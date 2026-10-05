# F09 回归：稀疏化不改变统计口径 + 零模型口径诚实化。
# golden 原则：暴力参考实现从定义独立写出（O(n²) 全距阵 + 边集合），
# 不复用被测实现；另保留历史稠密实现作全量数值等价对照（同 seed 下
# 零模型抽样序列也一致），锁定"观测统计量与 P 公式不变"。
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest
from conftest import run_script, SCRIPTS_DIR

NH_ENV = {"NH_X_COL": "x", "NH_Y_COL": "y", "NH_LABEL_COL": "label",
          "NH_K": "4", "NH_PERM": "200", "NH_SEED": "11"}


def make_table(path, n=40, seed=7, label_of=None, block_of=None):
    rng = np.random.default_rng(seed)
    coords = rng.normal(size=(n, 2))
    labels = ([label_of(i) for i in range(n)] if label_of
              else np.array(["A", "B", "C"])[np.arange(n) % 3].tolist())
    frame = pd.DataFrame({"cell": [f"c{i}" for i in range(n)],
                          "x": coords[:, 0], "y": coords[:, 1],
                          "label": labels})
    if block_of:
        frame["slice"] = [block_of(i) for i in range(n)]
    frame.to_csv(path, sep="\t", index=False)
    return coords, np.array(labels)


def brute_force_counts(coords, labels, k):
    """独立暴力参考（从定义写，不抄被测实现）：

    无向 kNN 图 = {i 的 k 近邻} ∪ {j: i 是 j 的 k 近邻}；每条无向边
    对其（排序后的）标签对计 1。
    """
    n = len(labels)
    levels = sorted(set(labels.tolist()))
    index = {name: i for i, name in enumerate(levels)}
    dist = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(dist, np.inf)
    edges = set()
    for i in range(n):
        for j in np.argsort(dist[i], kind="stable")[:k]:
            edges.add((min(i, int(j)), max(i, int(j))))
    counts = {}
    for i, j in edges:
        a, b = sorted((index[labels[i]], index[labels[j]]))
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return levels, counts


def historical_dense_reference(coords, labels, k, n_perm, seed):
    """审计前实现（稠密矩阵式）逐行保留，作全量数值等价对照。"""
    from sklearn.neighbors import NearestNeighbors
    n = len(labels)
    levels = sorted(set(labels))
    L = len(levels)
    code = np.array([levels.index(name) for name in labels])
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, neighbors = nn.kneighbors(coords)
    rows = np.repeat(np.arange(n), k)
    cols = neighbors[:, 1:].ravel()
    adjacency = np.zeros((n, n), dtype=bool)
    adjacency[rows, cols] = True
    adjacency |= adjacency.T
    adjacency = adjacency.astype(np.float64)
    upper = np.triu(np.ones((n, n)), 1) > 0
    pair_matrix = adjacency * upper

    def symmetrize(counts):
        sym = counts + counts.T
        np.fill_diagonal(sym, 0)
        return sym + np.diag(np.diag(counts))

    onehot = np.eye(L)[code]
    observed = symmetrize(onehot.T @ pair_matrix @ onehot)
    rng = np.random.default_rng(seed)
    null = np.zeros((n_perm, L, L))
    for b in range(n_perm):
        permuted = onehot[rng.permutation(n)]
        null[b] = symmetrize(permuted.T @ pair_matrix @ permuted)
    records = []
    for a in range(L):
        for b in range(a, L):
            null_values = null[:, a, b]
            mean = null_values.mean()
            sd = max(null_values.std(ddof=1), 1e-12)
            records.append((levels[a], levels[b], int(observed[a, b]),
                            mean, sd, (observed[a, b] - mean) / sd,
                            (1 + int((null_values >= observed[a, b]).sum())) / (n_perm + 1),
                            (1 + int((null_values <= observed[a, b]).sum())) / (n_perm + 1)))
    return pd.DataFrame(records, columns=[
        "label_a", "label_b", "count", "expected_mean", "expected_sd",
        "z", "p_upper", "p_lower"])


def run_nhood(tmp_path, table_name="cells.tsv", env_extra=None):
    table = tmp_path / table_name
    out_tsv = tmp_path / "nhood_enrichment.tsv"
    out_log = tmp_path / "spatial_nhood.log"
    result = run_script("spatial_nhood.py", {**NH_ENV, **(env_extra or {})},
                        [table], [out_tsv, out_log])
    if result.returncode != 0:
        return result, None, None
    return result, pd.read_csv(out_tsv, sep="\t"), out_log.read_text()


@pytest.fixture
def small_table(tmp_path):
    table = tmp_path / "cells.tsv"
    coords, labels = make_table(table)
    return table, coords, labels


def test_observed_counts_match_brute_force_golden(tmp_path, small_table):
    _, coords, labels = small_table
    result, out, _ = run_nhood(tmp_path)
    assert result.returncode == 0, result.stderr
    levels, golden = brute_force_counts(coords, labels, k=4)
    assert set(out["label_a"]) == set(levels)
    for _, row in out.iterrows():
        a = levels.index(row["label_a"])
        b = levels.index(row["label_b"])
        assert int(row["count"]) == golden.get((min(a, b), max(a, b)), 0), \
            (row["label_a"], row["label_b"])


def test_full_numeric_equivalence_with_historical_dense(tmp_path, small_table):
    """同 seed 下与审计前稠密实现逐格一致：统计量定义与 P 公式未变。"""
    _, coords, labels = small_table
    result, out, _ = run_nhood(tmp_path)
    assert result.returncode == 0, result.stderr
    reference = historical_dense_reference(coords, labels, k=4, n_perm=200, seed=11)
    pd.testing.assert_frame_equal(out, reference)


def test_same_seed_determinism(tmp_path, small_table):
    import shutil
    shutil.copy(small_table[0], tmp_path / "run_a.tsv")
    shutil.copy(small_table[0], tmp_path / "run_b.tsv")
    _, out_a, log_a = run_nhood(tmp_path, table_name="run_a.tsv")
    _, out_b, log_b = run_nhood(tmp_path, table_name="run_b.tsv")
    pd.testing.assert_frame_equal(out_a, out_b)
    assert log_a == log_b


def test_global_null_warns_in_log_and_stderr(tmp_path, small_table):
    result, _, log = run_nhood(tmp_path)
    assert result.returncode == 0
    assert "WARN: global label exchange null" in log
    assert "does not preserve spatial autocorrelation" in log
    assert "WARN" in result.stderr


def two_cluster_table(tmp_path):
    """两个远离的空间簇：A 簇标签 L1/L2，B 簇标签 L3/L4；k 小 → 无跨簇边。"""
    rng = np.random.default_rng(3)
    n_half = 30
    coords = np.vstack([
        rng.normal(scale=0.5, size=(n_half, 2)),
        rng.normal(scale=0.5, size=(n_half, 2)) + np.array([100.0, 100.0]),
    ])
    labels = (["L1", "L2"] * n_half)[:n_half] + (["L3", "L4"] * n_half)[:n_half]
    frame = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": labels,
                          "slice": ["SA"] * n_half + ["SB"] * n_half})
    frame.to_csv(tmp_path / "cells.tsv", sep="\t", index=False)
    return frame


def test_blocked_permutation_keeps_labels_within_blocks(tmp_path):
    two_cluster_table(tmp_path)
    result, out, log = run_nhood(tmp_path, env_extra={"NH_BLOCK_COL": "slice"})
    assert result.returncode == 0, result.stderr
    assert "perm_scope=within-block" in log
    assert "block_column=slice" in log
    assert "WARN" not in log
    cross = out[((out["label_a"] == "L1") & (out["label_b"] == "L3"))]
    assert len(cross) == 1
    row = cross.iloc[0]
    assert int(row["count"]) == 0            # 两簇无跨簇边
    assert row["expected_mean"] == 0.0       # 块内置换永远造不出 L1-L3 邻边


def test_global_permutation_mixes_across_blocks_unlike_blocked(tmp_path):
    two_cluster_table(tmp_path)
    _, out_global, log_global = run_nhood(tmp_path)
    _, out_block, _ = run_nhood(tmp_path, env_extra={"NH_BLOCK_COL": "slice"})
    cross_g = out_global[(out_global["label_a"] == "L1") & (out_global["label_b"] == "L3")].iloc[0]
    cross_b = out_block[(out_block["label_a"] == "L1") & (out_block["label_b"] == "L3")].iloc[0]
    # 全局置换会把 L1/L2 与 L3/L4 混进彼此簇内 → 跨簇标签对的期望邻边数 > 0；
    # 块内置换严格为 0。同 seed 下两者必然不同。
    assert cross_g["expected_mean"] > 0.0
    assert cross_b["expected_mean"] == 0.0
    assert "WARN" in log_global


def load_module():
    spec = importlib.util.spec_from_file_location(
        "spatial_nhood_mod", os.path.join(SCRIPTS_DIR, "spatial_nhood.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_permute_labels_within_block_unit():
    module = load_module()
    code = np.array([0, 1, 2, 2])
    blocks = np.array(["a", "a", "a", "b"])
    for seed in range(20):
        rng = np.random.default_rng(seed)
        permuted = module.permute_labels(code, blocks, rng)
        assert sorted(permuted[:3].tolist()) == [0, 1, 2]  # 块 a 组成不变
        assert permuted[3] == 2                             # 块 b 标签不外流


def test_permute_labels_global_unit():
    module = load_module()
    code = np.array([0] * 5 + [1] * 5)
    rng = np.random.default_rng(0)
    permuted = module.permute_labels(code, None, rng)
    assert sorted(permuted.tolist()) == [0] * 5 + [1] * 5  # 是置换
    assert permuted.tolist() != code.tolist()               # 且确实打乱（seed 0）


def test_missing_block_column_fails(tmp_path, small_table):
    result, _, _ = run_nhood(tmp_path, env_extra={"NH_BLOCK_COL": "donor"})
    assert result.returncode == 2
    assert "column missing: donor" in result.stderr


def test_single_label_fails(tmp_path):
    table = tmp_path / "cells.tsv"
    make_table(table, n=10, label_of=lambda i: "A")
    result, _, _ = run_nhood(tmp_path)
    assert result.returncode == 2
    assert "fewer than 2 distinct labels" in result.stderr


# --- fix-review R08（2026-10-05）：block 必须同时限制建图，不只限制置换 ---

def r08_coords():
    """评审原始反例坐标：两切片各 3 细胞，切片 2 每点横坐标 +0.1。

    不分块时每点的全局最近邻都是另一切片的对应点（距 0.1 < 切片内 10）
    → k=1 的 3 条边全为跨切片；分块后应为两段链内边。
    """
    return np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0],
                     [0.1, 0.0], [10.1, 0.0], [20.1, 0.0]])


def edge_set(ei, ej):
    return {(int(a), int(b)) for a, b in zip(ei.tolist(), ej.tolist())}


def test_build_edges_without_blocks_is_cross_slice_r08():
    # 无 block（v2/v3 行为一致）：全局 k=1 的 3 条边全为跨切片
    # (0,3),(1,4),(2,5)——这正是评审指出的问题形态。
    module = load_module()
    ei, ej = module.build_edges(r08_coords(), 1)
    assert edge_set(ei, ej) == {(0, 3), (1, 4), (2, 5)}


def test_build_edges_blocks_restrict_graph_within_slice_r08():
    # fix-review R08 修复本体：给 block 后逐块建 kNN——两段切片内链
    # {0,1},{1,2} 与 {3,4},{4,5}，零跨切片边。
    module = load_module()
    blocks = np.array(["S1"] * 3 + ["S2"] * 3)
    ei, ej = module.build_edges(r08_coords(), 1, blocks)
    assert edge_set(ei, ej) == {(0, 1), (1, 2), (3, 4), (4, 5)}
    # 块序确定性：块标签换序（值同集合）结果一致。
    blocks_renamed = np.array(["B"] * 3 + ["A"] * 3)
    ei2, ej2 = module.build_edges(r08_coords(), 1, blocks_renamed)
    assert edge_set(ei, ej) == edge_set(ei2, ej2)


def test_blocked_end_to_end_no_cross_slice_edges_r08(tmp_path):
    # 端到端：同反例坐标 + block_column → 观测计数里不可能出现跨切片
    # 才有的标签对（S1 全 A、S2 全 B 时 A-B 边数为 0）。
    coords = r08_coords()
    frame = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "label": ["A"] * 3 + ["B"] * 3,
        "slice": ["S1"] * 3 + ["S2"] * 3,
    })
    table = tmp_path / "cells.tsv"
    frame.to_csv(table, sep="\t", index=False)
    result, out, log = run_nhood(
        tmp_path, env_extra={"NH_BLOCK_COL": "slice", "NH_K": "1",
                             "NH_PERM": "100"})
    assert result.returncode == 0, result.stderr
    assert "perm_scope=within-block" in log
    row = out[(out["label_a"] == "A") & (out["label_b"] == "B")].iloc[0]
    assert int(row["count"]) == 0  # 无跨切片边；v2 此处 count=3


def test_small_block_reduces_effective_k_with_warn(tmp_path):
    # 块内细胞数 < k+1：有效 k 缩小并 WARN（诚实化，不静默降 k）。
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [100.0, 0.0], [101.0, 0.0]])
    frame = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "label": ["A", "B", "A", "B"],
        "slice": ["S1", "S1", "S2", "S2"],
    })
    table = tmp_path / "cells.tsv"
    frame.to_csv(table, sep="\t", index=False)
    result, _, _ = run_nhood(
        tmp_path, env_extra={"NH_BLOCK_COL": "slice", "NH_K": "4",
                             "NH_PERM": "100"})
    assert result.returncode == 0, result.stderr
    assert "effective k reduced" in result.stderr
