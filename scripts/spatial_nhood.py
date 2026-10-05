# 中文注释：标签邻域富集（标签置换零模型，稀疏实现）。
# 输入构念：细胞表（坐标 + 标签，可选块列 donor/slice/domain）。
# 审计 F09 修正：
# 1) 旧实现显式分配多个 n×n 稠密矩阵（邻接/上三角掩码/投影），10 万细胞
#    一张 float64 就 ~74.5GiB，远超单任务 8GiB 预算；本实现 kNN 图用
#    scipy.sparse 存储，计数/置换全在 O(n·k) 边表上做，无任何 n×n 分配。
# 2) 旧实现 rng.permutation(n) 全表打散标签且无块限制，跨切片混合随机化；
#    零假设口径诚实化：全局置换是"全切片标签可交换"零假设——它不保留标签
#    的空间自相关（这正是零模型的本意），日志打显式 WARN；提供 block_column
#    时置换只在块内交换，不跨块混合，块内标签组成保持不变。
# 3) fix-review R08（v3）：block_column 同时限制**建图**——逐块建 kNN 再并
#    回全局索引，合并多切片输入得到各切片内的图，不产生跨切片边（v2 的
#    block 只限制置换：两切片坐标贴近时 k=1 建出的边全为跨切片）。块内
#    细胞数 < k+1 时有效 k 缩小并 WARN。块内标签交换只保持标签数量，仍不
#    保持块内空间自相关（零模型本意，日志照记）。
# 统计口径不变（观测统计量定义与置换 P 公式均未改）：kNN 邻接（不含自身，
# k+1 取邻居后去自身），无向边每边计一次（i<j）；同标签对角只计一次；
# 经验 P=(b+1)/(B+1)，双侧各报 p_upper/p_lower；z 用 ddof=1 的置换 SD。
# 失败处理：列缺失/标签只有一类 → 退出 2。
# 解释边界：切片内统计，供者才是独立单位；空间邻近≠迁移/通讯证据（§15.5）。
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

X_COL = os.environ.get("NH_X_COL", "x_centroid")
Y_COL = os.environ.get("NH_Y_COL", "y_centroid")
LABEL_COL = os.environ.get("NH_LABEL_COL", "celltype")
BLOCK_COL = (os.environ.get("NH_BLOCK_COL", "") or "").strip()
K = int(os.environ.get("NH_K", "6"))
N_PERM = int(os.environ.get("NH_PERM", "1000"))
SEED = int(os.environ.get("NH_SEED", "42"))

GLOBAL_WARN = ("WARN: global label exchange null; does not preserve spatial "
               "autocorrelation")


def fail(message):
    print(message, file=sys.stderr)
    sys.exit(2)


def _knn_edges(coords, k):
    """单（块内）坐标集的 kNN 无向边（局部索引，ei<ej，每边恰一次）。

    n_neighbors 取 min(k+1, n)：块内细胞数不足 k+1 时按实际数取邻居
    （有效 k 缩到 n-1，由调用方 WARN）。稀疏实现：三元组建 csr，邻接
    对称化与去重（triu 取 i<j 侧）都是稀疏运算，内存 O(n·k)。
    """
    n = coords.shape[0]
    if n < 2 or k < 1:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(coords)
    _, neighbors = nn.kneighbors(coords)
    width = neighbors.shape[1] - 1
    if width < 1:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    rows = np.repeat(np.arange(n), width)
    cols = neighbors[:, 1:].ravel()  # 去掉首列自身
    directed = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n)
    )
    undirected = directed + directed.T
    undirected.data[:] = 1  # 互为邻居的边只算一条
    upper = sp.triu(undirected, k=1)  # 每条无向边保留 i<j 一侧
    ei, ej = upper.nonzero()
    return ei.astype(np.int64), ej.astype(np.int64)


def build_edges(coords, k, blocks=None):
    """kNN 无向边表：(ei, ej) 且 ei<ej，每边恰一次。

    blocks 为 None 时全表一次 kNN；给定块列时**逐块**建 kNN 再并回
    全局索引（fix-review R08）——块内各点只与本块近邻连边，合并多
    切片输入得到的是各切片内的图，不产生跨切片边（v2 的 block 只
    限制置换、不限制建图：两切片坐标贴近时 k=1 建出的边全为跨切片）。
    块序为排序后的块标签序，同 seed 确定。
    """
    if blocks is None:
        return _knn_edges(coords, k)
    ei_parts, ej_parts = [], []
    for block in sorted(set(blocks.tolist())):
        idx = np.flatnonzero(blocks == block)
        local_ei, local_ej = _knn_edges(coords[idx], k)
        ei_parts.append(idx[local_ei])
        ej_parts.append(idx[local_ej])
    if not ei_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return (np.concatenate(ei_parts).astype(np.int64),
            np.concatenate(ej_parts).astype(np.int64))


def pair_counts(code, ei, ej, n_levels):
    """标签对计数（观测统计量的定义实现，与旧实现的矩阵式等价）。

    raw[a,b] = 下端点标签为 a、上端点标签为 b 的边数（i<j 口径）；
    返回对称化口径：a≠b 时 raw[a,b]+raw[b,a]（无向每边一次），
    对角（同标签边）只计一次。bincount 在边表上计数，等价于
    onehot.T @ triu(adj) @ onehot 但内存 O(n·k + L^2)。
    """
    flat = code[ei].astype(np.int64) * n_levels + code[ej]
    raw = np.bincount(flat, minlength=n_levels * n_levels).reshape(n_levels, n_levels)
    sym = raw + raw.T
    np.fill_diagonal(sym, 0)
    return sym + np.diag(np.diag(raw))


def permute_labels(code, blocks, rng):
    """置换标签：blocks 为 None 时全表可交换；否则只在各块内部交换。

    块迭代顺序为排序后的块标签序，保证同 seed 确定性。
    """
    if blocks is None:
        return code[rng.permutation(len(code))]
    permuted = code.copy()
    for block in sorted(set(blocks.tolist())):
        idx = np.flatnonzero(blocks == block)
        permuted[idx] = code[idx][rng.permutation(len(idx))]
    return permuted


def main():
    input_path = os.environ["AUTONOMICS_INPUT0"]
    out_tsv = os.environ["AUTONOMICS_OUTPUT0"]
    out_log = os.environ["AUTONOMICS_OUTPUT1"]
    table = pd.read_csv(input_path, sep="\t")
    required = [X_COL, Y_COL, LABEL_COL] + ([BLOCK_COL] if BLOCK_COL else [])
    for column in required:
        if column not in table.columns:
            fail(f"column missing: {column}")
    coords = table[[X_COL, Y_COL]].to_numpy(dtype=float)
    labels = table[LABEL_COL].astype(str).to_numpy()
    levels = sorted(set(labels))
    if len(levels) < 2:
        fail(f"label column {LABEL_COL} has fewer than 2 distinct labels")
    L = len(levels)
    code = pd.Series(labels).map({name: i for i, name in enumerate(levels)}).to_numpy()
    n = len(labels)
    blocks = None
    if BLOCK_COL:
        blocks = table[BLOCK_COL].astype(str).to_numpy()
        # 块内细胞数不足 k+1 时有效 k 缩小——如实 WARN（fix-review R08）。
        for block in sorted(set(blocks.tolist())):
            block_n = int((blocks == block).sum())
            if block_n < K + 1:
                print(
                    f"WARN: block {block!r} has {block_n} cells < k+1={K + 1}; "
                    "each cell connects to at most block_n-1 within-block neighbors "
                    "(effective k reduced)",
                    file=sys.stderr,
                )

    ei, ej = build_edges(coords, K, blocks)

    observed = pair_counts(code, ei, ej, L)

    rng = np.random.default_rng(SEED)
    null = np.zeros((N_PERM, L, L))
    for b in range(N_PERM):
        permuted = permute_labels(code, blocks, rng)
        null[b] = pair_counts(permuted, ei, ej, L)

    log_lines = [
        f"cells={n}\tlabels={L}\tk={K}\tedges={len(ei)}\tn_perm={N_PERM}\tseed={SEED}",
        f"perm_scope={'within-block' if blocks is not None else 'global'}"
        + (f"\tblock_column={BLOCK_COL}" if blocks is not None else ""),
    ]
    if blocks is not None:
        log_lines.append(
            "note: within-block label exchange null; preserves per-block label "
            "composition, not within-block spatial autocorrelation"
        )
    else:
        print(GLOBAL_WARN, file=sys.stderr)
        log_lines.append(GLOBAL_WARN)

    records = []
    for a in range(L):
        for b in range(a, L):
            observed_count = observed[a, b]
            null_values = null[:, a, b]
            mean = null_values.mean()
            sd = max(null_values.std(ddof=1), 1e-12)
            z = (observed_count - mean) / sd
            p_upper = (1 + int((null_values >= observed_count).sum())) / (N_PERM + 1)
            p_lower = (1 + int((null_values <= observed_count).sum())) / (N_PERM + 1)
            records.append((levels[a], levels[b], int(observed_count),
                            mean, sd, z, p_upper, p_lower))

    pd.DataFrame(records, columns=[
        "label_a", "label_b", "count", "expected_mean", "expected_sd",
        "z", "p_upper", "p_lower",
    ]).to_csv(out_tsv, sep="\t", index=False)
    log_lines.append(f"pairs={len(records)}")
    with open(out_log, "w", encoding="utf-8") as handle:
        handle.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
