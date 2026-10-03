# 中文注释：标签邻域富集（组织内置换零模型）。
# 输入构念：细胞表（坐标 + 标签）。规则来源：方案 §15.5 第 4 条——置换只打散
# 标签、不动坐标与邻接图，保留空间自相关；经验 P=(b+1)/(B+1)，双侧各报
# p_upper/p_lower。统计口径：kNN 邻接（不含自身），标签对计数按无向边
# 每边计一次（i<j）。
# 失败处理：列缺失/标签只有一类 → 退出 2。
# 解释边界：切片内统计，供者才是独立单位；空间邻近≠迁移/通讯证据（§15.5）。
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

X_COL = os.environ.get("NH_X_COL", "x_centroid")
Y_COL = os.environ.get("NH_Y_COL", "y_centroid")
LABEL_COL = os.environ.get("NH_LABEL_COL", "celltype")
K = int(os.environ.get("NH_K", "6"))
N_PERM = int(os.environ.get("NH_PERM", "1000"))
SEED = int(os.environ.get("NH_SEED", "42"))
INPUT = os.environ.get("AUTONOMICS_INPUT0")
OUT_TSV = os.environ["AUTONOMICS_OUTPUT0"]
OUT_LOG = os.environ["AUTONOMICS_OUTPUT1"]


def fail(message):
    print(message, file=sys.stderr)
    sys.exit(2)


table = pd.read_csv(INPUT, sep="\t")
for column in (X_COL, Y_COL, LABEL_COL):
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

nn = NearestNeighbors(n_neighbors=K + 1).fit(coords)
_, neighbors = nn.kneighbors(coords)
rows = np.repeat(np.arange(n), K)
cols = neighbors[:, 1:].ravel()
adjacency = np.zeros((n, n), dtype=bool)
adjacency[rows, cols] = True
adjacency |= adjacency.T
adjacency = adjacency.astype(np.float64)
upper = np.triu(np.ones((n, n)), 1) > 0
pair_matrix = adjacency * upper


def symmetrize(counts):
    # 无向口径：跨标签对 X[a,b]+X[b,a]；对角（同标签）只计一次——
    # 之前用 triu(...,1) 把对角清零，同类邻域计数恒 0，已修。
    sym = counts + counts.T
    np.fill_diagonal(sym, 0)
    return sym + np.diag(np.diag(counts))


# observed[l, m] = 标签 l,m 邻接边数（无向，每边一次）。
onehot = np.eye(L)[code]
observed = symmetrize(onehot.T @ pair_matrix @ onehot)

rng = np.random.default_rng(SEED)
null = np.zeros((N_PERM, L, L))
for b in range(N_PERM):
    permuted = onehot[rng.permutation(n)]
    null[b] = symmetrize(permuted.T @ pair_matrix @ permuted)

log_lines = [
    f"cells={n}\tlabels={L}\tk={K}\tn_perm={N_PERM}\tseed={SEED}",
    "rule: label permutation only; coordinates and adjacency fixed (plan §15.5-4)",
]
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
]).to_csv(OUT_TSV, sep="\t", index=False)
log_lines.append(f"pairs={len(records)}")
with open(OUT_LOG, "w", encoding="utf-8") as handle:
    handle.write("\n".join(log_lines) + "\n")
