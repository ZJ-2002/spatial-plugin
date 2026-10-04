# 中文注释：Xenium 输出包摄取（h5 计数矩阵 + cells 元数据表）。
# 输入构念：官方 10x HDF5 布局——/matrix 组是 CSC：shape=(genes, cells)，
# indptr 按细胞列组织（长度 cells+1），barcodes 逐细胞、features/name 逐基因。
# 审计 F08 修正：旧实现按 CSR 读（indptr 当基因行指针）——非方阵直接
# ValueError（2 基因×3 细胞即报 index pointer size 错），方阵静默错读；
# 且只按行号对齐 cells 表，同长度不同顺序会把 counts 标错细胞。
# 本实现：csc_matrix 按官方布局直读；从 /matrix/barcodes 与 cells 表的
# barcode 列严格对齐（h5/cells 重复、缺失、多余均显式退出 2），
# counts 行序随 cells 表行序（cell 关联来自 barcode 映射而非行号）。
# 规则来源：方案 §15.5 第 1 条——panel 覆盖与背景探针必须核验，故基因
# 统计表带 probe_class 列。失败处理：h5 结构不符/cells 缺列/对齐失败 →
# 退出 2。解释边界：不做分割复算、不做去卷积；负控探针只作 QC，
# 不进下游定量。
import os
import sys

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

INPUT_H5 = os.environ["AUTONOMICS_INPUT0"]
INPUT_CELLS = os.environ["AUTONOMICS_INPUT1"]
CELLS_FORMAT = os.environ.get("XN_CELLS_FORMAT", "parquet").lower()
X_COL = os.environ.get("XN_X_COL", "x_centroid")
Y_COL = os.environ.get("XN_Y_COL", "y_centroid")
BARCODE_COL = os.environ.get("XN_BARCODE_COL", "") or "cell_id"
OUT_COUNTS = os.environ["AUTONOMICS_OUTPUT0"]
OUT_CELLS = os.environ["AUTONOMICS_OUTPUT1"]
OUT_GENE_STATS = os.environ["AUTONOMICS_OUTPUT2"]
OUT_LOG = os.environ["AUTONOMICS_OUTPUT3"]

CONTROL_PREFIXES = ("NegControlCode", "NegControlProbe", "UnassignedCode",
                    "DeprecatedCode", "BLANK", "antisense")


def fail(message):
    print(message, file=sys.stderr)
    sys.exit(2)


log_lines = []
with h5py.File(INPUT_H5, "r") as handle:
    if "matrix" not in handle:
        fail("h5 has no /matrix group; not a 10x/Xenium cell_feature_matrix?")
    group = handle["matrix"]
    if "barcodes" not in group:
        fail("h5 /matrix has no barcodes dataset; barcode alignment is mandatory "
             "(official 10x layout always provides it)")
    data = np.asarray(group["data"])
    indices = np.asarray(group["indices"])
    indptr = np.asarray(group["indptr"])
    shape = tuple(int(value) for value in np.asarray(group["shape"]))
    gene_names = np.asarray(group["features"]["name"]).astype(str)
    barcodes = np.asarray(group["barcodes"]).astype(str)
if len(shape) != 2:
    fail(f"matrix shape must be 2-D, got {shape}")
n_genes, n_cells = shape
# 官方 CSC 布局自检：indptr 按细胞列组织，长度必须是 cells+1。
if len(indptr) != n_cells + 1:
    fail(f"indptr length {len(indptr)} != cells+1 ({n_cells}+1); "
         "not official 10x CSC layout (indptr must be organized per cell column)")
if len(gene_names) != n_genes:
    fail(f"features/name has {len(gene_names)} entries != shape[0] genes ({n_genes})")
if len(barcodes) != n_cells:
    fail(f"barcodes has {len(barcodes)} entries != shape[1] cells ({n_cells})")

# 官方布局直读：CSC 的列就是细胞；转置成 cells×genes（行序=barcodes 序）。
counts = sp.csc_matrix((data, indices, indptr), shape=(n_genes, n_cells))
counts = counts.T.tocsr()
log_lines.append(f"matrix\tcells={counts.shape[0]}\tgenes={counts.shape[1]}")

if CELLS_FORMAT == "parquet":
    cells = pd.read_parquet(INPUT_CELLS)
elif CELLS_FORMAT == "csv":
    cells = pd.read_csv(INPUT_CELLS)
else:
    cells = pd.read_csv(INPUT_CELLS, sep="\t")
id_col = "cell_id" if "cell_id" in cells.columns else cells.columns[0]
for column in (id_col, X_COL, Y_COL, BARCODE_COL):
    if column not in cells.columns:
        fail(f"cells table missing column: {column}")

# barcode 严格对齐（审计 F08 核心）：唯一性 + 双向完整性，任何不符退出 2。
h5_index = pd.Index(barcodes)
h5_dups = sorted(set(barcodes[np.asarray(h5_index.duplicated())]))
if h5_dups:
    fail(f"duplicate barcodes in h5 /matrix/barcodes (e.g. {h5_dups[:3]}); "
         f"cell identity is ambiguous")
cell_bc = cells[BARCODE_COL].astype(str)
cells_dups = sorted(set(cell_bc[cell_bc.duplicated().to_numpy()]))
if cells_dups:
    fail(f"duplicate barcodes in cells column {BARCODE_COL!r} "
         f"(e.g. {cells_dups[:3]}); cannot align counts unambiguously")
extra = sorted(set(cell_bc) - set(barcodes))
missing = sorted(set(barcodes) - set(cell_bc))
if extra or missing:
    fail(f"barcode alignment failed: {len(missing)} barcode(s) in h5 missing from "
         f"cells table (e.g. {missing[:3]}), {len(extra)} barcode(s) in cells table "
         f"not in h5 (e.g. {extra[:3]})")
# 每个 cells 行定位到 h5 的细胞列；counts 行序改为随 cells 表行序。
position = h5_index.get_indexer(cell_bc)
counts = counts[position]
n_reordered = int((position != np.arange(len(position))).sum())
log_lines.append(f"aligned\tbarcodes={len(position)}\treordered_rows={n_reordered}")

def is_listlike(series):
    # pandas 3.x 移除了 is_list_dtype；按首元素类型探测（parquet 列表列的元素是
    # ndarray/list）。空列视为非列表。
    if len(series) == 0:
        return False
    first = series.iloc[0]
    return isinstance(first, (list, tuple, set, frozenset)) or hasattr(first, "shape")

flat = cells[[column for column in cells.columns if not is_listlike(cells[column])]]
flat.to_csv(OUT_CELLS, sep="\t", index=False)

# 基因统计 + 背景探针分类（§15.5 第 1 条的核验表）。
totals = np.asarray(counts.sum(axis=0)).ravel()
# 每基因检出细胞数按列求（indptr 是行口径，转置后不再适用）。
detected = np.asarray((counts > 0).sum(axis=0)).ravel()
is_control = pd.Series(gene_names).str.startswith(CONTROL_PREFIXES).to_numpy()
probe_class = np.where(is_control, "control", "target")
pd.DataFrame({
    "gene": gene_names,
    "total_counts": totals.astype(np.int64),
    "n_cells": detected.astype(np.int64),
    "probe_class": probe_class,
}).to_csv(OUT_GENE_STATS, sep="\t", index=False)
n_control = int((probe_class == "control").sum())
log_lines.append(f"genes\ttarget={len(gene_names) - n_control}\tcontrol={n_control}")

# 计数矩阵分块写 parquet，避免整块稠密化。
row_group = 20000
schema = pa.schema([("cell_id", pa.string())] +
                   [(str(gene), pa.int32()) for gene in gene_names])
writer = pq.ParquetWriter(OUT_COUNTS, schema)
ids = cells[id_col].astype(str).to_numpy()
for start in range(0, counts.shape[0], row_group):
    block = counts[start:start + row_group].toarray().astype(np.int32)
    columns = {"cell_id": ids[start:start + row_group]}
    for j, gene in enumerate(gene_names):
        columns[str(gene)] = block[:, j]
    writer.write_table(pa.table(columns, schema=schema))
writer.close()
log_lines.append("wrote parquet row groups")
with open(OUT_LOG, "w", encoding="utf-8") as handle:
    handle.write("\n".join(log_lines) + "\n")
