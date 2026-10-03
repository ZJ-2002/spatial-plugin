# 中文注释：Xenium 输出包摄取（h5 计数矩阵 + cells 元数据表）。
# 输入构念：10x 风格 cell_feature_matrix.h5（CSR: data/indices/indptr/shape，
# features/name）+ cells 表（parquet/csv/tsv）。规则来源：方案 §15.5 第 1 条
# ——panel 覆盖与背景探针必须核验，故基因统计表带 probe_class 列。
# 失败处理：h5 结构不符/cells 缺列 → 退出 2。解释边界：不做分割复算、
# 不做去卷积；负控探针只作 QC，不进下游定量。
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
    data = np.asarray(group["data"])
    indices = np.asarray(group["indices"])
    indptr = np.asarray(group["indptr"])
    shape = tuple(int(value) for value in np.asarray(group["shape"]))
    gene_names = np.asarray(group["features"]["name"]).astype(str)

counts = sp.csr_matrix((data, indices, indptr), shape=shape)
# h5 是 genes×cells（10x 转置），统一成 cells×genes。
counts = counts.T.tocsr()
log_lines.append(f"matrix\tcells={counts.shape[0]}\tgenes={counts.shape[1]}")

if CELLS_FORMAT == "parquet":
    cells = pd.read_parquet(INPUT_CELLS)
elif CELLS_FORMAT == "csv":
    cells = pd.read_csv(INPUT_CELLS)
else:
    cells = pd.read_csv(INPUT_CELLS, sep="\t")
id_col = "cell_id" if "cell_id" in cells.columns else cells.columns[0]
for column in (id_col, X_COL, Y_COL):
    if column not in cells.columns:
        fail(f"cells table missing column: {column}")
if len(cells) != counts.shape[0]:
    fail(f"cells rows ({len(cells)}) != matrix cells ({counts.shape[0]})")

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
