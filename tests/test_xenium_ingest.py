# F08 回归：官方 10x CSC 布局直读 + barcode 严格对齐。
# 审计反例形状（2 基因×3 细胞）下旧 CSR 读法必报 ValueError，新实现必读对；
# barcode 乱序证明按 barcode（而非行号）对齐；重复/缺失/多余均显式退出 2。
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from conftest import run_script

# 2 基因 × 3 细胞，每格计数互不相同——错位一眼可见。
GENES = ["GENE_A", "GENE_B", "NegControlCode_1"]
BARCODES = ["bc-1", "bc-2", "bc-3"]
# genes × cells（官方 shape 口径）
DENSE = np.array([
    [1, 20, 300],   # GENE_A
    [4, 50, 600],   # GENE_B
    [7, 80, 900],   # NegControlCode_1
], dtype=np.int64)


def write_official_h5(path, dense=DENSE, genes=GENES, barcodes=BARCODES,
                      with_barcodes=True):
    """按官方 10x cell_feature_matrix.h5 布局写 fixture（CSC：indptr 按细胞列）。"""
    csc = sp.csc_matrix(np.asarray(dense))
    with h5py.File(path, "w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("data", data=csc.data.astype(np.int32))
        group.create_dataset("indices", data=csc.indices.astype(np.int32))
        group.create_dataset("indptr", data=csc.indptr.astype(np.int32))
        group.create_dataset("shape", data=np.array(csc.shape, dtype=np.int64))
        features = group.create_group("features")
        features.create_dataset("name",
                                data=np.array(genes, dtype=h5py.string_dtype("utf-8")))
        if with_barcodes:
            group.create_dataset("barcodes",
                                 data=np.array(barcodes, dtype=h5py.string_dtype("utf-8")))


def cells_frame(order=BARCODES):
    """cells 表：可乱序、带 donor/slice 身份列。"""
    donor = {"bc-1": "D1", "bc-2": "D1", "bc-3": "D2"}
    return pd.DataFrame({
        "cell_id": list(order),
        "x_centroid": [1.0, 2.0, 3.0][: len(order)],
        "y_centroid": [4.0, 5.0, 6.0][: len(order)],
        "donor": [donor.get(b, "D9") for b in order],
        "slice": ["S1"] * len(order),
    })


@pytest.fixture
def out_paths(tmp_path):
    return [tmp_path / "counts.parquet", tmp_path / "cells.tsv",
            tmp_path / "gene_stats.tsv", tmp_path / "ingest.log"]


def run_xenium(tmp_path, cells, outputs, h5_kwargs=None, env_extra=None):
    h5_path = tmp_path / "cell_feature_matrix.h5"
    write_official_h5(h5_path, **(h5_kwargs or {}))
    cells_path = tmp_path / "cells.tsv"
    cells.to_csv(cells_path, sep="\t", index=False)
    result = run_script("xenium_ingest.py",
                        dict(XN_CELLS_FORMAT="tsv", **(env_extra or {})),
                        [h5_path, cells_path], outputs)
    return result


def test_audit_counterexample_old_csr_read_raises(tmp_path):
    """审计反例：2 基因×3 细胞官方 CSC 数组，旧 CSR 读法（审计 F08 原实现）必报错。"""
    h5_path = tmp_path / "matrix.h5"
    write_official_h5(h5_path)
    with h5py.File(h5_path, "r") as handle:
        group = handle["matrix"]
    # 非方阵官方数组（2 基因 × 3 细胞）：indptr 长 4，CSR 按 shape=(2,3) 要 3。
    dense23 = np.array([[1, 2, 3], [4, 5, 6]])
    csc23 = sp.csc_matrix(dense23)
    with pytest.raises(ValueError, match="index pointer size"):
        sp.csr_matrix((csc23.data, csc23.indices, csc23.indptr),
                      shape=(2, 3))


def test_official_layout_ingest_and_gene_stats(tmp_path, out_paths):
    result = run_xenium(tmp_path, cells_frame(), out_paths)
    assert result.returncode == 0, result.stderr
    counts = pd.read_parquet(out_paths[0])
    assert list(counts["cell_id"]) == BARCODES
    for j, gene in enumerate(GENES):
        col = counts[gene].to_numpy()
        assert col.tolist() == DENSE[j].tolist(), gene
    stats = pd.read_csv(out_paths[2], sep="\t")
    assert list(stats["gene"]) == GENES
    assert stats["total_counts"].tolist() == DENSE.sum(axis=1).tolist()
    assert stats["n_cells"].tolist() == [3, 3, 3]
    assert stats["probe_class"].tolist() == ["target", "target", "control"]
    cells_out = pd.read_csv(out_paths[1], sep="\t")
    # donor/slice 身份列保真。
    assert list(cells_out["donor"]) == ["D1", "D1", "D2"]
    assert list(cells_out["slice"]) == ["S1", "S1", "S1"]


def test_barcode_reorder_alignment(tmp_path, out_paths):
    """cells 表乱序：counts 行必须跟着 barcode 走，而不是跟着行号走。"""
    order = ["bc-3", "bc-1", "bc-2"]
    result = run_xenium(tmp_path, cells_frame(order=order), out_paths)
    assert result.returncode == 0, result.stderr
    counts = pd.read_parquet(out_paths[0])
    assert list(counts["cell_id"]) == order
    # bc-3 的观测是 DENSE 的第 3 列；行号对齐（旧缺陷）会拿到第 1 行的数。
    for j, gene in enumerate(GENES):
        col = counts[gene].to_numpy()
        expected = [DENSE[j][BARCODES.index(b)] for b in order]
        assert col.tolist() == expected, gene
    log = out_paths[3].read_text()
    assert "reordered_rows=3" in log
    # cells.tsv 行序随输入 cells 表。
    cells_out = pd.read_csv(out_paths[1], sep="\t")
    assert list(cells_out["cell_id"]) == order


def test_duplicate_barcode_in_cells_fails(tmp_path, out_paths):
    cells = cells_frame(order=["bc-1", "bc-2", "bc-1"])
    result = run_xenium(tmp_path, cells, out_paths)
    assert result.returncode == 2
    assert "duplicate barcodes in cells" in result.stderr


def test_duplicate_barcode_in_h5_fails(tmp_path, out_paths):
    result = run_xenium(tmp_path, cells_frame(), out_paths,
                        h5_kwargs={"barcodes": ["bc-1", "bc-2", "bc-2"]})
    assert result.returncode == 2
    assert "duplicate barcodes in h5" in result.stderr


def test_missing_barcode_in_cells_fails(tmp_path, out_paths):
    cells = cells_frame(order=["bc-1", "bc-2"])  # cells 少一个 barcode
    result = run_xenium(tmp_path, cells, out_paths)
    assert result.returncode == 2
    assert "barcode alignment failed" in result.stderr
    assert "bc-3" in result.stderr


def test_extra_barcode_in_cells_fails(tmp_path, out_paths):
    cells = cells_frame(order=["bc-1", "bc-2", "bc-9"])  # cells 多一个 h5 没有的
    result = run_xenium(tmp_path, cells, out_paths)
    assert result.returncode == 2
    assert "barcode alignment failed" in result.stderr
    assert "bc-9" in result.stderr


def test_h5_without_barcodes_fails(tmp_path, out_paths):
    result = run_xenium(tmp_path, cells_frame(), out_paths,
                        h5_kwargs={"with_barcodes": False})
    assert result.returncode == 2
    assert "no barcodes dataset" in result.stderr


def test_non_square_official_matrix_roundtrip(tmp_path, out_paths):
    """审计反例形状 2 基因×3 细胞：官方 CSC 直读 + 转置必须无损。"""
    dense = np.array([[1, 0, 3], [0, 2, 0]], dtype=np.int64)  # 2 genes × 3 cells
    result = run_xenium(tmp_path, cells_frame(), out_paths,
                        h5_kwargs={"dense": dense, "genes": ["G1", "G2"]})
    assert result.returncode == 0, result.stderr
    counts = pd.read_parquet(out_paths[0])
    assert counts["G1"].tolist() == [1, 0, 3]
    assert counts["G2"].tolist() == [0, 2, 0]
    stats = pd.read_csv(out_paths[2], sep="\t")
    assert stats["total_counts"].tolist() == [4, 2]
    assert stats["n_cells"].tolist() == [2, 1]


def test_custom_barcode_column(tmp_path, out_paths):
    """barcode 列名不是 cell_id 时用 barcode_col 指定。"""
    cells = cells_frame()
    cells = cells.rename(columns={"cell_id": "cell_name"})
    cells["barcode"] = BARCODES
    result = run_xenium(tmp_path, cells, out_paths,
                        env_extra={"XN_BARCODE_COL": "barcode"})
    assert result.returncode == 0, result.stderr
    counts = pd.read_parquet(out_paths[0])
    assert list(counts["cell_id"]) == BARCODES  # id 列取第一列 cell_name
