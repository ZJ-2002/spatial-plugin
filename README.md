# spatial

B-S 空间组学摄取与邻域统计（主计划 §15.5）。三个节点共享 biotools-py
镜像，共设一个插件家族。

## 节点

| kind | 输入 | 输出 | 方案条款 |
|---|---|---|---|
| `xenium_ingest` | 10x 风格 h5 + cells 表 | `xenium_counts.parquet`（cells×genes）、`xenium_cells.tsv`、`xenium_gene_stats.tsv`（带 `probe_class`） | §15.5-1 panel 覆盖核验 |
| `spatial_nhood` | 细胞表（坐标+标签 TSV） | `nhood_enrichment.tsv`（label_a/label_b/count/expected_mean/expected_sd/z/p_upper/p_lower） | §15.5-4 邻域富集 |
| `imzml_ingest` | imzML + ibd | `msi_pixels.tsv`（pixel_id/x/y/n_peaks/tic）、`msi_windows.tsv` | §15.5-3 空间代谢 |

参数见各自 manifest（xenium: `cells_format`/`x_col`/`y_col`/`barcode_col`；
nhood: `n_neighbors`=6/`n_perm`=1000/`seed`/`block_column`（块内**建图+**
**置换**，缺省全局图+全局置换并打 WARN）；imzml: `mz_windows` JSON，如
`[{"label":"mLPC_184","lo":184.07,"hi":184.08}]`）。

## 布局

```text
spatial/
├── manifest.toml
├── scripts/{xenium_ingest,spatial_nhood,imzml_ingest}.py
├── tests/（pytest：F08 barcode 对齐 + F09 稀疏置换/暴力 golden）
├── Dockerfile
└── README.md
```

镜像：`localhost/autonomics/biotools-py@sha256:21e14c1582a9d4c261c0a23a11864293313ee2b49b9b7d85bcabc89210b48c7e`。

## 口径（§15.5 硬约束，全部内嵌在脚本注释与 log）

- **nhood 置换只打散标签、坐标与邻接图不动**；经验 P=(b+1)/(B+1)，
  双侧 p_upper/p_lower；无向边计数 i<j、同标签对角只计一次。
  零假设口径：全局置换=全切片标签可交换，**不保留标签空间自相关**
  （log 显式 WARN）；`block_column` 给定时**建图与置换都在块内**
  （fix-review R08：逐块建 kNN 并回全局索引，合并多切片输入得到各
  切片内的图，不产生跨切片边；v2 只限制置换——两切片坐标贴近时
  k=1 的边全为跨切片。块内细胞数 < k+1 时有效 k 缩小并 WARN）。
  块内置换只保持标签数量，仍不保留块内空间自相关（零模型本意）。
  实现全程稀疏（kNN 图 scipy.sparse + O(n·k) 边表计数），无 n×n
  稠密分配。
- **probe_class** 列标出 NegControlCode/NegControlProbe/UnassignedCode/
  DeprecatedCode/BLANK/antisense——负控探针只作 QC，不进下游定量。
- **imzml 只取 TIC 与给定 m/z 窗强度**：峰强度≠通量；注释置信度核验
  在上游；空窗表（仅表头）是合法输出（只算 TIC 的用法）。
- 三个节点共同的解释边界：切片内统计，**供者才是独立单位**；空间
  邻近≠迁移/通讯证据。

## 实现说明

- pyimzML 版本间 API 漂移（`getspectrum`/`get_spectrum`）：getattr
  双兜底；输入按内容嗅探配对后恢复同基名（引擎暂存会改名，pyimzML
  按基名找 ibd）。
- h5 按官方 10x **CSC** 布局直读（indptr 按细胞列；审计 F08 修正——
  旧 CSR 读法非方阵报错、方阵静默错读），counts 行与 cells 行按
  `/matrix/barcodes` 严格对齐（重复/缺失/多余退出 2），统一转成
  cells×genes 再分块写 parquet（20k 行组）。

## 冒烟记录

5×5 合成 Xenium：parquet 行和与 gene_stats 交叉一致、2 个负控探针
被标出；nhood 同标签 z≈+34 / 跨标签 z≈−20（P 在置换下限）；MSI
25 像素、窗均值与设计值吻合。
