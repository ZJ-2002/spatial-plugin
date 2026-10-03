# spatial

B-S 空间组学摄取与邻域统计（主计划 §15.5）。三个节点共享 biotools-py
镜像，共设一个插件家族。

## 节点

| kind | 输入 | 输出 | 方案条款 |
|---|---|---|---|
| `xenium_ingest` | 10x 风格 h5 + cells 表 | `xenium_counts.parquet`（cells×genes）、`xenium_cells.tsv`、`xenium_gene_stats.tsv`（带 `probe_class`） | §15.5-1 panel 覆盖核验 |
| `spatial_nhood` | 细胞表（坐标+标签 TSV） | `nhood_enrichment.tsv`（label_a/label_b/count/expected_mean/expected_sd/z/p_upper/p_lower） | §15.5-4 邻域富集 |
| `imzml_ingest` | imzML + ibd | `msi_pixels.tsv`（pixel_id/x/y/n_peaks/tic）、`msi_windows.tsv` | §15.5-3 空间代谢 |

参数见各自 manifest（xenium: `cells_format`/`x_col`/`y_col`；nhood:
`n_neighbors`=6/`n_perm`=1000/`seed`；imzml: `mz_windows` JSON，如
`[{"label":"mLPC_184","lo":184.07,"hi":184.08}]`）。

## 布局

```text
spatial/
├── manifest.toml
├── scripts/{xenium_ingest,spatial_nhood,imzml_ingest}.py
├── Dockerfile
└── README.md
```

镜像：`localhost/autonomics/biotools-py@sha256:21e14c1582a9d4c261c0a23a11864293313ee2b49b9b7d85bcabc89210b48c7e`。

## 口径（§15.5 硬约束，全部内嵌在脚本注释与 log）

- **nhood 置换只打散标签、坐标与邻接图不动**——保留空间自相关，
  这是零模型成立的前提；经验 P=(b+1)/(B+1)，双侧 p_upper/p_lower。
  无向边计数 i<j、同标签对角只计一次。
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
- h5 是 genes×cells（10x 转置），统一转成 cells×genes 再分块写
  parquet（20k 行组）。

## 冒烟记录

5×5 合成 Xenium：parquet 行和与 gene_stats 交叉一致、2 个负控探针
被标出；nhood 同标签 z≈+34 / 跨标签 z≈−20（P 在置换下限）；MSI
25 像素、窗均值与设计值吻合。
