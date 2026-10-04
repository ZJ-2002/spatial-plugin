# 中文注释：imzML/ibd 空间代谢摄取。
# 输入构念：预处理 imzML（XML 文本）+ 同实验 ibd 二进制。规则来源：方案
# §15.5 第 3 条——注释置信度核验在上游；本节点只取像素 TIC 与给定 m/z 窗
# 强度，不做峰注释、不做通量解释。失败处理：两输入无法按内容区分或
# 解析异常 → 退出 2；窗口 JSON 非法 → 退出 2。空窗表仍是合法输出。
# 解释边界：峰强度是测量信号；像素嵌套切片、切片嵌套供者（§15.5-4）。
# 范围外注记（2026-10 审计 F08/F09 轮）：窗口输出语义（窗边界含否、
# mean/max/检出像素口径）不在本轮修复范围，见 manifest doc 注记。
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser

PATH_A = os.environ["AUTONOMICS_INPUT0"]
PATH_B = os.environ["AUTONOMICS_INPUT1"]
WINDOWS_JSON = os.environ.get("MZ_WINDOWS_JSON", "").strip()
WORKDIR = os.environ.get("AUTONOMICS_WORKDIR", "/work")
OUT_PIXELS = os.environ["AUTONOMICS_OUTPUT0"]
OUT_WINDOWS = os.environ["AUTONOMICS_OUTPUT1"]
OUT_LOG = os.environ["AUTONOMICS_OUTPUT2"]


def fail(message):
    print(message, file=sys.stderr)
    sys.exit(2)


def looks_like_imzml(path):
    with open(path, "rb") as handle:
        head = handle.read(512)
    return head.lstrip().startswith(b"<?xml") or b"<mzML" in head


log_lines = []
imzml, ibd = (PATH_A, PATH_B) if looks_like_imzml(PATH_A) else (PATH_B, PATH_A)
log_lines.append(f"pair\timzml={os.path.basename(imzml)}\tibd={os.path.basename(ibd)}")
base = os.path.join(WORKDIR, "msi_pair")
os.makedirs(base, exist_ok=True)
# pyimzML 按同基名找 ibd：暂存输入被改名过，这里恢复配对。
shutil.copy(imzml, os.path.join(base, "sample.imzML"))
shutil.copy(ibd, os.path.join(base, "sample.ibd"))

windows = []
if WINDOWS_JSON:
    try:
        windows = json.loads(WINDOWS_JSON)
    except json.JSONDecodeError as error:
        fail(f"mz_windows is not valid JSON: {error}")
    for window in windows:
        if not {"label", "lo", "hi"} <= set(window):
            fail(f"window needs label/lo/hi, got: {window}")
        if not window["lo"] < window["hi"]:
            fail(f"window lo must be < hi, got: {window}")

rows = []
window_rows = {w["label"]: [] for w in windows}
with ImzMLParser(os.path.join(base, "sample.imzML")) as parser:
    # pyimzML 版本间 API 名漂移：get_spectrum（新）/ getspectrum（旧）都兜住。
    get_spectrum = getattr(parser, "get_spectrum", None) or getattr(parser, "getspectrum")
    n_pixels = 0
    for idx, (x, y, _) in enumerate(parser.coordinates):
        mzs, intensities = get_spectrum(idx)
        tic = float(np.sum(intensities)) if len(intensities) else 0.0
        rows.append((idx, x, y, len(mzs), tic))
        for window in windows:
            mask = (mzs >= window["lo"]) & (mzs <= window["hi"])
            window_rows[window["label"]].append(float(intensities[mask].sum()) if mask.any() else 0.0)
        n_pixels += 1
        if n_pixels % 1000 == 0:
            log_lines.append(f"pixels parsed: {n_pixels}")

pd.DataFrame(rows, columns=["pixel_id", "x", "y", "n_peaks", "tic"]).to_csv(
    OUT_PIXELS, sep="\t", index=False
)

if windows:
    records = []
    for window in windows:
        values = np.asarray(window_rows[window["label"]])
        records.append((
            window["label"], window["lo"], window["hi"],
            float(values.mean()), float(values.max()),
            int((values > 0).sum()), n_pixels,
        ))
    pd.DataFrame(records, columns=[
        "label", "mz_lo", "mz_hi", "mean_intensity", "max_intensity",
        "pixels_detected", "pixels_total",
    ]).to_csv(OUT_WINDOWS, sep="\t", index=False)
    log_lines.append(f"windows={len(windows)}")
else:
    with open(OUT_WINDOWS, "w", encoding="utf-8") as handle:
        handle.write("label\tmz_lo\tmz_hi\tmean_intensity\tmax_intensity\tpixels_detected\tpixels_total\n")
    log_lines.append("windows=0 (TIC only; mz_windows param was empty)")

log_lines.append(f"pixels={len(rows)}")
with open(OUT_LOG, "w", encoding="utf-8") as handle:
    handle.write("\n".join(log_lines) + "\n")
