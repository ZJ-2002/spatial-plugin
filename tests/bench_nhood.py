# F09 规模基准（pytest 不收集；手动跑）：
#   python tests/bench_nhood.py <new|old> <n_cells> [n_perm]
# new = 当前 scripts/spatial_nhood.py；old = git 3f27564 的审计前实现
# （由调用方预先导出到 /tmp/spatial_nhood_old.py）。
# 本机无 GNU time（/usr/bin/time）：峰值 RSS 用
# resource.getrusage(RUSAGE_CHILDREN).ru_maxrss（内核与 time -v 的
# Maximum resident set size 同源），wall 用 perf_counter。
import os
import resource
import subprocess
import sys
import tempfile
import time as _time

import numpy as np
import pandas as pd

VENV_PY = os.path.expanduser("~/.cache/fix-venvs/spatial/bin/python")
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
OLD_SCRIPT = "/tmp/spatial_nhood_old.py"


def make_table(path, n, n_labels=8, seed=5):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=20.0, size=(n_labels, 2))
    which = rng.integers(0, n_labels, size=n)
    coords = rng.normal(scale=1.0, size=(n, 2)) + centers[which]
    labels = np.array([f"LT_{i}" for i in range(n_labels)])[which]
    slice_id = np.array([f"S{i % 4}" for i in range(4)])[np.arange(n) % 4]
    pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1],
                  "label": labels, "slice": slice_id}).to_csv(path, sep="\t", index=False)


def main():
    impl, n = sys.argv[1], int(sys.argv[2])
    n_perm = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    script = OLD_SCRIPT if impl == "old" else os.path.join(SCRIPTS_DIR, "spatial_nhood.py")
    with tempfile.TemporaryDirectory(prefix="nhood-bench-") as workdir:
        table = os.path.join(workdir, "cells.tsv")
        make_table(table, n)
        env = dict(os.environ)
        env.update({
            "AUTONOMICS_INPUT0": table,
            "AUTONOMICS_OUTPUT0": os.path.join(workdir, "nhood.tsv"),
            "AUTONOMICS_OUTPUT1": os.path.join(workdir, "nhood.log"),
            "NH_X_COL": "x", "NH_Y_COL": "y", "NH_LABEL_COL": "label",
            "NH_K": "6", "NH_PERM": str(n_perm), "NH_SEED": "42",
        })
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = _time.perf_counter()
        result = subprocess.run([VENV_PY, script],
                                env=env, capture_output=True, text=True)
        wall_seconds = _time.perf_counter() - start
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        if result.returncode != 0:
            print(result.stderr[-2000:])
            sys.exit(1)
        edges = ""
        log_path = env["AUTONOMICS_OUTPUT1"]
        with open(log_path) as handle:
            for line in handle:
                if line.startswith("cells="):
                    edges = line.strip()
        peak_kib = after.ru_maxrss  # 取 after；子进程只有这一个
        mib = peak_kib / 1024
        print(f"impl={impl}\tn={n}\tn_perm={n_perm}\t{edges}\t"
              f"peak_rss={mib:.1f}MiB\twall_s={wall_seconds:.1f}\t"
              f"user_s={after.ru_utime - before.ru_utime:.1f}")


if __name__ == "__main__":
    main()
