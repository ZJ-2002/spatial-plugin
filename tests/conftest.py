# 测试公共件：以子进程方式按引擎环境变量约定跑脚本。
import os
import subprocess
import sys

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")


def run_script(name, env_extra, inputs, outputs, cwd=None):
    """按 AUTONOMICS_INPUTn/OUTPUTn 约定执行 scripts/ 下脚本，返回 CompletedProcess。"""
    env = dict(os.environ)
    for i, path in enumerate(inputs):
        env[f"AUTONOMICS_INPUT{i}"] = str(path)
    for i, path in enumerate(outputs):
        env[f"AUTONOMICS_OUTPUT{i}"] = str(path)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, name)],
        env=env, capture_output=True, text=True, cwd=cwd,
    )


import pytest


@pytest.fixture
def run():
    return run_script
