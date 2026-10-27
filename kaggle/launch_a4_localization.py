"""Pinned inference-only Stage A.4 localization; never trains."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

PINNED_SHA = "__PINNED_SHA__"
REPO = Path("/tmp/amg-a4-localization")
OUT = Path("/kaggle/working/a4-localization")


def main():
    if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not torch.cuda.is_available():
        raise RuntimeError("Kaggle CUDA required")
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            "m104-e-and-m105-a",
            "https://github.com/hieulc0/adaptive-mini-gpt.git",
            str(REPO),
        ],
        check=True,
    )
    subprocess.run(["git", "checkout", PINNED_SHA], cwd=REPO, check=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO / "src"),
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    pre = list(Path("/kaggle/input").rglob("inherited_weights_inherited_moments.pt"))
    post = list(Path("/kaggle/input").rglob("augmented.pt"))
    if len(pre) != 1 or len(post) != 1:
        raise RuntimeError("Missing unique checkpoint inputs")
    with (OUT / "localization.log").open("w") as f:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_a4_localization.py",
                "--pre",
                str(pre[0]),
                "--post",
                str(post[0]),
                "--output-dir",
                str(OUT),
            ],
            cwd=REPO,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
    status = dict(
        sha=PINNED_SHA,
        gpu=torch.cuda.get_device_name(0),
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        exit_code=result.returncode,
    )
    (OUT / "execution.json").write_text(json.dumps(status, indent=2))
    if result.returncode:
        raise RuntimeError("Preserve inference diagnostic failure")


if __name__ == "__main__":
    main()
