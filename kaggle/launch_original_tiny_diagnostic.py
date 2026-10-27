"""Bounded original-training optimizer ablation only, no main/heldout training."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

PINNED_SHA = "__PINNED_SHA__"
REPO = Path("/tmp/amg-original-tiny-diagnostic")
OUT = Path("/kaggle/working/original-tiny-diagnostic")


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
    parents = list(Path("/kaggle/input").rglob("selective_plasticity.pt"))
    if len(parents) != 1:
        raise RuntimeError("Original parent input missing")
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO / "src"),
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }
    with (OUT / "diagnostic.log").open("w") as f:
        try:
            r = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_original_tiny_diagnostic.py",
                    "--parent",
                    str(parents[0]),
                    "--output-dir",
                    str(OUT),
                ],
                cwd=REPO,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
            result = {"sha": PINNED_SHA, "exit_code": r.returncode}
        except subprocess.TimeoutExpired:
            result = {"sha": PINNED_SHA, "exit_code": None, "status": "TIMEOUT_NOT_PASS"}
    (OUT / "execution.json").write_text(json.dumps(result, indent=2))
    if result["exit_code"] != 0:
        raise RuntimeError("Tiny diagnostic failed; preserve trace")


if __name__ == "__main__":
    main()
