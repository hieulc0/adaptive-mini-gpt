"""Pinned short GPU numerical reproducibility probe; no learning qualification."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

PINNED_SHA = "__PINNED_SHA__"
REPO = Path("/tmp/amg-numerics-probe")
OUT = Path("/kaggle/working/numerics-probe")


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
    paths = list(Path("/kaggle/input").rglob("inherited_weights_inherited_moments.pt"))
    if len(paths) != 1:
        raise RuntimeError("Missing immutable parent")
    status: dict[str, Any] = dict(
        sha=PINNED_SHA,
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        phases={},
    )
    for mode in ("default", "strict"):
        env = {
            **os.environ,
            "PYTHONPATH": str(REPO / "src"),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }
        env.pop("CUBLAS_WORKSPACE_CONFIG", None)
        if mode == "strict":
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        args = [
            sys.executable,
            "scripts/run_stable_numerics_probe.py",
            "--root",
            str(paths[0]),
            "--output",
            str(OUT / f"{mode}.json"),
        ]
        if mode == "strict":
            args.append("--strict")
        with (OUT / f"{mode}.log").open("w") as f:
            result = subprocess.run(
                args, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=180
            )
        status["phases"][mode] = result.returncode
        (OUT / "execution.json").write_text(json.dumps(status, indent=2))
    if any(status["phases"].values()):
        raise RuntimeError("Probe process failed; preserve traces")


if __name__ == "__main__":
    main()
