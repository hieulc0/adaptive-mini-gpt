"""Pinned Stage A.8 Neural Learning Investigation and Repair on Kaggle CUDA."""

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
REPO = Path("/tmp/amg-stage-a8")
OUT = Path("/kaggle/working/stage-a8")


def main():
    if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not torch.cuda.is_available():
        raise RuntimeError("Kaggle CUDA required; no CPU fallback")

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
        "PYTHONPATH": f"{REPO}/src:{REPO}/scripts",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "AMG_RUN_GPU_TESTS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(REPO)],
        cwd=REPO,
        env=env,
        check=True,
    )

    status: dict[str, Any] = dict(
        sha=PINNED_SHA,
        platform="kaggle",
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        phases={},
    )

    def run(args, name, timeout):
        with (OUT / f"{name}.log").open("w") as f:
            try:
                code = subprocess.run(
                    [sys.executable, *args],
                    cwd=REPO,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                ).returncode
            except subprocess.TimeoutExpired:
                code = None
        status["phases"][name] = dict(exit_code=code)
        (OUT / "execution.json").write_text(json.dumps(status, indent=2))
        return code == 0

    # Phase 1: Contracts
    ok = run(
        [
            "-m",
            "pytest",
            "-q",
            "tests/test_stage_a8_contracts.py",
            "tests/test_exposure_a7_mock.py",
            "tests/test_backend_update_remote.py",
            f"--junitxml={OUT / 'contracts.xml'}",
        ],
        "contracts",
        240,
    )

    # Phase 2: Stage A.8 Runner
    if ok:
        paths = list(Path("/kaggle/input").rglob("inherited_weights_inherited_moments.pt"))
        if len(paths) != 1:
            raise RuntimeError(
                f"Expected exactly 1 immutable persistent parent, found {len(paths)}"
            )

        run(
            [
                "scripts/run_stage_a8.py",
                "--root",
                str(paths[0]),
                "--output-dir",
                str(OUT),
            ],
            "stage_a8",
            2400,
        )

    # Phase 3: Full Regression Suite
    run(
        ["-m", "pytest", "-q", f"--junitxml={OUT / 'full-suite.xml'}"],
        "full-suite",
        1500,
    )

    if any(p["exit_code"] != 0 for p in status["phases"].values()):
        raise RuntimeError("One or more phases failed on Kaggle CUDA; see logs in OUT")


if __name__ == "__main__":
    main()
