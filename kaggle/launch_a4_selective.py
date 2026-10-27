"""Pinned Stage A.4 selective feasibility and full regression on Kaggle CUDA."""

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
REPO = Path("/tmp/amg-a4-selective")
OUT = Path("/kaggle/working/a4-selective")


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
    inspection = json.loads((REPO / "data/manifests/a4_localization_v1.json").read_text())
    if (
        inspection["status"] != "COMPLETE"
        or inspection["optimizer_steps"] != 0
        or not inspection["checkpoint_files_unchanged"]
        or inspection["pre_file_sha256"]
        != "3e93d7238bfdb706ddfd49ee9903fed8b1e6f577529537f1e21cae12e905b262"
        or inspection["post_file_sha256"]
        != "02dcbe3feedd975df34222b2a5a38e0481bedc7a52fe56dff7d655bb14c7adf3"
    ):
        raise RuntimeError("Completed inference-only localization is required before training")
    replay = json.loads((REPO / "data/manifests/a4_replay_v1.json").read_text())
    if replay["status"] != "COMPLETE" or replay["qualification_admission"]:
        raise RuntimeError("A completed, insufficient replay study is required first")
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO / "src"),
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

    ok = run(
        [
            "-m",
            "pytest",
            "-q",
            "tests/test_a4_selective_mock.py",
            "tests/test_a4_selective_remote.py",
            "tests/test_backend_update_remote.py",
            f"--junitxml={OUT / 'contracts.xml'}",
        ],
        "contracts",
        240,
    )
    if ok:
        paths = list(Path("/kaggle/input").rglob("inherited_weights_inherited_moments.pt"))
        if len(paths) != 1:
            raise RuntimeError("Missing immutable persistent parent")
        run(
            ["scripts/run_a4_selective.py", "--root", str(paths[0]), "--output-dir", str(OUT)],
            "selective",
            900,
        )
    run(["-m", "pytest", "-q", f"--junitxml={OUT / 'full-suite.xml'}"], "full-suite", 1200)
    if any(p["exit_code"] != 0 for p in status["phases"].values()):
        raise RuntimeError("Preserve failed phase and artifacts")


if __name__ == "__main__":
    main()
