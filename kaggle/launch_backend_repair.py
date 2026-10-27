"""Pinned backend repair checks, actual overflow traces and full GPU regression."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

PINNED_SHA = "__PINNED_SHA__"
REPO = Path("/tmp/amg-backend-repair")
OUT = Path("/kaggle/working/backend-repair")


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
        "AMG_RUN_GPU_TESTS": "1",
    }
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(REPO)],
        cwd=REPO,
        env=env,
        check=True,
    )
    status = {
        "sha": PINNED_SHA,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "phases": {},
    }

    def run(cmd, name, timeout):
        with (OUT / f"{name}.log").open("w") as f:
            try:
                r = subprocess.run(
                    cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout
                )
                status["phases"][name] = {"exit_code": r.returncode}
            except subprocess.TimeoutExpired:
                status["phases"][name] = {"exit_code": None, "status": "TIMEOUT_NOT_PASS"}
        (OUT / "execution.json").write_text(json.dumps(status, indent=2))
        print(name, status["phases"][name], flush=True)

    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_backend_mock_contracts.py",
            "tests/test_backend_update_remote.py",
            "tests/test_kaggle_backend.py",
            f"--basetemp={OUT / 'contracts-preserved'}",
            f"--junitxml={OUT / 'contracts.xml'}",
        ],
        "contracts",
        300,
    )
    for seed, forced in ((42, False), (43, False), (42, True)):
        name = f"seed{seed}" + ("-forced" if forced else "")
        first = OUT / name / "first"
        resumed = OUT / name / "resumed"
        common = [sys.executable, "scripts/trace_fp16_backend.py", "--ambient-seed", str(seed)]
        run(
            common
            + ["--output-dir", str(first), "--target", "2"]
            + (["--force-overflow"] if forced else []),
            name + "-first",
            90,
        )
        if (first / "latest.pt").exists():
            run(
                common
                + [
                    "--output-dir",
                    str(resumed),
                    "--target",
                    "4",
                    "--resume",
                    str(first / "latest.pt"),
                ],
                name + "-resumed",
                90,
            )
    if status["phases"]["contracts"]["exit_code"] == 0:
        run(
            [sys.executable, "-m", "pytest", "-q", f"--junitxml={OUT / 'full-suite.xml'}"],
            "full-suite",
            1200,
        )
    if any(v["exit_code"] != 0 for v in status["phases"].values()):
        raise RuntimeError("Backend phase failed; preserve all traces")


if __name__ == "__main__":
    main()
