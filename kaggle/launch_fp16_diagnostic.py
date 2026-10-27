"""Pinned historical backend traces; no source mutation or local CPU fallback."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

PINNED_SHA = "__PINNED_SHA__"
BASELINE = "2e228ea"
REPO = Path("/tmp/amg-fp16-diagnostic")
OUT = Path("/kaggle/working/fp16-diagnostic")


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
    shutil.copy(REPO / "scripts/trace_fp16_backend.py", OUT / "trace_fp16_backend.py")
    subprocess.run(["git", "checkout", BASELINE], cwd=REPO, check=True)
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
        "instrumentation_sha": PINNED_SHA,
        "baseline_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "gpu": torch.cuda.get_device_name(0),
        "phases": {},
    }

    def run(cmd, name):
        with (OUT / f"{name}.log").open("w") as f:
            try:
                r = subprocess.run(
                    cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=240
                )
                status["phases"][name] = {"exit_code": r.returncode}
            except subprocess.TimeoutExpired:
                status["phases"][name] = {"exit_code": None, "status": "TIMEOUT_NOT_PASS"}
        (OUT / "execution.json").write_text(json.dumps(status, indent=2))

    for seed, forced in ((42, False), (43, False), (42, True)):
        name = f"seed{seed}" + ("-forced" if forced else "")
        first = OUT / name / "first"
        resumed = OUT / name / "resumed"
        common = [sys.executable, str(OUT / "trace_fp16_backend.py"), "--ambient-seed", str(seed)]
        run(
            common
            + ["--output-dir", str(first), "--target", "2"]
            + (["--force-overflow"] if forced else []),
            name + "-first",
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
            )
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_kaggle_backend.py::test_gpu_qualification",
            f"--basetemp={OUT / 'pytest-preserved'}",
            f"--junitxml={OUT / 'original-qualification.xml'}",
        ],
        "original-qualification",
    )
    # Failures are diagnostic observations, never translated into qualification PASS.


if __name__ == "__main__":
    main()
