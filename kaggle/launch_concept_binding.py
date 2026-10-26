"""Canonical foundation recheck then one bounded concept study; pinned source."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

PINNED_SHA = "__PINNED_SHA__"
REPO = Path("/tmp/amg-concept-binding")
OUT = Path("/kaggle/working/concept-binding")


def run(command, name, timeout):
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(REPO / "src"),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        AMG_RUN_GPU_TESTS="1",
    )
    with (OUT / f"{name}.log").open("w") as f:
        try:
            result = subprocess.run(
                command, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout
            )
            return {"exit_code": result.returncode}
        except subprocess.TimeoutExpired:
            return {"exit_code": None, "status": "TIMEOUT_NOT_PASS"}


def main():
    if (
        not Path("/kaggle/working").is_dir()
        or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("Kaggle CUDA required; refusing CPU fallback")
    if len(PINNED_SHA) != 40:
        raise ValueError("Render pushed SHA")
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
    subprocess.run(["git", "checkout", "--detach", PINNED_SHA], cwd=REPO, check=True)
    status = {"sha": PINNED_SHA, "gpu": torch.cuda.get_device_name(0), "phases": {}}

    def record(name, result):
        status["phases"][name] = result
        (OUT / "execution.json").write_text(json.dumps(status, indent=2))

    record(
        "install",
        run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(REPO)], "install", 180),
    )
    record(
        "contracts",
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_foundation_structure.py",
                "tests/test_foundation_mock_contracts.py",
                f"--junitxml={OUT / 'contracts.xml'}",
            ],
            "contracts",
            240,
        ),
    )
    if status["phases"]["contracts"]["exit_code"] == 0:
        record(
            "concept_study",
            run(
                [sys.executable, "scripts/run_concept_binding.py", "--output-dir", str(OUT)],
                "study",
                600,
            ),
        )
        record(
            "full_suite",
            run(
                [sys.executable, "-m", "pytest", "-q", f"--junitxml={OUT / 'full-suite.xml'}"],
                "full-suite",
                1200,
            ),
        )
    if any(p["exit_code"] != 0 for p in status["phases"].values()):
        raise RuntimeError("Phase failed; preserve negative artifacts")


if __name__ == "__main__":
    main()
