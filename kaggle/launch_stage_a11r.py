"""Pinned Stage A.11-R Scientific Metric Audit & Readout Diagnosis on Kaggle CUDA."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

PINNED_SHA = "c88b659948ad253a1e70f84c6516ae2cf35127e8"
REPO = Path("/tmp/amg-stage-a11r")
OUT = Path("/kaggle/working/stage-a11r")


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

    # Phase 1: Focused Contracts
    ok_contracts = run(
        [
            "-m",
            "pytest",
            "-q",
            "tests/test_stage_a11r_contracts.py",
            "tests/test_stage_a11_contracts.py",
            "tests/test_stage_a10_contracts.py",
            "tests/test_stage_a9_contracts.py",
            "tests/test_stage_a8_contracts.py",
            "tests/test_exposure_a7_mock.py",
            f"--junitxml={OUT / 'contracts.xml'}",
        ],
        "contracts",
        300,
    )

    # Phase 2: Tasks 1 & 2 Audit on Checkpoints
    if ok_contracts:
        audit_out = OUT / "stage_a11r_audit_v1.json"
        ok_audit = run(
            [
                "scripts/audit_stage_a11r.py",
                "--task",
                "all",
                "--out_file",
                str(audit_out),
            ],
            "audit_tasks_1_and_2",
            1200,
        )

        if ok_audit and audit_out.exists():
            (Path("/kaggle/working") / "stage_a11r_audit_v1.json").write_text(audit_out.read_text())

        # Phase 3: Task 3 Minimal Readout Experiment
        if ok_audit:
            ok_exp = run(
                [
                    "scripts/run_stage_a11r.py",
                    "--out_dir",
                    str(OUT),
                ],
                "stage_a11r_readout_experiment",
                7200,
            )

            # Copy manifests to OUT and /kaggle/working
            for manifest_name in ("stage_a11r_v1.json", "stage_a11r_audit_v1.json"):
                for src_m in (OUT / manifest_name, REPO / "data/manifests" / manifest_name):
                    if src_m.exists():
                        (OUT / manifest_name).write_text(src_m.read_text())
                        (Path("/kaggle/working") / manifest_name).write_text(src_m.read_text())
                        break

    # Phase 4: Full GPU Regression Suite
    run(
        ["-m", "pytest", "-q", f"--junitxml={OUT / 'full-suite.xml'}"],
        "full-suite",
        1800,
    )

    if any(p["exit_code"] != 0 for p in status["phases"].values()):
        raise RuntimeError("One or more phases failed on Kaggle CUDA; see logs in OUT")


if __name__ == "__main__":
    main()
