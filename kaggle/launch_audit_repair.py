"""Render the full pushed SHA into an ignored job copy before submission."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

PINNED_SHA = "__PINNED_SHA__"
REPO = Path("/tmp/amg-audit-repair")
OUT = Path("/kaggle/working/audit-repair")


def run(command, name, timeout):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO / "src"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with (OUT / f"{name}.log").open("w") as log:
        try:
            result = subprocess.run(
                command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
            )
            return {"exit_code": result.returncode, "log": log.name}
        except subprocess.TimeoutExpired:
            return {"exit_code": None, "status": "TIMEOUT_NOT_PASS", "log": log.name}


def main():
    if (
        not Path("/kaggle/working").is_dir()
        or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("Kaggle CUDA required; refusing local/CPU execution")
    if len(PINNED_SHA) != 40:
        raise ValueError("Render committed and pushed source SHA")
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
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if actual != PINNED_SHA:
        raise RuntimeError("Source SHA mismatch")
    status = {
        "git_sha": actual,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "kernel_id": "sparklinghydraulics0/amg-approved-audit-repair",
        "phases": {},
    }

    def record(name, result):
        status["phases"][name] = result
        (OUT / "execution.json").write_text(json.dumps(status, indent=2))

    record(
        "install",
        run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(REPO)], "install", 180),
    )
    if status["phases"]["install"]["exit_code"] != 0:
        raise RuntimeError("Installation failed")
    baseline_test = OUT / "test_baseline_contracts.py"
    baseline_test.write_text((REPO / "tests/test_audit_neural_regressions.py").read_text())
    subprocess.run(
        ["git", "checkout", "--detach", "a6d8e11eda1ecfb7067f1552b7c8e7eb13d2eb11"],
        cwd=REPO,
        check=True,
    )
    for test in ("explicit_masks", "empty_target", "nul_byte"):
        record(
            f"baseline_expected_failure_{test}",
            run(
                [sys.executable, "-m", "pytest", "-q", str(baseline_test), "-k", test],
                f"baseline-{test}",
                90,
            ),
        )
    subprocess.run(["git", "checkout", "--detach", PINNED_SHA], cwd=REPO, check=True)
    record(
        "contracts",
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_architecture_audit_contracts.py",
                "tests/test_audit_neural_regressions.py",
                f"--junitxml={OUT / 'contracts.xml'}",
            ],
            "contracts",
            150,
        ),
    )
    if status["phases"]["contracts"]["exit_code"] == 0:
        record(
            "diagnosis_learning",
            run(
                [sys.executable, "scripts/run_audit_repair.py", "--output-dir", str(OUT)],
                "diagnosis-learning",
                900,
            ),
        )
    # Full suite runs independently of scientific competence; failures are preserved.
    record(
        "full_tests",
        run(
            [sys.executable, "-m", "pytest", "-q", f"--junitxml={OUT / 'full-tests.xml'}"],
            "full-tests",
            1200,
        ),
    )
    print(json.dumps(status, indent=2), flush=True)
    failures = [
        name
        for name, result in status["phases"].items()
        if (
            result["exit_code"] != 1
            if name.startswith("baseline_expected_failure")
            else result["exit_code"] != 0
        )
    ]
    if failures:
        raise RuntimeError(f"Failed phases, preserved artifacts: {failures}")


if __name__ == "__main__":
    main()
