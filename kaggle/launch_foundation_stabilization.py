"""Pinned, bounded foundation qualification. No local neural execution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

PINNED_SHA = "__PINNED_SHA__"
REPO = Path("/tmp/amg-foundation-stabilization")
OUT = Path("/kaggle/working/foundation-stabilization")


def run(args, label, timeout):
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(REPO / "src"),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        AMG_RUN_GPU_TESTS="1",
    )
    with (OUT / f"{label}.log").open("w") as f:
        try:
            p = subprocess.run(
                args, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout
            )
            return {"exit_code": p.returncode}
        except subprocess.TimeoutExpired:
            return {"exit_code": None, "status": "TIMEOUT_NOT_PASS"}


def main():
    if (
        not Path("/kaggle/working").is_dir()
        or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("Kaggle CUDA required")
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
    status = {
        "sha": PINNED_SHA,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "phases": {},
    }

    def record(name, result):
        status["phases"][name] = result
        (OUT / "execution.json").write_text(json.dumps(status, indent=2))

    record(
        "install",
        run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(REPO)], "install", 180),
    )
    baseline = OUT / "test_historical_foundation.py"
    baseline.write_text((REPO / "tests/test_foundation_structure.py").read_text())
    subprocess.run(
        ["git", "checkout", "--detach", "d7fbedd52cc052b0d296b02302f97ffcb228a9ab"],
        cwd=REPO,
        check=True,
    )
    for name in ("negative_logit", "dynamic_routing"):
        record(
            "historical_" + name,
            run(
                [sys.executable, "-m", "pytest", "-q", str(baseline), "-k", name],
                "historical-" + name,
                90,
            ),
        )
    alias_test = OUT / "test_historical_root_alias.py"
    subprocess.run(["git", "checkout", "--detach", PINNED_SHA], cwd=REPO, check=True)
    alias_test.write_text((REPO / "tests/test_foundation_mock_contracts.py").read_text())
    subprocess.run(
        ["git", "checkout", "--detach", "5a7b409fc0e8312708328fbddaca54cf0dd6e332"],
        cwd=REPO,
        check=True,
    )
    record(
        "historical_root_alias",
        run(
            [sys.executable, "-m", "pytest", "-q", str(alias_test), "-k", "rebinds"],
            "historical-root-alias",
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
                "tests/test_foundation_structure.py",
                "tests/test_foundation_mock_contracts.py",
                "tests/test_architecture_audit_contracts.py",
                "tests/test_audit_neural_regressions.py",
                f"--junitxml={OUT / 'contracts.xml'}",
            ],
            "contracts",
            240,
        ),
    )
    if status["phases"]["contracts"]["exit_code"] == 0:
        record(
            "full_suite",
            run(
                [sys.executable, "-m", "pytest", "-q", f"--junitxml={OUT / 'full-suite.xml'}"],
                "full-suite",
                1200,
            ),
        )
    if any(
        v["exit_code"] != (1 if k.startswith("historical_") else 0)
        for k, v in status["phases"].items()
    ):
        raise RuntimeError("Foundation phases failed; inspect preserved artifacts")


if __name__ == "__main__":
    main()
