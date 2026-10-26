"""Render __PINNED_SHA__ to a committed/pushed full SHA in an ignored job copy."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

PINNED_SHA = "__PINNED_SHA__"
REPO = Path("/tmp/amg-foundational-audit")
OUT = Path("/kaggle/working/foundational-audit")


def run(command, name, timeout):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
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
        raise RuntimeError("Kaggle CUDA runtime required; refusing local/CPU execution")
    if len(PINNED_SHA) != 40:
        raise ValueError("Render full committed and pushed SHA before submitting")
    OUT.mkdir(parents=True, exist_ok=True)
    if REPO.exists():
        raise RuntimeError("Expected fresh job checkout")
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
        "kernel_id": "sparklinghydraulics0/amg-foundational-audit",
        "phases": {},
    }
    status["phases"]["audit_tests"] = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_architecture_audit_contracts.py",
            f"--junitxml={OUT / 'audit-tests.xml'}",
        ],
        "audit-tests",
        90,
    )
    (OUT / "execution.json").write_text(json.dumps(status, indent=2))
    if status["phases"]["audit_tests"]["exit_code"] == 0:
        status["phases"]["feasibility"] = run(
            [
                sys.executable,
                "scripts/run_foundational_audit.py",
                "--seed",
                "42",
                "--steps",
                "200",
                "--output-dir",
                str(OUT),
            ],
            "feasibility",
            600,
        )
        (OUT / "execution.json").write_text(json.dumps(status, indent=2))
        report_path = OUT / "report.json"
        competent = report_path.exists() and any(
            a.get("sealed", {}).get("micro_f1", 0) >= 0.75
            for a in json.loads(report_path.read_text())["arms"].values()
        )
        if status["phases"]["feasibility"]["exit_code"] == 0 and competent:
            status["phases"]["full_tests"] = run(
                [sys.executable, "-m", "pytest", "-q", f"--junitxml={OUT / 'full-tests.xml'}"],
                "full-tests",
                600,
            )
        else:
            status["phases"]["full_tests"] = {
                "status": "NOT_RUN_BASIC_COMPETENCE_OR_FEASIBILITY_FAILED"
            }
    (OUT / "execution.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2), flush=True)
    if any(p.get("exit_code", 0) != 0 for p in status["phases"].values()):
        raise RuntimeError("Remote phase failed or timed out; inspect preserved artifacts")


if __name__ == "__main__":
    main()
