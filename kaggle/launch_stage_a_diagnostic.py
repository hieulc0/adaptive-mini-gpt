"""Stage A development factorial and history/loss diagnostics; no qualification."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

PINNED_SHA = "__PINNED_SHA__"
REPO = Path("/tmp/amg-stage-a-diagnostic")
OUT = Path("/kaggle/working/stage-a-diagnostic")


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
                "tests/test_attribute_learning_fixtures.py",
                "tests/test_attribute_diagnostic_fixtures.py",
                "tests/test_backend_mock_contracts.py",
                "tests/test_backend_update_remote.py",
                "tests/test_attribute_learning_remote.py",
                f"--junitxml={OUT / 'contracts.xml'}",
            ],
            "contracts",
            240,
        ),
    )
    if status["phases"]["contracts"]["exit_code"] == 0:
        parents = list(Path("/kaggle/input").rglob("selective_plasticity.pt"))
        if len(parents) != 1:
            record("parent", {"exit_code": 1, "found": [str(p) for p in parents]})
            raise RuntimeError("Attach the prior concept-binding kernel output; no fresh fallback")
        parent = parents[0]
        latest = list(Path("/kaggle/input").rglob("continuing_selective_two.pt"))
        if len(latest) != 1:
            raise RuntimeError("Attach latest failed Stage A output; no fresh fallback")
        record(
            "development_diagnostic",
            run(
                [
                    sys.executable,
                    "scripts/run_stage_a_diagnostic.py",
                    "--output-dir",
                    str(OUT),
                    "--original",
                    str(parent),
                    "--latest",
                    str(latest[0]),
                ],
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
