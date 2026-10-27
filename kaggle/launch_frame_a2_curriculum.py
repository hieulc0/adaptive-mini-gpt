"""Pinned finite Stage A.2 intervention and final-source full regression."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

PINNED_SHA = "__PINNED_SHA__"
REPO = Path("/tmp/amg-frame-a2-curriculum")
OUT = Path("/kaggle/working/frame-a2-curriculum")


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
    status = {"sha": PINNED_SHA, "gpu": torch.cuda.get_device_name(0), "phases": {}}

    def run(cmd, name, timeout):
        with (OUT / f"{name}.log").open("w") as f:
            try:
                r = subprocess.run(
                    cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout
                )
                code = r.returncode
            except subprocess.TimeoutExpired:
                code = None
        status["phases"][name] = {"exit_code": code}
        (OUT / "execution.json").write_text(json.dumps(status, indent=2))
        return code == 0

    if run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_frame_a2_mock.py",
            "tests/test_frame_a2_remote.py",
            "tests/test_frame_curriculum_fixtures.py",
            "tests/test_backend_update_remote.py",
            f"--junitxml={OUT / 'contracts.xml'}",
        ],
        "contracts",
        240,
    ):
        roots = list(Path("/kaggle/input").rglob("inherited_weights_inherited_moments.pt"))
        if len(roots) != 1:
            raise RuntimeError("Missing exact persistent parent")
        diagnostics = list(Path("/kaggle/input").rglob("optimizer_inherit.pt"))
        if len(diagnostics) != 1:
            raise RuntimeError("Missing diagnostic provenance input")
        if not run(
            [
                sys.executable,
                "scripts/annotate_frame_a2_diagnostics.py",
                "--input-dir",
                str(diagnostics[0].parent),
                "--output-dir",
                str(OUT / "annotated-diagnostics"),
            ],
            "provenance",
            180,
        ):
            raise RuntimeError("Provenance verification failed; preserve raw inputs")
        run(
            [
                sys.executable,
                "scripts/run_frame_a2_curriculum.py",
                "--root",
                str(roots[0]),
                "--output-dir",
                str(OUT),
            ],
            "curriculum",
            900,
        )
        run(
            [sys.executable, "-m", "pytest", "-q", f"--junitxml={OUT / 'full-suite.xml'}"],
            "full-suite",
            1200,
        )
    if any(p["exit_code"] != 0 for p in status["phases"].values()):
        raise RuntimeError("Phase failed; preserve all artifacts")


if __name__ == "__main__":
    main()
