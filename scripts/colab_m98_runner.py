#!/usr/bin/env python3
"""Remote Colab Orchestration Script for Milestone M98.

Executed via: uvx --from google-colab-cli colab exec -s m98-t4 -f scripts/colab_m98_runner.py
"""

import os
import subprocess
import sys


def run_cmd(cmd: str) -> str:
    print(f"\n>>> Running: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout)
    if res.stderr:
        print("STDERR:\n", res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {cmd}")
    return res.stdout


def main() -> None:
    # 1. Clone or pull repo on remote Colab instance
    if not os.path.exists("/content/adaptive-mini-gpt"):
        print("=== Cloning repo on remote Colab ===")
        run_cmd("git clone -b m98-cross-subject-skills https://github.com/hieulc0/adaptive-mini-gpt.git /content/adaptive-mini-gpt")
    else:
        print("=== Pulling latest changes on remote Colab ===")
        run_cmd("cd /content/adaptive-mini-gpt && git fetch origin && git checkout -f m98-cross-subject-skills && git reset --hard origin/m98-cross-subject-skills")

    # 2. Install package in editable mode
    print("=== Installing adaptive-mini-gpt ===")
    run_cmd("pip install -e /content/adaptive-mini-gpt")

    # 3. Run Pytest Suite on Tesla T4
    print("=== Running Milestone M98 Tests on Tesla T4 ===")
    os.chdir("/content/adaptive-mini-gpt")
    run_cmd("python3 -m pytest -q tests/test_m98_cross_subject_skills.py")

    print("=== Running Historical Regression Suite (M80-M98) on Tesla T4 ===")
    reg_out = run_cmd("python3 -m pytest -q tests/test_m8[0-9]*.py tests/test_m9[0-8]*.py")
    print("Regression suite summary snippet:\n", reg_out[-500:] if len(reg_out) > 500 else reg_out)

    # 4. Run Milestone M98 Benchmark
    print("=== Executing Milestone M98 Master Benchmark on Tesla T4 ===")
    run_cmd("python3 scripts/run_m98_benchmark.py --seeds 42 43 44 45 46 --device cuda --output-dir /content/runs/m98_output --epochs-trunk 250 --lr-trunk 0.01")

    # 5. Tar results for download
    print("=== Packaging Benchmark Outputs ===")
    run_cmd("tar -czf /content/m98_output.tar.gz -C /content/runs m98_output")
    print("\n=== Remote Job Successfully Finished ===")


if __name__ == "__main__":
    main()
