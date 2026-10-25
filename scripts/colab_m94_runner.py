"""Remote execution script for Google Colab T4: Milestone M94."""

import os
import subprocess
import sys


def run_cmd(cmd: str) -> None:
    print(f"\n>>> Running: {cmd}")
    res = subprocess.run(cmd, shell=True, check=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {cmd}")


def main() -> None:
    print("=== Colab Preflight Verification ===")
    import torch
    assert torch.cuda.is_available(), "CUDA is not available on Colab!"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"CUDA Available: True | Device: {gpu_name} | PyTorch: {torch.__version__}")

    # 1. Unpack repo
    print("=== Unpacking Repository Archive ===")
    run_cmd("rm -rf /content/adaptive-mini-gpt && mkdir -p /content/adaptive-mini-gpt")
    run_cmd("tar -xzf /content/repo_m94.tar.gz -C /content/adaptive-mini-gpt")

    # 2. Install editable package
    print("=== Installing Package ===")
    run_cmd("pip install -e /content/adaptive-mini-gpt pytest")

    # 3. Run Pytest Suite on Tesla T4
    print("=== Running Pytest Suite on Tesla T4 ===")
    os.chdir("/content/adaptive-mini-gpt")
    run_cmd("python3 -m pytest -q tests/test_m94_neural_program_learning.py")

    # 4. Run Milestone M94 Benchmark
    print("=== Executing Milestone M94 Master Benchmark on Tesla T4 ===")
    run_cmd(
        "python3 scripts/run_m94_benchmark.py "
        "--seeds 42 43 44 45 46 "
        "--device cuda "
        "--output-dir /content/runs/m94_output "
        "--epochs-trunk 250 "
        "--lr-trunk 0.01"
    )

    # 5. Archive results
    print("=== Packaging Benchmark Outputs ===")
    run_cmd("tar -czf /content/m94_output.tar.gz -C /content/runs m94_output")
    print("=== Remote Job Successfully Finished ===")


if __name__ == "__main__":
    main()
