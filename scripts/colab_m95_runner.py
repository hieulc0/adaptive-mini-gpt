"""Remote execution script for Google Colab T4: Milestone M95."""

import os
import subprocess
import sys


def run_cmd(cmd: str) -> str:
    print(f"\n>>> Running: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("STDERR:\n", res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {cmd}")
    return res.stdout


def main() -> None:
    print("=== Colab Preflight Verification ===")
    import torch
    assert torch.cuda.is_available(), "CUDA is not available on Colab!"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"CUDA Available: True | Device: {gpu_name} | PyTorch: {torch.__version__}")

    # 1. Pull/Clone repo branch from GitHub
    print("=== Syncing Repository Branch from GitHub ===")
    run_cmd("rm -rf /content/adaptive-mini-gpt && git clone -b m95-cross-domain-transfer https://github.com/hieulc0/adaptive-mini-gpt.git /content/adaptive-mini-gpt")

    # 2. Install editable package
    print("=== Installing Package ===")
    run_cmd("pip install -e /content/adaptive-mini-gpt pytest")

    # 3. Run Milestone M95 Tests and Historical Regression Suite on Tesla T4
    print("=== Running Milestone M95 Tests on Tesla T4 ===")
    os.chdir("/content/adaptive-mini-gpt")
    run_cmd("python3 -m pytest -q tests/test_m95_cross_domain_transfer.py")

    print("=== Running Historical Regression Suite (M80-M95) on Tesla T4 ===")
    reg_out = run_cmd("python3 -m pytest -q tests/test_m8[0-9]*.py tests/test_m9[0-5]*.py")
    print("Regression suite summary snippet:\n", reg_out[-500:] if len(reg_out) > 500 else reg_out)

    # 4. Run Milestone M95 Benchmark
    print("=== Executing Milestone M95 Master Benchmark on Tesla T4 ===")
    run_cmd(
        "python3 scripts/run_m95_benchmark.py "
        "--seeds 42 43 44 45 46 "
        "--device cuda "
        "--output-dir /content/runs/m95_output "
        "--epochs-trunk 250 "
        "--lr-trunk 0.01"
    )

    # 5. Archive results
    print("=== Packaging Benchmark Outputs ===")
    run_cmd("tar -czf /content/m95_output.tar.gz -C /content/runs m95_output")
    print("=== Remote Job Successfully Finished ===")


if __name__ == "__main__":
    main()
