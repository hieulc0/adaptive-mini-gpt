#!/usr/bin/env python3
"""CLI Entrypoint for Milestone M98 Master Benchmark.

Usage:
    python3 scripts/run_m98_benchmark.py --seeds 42 43 44 45 46 --device cuda --output-dir runs/m98_output
"""

import argparse
import sys
import torch

from adaptive_mini_gpt.cross_subject_skills.benchmark_m98 import M98CrossSubjectBenchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone M98 Benchmark Runner")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="runs/m98_output")
    parser.add_argument("--epochs-trunk", type=int, default=80)
    parser.add_argument("--lr-trunk", type=float, default=0.03)

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== Milestone M98 Benchmark Runner (Device: {device}) ===")

    report = M98CrossSubjectBenchmark.run_benchmark(
        device=device,
        seeds=args.seeds,
        output_dir=args.output_dir,
        epochs_trunk=args.epochs_trunk,
        lr_trunk=args.lr_trunk,
    )

    if report["status"] != "M98_QUALIFIED_PASS":
        print(f"\n[ERROR] Benchmark status: {report['status']}")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Milestone M98 Qualified Pass certified.")
        sys.exit(0)


if __name__ == "__main__":
    main()
