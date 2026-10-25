#!/usr/bin/env python3
"""CLI Entrypoint for Milestone M99 Master Benchmark.

Usage:
    python3 scripts/run_m99_benchmark.py --seeds 42 43 44 45 46 --device cuda --output-dir runs/m99_output --tier tier_20k
"""

import argparse
import sys
import torch

from adaptive_mini_gpt.self_directed_curriculum.adaptive_capacity_manager import CapacityTier
from adaptive_mini_gpt.self_directed_curriculum.benchmark_m99 import M99CurriculumDiscoveryBenchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone M99 Benchmark Runner")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="runs/m99_output")
    parser.add_argument("--epochs-trunk", type=int, default=150)
    parser.add_argument("--lr-trunk", type=float, default=0.01)
    parser.add_argument("--tier", type=str, default="tier_20k", choices=["tier_20k", "tier_30k", "tier_50k", "tier_100k", "tier_adaptive"])

    args = parser.parse_args()

    device = torch.device(args.device)
    tier = CapacityTier(args.tier)
    print(f"=== Milestone M99 Benchmark Runner (Device: {device} | Tier: {tier.value}) ===")

    report = M99CurriculumDiscoveryBenchmark.run_benchmark(
        device=device,
        seeds=args.seeds,
        output_dir=args.output_dir,
        epochs_trunk=args.epochs_trunk,
        lr_trunk=args.lr_trunk,
        capacity_tier=tier,
    )

    if report["status"] != "M99_QUALIFIED_PASS":
        print(f"\n[ERROR] Benchmark status: {report['status']}")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Milestone M99 Qualified Pass certified.")
        sys.exit(0)


if __name__ == "__main__":
    main()
