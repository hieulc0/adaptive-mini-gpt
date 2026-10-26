#!/usr/bin/env python3
"""Milestone M103-B CLI Benchmark Runner: Neural Language Acquisition.

Usage:
    python scripts/run_m103_b_benchmark.py --seed 42 --tier tier_20k --output-dir runs/m103_b_output
"""

import argparse
from pathlib import Path
import sys
import torch

from adaptive_mini_gpt.integrated_autonomous_student.structural_evolution_engine import (
    CapacityTierM100,
)
from adaptive_mini_gpt.natural_language_learning_m103.benchmark_m103_b import (
    M103BNeuralLanguageBenchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Milestone M103-B Neural Language Acquisition Benchmark"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Evaluation seed",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="tier_20k",
        choices=["tier_20k", "tier_30k", "tier_50k", "tier_100k", "tier_adaptive"],
        help="Capacity tier",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/m103_b_output",
        help="Directory to save report artifacts",
    )
    return parser.parse_args()


def main() -> None:
    from adaptive_mini_gpt.training.legacy_entrypoints import require_historical_reproduction

    require_historical_reproduction("run_m103_b_benchmark.py")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tier = CapacityTierM100(args.tier)

    report = M103BNeuralLanguageBenchmark.run_benchmark(
        device=device,
        seed=args.seed,
        tier=tier,
        output_dir=args.output_dir,
    )

    print(f"\n[+] M103-B Benchmark Completed with Status: {report.status}")
    out_path = Path(args.output_dir)
    print(f"[+] Artifacts written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
