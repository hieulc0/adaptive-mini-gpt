#!/usr/bin/env python3
"""CLI runner for Milestone M104-B: Grounded Concepts and Active Learning Benchmark."""

import argparse
from pathlib import Path

import torch

from adaptive_mini_gpt.natural_language_learning_m103.benchmark_m104_b import (
    BenchmarkM104_B,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Milestone M104-B Master Benchmark")
    parser.add_argument("--steps", type=int, default=150, help="Optimization steps per phase")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="List of seeds to evaluate")
    parser.add_argument("--output-dir", type=str, default="runs/m104_b_output", help="Directory for reports")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing M104-B Benchmark on {device} across seeds {args.seeds} with {args.steps} steps...", flush=True)

    out_path = Path(args.output_dir)
    benchmark = BenchmarkM104_B(device=device)
    report = benchmark.run_benchmark(seeds=args.seeds, steps=args.steps, output_dir=out_path)

    print("\n=== M104-B Execution Summary ===", flush=True)
    print(f"Status: {report.overall_status}", flush=True)
    print(f"Gates Passed: {report.gates_passed}/{report.total_gates}", flush=True)
    print(f"Duration: {report.execution_duration_sec:.2f}s", flush=True)

    for gate in report.acceptance_gates:
        status_sym = "[PASS]" if gate.passed else "[FAIL]"
        print(f"  {status_sym} {gate.gate_id}: {gate.name}", flush=True)
        print(f"         Metric: {gate.scientific_metric}", flush=True)
        print(f"         Evidence: {gate.evidence}", flush=True)


if __name__ == "__main__":
    main()
