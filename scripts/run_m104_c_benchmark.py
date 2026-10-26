#!/usr/bin/env python3
"""CLI runner for Milestone M104-C: Grounded Neural Learning and Learning-to-Learn."""

import argparse
from pathlib import Path

import torch

from adaptive_mini_gpt.natural_language_learning_m103.benchmark_m104_c import (
    BenchmarkM104_C,
)


def main() -> None:
    from adaptive_mini_gpt.training.legacy_entrypoints import require_historical_reproduction

    require_historical_reproduction("run_m104_c_benchmark.py")
    parser = argparse.ArgumentParser(description="Run Milestone M104-C Master Benchmark")
    parser.add_argument("--steps", type=int, default=150, help="Optimization steps per phase")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="List of seeds to evaluate")
    parser.add_argument("--output-dir", type=str, default="runs/m104_c_output", help="Directory for reports")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing M104-C Benchmark on {device} across seeds {args.seeds} with {args.steps} steps...", flush=True)

    out_path = Path(args.output_dir)
    benchmark = BenchmarkM104_C(device=device)
    report = benchmark.run_benchmark(seeds=args.seeds, steps=args.steps, output_dir=out_path)

    print("\n=== M104-C Execution Summary ===", flush=True)
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
