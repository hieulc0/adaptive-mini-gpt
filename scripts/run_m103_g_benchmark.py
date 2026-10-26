#!/usr/bin/env python3
\"\"\"CLI runner for Milestone M103-G Master Benchmark.\"\"\"

import argparse
from pathlib import Path
import sys
import torch

from adaptive_mini_gpt.natural_language_learning_m103.benchmark_m103_g import (
    BenchmarkM103_G,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Milestone M103-G Master Benchmark")
    parser.add_argument("--steps", type=int, default=150, help="Optimization steps per phase")
    parser.add_argument("--output-dir", type=str, default="runs/m103_g_output", help="Directory for reports")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing M103-G Benchmark on {device} with {args.steps} steps...")

    out_path = Path(args.output_dir)
    benchmark = BenchmarkM103_G(device=device)
    report = benchmark.run_benchmark(steps=args.steps, output_dir=out_path)

    print("\n=== M103-G Execution Summary ===")
    print(f"Status: {report.overall_status}")
    print(f"Gates Passed: {report.gates_passed}/{report.total_gates}")
    print(f"Duration: {report.execution_duration_sec:.2f}s")

    for gate in report.acceptance_gates:
        status_sym = "[PASS]" if gate.passed else "[FAIL]"
        print(f"  {status_sym} {gate.gate_id}: {gate.name}")


if __name__ == "__main__":
    main()
