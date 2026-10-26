#!/usr/bin/env python3
"""CLI runner for Milestone M104-D: Language Grounding and Multi-World Forward Transfer."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from adaptive_mini_gpt.natural_language_learning_m103.benchmark_m104_d import (
    BenchmarkRunnerM104_D,
)


def main() -> None:
    from adaptive_mini_gpt.training.legacy_entrypoints import require_historical_reproduction

    require_historical_reproduction("run_m104_d_benchmark.py")
    parser = argparse.ArgumentParser(description="Run Milestone M104-D Master Benchmark")
    parser.add_argument("--steps", type=int, default=150, help="Adaptation budget steps")
    parser.add_argument("--grounding-steps", type=int, default=60, help="Language grounding optimization steps")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="Seeds to evaluate")
    parser.add_argument("--output-dir", type=str, default="runs/m104_d_output", help="Directory for output deliverables")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing M104-D Benchmark on {device} across seeds {args.seeds}...", flush=True)

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunnerM104_D(device=device)
    report = runner.run_full_benchmark(
        seeds=args.seeds,
        adaptation_budget=args.steps,
        grounding_steps=args.grounding_steps,
    )

    # Save deliverables
    json_path = out_path / "m104_d_milestone_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    md_report = runner.generate_markdown_report(report)
    md_path = out_path / "m104_d_scientific_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    print("\n=== M104-D Execution Summary ===", flush=True)
    print(f"Verdict: {report.verdict}", flush=True)
    print(f"Gates Passed: {report.passed_gates_count}/{report.total_gates_count}", flush=True)
    print(f"Execution Time: {report.execution_time_seconds:.2f}s", flush=True)

    for gate in report.gates:
        status_sym = "[PASS]" if gate.passed else "[FAIL]"
        print(f"  {status_sym} {gate.gate_name}: {gate.observed} (Threshold: {gate.threshold})", flush=True)

    print(f"\nDeliverables written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
