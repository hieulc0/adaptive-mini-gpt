#!/usr/bin/env python3
"""CLI runner for Milestone M97: Adaptive Context Length, Working Memory & Lifelong Learning."""

import argparse
import sys
from pathlib import Path

import torch

from adaptive_mini_gpt.adaptive_context.benchmark_m97 import (
    M97AdaptiveContextBenchmark,
)
from adaptive_mini_gpt.model import MiniGPT, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Milestone M97 Benchmark")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="Random seeds")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="runs/m97_output", help="Output directory")
    parser.add_argument("--epochs-trunk", type=int, default=250, help="Epochs for concept trunk pretraining")
    parser.add_argument("--lr-trunk", type=float, default=0.01, help="Learning rate for concept trunk")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== Milestone M97 Benchmark Runner (Device: {device}) ===")

    config = ModelConfig(
        vocab_size=1024,
        context_length=128,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        pos_encoding="absolute",
    )
    model = MiniGPT(config).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    report = M97AdaptiveContextBenchmark.run_benchmark(
        model=model,
        device=device,
        seeds=args.seeds,
        output_dir=args.output_dir,
        epochs_trunk=args.epochs_trunk,
        lr_trunk=args.lr_trunk,
    )

    print(f"\nBenchmark finished with Status: {report['status']}")
    if report["status"] != "M97_QUALIFIED_PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
