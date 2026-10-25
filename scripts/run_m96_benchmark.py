#!/usr/bin/env python3
"""CLI runner for Milestone M96: Persistent Factual Memory & Long-Horizon Learning."""

import argparse
import sys
from pathlib import Path

import torch

from adaptive_mini_gpt.model import MiniGPT, MiniGPTConfig
from adaptive_mini_gpt.persistent_memory.benchmark_m96 import (
    M96PersistentMemoryBenchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Milestone M96 Benchmark")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="Random seeds")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="runs/m96_output", help="Output directory")
    parser.add_argument("--epochs-trunk", type=int, default=250, help="Epochs for concept trunk pretraining")
    parser.add_argument("--lr-trunk", type=float, default=0.01, help="Learning rate for concept trunk")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== Milestone M96 Benchmark Runner (Device: {device}) ===")

    config = MiniGPTConfig(
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
    )
    model = MiniGPT(config).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    report = M96PersistentMemoryBenchmark.run_benchmark(
        model=model,
        device=device,
        seeds=args.seeds,
        output_dir=args.output_dir,
        epochs_trunk=args.epochs_trunk,
        lr_trunk=args.lr_trunk,
    )

    print(f"\nBenchmark finished with Status: {report['status']}")
    if report["status"] != "M96_QUALIFIED_PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
