#!/usr/bin/env python3
"""CLI runner for Milestone M89 Benchmark."""

import argparse
import sys
from pathlib import Path

import torch

from adaptive_mini_gpt.continual_retention.benchmark_m89 import (
    M89RetentionBenchmark,
)
from adaptive_mini_gpt.model import MiniGPT, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Milestone M89 Continual Retention Benchmark")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--num-tasks", type=int, default=6)
    parser.add_argument("--budget-ceiling", type=int, default=20000)
    parser.add_argument("--epochs-per-task", type=int, default=35)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="runs/m89_output")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== Milestone M89 Benchmark Runner (Device: {device}) ===")

    cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = MiniGPT(cfg)

    ckpt_path = Path(args.checkpoint_dir) / "m18_pretrain_seed42.pt"
    if ckpt_path.exists():
        print(f"Loading base checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    else:
        print("Checkpoint not found, using initialized weights")

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    res = M89RetentionBenchmark.run_benchmark(
        model=model,
        device=device,
        seeds=args.seeds,
        num_tasks=args.num_tasks,
        budget_ceiling=args.budget_ceiling,
        epochs_per_task=args.epochs_per_task,
        output_dir=args.output_dir,
    )

    print(f"\nBenchmark finished with Status: {res['status']}")
    md_path = Path(args.output_dir) / "m89_report.md"
    if md_path.exists():
        print("\n" + md_path.read_text())


if __name__ == "__main__":
    main()
