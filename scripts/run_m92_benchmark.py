#!/usr/bin/env python3
"""CLI runner for Milestone M92 Benchmark: Autonomous Rule Discovery & Neural Acquisition."""

import argparse
from pathlib import Path

import torch

from adaptive_mini_gpt.autonomous_discovery.benchmark_m92 import (
    M92DiscoveryBenchmark,
)
from adaptive_mini_gpt.model import MiniGPT, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Milestone M92 Benchmark: Autonomous Rule Discovery & Persistent Neural Knowledge"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--output-dir", type=str, default="runs/m92_output")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs-trunk", type=int, default=70)
    parser.add_argument("--lr-trunk", type=float, default=0.02)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== Milestone M92 Benchmark Runner (Device: {device}) ===")

    cfg = ModelConfig(
        vocab_size=1024,
        context_length=128,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
    )
    model = MiniGPT(cfg)

    ckpt_path = Path(args.checkpoint_dir) / "m18_pretrain_seed42.pt"
    if ckpt_path.exists():
        print(f"Loading base checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    else:
        print("Checkpoint not found or using standardized frozen MiniGPT backbone")

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    res = M92DiscoveryBenchmark.run_benchmark(
        model=model,
        device=device,
        seeds=args.seeds,
        output_dir=args.output_dir,
        epochs_trunk=args.epochs_trunk,
        lr_trunk=args.lr_trunk,
    )

    print(f"\nBenchmark finished with Status: {res['status']}")
    md_path = Path(args.output_dir) / "m92_report.md"
    if md_path.exists():
        print("\n" + md_path.read_text())


if __name__ == "__main__":
    main()
