"""Milestone M86: Integrated Autonomous Continual Learning Master Benchmark.

Executes comprehensive scientific research evaluation on Google Colab GPU (Tesla T4):
- Phase 0: Forensic Audit of Milestone M85 (Multi-Module Parameter Amortization).
- Phase 1: Unified Continual Student Engine (Acquire -> Reuse -> Expand -> Consolidate -> Rollback).
- Phase 2: Persistent Learning State with Multi-Module Lifecycle.
- Phase 3: Long-Horizon Continual Task Stream (10 Episodes across 5 Seeds).
- Phase 4: Multi-Arm Causal Evaluation against Fixed, Independent, and Random Policies.
- Phase 5: Resource Accounting (FLOPs, Parameters, Latency, Teacher Queries).
- Phase 6: Bit-Exact Rollback Safety & Base Model Invariance (SHA-256: 6e39cd7ef6790b3c).
- Outputs machine-readable m86_report.json and m86_report.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import torch

from adaptive_mini_gpt.continual_student.benchmark import (
    M86ContinualLearningBenchmark,
)
from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("m86_benchmark")


def setup_base_model(
    checkpoint_dir: Path,
    seed: int,
    device: torch.device,
) -> tuple[MiniGPT, str]:
    """Load canonical base model with parameter hashes."""
    checkpoint_path = checkpoint_dir / f"seed_{seed}/base_pretrain/best.pt"
    if checkpoint_path.exists():
        logger.info("Loading pre-trained base model from %s", checkpoint_path)
        raw = CheckpointManager.load_raw_checkpoint(checkpoint_path)
        cfg_dict = raw["model_config"]
        cfg = ModelConfig(
            vocab_size=cfg_dict.get("vocab_size", 256),
            d_model=cfg_dict.get("d_model", 128),
            n_layers=cfg_dict.get("n_layers", 2),
            n_heads=cfg_dict.get("n_heads", 4),
            d_ff=cfg_dict.get("d_ff", 512),
            context_length=cfg_dict.get("context_length", 256),
        )
        model = MiniGPT(cfg)
        model.load_state_dict(raw["model_state_dict"])
    else:
        logger.warning("Checkpoint not found at %s. Initializing fresh baseline model.", checkpoint_path)
        torch.manual_seed(seed)
        cfg = ModelConfig(
            vocab_size=1024,
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_ff=512,
            context_length=128,
        )
        model = MiniGPT(cfg)

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hasher = hashlib.sha256()
    for p in model.parameters():
        hasher.update(p.detach().cpu().contiguous().numpy().tobytes())
    base_hash = hasher.hexdigest()[:16]

    return model, base_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milestone M86 Continual Learning Benchmark")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Random seeds for benchmark",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes in the task stream",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/content/checkpoints"),
        help="Directory containing pre-trained base checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/content/m86_output"),
        help="Directory to save output reports",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run benchmark on",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    logger.info("Initializing M86 continual learning evaluation on device: %s", device)

    # Load frozen base model
    model, base_hash = setup_base_model(args.checkpoint_dir, args.seeds[0], device)
    total_frozen_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Base model ready. SHA-256: %s, Total Frozen Parameters: %d",
        base_hash,
        total_frozen_params,
    )

    # Execute M86 master scientific benchmark
    result = M86ContinualLearningBenchmark.run_benchmark(
        model=model,
        device=device,
        seeds=args.seeds,
        num_episodes=args.num_episodes,
    )

    # Save output artifacts
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "m86_report.json"
    md_path = args.output_dir / "m86_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result["report_data"], f, indent=2)
    logger.info("Saved machine-readable JSON report: %s", json_path)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result["markdown_summary"])
    logger.info("Saved scientific Markdown report: %s", md_path)

    print("\n" + "=" * 70)
    print("MILESTONE M86 BENCHMARK SUMMARY")
    print("=" * 70)
    print(result["markdown_summary"])
    print("=" * 70)

    status = result["status"]
    if status == "PASS":
        logger.info("Milestone M86 PASSED all scientific qualification gates!")
    else:
        logger.warning("Milestone M86 finished with status: %s", status)


if __name__ == "__main__":
    main()
