"""Milestone M87: Long-Horizon Continual Learning Master Scientific Benchmark.

Executes comprehensive scientific research evaluation on Google Colab GPU (Tesla T4):
- Phase 0: Forensic Audit of Milestone M86 (Revisit reconciliation, baseline parameter accounting).
- Phase 1: 100-Episode Continual Task Curriculum (Foundations, Reuse, Conflicts, Delayed Revisits).
- Phase 2: Lifelong Knowledge Evaluation Matrix (R[i, j], ACC, BWT, FWT, F).
- Phase 3: Autonomous Capacity Regulation under 20,000 Parameter Ceiling (Shrinkage, Retirement).
- Phase 4: Six-Arm Causal Evaluation against Duplicate-Aware, Fixed, Ablation, and Random Controls.
- Phase 5: Bit-Exact Rollback Safety & Base Model Invariance (SHA-256: 6e39cd7ef6790b3c).
- Outputs machine-readable m87_report.json and m87_report.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import torch

from adaptive_mini_gpt.long_horizon.benchmark import M87LongHorizonBenchmark
from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("m87_benchmark")


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
    parser = argparse.ArgumentParser(description="Milestone M87 Long-Horizon Continual Learning Benchmark")
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
        default=100,
        help="Number of episodes in the task stream (default: 100)",
    )
    parser.add_argument(
        "--budget-ceiling",
        type=int,
        default=20_000,
        help="Parameter budget ceiling for all trainable components (default: 20000)",
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
        default=Path("/content/m87_output"),
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
    logger.info("Initializing M87 long-horizon continual learning evaluation on device: %s", device)

    # Load frozen base model
    model, base_hash = setup_base_model(args.checkpoint_dir, args.seeds[0], device)
    total_frozen_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Base model ready. SHA-256: %s, Total Frozen Parameters: %d",
        base_hash,
        total_frozen_params,
    )

    # Execute M87 master scientific benchmark
    result = M87LongHorizonBenchmark.run_benchmark(
        model=model,
        device=device,
        seeds=args.seeds,
        num_episodes=args.num_episodes,
        budget_ceiling=args.budget_ceiling,
    )

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "m87_report.json"
    md_path = args.output_dir / "m87_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result["report_data"], f, indent=2)
    logger.info("Saved scientific metrics JSON to %s", json_path)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result["markdown_summary"])
    logger.info("Saved scientific markdown report to %s", md_path)

    logger.info("================================================================")
    logger.info("Milestone M87 Benchmark Complete. Verdict: %s", result["status"])
    logger.info("================================================================")
    print(result["markdown_summary"])


if __name__ == "__main__":
    main()
