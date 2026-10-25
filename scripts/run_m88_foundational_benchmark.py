"""Milestone M88: Foundational Representation Learning Master Scientific Benchmark Runner.

Executes comprehensive scientific research evaluation on GPU / Colab:
- Phase 0: Forensic Audit of Milestone M87 (modular amnesia, budget-matched baseline).
- Phase 1 & 2: Foundational Multi-Domain Curriculum and Shared Neural Knowledge Substrate.
- Phase 3 & 4: Multi-Arm Causal Evaluation across 100-Episode Continual Streams.
- Phase 5: Scientific Qualification (Lifelong accuracy, forgetting, delayed retention, budget <= 20k).
- Outputs machine-readable m88_report.json and m88_report.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import torch

from adaptive_mini_gpt.foundational.benchmark import M88FoundationalBenchmark
from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("m88_benchmark")


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
        logger.warning("Checkpoint not found at %s. Initializing canonical baseline model.", checkpoint_path)
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
    parser = argparse.ArgumentParser(description="Milestone M88 Foundational Representation Benchmark")
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
        default=Path("/content/m88_output"),
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
    logger.info("Initializing M88 Foundational Representation Benchmark on device: %s", device)

    model, base_hash = setup_base_model(args.checkpoint_dir, args.seeds[0], device)
    total_frozen_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Base model ready. SHA-256: %s, Total Frozen Parameters: %d",
        base_hash,
        total_frozen_params,
    )

    result = M88FoundationalBenchmark.run_benchmark(
        model=model,
        device=device,
        seeds=args.seeds,
        num_episodes=args.num_episodes,
        budget_ceiling=args.budget_ceiling,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "m88_report.json"
    md_path = args.output_dir / "m88_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result["report_data"], f, indent=2)
    logger.info("Saved scientific metrics JSON to %s", json_path)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result["markdown_summary"])
    logger.info("Saved scientific markdown report to %s", md_path)

    logger.info("================================================================")
    logger.info("Milestone M88 Benchmark Complete. Verdict: %s", result["status"])
    logger.info("================================================================")
    print(result["markdown_summary"])


if __name__ == "__main__":
    main()
