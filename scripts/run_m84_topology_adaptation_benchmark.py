"""Milestone M84: Autonomous Neural Connection Pruning and Regrowth Master Benchmark.

Executes scientific research evaluation on Google Colab GPU (Tesla T4):
- Phase 0: Forensic Audit of Milestone M83 (Reconciled Net vs Marginal Compression).
- Phase 1: Dynamic Connection-Level Pruning (>= 20% Sparsity on Neural Adapters).
- Phase 2: Autonomous Neural Connection Regrowth (RigL Gradient Saliency + Learned Controller).
- Phase 3: 7-Arm Causal Topology Benchmark across Seeds [42, 43, 44, 45, 46].
- Phase 4: Skill Retention Integrity (>= 95% Relative Retention, >= 85% Floor).
- Phase 5: Future-Task Adaptation Efficiency (<= 35 Steps on Held-Out Cascade).
- Phase 6: Bit-Exact Rollback Safety & Base Model Invariance (SHA-256: 6e39cd7ef6790b3c).
- Outputs machine-readable m84_report.json and m84_report.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import torch

from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.topology_adaptation.benchmark import (
    M84TopologyAdaptationBenchmark,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("m84_benchmark")


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

    base_bytes = b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())
    base_hash = hashlib.sha256(base_bytes).hexdigest()[:16]
    return model, base_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Milestone M84 Benchmark on Colab GPU")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="Random seeds")
    parser.add_argument("--target-sparsity", type=float, default=0.30, help="Target connection sparsity")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/content/checkpoints"), help="Checkpoints dir")
    parser.add_argument("--output-dir", type=Path, default=Path("./m84_output"), help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Initializing M84 evaluation on device: %s", device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base model setup
    model, base_hash = setup_base_model(args.checkpoint_dir, seed=args.seeds[0], device=device)
    logger.info("Base model ready. SHA-256: %s, Total Frozen Parameters: %d",
                base_hash, sum(p.numel() for p in model.parameters()))

    # 2. Run Integrated M84 Master Benchmark
    results = M84TopologyAdaptationBenchmark.run_benchmark(
        model=model,
        device=device,
        seeds=args.seeds,
        target_sparsity=args.target_sparsity,
    )

    report_data = results["report_data"]
    markdown_summary = results["markdown_summary"]

    # 3. Save JSON and Markdown artifacts
    json_path = args.output_dir / "m84_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Saved machine-readable JSON report: %s", json_path)

    md_path = args.output_dir / "m84_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_summary)
    logger.info("Saved scientific Markdown report: %s", md_path)

    # Print summary to console
    print("\n" + "=" * 70)
    print("MILESTONE M84 BENCHMARK SUMMARY")
    print("=" * 70)
    print(markdown_summary)
    print("=" * 70)

    if report_data["status"] != "PASS":
        logger.error("M84 did not achieve all qualification gates! Status: %s", report_data["status"])
        raise SystemExit(1)
    else:
        logger.info("Milestone M84 PASSED all scientific qualification gates!")


if __name__ == "__main__":
    main()
