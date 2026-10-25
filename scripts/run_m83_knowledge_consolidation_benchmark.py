"""Milestone M83: Autonomous Neural Knowledge Consolidation and Self-Distillation Benchmark.

Executes scientific research evaluation on Google Colab GPU (Tesla T4):
- Phase 0: M82 Forensic Audit & Reconciliation (df = 4).
- Phase 1: Consolidation Candidate Discovery (Multi-metric compatibility analysis).
- Phase 2: Neural Self-Distillation into Consolidated Shared Adapter with Residual Heads.
- Phase 3: Selective Pruning & Bit-Exact Rollback Audit.
- Phase 4: Learned Consolidation Policy Evaluation across 5 Relation Regimes.
- Phase 5: Retention, Lineage Preservation, and 5-Seed Statistical Significance.
- Outputs machine-readable m83_report.json and m83_report.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import torch

from adaptive_mini_gpt.knowledge_consolidation.benchmark import (
    M83KnowledgeConsolidationBenchmark,
)
from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("m83_benchmark")


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
    parser = argparse.ArgumentParser(description="Run Milestone M83 Benchmark on Colab GPU")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="Random seeds")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/content/checkpoints"), help="Checkpoints dir")
    parser.add_argument("--output-dir", type=Path, default=Path("./m83_output"), help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Initializing M83 evaluation on device: %s", device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base model setup
    model, base_hash = setup_base_model(args.checkpoint_dir, seed=args.seeds[0], device=device)
    logger.info("Base model ready. SHA-256: %s, Total Frozen Parameters: %d",
                base_hash, sum(p.numel() for p in model.parameters()))

    # 2. Run Integrated M83 Benchmark
    results = M83KnowledgeConsolidationBenchmark.run_benchmark(
        model=model,
        device=device,
        seeds=args.seeds,
    )

    report_data = results["report_data"]
    markdown_summary = results["markdown_summary"]

    # 3. Save JSON and Markdown artifacts
    json_path = args.output_dir / "m83_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Saved machine-readable JSON report: %s", json_path)

    md_path = args.output_dir / "m83_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_summary)
    logger.info("Saved scientific Markdown report: %s", md_path)

    # Print summary to console
    print("\n" + "=" * 70)
    print("MILESTONE M83 BENCHMARK SUMMARY")
    print("=" * 70)
    print(markdown_summary)
    print("=" * 70)

    if report_data["status"] != "PASS":
        logger.error("M83 did not achieve all qualification gates! Status: %s", report_data["status"])
        raise SystemExit(1)
    else:
        logger.info("Milestone M83 PASSED all scientific qualification gates!")


if __name__ == "__main__":
    main()
