#!/usr/bin/env python3
"""M33: Repeated Self-Learning and Cumulative Skill Acquisition Benchmark Runner.

Executes the 7-Phase M33 Study across 3 independent lineages (Seeds 42, 43, 44):
  Phase 1: Audit M32 learning claims and directly matched before/after evaluation.
  Phase 2: Persistent evolving neural lineage tracking (V1 -> qual_V2 -> qual_V3 -> qual_V4).
  Phase 3: Multi-episode self-learning sequence (Arithmetic -> Logic -> Language).
  Phase 4: Practice and learning strategy selection with decision auditing.
  Phase 5: Cumulative multi-domain evaluation and comparison against baselines.
  Phase 6: Action selection utility testing.
  Phase 7: Multi-seed qualification report generation.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

# Ensure repository root and src are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from adaptive_mini_gpt.data.language_corpus import PRETRAIN_PROSE_CORPUS
from adaptive_mini_gpt.data.bpe_scaling import (
    expand_model_vocabulary,
    train_scaled_bpe_tokenizer,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.learning.audit_m32 import run_phase1_audit
from adaptive_mini_gpt.learning.cumulative_learner import (
    run_cumulative_learning_sequence,
)
from adaptive_mini_gpt.model.transformer import MiniGPT, ModelConfig
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("m33_benchmark")


def run_seed_m33_pipeline(
    seed: int,
    checkpoints_dir: Path,
    tokenizer: BPETokenizer,
    device: torch.device,
    epochs: int = 12,
    lr: float = 3e-4,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Execute complete M33 pipeline for a single seed lineage."""
    logger.info("#################################################################")
    logger.info(f"Executing M33 Cumulative Lineage Seed {seed}")
    logger.info("#################################################################")

    t_start = time.perf_counter()
    random.seed(seed)
    torch.manual_seed(seed)

    # 1. Load M18 Root Checkpoint & Expand to 1024-BPE
    relative = Path(f"seed_{seed}/base_pretrain/best.pt")
    checkpoint_path = checkpoints_dir / relative
    raw = CheckpointManager.load_raw_checkpoint(checkpoint_path)

    base_tok = CheckpointManager.load_tokenizer(raw)
    state_dict = raw.get("model_state_dict", raw.get("model_state"))
    base_model_400 = MiniGPT(ModelConfig(**raw["model_config"]))
    base_model_400.load_state_dict(state_dict)
    base_model = expand_model_vocabulary(
        model=base_model_400,
        old_tokenizer=base_tok,
        new_tokenizer=tokenizer,
    ).to(device)
    param_count = sum(p.numel() for p in base_model.parameters())

    # 2. Phase 1 Audit of M32 Learning Claims
    logger.info(f"[Seed {seed}] Running Phase 1 Audit...")
    phase1_audit = run_phase1_audit(
        base_model=base_model,
        tokenizer=tokenizer,
        device=device,
        seed=seed,
    )

    # 3. Phase 2-5 Multi-Episode Cumulative Learning Sequence
    logger.info(f"[Seed {seed}] Running Phase 2-5 Multi-Episode Learning Sequence...")
    cumulative_res = run_cumulative_learning_sequence(
        base_model=base_model,
        tokenizer=tokenizer,
        device=device,
        seed=seed,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    duration = round(time.perf_counter() - t_start, 2)
    logger.info(f"[Seed {seed}] Finished full pipeline in {duration}s")

    return {
        "seed": seed,
        "duration_seconds": duration,
        "parameters_count": param_count,
        "phase1_audit": phase1_audit,
        "cumulative_learning_results": cumulative_res,
    }


def compute_mean_std(values: list[float]) -> dict[str, float]:
    """Compute mean and sample standard deviation."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(values) / n
    if n == 1:
        return {"mean": round(mean, 4), "std": 0.0}
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="M33 Repeated Self-Learning and Cumulative Acquisition Study")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("======================================================================")
    logger.info("M33 — Repeated Self-Learning and Cumulative Skill Acquisition Study")
    logger.info(f"Device: {device}, Seeds: {args.seeds}, Epochs: {args.epochs}, LR: {args.lr}")
    logger.info("======================================================================")

    # 1. Scaled 1024-BPE Tokenizer (Consistent across all lineages)
    all_prose = " ".join(PRETRAIN_PROSE_CORPUS)
    tokenizer = train_scaled_bpe_tokenizer(all_prose, target_vocab_size=1024)
    tok_path = output_dir / "bpe_1024.json"
    tokenizer.save(tok_path)
    logger.info(f"1024-BPE Tokenizer saved. Fingerprint: {tokenizer.get_fingerprint()}")

    # 2. Execute Lineages Across Seeds
    seed_runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_res = run_seed_m33_pipeline(
            seed=seed,
            checkpoints_dir=checkpoints_dir,
            tokenizer=tokenizer,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        seed_runs.append(run_res)

    logger.info("\nAggregating M33 multi-lineage metrics across seeds 42, 43, 44...")

    # 3. Aggregate Cumulative Evolution across Steps (V1_root -> after_episode_a -> after_episode_b -> after_episode_c)
    step_names = ["V1_root", "after_episode_a", "after_episode_b", "after_episode_c"]
    metrics_to_track = [
        "arithmetic_exact",
        "arithmetic_step_acc",
        "logic_exact",
        "language_exact",
        "language_abstention",
        "carry_retention_200",
        "prose_retention_loss",
    ]

    progression_summary: dict[str, dict[str, dict[str, float]]] = {}
    for step in step_names:
        progression_summary[step] = {}
        for m in metrics_to_track:
            vals: list[float] = []
            for run in seed_runs:
                hist = run["cumulative_learning_results"]["cumulative_history"]
                step_entries = [h for h in hist if h["step"] == step]
                if step_entries:
                    vals.append(step_entries[0]["metrics"].get(m, 0.0))
            progression_summary[step][m] = compute_mean_std(vals)

    # 4. Aggregate Baselines (Frozen V1, Specialist Arithmetic, Specialist Logic)
    baseline_keys = ["v1_frozen_control", "specialist_arithmetic", "specialist_logic"]
    baseline_summary: dict[str, dict[str, dict[str, float]]] = {}
    for b_key in baseline_keys:
        baseline_summary[b_key] = {}
        for m in metrics_to_track:
            vals = [run["cumulative_learning_results"]["baselines"][b_key].get(m, 0.0) for run in seed_runs]
            baseline_summary[b_key][m] = compute_mean_std(vals)

    # 5. Qualification Gate Summary
    promotions_total = sum(run["cumulative_learning_results"]["promotions_count"] for run in seed_runs)
    rollbacks_total = sum(run["cumulative_learning_results"]["rollbacks_count"] for run in seed_runs)

    m33_report = {
        "metadata": {
            "milestone": "M33",
            "study_name": "Repeated Self-Learning and Cumulative Skill Acquisition",
            "seeds": args.seeds,
            "device": str(device),
            "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", 1024),
            "tokenizer_fingerprint": tokenizer.get_fingerprint(),
            "mean_duration_seconds": round(sum(r["duration_seconds"] for r in seed_runs) / len(seed_runs), 2),
        },
        "progression_summary": progression_summary,
        "baseline_summary": baseline_summary,
        "lineage_qualification_summary": {
            "total_episodes_attempted": promotions_total + rollbacks_total,
            "total_promotions": promotions_total,
            "total_rollbacks": rollbacks_total,
            "promotion_rate": round(promotions_total / max(1, promotions_total + rollbacks_total), 4),
            "per_seed_final_version": {
                f"seed_{run['seed']}": run["cumulative_learning_results"]["active_final_version"]
                for run in seed_runs
            },
        },
        "individual_seed_runs": seed_runs,
    }

    report_path = output_dir / "m33_report.json"
    with open(report_path, "w") as f:
        json.dump(m33_report, f, indent=2)
    logger.info(f"M33 Report successfully written to {report_path}")

    # Log summary tables
    logger.info("\n=================== M33 CUMULATIVE SKILL PROGRESSION ===================")
    for step in step_names:
        arith_m = progression_summary[step]["arithmetic_exact"]
        logic_m = progression_summary[step]["logic_exact"]
        lang_m = progression_summary[step]["language_exact"]
        carry_m = progression_summary[step]["carry_retention_200"]
        logger.info(
            f"[{step:<17}] Arith: {arith_m['mean']*100:.1f}% ± {arith_m['std']*100:.1f}% | "
            f"Logic: {logic_m['mean']*100:.1f}% ± {logic_m['std']*100:.1f}% | "
            f"Lang: {lang_m['mean']*100:.1f}% ± {lang_m['std']*100:.1f}% | "
            f"Carry Ret: {carry_m['mean']*100:.1f}% ± {carry_m['std']*100:.1f}%"
        )
    logger.info("========================================================================")


if __name__ == "__main__":
    main()
