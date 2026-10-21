"""M31 Benchmark: Verified Intermediate Reasoning and Generalizable Answer Generation.

Orchestrates:
  1. Phase 1: Diagnostic audit of answer-generation failures under teacher forcing vs autonomous rollout.
  2. Phase 2 & 3: Controlled comparison of 3 representations:
     - Approach A: Direct Answer (final answer only control)
     - Approach B: Structured Neural Scratchpad (intermediate steps + final answer)
     - Approach C: Compact Recurrent State (position-independent local transitions)
  3. Phase 4: Independent evaluation across:
     - Arithmetic length extrapolation (2, 4, 8, 16 digits)
     - Symbolic logic depth extrapolation (3, 4, 5 hops)
     - Grounded language comprehension on independent sealed documents
     - Retention audit on 200 elementary carry transitions
  4. Phase 5: Closed-loop learning integration with step-level error diagnosis.
  5. Multi-seed evaluation across Seeds 42, 43, 44 on Tesla T4 GPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from adaptive_mini_gpt.data.bpe_scaling import (
    expand_model_vocabulary,
    train_scaled_bpe_tokenizer,
)
from adaptive_mini_gpt.data.language_corpus import PRETRAIN_PROSE_CORPUS
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.interaction.closed_loop_reasoner import (
    run_closed_loop_reasoning_cycle,
)
from adaptive_mini_gpt.model.capacity_migration import create_1m_control_config
from adaptive_mini_gpt.model.transformer import MiniGPT, ModelConfig
from adaptive_mini_gpt.reasoning.audit import run_phase1_audit
from adaptive_mini_gpt.reasoning.dataset import (
    MultiRepresentationReasoningDataset,
    build_arithmetic_reasoning_suite,
    build_language_reasoning_suite,
    build_logic_reasoning_suite,
    collate_reasoning_batch,
)
from adaptive_mini_gpt.reasoning.evaluator import (
    evaluate_reasoning_suite,
)
from adaptive_mini_gpt.reasoning.scratchpad import (
    ApproachType,
    ReasoningDomain,
    ReasoningSample,
)
from adaptive_mini_gpt.rules.reliable_transition import (
    ElementaryTransitionDataset,
    collate_elementary_batch,
    get_all_200_transitions,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("m31_reasoning_benchmark")


def train_reasoning_arm(
    arm_name: str,
    approach: ApproachType,
    base_model: MiniGPT,
    tokenizer: BPETokenizer,
    practice_samples: list[ReasoningSample],
    device: torch.device,
    epochs: int = 12,
    lr: float = 3e-4,
    batch_size: int = 8,
) -> MiniGPT:
    """Train an experimental arm under matched token and compute exposure with balanced replay."""
    logger.info(f"Training Arm [{arm_name}] (Approach: {approach.value}) on {len(practice_samples)} samples...")
    model = copy.deepcopy(base_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    max_ctx = getattr(model.config, "context_length", 256)
    pad_id = getattr(tokenizer, "pad_id", 0)

    # Practice DataLoader
    ds_practice = MultiRepresentationReasoningDataset(
        samples=practice_samples,
        tokenizer=tokenizer,
        approach=approach,
        context_length=max_ctx,
    )
    loader_practice = DataLoader(
        ds_practice,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_reasoning_batch(b, pad_token_id=pad_id),
    )

    # Replay DataLoader (200 elementary transitions to preserve carry mechanics)
    transitions = get_all_200_transitions()
    ds_replay = ElementaryTransitionDataset(
        transitions=transitions,
        tokenizer=tokenizer,
        format_type="canonical_state",
        context_length=max_ctx,
    )
    loader_replay = DataLoader(
        ds_replay,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_elementary_batch(b, pad_token_id=pad_id),
    )

    model.train()
    steps_per_epoch = max(len(loader_practice), len(loader_replay))

    for ep in range(epochs):
        iter_p = iter(loader_practice)
        iter_r = iter(loader_replay)

        for _ in range(steps_per_epoch):
            try:
                xp, yp = next(iter_p)
            except StopIteration:
                iter_p = iter(loader_practice)
                xp, yp = next(iter_p)

            xp, yp = xp.to(device), yp.to(device)
            optimizer.zero_grad()
            out_p = model(xp)
            logits_p = out_p[0] if isinstance(out_p, tuple) else out_p
            loss_p = nn.functional.cross_entropy(
                logits_p[:, :-1, :].contiguous().view(-1, logits_p.size(-1)),
                yp[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            loss_p.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Replay step
            try:
                xr, yr = next(iter_r)
            except StopIteration:
                iter_r = iter(loader_replay)
                xr, yr = next(iter_r)

            xr, yr = xr.to(device), yr.to(device)
            optimizer.zero_grad()
            out_r = model(xr)
            logits_r = out_r[0] if isinstance(out_r, tuple) else out_r
            loss_r = nn.functional.cross_entropy(
                logits_r[:, :-1, :].contiguous().view(-1, logits_r.size(-1)),
                yr[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            loss_r.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="M31 Intermediate Reasoning Benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoints-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/m31_reasoning")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"Starting M31 Intermediate Reasoning Benchmark on {device}")
    logger.info(f"Seeds: {args.seeds}, Epochs: {args.epochs}, LR: {args.lr}")
    logger.info("=" * 70)

    # 1. 1024-BPE Tokenizer
    all_prose = " ".join(PRETRAIN_PROSE_CORPUS)
    tokenizer_1024 = train_scaled_bpe_tokenizer(all_prose, target_vocab_size=1024)
    tok_path = output_path / "bpe_1024.json"
    tokenizer_1024.save(tok_path)
    logger.info(f"1024-BPE Tokenizer fingerprint: {tokenizer_1024.get_fingerprint()}")

    # 2. Benchmark Suites
    arith_suite = build_arithmetic_reasoning_suite(seed=42)
    logic_suite = build_logic_reasoning_suite(seed=42)
    lang_suite = build_language_reasoning_suite()

    # Combined practice curriculum (128 arith + 64 logic + 24 lang = 216 verified samples)
    combined_practice = (
        arith_suite["practice_128"]
        + logic_suite["practice_64"]
        + lang_suite["practice_24"]
    )
    logger.info(f"Combined multi-domain practice curriculum size: {len(combined_practice)} samples.")

    seed_reports: list[dict[str, Any]] = []

    for seed in args.seeds:
        logger.info("\n" + "#" * 65)
        logger.info(f"Executing M31 Lineage Seed {seed}")
        logger.info("#" * 65)

        t_seed_start = time.time()
        cfg_1m = create_1m_control_config(vocab_size=tokenizer_1024.vocab_size)
        base_model = MiniGPT(cfg_1m).to(device)

        if args.checkpoints_dir:
            ckpt_file = Path(args.checkpoints_dir) / f"seed_{seed}/base_pretrain/best.pt"
            if ckpt_file.exists():
                raw_ckpt = CheckpointManager.load_raw_checkpoint(ckpt_file)
                state_dict = raw_ckpt.get("model_state_dict", raw_ckpt.get("model_state"))
                base_model_400 = MiniGPT(ModelConfig(**raw_ckpt["model_config"]))
                base_model_400.load_state_dict(state_dict)
                tok_old = CheckpointManager.load_tokenizer(raw_ckpt)
                base_model = expand_model_vocabulary(
                    model=base_model_400,
                    old_tokenizer=tok_old,
                    new_tokenizer=tokenizer_1024,
                ).to(device)
                logger.info(f"Expanded seed {seed} checkpoint to 1024 vocab.")

        # Phase 1: Diagnostic Audit of Generation Failures
        logger.info(f"[Seed {seed}] Running Phase 1 Failure Diagnostic Audit...")
        audit_samples = [
            {"id": s.sample_id, "prompt": s.prompt_direct, "expected": s.expected_final_answer}
            for s in (arith_suite["eval_2_digits"][:5] + logic_suite["eval_3_hop"][:5] + lang_suite["eval_fresh"][:5])
        ]
        phase1_res = run_phase1_audit(base_model, tokenizer_1024, audit_samples, device)

        # Phase 3 & 4: Controlled Arms Comparison
        arms = [
            ("arm_a_direct", ApproachType.DIRECT),
            ("arm_b_scratchpad", ApproachType.SCRATCHPAD),
            ("arm_c_recurrent", ApproachType.RECURRENT),
        ]
        arm_eval_results: dict[str, Any] = {}

        for arm_name, approach in arms:
            logger.info(f"\n--- [Seed {seed}] Evaluating {arm_name} ---")
            trained_model = train_reasoning_arm(
                arm_name=arm_name,
                approach=approach,
                base_model=base_model,
                tokenizer=tokenizer_1024,
                practice_samples=combined_practice,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
            )

            # Evaluate across splits
            eval_splits: dict[str, Any] = {}

            # Arithmetic Length Extrapolation
            eval_splits["arith_2_digits"] = asdict(
                evaluate_reasoning_suite(trained_model, tokenizer_1024, arith_suite["eval_2_digits"], approach, "arith_2_digits", device)
            )
            eval_splits["arith_4_digits"] = asdict(
                evaluate_reasoning_suite(trained_model, tokenizer_1024, arith_suite["eval_4_digits"], approach, "arith_4_digits", device)
            )
            eval_splits["arith_8_digits"] = asdict(
                evaluate_reasoning_suite(trained_model, tokenizer_1024, arith_suite["eval_8_digits"], approach, "arith_8_digits", device)
            )
            eval_splits["arith_16_digits"] = asdict(
                evaluate_reasoning_suite(trained_model, tokenizer_1024, arith_suite["eval_16_digits"], approach, "arith_16_digits", device)
            )

            # Symbolic Logic Depth Extrapolation
            eval_splits["logic_3_hop"] = asdict(
                evaluate_reasoning_suite(trained_model, tokenizer_1024, logic_suite["eval_3_hop"], approach, "logic_3_hop", device)
            )
            eval_splits["logic_4_hop"] = asdict(
                evaluate_reasoning_suite(trained_model, tokenizer_1024, logic_suite["eval_4_hop"], approach, "logic_4_hop", device)
            )
            eval_splits["logic_5_hop"] = asdict(
                evaluate_reasoning_suite(trained_model, tokenizer_1024, logic_suite["eval_5_hop"], approach, "logic_5_hop", device)
            )

            # Fresh Language Comprehension (Scratchpad vs Direct)
            if approach != ApproachType.RECURRENT:
                eval_splits["lang_fresh"] = asdict(
                    evaluate_reasoning_suite(trained_model, tokenizer_1024, lang_suite["eval_fresh"], approach, "lang_fresh", device)
                )

            arm_eval_results[arm_name] = eval_splits

        # Phase 5: Closed-Loop Interaction Integration
        logger.info(f"[Seed {seed}] Running Phase 5 Closed-Loop Reasoning Cycle...")
        failing_problem = arith_suite["eval_4_digits"][0]
        practice_curriculum = arith_suite["practice_128"][:32]
        fresh_eval_problems = arith_suite["eval_4_digits"][1:9]

        closed_loop_res = asdict(
            run_closed_loop_reasoning_cycle(
                base_model=base_model,
                tokenizer=tokenizer_1024,
                initial_sample=failing_problem,
                practice_pool=practice_curriculum,
                fresh_test_pool=fresh_eval_problems,
                device=device,
            )
        )

        seed_duration = round(time.time() - t_seed_start, 2)
        seed_reports.append({
            "seed": seed,
            "duration_seconds": seed_duration,
            "phase1_audit": phase1_res,
            "arms_evaluation": arm_eval_results,
            "closed_loop_interaction": closed_loop_res,
        })

    # Consolidate Across Seeds
    logger.info("\nAggregating M31 multi-lineage metrics across seeds 42, 43, 44...")
    summary_report: dict[str, Any] = {
        "milestone": "M31",
        "title": "Verified Intermediate Reasoning and Generalizable Answer Generation Study",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "seeds": args.seeds,
        "parameters_1m": cfg_1m.n_layer * 128 * 128 * 12,  # approx
        "mean_duration_seconds": round(sum(s["duration_seconds"] for s in seed_reports) / len(seed_reports), 2),
        "tokenizer_1024_fingerprint": tokenizer_1024.get_fingerprint(),
        "phase1_audit_summary": {
            "mean_exact_match_rate": round(sum(s["phase1_audit"]["exact_match_rate"] for s in seed_reports) / len(seed_reports), 4),
            "mean_semantic_match_rate": round(sum(s["phase1_audit"]["semantic_match_rate"] for s in seed_reports) / len(seed_reports), 4),
            "mean_target_token_prob": round(sum(s["phase1_audit"]["mean_target_token_prob"] for s in seed_reports) / len(seed_reports), 4),
            "mean_teacher_loss": round(sum(s["phase1_audit"]["mean_teacher_loss"] for s in seed_reports) / len(seed_reports), 4),
        },
        "arms_summary": {},
        "raw_seed_runs": seed_reports,
    }

    # Aggregate by arm and split
    arm_names = ["arm_a_direct", "arm_b_scratchpad", "arm_c_recurrent"]
    for arm in arm_names:
        summary_report["arms_summary"][arm] = {}
        sample_splits = seed_reports[0]["arms_evaluation"][arm].keys()
        for split in sample_splits:
            mean_exact = sum(s["arms_evaluation"][arm][split]["final_exact_accuracy"] for s in seed_reports) / len(seed_reports)
            mean_sem = sum(s["arms_evaluation"][arm][split]["final_semantic_accuracy"] for s in seed_reports) / len(seed_reports)
            mean_step_acc = sum(s["arms_evaluation"][arm][split]["intermediate_step_accuracy"] for s in seed_reports) / len(seed_reports)
            mean_lat = sum(s["arms_evaluation"][arm][split]["mean_latency_ms"] for s in seed_reports) / len(seed_reports)
            ret_vals = [s["arms_evaluation"][arm][split]["retention_carry_accuracy"] for s in seed_reports if s["arms_evaluation"][arm][split].get("retention_carry_accuracy") is not None]
            mean_ret = sum(ret_vals) / len(ret_vals) if ret_vals else None

            summary_report["arms_summary"][arm][split] = {
                "mean_final_exact_accuracy": round(mean_exact, 4),
                "mean_final_semantic_accuracy": round(mean_sem, 4),
                "mean_intermediate_step_accuracy": round(mean_step_acc, 4),
                "mean_latency_ms": round(mean_lat, 2),
                "mean_retention_carry": round(mean_ret, 4) if mean_ret is not None else None,
            }

    report_path = output_path / "m31_report.json"
    with open(report_path, "w") as f:
        json.dump(summary_report, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"M31 Benchmark complete! Summary saved to {report_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    main()
