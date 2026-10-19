"""M13 Procedural Reasoning Curriculum and Generalization Benchmark on NVIDIA Tesla T4.

Executes a 4-arm controlled experiment across 3 fully independent pretraining lineages:
- Lineages: 3 independent model initialization and pretraining seeds (42, 43, 44)
- Experimental Arms:
  * Arm A (Direct BPE): Direct-answer reversal using subword BPE (cat -> tac)
  * Arm B (Character Delimited): Space-delimited representation (c a t -> t a c)
  * Arm C (Explicit Stepwise Procedure): Intermediate verified procedural steps
  * Arm D (Progressive Curriculum): Copy -> 2-char swap -> short reverse -> full reverse
- Multi-dimensional diagnostics:
  * Exact match accuracy
  * Per-character position accuracy
  * Answer length match accuracy
  * Termination correctness
  * Output diversity (mode collapse detection)
  * Stepwise intermediate step validity (Step 1 read, Step 2 index)
  * Learning curves across adaptation step checkpoints (0, 50, 100, 150)
  * In-distribution (lens 3-6) vs OOD length extrapolation (lens 7-8)
- Retention and Continual Learning:
  * Raw cross-checkpoint loss drift on prose datasets
  * Old-skill retention (instruction_qa, instruction_reasoning)
  * Experience replay comparison
  * Strict autonomous gating on 'val' split and post-selection qualification on 'test' split
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import string
import time
from pathlib import Path
from typing import Any

import torch

from adaptive_mini_gpt.autonomous import GrowthAction, LearningController, LearningPlan
from adaptive_mini_gpt.autonomous.tasks import (
    TASK_REGISTRY,
    evaluate_symbolic_diagnostics,
    evaluate_symbolic_reversal_breakdown,
    evaluate_task_holdout,
)
from adaptive_mini_gpt.data.instruction_data import (
    SEED_INSTRUCTION_EXAMPLES,
    InstructionDataset,
    format_instruction_pair,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.evaluation.evaluator import evaluate_text_retention_suite
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.training.checkpoint import CheckpointManager
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m13_curriculum")


def pretrain_bpe_lineage(
    save_dir: Path,
    device: str = "cuda",
    precision: str = "fp16",
    steps: int = 150,
    seed: int = 42,
) -> tuple[Path, BPETokenizer, ModelConfig]:
    """Pretrain an independent BPE dense Transformer backbone from scratch for a given seed."""
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Pretraining independent BPE language backbone: Seed=%d, Steps=%d, Device=%s, Precision=%s",
        seed,
        steps,
        device,
        precision,
    )

    corpus_prose = (
        "From fairest creatures we desire increase, That thereby beauty's rose might never die.\n"
        "When forty winters shall besiege thy brow, And dig deep trenches in thy beauty's field.\n"
        "William Shakespeare wrote Hamlet, Macbeth, Othello, and King Lear in London.\n"
        "In quantum physics, photons, bosons, and fermions govern fundamental interactions.\n"
        "Paris is the capital of France, Tokyo is the capital of Japan, and Rome is in Italy.\n"
        "The speed of light in vacuum is approximately 300,000 km/s.\n"
    ) * 30

    instruction_pairs = []
    for ex in SEED_INSTRUCTION_EXAMPLES:
        instruction_pairs.append(format_instruction_pair(ex["instruction"], ex["response"]))
    full_corpus = corpus_prose + "".join(instruction_pairs) * 10

    tokenizer = BPETokenizer()
    tokenizer.train(full_corpus, target_vocab_size=350)

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        context_length=128,
        dropout=0.0,
    )
    model = MiniGPT(cfg)

    tokens = tokenizer.encode(full_corpus)
    split_idx = int(len(tokens) * 0.9)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    ds_train = InstructionDataset(
        token_ids=train_tokens,
        context_length=cfg.context_length,
        assistant_token_id=tokenizer.assistant_id,
        eos_token_id=tokenizer.eos_id,
        user_token_id=tokenizer.user_id,
    )
    ds_val = InstructionDataset(
        token_ids=val_tokens,
        context_length=cfg.context_length,
        assistant_token_id=tokenizer.assistant_id,
        eos_token_id=tokenizer.eos_id,
        user_token_id=tokenizer.user_id,
    )

    train_cfg = TrainingConfig(
        seed=seed,
        batch_size=16,
        gradient_accumulation=1,
        learning_rate=1.5e-3,
        max_steps=steps,
        eval_every_steps=max(1, steps // 2),
        device=device,
        precision=precision,
        save_dir=str(save_dir),
    )

    trainer = Trainer(
        model=model,
        model_config=cfg,
        training_config=train_cfg,
        train_dataset=ds_train,
        val_dataset=ds_val,
        tokenizer=tokenizer,
        dataset_fingerprint=f"pretrain_bpe_lineage_{seed}",
    )
    t0 = time.time()
    trainer.train()
    logger.info("Lineage seed=%d pretraining completed in %.2fs.", seed, time.time() - t0)

    # Publish baseline checkpoint using trainer checkpoint manager
    ckpt_path = trainer.checkpoint_manager.save_checkpoint(
        checkpoint_name="latest.pt",
        model=model,
        optimizer=trainer.optimizer,
        model_config=cfg,
        training_config=train_cfg,
        tokenizer=tokenizer,
        step=trainer.step,
        tokens_seen=trainer.tokens_seen,
    )

    # Initialize active pointer
    pointer = {
        "version_index": 0,
        "experiment_id": f"pretrain_lineage_{seed}",
        "published_dir": str(save_dir),
        "checkpoint_path": str(ckpt_path),
        "step": trainer.step,
        "tokens_seen": trainer.tokens_seen,
        "tokenizer_fingerprint": tokenizer.get_fingerprint() if hasattr(tokenizer, "get_fingerprint") else "",
    }
    with open(save_dir / "active_pointer.json", "w", encoding="utf-8") as f:
        json.dump(pointer, f, indent=2)

    return save_dir, tokenizer, cfg


def run_m13_curriculum_experiment(
    seeds: list[int],
    output_root: Path,
    device: str = "cuda",
    precision: str = "fp16",
    adaptation_steps: int = 150,
    checkpoint_interval: int = 50,
) -> dict[str, Any]:
    """Execute complete M13 procedural reasoning curriculum study across arms and independent seeds."""
    output_root.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    experimental_arms = [
        {"arm_id": "arm_a_direct", "task_id": "instruction_symbolic", "strategy": "direct", "name": "Arm A (Direct BPE)", "replay_ratio": 0.0},
        {"arm_id": "arm_b_char", "task_id": "instruction_symbolic_char", "strategy": "char", "name": "Arm B (Char Delimited)", "replay_ratio": 0.0},
        {"arm_id": "arm_c_stepwise", "task_id": "instruction_symbolic_stepwise", "strategy": "stepwise", "name": "Arm C (Explicit Stepwise)", "replay_ratio": 0.0},
        {"arm_id": "arm_d_curriculum", "task_id": "instruction_symbolic_curriculum", "strategy": "curriculum", "name": "Arm D (Progressive Curriculum)", "replay_ratio": 0.0},
        {"arm_id": "arm_b_replay", "task_id": "instruction_symbolic_char", "strategy": "char", "name": "Arm B + Replay (20% Replay)", "replay_ratio": 0.2},
    ]

    benchmark_report: dict[str, Any] = {
        "title": "M13 — Procedural Reasoning Curriculum and Generalization",
        "device": device,
        "precision": precision,
        "adaptation_steps": adaptation_steps,
        "seeds": seeds,
        "arms": [a["name"] for a in experimental_arms],
        "lineages": {},
        "summary": {},
        "timestamp": time.time(),
    }

    arm_aggregates: dict[str, list[dict[str, Any]]] = {a["arm_id"]: [] for a in experimental_arms}

    for seed in seeds:
        logger.info("\n" + "=" * 80)
        logger.info("STARTING INDEPENDENT LINEAGE SEED %d", seed)
        logger.info("=" * 80)

        seed_dir = output_root / f"lineage_seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # 1. Independent pretraining from scratch
        base_dir = seed_dir / "base_pretrain"
        _, tokenizer, cfg = pretrain_bpe_lineage(
            save_dir=base_dir,
            device=device,
            precision=precision,
            steps=150,
            seed=seed,
        )

        # Load fresh base model
        base_ckpt = CheckpointManager.find_latest_valid_checkpoint(base_dir)
        raw_base = CheckpointManager.load_raw_checkpoint(base_ckpt)
        base_model = MiniGPT(cfg)
        base_model.load_state_dict(raw_base["model_state_dict"])
        base_model.to(dev)

        # Baseline evaluation
        logger.info("Evaluating lineage seed=%d baseline metrics...", seed)
        baseline_retention = evaluate_text_retention_suite(model=base_model, tokenizer=tokenizer, device=dev)
        raw_baseline_losses = {
            dom: baseline_retention[dom]["raw_loss"]
            for dom in ("clean_prose", "numerical_prose", "mixed_prose_math")
            if dom in baseline_retention and "raw_loss" in baseline_retention[dom]
        }

        # Baseline old skill evaluations
        baseline_qa = evaluate_task_holdout(
            model=base_model, tokenizer=tokenizer, task=TASK_REGISTRY["instruction_qa"], device=dev, num_samples=30, split="val"
        )
        baseline_reasoning = evaluate_task_holdout(
            model=base_model, tokenizer=tokenizer, task=TASK_REGISTRY["instruction_reasoning"], device=dev, num_samples=30, split="val"
        )

        # Baseline diagnostics at step 0 for each arm
        baseline_diags: dict[str, Any] = {}
        for arm in experimental_arms:
            arm_strat = arm["strategy"]
            baseline_diags[arm["arm_id"]] = evaluate_symbolic_diagnostics(
                model=base_model,
                tokenizer=tokenizer,
                device=dev,
                arm=arm_strat,
                num_samples_per_cat=25,
                seed=seed + 100,
            )

        lineage_results: dict[str, Any] = {
            "seed": seed,
            "raw_baseline_losses": raw_baseline_losses,
            "baseline_qa_acc": baseline_qa["accuracy"],
            "baseline_reasoning_acc": baseline_reasoning["accuracy"],
            "baseline_diagnostics": baseline_diags,
            "arm_runs": {},
        }

        # 2. Run all experimental arms for this seed
        for arm_spec in experimental_arms:
            arm_id = arm_spec["arm_id"]
            task_id = arm_spec["task_id"]
            strategy = arm_spec["strategy"]
            name = arm_spec["name"]
            replay_ratio = arm_spec["replay_ratio"]

            logger.info("\n--- [Seed %d] Running %s (Task: %s, Replay: %.1f) ---", seed, name, task_id, replay_ratio)

            arm_dir = seed_dir / arm_id
            arm_dir.mkdir(parents=True, exist_ok=True)

            controller = LearningController(
                base_checkpoint_dir=base_dir,
                output_dir=arm_dir,
                budget_steps=adaptation_steps,
                max_budget_steps=adaptation_steps * 2,
                min_improvement=0.05,
                max_regression_tolerance=0.08,
                target_accuracy_threshold=0.95,
                growth_action=GrowthAction.CONTINUE,
                force_task=task_id,
                device=device,
                precision=precision,
                seed=seed,
            )

            # Step Evaluate
            controller.step_evaluate(model=base_model, tokenizer=tokenizer)
            # Step Select
            controller.step_select_task(tokenizer=tokenizer)

            # Step Plan with custom strategy and replay ratio
            plan = LearningPlan(
                task_id=task_id,
                growth_action="continue",
                budget_steps=adaptation_steps,
                learning_rate=1e-3,
                batch_size=16,
                data_samples=5000,
                seed=seed + 10,
                strategy=strategy,
                replay_ratio=replay_ratio,
            )
            controller.journal.plan = plan.__dict__
            controller.journal.state = "TRAIN"

            # Train with checkpoint learning curve monitoring
            candidate_model = copy.deepcopy(base_model)
            task = TASK_REGISTRY[task_id]

            if getattr(task, "split_support", False):
                task_text, task_fp = task.generator_fn(plan.data_samples, plan.seed, split="train")
            else:
                task_text, task_fp = task.generator_fn(plan.data_samples, plan.seed)

            # Support experience replay
            if replay_ratio > 0.0:
                replay_instructions = "".join(
                    format_instruction_pair(ex["instruction"], ex["response"]) for ex in SEED_INSTRUCTION_EXAMPLES
                ) * 5
                from adaptive_mini_gpt.autonomous.tasks import generate_instruction_qa_data
                qa_text, _ = generate_instruction_qa_data(num_samples=20, seed=plan.seed, split="train")
                replay_tokens = tokenizer.encode(replay_instructions + qa_text)
                task_tokens = tokenizer.encode(task_text)
                n_replay = int(len(task_tokens) * replay_ratio)
                combined_tokens = task_tokens + replay_tokens[:n_replay]
            else:
                combined_tokens = tokenizer.encode(task_text)

            split_idx = int(len(combined_tokens) * 0.9)
            ds_train = InstructionDataset(
                token_ids=combined_tokens[:split_idx],
                context_length=candidate_model.config.context_length,
                assistant_token_id=tokenizer.assistant_id,
                eos_token_id=tokenizer.eos_id,
                user_token_id=tokenizer.user_id,
            )
            ds_val = InstructionDataset(
                token_ids=combined_tokens[split_idx:],
                context_length=candidate_model.config.context_length,
                assistant_token_id=tokenizer.assistant_id,
                eos_token_id=tokenizer.eos_id,
                user_token_id=tokenizer.user_id,
            )

            # Train in intervals to record learning curves
            learning_curve: list[dict[str, Any]] = []

            # Step 0 diagnostics
            init_diags = baseline_diags[arm_id]
            learning_curve.append({
                "step": 0,
                "in_dist_exact": init_diags["unseen_in_distribution_len_3_6"]["exact_match_acc"],
                "in_dist_per_char": init_diags["unseen_in_distribution_len_3_6"]["per_char_acc"],
                "in_dist_length": init_diags["unseen_in_distribution_len_3_6"]["length_match_acc"],
                "in_dist_termination": init_diags["unseen_in_distribution_len_3_6"]["termination_acc"],
                "in_dist_diversity": init_diags["unseen_in_distribution_len_3_6"]["output_diversity"],
                "ood_exact": init_diags["ood_extrapolation_len_7_8"]["exact_match_acc"],
                "ood_per_char": init_diags["ood_extrapolation_len_7_8"]["per_char_acc"],
            })

            curr_step = 0
            candidate_dir = arm_dir / f"candidate_{task_id}"
            candidate_dir.mkdir(parents=True, exist_ok=True)

            while curr_step < adaptation_steps:
                next_step = min(curr_step + checkpoint_interval, adaptation_steps)
                steps_to_run = next_step - curr_step

                train_cfg = TrainingConfig(
                    seed=seed + curr_step,
                    batch_size=plan.batch_size,
                    gradient_accumulation=2,
                    learning_rate=plan.learning_rate,
                    max_steps=steps_to_run,
                    eval_every_steps=max(1, steps_to_run),
                    device=device,
                    precision=precision,
                    save_dir=str(candidate_dir),
                )
                trainer = Trainer(
                    model=candidate_model,
                    model_config=candidate_model.config,
                    training_config=train_cfg,
                    train_dataset=ds_train,
                    val_dataset=ds_val,
                    tokenizer=tokenizer,
                    dataset_fingerprint=task_fp,
                )
                trainer.train()
                curr_step = next_step

                # Evaluate checkpoint diagnostics
                ckpt_diags = evaluate_symbolic_diagnostics(
                    model=candidate_model,
                    tokenizer=tokenizer,
                    device=dev,
                    arm=strategy,
                    num_samples_per_cat=25,
                    seed=seed + curr_step,
                )
                learning_curve.append({
                    "step": curr_step,
                    "in_dist_exact": ckpt_diags["unseen_in_distribution_len_3_6"]["exact_match_acc"],
                    "in_dist_per_char": ckpt_diags["unseen_in_distribution_len_3_6"]["per_char_acc"],
                    "in_dist_length": ckpt_diags["unseen_in_distribution_len_3_6"]["length_match_acc"],
                    "in_dist_termination": ckpt_diags["unseen_in_distribution_len_3_6"]["termination_acc"],
                    "in_dist_diversity": ckpt_diags["unseen_in_distribution_len_3_6"]["output_diversity"],
                    "ood_exact": ckpt_diags["ood_extrapolation_len_7_8"]["exact_match_acc"],
                    "ood_per_char": ckpt_diags["ood_extrapolation_len_7_8"]["per_char_acc"],
                })
                logger.info(
                    "[%s Step %d] In-Dist Exact=%.2f%%, Per-Char=%.2f%%, LenMatch=%.2f%%, Diversity=%.2f, OOD Exact=%.2f%%",
                    arm_id,
                    curr_step,
                    ckpt_diags["unseen_in_distribution_len_3_6"]["exact_match_acc"] * 100,
                    ckpt_diags["unseen_in_distribution_len_3_6"]["per_char_acc"] * 100,
                    ckpt_diags["unseen_in_distribution_len_3_6"]["length_match_acc"] * 100,
                    ckpt_diags["unseen_in_distribution_len_3_6"]["output_diversity"],
                    ckpt_diags["ood_extrapolation_len_7_8"]["exact_match_acc"] * 100,
                )

            # Save candidate checkpoint using checkpoint manager
            trainer.checkpoint_manager.save_checkpoint(
                checkpoint_name="latest.pt",
                model=candidate_model,
                optimizer=trainer.optimizer,
                model_config=candidate_model.config,
                training_config=train_cfg,
                tokenizer=tokenizer,
                step=curr_step,
                tokens_seen=trainer.tokens_seen,
            )

            # Step Verify
            verified_promoted = controller.step_verify(
                candidate_dir=candidate_dir,
                tokenizer=tokenizer,
                task_id=task_id,
            )

            if verified_promoted:
                active_path = controller.step_promote(candidate_dir)
            else:
                controller.step_rollback(candidate_dir)
                active_path = base_dir

            # Qualification on untouched 'test' split
            qual_report = controller.step_qualify(
                checkpoint_dir=candidate_dir,
                tokenizer=tokenizer,
                num_samples=30,
            )

            # Detailed final diagnostics on candidate
            final_diags = evaluate_symbolic_diagnostics(
                model=candidate_model,
                tokenizer=tokenizer,
                device=dev,
                arm=strategy,
                num_samples_per_cat=30,
                seed=seed + 999,
            )

            # Extract retention audit
            cand_raw_audit = controller.journal.candidate_metrics.get("raw_retention_audit", {})
            deltas = cand_raw_audit.get("deltas", {})
            max_drift = max(deltas.values()) if deltas else 0.0

            # Evaluate old skill retention
            cand_qa = evaluate_task_holdout(
                model=candidate_model, tokenizer=tokenizer, task=TASK_REGISTRY["instruction_qa"], device=dev, num_samples=30, split="test"
            )
            cand_reasoning = evaluate_task_holdout(
                model=candidate_model, tokenizer=tokenizer, task=TASK_REGISTRY["instruction_reasoning"], device=dev, num_samples=30, split="test"
            )

            run_result = {
                "arm_id": arm_id,
                "name": name,
                "task_id": task_id,
                "decision": controller.journal.decision,
                "decision_reason": controller.journal.decision_reason,
                "learning_curve": learning_curve,
                "final_diagnostics": final_diags,
                "retention_drift": {
                    "deltas": deltas,
                    "max_drift": max_drift,
                },
                "old_skill_retention": {
                    "instruction_qa_baseline": baseline_qa["accuracy"],
                    "instruction_qa_candidate": cand_qa["accuracy"],
                    "instruction_reasoning_baseline": baseline_reasoning["accuracy"],
                    "instruction_reasoning_candidate": cand_reasoning["accuracy"],
                },
                "test_split_accuracy": {
                    "in_distribution_len_3_6": final_diags["unseen_in_distribution_len_3_6"]["exact_match_acc"],
                    "in_distribution_per_char": final_diags["unseen_in_distribution_len_3_6"]["per_char_acc"],
                    "ood_extrapolation_len_7_8": final_diags["ood_extrapolation_len_7_8"]["exact_match_acc"],
                    "ood_per_char": final_diags["ood_extrapolation_len_7_8"]["per_char_acc"],
                    "output_diversity": final_diags["unseen_in_distribution_len_3_6"]["output_diversity"],
                    "length_match": final_diags["unseen_in_distribution_len_3_6"]["length_match_acc"],
                },
            }
            lineage_results["arm_runs"][arm_id] = run_result
            arm_aggregates[arm_id].append(run_result)

        benchmark_report["lineages"][f"seed_{seed}"] = lineage_results

    # 3. Aggregate cross-seed analysis per arm
    arm_summaries: dict[str, Any] = {}
    for arm_spec in experimental_arms:
        arm_id = arm_spec["arm_id"]
        runs = arm_aggregates[arm_id]
        n = len(runs)
        if n == 0:
            continue

        mean_in_dist_exact = sum(r["test_split_accuracy"]["in_distribution_len_3_6"] for r in runs) / n
        mean_in_dist_pchar = sum(r["test_split_accuracy"]["in_distribution_per_char"] for r in runs) / n
        mean_ood_exact = sum(r["test_split_accuracy"]["ood_extrapolation_len_7_8"] for r in runs) / n
        mean_ood_pchar = sum(r["test_split_accuracy"]["ood_per_char"] for r in runs) / n
        mean_drift = sum(r["retention_drift"]["max_drift"] for r in runs) / n
        promotes = sum(1 for r in runs if r["decision"] == "PROMOTE")
        rollbacks = sum(1 for r in runs if r["decision"] == "ROLLBACK")

        arm_summaries[arm_id] = {
            "name": arm_spec["name"],
            "runs_evaluated": n,
            "promotions": promotes,
            "rollbacks": rollbacks,
            "mean_in_dist_exact_match": round(mean_in_dist_exact, 4),
            "mean_in_dist_per_char_acc": round(mean_in_dist_pchar, 4),
            "mean_ood_extrapolation_exact_match": round(mean_ood_exact, 4),
            "mean_ood_extrapolation_per_char_acc": round(mean_ood_pchar, 4),
            "mean_max_loss_drift": round(mean_drift, 4),
        }

    benchmark_report["summary"] = arm_summaries

    # Determine empirical qualification
    qualified_arms = [
        arm_id for arm_id, s in arm_summaries.items()
        if s["mean_in_dist_exact_match"] >= 0.80 and s["promotions"] >= 2 and s["mean_max_loss_drift"] <= 0.08
    ]
    benchmark_report["empirically_qualified"] = len(qualified_arms) > 0
    benchmark_report["qualified_arms"] = qualified_arms

    # Save final report
    report_file = output_root / "m13_t4_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("M13 EXPERIMENT COMPLETE! Report saved to: %s", report_file)
    logger.info("Summary of Arms:")
    for arm_id, s in arm_summaries.items():
        logger.info(
            "  * %s: In-Dist Exact=%.2f%%, Per-Char=%.2f%%, OOD Exact=%.2f%%, Drift=%+.4f, Decisions: %d PROMOTE / %d ROLLBACK",
            s["name"],
            s["mean_in_dist_exact_match"] * 100,
            s["mean_in_dist_per_char_acc"] * 100,
            s["mean_ood_extrapolation_exact_match"] * 100,
            s["mean_max_loss_drift"],
            s["promotions"],
            s["rollbacks"],
        )
    logger.info("=" * 80)

    return benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser(description="M13 Procedural Curriculum Benchmark")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Pretraining & adaptation seeds")
    parser.add_argument("--output-dir", type=str, default="runs/m13_curriculum", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="fp16" if torch.cuda.is_available() else "fp32")
    parser.add_argument("--steps", type=int, default=150, help="Adaptation steps per arm")
    args = parser.parse_args()

    run_m13_curriculum_experiment(
        seeds=args.seeds,
        output_root=Path(args.output_dir),
        device=args.device,
        precision=args.precision,
        adaptation_steps=args.steps,
    )


if __name__ == "__main__":
    main()
