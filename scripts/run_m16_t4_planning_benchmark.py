"""M16 Empirical Benchmark on NVIDIA Tesla T4: Verified Multi-Step Planning and Compositional Reasoning.

Evaluates 5 planning configurations across 3 independent lineages (seeds 42, 43, 44):
1. Arm A: Direct Mini-GPT Neural Answer (without tools)
2. Arm B: Oracle Multi-Step Plan + Deterministic Execution
3. Arm C: Deterministic Template Planner + Grounded Execution
4. Arm D: Mini-GPT Generated Plan + Deterministic Execution (no retry)
5. Arm E: Mini-GPT Generated Plan + Feedback Retry Loop (bounded, error-guided only)

Evaluates on identical held-out 2 to 5-step compositional problems (0% train/val/test overlap):
- Multi-segment average speed (2-segment = 3 steps, 3-segment = 5 steps)
- Chained arithmetic (3 operands = 2 steps, 4 operands = 3 steps)
- Transitive comparison chains (4+ entities)
- Abstention battery (dimensional incompatibility, non-positive duration, cyclic relations)

Records:
- Plan validity and exact/semantic match accuracy
- Step dependency resolution accuracy
- End-to-end verified answer accuracy
- Abstention correctness on malformed inputs
- Latency (ms) and retry effectiveness
- Accuracy broken down by number of reasoning steps (2, 3, 4, 5 steps)
- Commutative argument equivalence audit
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

from adaptive_mini_gpt.data.instruction_data import (
    SEED_INSTRUCTION_EXAMPLES,
    InstructionDataset,
    format_instruction_pair,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules import (
    PlanExecutor,
    ReasoningController,
    RuleExecutor,
    generate_multistep_abstention_problems,
    generate_multistep_instruction_corpus,
    generate_multistep_problems,
)
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m16_benchmark")


def pretrain_base_lineage(
    save_dir: Path,
    device: str = "cuda",
    precision: str = "fp16",
    steps: int = 150,
    seed: int = 42,
) -> tuple[MiniGPT, BPETokenizer, ModelConfig]:
    """Pretrain independent BPE dense Transformer backbone."""
    save_dir.mkdir(parents=True, exist_ok=True)
    corpus_prose = (
        "From fairest creatures we desire increase, That thereby beauty's rose might never die.\n"
        "William Shakespeare wrote Hamlet, Macbeth, Othello, and King Lear in London.\n"
        "In quantum physics, photons, bosons, and fermions govern fundamental interactions.\n"
        "Paris is the capital of France, Tokyo is the capital of Japan, and Rome is in Italy.\n"
        "The speed of light in vacuum is approximately 300,000 km/s.\n"
    ) * 30

    instruction_pairs = [
        format_instruction_pair(ex["instruction"], ex["response"])
        for ex in SEED_INSTRUCTION_EXAMPLES
    ]
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
    ds_train = InstructionDataset(
        token_ids=tokens[:split_idx],
        context_length=cfg.context_length,
        assistant_token_id=tokenizer.assistant_id,
        eos_token_id=tokenizer.eos_id,
        user_token_id=tokenizer.user_id,
    )
    ds_val = InstructionDataset(
        token_ids=tokens[split_idx:],
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
        dataset_fingerprint=f"pretrain_base_{seed}",
    )
    trainer.train()
    return model, tokenizer, cfg


def adapt_planner(
    base_model: MiniGPT,
    tokenizer: BPETokenizer,
    cfg: ModelConfig,
    save_dir: Path,
    device: str = "cuda",
    precision: str = "fp16",
    steps: int = 150,
    seed: int = 42,
) -> MiniGPT:
    """Adapt model to generate structured multi-step execution plans from natural language questions (train split only)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    plan_model = copy.deepcopy(base_model)

    # Strictly use training split for plan supervision
    plan_text, plan_fp = generate_multistep_instruction_corpus(
        num_samples_per_archetype=40, seed=seed, split="train", format_type="plan"
    )
    tokens = tokenizer.encode(plan_text)
    split_idx = int(len(tokens) * 0.9)

    ds_train = InstructionDataset(
        token_ids=tokens[:split_idx],
        context_length=cfg.context_length,
        assistant_token_id=tokenizer.assistant_id,
        eos_token_id=tokenizer.eos_id,
        user_token_id=tokenizer.user_id,
    )
    ds_val = InstructionDataset(
        token_ids=tokens[split_idx:],
        context_length=cfg.context_length,
        assistant_token_id=tokenizer.assistant_id,
        eos_token_id=tokenizer.eos_id,
        user_token_id=tokenizer.user_id,
    )

    train_cfg = TrainingConfig(
        seed=seed + 1,
        batch_size=16,
        gradient_accumulation=2,
        learning_rate=1e-3,
        max_steps=steps,
        eval_every_steps=max(1, steps // 2),
        device=device,
        precision=precision,
        save_dir=str(save_dir),
    )

    trainer = Trainer(
        model=plan_model,
        model_config=cfg,
        training_config=train_cfg,
        train_dataset=ds_train,
        val_dataset=ds_val,
        tokenizer=tokenizer,
        dataset_fingerprint=plan_fp,
    )
    trainer.train()
    return plan_model


def run_m16_benchmark(
    seeds: list[int],
    output_dir: Path,
    device: str = "cuda",
    precision: str = "fp16",
    samples_per_archetype: int = 10,
    num_abstention_samples: int = 8,
) -> dict[str, Any]:
    """Execute complete M16 empirical multi-step planning ablation study across lineages."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    rule_executor = RuleExecutor()
    plan_executor = PlanExecutor(rule_executor)

    configurations = [
        ("neural_only", "1. Arm A: Direct Neural Mini-GPT (No Tools)"),
        ("oracle", "2. Arm B: Oracle Multi-Step Plan + Deterministic Execution"),
        ("deterministic_template", "3. Arm C: Deterministic Template Planner + Grounded Execution"),
        ("model_plan", "4. Arm D: Mini-GPT Generated Plan (No Retry)"),
        ("model_plan_retry", "5. Arm E: Mini-GPT Generated Plan + Feedback Retry Loop"),
    ]

    benchmark_report: dict[str, Any] = {
        "title": "M16 — Verified Multi-Step Planning and Compositional Reasoning Qualification",
        "device": device,
        "precision": precision,
        "seeds": seeds,
        "configurations": [name for _, name in configurations],
        "lineage_runs": {},
        "summary": {},
        "timestamp": time.time(),
    }

    config_aggregates: dict[str, list[dict[str, Any]]] = {
        cfg_key: [] for cfg_key, _ in configurations
    }

    for seed in seeds:
        logger.info("\n" + "=" * 80)
        logger.info("STARTING SEED LINEAGE %d", seed)
        logger.info("=" * 80)

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # 1. Pretrain base model
        base_model, tokenizer, cfg = pretrain_base_lineage(
            save_dir=seed_dir / "base_pretrain",
            device=device,
            precision=precision,
            steps=150,
            seed=seed,
        )

        # 2. Adapt planner model on train split
        plan_model = adapt_planner(
            base_model=base_model,
            tokenizer=tokenizer,
            cfg=cfg,
            save_dir=seed_dir / "planner_adapter",
            device=device,
            precision=precision,
            steps=150,
            seed=seed,
        )

        # 3. Generate pristine held-out test split (0% train/val overlap)
        test_problems = generate_multistep_problems(
            num_samples_per_archetype=samples_per_archetype, seed=seed + 999, split="test"
        )
        # 4. Generate abstention / malformed test split
        abstention_problems = generate_multistep_abstention_problems(
            num_samples=num_abstention_samples, seed=seed + 777, split="test"
        )

        all_eval_problems = test_problems + abstention_problems
        logger.info(
            "Evaluating %d held-out test problems (%d standard + %d abstention) on lineage seed %d...",
            len(all_eval_problems),
            len(test_problems),
            len(abstention_problems),
            seed,
        )

        controller = ReasoningController(
            model=plan_model,
            tokenizer=tokenizer,
            executor=rule_executor,
            plan_executor=plan_executor,
            device=dev,
        )

        lineage_eval: dict[str, Any] = {}

        for cfg_key, cfg_title in configurations:
            logger.info("--- Evaluating: %s ---", cfg_title)
            results: list[dict[str, Any]] = []

            for prob in all_eval_problems:
                res = controller.solve_multistep(
                    question=prob.question,
                    expected_plan=prob.expected_plan,
                    expected_answer=prob.expected_answer,
                    mode=cfg_key,
                    max_retries=2,
                    should_abstain=prob.should_abstain,
                )
                results.append(res.to_dict())

            # Standard vs Abstention metrics
            std_results = results[: len(test_problems)]
            abst_results = results[len(test_problems) :]

            tot_std = len(std_results)
            plan_valid_corr = sum(1 for r in std_results if r["plan_valid"])
            plan_exact_corr = sum(1 for r in std_results if r["plan_exact_match"])
            ans_corr = sum(1 for r in std_results if r["answer_correct"])
            failures = sum(1 for r in std_results if r["status"] == "failed")
            mean_lat = sum(r["latency_ms"] for r in std_results) / tot_std if tot_std > 0 else 0.0
            mean_retries = sum(r["total_retries"] for r in std_results) / tot_std if tot_std > 0 else 0.0

            tot_abst = len(abst_results)
            abst_corr = sum(1 for r in abst_results if r["abstention_correct"])
            abst_acc = abst_corr / tot_abst if tot_abst > 0 else 0.0

            # Step-level breakdown (accuracy by number of reasoning steps)
            step_breakdown: dict[int, dict[str, Any]] = {}
            for n_steps in (2, 3, 4, 5):
                n_runs = [
                    r for r, p in zip(std_results, test_problems) if p.num_steps == n_steps
                ]
                tot_n = len(n_runs)
                if tot_n > 0:
                    n_correct = sum(1 for r in n_runs if r["answer_correct"])
                    n_valid = sum(1 for r in n_runs if r["plan_valid"])
                    step_breakdown[n_steps] = {
                        "count": tot_n,
                        "verified_accuracy": round(n_correct / tot_n, 4),
                        "plan_validity_rate": round(n_valid / tot_n, 4),
                    }

            cfg_summary = {
                "configuration": cfg_title,
                "total_standard_problems": tot_std,
                "plan_validity_rate": round(plan_valid_corr / tot_std, 4) if tot_std > 0 else 0.0,
                "plan_exact_match_rate": round(plan_exact_corr / tot_std, 4) if tot_std > 0 else 0.0,
                "verified_answer_accuracy": round(ans_corr / tot_std, 4) if tot_std > 0 else 0.0,
                "failure_rate": round(failures / tot_std, 4) if tot_std > 0 else 0.0,
                "total_abstention_problems": tot_abst,
                "abstention_correctness": round(abst_acc, 4),
                "mean_latency_ms": round(mean_lat, 2),
                "mean_retries": round(mean_retries, 2),
                "step_breakdown": step_breakdown,
                "sample_prediction": std_results[0] if std_results else None,
            }

            lineage_eval[cfg_key] = cfg_summary
            config_aggregates[cfg_key].append(cfg_summary)

            logger.info(
                "[%s] Verified Acc: %.2f%% | Plan Valid: %.2f%% | Plan Match: %.2f%% | Abstention: %.2f%% | Latency: %.2f ms",
                cfg_title,
                cfg_summary["verified_answer_accuracy"] * 100.0,
                cfg_summary["plan_validity_rate"] * 100.0,
                cfg_summary["plan_exact_match_rate"] * 100.0,
                cfg_summary["abstention_correctness"] * 100.0,
                cfg_summary["mean_latency_ms"],
            )

        lineage_eval["test_size"] = len(all_eval_problems)
        benchmark_report["lineage_runs"][f"seed_{seed}"] = lineage_eval

    # Multi-seed aggregate summary
    n_seeds = len(seeds)
    agg_summary: dict[str, Any] = {}
    for cfg_key, _ in configurations:
        runs = config_aggregates[cfg_key]
        mean_verified = sum(r["verified_answer_accuracy"] for r in runs) / n_seeds
        mean_valid = sum(r["plan_validity_rate"] for r in runs) / n_seeds
        mean_exact = sum(r["plan_exact_match_rate"] for r in runs) / n_seeds
        mean_failures = sum(r["failure_rate"] for r in runs) / n_seeds
        mean_abstention = sum(r["abstention_correctness"] for r in runs) / n_seeds
        mean_latency = sum(r["mean_latency_ms"] for r in runs) / n_seeds
        mean_retries = sum(r["mean_retries"] for r in runs) / n_seeds

        # Aggregate step breakdown
        agg_step_acc: dict[str, float] = {}
        for n_steps in (2, 3, 4, 5):
            runs_with_step = [r["step_breakdown"].get(n_steps) for r in runs if n_steps in r["step_breakdown"]]
            if runs_with_step:
                agg_step_acc[f"{n_steps}_steps"] = round(
                    sum(s["verified_accuracy"] for s in runs_with_step) / len(runs_with_step),
                    4,
                )

        agg_summary[cfg_key] = {
            "configuration": runs[0]["configuration"],
            "mean_verified_answer_accuracy": round(mean_verified, 4),
            "mean_plan_validity_rate": round(mean_valid, 4),
            "mean_plan_exact_match_rate": round(mean_exact, 4),
            "mean_failure_rate": round(mean_failures, 4),
            "mean_abstention_correctness": round(mean_abstention, 4),
            "mean_latency_ms": round(mean_latency, 2),
            "mean_retries": round(mean_retries, 2),
            "step_accuracy_breakdown": agg_step_acc,
        }

    benchmark_report["summary"] = agg_summary

    report_file = output_dir / "m16_t4_report.json"
    report_file.write_text(json.dumps(benchmark_report, indent=2))
    logger.info("\nSaved full M16 report to %s", report_file)
    return benchmark_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run M16 T4 Multi-Step Planning Benchmark")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output_dir", type=str, default="runs/m16_planning")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="fp16" if torch.cuda.is_available() else "fp32")
    parser.add_argument("--samples_per_archetype", type=int, default=10)
    parser.add_argument("--abstention_samples", type=int, default=8)
    args = parser.parse_args()

    run_m16_benchmark(
        seeds=args.seeds,
        output_dir=Path(args.output_dir),
        device=args.device,
        precision=args.precision,
        samples_per_archetype=args.samples_per_archetype,
        num_abstention_samples=args.abstention_samples,
    )
