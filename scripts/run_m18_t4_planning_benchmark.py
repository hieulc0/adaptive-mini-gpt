"""M18 Empirical Benchmark: Neural-Guided Search and Deep Compositional Reasoning.

Evaluates 5 planning configurations across 3 independent lineages (seeds 42, 43, 44):
1. Arm A: M17 Neural Full-Plan Generator with Grounded Slots (monolithic baseline)
2. Arm B: Grammar-Constrained Search without Model (pure grammar/search contribution)
3. Arm C: Grammar-Constrained Search with Mini-GPT Ranking (neural guidance contribution)
4. Arm D: Grammar Search + Mini-GPT + Bounded Retry Guidance
5. Arm E: Oracle Structural Plan with Grounded Slots (Upper Bound)

Depth ladders:
- Controlled Addition Chains across 2, 3, 4, 5 steps
- Multi-Segment Kinematics across 1, 2, 3 segments (1, 3, 5 steps)

Strict Runtime Ground-Truth Isolation:
- Search and runtime goals NEVER have access to expected answers or reference plans.
- Offline evaluation computes exact semantic answer accuracy, search expansions,
  exhaustion rates, and depth breakdown.
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
    DeterministicTemplatePlanner,
    ExecutionPlan,
    PlanExecutor,
    ReasoningController,
    RuleExecutor,
    generate_addition_depth_ladder,
    generate_multistep_abstention_problems,
    generate_multistep_instruction_corpus,
    generate_multistep_problems,
)
from adaptive_mini_gpt.rules.grammar_planner import (
    GrammarConstrainedPlanner,
    NeuralCandidateScorer,
    SearchBudgetExhaustedError,
    TargetGoal,
)
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m18_benchmark")


def pretrain_base_lineage(
    save_dir: Path,
    device: str = "cpu",
    precision: str = "fp32",
    steps: int = 150,
    seed: int = 42,
) -> tuple[MiniGPT, BPETokenizer, ModelConfig]:
    """Pretrain independent dense Transformer backbone."""
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
    tokenizer.train(full_corpus, target_vocab_size=400)

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        context_length=256,
        dropout=0.0,
    )
    model = MiniGPT(cfg)

    best_ckpt = save_dir / "best.pt"
    if best_ckpt.exists():
        logger.info("Found existing checkpoint %s, loading...", best_ckpt)
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        return model, tokenizer, cfg

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
    device: str = "cpu",
    precision: str = "fp32",
    steps: int = 150,
    seed: int = 42,
    format_type: str = "slot_plan",
) -> MiniGPT:
    """Adapt model to generate structured multi-step execution plans."""
    save_dir.mkdir(parents=True, exist_ok=True)
    plan_model = copy.deepcopy(base_model)

    best_ckpt = save_dir / "best.pt"
    if best_ckpt.exists():
        logger.info("Found existing checkpoint %s, loading...", best_ckpt)
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        plan_model.load_state_dict(ckpt["model_state_dict"])
        plan_model.to(device)
        return plan_model

    plan_text, plan_fp = generate_multistep_instruction_corpus(
        num_samples_per_archetype=40, seed=seed, split="train", format_type=format_type
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
        dataset_fingerprint=f"{plan_fp}_{format_type}",
    )
    trainer.train()
    return plan_model


def run_m18_benchmark(
    seeds: list[int],
    output_dir: Path,
    device: str = "cpu",
    precision: str = "fp32",
    samples_per_archetype: int = 10,
    num_abstention_samples: int = 6,
    samples_per_depth_ladder: int = 5,
) -> dict[str, Any]:
    """Execute complete M18 empirical neural-guided search ablation study across lineages."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    rule_executor = RuleExecutor()
    plan_executor = PlanExecutor(rule_executor)

    configurations = [
        ("neural_slots", "1. Arm A: M17 Neural Full-Plan Generation (Baseline)"),
        ("grammar_search", "2. Arm B: Grammar Search (No Neural Model)"),
        ("grammar_neural_ranked", "3. Arm C: Grammar Search + Mini-GPT Candidate Ranking"),
        ("neural_slots_retry", "4. Arm D: M17 Neural Plan + Bounded Feedback Retry"),
        ("oracle_slots", "5. Arm E: Oracle Structural Plan (Upper Bound)"),
    ]

    benchmark_report: dict[str, Any] = {
        "title": "M18 — Neural-Guided Search and Deep Compositional Reasoning Qualification",
        "device": device,
        "precision": precision,
        "seeds": seeds,
        "configurations": [name for _, name in configurations],
        "lineage_runs": {},
        "addition_depth_ladder": {},
        "summary": {},
        "timestamp": time.time(),
    }

    config_aggregates: dict[str, list[dict[str, Any]]] = {
        cfg_key: [] for cfg_key, _ in configurations
    }

    depth_aggregates: dict[str, dict[int, list[float]]] = {
        cfg_key: {2: [], 3: [], 4: [], 5: []} for cfg_key, _ in configurations
    }

    for seed in seeds:
        logger.info("\n" + "=" * 80)
        logger.info("STARTING M18 SEED LINEAGE %d", seed)
        logger.info("=" * 80)

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        seed_eval_file = seed_dir / "lineage_eval.json"
        if seed_eval_file.exists():
            logger.info("Found cached lineage evaluation for seed %d, loading %s...", seed, seed_eval_file)
            lineage_eval = json.loads(seed_eval_file.read_text())
            for cfg_key, _ in configurations:
                cfg_summary = lineage_eval[cfg_key]
                config_aggregates[cfg_key].append(cfg_summary)
                for depth in (2, 3, 4, 5):
                    d_str = str(depth)
                    if d_str in cfg_summary.get("addition_depth_ladder_accuracy", {}):
                        depth_aggregates[cfg_key][depth].append(cfg_summary["addition_depth_ladder_accuracy"][d_str])
                    elif depth in cfg_summary.get("addition_depth_ladder_accuracy", {}):
                        depth_aggregates[cfg_key][depth].append(cfg_summary["addition_depth_ladder_accuracy"][depth])
            benchmark_report["lineage_runs"][f"seed_{seed}"] = lineage_eval
            continue

        # 1. Pretrain base model
        base_model, tokenizer, cfg = pretrain_base_lineage(
            save_dir=seed_dir / "base_pretrain",
            device=device,
            precision=precision,
            steps=150,
            seed=seed,
        )

        # 2. Adapt planner model with grounded slot format
        slot_model = adapt_planner(
            base_model=base_model,
            tokenizer=tokenizer,
            cfg=cfg,
            save_dir=seed_dir / "slot_adapter",
            device=device,
            precision=precision,
            steps=150,
            seed=seed,
            format_type="slot_plan",
        )

        # 3. Generate pristine held-out test split (0% train/val overlap)
        test_problems = generate_multistep_problems(
            num_samples_per_archetype=samples_per_archetype, seed=seed + 999, split="test"
        )
        # 4. Generate abstention / malformed test split
        abstention_problems = generate_multistep_abstention_problems(
            num_samples=num_abstention_samples, seed=seed + 777, split="test"
        )

        # 5. Generate controlled addition depth ladder (2, 3, 4, 5 steps)
        addition_ladder = generate_addition_depth_ladder(
            num_samples_per_depth=samples_per_depth_ladder, seed=seed + 333, split="test"
        )

        all_eval_problems = test_problems + abstention_problems
        logger.info(
            "Evaluating %d standard + %d abstention + %d addition ladder problems on lineage seed %d...",
            len(test_problems),
            len(abstention_problems),
            len(addition_ladder),
            seed,
        )

        controller = ReasoningController(
            model=slot_model,
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
                    slots=prob.slots,
                    expected_slot_plan=prob.expected_slot_plan,
                )
                results.append(res.to_dict())

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

            # Step breakdown (2, 3, 5 steps)
            step_breakdown: dict[int, dict[str, Any]] = {}
            for n_steps in (1, 2, 3, 5):
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

            # Evaluate controlled addition depth ladder
            ladder_results = []
            for prob in addition_ladder:
                l_res = controller.solve_multistep(
                    question=prob.question,
                    expected_plan=prob.expected_plan,
                    expected_answer=prob.expected_answer,
                    mode=cfg_key,
                    max_retries=2,
                    slots=prob.slots,
                    expected_slot_plan=prob.expected_slot_plan,
                )
                ladder_results.append((prob.num_steps, l_res.answer_correct))

            ladder_depth_acc: dict[int, float] = {}
            for depth in (2, 3, 4, 5):
                d_items = [corr for d, corr in ladder_results if d == depth]
                if d_items:
                    acc = sum(1 for c in d_items if c) / len(d_items)
                    ladder_depth_acc[depth] = round(acc, 4)
                    depth_aggregates[cfg_key][depth].append(acc)

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
                "addition_depth_ladder_accuracy": ladder_depth_acc,
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

        benchmark_report["lineage_runs"][f"seed_{seed}"] = lineage_eval
        seed_eval_file.write_text(json.dumps(lineage_eval, indent=2))
        logger.info("Saved cached lineage evaluation for seed %d to %s", seed, seed_eval_file)

    # Multi-seed aggregate summary
    agg_summary: dict[str, Any] = {}
    for cfg_key, cfg_title in configurations:
        runs = config_aggregates[cfg_key]
        n_seeds = len(runs)
        mean_acc = sum(r["verified_answer_accuracy"] for r in runs) / n_seeds
        mean_valid = sum(r["plan_validity_rate"] for r in runs) / n_seeds
        mean_exact = sum(r["plan_exact_match_rate"] for r in runs) / n_seeds
        mean_failures = sum(r["failure_rate"] for r in runs) / n_seeds
        mean_abstention = sum(r["abstention_correctness"] for r in runs) / n_seeds
        mean_latency = sum(r["mean_latency_ms"] for r in runs) / n_seeds
        mean_retries = sum(r["mean_retries"] for r in runs) / n_seeds

        # Aggregate addition ladder depth curves
        ladder_summary: dict[str, float] = {}
        for d in (2, 3, 4, 5):
            d_vals = depth_aggregates[cfg_key][d]
            ladder_summary[f"depth_{d}_accuracy"] = (
                round(sum(d_vals) / len(d_vals), 4) if d_vals else 0.0
            )

        agg_summary[cfg_key] = {
            "configuration": cfg_title,
            "mean_verified_answer_accuracy": round(mean_acc, 4),
            "mean_plan_validity_rate": round(mean_valid, 4),
            "mean_plan_exact_match_rate": round(mean_exact, 4),
            "mean_failure_rate": round(mean_failures, 4),
            "mean_abstention_correctness": round(mean_abstention, 4),
            "mean_latency_ms": round(mean_latency, 2),
            "mean_retries": round(mean_retries, 2),
            "addition_depth_ladder": ladder_summary,
        }

    benchmark_report["summary"] = agg_summary

    report_file = output_dir / "m18_neural_search_report.json"
    report_file.write_text(json.dumps(benchmark_report, indent=2))
    logger.info("\nSaved full M18 report to %s", report_file)
    return benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M18 Neural-Guided Search Planning Benchmark")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Lineage seeds")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/m18_planning"), help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="fp16" if torch.cuda.is_available() else "fp32")
    parser.add_argument("--samples-per-archetype", type=int, default=10, help="Samples per archetype in test split")
    parser.add_argument("--abstention-samples", type=int, default=6, help="Abstention test queries")
    parser.add_argument("--samples-per-depth", type=int, default=5, help="Samples per depth in addition ladder")

    args = parser.parse_args()
    logger.info("M18 Benchmark starting on device=%s, precision=%s, seeds=%s", args.device, args.precision, args.seeds)

    t0 = time.time()
    run_m18_benchmark(
        seeds=args.seeds,
        output_dir=args.output_dir,
        device=args.device,
        precision=args.precision,
        samples_per_archetype=args.samples_per_archetype,
        num_abstention_samples=args.abstention_samples,
        samples_per_depth_ladder=args.samples_per_depth,
    )
    logger.info("Total M18 benchmark completed in %.2f seconds.", time.time() - t0)


if __name__ == "__main__":
    main()
