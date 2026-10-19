"""M17 Clean Benchmark Runner for NVIDIA Tesla T4.

Evaluates 5 planning configurations across 3 independent lineages (seeds 42, 43, 44):
1. Arm A: Deterministic Template Planner (Baseline)
2. Arm B: Neural Plan with Generated Literals (M16 Baseline)
3. Arm C: Neural Plan with Deterministic Slot Injection
4. Arm D: Neural Plan with Slot Injection + Bounded Retry
5. Arm E: Oracle Plan with Deterministic Slot Injection
Phase 5: Multi-Turn Conversational Working Memory
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
    ConversationMemory,
    DeterministicTemplatePlanner,
    ExecutionPlan,
    PlanExecutor,
    ReasoningController,
    RuleExecutor,
    SlotExtractor,
    format_available_slots,
    generate_conversational_problems,
    generate_multistep_abstention_problems,
    generate_multistep_instruction_corpus,
    generate_multistep_problems,
)
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m17_clean")


def get_or_train_base(
    save_dir: Path,
    device: str = "cuda",
    precision: str = "fp16",
    steps: int = 150,
    seed: int = 42,
) -> tuple[MiniGPT, BPETokenizer, ModelConfig]:
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / "best.pt"
    if best_ckpt.exists():
        logger.info("Loading existing base checkpoint: %s", best_ckpt)
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        tokenizer = BPETokenizer.from_dict(ckpt["tokenizer"])
        cfg = ModelConfig.from_dict(ckpt["model_config"])
        model = MiniGPT(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        return model, tokenizer, cfg

    logger.info("Training base lineage seed %d...", seed)
    corpus_prose = (
        "From fairest creatures we desire increase, That thereby beauty's rose might never die.\n"
        "William Shakespeare wrote Hamlet, Macbeth, Othello, and King Lear in London.\n"
        "In quantum physics, photons, bosons, and fermions govern fundamental interactions.\n"
        "Paris is the capital of France, Tokyo is the capital of Japan, and Rome is in Italy.\n"
        "The speed of light in vacuum is approximately 300,000 km/s.\n"
    ) * 10

    instruction_pairs = [
        format_instruction_pair(ex["instruction"], ex["response"])
        for ex in SEED_INSTRUCTION_EXAMPLES
    ]
    full_corpus = corpus_prose + "".join(instruction_pairs) * 4

    tokenizer = BPETokenizer()
    tokenizer.train(full_corpus, target_vocab_size=320)

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


def get_or_adapt_planner(
    base_model: MiniGPT,
    tokenizer: BPETokenizer,
    cfg: ModelConfig,
    save_dir: Path,
    device: str = "cuda",
    precision: str = "fp16",
    steps: int = 150,
    seed: int = 42,
    format_type: str = "slot_plan",
) -> MiniGPT:
    save_dir.mkdir(parents=True, exist_ok=True)
    plan_model = copy.deepcopy(base_model)
    best_ckpt = save_dir / "best.pt"
    if best_ckpt.exists():
        logger.info("Loading existing adapter checkpoint: %s", best_ckpt)
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        plan_model.load_state_dict(ckpt["model_state_dict"])
        plan_model.to(device)
        return plan_model

    logger.info("Adapting planner (%s) seed %d...", format_type, seed)
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


def run_benchmark(
    seeds: list[int],
    output_dir: Path,
    device: str = "cuda",
    precision: str = "fp16",
    samples_per_archetype: int = 10,
    num_abstention_samples: int = 8,
    num_conversational_sessions: int = 10,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    rule_executor = RuleExecutor()
    plan_executor = PlanExecutor(rule_executor)

    configurations = [
        ("deterministic_template", "1. Arm A: Deterministic Template Planner (Baseline)"),
        ("neural_literals", "2. Arm B: Neural Plan with Generated Literals (M16 Baseline)"),
        ("neural_slots", "3. Arm C: Neural Plan with Deterministic Slot Injection"),
        ("neural_slots_retry", "4. Arm D: Neural Plan with Slot Injection + Bounded Retry"),
        ("oracle_slots", "5. Arm E: Oracle Plan with Deterministic Slot Injection"),
    ]

    benchmark_report: dict[str, Any] = {
        "title": "M17 — Hybrid Neural-Structural Planning and Grounded Slot Binding Qualification",
        "device": device,
        "precision": precision,
        "seeds": seeds,
        "configurations": [name for _, name in configurations],
        "lineage_runs": {},
        "conversational_eval": {},
        "summary": {},
        "timestamp": time.time(),
    }

    config_aggregates: dict[str, list[dict[str, Any]]] = {
        cfg_key: [] for cfg_key, _ in configurations
    }

    for seed in seeds:
        t_seed_start = time.time()
        logger.info("\n" + "=" * 80)
        logger.info("STARTING SEED %d", seed)
        logger.info("=" * 80)

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_report_path = seed_dir / "lineage_report.json"

        if seed_report_path.exists():
            logger.info("Loading existing seed report: %s", seed_report_path)
            loaded = json.loads(seed_report_path.read_text())
            lineage_eval = loaded["lineage_eval"]
            conv_summary = loaded["conversational_eval"]
            benchmark_report["lineage_runs"][f"seed_{seed}"] = lineage_eval
            benchmark_report["conversational_eval"][f"seed_{seed}"] = conv_summary
            for cfg_key, _ in configurations:
                config_aggregates[cfg_key].append(lineage_eval[cfg_key])
            continue

        base_model, tokenizer, cfg = get_or_train_base(
            save_dir=seed_dir / "base_pretrain",
            device=device,
            precision=precision,
            steps=150,
            seed=seed,
        )

        literal_model = get_or_adapt_planner(
            base_model=base_model,
            tokenizer=tokenizer,
            cfg=cfg,
            save_dir=seed_dir / "literal_adapter",
            device=device,
            precision=precision,
            steps=150,
            seed=seed,
            format_type="plan",
        )

        slot_model = get_or_adapt_planner(
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

        test_problems = generate_multistep_problems(
            num_samples_per_archetype=samples_per_archetype, seed=seed + 999, split="test"
        )
        abstention_problems = generate_multistep_abstention_problems(
            num_samples=num_abstention_samples, seed=seed + 777, split="test"
        )
        all_eval_problems = test_problems + abstention_problems

        literal_controller = ReasoningController(
            model=literal_model,
            tokenizer=tokenizer,
            executor=rule_executor,
            plan_executor=plan_executor,
            device=dev,
        )
        slot_controller = ReasoningController(
            model=slot_model,
            tokenizer=tokenizer,
            executor=rule_executor,
            plan_executor=plan_executor,
            device=dev,
        )

        lineage_eval: dict[str, Any] = {}

        for cfg_key, cfg_title in configurations:
            t_cfg = time.time()
            active_controller = literal_controller if cfg_key == "neural_literals" else slot_controller
            results: list[dict[str, Any]] = []

            for prob in all_eval_problems:
                res = active_controller.solve_multistep(
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
                "[%s] Verified: %.1f%% | Plan Valid: %.1f%% | Match: %.1f%% | Abstain: %.1f%% | Lat: %.1f ms | Duration: %.2fs",
                cfg_title,
                cfg_summary["verified_answer_accuracy"] * 100.0,
                cfg_summary["plan_validity_rate"] * 100.0,
                cfg_summary["plan_exact_match_rate"] * 100.0,
                cfg_summary["abstention_correctness"] * 100.0,
                cfg_summary["mean_latency_ms"],
                time.time() - t_cfg,
            )

        # Conversational Working Memory
        t_conv = time.time()
        conv_sessions = generate_conversational_problems(
            num_sessions=num_conversational_sessions, seed=seed + 555, split="test"
        )
        conv_results = []
        for s in conv_sessions:
            conv_mem = ConversationMemory(session_id=s.session_id)
            t1_res = slot_controller.solve_conversational_turn(
                question=s.turn1_question,
                conversation_memory=conv_mem,
                expected_answer=s.turn1_expected_answer,
                expected_plan=s.turn1_plan,
                slots=s.turn1_slots,
            )
            t2_res = slot_controller.solve_conversational_turn(
                question=s.turn2_question,
                conversation_memory=conv_mem,
                expected_answer=s.turn2_expected_answer,
                expected_plan=s.turn2_plan,
                slots=s.turn2_slots,
                should_abstain=s.should_abstain_turn2,
            )
            conv_results.append({
                "session_id": s.session_id,
                "turn1_correct": t1_res["answer_correct"],
                "turn2_correct": t2_res["answer_correct"],
                "turn2_abstained": t2_res["status"] == "abstained",
                "is_ambiguous": s.is_ambiguous,
            })

        t1_acc = sum(1 for r in conv_results if r["turn1_correct"]) / len(conv_results)
        t2_acc = sum(1 for r in conv_results if r["turn2_correct"]) / len(conv_results)
        amb_correct = sum(
            1 for r in conv_results if r["is_ambiguous"] and r["turn2_abstained"]
        ) / max(1, sum(1 for r in conv_results if r["is_ambiguous"]))

        conv_summary = {
            "num_sessions": len(conv_sessions),
            "turn1_accuracy": round(t1_acc, 4),
            "turn2_accuracy": round(t2_acc, 4),
            "ambiguous_reference_abstention": round(amb_correct, 4),
        }
        benchmark_report["conversational_eval"][f"seed_{seed}"] = conv_summary
        lineage_eval["test_size"] = len(all_eval_problems)
        benchmark_report["lineage_runs"][f"seed_{seed}"] = lineage_eval

        seed_report_path.write_text(
            json.dumps({"lineage_eval": lineage_eval, "conversational_eval": conv_summary}, indent=2)
        )
        logger.info(
            "[Conversational Memory] Turn 1: %.1f%% | Turn 2: %.1f%% | Ambiguous: %.1f%% | Total Seed Time: %.2fs",
            t1_acc * 100.0,
            t2_acc * 100.0,
            amb_correct * 100.0,
            time.time() - t_seed_start,
        )

    # Multi-seed aggregate summary
    completed_seeds = len(benchmark_report["lineage_runs"])
    if completed_seeds > 0:
        agg_summary: dict[str, Any] = {}
        for cfg_key, _ in configurations:
            runs = config_aggregates[cfg_key]
            if not runs:
                continue
            mean_verified = sum(r["verified_answer_accuracy"] for r in runs) / len(runs)
            mean_valid = sum(r["plan_validity_rate"] for r in runs) / len(runs)
            mean_exact = sum(r["plan_exact_match_rate"] for r in runs) / len(runs)
            mean_abst = sum(r["abstention_correctness"] for r in runs) / len(runs)
            mean_fail = sum(r["failure_rate"] for r in runs) / len(runs)
            mean_lat = sum(r["mean_latency_ms"] for r in runs) / len(runs)
            mean_ret = sum(r["mean_retries"] for r in runs) / len(runs)

            agg_steps: dict[int, dict[str, Any]] = {}
            for n_steps in (2, 3, 4, 5):
                step_accs = [
                    r["step_breakdown"][n_steps]["verified_accuracy"]
                    for r in runs
                    if n_steps in r["step_breakdown"]
                ]
                if step_accs:
                    agg_steps[n_steps] = {
                        "verified_accuracy": round(sum(step_accs) / len(step_accs), 4),
                    }

            agg_summary[cfg_key] = {
                "configuration": runs[0]["configuration"],
                "mean_verified_answer_accuracy": round(mean_verified, 4),
                "mean_plan_validity_rate": round(mean_valid, 4),
                "mean_plan_exact_match_rate": round(mean_exact, 4),
                "mean_abstention_correctness": round(mean_abst, 4),
                "mean_failure_rate": round(mean_fail, 4),
                "mean_latency_ms": round(mean_lat, 2),
                "mean_retries": round(mean_ret, 2),
                "step_breakdown": agg_steps,
            }

        conv_runs = list(benchmark_report["conversational_eval"].values())
        if conv_runs:
            mean_t1 = sum(r["turn1_accuracy"] for r in conv_runs) / len(conv_runs)
            mean_t2 = sum(r["turn2_accuracy"] for r in conv_runs) / len(conv_runs)
            mean_amb = sum(r["ambiguous_reference_abstention"] for r in conv_runs) / len(conv_runs)
            agg_summary["conversational_memory"] = {
                "mean_turn1_accuracy": round(mean_t1, 4),
                "mean_turn2_accuracy": round(mean_t2, 4),
                "mean_ambiguous_abstention": round(mean_amb, 4),
            }

        benchmark_report["summary"] = agg_summary
        report_path = output_dir / "m17_t4_report.json"
        report_path.write_text(json.dumps(benchmark_report, indent=2))
        logger.info("Saved complete M17 benchmark report to %s", report_path)

    return benchmark_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run M17 Clean Benchmark on NVIDIA Tesla T4")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output-dir", type=Path, default=Path("runs/m17_planning"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="fp16" if torch.cuda.is_available() else "fp32")
    parser.add_argument("--samples-per-archetype", type=int, default=10)
    parser.add_argument("--abstention-samples", type=int, default=8)
    parser.add_argument("--conversational-sessions", type=int, default=10)
    args = parser.parse_args()

    run_benchmark(
        seeds=args.seeds,
        output_dir=args.output_dir,
        device=args.device,
        precision=args.precision,
        samples_per_archetype=args.samples_per_archetype,
        num_abstention_samples=args.abstention_samples,
        num_conversational_sessions=args.conversational_sessions,
    )
