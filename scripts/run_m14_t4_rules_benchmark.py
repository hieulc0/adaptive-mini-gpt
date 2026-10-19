"""M14 Empirical Benchmark on NVIDIA Tesla T4: Verified Rules and Reasoning Loop.

Compares four experimental configurations across 3 independent seeds:
1. Model alone (Baseline neural capability)
2. Oracle-selected rule engine (Upper bound with ground-truth rule selection & args)
3. Mini-GPT + rule engine (End-to-end tool calling without retry)
4. Mini-GPT + rules + verification/retry (Feedback loop with bounded error recovery)

Evaluates on unfamiliar, paraphrased natural-language questions with 0% train/test overlap across 4 operations:
- 'reverse'
- 'add'
- 'compare'
- 'average_speed'

Records:
- Rule-selection accuracy
- Argument-extraction accuracy
- Verified-answer exact match accuracy
- Failure rate
- Wall-clock latency (ms)
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
    ReasoningController,
    RuleExecutor,
    generate_rule_instruction_corpus,
    generate_rule_problems,
)
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m14_benchmark")


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


def adapt_tool_calling(
    base_model: MiniGPT,
    tokenizer: BPETokenizer,
    cfg: ModelConfig,
    save_dir: Path,
    device: str = "cuda",
    precision: str = "fp16",
    steps: int = 150,
    seed: int = 42,
) -> MiniGPT:
    """Adapt model to propose structured tool calls from natural language questions (train split only)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    tool_model = copy.deepcopy(base_model)

    # Strictly use training split for tool supervision
    tool_text, tool_fp = generate_rule_instruction_corpus(
        num_samples_per_rule=100, seed=seed, split="train", format_type="tool"
    )
    tokens = tokenizer.encode(tool_text)
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
        model=tool_model,
        model_config=cfg,
        training_config=train_cfg,
        train_dataset=ds_train,
        val_dataset=ds_val,
        tokenizer=tokenizer,
        dataset_fingerprint=tool_fp,
    )
    trainer.train()
    return tool_model


def run_m14_benchmark(
    seeds: list[int],
    output_dir: Path,
    device: str = "cuda",
    precision: str = "fp16",
    samples_per_rule: int = 15,
) -> dict[str, Any]:
    """Execute complete M14 empirical ablation study across all 4 configurations and seeds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    executor = RuleExecutor()

    configurations = [
        ("neural_only", "1. Model Alone (Baseline)"),
        ("oracle", "2. Oracle-Selected Rule Engine"),
        ("rule_engine", "3. Mini-GPT + Rule Engine"),
        ("rule_engine_retry", "4. Mini-GPT + Rules + Verification/Retry"),
    ]

    benchmark_report: dict[str, Any] = {
        "title": "M14 — Verified Rules and Reasoning Loop Empirical Qualification",
        "device": device,
        "precision": precision,
        "seeds": seeds,
        "operations": ["reverse", "add", "compare", "average_speed"],
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

        # 2. Adapt tool calling model on train split
        tool_model = adapt_tool_calling(
            base_model=base_model,
            tokenizer=tokenizer,
            cfg=cfg,
            save_dir=seed_dir / "tool_calling_adapter",
            device=device,
            precision=precision,
            steps=150,
            seed=seed,
        )

        # 3. Generate pristine held-out test split (0% train/val overlap)
        test_problems = generate_rule_problems(
            num_samples_per_rule=samples_per_rule, seed=seed + 999, split="test"
        )
        logger.info(
            "Evaluating %d held-out test problems on lineage seed %d...",
            len(test_problems),
            seed,
        )

        # Set up reasoning controllers
        neural_controller = ReasoningController(
            model=base_model, tokenizer=tokenizer, executor=executor, device=dev
        )
        tool_controller = ReasoningController(
            model=tool_model, tokenizer=tokenizer, executor=executor, device=dev
        )

        lineage_eval: dict[str, Any] = {}

        for cfg_key, cfg_title in configurations:
            logger.info("--- Evaluating: %s ---", cfg_title)
            results: list[dict[str, Any]] = []

            for prob in test_problems:
                ctrl = neural_controller if cfg_key == "neural_only" else tool_controller
                res = ctrl.solve(
                    question=prob.question,
                    expected_rule=prob.rule_name,
                    expected_answer=prob.expected_answer,
                    ground_truth_args=prob.args,
                    mode=cfg_key,
                    max_retries=2,
                )
                results.append(res.to_dict())

            tot = len(results)
            rule_sel_corr = sum(1 for r in results if r["rule_selection_correct"])
            arg_ext_corr = sum(1 for r in results if r["arg_extraction_correct"])
            ans_corr = sum(1 for r in results if r["answer_correct"])
            failures = sum(1 for r in results if r["status"] == "failed")
            mean_lat = sum(r["latency_ms"] for r in results) / tot if tot > 0 else 0.0
            mean_retries = sum(r["total_retries"] for r in results) / tot if tot > 0 else 0.0

            # Per-rule breakdown
            per_rule: dict[str, Any] = {}
            for op in ("reverse", "add", "compare", "average_speed"):
                op_runs = [r for r in results if r["ground_truth_rule"] == op]
                n_op = len(op_runs)
                op_acc = sum(1 for r in op_runs if r["answer_correct"]) / n_op if n_op > 0 else 0.0
                op_rule_acc = (
                    sum(1 for r in op_runs if r["rule_selection_correct"]) / n_op
                    if n_op > 0
                    else 0.0
                )
                op_arg_acc = (
                    sum(1 for r in op_runs if r["arg_extraction_correct"]) / n_op
                    if n_op > 0
                    else 0.0
                )
                per_rule[op] = {
                    "count": n_op,
                    "answer_accuracy": round(op_acc, 4),
                    "rule_selection_acc": round(op_rule_acc, 4),
                    "arg_extraction_acc": round(op_arg_acc, 4),
                }

            cfg_summary = {
                "configuration": cfg_title,
                "total_problems": tot,
                "rule_selection_accuracy": round(rule_sel_corr / tot, 4) if tot > 0 else 0.0,
                "arg_extraction_accuracy": round(arg_ext_corr / tot, 4) if tot > 0 else 0.0,
                "verified_answer_accuracy": round(ans_corr / tot, 4) if tot > 0 else 0.0,
                "failure_rate": round(failures / tot, 4) if tot > 0 else 0.0,
                "mean_latency_ms": round(mean_lat, 2),
                "mean_retries": round(mean_retries, 2),
                "per_rule_breakdown": per_rule,
                "sample_predictions": results[:3],
            }

            lineage_eval[cfg_key] = cfg_summary
            config_aggregates[cfg_key].append(cfg_summary)

            logger.info(
                "[%s] Verified Answer Acc: %.2f%%, Rule Sel Acc: %.2f%%, Arg Ext Acc: %.2f%%, Failures: %.2f%%, Latency: %.1fms",
                cfg_title,
                cfg_summary["verified_answer_accuracy"] * 100,
                cfg_summary["rule_selection_accuracy"] * 100,
                cfg_summary["arg_extraction_accuracy"] * 100,
                cfg_summary["failure_rate"] * 100,
                cfg_summary["mean_latency_ms"],
            )

        benchmark_report["lineage_runs"][f"seed_{seed}"] = lineage_eval

    # 4. Multi-seed Cross-Lineage Summaries
    overall_summary: dict[str, Any] = {}
    for cfg_key, cfg_title in configurations:
        runs = config_aggregates[cfg_key]
        n_seeds = len(runs)
        mean_rule_sel = sum(r["rule_selection_accuracy"] for r in runs) / n_seeds
        mean_arg_ext = sum(r["arg_extraction_accuracy"] for r in runs) / n_seeds
        mean_ans_acc = sum(r["verified_answer_accuracy"] for r in runs) / n_seeds
        mean_fail = sum(r["failure_rate"] for r in runs) / n_seeds
        mean_lat = sum(r["mean_latency_ms"] for r in runs) / n_seeds
        mean_retries = sum(r["mean_retries"] for r in runs) / n_seeds

        overall_summary[cfg_key] = {
            "configuration": cfg_title,
            "seeds_evaluated": n_seeds,
            "mean_rule_selection_accuracy": round(mean_rule_sel, 4),
            "mean_arg_extraction_accuracy": round(mean_arg_ext, 4),
            "mean_verified_answer_accuracy": round(mean_ans_acc, 4),
            "mean_failure_rate": round(mean_fail, 4),
            "mean_latency_ms": round(mean_lat, 2),
            "mean_retries": round(mean_retries, 2),
        }

    benchmark_report["summary"] = overall_summary

    # Save benchmark report
    report_file = output_dir / "m14_t4_report.json"
    report_file.write_text(json.dumps(benchmark_report, indent=2))
    logger.info("\n" + "=" * 80)
    logger.info("M14 BENCHMARK COMPLETE! Consolidated report written to: %s", report_file)
    logger.info("Summary Across All Seeds:")
    for cfg_key, s in overall_summary.items():
        logger.info(
            "  * %s: Verified Acc=%.2f%%, Rule Sel=%.2f%%, Arg Ext=%.2f%%, Failures=%.2f%%, Latency=%.1fms",
            s["configuration"],
            s["mean_verified_answer_accuracy"] * 100,
            s["mean_rule_selection_accuracy"] * 100,
            s["mean_arg_extraction_accuracy"] * 100,
            s["mean_failure_rate"] * 100,
            s["mean_latency_ms"],
        )
    logger.info("=" * 80)

    return benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser(description="M14 Verified Rules and Reasoning Loop Benchmark")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output-dir", type=str, default="runs/m14_rules")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="fp16" if torch.cuda.is_available() else "fp32")
    parser.add_argument("--samples-per-rule", type=int, default=15)
    args = parser.parse_args()

    run_m14_benchmark(
        seeds=args.seeds,
        output_dir=Path(args.output_dir),
        device=args.device,
        precision=args.precision,
        samples_per_rule=args.samples_per_rule,
    )


if __name__ == "__main__":
    main()
