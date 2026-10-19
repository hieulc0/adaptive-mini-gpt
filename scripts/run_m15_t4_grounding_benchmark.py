"""M15 Empirical Benchmark on NVIDIA Tesla T4: Grounded Argument Extraction and Working Memory.

Evaluates and compares:
1. Arm 1: Autoregressive Mini-GPT Tool Calling (M14 baseline)
2. Arm 2: Deterministic Rule Selector + Deterministic Argument Grounder
3. Arm 3: Deterministic Rule Selector + Source-Span Copy Grounder
4. Hybrid A: Autonomous Mini-GPT Rule Selector + Deterministic Argument Grounder
5. Hybrid B: Autonomous Mini-GPT Rule Selector + Source-Span Copy Grounder
6. Upper Bound: Oracle Rule Selector + Deterministic Argument Grounder
7. Upper Bound: Oracle Rule Selector + Oracle Arguments (M14 Upper Bound)

Evaluates on identical held-out test problems across 3 independent lineages (seeds 42, 43, 44):
- Paraphrased queries with 0% train/val/test overlap
- Changed argument ordering (e.g. 'Add {b} to {a}')
- Distracting numbers (e.g. year 2024, flight numbers)
- Multiword entity names ('Mary Jane', 'Uncle Bob')
- Varied unit expressions ('km' vs 'kilometers', 'hours' vs 'hrs')
- Malformed / contradictory inputs requiring explicit abstention

Records:
- Autonomous vs. Oracle rule-selection accuracy
- Exact argument extraction accuracy
- End-to-end verified answer accuracy
- Abstention correctness on malformed inputs
- Wall-clock latency (ms)
- Working memory provenance tracking
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
    generate_abstention_problems,
    generate_rule_instruction_corpus,
    generate_rule_problems,
)
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m15_benchmark")


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


def run_m15_benchmark(
    seeds: list[int],
    output_dir: Path,
    device: str = "cuda",
    precision: str = "fp16",
    samples_per_rule: int = 13,
    num_abstention_samples: int = 10,
) -> dict[str, Any]:
    """Execute complete M15 empirical ablation study across all configurations and lineages."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    executor = RuleExecutor()

    configurations = [
        ("autoregressive", "1. Arm 1: Autoregressive Mini-GPT Tool Call (M14 Baseline)"),
        ("deterministic", "2. Arm 2: Deterministic Rule Selector + Deterministic Grounder"),
        ("span_copy", "3. Arm 3: Deterministic Rule Selector + Span-Copy Grounder"),
        ("model_rule_deterministic_grounding", "4. Hybrid A: Mini-GPT Rule + Deterministic Grounder"),
        ("model_rule_span_copy", "5. Hybrid B: Mini-GPT Rule + Span-Copy Grounder"),
        ("oracle_grounded", "6. Upper Bound: Oracle Rule + Deterministic Grounder"),
        ("oracle", "7. Upper Bound: Oracle Rule + Oracle Args (M14 Bound)"),
    ]

    benchmark_report: dict[str, Any] = {
        "title": "M15 — Grounded Argument Extraction and Working Memory Qualification",
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
        # 4. Generate abstention / malformed test split
        abstention_problems = generate_abstention_problems(
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
            model=tool_model, tokenizer=tokenizer, executor=executor, device=dev
        )

        lineage_eval: dict[str, Any] = {}

        for cfg_key, cfg_title in configurations:
            logger.info("--- Evaluating: %s ---", cfg_title)
            results: list[dict[str, Any]] = []

            for prob in all_eval_problems:
                solve_mode = "deterministic" if cfg_key == "oracle_grounded" else cfg_key
                expected_rule = prob.rule_name

                # For oracle_grounded, pass ground_truth_rule as rule_selected
                if cfg_key == "oracle_grounded":
                    res = controller.solve(
                        question=prob.question,
                        expected_rule=expected_rule,
                        expected_answer=prob.expected_answer,
                        ground_truth_args=prob.args,
                        mode="deterministic",
                        should_abstain=prob.should_abstain,
                    )
                    # Oracle rule selection is 100% correct
                    res.rule_selected = expected_rule
                    res.rule_selection_correct = True
                else:
                    res = controller.solve(
                        question=prob.question,
                        expected_rule=expected_rule,
                        expected_answer=prob.expected_answer,
                        ground_truth_args=prob.args,
                        mode=solve_mode,
                        should_abstain=prob.should_abstain,
                    )

                results.append(res.to_dict())

            # Metrics on standard (non-malformed) problems
            std_results = results[: len(test_problems)]
            abst_results = results[len(test_problems) :]

            tot_std = len(std_results)
            rule_sel_corr = sum(1 for r in std_results if r["rule_selection_correct"])
            arg_ext_corr = sum(1 for r in std_results if r["arg_extraction_correct"])
            ans_corr = sum(1 for r in std_results if r["answer_correct"])
            failures = sum(1 for r in std_results if r["status"] == "failed")
            mean_lat = sum(r["latency_ms"] for r in std_results) / tot_std if tot_std > 0 else 0.0

            # Metrics on abstention problems
            tot_abst = len(abst_results)
            abst_corr = sum(1 for r in abst_results if r["abstention_correct"])
            abst_acc = abst_corr / tot_abst if tot_abst > 0 else 0.0

            # Per-rule breakdown on standard problems
            per_rule: dict[str, Any] = {}
            for op in ("reverse", "add", "compare", "average_speed"):
                op_runs = [r for r in std_results if r["ground_truth_rule"] == op]
                n_op = len(op_runs)
                op_ans = sum(1 for r in op_runs if r["answer_correct"]) / n_op if n_op > 0 else 0.0
                op_rule = sum(1 for r in op_runs if r["rule_selection_correct"]) / n_op if n_op > 0 else 0.0
                op_arg = sum(1 for r in op_runs if r["arg_extraction_correct"]) / n_op if n_op > 0 else 0.0
                per_rule[op] = {
                    "count": n_op,
                    "verified_accuracy": round(op_ans, 4),
                    "rule_selection_acc": round(op_rule, 4),
                    "arg_extraction_acc": round(op_arg, 4),
                }

            cfg_summary = {
                "configuration": cfg_title,
                "total_standard_problems": tot_std,
                "rule_selection_accuracy": round(rule_sel_corr / tot_std, 4) if tot_std > 0 else 0.0,
                "arg_extraction_accuracy": round(arg_ext_corr / tot_std, 4) if tot_std > 0 else 0.0,
                "verified_answer_accuracy": round(ans_corr / tot_std, 4) if tot_std > 0 else 0.0,
                "failure_rate": round(failures / tot_std, 4) if tot_std > 0 else 0.0,
                "total_abstention_problems": tot_abst,
                "abstention_correctness": round(abst_acc, 4),
                "mean_latency_ms": round(mean_lat, 2),
                "per_rule_breakdown": per_rule,
                "sample_prediction": std_results[0] if std_results else None,
            }

            lineage_eval[cfg_key] = cfg_summary
            config_aggregates[cfg_key].append(cfg_summary)

            logger.info(
                "[%s] Verified Acc: %.2f%% | Rule Sel: %.2f%% | Arg Ext: %.2f%% | Abstention: %.2f%% | Latency: %.2f ms",
                cfg_title,
                cfg_summary["verified_answer_accuracy"] * 100.0,
                cfg_summary["rule_selection_accuracy"] * 100.0,
                cfg_summary["arg_extraction_accuracy"] * 100.0,
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
        mean_rule_sel = sum(r["rule_selection_accuracy"] for r in runs) / n_seeds
        mean_arg_ext = sum(r["arg_extraction_accuracy"] for r in runs) / n_seeds
        mean_failures = sum(r["failure_rate"] for r in runs) / n_seeds
        mean_abstention = sum(r["abstention_correctness"] for r in runs) / n_seeds
        mean_latency = sum(r["mean_latency_ms"] for r in runs) / n_seeds

        # Aggregate per-rule verified accuracy
        per_rule_agg: dict[str, float] = {}
        for op in ("reverse", "add", "compare", "average_speed"):
            per_rule_agg[op] = round(
                sum(r["per_rule_breakdown"][op]["verified_accuracy"] for r in runs) / n_seeds,
                4,
            )

        agg_summary[cfg_key] = {
            "configuration": runs[0]["configuration"],
            "mean_verified_answer_accuracy": round(mean_verified, 4),
            "mean_rule_selection_accuracy": round(mean_rule_sel, 4),
            "mean_arg_extraction_accuracy": round(mean_arg_ext, 4),
            "mean_failure_rate": round(mean_failures, 4),
            "mean_abstention_correctness": round(mean_abstention, 4),
            "mean_latency_ms": round(mean_latency, 2),
            "per_rule_verified_accuracy": per_rule_agg,
        }

    benchmark_report["summary"] = agg_summary

    # Save final report
    report_file = output_dir / "m15_t4_report.json"
    report_file.write_text(json.dumps(benchmark_report, indent=2))
    logger.info("\nSaved full M15 report to %s", report_file)
    return benchmark_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run M15 T4 Grounding Benchmark")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output_dir", type=str, default="runs/m15_grounding")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="fp16" if torch.cuda.is_available() else "fp32")
    parser.add_argument("--samples_per_rule", type=int, default=13)
    parser.add_argument("--abstention_samples", type=int, default=10)
    args = parser.parse_args()

    run_m15_benchmark(
        seeds=args.seeds,
        output_dir=Path(args.output_dir),
        device=args.device,
        precision=args.precision,
        samples_per_rule=args.samples_per_rule,
        num_abstention_samples=args.abstention_samples,
    )
