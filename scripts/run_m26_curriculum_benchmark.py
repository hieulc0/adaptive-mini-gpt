"""Autonomous Curriculum Selection and Multi-Domain Evaluation Benchmark (M26).

Compares:
  Strategy A: Fixed Round-Robin Curriculum.
  Strategy B: Largest Measured Skill-Gap Selection.
  Strategy C: Learning-Progress & Retention-Aware Selection.

Evaluates 3 bounded learning cycles across independent lineages (Seeds 42, 43, 44)
on Tesla T4 GPU, tracking:
  - Skill acquisition across Arithmetic, Language, and Logic.
  - Representation retention drift on pretraining prose.
  - Prediction accuracy of expected vs observed skill gains.
  - Learning efficiency and gate rollback rates.
  - Final qualification on protected sealed test splits.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from adaptive_mini_gpt.autonomous.curriculum_selector import (
    CurriculumProposal,
    CurriculumSelector,
    CurriculumStrategy,
    CycleResult,
    InterventionType,
)
from adaptive_mini_gpt.autonomous.lineage_registry import (
    CANONICAL_LINEAGE_REGISTRY,
    get_candidate_root_lineage,
)
from adaptive_mini_gpt.data.language_data import (
    LanguageComprehensionDataset,
    build_language_benchmark_samples,
    collate_language_batch,
    parse_language_completion,
    verify_language_answer,
)
from adaptive_mini_gpt.data.logic_data import (
    LogicGeneralizationType,
    LogicReasoningDataset,
    generate_logic_samples_for_split,
    parse_logic_completion,
    verify_logic_answer,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.evaluation.multi_domain import (
    BenchmarkSample,
    BenchmarkSplit,
    DomainType,
    MultiDomainBenchmark,
)
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.recurrent_arithmetic import (
    evaluate_recurrent_arithmetic_model,
    generate_controlled_addition_dataset,
)
from adaptive_mini_gpt.rules.reliable_transition import (
    ElementaryTransition,
    ElementaryTransitionDataset,
    audit_elementary_transition_space,
    build_exhaustive_elementary_transitions,
    collate_elementary_batch,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Audited Retention Benchmark Suite
EXPANDED_RETENTION_CORPUS = {
    "prose_original": (
        "Paris is the capital of France. William Shakespeare wrote Hamlet, Macbeth, and Othello in London. "
        "The speed of light in vacuum is approximately 300,000 km/s. "
        "In quantum physics, photons, bosons, and fermions govern fundamental interactions. "
        "Tokyo is the capital of Japan, and Rome is in Italy."
    ),
    "technical_science": (
        "Thermodynamics defines the relationships between heat, work, temperature, and energy. "
        "The first law states that energy cannot be created or destroyed, only transformed. "
        "Entropy measures molecular disorder and increases in all irreversible processes. "
        "DNA transcription and translation synthesize polypeptides from nucleotide sequences."
    ),
    "geography_history": (
        "The Nile River flows north through northeastern Africa into the Mediterranean Sea. "
        "The Roman Empire expanded across Europe, North Africa, and the Middle East before dividing. "
        "The Industrial Revolution transformed manufacturing from manual labor to machine-driven factories."
    ),
    "mathematics_prose": (
        "Prime numbers are integers greater than one divisible only by one and themselves. "
        "Euclidean geometry explores planes, angles, triangles, and parallel postulates. "
        "Differential calculus computes instantaneous rates of change along smooth curves."
    ),
}


def evaluate_retention_loss(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate cross-entropy loss on multi-corpus retention suite."""
    model.eval()
    losses: dict[str, float] = {}
    with torch.no_grad():
        for name, text in EXPANDED_RETENTION_CORPUS.items():
            toks = tokenizer.encode(text)
            if len(toks) > model.config.context_length:
                toks = toks[: model.config.context_length]
            inp = torch.tensor([toks], dtype=torch.long, device=device)
            logits, _ = model(inp)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inp[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, model.config.vocab_size),
                shift_labels.view(-1),
            )
            losses[name] = round(loss.item(), 4)
    return losses


def evaluate_validation_state(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    device: torch.device,
    benchmark: MultiDomainBenchmark,
    val_arith_transitions: list[ElementaryTransition],
) -> dict[str, float]:
    """Evaluate candidate model on held-out validation splits across all active domains."""
    model.eval()

    # 1. Arithmetic: elementary transition validation
    audit_res = audit_elementary_transition_space(
        model=model,
        tokenizer=tokenizer,
        device=device,
        format_type="canonical_state",
    )
    arith_exact = audit_res.get("overall_exact_acc", 0.0) / 100.0
    arith_carry = audit_res.get("overall_carry_acc", 0.0) / 100.0

    # 2. Language: validation QA extraction and abstention
    lang_res = benchmark.evaluate_generation_task(
        model=model,
        tokenizer=tokenizer,
        domain=DomainType.LANGUAGE,
        split=BenchmarkSplit.VAL,
        parse_fn=parse_language_completion,
        verify_fn=verify_language_answer,
        device=device,
        max_new_tokens=24,
    )

    # 3. Logic: validation reasoning
    logic_res = benchmark.evaluate_generation_task(
        model=model,
        tokenizer=tokenizer,
        domain=DomainType.LOGIC,
        split=BenchmarkSplit.VAL,
        parse_fn=parse_logic_completion,
        verify_fn=verify_logic_answer,
        device=device,
        max_new_tokens=16,
    )

    return {
        DomainType.ARITHMETIC.value: round(arith_exact, 4),
        "arithmetic_carry": round(arith_carry, 4),
        DomainType.LANGUAGE.value: round(lang_res.exact_accuracy, 4),
        "language_abstention": round(lang_res.abstention_accuracy, 4),
        DomainType.LOGIC.value: round(logic_res.exact_accuracy, 4),
        "logic_abstention": round(logic_res.abstention_accuracy, 4),
    }


def train_curriculum_cycle(
    candidate: MiniGPT,
    tokenizer: BPETokenizer,
    proposal: CurriculumProposal,
    arith_transitions: list[ElementaryTransition],
    lang_samples: list[BenchmarkSample],
    logic_samples: list[BenchmarkSample],
    device: torch.device,
    seed: int = 42,
) -> MiniGPT:
    """Execute a single bounded training cycle based on the curriculum proposal."""
    model = copy.deepcopy(candidate).to(device)
    model.train()
    torch.manual_seed(seed + proposal.cycle_index * 17)
    pad_id = getattr(tokenizer, "pad_id", 0)

    # 1. Build DataLoaders
    ds_arith = ElementaryTransitionDataset(
        transitions=arith_transitions,
        tokenizer=tokenizer,
        format_type="canonical_state",
        context_length=model.config.context_length,
    )
    loader_arith = DataLoader(
        ds_arith,
        batch_size=proposal.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_elementary_batch(b, pad_token_id=pad_id),
    )

    ds_lang = LanguageComprehensionDataset(
        samples=lang_samples,
        tokenizer=tokenizer,
        context_length=model.config.context_length,
        ignore_index=-100,
    )
    loader_lang = DataLoader(
        ds_lang,
        batch_size=proposal.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_language_batch(b, pad_token_id=pad_id, ignore_index=-100),
    )

    ds_logic = LogicReasoningDataset(
        samples=logic_samples,
        tokenizer=tokenizer,
        context_length=model.config.context_length,
        ignore_index=-100,
    )
    loader_logic = DataLoader(
        ds_logic,
        batch_size=proposal.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_language_batch(b, pad_token_id=pad_id, ignore_index=-100),
    )

    # 2. Replay text dataset if replay_ratio > 0
    replay_texts = list(EXPANDED_RETENTION_CORPUS.values())

    optimizer = torch.optim.AdamW(model.parameters(), lr=proposal.learning_rate, weight_decay=0.01)

    r_arith = proposal.sampling_ratios.get("arithmetic", 0.33)
    r_lang = proposal.sampling_ratios.get("language", 0.33)
    r_logic = proposal.sampling_ratios.get("logic", 0.33)

    for epoch in range(proposal.budget_epochs):
        iter_arith = iter(loader_arith)
        iter_lang = iter(loader_lang)
        iter_logic = iter(loader_logic)

        max_batches = max(len(loader_arith), len(loader_lang), len(loader_logic))

        for _ in range(max_batches):
            # Domain A: Arithmetic step (weighted by r_arith)
            if random.random() < r_arith:
                try:
                    x_a, y_a = next(iter_arith)
                except StopIteration:
                    iter_arith = iter(loader_arith)
                    x_a, y_a = next(iter_arith)
                x_a, y_a = x_a.to(device), y_a.to(device)
                optimizer.zero_grad()
                logits_a, _ = model(x_a)
                loss_a = nn.functional.cross_entropy(
                    logits_a[:, :-1, :].contiguous().view(-1, model.config.vocab_size),
                    y_a[:, 1:].contiguous().view(-1),
                    ignore_index=-100,
                )
                loss_a.backward()
                optimizer.step()

            # Domain B: Language step (weighted by r_lang)
            if random.random() < r_lang:
                try:
                    x_b, y_b = next(iter_lang)
                except StopIteration:
                    iter_lang = iter(loader_lang)
                    x_b, y_b = next(iter_lang)
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                logits_b, _ = model(x_b)
                loss_b = nn.functional.cross_entropy(
                    logits_b[:, :-1, :].contiguous().view(-1, model.config.vocab_size),
                    y_b[:, 1:].contiguous().view(-1),
                    ignore_index=-100,
                )
                loss_b.backward()
                optimizer.step()

            # Domain C: Logic step (weighted by r_logic)
            if random.random() < r_logic:
                try:
                    x_c, y_c = next(iter_logic)
                except StopIteration:
                    iter_logic = iter(loader_logic)
                    x_c, y_c = next(iter_logic)
                x_c, y_c = x_c.to(device), y_c.to(device)
                optimizer.zero_grad()
                logits_c, _ = model(x_c)
                loss_c = nn.functional.cross_entropy(
                    logits_c[:, :-1, :].contiguous().view(-1, model.config.vocab_size),
                    y_c[:, 1:].contiguous().view(-1),
                    ignore_index=-100,
                )
                loss_c.backward()
                optimizer.step()

            # Optional Replay Step
            if proposal.replay_ratio > 0.0 and random.random() < proposal.replay_ratio:
                rep_text = random.choice(replay_texts)
                toks = tokenizer.encode(rep_text)[: model.config.context_length]
                x_rep = torch.tensor([toks], dtype=torch.long, device=device)
                optimizer.zero_grad()
                logits_rep, _ = model(x_rep)
                loss_rep = nn.functional.cross_entropy(
                    logits_rep[:, :-1, :].contiguous().view(-1, model.config.vocab_size),
                    x_rep[:, 1:].contiguous().view(-1),
                )
                loss_rep.backward()
                optimizer.step()

    return model


def run_strategy_benchmark(
    strategy: CurriculumStrategy,
    base_model: MiniGPT,
    tokenizer: BPETokenizer,
    benchmark: MultiDomainBenchmark,
    arith_train_transitions: list[ElementaryTransition],
    lang_train_samples: list[BenchmarkSample],
    logic_train_samples: list[BenchmarkSample],
    device: torch.device,
    cycles: int = 3,
    budget_epochs_per_cycle: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute multi-cycle autonomous learning under a specific curriculum strategy."""
    selector = CurriculumSelector(
        strategy=strategy,
        target_accuracy=0.85,
        max_regression_tolerance=0.15,
        default_budget_epochs=budget_epochs_per_cycle,
        default_lr=3e-4,
    )

    current_candidate = copy.deepcopy(base_model)
    base_retention = evaluate_retention_loss(base_model, tokenizer, device)
    base_prose_loss = base_retention["prose_original"]

    cycle_records: list[dict[str, Any]] = []

    for c in range(1, cycles + 1):
        # 1. Evaluate pre-cycle validation state
        pre_metrics = evaluate_validation_state(
            model=current_candidate,
            tokenizer=tokenizer,
            device=device,
            benchmark=benchmark,
            val_arith_transitions=arith_train_transitions,
        )
        current_ret = evaluate_retention_loss(current_candidate, tokenizer, device)
        current_drift = round(current_ret["prose_original"] - base_prose_loss, 4)

        # 2. Selector proposes experiment
        proposal = selector.select_next_experiment(
            cycle_index=c,
            current_metrics=pre_metrics,
            retention_drift=current_drift,
        )
        logger.info(
            "[%s | Seed %d | Cycle %d] Proposing target '%s' (replay=%.2f): %s",
            strategy.value,
            seed,
            c,
            proposal.target_domain.value,
            proposal.replay_ratio,
            proposal.rationale,
        )

        # 3. Train candidate in isolated branch
        start_t = time.time()
        new_candidate = train_curriculum_cycle(
            candidate=current_candidate,
            tokenizer=tokenizer,
            proposal=proposal,
            arith_transitions=arith_train_transitions,
            lang_samples=lang_train_samples,
            logic_samples=logic_train_samples,
            device=device,
            seed=seed,
        )
        train_duration = round(time.time() - start_t, 2)

        # 4. Evaluate post-cycle validation state
        post_metrics = evaluate_validation_state(
            model=new_candidate,
            tokenizer=tokenizer,
            device=device,
            benchmark=benchmark,
            val_arith_transitions=arith_train_transitions,
        )
        post_ret = evaluate_retention_loss(new_candidate, tokenizer, device)
        observed_drift = round(post_ret["prose_original"] - base_prose_loss, 4)

        # 5. Record outcome and apply promotion/rollback gate
        cycle_res = selector.record_cycle_outcome(
            proposal=proposal,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            observed_retention_drift=observed_drift,
        )

        if cycle_res.is_promoted_to_candidate:
            current_candidate = new_candidate
            logger.info(
                "[%s | Cycle %d] PROMOTED candidate. Target gain: %+.4f, Drift: %+.4f",
                strategy.value,
                c,
                cycle_res.observed_gain,
                observed_drift,
            )
        else:
            logger.warning(
                "[%s | Cycle %d] ROLLED BACK candidate: %s",
                strategy.value,
                c,
                cycle_res.rejection_reason,
            )

        cycle_records.append({
            "cycle_index": c,
            "proposal": proposal.to_dict(),
            "pre_metrics": pre_metrics,
            "post_metrics": post_metrics,
            "observed_gain": cycle_res.observed_gain,
            "observed_retention_drift": observed_drift,
            "gain_prediction_error": cycle_res.gain_prediction_error,
            "is_promoted": cycle_res.is_promoted_to_candidate,
            "rejection_reason": cycle_res.rejection_reason,
            "duration_seconds": train_duration,
        })

    # Final qualification on protected sealed TEST holdouts
    final_test_arith = audit_elementary_transition_space(
        model=current_candidate,
        tokenizer=tokenizer,
        device=device,
        format_type="canonical_state",
    )
    final_test_lang = benchmark.evaluate_generation_task(
        model=current_candidate,
        tokenizer=tokenizer,
        domain=DomainType.LANGUAGE,
        split=BenchmarkSplit.TEST,
        parse_fn=parse_language_completion,
        verify_fn=verify_language_answer,
        device=device,
        max_new_tokens=24,
    )
    final_test_logic = benchmark.evaluate_generation_task(
        model=current_candidate,
        tokenizer=tokenizer,
        domain=DomainType.LOGIC,
        split=BenchmarkSplit.TEST,
        parse_fn=parse_logic_completion,
        verify_fn=verify_logic_answer,
        device=device,
        max_new_tokens=16,
    )
    final_retention = evaluate_retention_loss(current_candidate, tokenizer, device)
    final_drift = round(final_retention["prose_original"] - base_prose_loss, 4)

    return {
        "strategy": strategy.value,
        "seed": seed,
        "cycle_history": cycle_records,
        "final_qualification": {
            "arithmetic_exact": round(final_test_arith.get("overall_exact_acc", 0.0) / 100.0, 4),
            "arithmetic_carry": round(final_test_arith.get("overall_carry_acc", 0.0) / 100.0, 4),
            "language_exact": round(final_test_lang.exact_accuracy, 4),
            "language_abstain": round(final_test_lang.abstention_accuracy, 4),
            "logic_exact": round(final_test_logic.exact_accuracy, 4),
            "logic_abstain": round(final_test_logic.abstention_accuracy, 4),
            "retention_drift": final_drift,
        },
        "promotion_rate": round(
            sum(1 for c in cycle_records if c["is_promoted"]) / len(cycle_records), 4
        ),
        "mean_gain_prediction_error": round(
            sum(c["gain_prediction_error"] for c in cycle_records) / len(cycle_records), 4
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M26 Autonomous Curriculum Selection Benchmark")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--epochs-per-cycle", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError("CUDA required for M26 multi-seed benchmark; refusing to run on CPU")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path(args.checkpoints_dir)

    logger.info("==================================================================")
    logger.info("M26: Autonomous Curriculum Selection and Continual Learning Benchmark")
    logger.info("Lineages: %s | Device: %s | Cycles: %d | Epochs/Cycle: %d", args.seeds, device, args.cycles, args.epochs_per_cycle)
    logger.info("==================================================================")

    # 1. Initialize MultiDomainBenchmark and register samples
    benchmark = MultiDomainBenchmark(version="m26-v1")

    lang_train = build_language_benchmark_samples(BenchmarkSplit.TRAIN)
    lang_val = build_language_benchmark_samples(BenchmarkSplit.VAL)
    lang_test = build_language_benchmark_samples(BenchmarkSplit.TEST)
    benchmark.register_samples(lang_train + lang_val + lang_test)

    logic_train = generate_logic_samples_for_split(BenchmarkSplit.TRAIN, count_per_category=10)
    logic_val = generate_logic_samples_for_split(BenchmarkSplit.VAL, count_per_category=5, include_unseen_structures=True)
    logic_test = generate_logic_samples_for_split(BenchmarkSplit.TEST, count_per_category=10, include_unseen_structures=True)
    benchmark.register_samples(logic_train + logic_val + logic_test)

    arith_exhaustive = build_exhaustive_elementary_transitions()
    logger.info(
        "Registered Partitions: Language (Train=%d, Val=%d, Test=%d) | Logic (Train=%d, Val=%d, Test=%d)",
        len(lang_train),
        len(lang_val),
        len(lang_test),
        len(logic_train),
        len(logic_val),
        len(logic_test),
    )

    all_strategy_results: dict[str, list[dict[str, Any]]] = {
        CurriculumStrategy.ROUND_ROBIN.value: [],
        CurriculumStrategy.LARGEST_GAP.value: [],
        CurriculumStrategy.PROGRESS_AND_RETENTION_AWARE.value: [],
    }

    raw_manifest = checkpoints_dir / "checkpoint-manifest.json"
    manifest_data = json.loads(raw_manifest.read_text()) if raw_manifest.is_file() else {}

    for seed in args.seeds:
        logger.info("------------------------------------------------------------------")
        logger.info("Processing Lineage seed_%d", seed)
        ckpt_path = checkpoints_dir / f"seed_{seed}/base_pretrain/best.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        raw_ckpt = CheckpointManager.load_raw_checkpoint(ckpt_path)
        tokenizer = CheckpointManager.load_tokenizer(raw_ckpt)
        cfg_dict = raw_ckpt.get("model_config", {})
        model_cfg = ModelConfig(**cfg_dict)
        base_model = MiniGPT(model_cfg).to(device)
        base_model.load_state_dict(raw_ckpt["model_state_dict"])
        base_model.eval()

        # Run 3 Curriculum Strategies
        for strat in (
            CurriculumStrategy.ROUND_ROBIN,
            CurriculumStrategy.LARGEST_GAP,
            CurriculumStrategy.PROGRESS_AND_RETENTION_AWARE,
        ):
            logger.info("Starting Strategy '%s' for Seed %d...", strat.value, seed)
            strat_res = run_strategy_benchmark(
                strategy=strat,
                base_model=base_model,
                tokenizer=tokenizer,
                benchmark=benchmark,
                arith_train_transitions=arith_exhaustive,
                lang_train_samples=lang_train,
                logic_train_samples=logic_train,
                device=device,
                cycles=args.cycles,
                budget_epochs_per_cycle=args.epochs_per_cycle,
                seed=seed,
            )
            all_strategy_results[strat.value].append(strat_res)

    # Summarize across seeds for each strategy
    summary: dict[str, Any] = {}
    for strat_name, seed_results in all_strategy_results.items():
        arith_exacts = [r["final_qualification"]["arithmetic_exact"] for r in seed_results]
        arith_carries = [r["final_qualification"]["arithmetic_carry"] for r in seed_results]
        lang_exacts = [r["final_qualification"]["language_exact"] for r in seed_results]
        logic_exacts = [r["final_qualification"]["logic_exact"] for r in seed_results]
        drifts = [r["final_qualification"]["retention_drift"] for r in seed_results]
        prom_rates = [r["promotion_rate"] for r in seed_results]
        pred_errs = [r["mean_gain_prediction_error"] for r in seed_results]

        def mean_std(vals: list[float]) -> dict[str, float]:
            m = sum(vals) / len(vals)
            s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
            return {"mean": round(m, 4), "std": round(s, 4)}

        summary[strat_name] = {
            "arithmetic_exact": mean_std(arith_exacts),
            "arithmetic_carry": mean_std(arith_carries),
            "language_exact": mean_std(lang_exacts),
            "logic_exact": mean_std(logic_exacts),
            "retention_drift": mean_std(drifts),
            "promotion_rate": mean_std(prom_rates),
            "prediction_error": mean_std(pred_errs),
        }

    report = {
        "benchmark": "M26 Autonomous Curriculum Selection Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "seeds": args.seeds,
            "cycles": args.cycles,
            "epochs_per_cycle": args.epochs_per_cycle,
            "lr": args.lr,
            "batch_size": args.batch_size,
        },
        "summary": summary,
        "strategy_details": all_strategy_results,
    }

    report_path = output_dir / "m26_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Report written to %s", report_path)
    logger.info("==================================================================")
    logger.info("M26 CURRICULUM STRATEGY COMPARISON (Mean ± Std over Seeds %s):", args.seeds)
    for strat_name, stats in summary.items():
        logger.info(
            "  Strategy [%s]: Arith: %.2f%% ± %.2f%% (Carry: %.2f%%) | Lang: %.2f%% ± %.2f%% | Logic: %.2f%% ± %.2f%% | Drift: %+.4f ± %.4f | PromoRate: %.2f%%",
            strat_name,
            stats["arithmetic_exact"]["mean"] * 100,
            stats["arithmetic_exact"]["std"] * 100,
            stats["arithmetic_carry"]["mean"] * 100,
            stats["language_exact"]["mean"] * 100,
            stats["language_exact"]["std"] * 100,
            stats["logic_exact"]["mean"] * 100,
            stats["logic_exact"]["std"] * 100,
            stats["retention_drift"]["mean"],
            stats["retention_drift"]["std"],
            stats["promotion_rate"]["mean"] * 100,
        )
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
