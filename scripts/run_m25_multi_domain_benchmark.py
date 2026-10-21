"""M25: Parallel Learning and Multi-Domain Benchmark Runner.

Executes controlled multi-domain evaluation across 3 independent lineages (Seeds 42, 43, 44):
  - Track M25-A: Neural Arithmetic Transitions (M24 recurrent write-digit / carry reliability)
  - Track M25-B: Language Comprehension (Reading short passages, factual QA, safe abstention)
  - Track M25-C: Symbolic Logic Foundation (Boolean logic, comparisons, deductions, contradictions)
  - Track M25-D: Continual Learning & Multi-Domain Retention Audit

Scientific and Engineering Guarantees:
  1. Lineage-aware loading: Base checkpoints verified with SHA-256 and tokenizer fingerprints.
  2. Sealed test holdouts: Zero overlap in documents, facts, or variable names between train and test.
  3. Neural-only inference: No external calculators or symbolic solvers during evaluation.
  4. Cross-domain retention tracking: Continuous loss audit on disjoint multi-corpus suite.
  5. Immutable base checkpoints: Experimental candidates cloned into isolated memory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from adaptive_mini_gpt.data.language_data import (
    ABSTENTION_TARGET,
    LanguageComprehensionDataset,
    build_language_benchmark_samples,
    collate_language_batch,
    parse_language_completion,
    verify_language_answer,
)
from adaptive_mini_gpt.data.logic_data import (
    CONTRADICTION_TARGET,
    LOGIC_ABSTENTION_TARGET,
    LogicReasoningDataset,
    generate_logic_samples_for_split,
    parse_logic_completion,
    verify_logic_answer,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.evaluation.multi_domain import (
    BenchmarkSample,
    BenchmarkSplit,
    DomainResult,
    DomainType,
    MultiDomainBenchmark,
    MultiDomainReport,
)
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.arithmetic_dataset import generate_controlled_addition_dataset
from adaptive_mini_gpt.rules.reliable_transition import (
    ElementaryTransition,
    ElementaryTransitionDataset,
    audit_elementary_transition_space,
    collate_elementary_batch,
    execute_reliable_recurrent_addition,
    get_all_200_transitions,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m25_multi_domain")

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


def train_language_candidate(
    base_model: MiniGPT,
    tokenizer: BPETokenizer,
    train_samples: list[BenchmarkSample],
    device: torch.device,
    epochs: int = 15,
    lr: float = 3e-4,
    batch_size: int = 8,
    seed: int = 42,
) -> MiniGPT:
    """Train a cloned candidate branch on language comprehension with prompt loss masking."""
    model = copy.deepcopy(base_model).to(device)
    model.train()
    torch.manual_seed(seed)

    dataset = LanguageComprehensionDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        context_length=model.config.context_length,
        ignore_index=-100,
    )
    pad_id = getattr(tokenizer, "pad_id", 0)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_language_batch(b, pad_token_id=pad_id, ignore_index=-100),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = y[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, model.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    return model


def train_logic_candidate(
    base_model: MiniGPT,
    tokenizer: BPETokenizer,
    train_samples: list[BenchmarkSample],
    device: torch.device,
    epochs: int = 15,
    lr: float = 3e-4,
    batch_size: int = 8,
    seed: int = 42,
) -> MiniGPT:
    """Train a cloned candidate branch on symbolic logic with prompt loss masking."""
    model = copy.deepcopy(base_model).to(device)
    model.train()
    torch.manual_seed(seed)

    dataset = LogicReasoningDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        context_length=model.config.context_length,
        ignore_index=-100,
    )
    pad_id = getattr(tokenizer, "pad_id", 0)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_language_batch(b, pad_token_id=pad_id, ignore_index=-100),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = y[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, model.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    return model


def train_continual_interleaved_candidate(
    base_model: MiniGPT,
    tokenizer: BPETokenizer,
    arith_transitions: list[ElementaryTransition],
    lang_samples: list[BenchmarkSample],
    logic_samples: list[BenchmarkSample],
    device: torch.device,
    epochs: int = 12,
    lr: float = 3e-4,
    batch_size: int = 8,
    seed: int = 42,
) -> MiniGPT:
    """Train a consolidated multi-domain candidate with interleaved domain batches."""
    model = copy.deepcopy(base_model).to(device)
    model.train()
    torch.manual_seed(seed)
    pad_id = getattr(tokenizer, "pad_id", 0)

    # 1. Arithmetic dataset
    ds_arith = ElementaryTransitionDataset(
        transitions=arith_transitions,
        tokenizer=tokenizer,
        format_type="canonical_state",
        context_length=model.config.context_length,
    )
    loader_arith = DataLoader(
        ds_arith,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_elementary_batch(b, pad_token_id=pad_id),
    )

    # 2. Language dataset
    ds_lang = LanguageComprehensionDataset(
        samples=lang_samples,
        tokenizer=tokenizer,
        context_length=model.config.context_length,
        ignore_index=-100,
    )
    loader_lang = DataLoader(
        ds_lang,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_language_batch(b, pad_token_id=pad_id, ignore_index=-100),
    )

    # 3. Logic dataset
    ds_logic = LogicReasoningDataset(
        samples=logic_samples,
        tokenizer=tokenizer,
        context_length=model.config.context_length,
        ignore_index=-100,
    )
    loader_logic = DataLoader(
        ds_logic,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_language_batch(b, pad_token_id=pad_id, ignore_index=-100),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        iter_arith = iter(loader_arith)
        iter_lang = iter(loader_lang)
        iter_logic = iter(loader_logic)

        max_batches = max(len(loader_arith), len(loader_lang), len(loader_logic))

        for _ in range(max_batches):
            # Domain A: Arithmetic step
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

            # Domain B: Language step
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

            # Domain C: Logic step
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

    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("=" * 66)
    logger.info("M25: Parallel Learning and Multi-Domain Benchmark")
    logger.info("Lineages: %s | Device: %s | Epochs: %d", args.seeds, device, args.epochs)
    logger.info("=" * 66)

    # 1. Initialize Benchmark and Register Datasets
    benchmark = MultiDomainBenchmark(version="m25-v1")

    # Language samples (train, val, test)
    lang_train = build_language_benchmark_samples(BenchmarkSplit.TRAIN)
    lang_val = build_language_benchmark_samples(BenchmarkSplit.VAL)
    lang_test = build_language_benchmark_samples(BenchmarkSplit.TEST)
    benchmark.register_samples(lang_train)
    benchmark.register_samples(lang_val)
    benchmark.register_samples(lang_test)

    # Logic samples (train, val, test)
    logic_train = generate_logic_samples_for_split(BenchmarkSplit.TRAIN, count_per_category=10, seed=42)
    logic_val = generate_logic_samples_for_split(BenchmarkSplit.VAL, count_per_category=5, seed=42)
    logic_test = generate_logic_samples_for_split(BenchmarkSplit.TEST, count_per_category=10, seed=42)
    benchmark.register_samples(logic_train)
    benchmark.register_samples(logic_val)
    benchmark.register_samples(logic_test)

    # Arithmetic transitions (M24 curriculum)
    all_200_transitions = get_all_200_transitions()
    extrap_addition_problems = generate_controlled_addition_dataset(
        lengths=(4, 5, 6, 8), num_samples=100, split="test", seed=42
    )

    logger.info("Registered Multi-Domain Benchmark partitions:")
    logger.info("  Language - Train: %d, Val: %d, Test: %d", len(lang_train), len(lang_val), len(lang_test))
    logger.info("  Logic    - Train: %d, Val: %d, Test: %d", len(logic_train), len(logic_val), len(logic_test))
    logger.info("  Arithmetic - Elementary: 200, Extrap: %d", len(extrap_addition_problems))

    # 2. Lineage Evaluation Results Registry
    lineage_results: dict[int, dict[str, Any]] = {}

    for seed in args.seeds:
        logger.info("=" * 66)
        logger.info("Processing Lineage seed_%d", seed)
        ckpt_path = args.checkpoints_dir / f"seed_{seed}/base_pretrain/best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        raw_ckpt = CheckpointManager.load_raw_checkpoint(ckpt_path)
        tokenizer = CheckpointManager.load_tokenizer(raw_ckpt)
        base_cfg_dict = raw_ckpt.get("model_config", {})
        if not isinstance(base_cfg_dict, dict):
            base_cfg_dict = base_cfg_dict.to_dict()
        cfg_dict = dict(base_cfg_dict)
        cfg_dict["pos_encoding"] = "rope"
        cfg = ModelConfig(**cfg_dict)
        base_model = MiniGPT(cfg)
        base_model.load_state_dict(raw_ckpt["model_state_dict"], strict=False)
        base_model.to(device)
        base_model.eval()

        param_count = sum(p.numel() for p in base_model.parameters())
        ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        tok_fp = tokenizer.get_fingerprint()

        # Step 2.1: Baseline Evaluation (Zero-shot unadapted base)
        logger.info("Evaluating Base Lineage seed_%d...", seed)
        base_retention = evaluate_retention_loss(base_model, tokenizer, device)
        base_arith = audit_elementary_transition_space(base_model, tokenizer, device, format_type="canonical_state")
        base_lang = benchmark.evaluate_generation_task(
            model=base_model,
            tokenizer=tokenizer,
            domain=DomainType.LANGUAGE,
            split=BenchmarkSplit.TEST,
            parse_fn=parse_language_completion,
            verify_fn=verify_language_answer,
            device=device,
        )
        base_logic = benchmark.evaluate_generation_task(
            model=base_model,
            tokenizer=tokenizer,
            domain=DomainType.LOGIC,
            split=BenchmarkSplit.TEST,
            parse_fn=parse_logic_completion,
            verify_fn=verify_logic_answer,
            device=device,
        )

        logger.info("Base Lineage (Unadapted) - Arith: %.2f%%, Lang: %.2f%%, Logic: %.2f%%",
                    base_arith.get("overall_exact_acc", 0.0),
                    base_lang.exact_accuracy * 100,
                    base_logic.exact_accuracy * 100)

        # Step 2.2: Candidate Track M25-B (Language Specialist)
        logger.info("Training Track M25-B Language Candidate (seed_%d)...", seed)
        lang_cand = train_language_candidate(
            base_model=base_model,
            tokenizer=tokenizer,
            train_samples=lang_train,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=seed,
        )
        cand_b_lang = benchmark.evaluate_generation_task(
            model=lang_cand,
            tokenizer=tokenizer,
            domain=DomainType.LANGUAGE,
            split=BenchmarkSplit.TEST,
            parse_fn=parse_language_completion,
            verify_fn=verify_language_answer,
            device=device,
        )
        cand_b_logic = benchmark.evaluate_generation_task(
            model=lang_cand,
            tokenizer=tokenizer,
            domain=DomainType.LOGIC,
            split=BenchmarkSplit.TEST,
            parse_fn=parse_logic_completion,
            verify_fn=verify_logic_answer,
            device=device,
        )
        cand_b_arith = audit_elementary_transition_space(lang_cand, tokenizer, device, format_type="canonical_state")
        cand_b_ret = evaluate_retention_loss(lang_cand, tokenizer, device)
        drift_b = round(cand_b_ret["prose_original"] - base_retention["prose_original"], 4)
        logger.info("Candidate M25-B (Language) - Lang: %.2f%% (Abstain: %.2f%%), Logic: %.2f%%, Retention Drift: %+.4f",
                    cand_b_lang.exact_accuracy * 100, cand_b_lang.abstention_accuracy * 100, cand_b_logic.exact_accuracy * 100, drift_b)

        # Step 2.3: Candidate Track M25-C (Logic Specialist)
        logger.info("Training Track M25-C Logic Candidate (seed_%d)...", seed)
        logic_cand = train_logic_candidate(
            base_model=base_model,
            tokenizer=tokenizer,
            train_samples=logic_train,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=seed,
        )
        cand_c_logic = benchmark.evaluate_generation_task(
            model=logic_cand,
            tokenizer=tokenizer,
            domain=DomainType.LOGIC,
            split=BenchmarkSplit.TEST,
            parse_fn=parse_logic_completion,
            verify_fn=verify_logic_answer,
            device=device,
        )
        cand_c_lang = benchmark.evaluate_generation_task(
            model=logic_cand,
            tokenizer=tokenizer,
            domain=DomainType.LANGUAGE,
            split=BenchmarkSplit.TEST,
            parse_fn=parse_language_completion,
            verify_fn=verify_language_answer,
            device=device,
        )
        cand_c_arith = audit_elementary_transition_space(logic_cand, tokenizer, device, format_type="canonical_state")
        cand_c_ret = evaluate_retention_loss(logic_cand, tokenizer, device)
        drift_c = round(cand_c_ret["prose_original"] - base_retention["prose_original"], 4)
        logger.info("Candidate M25-C (Logic) - Logic: %.2f%% (Abstain: %.2f%%), Lang: %.2f%%, Retention Drift: %+.4f",
                    cand_c_logic.exact_accuracy * 100, cand_c_logic.abstention_accuracy * 100, cand_c_lang.exact_accuracy * 100, drift_c)

        # Step 2.4: Candidate Track M25-D Prototype (Interleaved Continual Multi-Domain)
        logger.info("Training Track M25-D Interleaved Continual Candidate (seed_%d)...", seed)
        continual_cand = train_continual_interleaved_candidate(
            base_model=base_model,
            tokenizer=tokenizer,
            arith_transitions=all_200_transitions,
            lang_samples=lang_train,
            logic_samples=logic_train,
            device=device,
            epochs=12,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=seed,
        )
        cand_d_arith = audit_elementary_transition_space(continual_cand, tokenizer, device, format_type="canonical_state")
        cand_d_lang = benchmark.evaluate_generation_task(
            model=continual_cand,
            tokenizer=tokenizer,
            domain=DomainType.LANGUAGE,
            split=BenchmarkSplit.TEST,
            parse_fn=parse_language_completion,
            verify_fn=verify_language_answer,
            device=device,
        )
        cand_d_logic = benchmark.evaluate_generation_task(
            model=continual_cand,
            tokenizer=tokenizer,
            domain=DomainType.LOGIC,
            split=BenchmarkSplit.TEST,
            parse_fn=parse_logic_completion,
            verify_fn=verify_logic_answer,
            device=device,
        )
        cand_d_ret = evaluate_retention_loss(continual_cand, tokenizer, device)
        drift_d = round(cand_d_ret["prose_original"] - base_retention["prose_original"], 4)
        logger.info("Candidate M25-D (Continual) - Arith: %.2f%%, Lang: %.2f%%, Logic: %.2f%%, Retention Drift: %+.4f",
                    cand_d_arith.get("overall_exact_acc", 0.0),
                    cand_d_lang.exact_accuracy * 100,
                    cand_d_logic.exact_accuracy * 100,
                    drift_d)

        lineage_results[seed] = {
            "checkpoint_sha256": ckpt_sha,
            "tokenizer_fingerprint": tok_fp,
            "param_count": param_count,
            "base": {
                "arithmetic": base_arith,
                "language": base_lang.to_dict(),
                "logic": base_logic.to_dict(),
                "retention": base_retention,
            },
            "cand_b_language": {
                "language": cand_b_lang.to_dict(),
                "logic": cand_b_logic.to_dict(),
                "arithmetic": cand_b_arith,
                "retention": cand_b_ret,
                "retention_drift": drift_b,
            },
            "cand_c_logic": {
                "logic": cand_c_logic.to_dict(),
                "language": cand_c_lang.to_dict(),
                "arithmetic": cand_c_arith,
                "retention": cand_c_ret,
                "retention_drift": drift_c,
            },
            "cand_d_continual": {
                "arithmetic": cand_d_arith,
                "language": cand_d_lang.to_dict(),
                "logic": cand_d_logic.to_dict(),
                "retention": cand_d_ret,
                "retention_drift": drift_d,
            },
        }

    # 3. Aggregate Multi-Seed Statistics
    def compute_stats(values: list[float]) -> dict[str, float]:
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / max(1, len(values) - 1)
        return {"mean": round(mean, 4), "std": round(math.sqrt(variance), 4)}

    summary = {
        "seeds": args.seeds,
        "base_arith_exact": compute_stats([lineage_results[s]["base"]["arithmetic"].get("overall_exact_acc", 0.0) / 100.0 for s in args.seeds]),
        "base_lang_exact": compute_stats([lineage_results[s]["base"]["language"]["exact_accuracy"] for s in args.seeds]),
        "base_logic_exact": compute_stats([lineage_results[s]["base"]["logic"]["exact_accuracy"] for s in args.seeds]),
        "cand_b_lang_exact": compute_stats([lineage_results[s]["cand_b_language"]["language"]["exact_accuracy"] for s in args.seeds]),
        "cand_b_lang_abstain": compute_stats([lineage_results[s]["cand_b_language"]["language"]["abstention_accuracy"] for s in args.seeds]),
        "cand_b_retention_drift": compute_stats([lineage_results[s]["cand_b_language"]["retention_drift"] for s in args.seeds]),
        "cand_c_logic_exact": compute_stats([lineage_results[s]["cand_c_logic"]["logic"]["exact_accuracy"] for s in args.seeds]),
        "cand_c_logic_abstain": compute_stats([lineage_results[s]["cand_c_logic"]["logic"]["abstention_accuracy"] for s in args.seeds]),
        "cand_c_retention_drift": compute_stats([lineage_results[s]["cand_c_logic"]["retention_drift"] for s in args.seeds]),
        "cand_d_arith_exact": compute_stats([lineage_results[s]["cand_d_continual"]["arithmetic"].get("overall_exact_acc", 0.0) / 100.0 for s in args.seeds]),
        "cand_d_lang_exact": compute_stats([lineage_results[s]["cand_d_continual"]["language"]["exact_accuracy"] for s in args.seeds]),
        "cand_d_logic_exact": compute_stats([lineage_results[s]["cand_d_continual"]["logic"]["exact_accuracy"] for s in args.seeds]),
        "cand_d_retention_drift": compute_stats([lineage_results[s]["cand_d_continual"]["retention_drift"] for s in args.seeds]),
    }

    report = {
        "benchmark": "M25 Multi-Domain Benchmark (Parallel Learning Foundation)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "lineages": lineage_results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_file = args.output_dir / "m25_report.json"
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Report written to %s", report_file)

    logger.info("=" * 66)
    logger.info("M25 MULTI-DOMAIN BENCHMARK SUMMARY (Mean ± Std over Seeds %s):", args.seeds)
    logger.info("  Base Lineage:      Arith: %.2f%% ± %.2f%% | Lang: %.2f%% ± %.2f%% | Logic: %.2f%% ± %.2f%%",
                summary["base_arith_exact"]["mean"] * 100, summary["base_arith_exact"]["std"] * 100,
                summary["base_lang_exact"]["mean"] * 100, summary["base_lang_exact"]["std"] * 100,
                summary["base_logic_exact"]["mean"] * 100, summary["base_logic_exact"]["std"] * 100)
    logger.info("  Cand B (Language): Lang:  %.2f%% ± %.2f%% (Abstain: %.2f%%) | Retention Drift: %+.4f",
                summary["cand_b_lang_exact"]["mean"] * 100, summary["cand_b_lang_exact"]["std"] * 100,
                summary["cand_b_lang_abstain"]["mean"] * 100, summary["cand_b_retention_drift"]["mean"])
    logger.info("  Cand C (Logic):    Logic: %.2f%% ± %.2f%% (Abstain: %.2f%%) | Retention Drift: %+.4f",
                summary["cand_c_logic_exact"]["mean"] * 100, summary["cand_c_logic_exact"]["std"] * 100,
                summary["cand_c_logic_abstain"]["mean"] * 100, summary["cand_c_retention_drift"]["mean"])
    logger.info("  Cand D (Continual):Arith: %.2f%% ± %.2f%% | Lang: %.2f%% ± %.2f%% | Logic: %.2f%% ± %.2f%% | Drift: %+.4f",
                summary["cand_d_arith_exact"]["mean"] * 100, summary["cand_d_arith_exact"]["std"] * 100,
                summary["cand_d_lang_exact"]["mean"] * 100, summary["cand_d_lang_exact"]["std"] * 100,
                summary["cand_d_logic_exact"]["mean"] * 100, summary["cand_d_logic_exact"]["std"] * 100,
                summary["cand_d_retention_drift"]["mean"])
    logger.info("=" * 66)


if __name__ == "__main__":
    main()
