"""M28 Language Foundation Benchmark: Pretraining, Tokenizer Scaling, and Multi-Domain Retention.

Implements the experimental protocol for Adaptive Mini-GPT M28:
  1. Phase 1: Freeze and audit M27 baseline and language failure path.
  2. Phase 2: Curated diverse language corpus with verified 0.0% split disjointness.
  3. Phase 3: Tokenizer scaling (400-BPE vs 1024-BPE) and causal language pretraining.
  4. Phase 4: Grounded reading comprehension, instruction following, and calibrated safe abstention.
  5. Phase 5: Continual multi-domain learning (arithmetic, logic, language, and audited retention).
  6. Phase 6: Multi-lineage qualification across seeds 42, 43, 44 on Tesla T4 GPU.
  7. Phase 7: Autonomous curriculum proposal pilot.
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
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from adaptive_mini_gpt.autonomous.curriculum_selector import (
    CurriculumProposal,
    CurriculumSelector,
    CurriculumStrategy,
)
from adaptive_mini_gpt.data.bpe_scaling import (
    compare_tokenizers,
    expand_model_vocabulary,
    train_scaled_bpe_tokenizer,
)
from adaptive_mini_gpt.data.language_corpus import (
    ABSTENTION_TARGET,
    PRETRAIN_PROSE_CORPUS,
    SEALED_TEST_QA_DOCUMENTS,
    TRAIN_QA_DOCUMENTS,
    VAL_DOCUMENTS,
    compute_corpus_fingerprint,
    format_qa_completion,
    format_qa_prompt,
    get_corpus_statistics,
    normalize_text,
    parse_language_completion,
    verify_all_language_splits_disjoint,
    verify_format_compliance,
)
from adaptive_mini_gpt.data.language_data import (
    BenchmarkSample,
    BenchmarkSplit,
    DomainType,
    LanguageComprehensionDataset,
    build_language_benchmark_samples,
    collate_language_batch,
    verify_language_exact,
    verify_language_grounded,
)
from adaptive_mini_gpt.data.logic_data import (
    LogicReasoningDataset,
    build_logic_benchmark_samples,
    parse_logic_completion,
    verify_logic_answer,
)
from adaptive_mini_gpt.data.pretrain_dataset import (
    PretrainLanguageDataset,
    collate_pretrain_batch,
    evaluate_language_modeling,
)
from adaptive_mini_gpt.data.retention_data import (
    EXPANDED_RETENTION_CORPUS,
    INDEPENDENT_REPLAY_CORPUS,
    evaluate_audited_retention,
    verify_corpora_disjointness,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.evaluation.multi_domain import (
    DomainResult,
    MultiDomainBenchmark,
    MultiDomainReport,
)
from adaptive_mini_gpt.model.capacity_migration import (
    count_parameters,
    create_1m_control_config,
    create_3m_scaled_config,
    estimate_training_flops_per_token,
)
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.reliable_transition import (
    ElementaryTransition,
    ElementaryTransitionDataset,
    audit_elementary_transition_space,
    collate_elementary_batch,
    get_all_200_transitions,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("m28_benchmark")


def train_single_step(
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> float:
    """Execute forward and backward step with prompt loss masking."""
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, _ = model(x)
    loss = nn.functional.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, model.config.vocab_size),
        y[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    loss.backward()
    optimizer.step()
    return loss.item()


def train_pretrain_step(
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    pad_id: int,
    device: torch.device,
) -> float:
    """Execute autoregressive causal next-token pretraining step."""
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, _ = model(x)
    loss = nn.functional.cross_entropy(
        logits.contiguous().view(-1, model.config.vocab_size),
        y.contiguous().view(-1),
        ignore_index=pad_id,
    )
    loss.backward()
    optimizer.step()
    return loss.item()


def train_replay_step(
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    tokenizer: BPETokenizer,
    replay_texts: list[str],
    device: torch.device,
) -> float:
    """Execute continual replay step using strictly disjoint replay texts."""
    text = random.choice(replay_texts)
    toks = tokenizer.encode(text)[: model.config.context_length]
    if len(toks) < 2:
        return 0.0
    x_rep = torch.tensor([toks[:-1]], dtype=torch.long, device=device)
    y_rep = torch.tensor([toks[1:]], dtype=torch.long, device=device)
    optimizer.zero_grad()
    logits, _ = model(x_rep)
    loss = nn.functional.cross_entropy(
        logits.contiguous().view(-1, model.config.vocab_size),
        y_rep.contiguous().view(-1),
    )
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_reading_comprehension(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    samples: list[BenchmarkSample],
    device: torch.device,
    max_new_tokens: int = 24,
) -> dict[str, float]:
    """Evaluate reading comprehension on benchmark samples."""
    model.eval()
    correct_exact = 0
    correct_grounded = 0
    compliant_count = 0
    unans_total = 0
    unans_abstained = 0
    unans_unsupported = 0
    latencies: list[float] = []

    start_ev = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and "cuda" in str(device) else None
    end_ev = torch.cuda.Event(enable_timing=True) if start_ev is not None else None

    with torch.no_grad():
        for s in samples:
            prompt_ids = tokenizer.encode(s.prompt)
            if len(prompt_ids) >= model.config.context_length:
                prompt_ids = prompt_ids[-(model.config.context_length - max_new_tokens - 1) :]

            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            if start_ev is not None:
                start_ev.record()

            # Autoregressive greedy generation
            for _ in range(max_new_tokens):
                if input_tensor.size(1) >= model.config.context_length:
                    break
                logits, _ = model(input_tensor)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_tensor = torch.cat([input_tensor, next_tok], dim=1)
                if next_tok.item() == tokenizer.eos_id:
                    break

            if end_ev is not None:
                end_ev.record()
                torch.cuda.synchronize()
                latencies.append(start_ev.elapsed_time(end_ev))

            generated_toks = input_tensor[0, len(prompt_ids) :].tolist()
            completion = tokenizer.decode(generated_toks).strip()

            parsed_ans, is_comp, is_abst = parse_language_completion(completion)
            if is_comp:
                compliant_count += 1

            if s.is_unanswerable:
                unans_total += 1
                if is_abst:
                    unans_abstained += 1
                else:
                    unans_unsupported += 1

            if verify_language_exact(parsed_ans, s):
                correct_exact += 1
            if verify_language_grounded(parsed_ans, s):
                correct_grounded += 1

    total = len(samples)
    exact_acc = round(correct_exact / total, 4) if total > 0 else 0.0
    grounded_acc = round(correct_grounded / total, 4) if total > 0 else 0.0
    comp_rate = round(compliant_count / total, 4) if total > 0 else 0.0
    abst_acc = round(unans_abstained / unans_total, 4) if unans_total > 0 else 1.0
    unsupp_rate = round(unans_unsupported / unans_total, 4) if unans_total > 0 else 0.0
    mean_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    return {
        "exact_accuracy": exact_acc,
        "grounded_accuracy": grounded_acc,
        "format_compliance": comp_rate,
        "abstention_accuracy": abst_acc,
        "unsupported_answer_rate": unsupp_rate,
        "mean_latency_ms": mean_lat,
    }


def evaluate_all_capabilities(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    val_samples: list[BenchmarkSample],
    test_samples: list[BenchmarkSample],
    logic_test_samples: list[BenchmarkSample],
    benchmark: MultiDomainBenchmark,
    device: torch.device,
) -> dict[str, Any]:
    """Comprehensive evaluation across reading comprehension, arithmetic, logic, and retention."""
    model.eval()

    # 1. Arithmetic evaluation (200 elementary transitions)
    arith_exact = 0.0
    arith_carry = 0.0
    try:
        audit_res = audit_elementary_transition_space(
            model=model,
            tokenizer=tokenizer,
            device=device,
            format_type="canonical_state",
        )
        arith_exact = round(audit_res.get("overall_exact_acc", 0.0) / 100.0, 4)
        arith_carry = round(audit_res.get("overall_carry_acc", 0.0) / 100.0, 4)
    except Exception as e:
        logger.warning(f"Arithmetic evaluation skipped or encountered error: {e}")

    # 2. Reading comprehension on Validation Split
    val_metrics = evaluate_reading_comprehension(
        model=model,
        tokenizer=tokenizer,
        samples=val_samples,
        device=device,
    )

    # 3. Reading comprehension on Sealed Test Split
    test_metrics = evaluate_reading_comprehension(
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
        device=device,
    )

    # 4. Symbolic Logic evaluation
    logic_exact = 0.0
    try:
        logic_res = benchmark.evaluate_generation_task(
            model=model,
            tokenizer=tokenizer,
            domain=DomainType.LOGIC,
            split=BenchmarkSplit.TEST,
            parse_fn=parse_logic_completion,
            verify_fn=verify_logic_answer,
            device=device,
            max_new_tokens=16,
        )
        logic_exact = round(logic_res.exact_accuracy, 4)
    except Exception as e:
        logger.warning(f"Logic evaluation skipped or encountered error: {e}")

    # 5. Audited Retention evaluation
    ret_mean = 0.0
    ret_prose = 0.0
    try:
        ret_losses = evaluate_audited_retention(model, tokenizer, device)
        ret_mean = round(ret_losses.get("mean_loss", 0.0), 4)
        ret_prose = round(ret_losses.get("prose_original", 0.0), 4)
    except Exception as e:
        logger.warning(f"Retention evaluation encountered error: {e}")

    return {
        "arithmetic_exact": arith_exact,
        "arithmetic_carry": arith_carry,
        "logic_exact": logic_exact,
        "retention_loss_mean": ret_mean,
        "retention_loss_prose": ret_prose,
        "val_exact": val_metrics["exact_accuracy"],
        "val_grounded": val_metrics["grounded_accuracy"],
        "val_abstention": val_metrics["abstention_accuracy"],
        "val_unsupported": val_metrics["unsupported_answer_rate"],
        "val_compliance": val_metrics["format_compliance"],
        "test_exact": test_metrics["exact_accuracy"],
        "test_grounded": test_metrics["grounded_accuracy"],
        "test_abstention": test_metrics["abstention_accuracy"],
        "test_unsupported": test_metrics["unsupported_answer_rate"],
        "test_compliance": test_metrics["format_compliance"],
        "test_latency_ms": test_metrics["mean_latency_ms"],
    }


def execute_arm_experiment(
    arm_name: str,
    base_model_1m: MiniGPT,
    tokenizer_400: BPETokenizer,
    tokenizer_1024: BPETokenizer,
    arith_transitions: list[ElementaryTransition],
    train_lang_samples: list[BenchmarkSample],
    val_lang_samples: list[BenchmarkSample],
    test_lang_samples: list[BenchmarkSample],
    logic_train_samples: list[BenchmarkSample],
    logic_test_samples: list[BenchmarkSample],
    benchmark: MultiDomainBenchmark,
    pretrain_epochs: int,
    finetune_epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Run one experimental arm under matched compute protocol."""
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    steps_executed = 0
    pretrain_loss = 0.0
    pretrain_ppl = 1.0

    # Model and Tokenizer setup per Arm
    if arm_name == "arm1_m27_control":
        # 400-BPE, 1M dense backbone, NO pretraining, direct QA tuning
        model = copy.deepcopy(base_model_1m).to(device)
        tokenizer = tokenizer_400
        do_pretrain = False
        multi_domain = False

    elif arm_name == "arm2_pretrain_400":
        # 400-BPE, 1M dense backbone, WITH pretraining, direct QA tuning
        model = copy.deepcopy(base_model_1m).to(device)
        tokenizer = tokenizer_400
        do_pretrain = True
        multi_domain = False

    elif arm_name == "arm3_pretrain_1024_1m":
        # 1024-BPE, 1M dense backbone (~1.05M params), WITH pretraining, direct QA tuning
        cfg_1m_scaled = create_1m_control_config(
            vocab_size=tokenizer_1024.vocab_size,
            context_length=base_model_1m.config.context_length,
            pos_encoding=base_model_1m.config.pos_encoding,
        )
        model = MiniGPT(cfg_1m_scaled).to(device)
        tokenizer = tokenizer_1024
        do_pretrain = True
        multi_domain = False

    elif arm_name == "arm4_pretrain_1024_3m_multidomain":
        # 1024-BPE, 3M dense backbone (~3.59M params), WITH pretraining + multi-domain replay
        cfg_3m = create_3m_scaled_config(
            vocab_size=tokenizer_1024.vocab_size,
            context_length=base_model_1m.config.context_length,
            pos_encoding=base_model_1m.config.pos_encoding,
        )
        model = MiniGPT(cfg_3m).to(device)
        tokenizer = tokenizer_1024
        do_pretrain = True
        multi_domain = True
    else:
        raise ValueError(f"Unknown arm name: {arm_name}")

    param_count = count_parameters(model)
    flops_per_tok = estimate_training_flops_per_token(param_count)

    # -------------------------------------------------------------
    # Stage 1: Autoregressive Language Pretraining (Arms 2, 3, 4)
    # -------------------------------------------------------------
    if do_pretrain:
        pad_id = getattr(tokenizer, "pad_id", 0)
        ds_pretrain = PretrainLanguageDataset(
            texts=PRETRAIN_PROSE_CORPUS,
            tokenizer=tokenizer,
            context_length=model.config.context_length,
        )
        loader_pretrain = DataLoader(
            ds_pretrain,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_pretrain_batch(b, pad_token_id=pad_id),
        )
        opt_pretrain = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        model.train()
        for ep in range(pretrain_epochs):
            for x_p, y_p in loader_pretrain:
                loss_val = train_pretrain_step(model, opt_pretrain, x_p, y_p, pad_id, device)
                steps_executed += 1

        # Evaluate language modeling perplexity on held-out prose
        val_prose = [d.text for d in VAL_DOCUMENTS]
        lm_res = evaluate_language_modeling(model, tokenizer, val_prose, device, batch_size=batch_size)
        pretrain_loss = lm_res["loss"]
        pretrain_ppl = lm_res["perplexity"]

    # -------------------------------------------------------------
    # Stage 2: Supervised Instruction & Reading Comprehension Tuning
    # -------------------------------------------------------------
    pad_id = getattr(tokenizer, "pad_id", 0)
    opt_finetune = torch.optim.AdamW(model.parameters(), lr=lr * 0.75, weight_decay=0.01)

    ds_lang = LanguageComprehensionDataset(
        samples=train_lang_samples,
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

    if multi_domain:
        # Multi-domain balanced dataloaders for Arm 4
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

        ds_logic = LogicReasoningDataset(
            samples=logic_train_samples,
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

        replay_texts = INDEPENDENT_REPLAY_CORPUS

        # Balanced interleaved multi-domain replay
        model.train()
        for ep in range(finetune_epochs):
            iter_a = iter(loader_arith)
            iter_b = iter(loader_lang)
            iter_c = iter(loader_logic)
            max_len = max(len(loader_arith), len(loader_lang), len(loader_logic))

            for _ in range(max_len):
                # Arithmetic
                try:
                    x_a, y_a = next(iter_a)
                except StopIteration:
                    iter_a = iter(loader_arith)
                    x_a, y_a = next(iter_a)
                train_single_step(model, opt_finetune, x_a, y_a, device)
                steps_executed += 1

                # Language QA
                try:
                    x_b, y_b = next(iter_b)
                except StopIteration:
                    iter_b = iter(loader_lang)
                    x_b, y_b = next(iter_b)
                train_single_step(model, opt_finetune, x_b, y_b, device)
                steps_executed += 1

                # Logic
                try:
                    x_c, y_c = next(iter_c)
                except StopIteration:
                    iter_c = iter(loader_logic)
                    x_c, y_c = next(iter_c)
                train_single_step(model, opt_finetune, x_c, y_c, device)
                steps_executed += 1

                # Replay
                if random.random() < 0.10:
                    train_replay_step(model, opt_finetune, tokenizer, replay_texts, device)
                    steps_executed += 1

    else:
        # Standard Language-only fine-tuning (Arms 1, 2, 3)
        model.train()
        for ep in range(finetune_epochs):
            for x_l, y_l in loader_lang:
                train_single_step(model, opt_finetune, x_l, y_l, device)
                steps_executed += 1

    total_time = round(time.time() - start_time, 2)
    step_latency_ms = round((total_time / max(1, steps_executed)) * 1000.0, 2)
    peak_mem_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0

    # -------------------------------------------------------------
    # Stage 3: Full Multi-Domain & Reading Comprehension Evaluation
    # -------------------------------------------------------------
    eval_metrics = evaluate_all_capabilities(
        model=model,
        tokenizer=tokenizer,
        val_samples=val_lang_samples,
        test_samples=test_lang_samples,
        logic_test_samples=logic_test_samples,
        benchmark=benchmark,
        device=device,
    )

    eval_metrics["pretrain_loss"] = pretrain_loss
    eval_metrics["pretrain_ppl"] = pretrain_ppl

    return {
        "arm": arm_name,
        "seed": seed,
        "param_count": param_count,
        "flops_per_token": flops_per_tok,
        "steps_executed": steps_executed,
        "step_latency_ms": step_latency_ms,
        "peak_memory_mb": peak_mem_mb,
        "duration_seconds": total_time,
        "metrics": eval_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M28 Language Foundation Benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoints-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/m28_language_foundation")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting M28 Language Foundation Benchmark on {device}")
    logger.info(f"Seeds: {args.seeds}, Pretrain Epochs: {args.pretrain_epochs}, Finetune Epochs: {args.finetune_epochs}")

    # =====================================================================
    # Phase 1 & 2: Audit and Corpus Disjointness Verification
    # =====================================================================
    logger.info("Phase 1 & 2: Auditing corpus disjointness and statistics...")
    corpus_stats = get_corpus_statistics()
    disjoint_audit = verify_all_language_splits_disjoint()
    replay_disjoint_audit = verify_corpora_disjointness()

    logger.info(f"Corpus statistics: {corpus_stats}")
    logger.info(f"Language split audit: {disjoint_audit}")
    logger.info(f"Replay split audit: {replay_disjoint_audit}")

    # =====================================================================
    # Phase 3: Train and Compare 1024-BPE Tokenizer
    # =====================================================================
    logger.info("Phase 3: Training scaled 1024-vocab BPE tokenizer...")
    all_prose = " ".join(PRETRAIN_PROSE_CORPUS)
    tokenizer_1024 = train_scaled_bpe_tokenizer(all_prose, target_vocab_size=1024)
    tok_1024_path = output_path / "bpe_1024.json"
    tokenizer_1024.save(tok_1024_path)
    logger.info(f"Saved 1024-BPE tokenizer to {tok_1024_path} with fingerprint {tokenizer_1024.get_fingerprint()}")

    # =====================================================================
    # Benchmark Setup & Sample Preparation
    # =====================================================================
    benchmark = MultiDomainBenchmark()
    arith_transitions = get_all_200_transitions()
    logic_train_samples = build_logic_benchmark_samples(split=BenchmarkSplit.TRAIN, seed=42)
    logic_test_samples = build_logic_benchmark_samples(split=BenchmarkSplit.TEST, seed=42)
    benchmark.register_samples(DomainType.LOGIC, BenchmarkSplit.TEST, logic_test_samples)

    train_lang_samples = build_language_benchmark_samples(BenchmarkSplit.TRAIN, include_paraphrases=True, seed=42)
    val_lang_samples = build_language_benchmark_samples(BenchmarkSplit.VAL, include_paraphrases=True, seed=42)
    test_lang_samples = build_language_benchmark_samples(BenchmarkSplit.TEST, include_paraphrases=True, seed=42)

    logger.info(f"Registered {len(train_lang_samples)} train QA, {len(val_lang_samples)} val QA, {len(test_lang_samples)} test QA samples.")

    # =====================================================================
    # Multi-Lineage Experimental Execution across Seeds
    # =====================================================================
    arms = [
        "arm1_m27_control",
        "arm2_pretrain_400",
        "arm3_pretrain_1024_1m",
        "arm4_pretrain_1024_3m_multidomain",
    ]

    all_arm_results: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    tokenizer_comparison: dict[str, Any] = {}

    for seed in args.seeds:
        logger.info(f"\n=======================================================")
        logger.info(f"Evaluating Neural Lineage Seed {seed}")
        logger.info(f"=======================================================")

        # Load root baseline model from M18 checkpoint
        if args.checkpoints_dir:
            ckpt_file = Path(args.checkpoints_dir) / f"seed_{seed}/base_pretrain/best.pt"
            raw_ckpt = CheckpointManager.load_raw_checkpoint(ckpt_file)
            base_model_1m = MiniGPT(ModelConfig(**raw_ckpt["model_config"]))
            base_model_1m.load_state_dict(raw_ckpt["model_state"])
            tokenizer_400 = CheckpointManager.load_tokenizer(raw_ckpt)
        else:
            cfg_1m = create_1m_control_config(vocab_size=400)
            base_model_1m = MiniGPT(cfg_1m)
            tokenizer_400 = BPETokenizer()
            tokenizer_400.train(all_prose, target_vocab_size=400)

        # Quantitative tokenizer comparison on first seed
        if not tokenizer_comparison:
            tokenizer_comparison = compare_tokenizers(
                tok_baseline=tokenizer_400,
                tok_scaled=tokenizer_1024,
                sample_texts=PRETRAIN_PROSE_CORPUS[:15],
            )
            logger.info(f"Tokenizer comparison: {tokenizer_comparison}")

        # Run each experimental arm
        for arm_name in arms:
            logger.info(f"\n--- Running {arm_name} on Seed {seed} ---")
            arm_record = execute_arm_experiment(
                arm_name=arm_name,
                base_model_1m=base_model_1m,
                tokenizer_400=tokenizer_400,
                tokenizer_1024=tokenizer_1024,
                arith_transitions=arith_transitions,
                train_lang_samples=train_lang_samples,
                val_lang_samples=val_lang_samples,
                test_lang_samples=test_lang_samples,
                logic_train_samples=logic_train_samples,
                logic_test_samples=logic_test_samples,
                benchmark=benchmark,
                pretrain_epochs=args.pretrain_epochs,
                finetune_epochs=args.finetune_epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                device=device,
                seed=seed,
            )
            all_arm_results[arm_name].append(arm_record)
            m = arm_record["metrics"]
            logger.info(
                f"[{arm_name} Seed {seed}] Test Exact: {m['test_exact']*100:.1f}%, "
                f"Test Grounded: {m['test_grounded']*100:.1f}%, Abstention: {m['test_abstention']*100:.1f}%, "
                f"Arith Carry: {m['arithmetic_carry']*100:.1f}%, Logic: {m['logic_exact']*100:.1f}%, "
                f"PPL: {m.get('pretrain_ppl', 1.0):.2f}"
            )

    # =====================================================================
    # Aggregate Multi-Lineage Statistics
    # =====================================================================
    aggregated_arms: dict[str, dict[str, Any]] = {}

    for arm_name, runs in all_arm_results.items():
        n = len(runs)
        keys = list(runs[0]["metrics"].keys())
        mean_metrics = {k: round(sum(r["metrics"][k] for r in runs) / n, 4) for k in keys}
        std_metrics = {k: round(math.sqrt(sum((r["metrics"][k] - mean_metrics[k]) ** 2 for r in runs) / n), 4) for k in keys}
        mean_duration = round(sum(r["duration_seconds"] for r in runs) / n, 2)
        mean_latency = round(sum(r["step_latency_ms"] for r in runs) / n, 2)

        aggregated_arms[arm_name] = {
            "param_count": runs[0]["param_count"],
            "flops_per_token": runs[0]["flops_per_token"],
            "mean_duration_seconds": mean_duration,
            "mean_step_latency_ms": mean_latency,
            "peak_memory_mb": max(r["peak_memory_mb"] for r in runs),
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
            "raw_runs": runs,
        }

    # =====================================================================
    # Phase 7: Autonomous Curriculum Proposal Pilot
    # =====================================================================
    logger.info("\nPhase 7: Running Autonomous Curriculum Selector Proposal...")
    selector = CurriculumSelector(strategy=CurriculumStrategy.GAP_DIRECTED)
    arm4_means = aggregated_arms["arm4_pretrain_1024_3m_multidomain"]["mean_metrics"]
    initial_accs = {
        "arithmetic": arm4_means["arithmetic_carry"],
        "logic": arm4_means["logic_exact"],
        "language": arm4_means["test_exact"],
    }
    autonomous_proposal = selector.propose_curriculum(
        domain_accuracies=initial_accs,
        step_budget=200,
        cycle_index=1,
    )
    logger.info(f"Autonomous proposal: {autonomous_proposal}")

    # =====================================================================
    # Final Report Assembly
    # =====================================================================
    report: dict[str, Any] = {
        "milestone": "M28",
        "title": "Language Foundation for a Continually Self-Learning Mini-GPT",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "seeds": args.seeds,
        "corpus_statistics": corpus_stats,
        "disjointness_audit": disjoint_audit,
        "replay_disjointness_audit": replay_disjoint_audit,
        "tokenizer_comparison": tokenizer_comparison,
        "tokenizer_1024_fingerprint": tokenizer_1024.get_fingerprint(),
        "arms": aggregated_arms,
        "autonomous_proposal": asdict(autonomous_proposal),
    }

    report_path = output_path / "m28_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n=======================================================")
    logger.info(f"M28 Benchmark Complete! Report successfully saved to {report_path}")
    logger.info(f"=======================================================")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    main()
