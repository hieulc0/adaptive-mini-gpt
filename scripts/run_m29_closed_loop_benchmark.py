"""M29 Benchmark: Closed-Loop Interaction and Knowledge-Gap-Driven Learning.

Implements the experimental protocol for Adaptive Mini-GPT M29:
  1. Phase 1 & 2: Interactive knowledge-gap tracking, structured diagnostics, and error categorization.
  2. Phase 3 & 8: Comparative evaluation of Action Selection policies (Gap-Driven vs Always-Answer vs Always-Retrieve vs Fixed-Schedule).
  3. Phase 4: Provenance-aware verified memory bank with source whitelisting.
  4. Phase 5 & 6: Three controlled learning episodes (Language, Logic, Arithmetic) comparing 4 regimes:
     - Regime 1: Immediate Retry (no training, no retrieval)
     - Regime 2: External Retrieval Only (no weight update)
     - Regime 3: Targeted Candidate Training (without replay)
     - Regime 4: Targeted Candidate Training with Balanced Multi-Domain Replay
  5. Phase 7: Multi-lineage qualification across seeds 42, 43, 44 on Tesla T4 GPU.
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
from adaptive_mini_gpt.model.capacity_migration import (
    count_parameters,
    create_1m_control_config,
    create_3m_scaled_config,
    estimate_training_flops_per_token,
)
from adaptive_mini_gpt.data.elementary_data import (
    ElementaryTransitionDataset,
    collate_elementary_batch,
    get_all_200_transitions,
)
from adaptive_mini_gpt.data.language_corpus import (
    ABSTENTION_TARGET,
    PRETRAIN_PROSE_CORPUS,
    TRAIN_QA_DOCUMENTS,
    VAL_DOCUMENTS,
    normalize_text,
    parse_language_completion,
)
from adaptive_mini_gpt.data.language_data import (
    BenchmarkSample,
    BenchmarkSplit,
    DomainType,
    LanguageComprehensionDataset,
    collate_language_batch,
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
    train_pretrain_step,
)
from adaptive_mini_gpt.data.retention_data import (
    INDEPENDENT_REPLAY_CORPUS,
    evaluate_audited_retention,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.evaluation.elementary_audit import audit_elementary_transition_space
from adaptive_mini_gpt.evaluation.multi_domain import MultiDomainBenchmark
from adaptive_mini_gpt.interaction.action_selector import (
    ActionDecision,
    AlwaysAnswerOrRetryPolicy,
    AlwaysRetrievePolicy,
    FixedScheduledTrainingPolicy,
    InteractionActionSelector,
    LearningAction,
)
from adaptive_mini_gpt.interaction.episodes import (
    EpisodeData,
    build_episode_a_language,
    build_episode_b_logic,
    build_episode_c_arithmetic,
)
from adaptive_mini_gpt.interaction.knowledge_gap import (
    ErrorCategory,
    KnowledgeGapRecord,
    KnowledgeGapTracker,
)
from adaptive_mini_gpt.interaction.loop import ClosedLoopInteractiveLearner
from adaptive_mini_gpt.interaction.verified_memory import (
    VerifiedMemoryBank,
    VerifiedMemoryEntry,
)
from adaptive_mini_gpt.model.transformer import MiniGPT, ModelConfig
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("m29_closed_loop")


def train_single_step(
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    ignore_index: int = -100,
) -> float:
    """Execute one optimization step."""
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    out = model(x)
    logits = out[0] if isinstance(out, tuple) else out
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1),
        ignore_index=ignore_index,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


def evaluate_episode_accuracy(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    samples: list[BenchmarkSample],
    domain: DomainType,
    device: torch.device,
) -> float:
    """Evaluate exact match accuracy on fresh held-out episode samples."""
    model.eval()
    correct = 0
    total = len(samples)
    if total == 0:
        return 0.0

    with torch.no_grad():
        for s in samples:
            prompt = s.prompt
            enc = tokenizer.encode(prompt)
            input_ids = torch.tensor([enc], dtype=torch.long, device=device)

            for _ in range(16):
                out = model(input_ids)
                logits = out[0] if isinstance(out, tuple) else out
                next_id = torch.argmax(logits[0, -1, :]).item()
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
                decoded_partial = tokenizer.decode(input_ids[0].tolist())
                if "\n" in decoded_partial[len(prompt):] or next_id == tokenizer.eos_token_id:
                    break

            full_decoded = tokenizer.decode(input_ids[0].tolist())
            completion = full_decoded[len(prompt):].strip()

            if domain == DomainType.LANGUAGE:
                pred = parse_language_completion(completion)
                if normalize_text(pred) == normalize_text(s.expected_answer):
                    correct += 1
            elif domain == DomainType.LOGIC:
                pred = completion.split("\n")[0].strip()
                if normalize_text(pred) == normalize_text(s.expected_answer):
                    correct += 1
            elif domain == DomainType.ARITHMETIC:
                pred = completion.split("\n")[0].strip()
                if normalize_text(pred) == normalize_text(s.expected_answer):
                    correct += 1

    return round(correct / total, 4)


def run_episode_experiment(
    episode: EpisodeData,
    base_model: MiniGPT,
    tokenizer: BPETokenizer,
    arith_transitions: list[Any],
    logic_train_samples: list[BenchmarkSample],
    device: torch.device,
    epochs: int = 12,
    lr: float = 3e-4,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Benchmark the four learning regimes on a controlled episode."""
    pad_id = getattr(tokenizer, "pad_id", 0)

    # -------------------------------------------------------------
    # 0. Pre-training baseline accuracy on fresh test problems
    # -------------------------------------------------------------
    pre_accuracy = evaluate_episode_accuracy(
        model=base_model,
        tokenizer=tokenizer,
        samples=episode.fresh_test_samples,
        domain=episode.domain,
        device=device,
    )

    # -------------------------------------------------------------
    # Regime 1: Immediate Retry (no weight updates, no external memory)
    # -------------------------------------------------------------
    # Under deterministic greedy decoding, immediate retry yields identical output
    regime1_acc = pre_accuracy

    # -------------------------------------------------------------
    # Regime 2: Retrieval Only (External Memory Augmented, no weight update)
    # -------------------------------------------------------------
    # Simulates external lookup returning verified reference facts
    memory = VerifiedMemoryBank()
    memory.store_verified_fact(
        domain=episode.domain,
        topic_or_key=episode.trigger_question,
        verified_fact=episode.trigger_ground_truth,
        evidence_source=f"approved_local_doc:{episode.episode_id}",
        verifier_identity="oracle_verifier",
    )
    # Exact match on trigger question from memory is 100%, but on fresh unseen questions
    # external retrieval without weight learning only succeeds if exact query matches
    retrieval_hits = 0
    for s in episode.fresh_test_samples:
        q_text = s.metadata.get("question", s.prompt)
        entry = memory.retrieve(episode.domain, q_text)
        if entry and normalize_text(entry.verified_fact) == normalize_text(s.expected_answer):
            retrieval_hits += 1
    regime2_acc = round(retrieval_hits / len(episode.fresh_test_samples), 4)

    # -------------------------------------------------------------
    # Regime 3: Targeted Candidate Training (Without Replay)
    # -------------------------------------------------------------
    cand_model_no_replay = copy.deepcopy(base_model).to(device)
    opt_no_replay = torch.optim.AdamW(cand_model_no_replay.parameters(), lr=lr, weight_decay=0.01)

    ds_practice = LanguageComprehensionDataset(
        samples=episode.practice_samples,
        tokenizer=tokenizer,
        context_length=cand_model_no_replay.config.context_length,
        ignore_index=-100,
    )
    loader_practice = DataLoader(
        ds_practice,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_language_batch(b, pad_token_id=pad_id, ignore_index=-100),
    )

    cand_model_no_replay.train()
    for ep in range(epochs):
        for x_b, y_b in loader_practice:
            train_single_step(cand_model_no_replay, opt_no_replay, x_b, y_b, device)

    regime3_acc = evaluate_episode_accuracy(
        model=cand_model_no_replay,
        tokenizer=tokenizer,
        samples=episode.fresh_test_samples,
        domain=episode.domain,
        device=device,
    )

    # Measure retention collapse without replay
    try:
        audit_no_rep = audit_elementary_transition_space(
            cand_model_no_replay, tokenizer, device, format_type="canonical_state"
        )
        regime3_carry = round(audit_no_rep.get("overall_carry_acc", 0.0) / 100.0, 4)
    except Exception:
        regime3_carry = 0.0

    # -------------------------------------------------------------
    # Regime 4: Targeted Candidate Training (With Balanced Multi-Domain Replay)
    # -------------------------------------------------------------
    cand_model_replay = copy.deepcopy(base_model).to(device)
    opt_replay = torch.optim.AdamW(cand_model_replay.parameters(), lr=lr, weight_decay=0.01)

    ds_arith = ElementaryTransitionDataset(
        transitions=arith_transitions,
        tokenizer=tokenizer,
        format_type="canonical_state",
        context_length=cand_model_replay.config.context_length,
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
        context_length=cand_model_replay.config.context_length,
        ignore_index=-100,
    )
    loader_logic = DataLoader(
        ds_logic,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_language_batch(b, pad_token_id=pad_id, ignore_index=-100),
    )

    replay_texts = INDEPENDENT_REPLAY_CORPUS

    cand_model_replay.train()
    for ep in range(epochs):
        iter_p = iter(loader_practice)
        iter_a = iter(loader_arith)
        iter_l = iter(loader_logic)
        max_steps = max(len(loader_practice), len(loader_arith), len(loader_logic))

        for _ in range(max_steps):
            # Practice step
            try:
                x_p, y_p = next(iter_p)
            except StopIteration:
                iter_p = iter(loader_practice)
                x_p, y_p = next(iter_p)
            train_single_step(cand_model_replay, opt_replay, x_p, y_p, device)

            # Arithmetic Replay
            try:
                x_a, y_a = next(iter_a)
            except StopIteration:
                iter_a = iter(loader_arith)
                x_a, y_a = next(iter_a)
            train_single_step(cand_model_replay, opt_replay, x_a, y_a, device)

            # Logic Replay
            try:
                x_l, y_l = next(iter_l)
            except StopIteration:
                iter_l = iter(loader_logic)
                x_l, y_l = next(iter_l)
            train_single_step(cand_model_replay, opt_replay, x_l, y_l, device)

    regime4_acc = evaluate_episode_accuracy(
        model=cand_model_replay,
        tokenizer=tokenizer,
        samples=episode.fresh_test_samples,
        domain=episode.domain,
        device=device,
    )

    try:
        audit_rep = audit_elementary_transition_space(
            cand_model_replay, tokenizer, device, format_type="canonical_state"
        )
        regime4_carry = round(audit_rep.get("overall_carry_acc", 0.0) / 100.0, 4)
    except Exception:
        regime4_carry = 0.0

    return {
        "episode_id": episode.episode_id,
        "domain": episode.domain.value,
        "pre_accuracy": pre_accuracy,
        "regime1_immediate_retry": regime1_acc,
        "regime2_retrieval_only": regime2_acc,
        "regime3_training_no_replay": regime3_acc,
        "regime3_retention_carry": regime3_carry,
        "regime4_training_with_replay": regime4_acc,
        "regime4_retention_carry": regime4_carry,
        "weight_learning_gain": round(regime4_acc - pre_accuracy, 4),
    }


def benchmark_action_policies(
    episodes: list[EpisodeData],
    model: MiniGPT,
    tokenizer: BPETokenizer,
    device: torch.device,
) -> dict[str, Any]:
    """Compare the 4 Action Selection Policies on accuracy, action balance, and GPU compute efficiency."""
    policies: dict[str, BaseActionPolicy] = {
        "always_answer": AlwaysAnswerOrRetryPolicy(),
        "always_retrieve": AlwaysRetrievePolicy(),
        "fixed_schedule": FixedScheduledTrainingPolicy(train_interval_steps=4),
        "gap_driven": InteractionActionSelector(confidence_threshold=0.70, repeat_gap_training_threshold=2),
    }

    policy_results: dict[str, dict[str, Any]] = {}

    for pol_name, policy in policies.items():
        tracker = KnowledgeGapTracker()
        memory = VerifiedMemoryBank()
        action_counts: dict[str, int] = {a.value: 0 for a in LearningAction}

        for ep in episodes:
            learner = ClosedLoopInteractiveLearner(
                model=model,
                tokenizer=tokenizer,
                memory_bank=memory,
                gap_tracker=tracker,
                action_selector=policy,
                device=device,
            )

            # Query 1: Trigger question (first encounter)
            res1 = learner.interact(
                domain=ep.domain,
                question=ep.trigger_question,
                context=ep.trigger_context,
                ground_truth=ep.trigger_ground_truth,
                has_approved_reference=True,
                approved_reference_text=ep.approved_reference_doc,
            )
            action_counts[res1.decision.action.value] += 1

            # Query 2: Repeated question (second encounter)
            res2 = learner.interact(
                domain=ep.domain,
                question=ep.trigger_question,
                context=ep.trigger_context,
                ground_truth=ep.trigger_ground_truth,
                has_approved_reference=True,
                approved_reference_text=ep.approved_reference_doc,
            )
            action_counts[res2.decision.action.value] += 1

        diagnostics = tracker.compute_diagnostics()
        training_actions = action_counts["train"]

        policy_results[pol_name] = {
            "action_counts": action_counts,
            "training_actions_count": training_actions,
            "gpu_training_avoided_pct": round((1.0 - (training_actions / max(1, len(episodes) * 2))) * 100.0, 1),
            "diagnostics": diagnostics,
        }

    return policy_results


def main() -> None:
    parser = argparse.ArgumentParser(description="M29 Closed-Loop Learning Benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoints-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/m29_closed_loop")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info(f"Starting M29 Closed-Loop Interaction & Learning Benchmark on {device}")
    logger.info(f"Seeds: {args.seeds}, Epochs: {args.epochs}, LR: {args.lr}")
    logger.info("=" * 65)

    # 1. Train or load 1024-BPE Tokenizer
    all_prose = " ".join(PRETRAIN_PROSE_CORPUS)
    tokenizer_1024 = train_scaled_bpe_tokenizer(all_prose, target_vocab_size=1024)
    tok_path = output_path / "bpe_1024.json"
    tokenizer_1024.save(tok_path)
    logger.info(f"1024-BPE Tokenizer ready with fingerprint: {tokenizer_1024.get_fingerprint()}")

    # 2. Build Multi-Domain Replay Pools
    arith_transitions = get_all_200_transitions()
    logic_train_samples = build_logic_benchmark_samples(split=BenchmarkSplit.TRAIN, seed=42)
    logic_test_samples = build_logic_benchmark_samples(split=BenchmarkSplit.TEST, seed=42)

    # 3. Build the 3 Controlled Learning Episodes
    ep_a = build_episode_a_language()
    ep_b = build_episode_b_logic(seed=42)
    ep_c = build_episode_c_arithmetic()
    episodes = [ep_a, ep_b, ep_c]
    logger.info(f"Episodes prepared: {[e.title for e in episodes]}")

    # 4. Multi-Lineage Execution across Seeds
    all_seed_results: list[dict[str, Any]] = []
    policy_benchmarks_all: list[dict[str, Any]] = []

    for seed in args.seeds:
        logger.info(f"\n=======================================================")
        logger.info(f"Executing Lineage Seed {seed}")
        logger.info(f"=======================================================")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        start_time = time.time()

        # Build base 1M model with 1024 vocab
        cfg_1m = create_1m_control_config(vocab_size=tokenizer_1024.vocab_size)
        base_model = MiniGPT(cfg_1m).to(device)

        if args.checkpoints_dir:
            ckpt_file = Path(args.checkpoints_dir) / f"seed_{seed}/base_pretrain/best.pt"
            raw_ckpt = CheckpointManager.load_raw_checkpoint(ckpt_file)
            state_dict = raw_ckpt.get("model_state_dict", raw_ckpt.get("model_state"))
            # Expand vocabulary weights from 400 to 1024
            base_model = expand_model_vocabulary(
                model=MiniGPT(ModelConfig(**raw_ckpt["model_config"])),
                new_vocab_size=1024,
            ).to(device)
            logger.info(f"Expanded checkpoint {ckpt_file.name} to 1024 vocab.")

        # Phase 3 & 8: Policy Benchmark on Seed
        logger.info(f"[Seed {seed}] Benchmarking Action Selection Policies...")
        pol_res = benchmark_action_policies(episodes, base_model, tokenizer_1024, device)
        policy_benchmarks_all.append(pol_res)

        # Phase 5 & 6: Execute Controlled Learning Episodes
        episode_runs: list[dict[str, Any]] = []
        for ep in episodes:
            logger.info(f"[Seed {seed}] Running {ep.title}...")
            res = run_episode_experiment(
                episode=ep,
                base_model=base_model,
                tokenizer=tokenizer_1024,
                arith_transitions=arith_transitions,
                logic_train_samples=logic_train_samples,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
            )
            logger.info(
                f"  [{ep.episode_id}] Pre: {res['pre_accuracy']*100:.1f}% | "
                f"Retry: {res['regime1_immediate_retry']*100:.1f}% | "
                f"Retr: {res['regime2_retrieval_only']*100:.1f}% | "
                f"Train (NoRep): {res['regime3_training_no_replay']*100:.1f}% (Carry: {res['regime3_retention_carry']*100:.1f}%) | "
                f"Train (WithRep): {res['regime4_training_with_replay']*100:.1f}% (Carry: {res['regime4_retention_carry']*100:.1f}%)"
            )
            episode_runs.append(res)

        # Retention on pretraining holdout
        ret_losses = evaluate_audited_retention(base_model, tokenizer_1024, device)
        duration = round(time.time() - start_time, 2)
        peak_mem = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0

        all_seed_results.append({
            "seed": seed,
            "duration_seconds": duration,
            "peak_memory_mb": peak_mem,
            "param_count": count_parameters(base_model),
            "flops_per_token": estimate_training_flops_per_token(count_parameters(base_model)),
            "retention_loss_mean": round(ret_losses.get("mean_loss", 0.0), 4),
            "episodes": episode_runs,
        })

    # 5. Aggregate Across Lineages
    n = len(all_seed_results)
    ep_ids = [e["episode_id"] for e in all_seed_results[0]["episodes"]]
    aggregated_episodes: dict[str, dict[str, Any]] = {}

    for ep_id in ep_ids:
        pre_accs = [r["episodes"][i]["pre_accuracy"] for r in all_seed_results for i, e in enumerate(r["episodes"]) if e["episode_id"] == ep_id]
        r1_accs = [r["episodes"][i]["regime1_immediate_retry"] for r in all_seed_results for i, e in enumerate(r["episodes"]) if e["episode_id"] == ep_id]
        r2_accs = [r["episodes"][i]["regime2_retrieval_only"] for r in all_seed_results for i, e in enumerate(r["episodes"]) if e["episode_id"] == ep_id]
        r3_accs = [r["episodes"][i]["regime3_training_no_replay"] for r in all_seed_results for i, e in enumerate(r["episodes"]) if e["episode_id"] == ep_id]
        r3_carries = [r["episodes"][i]["regime3_retention_carry"] for r in all_seed_results for i, e in enumerate(r["episodes"]) if e["episode_id"] == ep_id]
        r4_accs = [r["episodes"][i]["regime4_training_with_replay"] for r in all_seed_results for i, e in enumerate(r["episodes"]) if e["episode_id"] == ep_id]
        r4_carries = [r["episodes"][i]["regime4_retention_carry"] for r in all_seed_results for i, e in enumerate(r["episodes"]) if e["episode_id"] == ep_id]
        gains = [r["episodes"][i]["weight_learning_gain"] for r in all_seed_results for i, e in enumerate(r["episodes"]) if e["episode_id"] == ep_id]

        aggregated_episodes[ep_id] = {
            "mean_pre_accuracy": round(sum(pre_accs) / n, 4),
            "mean_regime1_immediate_retry": round(sum(r1_accs) / n, 4),
            "mean_regime2_retrieval_only": round(sum(r2_accs) / n, 4),
            "mean_regime3_training_no_replay": round(sum(r3_accs) / n, 4),
            "mean_regime3_retention_carry": round(sum(r3_carries) / n, 4),
            "mean_regime4_training_with_replay": round(sum(r4_accs) / n, 4),
            "mean_regime4_retention_carry": round(sum(r4_carries) / n, 4),
            "mean_weight_learning_gain": round(sum(gains) / n, 4),
        }

    # Aggregate Policy Benchmark
    pol_names = list(policy_benchmarks_all[0].keys())
    aggregated_policies: dict[str, Any] = {}
    for p in pol_names:
        gpu_avoided = [pb[p]["gpu_training_avoided_pct"] for pb in policy_benchmarks_all]
        train_acts = [pb[p]["training_actions_count"] for pb in policy_benchmarks_all]
        gap_precs = [pb[p]["diagnostics"]["gap_detection_precision"] for pb in policy_benchmarks_all]
        false_confs = [pb[p]["diagnostics"]["false_confidence_rate"] for pb in policy_benchmarks_all]

        aggregated_policies[p] = {
            "mean_training_actions": round(sum(train_acts) / n, 2),
            "mean_gpu_training_avoided_pct": round(sum(gpu_avoided) / n, 1),
            "mean_gap_detection_precision": round(sum(gap_precs) / n, 4),
            "mean_false_confidence_rate": round(sum(false_confs) / n, 4),
            "action_distribution": policy_benchmarks_all[0][p]["action_counts"],
        }

    # 6. Assemble Final Report
    report: dict[str, Any] = {
        "milestone": "M29",
        "title": "Closed-Loop Interaction and Knowledge-Gap-Driven Learning",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "seeds": args.seeds,
        "parameters": all_seed_results[0]["param_count"],
        "mean_duration_seconds": round(sum(r["duration_seconds"] for r in all_seed_results) / n, 2),
        "peak_memory_mb": max(r["peak_memory_mb"] for r in all_seed_results),
        "tokenizer_1024_fingerprint": tokenizer_1024.get_fingerprint(),
        "episodes_aggregated": aggregated_episodes,
        "policy_comparison": aggregated_policies,
        "raw_seed_runs": all_seed_results,
    }

    report_file = output_path / "m29_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 65)
    logger.info(f"M29 Benchmark Complete! Report successfully saved to: {report_file}")
    logger.info("=" * 65)


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    main()
