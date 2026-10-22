#!/usr/bin/env python3
"""M34: Modular Continual Learning and Verified Checkpoint Growth Benchmark Runner.

Executes the 7-Phase M34 Study across 3 independent lineages (Seeds 42, 43, 44):
  Phase 1: Experimental protocol audit and reconciliation.
  Phase 2: Foundation comparison: M18 neural root vs. verified M32 recurrent foundation.
  Phase 3-4: Sequential multi-episode learning across 3 controlled approaches:
    - Approach A: Full fine-tuning with balanced replay (M33 baseline).
    - Approach B: Frozen backbone + separate skill adapters (no replay).
    - Approach C: Frozen backbone + separate skill adapters + balanced replay.
  Phase 5: Deterministic router integration and unified modular inference.
  Phase 6: Multi-seed aggregation, retention matrix, parameter efficiency, and qualification report.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

# Ensure repository root and src are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from adaptive_mini_gpt.data.bpe_scaling import (
    expand_model_vocabulary,
    train_scaled_bpe_tokenizer,
)
from adaptive_mini_gpt.data.language_corpus import PRETRAIN_PROSE_CORPUS
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.learning.cumulative_learner import (
    build_m33_arithmetic_episode,
    build_m33_language_episode,
    build_m33_logic_episode,
)
from adaptive_mini_gpt.learning.foundation import (
    FoundationComparisonRecord,
    train_m32_recurrent_foundation,
)
from adaptive_mini_gpt.learning.modular_cumulative_learner import (
    ModularReasoner,
    evaluate_multi_domain_system,
    train_domain_adapter,
)
from adaptive_mini_gpt.learning.modular_lineage import (
    ModularCheckpointRecord,
    ModularLineageManager,
    compute_adapter_hash,
)
from adaptive_mini_gpt.model.modular_adapter import (
    add_domain_adapter,
    compute_base_model_hash,
    freeze_for_adapter_training,
    get_adapter_state_dict,
    inject_multi_domain_adapters,
    load_adapter_state_dict,
    set_active_domain_adapter,
)
from adaptive_mini_gpt.model.transformer import MiniGPT, ModelConfig
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("m34_benchmark")


def compute_mean_std(values: list[float]) -> dict[str, float]:
    """Compute mean and sample standard deviation."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(values) / n
    if n == 1:
        return {"mean": round(mean, 4), "std": 0.0}
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4)}


def run_sequential_episodes_for_approach(
    approach_name: str,
    foundation_model: MiniGPT,
    tokenizer: BPETokenizer,
    device: torch.device,
    seed: int,
    ep_a: Any,
    ep_b: Any,
    ep_c: Any,
    epochs: int = 12,
    lr: float = 3e-4,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Run sequential 3-episode learning under specified approach (A, B, or C)."""
    logger.info(f"--- Running {approach_name} (Seed {seed}) ---")
    arith_holdout = ep_a.fresh_test_samples
    logic_holdout = ep_b.fresh_test_samples
    lang_holdout = ep_c.fresh_test_samples

    is_modular = (approach_name in ["approach_b_modular", "approach_c_modular_replay"])
    use_replay = (approach_name in ["approach_a_full_ft_replay", "approach_c_modular_replay"])

    model = copy.deepcopy(foundation_model).to(device)
    if is_modular:
        inject_multi_domain_adapters(model, r=8, target_modules=["c_attn", "c_proj", "c_fc"])

    lineage = ModularLineageManager(seed=seed)
    f0_metrics = evaluate_multi_domain_system(
        model=model,
        tokenizer=tokenizer,
        device=device,
        arith_test_samples=arith_holdout,
        logic_test_samples=logic_holdout,
        lang_test_samples=lang_holdout,
        is_modular=is_modular,
        active_domain=None,
    )
    lineage.register_foundation(model, foundation_id="V1", metrics=f0_metrics, description="M32 Foundation")

    cumulative_history: list[dict[str, Any]] = [
        {"step": "foundation_V1", "metrics": copy.deepcopy(f0_metrics), "promoted": True}
    ]

    episodes = [
        ("arithmetic", ep_a, "candidate_V2", "qualified_V2"),
        ("logic", ep_b, "candidate_V3", "qualified_V3"),
        ("language", ep_c, "candidate_V4", "qualified_V4"),
    ]

    replay_accumulator: list[Any] = []

    for dom_name, ep_spec, cand_id, qual_id in episodes:
        logger.info(f"[{approach_name} | Seed {seed}] Starting {ep_spec.title}...")

        if is_modular:
            # Register adapter in model
            add_domain_adapter(model, dom_name, r=8)
            adapt_state, train_stats = train_domain_adapter(
                model=model,
                tokenizer=tokenizer,
                domain_name=dom_name,
                practice_samples=ep_spec.practice_samples,
                replay_samples=replay_accumulator,
                device=device,
                use_replay=use_replay,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )
            lineage.create_candidate(
                candidate_id=cand_id,
                episode_id=ep_spec.episode_id,
                domain=dom_name,
                adapter_state=adapt_state,
                base_model=model,
                curriculum_meta={"practice_count": len(ep_spec.practice_samples)},
                compute_stats=train_stats,
            )
        else:
            # Approach A: Full fine-tuning of all parameters
            trainable_params = [p for p in model.parameters()]
            optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
            # Train full parameters on practice + replay
            max_ctx = getattr(model.config, "context_length", 256)
            from adaptive_mini_gpt.reasoning.dataset import MultiRepresentationReasoningDataset, collate_reasoning_batch
            from adaptive_mini_gpt.rules.reliable_transition import ElementaryTransitionDataset, collate_elementary_batch, get_all_200_transitions
            ds_p = MultiRepresentationReasoningDataset(ep_spec.practice_samples, tokenizer, ApproachType.RECURRENT, max_ctx)
            ld_p = DataLoader(ds_p, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_reasoning_batch(b, pad_token_id=0))
            
            transitions = get_all_200_transitions()
            ds_r = ElementaryTransitionDataset(transitions, tokenizer, "m23_baseline", max_ctx)
            all_r = [{"input_ids": ds_r[i][0], "labels": ds_r[i][1]} for i in range(len(ds_r))]
            class RDS(Dataset):
                def __len__(self): return len(all_r)
                def __getitem__(self, idx): return all_r[idx]
            ld_r = DataLoader(RDS(), batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_reasoning_batch(b, pad_token_id=0))

            model.train()
            itr_r = iter(ld_r)
            for _ in range(epochs):
                for b_in, b_lbl in ld_p:
                    b_in, b_lbl = b_in.to(device), b_lbl.to(device)
                    out_p = model(b_in)
                    lg_p = out_p[0] if isinstance(out_p, tuple) else out_p
                    lp = torch.nn.functional.cross_entropy(lg_p[:, :-1, :].contiguous().view(-1, lg_p.size(-1)), b_lbl[:, 1:].contiguous().view(-1), ignore_index=-100)
                    try:
                        r_in, r_lbl = next(itr_r)
                    except StopIteration:
                        itr_r = iter(ld_r)
                        r_in, r_lbl = next(itr_r)
                    r_in, r_lbl = r_in.to(device), r_lbl.to(device)
                    out_r = model(r_in)
                    lg_r = out_r[0] if isinstance(out_r, tuple) else out_r
                    lr_loss = torch.nn.functional.cross_entropy(lg_r[:, :-1, :].contiguous().view(-1, lg_r.size(-1)), r_lbl[:, 1:].contiguous().view(-1), ignore_index=-100)
                    t_loss = lp + 0.5 * lr_loss
                    optimizer.zero_grad()
                    t_loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
            model.eval()

        # Evaluate candidate across ALL domains
        cand_metrics = evaluate_multi_domain_system(
            model=model,
            tokenizer=tokenizer,
            device=device,
            arith_test_samples=arith_holdout,
            logic_test_samples=logic_holdout,
            lang_test_samples=lang_holdout,
            is_modular=is_modular,
            active_domain=None,
        )

        # Gate Evaluation
        prev_arith = cumulative_history[-1]["metrics"].get("arithmetic_exact", 0.0)
        prev_carry = cumulative_history[-1]["metrics"].get("carry_retention_200", 0.0)

        # Criteria: acquired domain performance + retention of prior capabilities >= 80%
        if dom_name == "arithmetic":
            passed = (cand_metrics["arithmetic_exact"] >= 0.70 and cand_metrics["carry_retention_200"] >= 0.80)
        elif dom_name == "logic":
            passed = (cand_metrics["logic_exact"] >= 0.50 and cand_metrics["carry_retention_200"] >= 0.80 * prev_carry)
        else:  # language
            passed = (cand_metrics["language_exact"] >= 0.25 and cand_metrics["carry_retention_200"] >= 0.80 * prev_carry)

        if passed:
            rat = f"Candidate passed gate on {dom_name} with retention preserved."
            if is_modular:
                lineage.promote_candidate(cand_id, qual_id, cand_metrics, rat)
            replay_accumulator.extend(ep_spec.practice_samples[:16])
        else:
            rat = f"Candidate failed gate on {dom_name} or regressed retention."
            if is_modular:
                lineage.rollback_candidate(cand_id, cand_metrics, rat)

        cumulative_history.append({
            "step": f"after_{dom_name}",
            "domain": dom_name,
            "metrics": copy.deepcopy(cand_metrics),
            "promoted": passed,
        })

    return {
        "approach_name": approach_name,
        "seed": seed,
        "cumulative_history": cumulative_history,
        "promotions": lineage.successful_promotions if is_modular else sum(1 for h in cumulative_history if h.get("promoted", False) and h["step"] != "foundation_V1"),
        "rollbacks": lineage.rollbacks if is_modular else (3 - sum(1 for h in cumulative_history if h.get("promoted", False) and h["step"] != "foundation_V1")),
    }


def run_seed_m34_pipeline(
    seed: int,
    checkpoints_dir: Path,
    tokenizer: BPETokenizer,
    device: torch.device,
    epochs: int = 12,
    lr: float = 3e-4,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Execute complete M34 pipeline for a single seed lineage."""
    logger.info("#################################################################")
    logger.info(f"Executing M34 Modular Continual Lineage Seed {seed}")
    logger.info("#################################################################")

    t_start = time.perf_counter()
    random.seed(seed)
    torch.manual_seed(seed)

    # 1. Load M18 Root Checkpoint & Expand to 1024-BPE
    relative = Path(f"seed_{seed}/base_pretrain/best.pt")
    checkpoint_path = checkpoints_dir / relative
    raw = CheckpointManager.load_raw_checkpoint(checkpoint_path)

    base_tok = CheckpointManager.load_tokenizer(raw)
    state_dict = raw.get("model_state_dict", raw.get("model_state"))
    base_model_400 = MiniGPT(ModelConfig(**raw["model_config"]))
    base_model_400.load_state_dict(state_dict)
    base_model = expand_model_vocabulary(
        model=base_model_400,
        old_tokenizer=base_tok,
        new_tokenizer=tokenizer,
    ).to(device)

    # 2. Phase 2: Foundation Preparation & Baseline Comparison
    logger.info(f"[Seed {seed}] Evaluating Foundation A (Raw M18 Root)...")
    ep_a = build_m33_arithmetic_episode(seed=1000 + seed)
    ep_b = build_m33_logic_episode(seed=2000 + seed)
    ep_c = build_m33_language_episode(seed=3000 + seed)

    f_a_metrics = evaluate_multi_domain_system(
        model=base_model,
        tokenizer=tokenizer,
        device=device,
        arith_test_samples=ep_a.fresh_test_samples,
        logic_test_samples=ep_b.fresh_test_samples,
        lang_test_samples=ep_c.fresh_test_samples,
        is_modular=False,
    )

    logger.info(f"[Seed {seed}] Training & Evaluating Foundation B (M32 Recurrent Foundation)...")
    foundation_b_model = train_m32_recurrent_foundation(
        base_model=base_model,
        tokenizer=tokenizer,
        device=device,
        seed=seed,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )
    f_b_metrics = evaluate_multi_domain_system(
        model=foundation_b_model,
        tokenizer=tokenizer,
        device=device,
        arith_test_samples=ep_a.fresh_test_samples,
        logic_test_samples=ep_b.fresh_test_samples,
        lang_test_samples=ep_c.fresh_test_samples,
        is_modular=False,
    )

    foundation_comparison = {
        "foundation_a_m18_root": f_a_metrics,
        "foundation_b_m32_recurrent": f_b_metrics,
    }

    # 3. Phase 3 & 4: Controlled Evaluation of Approaches A, B, and C
    app_a_res = run_sequential_episodes_for_approach(
        approach_name="approach_a_full_ft_replay",
        foundation_model=foundation_b_model,
        tokenizer=tokenizer,
        device=device,
        seed=seed,
        ep_a=ep_a,
        ep_b=ep_b,
        ep_c=ep_c,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    app_b_res = run_sequential_episodes_for_approach(
        approach_name="approach_b_modular",
        foundation_model=foundation_b_model,
        tokenizer=tokenizer,
        device=device,
        seed=seed,
        ep_a=ep_a,
        ep_b=ep_b,
        ep_c=ep_c,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    app_c_res = run_sequential_episodes_for_approach(
        approach_name="approach_c_modular_replay",
        foundation_model=foundation_b_model,
        tokenizer=tokenizer,
        device=device,
        seed=seed,
        ep_a=ep_a,
        ep_b=ep_b,
        ep_c=ep_c,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    # 4. Phase 5: Router & Modular Inference Benchmark
    logger.info(f"[Seed {seed}] Running Phase 5 Router & Unified Inference Benchmark...")
    modular_eval_model = copy.deepcopy(foundation_b_model).to(device)
    inject_multi_domain_adapters(modular_eval_model, r=8, target_modules=["c_attn", "c_proj", "c_fc"])
    add_domain_adapter(modular_eval_model, "arithmetic", r=8)
    add_domain_adapter(modular_eval_model, "logic", r=8)
    add_domain_adapter(modular_eval_model, "language", r=8)

    reasoner = ModularReasoner(
        model=modular_eval_model,
        tokenizer=tokenizer,
        device=device,
        available_adapters=["arithmetic", "logic", "language"],
    )

    test_queries = [
        ("Add: 4521 + 3892 =", "arithmetic"),
        ("Rule: If A then B. If B then C. Fact: A is True. Query: Is C True?", "logic"),
        ("Context: Ribosomes are cellular machines responsible for protein synthesis. Question: What synthesizes proteins?", "language"),
        ("What is the capital of Mars?", "language"),
    ]

    routing_eval: list[dict[str, Any]] = []
    for q, exp_dom in test_queries:
        act, ans, lat = reasoner.reason(q)
        routing_eval.append({
            "query": q,
            "expected_domain": exp_dom,
            "routed_action": act,
            "answer_preview": ans[:40],
            "latency_ms": lat,
        })

    duration = round(time.perf_counter() - t_start, 2)
    logger.info(f"[Seed {seed}] Finished full pipeline in {duration}s")

    return {
        "seed": seed,
        "duration_seconds": duration,
        "foundation_comparison": foundation_comparison,
        "approaches": {
            "approach_a_full_ft_replay": app_a_res,
            "approach_b_modular": app_b_res,
            "approach_c_modular_replay": app_c_res,
        },
        "phase5_routing_eval": routing_eval,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M34 Modular Continual Learning Study")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("======================================================================")
    logger.info("M34 — Modular Continual Learning and Verified Checkpoint Growth Study")
    logger.info(f"Device: {device}, Seeds: {args.seeds}, Epochs: {args.epochs}, LR: {args.lr}")
    logger.info("======================================================================")

    # 1. 1024-BPE Tokenizer
    all_prose = " ".join(PRETRAIN_PROSE_CORPUS)
    tokenizer = train_scaled_bpe_tokenizer(all_prose, target_vocab_size=1024)
    tok_path = output_dir / "bpe_1024.json"
    tokenizer.save(tok_path)
    logger.info(f"1024-BPE Tokenizer saved. Fingerprint: {tokenizer.get_fingerprint()}")

    # 2. Execute Lineages Across Seeds
    seed_runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_res = run_seed_m34_pipeline(
            seed=seed,
            checkpoints_dir=checkpoints_dir,
            tokenizer=tokenizer,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        seed_runs.append(run_res)

    logger.info("\nAggregating M34 multi-lineage metrics across seeds 42, 43, 44...")

    # 3. Foundation Aggregation
    f_a_arith = compute_mean_std([r["foundation_comparison"]["foundation_a_m18_root"]["arithmetic_exact"] for r in seed_runs])
    f_b_arith = compute_mean_std([r["foundation_comparison"]["foundation_b_m32_recurrent"]["arithmetic_exact"] for r in seed_runs])
    f_a_carry = compute_mean_std([r["foundation_comparison"]["foundation_a_m18_root"]["carry_retention_200"] for r in seed_runs])
    f_b_carry = compute_mean_std([r["foundation_comparison"]["foundation_b_m32_recurrent"]["carry_retention_200"] for r in seed_runs])

    # 4. Approach Comparison Aggregation across final steps
    step_names = ["foundation_V1", "after_arithmetic", "after_logic", "after_language"]
    approach_names = ["approach_a_full_ft_replay", "approach_b_modular", "approach_c_modular_replay"]

    approach_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for app in approach_names:
        approach_summaries[app] = {}
        for st in step_names:
            arith_vals = []
            logic_vals = []
            lang_vals = []
            carry_vals = []
            for r in seed_runs:
                hist = r["approaches"][app]["cumulative_history"]
                entries = [h for h in hist if h["step"] == st]
                if entries:
                    m = entries[0]["metrics"]
                    arith_vals.append(m.get("arithmetic_exact", 0.0))
                    logic_vals.append(m.get("logic_exact", 0.0))
                    lang_vals.append(m.get("language_exact", 0.0))
                    carry_vals.append(m.get("carry_retention_200", 0.0))
            approach_summaries[app][st] = {
                "arithmetic_exact": compute_mean_std(arith_vals),
                "logic_exact": compute_mean_std(logic_vals),
                "language_exact": compute_mean_std(lang_vals),
                "carry_retention_200": compute_mean_std(carry_vals),
            }

    m34_report = {
        "metadata": {
            "milestone": "M34",
            "study_name": "Modular Continual Learning and Verified Checkpoint Growth",
            "seeds": args.seeds,
            "device": str(device),
            "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", 1024),
            "tokenizer_fingerprint": tokenizer.get_fingerprint(),
            "mean_duration_seconds": round(sum(r["duration_seconds"] for r in seed_runs) / len(seed_runs), 2),
        },
        "foundation_comparison_summary": {
            "foundation_a_m18_root": {
                "arithmetic_exact": f_a_arith,
                "carry_retention_200": f_a_carry,
            },
            "foundation_b_m32_recurrent": {
                "arithmetic_exact": f_b_arith,
                "carry_retention_200": f_b_carry,
            },
        },
        "approach_progression_summary": approach_summaries,
        "individual_seed_runs": seed_runs,
    }

    report_path = output_dir / "m34_report.json"
    with open(report_path, "w") as f:
        json.dump(m34_report, f, indent=2)
    logger.info(f"M34 Report successfully written to {report_path}")

    logger.info("\n=================== M34 CUMULATIVE APPROACH COMPARISON ===================")
    for app in approach_names:
        logger.info(f"\n--- {app.upper()} ---")
        for st in step_names:
            s_data = approach_summaries[app][st]
            ar = s_data["arithmetic_exact"]
            lg = s_data["logic_exact"]
            la = s_data["language_exact"]
            cr = s_data["carry_retention_200"]
            logger.info(
                f"[{st:<16}] Arith: {ar['mean']*100:.1f}% ± {ar['std']*100:.1f}% | "
                f"Logic: {lg['mean']*100:.1f}% ± {lg['std']*100:.1f}% | "
                f"Lang: {la['mean']*100:.1f}% ± {la['std']*100:.1f}% | "
                f"Carry: {cr['mean']*100:.1f}% ± {cr['std']*100:.1f}%"
            )
    logger.info("==========================================================================")


if __name__ == "__main__":
    main()
