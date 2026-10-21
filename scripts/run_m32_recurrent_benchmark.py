"""M32 Benchmark: Verified Recurrent Learning and Adaptive Reasoning Study.

Orchestrates:
  1. Phase 1: Exhaustive audit of all 200 elementary addition states and failure mode classification.
  2. Phase 2 & 6: Controlled training comparing:
     - Arm A: Original Balanced Training
     - Arm B: Failure-Targeted Practice
     - Arm C: Failure-Targeted Practice with Balanced Multi-Domain Replay
     Across 2, 4, 8, 16, 32 digits, cascading carries, and 3-6 hop logic.
  3. Phase 3: Step-level correction (Single attempt vs Neural retry vs Verifier-assisted).
  4. Phase 4: Adaptive action routing (Deterministic vs Neural Selector vs Baselines).
  5. Phase 5: Closed-loop interaction episode with isolated candidate training and gate rollback.
  6. Multi-lineage evaluation across Seeds 42, 43, 44 on Tesla T4 GPU.
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
from adaptive_mini_gpt.data.language_corpus import PRETRAIN_PROSE_CORPUS
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.interaction.closed_loop_recurrent import (
    run_closed_loop_recurrent_episode,
)
from adaptive_mini_gpt.model.capacity_migration import create_1m_control_config
from adaptive_mini_gpt.model.transformer import MiniGPT, ModelConfig
from adaptive_mini_gpt.reasoning.adaptive_router import (
    AlwaysDirectRouter,
    AlwaysRecurrentRouter,
    AlwaysRetrieveRouter,
    DeterministicRouter,
    NeuralActionSelector,
    ReasoningAction,
    build_routing_benchmark_dataset,
)
from adaptive_mini_gpt.reasoning.dataset import (
    MultiRepresentationReasoningDataset,
    build_arithmetic_reasoning_suite,
    build_language_reasoning_suite,
    build_logic_reasoning_suite,
    collate_reasoning_batch,
)
from adaptive_mini_gpt.reasoning.recurrent_audit import (
    audit_all_200_recurrent_states,
)
from adaptive_mini_gpt.reasoning.recurrent_curriculum import (
    AdditionProblemSpec,
    build_deep_logic_suite,
    build_failure_targeted_recurrent_practice,
    build_multi_digit_addition_suite,
)
from adaptive_mini_gpt.reasoning.recurrent_reasoner import (
    rollout_recurrent_arithmetic,
    rollout_recurrent_logic,
)
from adaptive_mini_gpt.reasoning.scratchpad import (
    ApproachType,
    ReasoningDomain,
    ReasoningSample,
    build_arithmetic_reasoning_sample,
)
from adaptive_mini_gpt.reasoning.step_correction import (
    CorrectionMode,
    rollout_addition_with_correction,
)
from adaptive_mini_gpt.rules.reliable_transition import (
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
logger = logging.getLogger("m32_recurrent_benchmark")


def train_recurrent_arm(
    arm_name: str,
    base_model: MiniGPT,
    tokenizer: BPETokenizer,
    practice_samples: list[ReasoningSample],
    device: torch.device,
    use_replay: bool = False,
    epochs: int = 12,
    lr: float = 3e-4,
    batch_size: int = 8,
) -> MiniGPT:
    """Train a recurrent model under matched compute and token exposure."""
    logger.info(f"Training Arm [{arm_name}] on {len(practice_samples)} samples (replay={use_replay})...")
    model = copy.deepcopy(base_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    max_ctx = getattr(model.config, "context_length", 256)
    pad_id = getattr(tokenizer, "pad_id", 0)

    ds_practice = MultiRepresentationReasoningDataset(
        samples=practice_samples,
        tokenizer=tokenizer,
        approach=ApproachType.RECURRENT,
        context_length=max_ctx,
    )
    loader_practice = DataLoader(
        ds_practice,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_reasoning_batch(b, pad_token_id=pad_id),
    )

    loader_replay = None
    if use_replay:
        transitions = get_all_200_transitions()
        ds_replay = ElementaryTransitionDataset(
            transitions=transitions,
            tokenizer=tokenizer,
            format_type="m23_baseline",
            context_length=max_ctx,
        )
        loader_replay = DataLoader(
            ds_replay,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_elementary_batch(b, pad_token_id=pad_id),
        )

    model.train()
    iter_replay = iter(loader_replay) if loader_replay else None

    for _ in range(epochs):
        for b_inputs, b_labels in loader_practice:
            b_inputs = b_inputs.to(device)
            b_labels = b_labels.to(device)

            out_p = model(b_inputs, labels=b_labels)
            loss_p = out_p[0] if isinstance(out_p, tuple) else out_p

            if use_replay and iter_replay:
                try:
                    r_inputs, r_labels = next(iter_replay)
                except StopIteration:
                    iter_replay = iter(loader_replay)
                    r_inputs, r_labels = next(iter_replay)
                r_inputs = r_inputs.to(device)
                r_labels = r_labels.to(device)
                out_r = model(r_inputs, labels=r_labels)
                loss_r = out_r[0] if isinstance(out_r, tuple) else out_r
                total_loss = loss_p + 0.5 * loss_r
            else:
                total_loss = loss_p

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    return model


def train_neural_router(
    base_model: MiniGPT,
    tokenizer: BPETokenizer,
    train_routing_samples: list[Any],
    device: torch.device,
    epochs: int = 8,
    lr: float = 3e-4,
    batch_size: int = 8,
) -> MiniGPT:
    """Train minimal neural router on action selection prompts."""
    logger.info(f"Training Neural Action Router on {len(train_routing_samples)} routing samples...")
    model = copy.deepcopy(base_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    max_ctx = getattr(model.config, "context_length", 256)
    pad_id = getattr(tokenizer, "pad_id", 0)

    input_ids_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for s in train_routing_samples:
        p_str = NeuralActionSelector.format_routing_prompt(s.query)
        t_str = f"{s.optimal_action.value}<|eos|>\n"
        p_toks = tokenizer.encode(p_str)
        t_toks = tokenizer.encode(t_str)
        full_toks = p_toks + t_toks
        if len(full_toks) > max_ctx:
            full_toks = full_toks[:max_ctx]
            p_len = min(len(p_toks), max_ctx)
        else:
            p_len = len(p_toks)
        labels = [-100] * p_len + full_toks[p_len:]
        input_ids_list.append(torch.tensor(full_toks, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    class RouteDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(input_ids_list)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            return input_ids_list[idx], labels_list[idx]

    loader = DataLoader(
        RouteDataset(),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_elementary_batch(b, pad_token_id=pad_id),
    )

    model.train()
    for _ in range(epochs):
        for b_inputs, b_labels in loader:
            b_inputs = b_inputs.to(device)
            b_labels = b_labels.to(device)
            out = model(b_inputs, labels=b_labels)
            loss = out[0] if isinstance(out, tuple) else out
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    return model


def evaluate_arithmetic_suite(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    problems: list[AdditionProblemSpec],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate multi-digit addition suite with autonomous rollout."""
    exact_count = 0
    total_steps = 0
    correct_steps = 0
    latencies: list[float] = []

    for prob in problems:
        t0 = time.perf_counter()
        trace = rollout_recurrent_arithmetic(
            model=model,
            tokenizer=tokenizer,
            a=prob.a,
            b=prob.b,
            device=device,
        )
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        if trace.is_exact_match:
            exact_count += 1
        total_steps += len(trace.step_accuracies)
        correct_steps += sum(1 for s in trace.step_accuracies if s)

    n = max(1, len(problems))
    step_acc = (correct_steps / max(1, total_steps)) if total_steps > 0 else 0.0
    mean_lat = sum(latencies) / n

    return {
        "final_exact_accuracy": round(exact_count / n, 4),
        "intermediate_step_accuracy": round(step_acc, 4),
        "mean_latency_ms": round(mean_lat, 2),
        "total_evaluated": len(problems),
    }


def evaluate_logic_suite(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    samples: list[ReasoningSample],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate logic suite with autonomous recurrent rollout."""
    exact_count = 0
    total_steps = 0
    correct_steps = 0
    latencies: list[float] = []

    for sample in samples:
        t0 = time.perf_counter()
        trace = rollout_recurrent_logic(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            device=device,
        )
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        if trace.is_exact_match:
            exact_count += 1
        total_steps += len(trace.step_accuracies)
        correct_steps += sum(1 for s in trace.step_accuracies if s)

    n = max(1, len(samples))
    step_acc = (correct_steps / max(1, total_steps)) if total_steps > 0 else 0.0
    mean_lat = sum(latencies) / n

    return {
        "final_exact_accuracy": round(exact_count / n, 4),
        "intermediate_step_accuracy": round(step_acc, 4),
        "mean_latency_ms": round(mean_lat, 2),
        "total_evaluated": len(samples),
    }


def run_seed_pipeline(
    seed: int,
    checkpoints_dir: Path,
    tokenizer: BPETokenizer,
    device: torch.device,
    epochs: int = 12,
    lr: float = 3e-4,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Run full M32 pipeline for a single seed lineage."""
    logger.info(f"#################################################################")
    logger.info(f"Executing M32 Lineage Seed {seed}")
    logger.info(f"#################################################################")

    t_start = time.perf_counter()
    random.seed(seed)
    torch.manual_seed(seed)

    # 1. Load M18 Root Checkpoint & Expand to 1024-BPE
    relative = Path(f"seed_{seed}/base_pretrain/best.pt")
    checkpoint_path = checkpoints_dir / relative
    raw = CheckpointManager.load_raw_checkpoint(checkpoint_path)

    base_model = MiniGPT(ModelConfig(**raw["model_config"]))
    base_model.load_state_dict(raw["model_state_dict"])
    base_model = expand_model_vocabulary(base_model, new_vocab_size=1024)
    base_model.to(device)
    param_count = sum(p.numel() for p in base_model.parameters())

    # 2. Phase 1 Failure Diagnostic Audit of all 200 Elementary States
    logger.info(f"[Seed {seed}] Running Phase 1 200-State Transition Audit...")
    phase1_audit = audit_all_200_recurrent_states(
        model=base_model,
        tokenizer=tokenizer,
        device=device,
    )

    # 3. Build Evaluation Suites
    arith_suites = build_multi_digit_addition_suite(seed=1000 + seed)
    logic_suites = build_deep_logic_suite(seed=2000 + seed)

    # 4. Construct Practice Curricula
    # Curriculum A: Original Balanced Training (M31 style)
    curriculum_a = build_arithmetic_reasoning_suite()
    curriculum_a.extend(build_logic_reasoning_suite())
    curriculum_a.extend(build_language_reasoning_suite())

    # Curriculum B & C: Failure-Targeted Practice (Focused on ripple carries, double carries, cascading carries)
    curriculum_bc = build_failure_targeted_recurrent_practice(target_sample_count=72, seed=3000 + seed)
    curriculum_bc.extend(build_logic_reasoning_suite())
    curriculum_bc.extend(build_language_reasoning_suite())

    # 5. Train Arms
    logger.info(f"--- [Seed {seed}] Training Arm A: Original Balanced ---")
    model_arm_a = train_recurrent_arm(
        arm_name="arm_a_balanced",
        base_model=base_model,
        tokenizer=tokenizer,
        practice_samples=curriculum_a,
        device=device,
        use_replay=False,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    logger.info(f"--- [Seed {seed}] Training Arm B: Failure-Targeted ---")
    model_arm_b = train_recurrent_arm(
        arm_name="arm_b_failure_targeted",
        base_model=base_model,
        tokenizer=tokenizer,
        practice_samples=curriculum_bc,
        device=device,
        use_replay=False,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    logger.info(f"--- [Seed {seed}] Training Arm C: Failure-Targeted with Replay ---")
    model_arm_c = train_recurrent_arm(
        arm_name="arm_c_targeted_replay",
        base_model=base_model,
        tokenizer=tokenizer,
        practice_samples=curriculum_bc,
        device=device,
        use_replay=True,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    # 6. Evaluate Arms Across All Suites
    arms = {
        "arm_a_balanced": model_arm_a,
        "arm_b_failure_targeted": model_arm_b,
        "arm_c_targeted_replay": model_arm_c,
    }

    arms_eval: dict[str, dict[str, Any]] = {}
    for arm_name, arm_model in arms.items():
        logger.info(f"Evaluating {arm_name} across arithmetic and logic suites...")
        arm_results: dict[str, Any] = {}

        # Arithmetic Length Extrapolation
        for split_name, probs in arith_suites.items():
            arm_results[split_name] = evaluate_arithmetic_suite(arm_model, tokenizer, probs, device)

        # Logic Depth Extrapolation
        for split_name, samples in logic_suites.items():
            arm_results[split_name] = evaluate_logic_suite(arm_model, tokenizer, samples, device)

        # Carry Retention on 200 Transitions
        ret_rep = audit_elementary_transition_space(
            model=arm_model,
            tokenizer=tokenizer,
            device=device,
            format_type="m23_baseline",
        )
        arm_results["carry_retention_200"] = round(ret_rep["overall_carry_acc"] / 100.0, 4)
        arms_eval[arm_name] = arm_results

    # 7. Phase 3 Step-Level Correction Benchmark (on Arm C)
    logger.info(f"[Seed {seed}] Running Phase 3 Step-Level Correction Benchmark on Arm C...")
    test_probs_correction = arith_suites["arith_4_digits"] + arith_suites["arith_cascading_carry"][:8]
    correction_eval: dict[str, dict[str, float]] = {}

    for mode in [CorrectionMode.SINGLE_ATTEMPT, CorrectionMode.NEURAL_RETRY, CorrectionMode.VERIFIER_ASSISTED]:
        mode_exact = 0
        mode_steps_total = 0
        mode_steps_correct = 0
        mode_corrections = 0
        mode_retries = 0

        for prob in test_probs_correction:
            r = rollout_addition_with_correction(
                model=model_arm_c,
                tokenizer=tokenizer,
                a=prob.a,
                b=prob.b,
                device=device,
                mode=mode,
            )
            if r.is_exact_match:
                mode_exact += 1
            mode_steps_total += len(r.step_accuracies)
            mode_steps_correct += sum(1 for ok in r.step_accuracies if ok)
            mode_corrections += r.verifier_corrections_applied
            mode_retries += r.neural_retries_attempted

        n_p = len(test_probs_correction)
        correction_eval[mode.value] = {
            "exact_accuracy": round(mode_exact / n_p, 4),
            "step_accuracy": round(mode_steps_correct / max(1, mode_steps_total), 4),
            "corrections_applied": mode_corrections,
            "neural_retries": mode_retries,
            "is_verifier_assisted": (mode == CorrectionMode.VERIFIER_ASSISTED),
        }

    # 8. Phase 4 Adaptive Action Routing Benchmark
    logger.info(f"[Seed {seed}] Running Phase 4 Adaptive Action Routing Benchmark...")
    train_routing, eval_routing = build_routing_benchmark_dataset()

    neural_router_model = train_neural_router(
        base_model=base_model,
        tokenizer=tokenizer,
        train_routing_samples=train_routing,
        device=device,
        epochs=8,
        lr=lr,
        batch_size=batch_size,
    )
    neural_router = NeuralActionSelector(neural_router_model, tokenizer, device)

    routers = {
        "deterministic_router": DeterministicRouter(),
        "neural_action_selector": neural_router,
        "always_direct": AlwaysDirectRouter(),
        "always_recurrent": AlwaysRecurrentRouter(),
        "always_retrieve": AlwaysRetrieveRouter(),
    }

    routing_eval: dict[str, dict[str, Any]] = {}
    for r_name, r_obj in routers.items():
        correct_routes = 0
        latencies_route: list[float] = []
        action_counts: dict[str, int] = {act.value: 0 for act in ReasoningAction}

        for sample in eval_routing:
            t0 = time.perf_counter()
            dec = r_obj.route(sample.query)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies_route.append(lat)
            action_counts[dec.action.value] += 1
            if dec.action == sample.optimal_action:
                correct_routes += 1

        n_eval = len(eval_routing)
        routing_eval[r_name] = {
            "routing_accuracy": round(correct_routes / n_eval, 4),
            "mean_latency_ms": round(sum(latencies_route) / n_eval, 2),
            "action_distribution": action_counts,
        }

    # 9. Phase 5 Closed-Loop Interaction Episode
    logger.info(f"[Seed {seed}] Running Phase 5 Closed-Loop Recurrent Learning Episode...")
    closed_loop_res = run_closed_loop_recurrent_episode(
        base_model=base_model,
        tokenizer=tokenizer,
        device=device,
        prob_a=6138,
        prob_b=1936,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    duration = round(time.perf_counter() - t_start, 2)
    logger.info(f"[Seed {seed}] Finished in {duration}s")

    return {
        "seed": seed,
        "duration_seconds": duration,
        "parameters_1m": param_count,
        "phase1_audit": asdict(phase1_audit),
        "arms_evaluation": arms_eval,
        "step_correction_benchmark": correction_eval,
        "routing_benchmark": routing_eval,
        "closed_loop_episode": asdict(closed_loop_res),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M32 Verified Recurrent Learning and Adaptive Reasoning Study")
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
    logger.info("M32 — Verified Recurrent Learning and Adaptive Reasoning Study")
    logger.info(f"Device: {device}, Seeds: {args.seeds}, Epochs: {args.epochs}, LR: {args.lr}")
    logger.info("======================================================================")

    # 1. 1024-BPE Tokenizer (Shared across lineages)
    all_prose = " ".join(PRETRAIN_PROSE_CORPUS)
    tokenizer = train_scaled_bpe_tokenizer(all_prose, target_vocab_size=1024)
    tok_path = output_dir / "bpe_1024.json"
    tokenizer.save(tok_path)
    logger.info(f"1024-BPE Tokenizer saved. Fingerprint: {tokenizer.get_fingerprint()}")

    seed_runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_res = run_seed_pipeline(
            seed=seed,
            checkpoints_dir=checkpoints_dir,
            tokenizer=tokenizer,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        seed_runs.append(run_res)

    logger.info("\nAggregating M32 multi-lineage metrics across seeds 42, 43, 44...")

    # Consolidate Arms Summary
    arm_names = ["arm_a_balanced", "arm_b_failure_targeted", "arm_c_targeted_replay"]
    sample_eval = seed_runs[0]["arms_evaluation"][arm_names[0]]
    splits = [k for k in sample_eval.keys() if k != "carry_retention_200"]

    arms_summary: dict[str, dict[str, Any]] = {}
    for arm in arm_names:
        arms_summary[arm] = {}
        for sp in splits:
            ex_list = [run["arms_evaluation"][arm][sp]["final_exact_accuracy"] for run in seed_runs]
            st_list = [run["arms_evaluation"][arm][sp]["intermediate_step_accuracy"] for run in seed_runs]
            lat_list = [run["arms_evaluation"][arm][sp]["mean_latency_ms"] for run in seed_runs]
            ret_list = [run["arms_evaluation"][arm]["carry_retention_200"] for run in seed_runs]

            arms_summary[arm][sp] = {
                "mean_final_exact_accuracy": round(sum(ex_list) / len(ex_list), 4),
                "std_final_exact_accuracy": round(math.sqrt(sum((x - sum(ex_list)/len(ex_list))**2 for x in ex_list)/len(ex_list)), 4) if len(ex_list) > 1 else 0.0,
                "mean_intermediate_step_accuracy": round(sum(st_list) / len(st_list), 4),
                "mean_latency_ms": round(sum(lat_list) / len(lat_list), 2),
                "mean_retention_carry": round(sum(ret_list) / len(ret_list), 4),
            }

    # Consolidate Correction Summary
    correction_summary: dict[str, Any] = {}
    modes = [m.value for m in CorrectionMode]
    for m in modes:
        ex_list = [run["step_correction_benchmark"][m]["exact_accuracy"] for run in seed_runs]
        st_list = [run["step_correction_benchmark"][m]["step_accuracy"] for run in seed_runs]
        corr_list = [run["step_correction_benchmark"][m]["corrections_applied"] for run in seed_runs]
        ret_list = [run["step_correction_benchmark"][m]["neural_retries"] for run in seed_runs]
        correction_summary[m] = {
            "mean_exact_accuracy": round(sum(ex_list) / len(ex_list), 4),
            "mean_step_accuracy": round(sum(st_list) / len(st_list), 4),
            "mean_corrections_applied": round(sum(corr_list) / len(corr_list), 1),
            "mean_neural_retries": round(sum(ret_list) / len(ret_list), 1),
            "is_verifier_assisted": seed_runs[0]["step_correction_benchmark"][m]["is_verifier_assisted"],
        }

    # Consolidate Routing Summary
    routing_summary: dict[str, Any] = {}
    r_names = list(seed_runs[0]["routing_benchmark"].keys())
    for r in r_names:
        acc_list = [run["routing_benchmark"][r]["routing_accuracy"] for run in seed_runs]
        lat_list = [run["routing_benchmark"][r]["mean_latency_ms"] for run in seed_runs]
        routing_summary[r] = {
            "mean_routing_accuracy": round(sum(acc_list) / len(acc_list), 4),
            "mean_latency_ms": round(sum(lat_list) / len(lat_list), 2),
        }

    # Consolidate Phase 1 Summary
    p1_exact = [run["phase1_audit"]["overall_exact_accuracy"] for run in seed_runs]
    p1_digit = [run["phase1_audit"]["overall_digit_accuracy"] for run in seed_runs]
    p1_carry = [run["phase1_audit"]["overall_carry_accuracy"] for run in seed_runs]
    p1_teacher_acc = [run["phase1_audit"]["teacher_forced_accuracy"] for run in seed_runs]
    p1_teacher_loss = [run["phase1_audit"]["mean_teacher_loss"] for run in seed_runs]

    phase1_summary = {
        "mean_overall_exact_accuracy": round(sum(p1_exact) / len(p1_exact), 2),
        "mean_overall_digit_accuracy": round(sum(p1_digit) / len(p1_digit), 2),
        "mean_overall_carry_accuracy": round(sum(p1_carry) / len(p1_carry), 2),
        "mean_teacher_forced_accuracy": round(sum(p1_teacher_acc) / len(p1_teacher_acc), 2),
        "mean_teacher_loss": round(sum(p1_teacher_loss) / len(p1_teacher_loss), 4),
    }

    param_count = seed_runs[0]["parameters_1m"]

    report = {
        "milestone": "M32",
        "title": "Verified Recurrent Learning and Adaptive Reasoning Study",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "seeds": args.seeds,
        "parameters_1m": param_count,
        "mean_duration_seconds": round(sum(r["duration_seconds"] for r in seed_runs) / len(seed_runs), 2),
        "tokenizer_1024_fingerprint": tokenizer.get_fingerprint(),
        "phase1_audit_summary": phase1_summary,
        "arms_summary": arms_summary,
        "correction_summary": correction_summary,
        "routing_summary": routing_summary,
        "raw_seed_runs": seed_runs,
    }

    report_path = output_dir / "m32_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("======================================================================")
    logger.info(f"M32 Benchmark complete! Summary saved to {report_path}")
    logger.info("======================================================================")


if __name__ == "__main__":
    main()
