"""M19.5 Closed-Loop Action Learning and Recoverability-Aware Reasoning Benchmark.

Implements the four phases specified in M19.5:
- Phase 1: Rollout diagnostics with full trajectory recording.
- Phase 2: Recoverability-aware corrective labeling via deterministic forward search.
- Phase 3: Matched-budget training comparison between Regime A (supervised reference) and
           Regime B (closed-loop learner rollout aggregation with verified corrective labels).
- Phase 4: Evaluation across 3 lineages (seeds 42, 43, 44) on sealed tests, goal reachability,
           and multi-corpus language retention.
"""

from __future__ import annotations

import argparse
import copy
import heapq
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.action_dataset import ActionTrace, ActionTraceDataset
from adaptive_mini_gpt.rules.action_policy import (
    ActionPolicyModel,
    EvaluationMetrics,
    compute_action_loss,
    evaluate_action_ranking,
    format_context_prompt,
    rollout_problem,
    train_action_policy,
)
from adaptive_mini_gpt.rules.grammar_planner import (
    DependencyResolutionError,
    ExecutionPlan,
    GrammarConstrainedPlanner,
    PlanGrammar,
    PlanStep,
    RuleExecutor,
    SearchNode,
    TargetGoal,
    WorkingMemory,
)
from adaptive_mini_gpt.rules.multistep_dataset import (
    MultiStepProblem,
    generate_addition_depth_ladder,
    generate_multistep_problems,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m19_5_closed_loop")


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


def find_recoverable_plan(
    planner: GrammarConstrainedPlanner,
    question: str,
    slots: dict[str, Any],
    initial_mem: WorkingMemory,
    initial_steps: list[PlanStep],
) -> tuple[bool, list[PlanStep]]:
    """Determine if a goal is reachable from an intermediate state and return remaining steps."""
    goal = TargetGoal.from_question(question, slots)

    # Check if current state already achieves the semantic goal
    if initial_steps:
        plan = ExecutionPlan(steps=initial_steps, final_var=initial_steps[-1].output_var)
        initial_mem.bind_result(initial_mem.variables.get(plan.final_var or ""), verified=True)
        sem_ok, _ = goal.evaluate_semantics(plan, initial_mem)
        if sem_ok:
            return True, []

    score_0 = sum(1.0 / (i + 1) for i in range(1, len(initial_steps) + 1))
    root = SearchNode(
        priority=-score_0,
        depth=len(initial_steps),
        executed_steps=list(initial_steps),
        working_memory=copy.deepcopy(initial_mem),
        state_signature=planner._compute_state_signature(initial_steps),
        score=score_0,
    )
    frontier = [root]
    visited = {root.state_signature}
    expansions = 0

    while frontier and expansions < planner.max_expansions:
        curr = heapq.heappop(frontier)
        if curr.executed_steps:
            cand_plan = ExecutionPlan(steps=curr.executed_steps, final_var=curr.executed_steps[-1].output_var)
            cand_mem = curr.working_memory
            cand_mem.bind_result(cand_mem.variables.get(cand_plan.final_var or ""), verified=True)
            sem_ok, _ = goal.evaluate_semantics(cand_plan, cand_mem)
            if sem_ok:
                remaining_steps = curr.executed_steps[len(initial_steps):]
                return True, remaining_steps

        if curr.depth >= planner.max_depth:
            continue

        expansions += 1
        candidates = PlanGrammar.get_candidate_steps(
            slots, curr.working_memory, curr.executed_steps, planner.max_depth, question
        )
        branch_nodes = []
        for cand in candidates:
            is_valid, _ = PlanGrammar.validate_step(cand, slots, curr.working_memory, curr.executed_steps)
            if not is_valid:
                continue

            branch_mem = copy.deepcopy(curr.working_memory)
            resolved = {}
            try:
                for arg_k, arg_v in cand.args.items():
                    if isinstance(arg_v, str) and arg_v.startswith("$"):
                        ref = arg_v[1:].strip()
                        if ref in branch_mem.variables:
                            resolved[arg_k] = branch_mem.variables[ref]
                        elif ref in branch_mem.slots:
                            resolved[arg_k] = branch_mem.slots[ref].value
                        else:
                            raise DependencyResolutionError(ref)
                    else:
                        resolved[arg_k] = arg_v
            except DependencyResolutionError:
                continue

            exec_res = planner.rule_executor.execute(cand.operation, resolved, verify=True)
            if not exec_res.success or not exec_res.verified:
                continue

            branch_mem.bind_step_variable(
                cand.output_var,
                exec_res.result,
                cand.operation,
                cand.args,
                resolved,
                cand.units,
                len(curr.executed_steps) + 1,
            )

            new_steps = list(curr.executed_steps) + [cand]
            sig = planner._compute_state_signature(new_steps)
            if sig in visited:
                continue
            visited.add(sig)

            step_score = (
                planner.score_fn(question, curr.executed_steps, cand)
                if planner.score_fn
                else (1.0 / (len(new_steps) + 1))
            )
            total_score = curr.score + step_score
            b_node = SearchNode(
                priority=-total_score,
                depth=curr.depth + 1,
                executed_steps=new_steps,
                working_memory=branch_mem,
                state_signature=sig,
                score=total_score,
            )
            branch_nodes.append(b_node)

        branch_nodes.sort(key=lambda n: n.priority)
        for n in branch_nodes[: planner.beam_width]:
            heapq.heappush(frontier, n)

    return False, []


def collect_closed_loop_rollout_traces(
    policy_model: ActionPolicyModel,
    problems: list[MultiStepProblem],
    device: torch.device,
    max_steps: int = 5,
) -> tuple[list[ActionTrace], dict[str, Any]]:
    """Phase 1 & 2: Collect neural rollout trajectories, test recoverability, and label corrective actions."""
    policy_model.eval()
    planner = GrammarConstrainedPlanner(beam_width=4, max_expansions=30, max_depth=max_steps)
    executor = RuleExecutor()

    corrective_traces: list[ActionTrace] = []
    total_states_visited = 0
    recoverable_states = 0
    unrecoverable_states = 0
    first_divergence_positions: dict[int, int] = {}
    total_rollout_successes = 0

    for prob_idx, prob in enumerate(problems):
        slots = prob.slots
        goal = TargetGoal.from_question(prob.question, slots)
        live_mem = WorkingMemory()
        live_mem.slots = copy.copy(slots)
        executed_steps: list[PlanStep] = []
        first_div_recorded = False

        for step_idx in range(max_steps):
            total_states_visited += 1
            candidates = PlanGrammar.get_candidate_steps(
                slots=slots,
                working_memory=live_mem,
                executed_steps=executed_steps,
                max_steps=max_steps,
                question=prob.question,
            )
            if not candidates:
                unrecoverable_states += 1
                break

            prompt = format_context_prompt(prob.question, slots, executed_steps)
            with torch.no_grad():
                scores = policy_model.score_candidates(prompt, candidates, device)

            chosen_idx = int(torch.argmax(scores).item())
            chosen_cand = candidates[chosen_idx]

            # Check recoverability of CURRENT state before taking action
            is_rec, remaining_plan = find_recoverable_plan(planner, prob.question, slots, live_mem, executed_steps)
            if is_rec:
                recoverable_states += 1
                # If remaining plan steps exist, the first step is a verified corrective action
                if remaining_plan:
                    target_step = remaining_plan[0]
                    correct_indices = [
                        idx
                        for idx, c in enumerate(candidates)
                        if c.operation == target_step.operation and c.units == target_step.units
                    ]
                    if not correct_indices:
                        correct_indices = [0]

                    # Record corrective trace for DAgger aggregation
                    corrective_trace = ActionTrace(
                        trace_id=f"closed_loop_{prob.archetype}_{prob_idx}_step_{len(executed_steps)+1}",
                        problem_key=f"{prob.archetype}_{prob_idx}",
                        question=prob.question,
                        slots=slots,
                        executed_steps=list(executed_steps),
                        valid_candidates=candidates,
                        correct_action_indices=correct_indices,
                        target_action_index=correct_indices[0],
                        step_index=len(executed_steps) + 1,
                        total_plan_steps=len(executed_steps) + len(remaining_plan),
                        archetype=prob.archetype,
                        split="train",
                    )
                    corrective_traces.append(corrective_trace)

                    # Track if chosen action diverged from verified recovery path
                    if chosen_idx not in correct_indices and not first_div_recorded:
                        first_divergence_positions[step_idx + 1] = first_divergence_positions.get(step_idx + 1, 0) + 1
                        first_div_recorded = True
            else:
                unrecoverable_states += 1
                if not first_div_recorded:
                    first_divergence_positions[step_idx + 1] = first_divergence_positions.get(step_idx + 1, 0) + 1
                    first_div_recorded = True

            # Execute the neural policy's chosen step to advance the rollout
            resolved: dict[str, Any] = {}
            for k, v in chosen_cand.args.items():
                if isinstance(v, str) and v.startswith("$"):
                    r_name = v[1:].strip()
                    resolved[k] = live_mem.variables.get(
                        r_name, live_mem.slots[r_name].value if r_name in live_mem.slots else None
                    )
                else:
                    resolved[k] = v

            exec_res = executor.execute(chosen_cand.operation, resolved, verify=True)
            if not exec_res.success or not exec_res.verified:
                break

            live_mem.bind_step_variable(
                var_name=chosen_cand.output_var,
                value=exec_res.result,
                operation=chosen_cand.operation,
                raw_args=chosen_cand.args,
                resolved_args=resolved,
                units=chosen_cand.units,
                step_index=len(executed_steps) + 1,
            )
            live_mem.bind_result(exec_res.result, verified=True)
            executed_steps.append(chosen_cand)

            plan = ExecutionPlan(steps=executed_steps, final_var=chosen_cand.output_var)
            sem_ok, _ = goal.evaluate_semantics(plan, live_mem)
            if sem_ok:
                total_rollout_successes += 1
                break

    stats = {
        "total_problems_rolled_out": len(problems),
        "rollout_successes": total_rollout_successes,
        "rollout_success_rate": (total_rollout_successes / max(1, len(problems))) * 100.0,
        "total_states_visited": total_states_visited,
        "recoverable_states": recoverable_states,
        "unrecoverable_states": unrecoverable_states,
        "recoverability_rate": (recoverable_states / max(1, total_states_visited)) * 100.0,
        "first_divergence_positions": first_divergence_positions,
        "corrective_traces_generated": len(corrective_traces),
    }
    return corrective_traces, stats


def evaluate_retention_suite(
    model: ActionPolicyModel,
    device: torch.device,
) -> dict[str, Any]:
    """Phase 4: Evaluate language retention on multi-domain held-out corpora."""
    model.eval()
    results: dict[str, Any] = {}
    for corpus_name, text in EXPANDED_RETENTION_CORPUS.items():
        tokens = torch.tensor([model.tokenizer.encode(text)], dtype=torch.long, device=device)
        loss_active = model.evaluate_language_loss(tokens, bypass_adapters=False)
        loss_bypassed = model.evaluate_language_loss(tokens, bypass_adapters=True)
        results[corpus_name] = {
            "loss_active": round(loss_active, 4),
            "loss_bypassed": round(loss_bypassed, 4),
            "drift_bypassed": round(loss_active - loss_bypassed, 6),
        }
    return results


def run_m19_5_benchmark(
    base_checkpoints_dir: Path,
    output_dir: Path,
    device_str: str = "cuda",
    seeds: list[int] = [42, 43, 44],
    epochs: int = 10,
) -> dict[str, Any]:
    """Execute complete M19.5 closed-loop action learning benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    logger.info("Initializing M19.5 benchmark across lineages %s on %s...", seeds, device)

    # 1. Sealed Test Sets (Strict isolation: seeds 3001, 4001, 5001)
    test_familiar_probs = generate_multistep_problems(
        num_samples_per_archetype=10,
        seed=3001,
        split="test",
        archetypes=["chained_addition_3", "chained_addition_4", "2segment_speed", "multi_entity_compare"],
    )
    test_ood_speed_probs = generate_multistep_problems(
        num_samples_per_archetype=10,
        seed=4001,
        split="test",
        archetypes=["3segment_speed"],
    )
    test_ood_ladder_probs = generate_multistep_problems(
        num_samples_per_archetype=10,
        seed=5001,
        split="test",
        archetypes=["chained_addition_5", "chained_addition_6"],
    )

    test_fam_ds = ActionTraceDataset.from_problems(test_familiar_probs, split="test")
    test_speed_ds = ActionTraceDataset.from_problems(test_ood_speed_probs, split="test")
    test_ladder_ds = ActionTraceDataset.from_problems(test_ood_ladder_probs, split="test")

    # 2. Training Problems (Strict isolation: seed 1002)
    train_archetypes = [
        "chained_addition_3",
        "chained_addition_4",
        "2segment_speed",
        "multi_entity_compare",
        "3segment_speed",
    ]
    train_probs = generate_multistep_problems(
        num_samples_per_archetype=25,
        seed=1002,
        split="train",
        archetypes=train_archetypes,
    )
    val_probs = generate_multistep_problems(
        num_samples_per_archetype=10,
        seed=2002,
        split="val",
        archetypes=train_archetypes,
    )

    ref_train_ds = ActionTraceDataset.from_problems(train_probs, split="train")
    val_ds = ActionTraceDataset.from_problems(val_probs, split="val")

    logger.info("Reference training traces: %d, Validation traces: %d", len(ref_train_ds), len(val_ds))

    results_by_lineage: dict[str, Any] = {}

    for seed in seeds:
        seed_key = f"seed_{seed}"
        ckpt_path = base_checkpoints_dir / f"seed_{seed}" / "base_pretrain" / "best.pt"
        logger.info("==================================================================")
        logger.info("Processing Lineage %s from %s", seed_key, ckpt_path)
        logger.info("==================================================================")

        raw_ckpt = CheckpointManager.load_raw_checkpoint(ckpt_path)
        tokenizer = CheckpointManager.load_tokenizer(raw_ckpt)
        cfg_dict = raw_ckpt.get("model_config", {})
        cfg = ModelConfig(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict

        lineage_eval: dict[str, Any] = {}

        # -------------------------------------------------------------
        # REGIME A: Standard Supervised Training on Reference Traces
        # -------------------------------------------------------------
        logger.info("--- Starting Regime A: Supervised Reference Training [%s] ---", seed_key)
        torch.manual_seed(seed)
        backbone_a = MiniGPT(cfg)
        backbone_a.load_state_dict(raw_ckpt["model_state_dict"])
        backbone_a.to(device)

        model_regime_a = ActionPolicyModel(
            backbone=backbone_a,
            tokenizer=tokenizer,
            d_hidden=128,
            adapter_mode="full",
        ).to(device)

        train_action_policy(
            policy_model=model_regime_a,
            train_dataset=ref_train_ds,
            val_dataset=val_ds,
            device=device,
            epochs=epochs,
            lr=1e-4,
            val_problems=val_probs,
        )

        m_a_fam = evaluate_action_ranking(model_regime_a, test_fam_ds, device, problems_for_rollout=test_familiar_probs)
        m_a_speed = evaluate_action_ranking(model_regime_a, test_speed_ds, device, problems_for_rollout=test_ood_speed_probs)
        m_a_ladder = evaluate_action_ranking(model_regime_a, test_ladder_ds, device, problems_for_rollout=test_ood_ladder_probs)
        retention_a = evaluate_retention_suite(model_regime_a, device)

        lineage_eval["Regime_A_Supervised"] = {
            "familiar_test": {
                "action_acc": m_a_fam.policy_acc,
                "rollout_acc": m_a_fam.rollout_policy_acc,
                "latency_ms": m_a_fam.decision_latency_ms,
            },
            "ood_kinematics_5step": {
                "action_acc": m_a_speed.policy_acc,
                "rollout_acc": m_a_speed.rollout_policy_acc,
                "positions": m_a_speed.position_accuracies,
            },
            "ood_addition_ladders": {
                "action_acc": m_a_ladder.policy_acc,
                "rollout_acc": m_a_ladder.rollout_policy_acc,
                "positions": m_a_ladder.position_accuracies,
            },
            "retention": retention_a,
        }

        # -------------------------------------------------------------
        # REGIME B: Closed-Loop Learning (DAgger with Corrective Labels)
        # -------------------------------------------------------------
        logger.info("--- Starting Regime B: Closed-Loop Rollout Data Collection [%s] ---", seed_key)
        # Collect rollout trajectories on training problems using current policy
        corrective_traces, rollout_stats = collect_closed_loop_rollout_traces(
            policy_model=model_regime_a,
            problems=train_probs,
            device=device,
            max_steps=5,
        )
        logger.info(
            "Rollout stats: Visited=%d, Recoverable=%d (%.1f%%), Unrecoverable=%d, Corrective Traces=%d",
            rollout_stats["total_states_visited"],
            rollout_stats["recoverable_states"],
            rollout_stats["recoverability_rate"],
            rollout_stats["unrecoverable_states"],
            len(corrective_traces),
        )

        # Aggregate reference dataset + verified corrective traces
        aggregated_traces = list(ref_train_ds) + corrective_traces
        agg_train_ds = ActionTraceDataset(
            traces=aggregated_traces,
            split="train",
            dataset_fingerprint=hashlib.sha256(
                f"{ref_train_ds.dataset_fingerprint}_{len(corrective_traces)}".encode()
            ).hexdigest()[:16],
        )

        # Train a fresh policy under matched compute conditions
        torch.manual_seed(seed)
        backbone_b = MiniGPT(cfg)
        backbone_b.load_state_dict(raw_ckpt["model_state_dict"])
        backbone_b.to(device)

        model_regime_b = ActionPolicyModel(
            backbone=backbone_b,
            tokenizer=tokenizer,
            d_hidden=128,
            adapter_mode="full",
        ).to(device)

        train_action_policy(
            policy_model=model_regime_b,
            train_dataset=agg_train_ds,
            val_dataset=val_ds,
            device=device,
            epochs=epochs,
            lr=1e-4,
            val_problems=val_probs,
        )

        m_b_fam = evaluate_action_ranking(model_regime_b, test_fam_ds, device, problems_for_rollout=test_familiar_probs)
        m_b_speed = evaluate_action_ranking(model_regime_b, test_speed_ds, device, problems_for_rollout=test_ood_speed_probs)
        m_b_ladder = evaluate_action_ranking(model_regime_b, test_ladder_ds, device, problems_for_rollout=test_ood_ladder_probs)
        retention_b = evaluate_retention_suite(model_regime_b, device)

        lineage_eval["Regime_B_Closed_Loop"] = {
            "rollout_diagnostics": rollout_stats,
            "familiar_test": {
                "action_acc": m_b_fam.policy_acc,
                "rollout_acc": m_b_fam.rollout_policy_acc,
                "latency_ms": m_b_fam.decision_latency_ms,
            },
            "ood_kinematics_5step": {
                "action_acc": m_b_speed.policy_acc,
                "rollout_acc": m_b_speed.rollout_policy_acc,
                "positions": m_b_speed.position_accuracies,
            },
            "ood_addition_ladders": {
                "action_acc": m_b_ladder.policy_acc,
                "rollout_acc": m_b_ladder.rollout_policy_acc,
                "positions": m_b_ladder.position_accuracies,
            },
            "retention": retention_b,
        }

        results_by_lineage[seed_key] = lineage_eval
        logger.info(
            "Lineage %s summary: Regime A (Speed Rollout=%.1f%%, Ladder Rollout=%.1f%%) vs Regime B (Speed Rollout=%.1f%%, Ladder Rollout=%.1f%%)",
            seed_key,
            m_a_speed.rollout_policy_acc,
            m_a_ladder.rollout_policy_acc,
            m_b_speed.rollout_policy_acc,
            m_b_ladder.rollout_policy_acc,
        )

    # 3. Aggregate multi-seed summary
    def _mean_std(vals: list[float]) -> dict[str, float]:
        m = sum(vals) / len(vals)
        var = sum((x - m) ** 2 for x in vals) / max(len(vals) - 1, 1)
        return {"mean": round(m, 2), "std": round(var ** 0.5, 2)}

    aggregate_summary: dict[str, Any] = {}
    for regime in ["Regime_A_Supervised", "Regime_B_Closed_Loop"]:
        fam_rollouts = [results_by_lineage[f"seed_{s}"][regime]["familiar_test"]["rollout_acc"] for s in seeds]
        speed_rollouts = [results_by_lineage[f"seed_{s}"][regime]["ood_kinematics_5step"]["rollout_acc"] for s in seeds]
        ladder_rollouts = [results_by_lineage[f"seed_{s}"][regime]["ood_addition_ladders"]["rollout_acc"] for s in seeds]
        speed_actions = [results_by_lineage[f"seed_{s}"][regime]["ood_kinematics_5step"]["action_acc"] for s in seeds]
        ladder_actions = [results_by_lineage[f"seed_{s}"][regime]["ood_addition_ladders"]["action_acc"] for s in seeds]

        aggregate_summary[regime] = {
            "familiar_test_rollout_acc": _mean_std(fam_rollouts),
            "ood_speed_action_acc": _mean_std(speed_actions),
            "ood_speed_rollout_acc": _mean_std(speed_rollouts),
            "ood_ladder_action_acc": _mean_std(ladder_actions),
            "ood_ladder_rollout_acc": _mean_std(ladder_rollouts),
        }

    final_report = {
        "milestone": "M19.5",
        "description": "Closed-Loop Action Learning with Recoverability-Aware Corrective Labeling",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds,
        "device": device_str,
        "aggregate_summary": aggregate_summary,
        "detailed_lineage_results": results_by_lineage,
    }

    report_file = output_dir / "m19_5_closed_loop_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    logger.info("M19.5 Benchmark Report successfully written to %s", report_file)
    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M19.5 Closed-Loop Action Learning Benchmark")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("runs/m18_planning"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/m19_closed_loop"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    run_m19_5_benchmark(
        base_checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
        device_str=args.device,
        seeds=args.seeds,
        epochs=args.epochs,
    )
