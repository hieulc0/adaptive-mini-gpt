"""Verified Action Trace Dataset and Extraction Engine (M19.1).

Extracts state-action tuples from verified multi-step reasoning plans:
    (q, s_t, A_valid) -> a_t*

Guarantees:
- Derives training labels ONLY from plans that independently solve the target goal.
- Equivalence-aware: Recognizes semantically equivalent next actions (commutative addition,
  parallel independent DAG steps) without penalizing valid alternatives.
- Strict train/val/test split isolation: 0% data leakage into benchmark test splits.
- Zero answer leakage: Reference plans and expected answers are never stored in the traces.
- Cryptographic reproducibility and SHA-256 dataset fingerprinting.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.rules.executor import RuleExecutor
from adaptive_mini_gpt.rules.grammar_planner import PlanGrammar, TargetGoal
from adaptive_mini_gpt.rules.memory import GroundedSlot, WorkingMemory
from adaptive_mini_gpt.rules.multistep_dataset import (
    DEFAULT_ARCHETYPES,
    MultiStepProblem,
    format_available_slots,
    generate_multistep_problems,
    get_multistep_split,
)
from adaptive_mini_gpt.rules.plan import ExecutionPlan, PlanStep

logger = logging.getLogger(__name__)


def get_slot_dependencies(
    arg_val: Any,
    var_slot_deps: dict[str, frozenset[str]],
    slots: dict[str, GroundedSlot],
) -> frozenset[str]:
    """Recursively extract the set of grounded input slot IDs that an argument depends upon."""
    if isinstance(arg_val, str) and arg_val.startswith("$"):
        ref = arg_val[1:].strip()
        if ref in slots:
            return frozenset([ref])
        if ref in var_slot_deps:
            return var_slot_deps[ref]
    return frozenset()


def step_canonical_signature(
    step: PlanStep,
    var_slot_deps: dict[str, frozenset[str]],
    slots: dict[str, GroundedSlot],
) -> tuple[Any, ...]:
    """Compute a canonical, equivalence-invariant structural signature for an operation step.
    
    Invariance guarantees:
    - Commutative addition: add(a, b) and add(b, a) yield identical signatures.
    - Slot provenance: tracks underlying input slots, independent of temporary variable names.
    """
    op = step.operation
    u = step.units
    if op == "add":
        d_a = get_slot_dependencies(step.args.get("a"), var_slot_deps, slots)
        d_b = get_slot_dependencies(step.args.get("b"), var_slot_deps, slots)
        return (op, u, frozenset([d_a, d_b]))
    elif op == "average_speed":
        d_dist = get_slot_dependencies(step.args.get("distance_km"), var_slot_deps, slots)
        d_time = get_slot_dependencies(step.args.get("time_hours"), var_slot_deps, slots)
        return (op, u, d_dist, d_time)
    elif op == "compare":
        return (
            op,
            u,
            str(step.args.get("entity_a")),
            str(step.args.get("entity_b")),
            tuple(sorted(step.args.get("relations", []))),
        )
    elif op == "reverse":
        d_text = get_slot_dependencies(step.args.get("text"), var_slot_deps, slots)
        return (op, u, d_text)
    return (op, u, tuple(sorted(step.args.items())))


@dataclass
class ActionTrace:
    """A single decision step in a verified multi-step reasoning trajectory.
    
    Represents: (question, current_state, valid_candidates) -> correct_action_indices
    """

    trace_id: str
    problem_key: str
    question: str
    slots: dict[str, GroundedSlot]
    executed_steps: list[PlanStep]
    valid_candidates: list[PlanStep]
    correct_action_indices: list[int]
    target_action_index: int
    step_index: int
    total_plan_steps: int
    archetype: str
    split: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "problem_key": self.problem_key,
            "question": self.question,
            "slots": {k: s.to_dict() for k, s in self.slots.items()},
            "executed_steps": [s.to_dict() for s in self.executed_steps],
            "valid_candidates": [s.to_dict() for s in self.valid_candidates],
            "correct_action_indices": self.correct_action_indices,
            "target_action_index": self.target_action_index,
            "step_index": self.step_index,
            "total_plan_steps": self.total_plan_steps,
            "archetype": self.archetype,
            "split": self.split,
        }

    def format_context_prompt(self) -> str:
        """Format the state prefix (q, s_t) for causal Transformer ingestion."""
        slots_text = format_available_slots(self.slots).strip()
        user_body = f"{self.question}\n{slots_text}" if slots_text else self.question
        lines = [f"<|user|>\n{user_body}\n<|assistant|>\nPlan:"]
        for idx, s in enumerate(self.executed_steps, start=1):
            u_str = f" [{s.units}]" if s.units else ""
            args_formatted = ", ".join(
                f"{k}={repr(v) if isinstance(v, str) else v}" for k, v in s.args.items()
            )
            lines.append(f"{idx}. {s.output_var} = {s.operation}({args_formatted}){u_str}")

        next_idx = len(self.executed_steps) + 1
        return "\n".join(lines) + f"\n{next_idx}. "


class ActionTraceDataset:
    """Dataset of verified state-action reasoning traces with strict split isolation."""

    def __init__(
        self,
        traces: list[ActionTrace],
        split: str = "train",
        dataset_fingerprint: str = "",
    ) -> None:
        self.traces = traces
        self.split = split
        self.dataset_fingerprint = dataset_fingerprint or self._compute_fingerprint()

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> ActionTrace:
        return self.traces[idx]

    def _compute_fingerprint(self) -> str:
        """Generate SHA-256 fingerprint from immutable trace IDs and signatures."""
        h = hashlib.sha256()
        for t in self.traces:
            h.update(f"{t.trace_id}:{t.step_index}:{sorted(t.correct_action_indices)}".encode())
        return h.hexdigest()[:16]

    @classmethod
    def from_problems(
        cls,
        problems: list[MultiStepProblem],
        split: str = "train",
        executor: RuleExecutor | None = None,
    ) -> ActionTraceDataset:
        """Extract verified traces from a collection of multi-step problems.
        
        Guarantees:
        - Only extracts from problems belonging strictly to the requested `split`.
        - Rejects malformed / abstention problems (no valid execution plan).
        - Verifies that the plan independently achieves the TargetGoal before trace extraction.
        - Identifies all semantically equivalent next actions at each step.
        """
        rule_executor = executor or RuleExecutor()
        traces: list[ActionTrace] = []

        for p_idx, prob in enumerate(problems):
            # 1. Enforce split partition check
            if prob.split != split:
                continue

            # 2. Skip abstention or malformed problems (no valid execution plan)
            if prob.should_abstain or prob.is_malformed or prob.expected_slot_plan is None:
                continue

            ref_plan = prob.expected_slot_plan
            slots = prob.slots
            goal = TargetGoal.from_question(prob.question, slots)

            # 3. Pre-verify reference plan semantics independently
            sim_mem = WorkingMemory()
            sim_mem.slots = copy.copy(slots)
            plan_valid = True
            for s in ref_plan.steps:
                resolved: dict[str, Any] = {}
                for k, v in s.args.items():
                    if isinstance(v, str) and v.startswith("$"):
                        ref_name = v[1:].strip()
                        resolved[k] = sim_mem.variables.get(
                            ref_name, sim_mem.slots[ref_name].value if ref_name in sim_mem.slots else None
                        )
                    else:
                        resolved[k] = v
                exec_res = rule_executor.execute(s.operation, resolved, verify=True)
                if not exec_res.success or not exec_res.verified:
                    plan_valid = False
                    break
                sim_mem.bind_step_variable(
                    var_name=s.output_var,
                    value=exec_res.result,
                    operation=s.operation,
                    raw_args=s.args,
                    resolved_args=resolved,
                    units=s.units,
                    step_index=len(sim_mem.variables),
                )
            if not plan_valid:
                continue
            sim_mem.bind_result(sim_mem.variables.get(ref_plan.final_var or ""), verified=True)
            sem_ok, _ = goal.evaluate_semantics(ref_plan, sim_mem)
            if not sem_ok:
                continue

            # 4. Extract step signatures of all remaining required steps in the DAG
            ref_var_slot_deps: dict[str, frozenset[str]] = {}
            ref_step_sigs: list[tuple[Any, ...]] = []
            for s in ref_plan.steps:
                sig = step_canonical_signature(s, ref_var_slot_deps, slots)
                ref_step_sigs.append(sig)
                step_deps: set[str] = set()
                for v in s.args.values():
                    step_deps.update(get_slot_dependencies(v, ref_var_slot_deps, slots))
                ref_var_slot_deps[s.output_var] = frozenset(step_deps)

            # 5. Stepwise simulation to extract state-action tuples
            live_mem = WorkingMemory()
            live_mem.slots = copy.copy(slots)
            executed_steps: list[PlanStep] = []
            live_var_slot_deps: dict[str, frozenset[str]] = {}
            active_terms: set[str] = set(slots.keys())

            total_steps = len(ref_plan.steps)
            for step_idx, target_step in enumerate(ref_plan.steps):
                rem_steps = total_steps - step_idx

                # Candidate actions generated by PlanGrammar from current state
                candidates = PlanGrammar.get_candidate_steps(
                    slots=slots,
                    working_memory=live_mem,
                    executed_steps=executed_steps,
                    max_steps=total_steps,
                    question=prob.question,
                )

                if not candidates:
                    break

                remaining_ref_sigs = set(ref_step_sigs[step_idx:])
                target_sig = ref_step_sigs[step_idx]

                correct_indices: list[int] = []
                target_cand_idx = -1

                for c_idx, cand in enumerate(candidates):
                    cand_sig = step_canonical_signature(cand, live_var_slot_deps, slots)
                    # Case A: Exact canonical match to current target step
                    if cand_sig == target_sig:
                        if target_cand_idx == -1:
                            target_cand_idx = c_idx
                        correct_indices.append(c_idx)
                        continue

                    # Case B: Parallel DAG step remaining in the reference plan
                    if cand_sig in remaining_ref_sigs:
                        correct_indices.append(c_idx)
                        continue

                    # Case C: Commutative / topological equivalence check for sum goals
                    if goal.goal_type == "sum" and cand.operation == "add":
                        a_ref = str(cand.args.get("a", "")).lstrip("$")
                        b_ref = str(cand.args.get("b", "")).lstrip("$")
                        if a_ref in active_terms and b_ref in active_terms:
                            c_a = get_slot_dependencies(cand.args.get("a"), live_var_slot_deps, slots)
                            c_b = get_slot_dependencies(cand.args.get("b"), live_var_slot_deps, slots)
                            req_ops = goal.required_slots or {s for s in slots if s.startswith("operand_")}
                            if c_a and c_b and not (c_a & c_b) and (c_a | c_b).issubset(req_ops):
                                correct_indices.append(c_idx)
                        continue

                    # Case D: Kinematics (speed) independent additions
                    if goal.goal_type == "speed":
                        if rem_steps > 1 and cand.operation == "add":
                            a_ref = str(cand.args.get("a", "")).lstrip("$")
                            b_ref = str(cand.args.get("b", "")).lstrip("$")
                            if a_ref in active_terms and b_ref in active_terms:
                                c_a = get_slot_dependencies(cand.args.get("a"), live_var_slot_deps, slots)
                                c_b = get_slot_dependencies(cand.args.get("b"), live_var_slot_deps, slots)
                                if c_a and c_b and not (c_a & c_b):
                                    dist_slots = {s for s in slots if s.startswith("distance_")}
                                    dur_slots = {s for s in slots if s.startswith("duration_")}
                                    if (c_a | c_b).issubset(dist_slots) or (c_a | c_b).issubset(dur_slots):
                                        correct_indices.append(c_idx)
                        elif rem_steps == 1 and cand.operation == "average_speed":
                            d_ref = str(cand.args.get("distance_km", "")).lstrip("$")
                            t_ref = str(cand.args.get("time_hours", "")).lstrip("$")
                            if d_ref in active_terms and t_ref in active_terms:
                                c_dist = get_slot_dependencies(cand.args.get("distance_km"), live_var_slot_deps, slots)
                                c_dur = get_slot_dependencies(cand.args.get("time_hours"), live_var_slot_deps, slots)
                                dist_slots = {s for s in slots if s.startswith("distance_")}
                                dur_slots = {s for s in slots if s.startswith("duration_")}
                                if c_dist == dist_slots and c_dur == dur_slots:
                                    correct_indices.append(c_idx)

                correct_indices = sorted(list(set(correct_indices)))
                if target_cand_idx == -1 and correct_indices:
                    target_cand_idx = correct_indices[0]

                if target_cand_idx != -1:
                    prob_key = f"{prob.archetype}_{p_idx}_{prob.split}"
                    trace = ActionTrace(
                        trace_id=f"trace_{prob_key}_step_{step_idx + 1}",
                        problem_key=prob_key,
                        question=prob.question,
                        slots=copy.copy(slots),
                        executed_steps=list(executed_steps),
                        valid_candidates=list(candidates),
                        correct_action_indices=correct_indices,
                        target_action_index=target_cand_idx,
                        step_index=step_idx + 1,
                        total_plan_steps=total_steps,
                        archetype=prob.archetype,
                        split=split,
                    )
                    traces.append(trace)

                # Advance live execution state with the canonical target step
                resolved_live: dict[str, Any] = {}
                for k, v in target_step.args.items():
                    if isinstance(v, str) and v.startswith("$"):
                        r_name = v[1:].strip()
                        resolved_live[k] = live_mem.variables.get(
                            r_name, live_mem.slots[r_name].value if r_name in live_mem.slots else None
                        )
                        active_terms.discard(r_name)
                    else:
                        resolved_live[k] = v

                exec_res = rule_executor.execute(target_step.operation, resolved_live, verify=True)
                live_mem.bind_step_variable(
                    var_name=target_step.output_var,
                    value=exec_res.result,
                    operation=target_step.operation,
                    raw_args=target_step.args,
                    resolved_args=resolved_live,
                    units=target_step.units,
                    step_index=len(executed_steps) + 1,
                )
                executed_steps.append(target_step)
                active_terms.add(target_step.output_var)

                step_deps = set()
                for v in target_step.args.values():
                    step_deps.update(get_slot_dependencies(v, live_var_slot_deps, slots))
                live_var_slot_deps[target_step.output_var] = frozenset(step_deps)

        return cls(traces=traces, split=split)


def extract_verified_m19_traces(
    num_samples_per_archetype: int = 40,
    seed: int = 42,
    held_out_archetypes: list[str] | None = None,
) -> tuple[ActionTraceDataset, ActionTraceDataset, ActionTraceDataset]:
    """Generate strictly partitioned ActionTraceDatasets for Train, Validation, and Held-out OOD.
    
    Partition strategy:
    - Train dataset: split="train" on familiar archetypes (e.g. chained addition, 2-segment speed, comparison).
    - Val dataset: split="val" on familiar archetypes for model checkpoint selection.
    - OOD / Held-out structure dataset: unseen composition family (e.g. 3-segment speed / deep ladders).
    """
    all_archs = list(DEFAULT_ARCHETYPES)
    held_out = set(held_out_archetypes or ["3segment_speed"])
    familiar_archs = [a for a in all_archs if a not in held_out]

    # Generate problems
    train_problems = generate_multistep_problems(
        num_samples_per_archetype=num_samples_per_archetype,
        seed=seed,
        split="train",
        archetypes=familiar_archs,
    )
    val_problems = generate_multistep_problems(
        num_samples_per_archetype=max(5, num_samples_per_archetype // 4),
        seed=seed + 100,
        split="val",
        archetypes=familiar_archs,
    )
    ood_problems = generate_multistep_problems(
        num_samples_per_archetype=max(5, num_samples_per_archetype // 4),
        seed=seed + 200,
        split="test",
        archetypes=list(held_out),
    )

    train_ds = ActionTraceDataset.from_problems(train_problems, split="train")
    val_ds = ActionTraceDataset.from_problems(val_problems, split="val")
    ood_ds = ActionTraceDataset.from_problems(ood_problems, split="test")

    logger.info(
        "M19.1 Action Traces extracted: Train=%d traces, Val=%d traces, Held-out OOD (%s)=%d traces",
        len(train_ds),
        len(val_ds),
        list(held_out),
        len(ood_ds),
    )
    return train_ds, val_ds, ood_ds
