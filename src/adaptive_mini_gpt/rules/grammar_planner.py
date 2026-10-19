"""Grammar-Constrained, Execution-Guided Planning Engine (M18).

Implements bounded, stepwise plan construction with isolated branch memory:
1. PlanGrammar: Strict type-, unit-, and dependency-constrained next-step candidate generation.
   - Restricts candidates to registered rules, available slots, and verified intermediate variables.
   - Enforces Single Static Assignment (SSA) and rejects forward/cyclic references.
   - Prunes dimensionally incompatible operations (e.g. adding distance to duration).
   - Eliminates redundant self-accumulation and duplicate operations.
2. Isolated Search State & Frontier:
   - Deep-copied, isolated WorkingMemory per search branch.
   - Rejected branches leave zero trace in active memory or sibling branches.
   - Bounded search budget (configurable beam width, max expansions, max depth).
   - Loop and state deduplication to prevent cyclical exploration.
   - Explicit failure upon search budget exhaustion.
3. TargetGoal & Semantic Evaluation:
   - Decouples structural plan validity (executable DAG) from semantic correctness (answering the question).
   - Verifies transitive slot dependencies, final unit compatibility, and expected target types.
"""

from __future__ import annotations

import copy
import heapq
import itertools
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from adaptive_mini_gpt.rules.executor import RuleExecutor
from adaptive_mini_gpt.rules.memory import GroundedSlot, WorkingMemory
from adaptive_mini_gpt.rules.plan import (
    DependencyResolutionError,
    ExecutionPlan,
    PlanExecutionResult,
    PlanStep,
    PlanValidationError,
)
from adaptive_mini_gpt.rules.registry import RULE_REGISTRY

logger = logging.getLogger(__name__)


class SearchBudgetExhaustedError(RuntimeError):
    """Raised when search space or expansion budget is exhausted without reaching a valid goal."""


@dataclass
class TargetGoal:
    """Runtime goal specification decoupled from benchmark ground-truth answers.
    
    Guarantees runtime isolation:
    - Never accesses expected numerical answers or reference plans during search.
    - Evaluates structural preconditions, output units, and transitive slot coverage.
    """

    goal_type: str  # "speed", "sum", "compare", "reverse", "generic"
    required_slots: set[str] = field(default_factory=set)
    expected_unit: str | None = None

    @classmethod
    def from_question(
        cls,
        question: str,
        slots: dict[str, GroundedSlot] | None = None,
    ) -> TargetGoal:
        """Infer target goal semantics strictly from question text and available slots."""
        q_lower = question.lower()
        available_slots = set(slots.keys()) if slots else set()

        if "average speed" in q_lower or (
            ("travel" in q_lower or "cover" in q_lower or "drive" in q_lower or "fly" in q_lower)
            and "km" in q_lower
            and ("hour" in q_lower or "hrs" in q_lower or "h" in q_lower)
        ):
            kin_slots = {s for s in available_slots if s.startswith("distance_") or s.startswith("duration_")}
            return cls(
                goal_type="speed",
                required_slots=kin_slots,
                expected_unit="km/h",
            )

        if any(kw in q_lower for kw in ("sum", "total", "plus", "add")) and "than" not in q_lower:
            op_slots = {s for s in available_slots if s.startswith("operand_")}
            return cls(
                goal_type="sum",
                required_slots=op_slots,
                expected_unit=None,
            )

        if any(kw in q_lower for kw in ("taller", "faster", "heavier", "older", "shorter")):
            ent_slots = {s for s in available_slots if s.startswith("entity_")}
            return cls(
                goal_type="compare",
                required_slots=ent_slots,
                expected_unit=None,
            )

        if any(kw in q_lower for kw in ("reverse", "backward", "invert")):
            return cls(
                goal_type="reverse",
                required_slots=available_slots,
                expected_unit=None,
            )

        return cls(
            goal_type="generic",
            required_slots=available_slots,
            expected_unit=None,
        )

    def evaluate_semantics(
        self,
        plan: ExecutionPlan,
        working_memory: WorkingMemory,
    ) -> tuple[bool, str]:
        """Check whether the executed plan satisfies the semantic goal without expected answers.
        
        Returns:
            (is_semantically_correct, diagnostic_message)
        """
        if not working_memory.verified:
            return False, "Plan did not finish with verified status in working memory"

        if not plan.final_var or plan.final_var not in working_memory.variables:
            return False, f"Final variable '{plan.final_var}' not found in working memory"

        final_val = working_memory.variables[plan.final_var]
        final_unit = working_memory.step_units.get(plan.final_var)

        # 1. Check unit requirements
        if self.expected_unit is not None:
            if final_unit != self.expected_unit:
                if not (isinstance(final_val, str) and final_val.endswith(self.expected_unit)):
                    return False, f"Final unit mismatch: expected '{self.expected_unit}', got '{final_unit}'"

        # 2. Transitive slot dependency completeness check
        if self.required_slots:
            var_deps: dict[str, set[str]] = {}
            for step in plan.steps:
                deps: set[str] = set()
                for v in step.args.values():
                    if isinstance(v, str) and v.startswith("$"):
                        ref = v[1:].strip()
                        deps.add(ref)
                        if ref in var_deps:
                            deps.update(var_deps[ref])
                    elif isinstance(v, str):
                        # Match literal entity names or values
                        for sid, slot in working_memory.slots.items():
                            if str(slot.value) == v or sid == v:
                                deps.add(sid)
                    elif isinstance(v, list):
                        for item in v:
                            for sid, slot in working_memory.slots.items():
                                if str(slot.value) in str(item) or sid in str(item):
                                    deps.add(sid)
                var_deps[step.output_var] = deps

            final_deps = var_deps.get(plan.final_var, set())
            missing_slots = self.required_slots - final_deps
            if missing_slots:
                return (
                    False,
                    f"Semantically incomplete plan: final answer does not depend on required slots {sorted(missing_slots)}",
                )

        return True, "Plan satisfies semantic goal"


class PlanGrammar:
    """Grammar and type validator for generating valid candidate plan steps.
    
    Guarantees:
    - Never generates invalid operations or unknown rules.
    - Never references undefined variables or forward variables.
    - Eliminates self-accumulation and duplicate operations.
    - Prioritizes slot coverage and structural building blocks.
    """

    @classmethod
    def get_candidate_steps(
        cls,
        slots: dict[str, GroundedSlot],
        working_memory: WorkingMemory,
        executed_steps: list[PlanStep],
        max_steps: int = 5,
        question: str = "",
    ) -> list[PlanStep]:
        """Generate all grammatically valid, type-safe next candidate steps from the current state."""
        if len(executed_steps) >= max_steps:
            return []

        # 1. Build transitive dependency map for all previously executed steps
        var_deps: dict[str, set[str]] = {}
        for s in executed_steps:
            d: set[str] = set()
            for v in s.args.values():
                if isinstance(v, str) and v.startswith("$"):
                    ref = v[1:].strip()
                    if ref in slots:
                        d.add(ref)
                    elif ref in var_deps:
                        d.update(var_deps[ref])
                elif isinstance(v, str):
                    for sid, slot in slots.items():
                        if str(slot.value) == v or sid == v:
                            d.add(sid)
            var_deps[s.output_var] = d

        def get_ref_deps(ref_str: str) -> set[str]:
            if ref_str.startswith("$"):
                name = ref_str[1:].strip()
                if name in slots:
                    return {name}
                if name in var_deps:
                    return var_deps[name]
            return set()

        # 2. Track executed step signatures to avoid duplicate exploration
        executed_add_pairs: set[frozenset[str]] = set()
        executed_speed_pairs: set[tuple[str, str]] = set()
        for s in executed_steps:
            if s.operation == "add":
                vals = [str(v) for v in s.args.values()]
                executed_add_pairs.add(frozenset(vals))
            elif s.operation == "average_speed":
                d_val = str(s.args.get("distance_km", ""))
                t_val = str(s.args.get("time_hours", ""))
                executed_speed_pairs.add((d_val, t_val))

        next_step_idx = len(executed_steps) + 1

        # Collect available variable and slot references: (ref_id, value, unit, type_name)
        available_refs: list[tuple[str, Any, str | None, str]] = []
        for sid, slot in slots.items():
            t_name = slot.type_name or (type(slot.value).__name__)
            available_refs.append((f"${sid}", slot.value, slot.unit, t_name))

        for var_name, var_val in working_memory.variables.items():
            u = working_memory.step_units.get(var_name)
            t_name = type(var_val).__name__
            available_refs.append((f"${var_name}", var_val, u, t_name))

        # Output variable name based on operation convention
        add_out_var = f"sum_{next_step_idx}" if not any(s.startswith("distance") or s.startswith("duration") for s in slots) else f"step_{next_step_idx}"

        candidates_with_meta: list[tuple[int, int, PlanStep]] = []

        # -------------------------------------------------------------
        # Rule 1: 'add' (arithmetic addition)
        # -------------------------------------------------------------
        numeric_refs = [
            r for r in available_refs if isinstance(r[1], (int, float)) and r[3] in ("int", "float")
        ]
        unit_groups: dict[str | None, list[tuple[str, Any, str | None, str]]] = {}
        for r in numeric_refs:
            unit_groups.setdefault(r[2], []).append(r)

        for u, refs in unit_groups.items():
            for i in range(len(refs)):
                for j in range(i + 1, len(refs)):
                    r1, r2 = refs[i], refs[j]
                    if r1[0] == r2[0]:
                        continue
                    # Prune 1: Never duplicate exact executed addition
                    if frozenset((r1[0], r2[0])) in executed_add_pairs:
                        continue
                    # Prune 2: Disjoint slot dependencies (no self-accumulation or double-counting operands)
                    d1 = get_ref_deps(r1[0])
                    d2 = get_ref_deps(r2[0])
                    if d1 and d2 and (d1 & d2):
                        continue

                    # Canonical order: intermediate variables first
                    if (r2[0].startswith("$step_") or r2[0].startswith("$sum_")) and not (r1[0].startswith("$step_") or r1[0].startswith("$sum_")):
                        r1, r2 = r2, r1

                    cand_args = {"a": r1[0], "b": r2[0]}
                    cov = len(d1 | d2)
                    step_obj = PlanStep(
                        output_var=add_out_var,
                        operation="add",
                        args=cand_args,
                        units=u,
                    )
                    # Normal rank 0
                    candidates_with_meta.append((cov, 0, step_obj))

        # -------------------------------------------------------------
        # Rule 2: 'average_speed' (kinematics sink operation)
        # -------------------------------------------------------------
        dist_refs = [
            r for r in numeric_refs
            if (r[2] in ("km", "kilometer", "kilometers") or "distance" in r[0] or "d_" in r[0])
            and r[1] >= 0
        ]
        time_refs = [
            r for r in numeric_refs
            if (r[2] in ("h", "hour", "hours", "hrs") or "duration" in r[0] or "t_" in r[0])
            and r[1] > 0
        ]
        all_dist_slots = {s for s, slot in slots.items() if slot.unit in ("km", "kilometer", "kilometers") or "distance" in s}
        all_time_slots = {s for s, slot in slots.items() if slot.unit in ("h", "hour", "hours", "hrs") or "duration" in s}

        for d_ref in dist_refs:
            for t_ref in time_refs:
                if (d_ref[0], t_ref[0]) in executed_speed_pairs:
                    continue
                cand_args = {"distance_km": d_ref[0], "time_hours": t_ref[0]}
                d1 = get_ref_deps(d_ref[0])
                d2 = get_ref_deps(t_ref[0])
                cov = len(d1 | d2)
                has_unconsumed = bool((all_dist_slots - d1) or (all_time_slots - d2))
                # Terminal sink (covering all slots) gets highest priority rank -1; partial sink gets lower priority 1
                rank = 1 if has_unconsumed else -1
                step_obj = PlanStep(
                    output_var=f"speed_{next_step_idx}",
                    operation="average_speed",
                    args=cand_args,
                    units="km/h",
                )
                candidates_with_meta.append((cov, rank, step_obj))

        # -------------------------------------------------------------
        # Rule 3: 'compare' (multi-entity comparison chain)
        # -------------------------------------------------------------
        entity_refs = [r for r in available_refs if r[3] == "str" and "entity" in r[0]]
        if len(entity_refs) >= 2 and not executed_steps:
            adj_pairs = {"taller": "shorter", "faster": "slower", "heavier": "lighter", "older": "younger"}
            all_adjs = list(adj_pairs.keys()) + list(adj_pairs.values())
            adj_pat = "|".join(all_adjs)
            ent_pat = r"([A-Z][a-zA-Z]*)"
            prem_matches = list(re.finditer(rf"{ent_pat}\s+is\s+({adj_pat})\s+than\s+{ent_pat}", question))
            if len(prem_matches) >= 2:
                relations = []
                for pm in prem_matches:
                    e1, adj, e2 = pm.group(1).strip(), pm.group(2).strip().lower(), pm.group(3).strip()
                    if adj in adj_pairs:
                        relations.append(f"{e1} > {e2}")
                    else:
                        relations.append(f"{e2} > {e1}")
                tm = re.search(rf"(?:who|which one)\s+is\s+(?:[a-zA-Z]+),\s*{ent_pat}\s+or\s+{ent_pat}", question, re.IGNORECASE)
                if tm:
                    ta, tb = tm.group(1).strip(), tm.group(2).strip()
                    step_obj = PlanStep(
                        output_var="comparison_result",
                        operation="compare",
                        args={"entity_a": ta, "entity_b": tb, "relations": relations},
                    )
                    candidates_with_meta.append((len(entity_refs), -2, step_obj))

        # -------------------------------------------------------------
        # Rule 4: 'reverse' (string reversal)
        # -------------------------------------------------------------
        text_refs = [r for r in available_refs if isinstance(r[1], str) and r[3] == "str" and not r[0].startswith("$entity")]
        for t_ref in text_refs:
            step_obj = PlanStep(
                output_var=f"reversed_{next_step_idx}",
                operation="reverse",
                args={"text": t_ref[0]},
            )
            candidates_with_meta.append((1, 0, step_obj))

        # Sort candidates deterministically:
        # 1. Structural rank (-2 compare, -1 terminal sink, 0 standard, 1 partial sink)
        # 2. Transitive slot coverage (higher is better)
        # 3. Intermediate variable connections
        candidates_with_meta.sort(
            key=lambda item: (
                item[1],
                -item[0],
                0 if any(str(v).startswith("$step_") or str(v).startswith("$sum_") for v in item[2].args.values()) else 1,
                item[2].operation,
                str(item[2].args),
            )
        )

        return [c for _, _, c in candidates_with_meta]

    @classmethod
    def validate_step(
        cls,
        step: PlanStep,
        slots: dict[str, GroundedSlot],
        working_memory: WorkingMemory,
        executed_steps: list[PlanStep],
    ) -> tuple[bool, str]:
        """Validate an individual proposed step against grammar, SSA, and type/unit constraints."""
        # 1. Operation registration
        if step.operation not in RULE_REGISTRY:
            return False, f"Unknown operation '{step.operation}' not in RULE_REGISTRY"

        # 2. SSA enforcement: output variable must be unique
        used_vars = {s.output_var for s in executed_steps}
        if step.output_var in used_vars or step.output_var in working_memory.variables:
            return False, f"SSA violation: variable '{step.output_var}' is already bound"

        # 3. Check for self-dependency or cycle
        for arg_key, arg_val in step.args.items():
            if isinstance(arg_val, str) and arg_val.startswith("$"):
                ref = arg_val[1:].strip()
                if ref == step.output_var:
                    return False, f"Self-dependency: step references its own output '${ref}'"
                if ref not in slots and ref not in working_memory.variables:
                    return False, f"Undefined reference '${ref}': not found in slots or intermediate variables"

        # 4. Unit consistency for addition
        if step.operation == "add":
            units: list[str | None] = []
            for arg_val in step.args.values():
                if isinstance(arg_val, str) and arg_val.startswith("$"):
                    ref = arg_val[1:].strip()
                    if ref in slots:
                        units.append(slots[ref].unit)
                    elif ref in working_memory.variables:
                        units.append(working_memory.step_units.get(ref))
            if len(units) == 2 and units[0] is not None and units[1] is not None:
                if units[0] != units[1]:
                    return (
                        False,
                        f"Unit mismatch for addition: '{units[0]}' != '{units[1]}'",
                    )

        return True, "Valid"


@dataclass(order=True)
class SearchNode:
    """A search node in the execution-guided branch space."""

    priority: float  # Min-heap priority (lower value popped first)
    depth: int = field(compare=False, default=0)
    executed_steps: list[PlanStep] = field(compare=False, default_factory=list)
    working_memory: WorkingMemory = field(compare=False, default_factory=WorkingMemory)
    state_signature: str = field(compare=False, default="")
    is_terminal: bool = field(compare=False, default=False)
    score: float = field(compare=False, default=0.0)


class GrammarConstrainedPlanner:
    """Bounded, execution-guided planner with branch memory isolation and grammar constraints.
    
    Guarantees:
    - Bounded search: strictly respects beam_width, max_expansions, and max_depth.
    - Isolated branch memory: rejected branches leave zero state in siblings or parent.
    - Zero arbitrary code execution: deterministic execution via RuleExecutor.
    - Independent goal & semantic verification: does not treat all executable plans as correct.
    - Explicit failure on budget exhaustion.
    """

    def __init__(
        self,
        rule_executor: RuleExecutor | None = None,
        beam_width: int = 4,
        max_expansions: int = 30,
        max_depth: int = 5,
        score_fn: Callable[[str, list[PlanStep], PlanStep], float] | None = None,
    ) -> None:
        self.rule_executor = rule_executor or RuleExecutor()
        self.beam_width = max(1, beam_width)
        self.max_expansions = max(1, max_expansions)
        self.max_depth = max(1, min(max_depth, 5))
        self.score_fn = score_fn

    def _compute_state_signature(self, steps: list[PlanStep]) -> str:
        """Canonical signature to detect duplicate or equivalent DAG states."""
        step_strs = [
            f"{s.operation}({sorted(s.args.items())})" for s in steps
        ]
        return " -> ".join(step_strs)

    def plan(
        self,
        question: str,
        slots: dict[str, GroundedSlot],
        target_goal: TargetGoal | None = None,
    ) -> tuple[ExecutionPlan, WorkingMemory]:
        """Synthesize and execute a verified plan within strict search bounds.
        
        Raises:
            SearchBudgetExhaustedError: if search frontier is exhausted or budget limit is reached
            PlanValidationError: if question has invalid preconditions
        """
        goal = target_goal or TargetGoal.from_question(question, slots)

        # Initialize root working memory with slots
        root_mem = WorkingMemory()
        root_mem.slots = copy.copy(slots)

        root_node = SearchNode(
            priority=0.0,
            depth=0,
            executed_steps=[],
            working_memory=root_mem,
            state_signature="",
            is_terminal=False,
            score=0.0,
        )

        frontier: list[SearchNode] = [root_node]
        visited_signatures: set[str] = set()

        expansions = 0
        best_candidate_plan: ExecutionPlan | None = None
        best_candidate_mem: WorkingMemory | None = None

        while frontier and expansions < self.max_expansions:
            current = heapq.heappop(frontier)

            # Check if current node is a completed goal state
            if current.executed_steps:
                candidate_plan = ExecutionPlan(
                    steps=current.executed_steps,
                    final_var=current.executed_steps[-1].output_var,
                )
                candidate_mem = current.working_memory
                final_val = candidate_mem.variables.get(candidate_plan.final_var or "")
                candidate_mem.bind_result(final_val, verified=True)

                is_sem_correct, msg = goal.evaluate_semantics(candidate_plan, candidate_mem)
                if is_sem_correct:
                    logger.debug("Goal achieved at depth %d in %d expansions", current.depth, expansions)
                    return candidate_plan, candidate_mem
                else:
                    best_candidate_plan = candidate_plan
                    best_candidate_mem = candidate_mem

            if current.depth >= self.max_depth:
                continue

            expansions += 1

            # Generate grammar-constrained candidates
            candidates = PlanGrammar.get_candidate_steps(
                slots=slots,
                working_memory=current.working_memory,
                executed_steps=current.executed_steps,
                max_steps=self.max_depth,
                question=question,
            )

            branch_nodes: list[SearchNode] = []
            for cand in candidates:
                # 1. Grammar validation
                is_valid, err_msg = PlanGrammar.validate_step(
                    step=cand,
                    slots=slots,
                    working_memory=current.working_memory,
                    executed_steps=current.executed_steps,
                )
                if not is_valid:
                    continue

                # 2. Memory Isolation: Deep-copy working memory for this branch
                branch_mem = copy.deepcopy(current.working_memory)

                # 3. Resolve arguments against branch memory
                resolved_args: dict[str, Any] = {}
                try:
                    for arg_k, arg_v in cand.args.items():
                        if isinstance(arg_v, str) and arg_v.startswith("$"):
                            ref = arg_v[1:].strip()
                            if ref in branch_mem.variables:
                                resolved_args[arg_k] = branch_mem.variables[ref]
                            elif ref in branch_mem.slots:
                                resolved_args[arg_k] = branch_mem.slots[ref].value
                            else:
                                raise DependencyResolutionError(f"Undefined reference '${ref}'")
                        else:
                            resolved_args[arg_k] = arg_v
                except DependencyResolutionError:
                    continue

                # 4. Deterministic step execution and verification
                exec_res = self.rule_executor.execute(cand.operation, resolved_args, verify=True)
                if not exec_res.success or not exec_res.verified:
                    continue

                # 5. Bind verified result to isolated branch memory
                step_idx = len(current.executed_steps) + 1
                branch_mem.bind_step_variable(
                    var_name=cand.output_var,
                    value=exec_res.result,
                    operation=cand.operation,
                    raw_args=cand.args,
                    resolved_args=resolved_args,
                    units=cand.units,
                    step_index=step_idx,
                )

                new_steps = list(current.executed_steps) + [cand]
                sig = self._compute_state_signature(new_steps)

                if sig in visited_signatures:
                    continue
                visited_signatures.add(sig)

                # 6. Score candidate step
                step_score = (
                    self.score_fn(question, current.executed_steps, cand)
                    if self.score_fn is not None
                    else (1.0 / (len(new_steps) + 1))
                )
                total_score = current.score + step_score

                branch_node = SearchNode(
                    priority=-total_score,
                    depth=current.depth + 1,
                    executed_steps=new_steps,
                    working_memory=branch_mem,
                    state_signature=sig,
                    score=total_score,
                )
                branch_nodes.append(branch_node)

            # Beam pruning
            branch_nodes.sort(key=lambda n: n.priority)
            for node in branch_nodes[: self.beam_width]:
                heapq.heappush(frontier, node)

        # Budget exhaustion
        raise SearchBudgetExhaustedError(
            f"Search budget exhausted ({expansions}/{self.max_expansions} expansions) "
            f"without reaching a valid semantic goal for query: '{question[:80]}'"
        )


class NeuralCandidateScorer:
    """Scores next-step candidate operations using Mini-GPT log-probabilities.
    
    Prompts the causal Transformer with the question, available grounded slots,
    and partial plan generated so far, scoring the candidate step token sequence.
    Strictly isolated from ground-truth target answers.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Any = None,
        slots: dict[str, GroundedSlot] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or getattr(model, "device", "cpu")
        self.slots = slots or {}

    def score_step(
        self,
        question: str,
        executed_steps: list[PlanStep],
        candidate: PlanStep,
    ) -> float:
        """Compute log-probability score of proposing candidate step given prefix history."""
        import torch
        import torch.nn.functional as F
        from adaptive_mini_gpt.rules.grounding import format_available_slots

        was_training = self.model.training
        self.model.eval()

        # Build prompt: question + available slots + executed steps so far
        slots_text = format_available_slots(self.slots).strip() if self.slots else ""
        user_body = f"{question}\n{slots_text}" if slots_text else question
        lines = [f"<|user|>\n{user_body}\n<|assistant|>\nPlan:"]
        for idx, s in enumerate(executed_steps, start=1):
            u_str = f" [{s.units}]" if s.units else ""
            args_formatted = ", ".join(
                f"{k}={repr(v) if isinstance(v, str) else v}" for k, v in s.args.items()
            )
            lines.append(f"{idx}. {s.output_var} = {s.operation}({args_formatted}){u_str}")

        next_idx = len(executed_steps) + 1
        prefix = "\n".join(lines) + f"\n{next_idx}. "

        # Candidate step string representation
        cand_u_str = f" [{candidate.units}]" if candidate.units else ""
        cand_args_formatted = ", ".join(
            f"{k}={repr(v) if isinstance(v, str) else v}" for k, v in candidate.args.items()
        )
        candidate_text = f"{candidate.output_var} = {candidate.operation}({cand_args_formatted}){cand_u_str}"

        prefix_tokens = self.tokenizer.encode(prefix)
        cand_tokens = self.tokenizer.encode(candidate_text)
        full_tokens = prefix_tokens + cand_tokens

        if len(full_tokens) > self.model.config.context_length:
            return -100.0

        input_tensor = torch.tensor([full_tokens], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(input_tensor)
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = input_tensor[:, 1:]
            token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

            start_idx = len(prefix_tokens) - 1
            step_log_probs = token_log_probs[0, start_idx:]
            mean_log_prob = step_log_probs.mean().item()

        if was_training:
            self.model.train()

        return float(mean_log_prob)
