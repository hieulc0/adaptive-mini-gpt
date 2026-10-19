"""Grammar-Constrained, Execution-Guided Planning Engine (M18).

Implements bounded, stepwise plan construction with isolated branch memory:
1. PlanGrammar: Strict type-, unit-, and dependency-constrained next-step candidate generation.
   - Restricts candidates to registered rules, available slots, and verified intermediate variables.
   - Enforces Single Static Assignment (SSA) and rejects forward/cyclic references.
   - Prunes dimensionally incompatible operations (e.g. adding distance to duration).
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
    """Explicit semantic specification of the requested goal for a reasoning task.
    
    Decouples structural validity (whether the plan executes) from semantic correctness
    (whether the plan actually answers what was asked).
    """

    goal_type: str  # "speed", "sum", "compare", "reverse", "generic"
    required_slots: set[str] = field(default_factory=set)
    expected_unit: str | None = None
    expected_answer: str | None = None

    @classmethod
    def from_question(
        cls,
        question: str,
        slots: dict[str, GroundedSlot] | None = None,
        expected_answer: str | None = None,
    ) -> TargetGoal:
        """Infer target goal semantics from question text and extracted slots."""
        q_lower = question.lower()
        available_slots = set(slots.keys()) if slots else set()

        if "average speed" in q_lower or (
            ("travel" in q_lower or "cover" in q_lower or "drive" in q_lower or "fly" in q_lower)
            and "km" in q_lower
            and ("hour" in q_lower or "hrs" in q_lower or "h" in q_lower)
        ):
            # Requires all distance and duration slots
            kin_slots = {s for s in available_slots if s.startswith("distance_") or s.startswith("duration_")}
            return cls(
                goal_type="speed",
                required_slots=kin_slots,
                expected_unit="km/h",
                expected_answer=expected_answer,
            )

        if any(kw in q_lower for kw in ("sum", "total", "plus", "add")) and "than" not in q_lower:
            op_slots = {s for s in available_slots if s.startswith("operand_")}
            return cls(
                goal_type="sum",
                required_slots=op_slots,
                expected_unit=None,
                expected_answer=expected_answer,
            )

        if any(kw in q_lower for kw in ("taller", "faster", "heavier", "older", "shorter")):
            ent_slots = {s for s in available_slots if s.startswith("entity_")}
            return cls(
                goal_type="compare",
                required_slots=ent_slots,
                expected_unit=None,
                expected_answer=expected_answer,
            )

        if any(kw in q_lower for kw in ("reverse", "backward", "invert")):
            return cls(
                goal_type="reverse",
                required_slots=available_slots,
                expected_unit=None,
                expected_answer=expected_answer,
            )

        return cls(
            goal_type="generic",
            required_slots=available_slots,
            expected_unit=None,
            expected_answer=expected_answer,
        )

    def evaluate_semantics(
        self,
        plan: ExecutionPlan,
        working_memory: WorkingMemory,
    ) -> tuple[bool, str]:
        """Check whether the executed plan satisfies the semantic goal.
        
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
                # Also check string value e.g. "60 km/h"
                if not (isinstance(final_val, str) and final_val.endswith(self.expected_unit)):
                    return False, f"Final unit mismatch: expected '{self.expected_unit}', got '{final_unit}'"

        # 2. Check dependency completeness (transitive coverage of required slots)
        if self.required_slots:
            covered_slots = set()
            for step in plan.steps:
                for arg_val in step.args.values():
                    if isinstance(arg_val, str) and arg_val.startswith("$"):
                        ref = arg_val[1:].strip()
                        if ref in self.required_slots:
                            covered_slots.add(ref)

            missing_slots = self.required_slots - covered_slots
            if missing_slots:
                return (
                    False,
                    f"Semantically incomplete plan: final answer does not depend on required slots {sorted(missing_slots)}",
                )

        # 3. Ground truth evaluation when expected answer is available
        if self.expected_answer is not None:
            if str(final_val).strip().lower() != str(self.expected_answer).strip().lower():
                return (
                    False,
                    f"Final answer '{final_val}' does not match expected answer '{self.expected_answer}'",
                )

        return True, "Plan satisfies semantic goal"


class PlanGrammar:
    """Grammar and type validator for generating valid candidate plan steps.
    
    Guarantees:
    - Never generates invalid operations or unknown rules.
    - Never references undefined variables or forward variables.
    - Enforces unit compatibility (rejects adding km + h).
    - Prevents cyclic dependencies and self-references.
    - Maintains Single Static Assignment (SSA).
    """

    @classmethod
    def get_candidate_steps(
        cls,
        slots: dict[str, GroundedSlot],
        working_memory: WorkingMemory,
        executed_steps: list[PlanStep],
        max_steps: int = 5,
    ) -> list[PlanStep]:
        """Generate all grammatically valid, type-safe next candidate steps from the current state."""
        if len(executed_steps) >= max_steps:
            return []

        candidates: list[PlanStep] = []
        next_step_idx = len(executed_steps) + 1
        output_var = f"step_{next_step_idx}"

        # Collect available variable and slot references
        # Each reference is (ref_id, value, unit, type_name)
        available_refs: list[tuple[str, Any, str | None, str]] = []

        # 1. Grounded input slots
        for sid, slot in slots.items():
            t_name = slot.type_name or (type(slot.value).__name__)
            available_refs.append((f"${sid}", slot.value, slot.unit, t_name))

        # 2. Verified intermediate variables from executed steps
        for var_name, var_val in working_memory.variables.items():
            u = working_memory.step_units.get(var_name)
            t_name = type(var_val).__name__
            available_refs.append((f"${var_name}", var_val, u, t_name))

        # -------------------------------------------------------------
        # Rule 1: 'add' (arithmetic addition)
        # -------------------------------------------------------------
        # Requires two numeric inputs (int or float) with matching or compatible units
        numeric_refs = [
            r for r in available_refs if isinstance(r[1], (int, float)) and r[3] in ("int", "float")
        ]

        # Group by unit to enforce dimensional homogeneity
        unit_groups: dict[str | None, list[tuple[str, Any, str | None, str]]] = {}
        for r in numeric_refs:
            unit_groups.setdefault(r[2], []).append(r)

        for u, refs in unit_groups.items():
            # Generate pairs of operands
            for i in range(len(refs)):
                for j in range(i + 1, len(refs)):
                    r1, r2 = refs[i], refs[j]
                    # Avoid adding identical variable to itself unless from distinct slots
                    if r1[0] == r2[0]:
                        continue
                    # Canonical order: intermediate variables first, or alphabetical
                    cand_args = {"a": r1[0], "b": r2[0]}
                    candidates.append(
                        PlanStep(
                            output_var=output_var,
                            operation="add",
                            args=cand_args,
                            units=u,
                        )
                    )

        # -------------------------------------------------------------
        # Rule 2: 'average_speed' (kinematics)
        # -------------------------------------------------------------
        # Requires distance_km and time_hours
        distance_refs = [
            r for r in numeric_refs
            if (r[2] in ("km", "kilometer", "kilometers") or "distance" in r[0] or "d_" in r[0])
            and r[1] >= 0
        ]
        time_refs = [
            r for r in numeric_refs
            if (r[2] in ("h", "hour", "hours", "hrs") or "duration" in r[0] or "t_" in r[0])
            and r[1] > 0
        ]

        for d_ref in distance_refs:
            for t_ref in time_refs:
                candidates.append(
                    PlanStep(
                        output_var=output_var,
                        operation="average_speed",
                        args={"distance_km": d_ref[0], "time_hours": t_ref[0]},
                        units="km/h",
                    )
                )

        # -------------------------------------------------------------
        # Rule 3: 'reverse' (string reversal)
        # -------------------------------------------------------------
        text_refs = [r for r in available_refs if isinstance(r[1], str) and r[3] == "str"]
        for t_ref in text_refs:
            candidates.append(
                PlanStep(
                    output_var=output_var,
                    operation="reverse",
                    args={"text": t_ref[0]},
                )
            )

        return candidates

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
                # Reference must exist in slots or already verified intermediate variables
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
                    elif ref in working_memory.step_units:
                        units.append(working_memory.step_units[ref])
            non_null_units = [u for u in units if u is not None]
            if len(non_null_units) == 2 and non_null_units[0] != non_null_units[1]:
                return False, f"Unit mismatch in addition: '{non_null_units[0]}' vs '{non_null_units[1]}'"

        return True, "Valid grammar step"


@dataclass(order=True)
class SearchNode:
    """A search state node in the bounded frontier with isolated branch working memory."""

    priority: float  # For min-heap (lower is better, or -score)
    depth: int = field(compare=False)
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
        beam_width: int = 3,
        max_expansions: int = 15,
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

        # Frontier: priority queue of search nodes
        frontier: list[SearchNode] = [root_node]
        visited_signatures: set[str] = set()

        expansions = 0
        best_candidate_plan: ExecutionPlan | None = None
        best_candidate_mem: WorkingMemory | None = None

        while frontier and expansions < self.max_expansions:
            # Pop best node
            current = heapq.heappop(frontier)

            # Check if current node is a completed goal state
            if current.executed_steps:
                candidate_plan = ExecutionPlan(
                    steps=current.executed_steps,
                    final_var=current.executed_steps[-1].output_var,
                )
                is_sem_correct, msg = goal.evaluate_semantics(candidate_plan, current.working_memory)
                if is_sem_correct:
                    logger.debug("Goal achieved at depth %d in %d expansions", current.depth, expansions)
                    return candidate_plan, current.working_memory
                else:
                    best_candidate_plan = candidate_plan
                    best_candidate_mem = current.working_memory

            # Check depth limit
            if current.depth >= self.max_depth:
                continue

            expansions += 1

            # Generate grammar-constrained candidates
            candidates = PlanGrammar.get_candidate_steps(
                slots=slots,
                working_memory=current.working_memory,
                executed_steps=current.executed_steps,
                max_steps=self.max_depth,
            )

            # Expand each candidate with isolated branch memory
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
                    # Prune branch
                    continue

                # 4. Deterministic step execution and precondition verification
                exec_res = self.rule_executor.execute(cand.operation, resolved_args, verify=True)
                if not exec_res.success or not exec_res.verified:
                    # Prune branch on execution failure (leaves zero trace in siblings)
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

                # Loop and state deduplication
                if sig in visited_signatures:
                    continue
                visited_signatures.add(sig)

                # Score the candidate step (model score or heuristic depth preference)
                step_score = (
                    self.score_fn(question, current.executed_steps, cand)
                    if self.score_fn is not None
                    else (1.0 / (len(new_steps) + 1))
                )
                total_score = current.score + step_score

                branch_node = SearchNode(
                    priority=-total_score,  # Max-priority in min-heap
                    depth=current.depth + 1,
                    executed_steps=new_steps,
                    working_memory=branch_mem,
                    state_signature=sig,
                    score=total_score,
                )
                branch_nodes.append(branch_node)

            # Beam pruning: keep top beam_width branches
            branch_nodes.sort(key=lambda n: n.priority)
            for node in branch_nodes[: self.beam_width]:
                heapq.heappush(frontier, node)

        # If loop finishes without returning, search budget is exhausted
        raise SearchBudgetExhaustedError(
            f"Search budget exhausted ({expansions}/{self.max_expansions} expansions) "
            f"without reaching a valid semantic goal for query: '{question[:80]}'"
        )
