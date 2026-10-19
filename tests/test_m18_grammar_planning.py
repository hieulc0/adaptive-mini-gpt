"""Unit and Integration Tests for M18 Grammar-Constrained Planning and Search Engine."""

from __future__ import annotations

import pytest

from adaptive_mini_gpt.rules import (
    DeterministicTemplatePlanner,
    ExecutionPlan,
    GroundedSlot,
    PlanExecutor,
    PlanStep,
    RuleExecutor,
    WorkingMemory,
    generate_addition_depth_ladder,
    generate_multistep_problems,
)
from adaptive_mini_gpt.rules.grammar_planner import (
    GrammarConstrainedPlanner,
    PlanGrammar,
    SearchBudgetExhaustedError,
    TargetGoal,
)


def test_baseline_reconciliation_five_step_speed_100_percent() -> None:
    """Verify that the 5-step, 3-segment speed cases achieve 100% verified accuracy on frozen test seeds."""
    executor = PlanExecutor()
    seeds = [42, 43, 44]

    for seed in seeds:
        test_problems = generate_multistep_problems(num_samples_per_archetype=10, seed=seed + 999, split="test")
        five_step_problems = [p for p in test_problems if p.num_steps == 5]
        assert len(five_step_problems) == 10, f"Expected 10 five-step problems for seed {seed}"

        correct = 0
        for prob in five_step_problems:
            plan = DeterministicTemplatePlanner.plan(prob.question)
            res = executor.execute(plan)
            assert res.success, f"Plan execution failed: {res.error}"
            assert str(res.final_result).lower() == str(prob.expected_answer).lower()
            correct += 1

        assert correct == 10, f"Seed {seed} failed 5-step speed regression: got {correct}/10"


def test_plan_grammar_type_and_unit_candidate_generation() -> None:
    """Verify that PlanGrammar generates valid candidates and rejects dimensional mismatches (km + h)."""
    slots = {
        "distance_1": GroundedSlot("distance_1", 120.0, "km", "120 km", 0, 0, type_name="float"),
        "duration_1": GroundedSlot("duration_1", 2.0, "h", "2 hours", 0, 0, type_name="float"),
        "distance_2": GroundedSlot("distance_2", 80.0, "km", "80 km", 0, 0, type_name="float"),
        "duration_2": GroundedSlot("duration_2", 1.0, "h", "1 hours", 0, 0, type_name="float"),
    }
    mem = WorkingMemory()
    mem.slots = slots

    candidates = PlanGrammar.get_candidate_steps(slots=slots, working_memory=mem, executed_steps=[])
    assert len(candidates) > 0

    # Ensure no candidate adds distance to duration
    for c in candidates:
        if c.operation == "add":
            arg_a = c.args["a"]
            arg_b = c.args["b"]
            # Distance + duration should never appear
            is_cross_dim = ("distance" in arg_a and "duration" in arg_b) or ("duration" in arg_a and "distance" in arg_b)
            assert not is_cross_dim, f"Grammar generated dimensionally incompatible candidate: {c}"


def test_invalid_operations_and_references_rejected() -> None:
    """Verify that unknown rules, undefined variables, and cycles are rejected by PlanGrammar."""
    slots = {"operand_1": GroundedSlot("operand_1", 10, None, "10", 0, 0, type_name="int")}
    mem = WorkingMemory()
    mem.slots = slots

    # 1. Unknown rule
    bad_rule_step = PlanStep(output_var="s1", operation="nonexistent_rule", args={"x": "$operand_1"})
    valid, msg = PlanGrammar.validate_step(bad_rule_step, slots, mem, [])
    assert not valid
    assert "Unknown operation" in msg

    # 2. Undefined reference
    bad_ref_step = PlanStep(output_var="s1", operation="add", args={"a": "$undefined_var", "b": 10})
    valid, msg = PlanGrammar.validate_step(bad_ref_step, slots, mem, [])
    assert not valid
    assert "Undefined reference" in msg

    # 3. Self-dependency / cycle
    cycle_step = PlanStep(output_var="s1", operation="add", args={"a": "$s1", "b": 10})
    valid, msg = PlanGrammar.validate_step(cycle_step, slots, mem, [])
    assert not valid
    assert "Self-dependency" in msg

    # 4. SSA violation (reusing existing variable name)
    mem.variables["existing_var"] = 42
    ssa_step = PlanStep(output_var="existing_var", operation="add", args={"a": "$operand_1", "b": 5})
    valid, msg = PlanGrammar.validate_step(ssa_step, slots, mem, [])
    assert not valid
    assert "SSA violation" in msg


def test_branch_memory_isolation() -> None:
    """Verify that executing a candidate step on branch A does not contaminate branch B or parent."""
    slots = {
        "operand_1": GroundedSlot("operand_1", 100, None, "100", 0, 0, type_name="int"),
        "operand_2": GroundedSlot("operand_2", 50, None, "50", 0, 0, type_name="int"),
    }
    parent_mem = WorkingMemory()
    parent_mem.slots = slots
    parent_mem.variables["base_val"] = 100

    # Branch A execution
    branch_a_mem = PlanGrammar.get_candidate_steps(slots, parent_mem, [])
    planner = GrammarConstrainedPlanner()

    # Create two isolated branch memories
    import copy
    mem_a = copy.deepcopy(parent_mem)
    mem_b = copy.deepcopy(parent_mem)

    # Mutate branch A
    executor = RuleExecutor()
    res_a = executor.execute("add", {"a": 100, "b": 50})
    mem_a.bind_step_variable("step_1", res_a.result, "add", {}, {}, None, 1)

    assert "step_1" in mem_a.variables
    assert mem_a.variables["step_1"] == 150

    # Verify branch B and parent remain completely clean
    assert "step_1" not in mem_b.variables
    assert "step_1" not in parent_mem.variables
    assert parent_mem.variables == {"base_val": 100}


def test_search_budget_exhaustion_explicit_failure() -> None:
    """Verify that GrammarConstrainedPlanner raises SearchBudgetExhaustedError when budget is exhausted."""
    slots = {
        "operand_1": GroundedSlot("operand_1", 10, None, "10", 0, 0, type_name="int"),
    }
    # Unreachable goal requiring unit "km/h" with only integer operands
    impossible_goal = TargetGoal(goal_type="speed", required_slots={"operand_1"}, expected_unit="km/h")

    # Set very small expansion budget
    planner = GrammarConstrainedPlanner(max_expansions=3, max_depth=2)

    with pytest.raises(SearchBudgetExhaustedError, match="Search budget exhausted"):
        planner.plan(
            question="What is the speed?",
            slots=slots,
            target_goal=impossible_goal,
        )


def test_semantic_correctness_vs_structural_validity() -> None:
    """Verify that an executable plan that does not solve the target problem is rejected semantically."""
    slots = {
        "operand_1": GroundedSlot("operand_1", 10, None, "10", 0, 0, type_name="int"),
        "operand_2": GroundedSlot("operand_2", 20, None, "20", 0, 0, type_name="int"),
        "operand_3": GroundedSlot("operand_3", 30, None, "30", 0, 0, type_name="int"),
        "operand_4": GroundedSlot("operand_4", 40, None, "40", 0, 0, type_name="int"),
    }
    # Goal: sum of all 4 operands (expected answer 100)
    goal = TargetGoal(
        goal_type="sum",
        required_slots={"operand_1", "operand_2", "operand_3", "operand_4"},
        expected_answer="100",
    )

    # Incomplete plan: only adds operand_1 and operand_2
    partial_plan = ExecutionPlan(
        steps=[PlanStep(output_var="sum_1", operation="add", args={"a": "$operand_1", "b": "$operand_2"})],
        final_var="sum_1",
    )
    # 1. Structural validity: plan is valid and executes successfully
    partial_plan.validate(allowed_slots=set(slots.keys()))
    executor = PlanExecutor()
    mem = WorkingMemory()
    mem.slots = slots
    res = executor.execute(partial_plan, working_memory=mem)
    assert res.success
    assert res.final_result == 30

    # 2. Semantic correctness: must FAIL because operand_3 and operand_4 were omitted
    is_sem_correct, diagnostic = goal.evaluate_semantics(partial_plan, res.working_memory)
    assert not is_sem_correct
    assert "Semantically incomplete plan" in diagnostic


def test_addition_depth_ladder_generation_and_execution() -> None:
    """Verify generation of addition depth ladder across 2, 3, 4, and 5 reasoning steps."""
    ladder = generate_addition_depth_ladder(num_samples_per_depth=5, seed=42, split="test")
    assert len(ladder) == 20

    depth_counts: dict[int, int] = {}
    executor = PlanExecutor()

    for prob in ladder:
        depth_counts[prob.num_steps] = depth_counts.get(prob.num_steps, 0) + 1
        # Verify slot plan executes to expected answer
        res = executor.execute(prob.expected_slot_plan, working_memory=WorkingMemory())
        # Set slots into memory
        mem = WorkingMemory()
        mem.slots = prob.slots
        res = executor.execute(prob.expected_slot_plan, working_memory=mem)
        assert res.success
        assert str(res.final_result) == str(prob.expected_answer)

    # 5 samples each for depths 2, 3, 4, 5
    assert depth_counts == {2: 5, 3: 5, 4: 5, 5: 5}
