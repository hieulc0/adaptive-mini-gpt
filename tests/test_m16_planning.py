"""Unit and Integration Tests for Multi-Step Planning and Compositional Reasoning (M16)."""

from __future__ import annotations

import pytest

from adaptive_mini_gpt.rules import (
    DeterministicTemplatePlanner,
    ExecutionPlan,
    PlanExecutionResult,
    PlanExecutor,
    PlanStep,
    PlanValidationError,
    ReasoningController,
    WorkingMemory,
    are_arguments_semantically_equivalent,
    generate_multistep_abstention_problems,
    generate_multistep_problems,
    parse_plan,
)


def test_plan_validation_and_safety_checks() -> None:
    """Verify plan topological validation, cycle detection, and step limit constraints."""
    # 1. Valid 2-step plan
    step1 = PlanStep(output_var="d_tot", operation="add", args={"a": 120, "b": 90}, units="km")
    step2 = PlanStep(output_var="t_tot", operation="add", args={"a": 2, "b": 3}, units="h")
    plan = ExecutionPlan(steps=[step1, step2], final_var="d_tot")
    plan.validate()
    assert plan.final_var == "d_tot"

    # 2. Cycle / Self-dependency rejection
    bad_self = ExecutionPlan(
        steps=[PlanStep(output_var="x", operation="add", args={"a": "$x", "b": 10})],
        final_var="x",
    )
    with pytest.raises(PlanValidationError, match="self-dependency"):
        bad_self.validate()

    # 3. Forward reference rejection
    bad_fwd = ExecutionPlan(
        steps=[
            PlanStep(output_var="step1", operation="add", args={"a": "$step2", "b": 10}),
            PlanStep(output_var="step2", operation="add", args={"a": 5, "b": 5}),
        ],
        final_var="step1",
    )
    with pytest.raises(PlanValidationError, match="undefined variable"):
        bad_fwd.validate()

    # 4. Redefining variable (SSA violation)
    bad_ssa = ExecutionPlan(
        steps=[
            PlanStep(output_var="x", operation="add", args={"a": 1, "b": 2}),
            PlanStep(output_var="x", operation="add", args={"a": 3, "b": 4}),
        ],
        final_var="x",
    )
    with pytest.raises(PlanValidationError, match="redefined"):
        bad_ssa.validate()

    # 5. Step limit (max 5 steps)
    too_many_steps = [
        PlanStep(output_var=f"v_{i}", operation="add", args={"a": i, "b": 1}) for i in range(6)
    ]
    bad_len = ExecutionPlan(steps=too_many_steps)
    with pytest.raises(PlanValidationError, match="maximum limit of 5 steps"):
        bad_len.validate()


def test_plan_executor_multi_segment_speed() -> None:
    """Verify execution of 3-step average speed composition with provenance and unit consistency."""
    # Segment 1: 120 km in 2 h | Segment 2: 90 km in 3 h -> 210 km in 5 h = 42 km/h
    plan = ExecutionPlan(
        steps=[
            PlanStep(output_var="d_total", operation="add", args={"a": 120.0, "b": 90.0}, units="km"),
            PlanStep(output_var="t_total", operation="add", args={"a": 2.0, "b": 3.0}, units="h"),
            PlanStep(
                output_var="overall_speed",
                operation="average_speed",
                args={"distance_km": "$d_total", "time_hours": "$t_total"},
                units="km/h",
            ),
        ],
        final_var="overall_speed",
    )

    executor = PlanExecutor()
    res = executor.execute(plan)

    assert res.success
    assert res.final_result == "42 km/h"
    assert res.executed_steps == 3
    assert res.working_memory is not None
    assert res.working_memory.variables["d_total"] == 210.0
    assert res.working_memory.variables["t_total"] == 5.0
    assert res.working_memory.variables["overall_speed"] == "42 km/h"
    assert "d_total" in res.working_memory.step_provenance
    assert res.working_memory.step_provenance["overall_speed"].verified


def test_plan_executor_dimensional_mismatch_rejection() -> None:
    """Verify that attempting to add distance (km) and time (h) is rejected by unit verification."""
    plan = ExecutionPlan(
        steps=[
            PlanStep(output_var="d1", operation="add", args={"a": 100, "b": 20}, units="km"),
            PlanStep(output_var="t1", operation="add", args={"a": 1, "b": 1}, units="h"),
            PlanStep(output_var="bad_sum", operation="add", args={"a": "$d1", "b": "$t1"}),
        ],
        final_var="bad_sum",
    )

    executor = PlanExecutor()
    res = executor.execute(plan)
    assert not res.success
    assert "Unit mismatch" in (res.error or "")
    assert res.executed_steps == 2


def test_plan_parser() -> None:
    """Verify parsing of line-based plan strings into typed ExecutionPlans."""
    plan_text = """
    Plan:
    1. d_tot = add(a=120, b=90) [km]
    2. t_tot = add(a=2, b=3) [h]
    3. speed = average_speed(distance_km=$d_tot, time_hours=$t_tot) [km/h]
    Result: $speed
    """
    plan = parse_plan(plan_text)
    assert len(plan.steps) == 3
    assert plan.final_var == "speed"
    assert plan.steps[0].output_var == "d_tot"
    assert plan.steps[0].operation == "add"
    assert plan.steps[0].units == "km"
    assert plan.steps[2].args["distance_km"] == "$d_tot"


def test_deterministic_template_planner() -> None:
    """Verify deterministic plan synthesis on multi-segment speed, chained addition, and comparisons."""
    # 1. 2-segment speed
    q_speed = "A car travels 120 km in 2 hours, then 90 km in 3 hours. What is its overall average speed?"
    plan_speed = DeterministicTemplatePlanner.plan(q_speed)
    assert len(plan_speed.steps) == 3
    assert plan_speed.final_var == "overall_speed"
    assert plan_speed.steps[0].operation == "add"
    assert plan_speed.steps[1].operation == "add"
    assert plan_speed.steps[2].operation == "average_speed"

    # 2. Chained addition (4 operands)
    q_add = "Find the total sum of 100, 200, 300, and 400."
    plan_add = DeterministicTemplatePlanner.plan(q_add)
    assert len(plan_add.steps) == 3
    assert plan_add.steps[0].args == {"a": 100, "b": 200}
    assert plan_add.steps[1].args == {"a": "$sum_1", "b": 300}
    assert plan_add.steps[2].args == {"a": "$sum_2", "b": 400}

    # 3. Incompatible dimensions -> AbstentionException
    with pytest.raises(Exception, match="Incompatible physical dimensions"):
        DeterministicTemplatePlanner.plan("What is 120 km plus 3 hours?")


def test_reasoning_controller_multistep_end_to_end() -> None:
    """Verify end-to-end multi-step reasoning execution with ReasoningController."""
    controller = ReasoningController()

    q = "A train covers 150 km in 2 hours, followed by another 90 km in 2 hours. Calculate average speed for the whole journey."
    res = controller.solve_multistep(
        question=q,
        expected_plan=None,
        expected_answer="60 km/h",
        mode="deterministic_template",
    )
    assert res.status == "success"
    assert res.answer_correct
    assert res.final_answer == "60 km/h"
    assert res.num_steps == 3
    assert res.working_memory is not None
    assert res.working_memory.verified


def test_multistep_abstention_battery() -> None:
    """Verify that multi-step malformed, dimensional mismatch, and cyclic queries trigger safe abstention."""
    controller = ReasoningController()
    abstention_problems = generate_multistep_abstention_problems(num_samples=6, seed=42)
    assert len(abstention_problems) > 0

    for prob in abstention_problems:
        res = controller.solve_multistep(
            question=prob.question,
            expected_plan=prob.expected_plan,
            expected_answer=prob.expected_answer,
            mode="deterministic_template",
            should_abstain=True,
        )
        assert res.abstained is True
        assert res.abstention_correct is True
        assert res.status == "abstained"


def test_semantic_argument_equivalence_audit() -> None:
    """Verify that commutative argument orders are recognized as semantically equivalent."""
    # Addition: order invariant
    args_std = {"a": 181, "b": 319}
    args_rev = {"a": 319, "b": 181}
    assert are_arguments_semantically_equivalent("add", args_std, args_rev)
    assert are_arguments_semantically_equivalent("add", args_std, args_std)

    # Subtraction / Average speed: order sensitive
    args_speed1 = {"distance_km": 120.0, "time_hours": 2.0}
    args_speed2 = {"distance_km": 2.0, "time_hours": 120.0}
    assert not are_arguments_semantically_equivalent("average_speed", args_speed1, args_speed2)
