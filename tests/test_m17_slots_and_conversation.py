"""Unit and Integration Tests for Grounded Slot Binding and Conversational Memory (M17)."""

from __future__ import annotations

import pytest

from adaptive_mini_gpt.rules import (
    AbstentionException,
    AmbiguousReferenceError,
    ConversationMemory,
    DeterministicTemplatePlanner,
    ExecutionPlan,
    GroundedSlot,
    PlanExecutionResult,
    PlanExecutor,
    PlanStep,
    PlanValidationError,
    ReasoningController,
    SlotExtractor,
    WorkingMemory,
    format_available_slots,
    generate_conversational_problems,
    generate_multistep_problems,
    parse_plan,
)


def test_grounded_slot_registry() -> None:
    """Verify GroundedSlot creation, registration in WorkingMemory, and serialization."""
    mem = WorkingMemory()
    slot1 = GroundedSlot(
        slot_id="distance_1",
        value=120.0,
        unit="km",
        source_text="120 km",
        start_idx=14,
        end_idx=20,
        extraction_method="deterministic",
        verified=True,
    )
    mem.bind_slot(slot1)
    assert mem.has_slot("distance_1")
    assert not mem.has_slot("distance_2")
    assert mem.get_slot("distance_1") == slot1
    assert mem.get_slot("nonexistent") is None

    mem_dict = mem.to_dict()
    assert "slots" in mem_dict
    assert "distance_1" in mem_dict["slots"]
    assert mem_dict["slots"]["distance_1"]["value"] == 120.0
    assert mem_dict["slots"]["distance_1"]["unit"] == "km"


def test_slot_extractor_kinematics() -> None:
    """Verify slot extraction from 2-segment and 3-segment journey descriptions with units."""
    q2 = "A car travels 120 km in 2 hours, then 90 km in 3 hours. What is its overall average speed?"
    slots2 = SlotExtractor.extract(q2)
    assert len(slots2) == 4
    assert slots2["distance_1"].value == 120.0
    assert slots2["distance_1"].unit == "km"
    assert slots2["duration_1"].value == 2.0
    assert slots2["duration_1"].unit == "h"
    assert slots2["distance_2"].value == 90.0
    assert slots2["distance_2"].unit == "km"
    assert slots2["duration_2"].value == 3.0
    assert slots2["duration_2"].unit == "h"

    # Inverted order
    q_inv = "A cyclist takes 2 hours to cover 60 km, then 3 hours to cover 90 km."
    slots_inv = SlotExtractor.extract(q_inv)
    assert slots_inv["distance_1"].value == 60.0
    assert slots_inv["duration_1"].value == 2.0
    assert slots_inv["distance_2"].value == 90.0
    assert slots_inv["duration_2"].value == 3.0


def test_slot_extractor_arithmetic_and_distractor_filtering() -> None:
    """Verify arithmetic operand extraction and contextual year filtering."""
    q = "In the year 2024, calculate the sum of 150, 250, and 350."
    slots = SlotExtractor.extract(q)
    assert "operand_1" in slots
    assert "operand_2" in slots
    assert "operand_3" in slots
    assert slots["operand_1"].value == 150
    assert slots["operand_2"].value == 250
    assert slots["operand_3"].value == 350
    assert "operand_4" not in slots


def test_slot_extractor_abstentions() -> None:
    """Verify safe abstention on dimensional mismatches, non-positive times, and negative distances."""
    # Dimensional mismatch
    with pytest.raises(AbstentionException, match="Incompatible physical dimensions"):
        SlotExtractor.extract("What is 120 km plus 3 hours?")

    # Non-positive duration
    with pytest.raises(AbstentionException, match="violates physical kinematics"):
        SlotExtractor.extract("A car travels 100 km in 0 hours, then 50 km in 2 hours.")

    # Negative distance
    with pytest.raises(AbstentionException, match="violates physical constraints"):
        SlotExtractor.extract("A vehicle travels -50 km in 2 hours, then 80 km in 3 hours.")


def test_plan_executor_slot_injection() -> None:
    """Verify execution of structural plan by injecting values from GroundedSlot registry."""
    slots = {
        "distance_1": GroundedSlot("distance_1", 120.0, "km"),
        "duration_1": GroundedSlot("duration_1", 2.0, "h"),
        "distance_2": GroundedSlot("distance_2", 90.0, "km"),
        "duration_2": GroundedSlot("duration_2", 3.0, "h"),
    }
    mem = WorkingMemory()
    for s in slots.values():
        mem.bind_slot(s)

    plan = ExecutionPlan(
        steps=[
            PlanStep(output_var="d_total", operation="add", args={"a": "$distance_1", "b": "$distance_2"}, units="km"),
            PlanStep(output_var="t_total", operation="add", args={"a": "$duration_1", "b": "$duration_2"}, units="h"),
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
    res = executor.execute(plan, working_memory=mem)

    assert res.success
    assert res.final_result == "42 km/h"
    assert res.step_results["d_total"] == 210.0
    assert res.step_results["t_total"] == 5.0
    assert res.executed_steps == 3


def test_plan_executor_rejects_unknown_slot_id() -> None:
    """Verify that referencing an nonexistent slot identifier is rejected by the validator."""
    slots = {"distance_1": GroundedSlot("distance_1", 100.0, "km")}
    mem = WorkingMemory()
    for s in slots.values():
        mem.bind_slot(s)

    plan = ExecutionPlan(
        steps=[PlanStep(output_var="d", operation="add", args={"a": "$distance_1", "b": "$distance_99"})],
        final_var="d",
    )

    executor = PlanExecutor()
    res = executor.execute(plan, working_memory=mem)
    assert not res.success
    assert "unknown slot or undefined variable '$distance_99'" in (res.error or "")


def test_plan_executor_unit_mismatch_with_slots() -> None:
    """Verify that adding distance slot (km) and time slot (h) is rejected as a unit mismatch."""
    slots = {
        "distance_1": GroundedSlot("distance_1", 120.0, "km"),
        "duration_1": GroundedSlot("duration_1", 2.0, "h"),
    }
    mem = WorkingMemory()
    for s in slots.values():
        mem.bind_slot(s)

    plan = ExecutionPlan(
        steps=[PlanStep(output_var="bad_add", operation="add", args={"a": "$distance_1", "b": "$duration_1"})],
        final_var="bad_add",
    )

    executor = PlanExecutor()
    res = executor.execute(plan, working_memory=mem)
    assert not res.success
    assert "Unit mismatch in addition: 'km' vs 'h'" in (res.error or "")


def test_positional_and_slot_parsing() -> None:
    """Verify that parse_plan handles positional args and $slot identifiers correctly."""
    text = (
        "Plan:\n"
        "1. d_total = add($distance_1, $distance_2) [km]\n"
        "2. t_total = add($duration_1, $duration_2) [h]\n"
        "3. speed = average_speed($d_total, $t_total) [km/h]\n"
        "Result: $speed"
    )
    plan = parse_plan(text)
    assert len(plan.steps) == 3
    assert plan.steps[0].args == {"a": "$distance_1", "b": "$distance_2"}
    assert plan.steps[1].args == {"a": "$duration_1", "b": "$duration_2"}
    assert plan.steps[2].args == {"distance_km": "$d_total", "time_hours": "$t_total"}
    assert plan.final_var == "speed"


def test_deterministic_template_planner_with_slots() -> None:
    """Verify deterministic template planner slot generation."""
    q = "A car travels 120 km in 2 hours, then 90 km in 3 hours. What is its overall average speed?"
    plan, slots = DeterministicTemplatePlanner.plan_with_slots(q)

    assert len(slots) == 4
    assert len(plan.steps) == 3
    assert plan.steps[0].args["a"] == "$distance_1"
    assert plan.steps[0].args["b"] == "$distance_2"

    executor = PlanExecutor()
    mem = WorkingMemory()
    for s in slots.values():
        mem.bind_slot(s)
    res = executor.execute(plan, working_memory=mem)
    assert res.success
    assert res.final_result == "42 km/h"


def test_conversational_working_memory_2turn() -> None:
    """Verify multi-turn conversational reasoning where Turn 2 references Turn 1 verified quantities."""
    conv_mem = ConversationMemory(session_id="test_session_1")
    controller = ReasoningController()

    # Turn 1
    t1_q = "A car travels 120 km in 2 hours. What was its average speed?"
    t1_slots = {
        "distance_1": GroundedSlot("distance_1", 120.0, "km"),
        "duration_1": GroundedSlot("duration_1", 2.0, "h"),
    }
    t1_plan = ExecutionPlan(
        steps=[PlanStep(output_var="speed_1", operation="average_speed", args={"distance_km": "$distance_1", "time_hours": "$duration_1"}, units="km/h")],
        final_var="speed_1",
    )
    t1_res = controller.solve_conversational_turn(
        question=t1_q,
        conversation_memory=conv_mem,
        expected_answer="60 km/h",
        expected_plan=t1_plan,
        slots=t1_slots,
    )
    assert t1_res["status"] == "success"
    assert t1_res["answer"] == "60 km/h"

    # Turn 2: uses Turn 1 distance and time
    t2_q = "It then traveled another 90 km in 3 hours. What was the average speed for the whole journey?"
    t2_slots = {
        "distance_2": GroundedSlot("distance_2", 90.0, "km"),
        "duration_2": GroundedSlot("duration_2", 3.0, "h"),
    }
    t2_plan = ExecutionPlan(
        steps=[
            PlanStep(output_var="d_total", operation="add", args={"a": "$turn1.distance_1", "b": "$distance_2"}, units="km"),
            PlanStep(output_var="t_total", operation="add", args={"a": "$turn1.duration_1", "b": "$duration_2"}, units="h"),
            PlanStep(output_var="overall_speed", operation="average_speed", args={"distance_km": "$d_total", "time_hours": "$t_total"}, units="km/h"),
        ],
        final_var="overall_speed",
    )
    t2_res = controller.solve_conversational_turn(
        question=t2_q,
        conversation_memory=conv_mem,
        expected_answer="42 km/h",
        expected_plan=t2_plan,
        slots=t2_slots,
    )
    assert t2_res["status"] == "success"
    assert t2_res["answer"] == "42 km/h"


def test_conversational_ambiguous_referent_abstention() -> None:
    """Verify that ambiguous pronouns across multiple prior values safely trigger abstention."""
    conv_mem = ConversationMemory(session_id="test_session_amb")

    # Turn 1 has 2 distance quantities
    turn1 = conv_mem.start_turn("First turn with 2 distances")
    turn1.working_memory.bind_step_variable("d1", 50.0, "add", {}, {}, units="km")
    turn1.working_memory.bind_step_variable("d2", 80.0, "add", {}, {}, units="km")

    # Attempting to resolve "previous distance" should raise AmbiguousReferenceError
    with pytest.raises(AmbiguousReferenceError, match="Ambiguous distance reference"):
        conv_mem.resolve_reference("previous_distance")


def test_multistep_dataset_slots_generation() -> None:
    """Verify that generate_multistep_problems produces both literal and slot plans for all archetypes."""
    problems = generate_multistep_problems(num_samples_per_archetype=3, seed=42, split="train")
    for p in problems:
        assert p.slots is not None
        assert p.expected_slot_plan is not None
        pair = p.to_slot_plan_pair()
        assert "Available slots:" in pair
        assert "<|assistant|>" in pair
