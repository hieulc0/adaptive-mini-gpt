"""Unit and Integration Tests for Grounded Argument Extraction and Working Memory (M15)."""

from __future__ import annotations

import pytest

from adaptive_mini_gpt.rules import (
    DeterministicArgumentGrounder,
    DeterministicRuleSelector,
    ReasoningController,
    RuleExecutor,
    SpanCopyArgumentGrounder,
    WorkingMemory,
    generate_abstention_problems,
    generate_rule_problems,
)


def test_working_memory_schema_and_provenance() -> None:
    """Verify working memory state management, provenance binding, and JSON serialization."""
    mem = WorkingMemory()
    assert mem.status == "empty"
    assert not mem.verified

    mem.bind_operation("add")
    assert mem.operation == "add"
    assert mem.status == "operation_selected"

    mem.bind_argument("a", 116, "116", (8, 11), "int", "regex_span")
    mem.bind_argument("b", 722, "722", (17, 20), "int", "regex_span")

    assert mem.status == "grounded"
    assert mem.is_complete_for_rule(["a", "b"])
    assert not mem.is_complete_for_rule(["a", "b", "c"])

    prov_a = mem.provenance["a"]
    assert prov_a.slot_name == "a"
    assert prov_a.source_text == "116"
    assert prov_a.start_idx == 8
    assert prov_a.end_idx == 11
    assert prov_a.parsed_value == 116
    assert prov_a.type_name == "int"

    mem.bind_result(838, verified=True)
    assert mem.verified
    assert mem.result == 838
    assert mem.status == "verified"

    d = mem.to_dict()
    assert d["operation"] == "add"
    assert d["arguments"] == {"a": 116, "b": 722}
    assert d["verified"] is True
    assert d["result"] == 838
    assert "a" in d["provenance"]
    assert d["provenance"]["a"]["span"] == [8, 11]


def test_deterministic_rule_selector() -> None:
    """Verify syntactic intent classification across all 4 rules and rejection of unknown intents."""
    assert DeterministicRuleSelector.select_rule("Write 'photon' backward.") == "reverse"
    assert DeterministicRuleSelector.select_rule("What is 116 plus 722?") == "add"
    assert DeterministicRuleSelector.select_rule("A train travels 120 km in 2 hours. Calculate average speed.") == "average_speed"
    assert (
        DeterministicRuleSelector.select_rule(
            "If Anna is taller than Ben and Ben is taller than Chris, who is taller, Anna or Chris?"
        )
        == "compare"
    )

    with pytest.raises(Exception):
        DeterministicRuleSelector.select_rule("What is the capital of France?")


def test_grounder_reverse() -> None:
    """Verify reverse argument extraction with exact character span tracking."""
    # Quoted string
    mem1 = DeterministicArgumentGrounder.ground("reverse", "Write 'k8f2a' backward.")
    assert mem1.status == "grounded"
    assert mem1.arguments == {"s": "k8f2a"}
    assert mem1.provenance["s"].source_text == "k8f2a"
    span1 = (mem1.provenance["s"].start_idx, mem1.provenance["s"].end_idx)
    assert "Write 'k8f2a' backward."[span1[0] : span1[1]] == "k8f2a"

    # Missing string sequence
    mem_bad = DeterministicArgumentGrounder.ground("reverse", "Spell the word in reverse order.")
    assert mem_bad.status == "rejected"
    assert "Missing" in (mem_bad.error or "")


def test_grounder_add_ordering_and_distractors() -> None:
    """Verify addition argument extraction under changed ordering and distracting numbers."""
    # Standard: X plus Y
    mem1 = DeterministicArgumentGrounder.ground("add", "What is 116 plus 722?")
    assert mem1.status == "grounded"
    assert mem1.arguments == {"a": 116, "b": 722}

    # Changed argument ordering: Add B to A
    mem2 = DeterministicArgumentGrounder.ground("add", "Add 722 to 116.")
    assert mem2.status == "grounded"
    assert mem2.arguments == {"a": 722, "b": 116}

    # Distracting numbers: year 2024 should be filtered out
    mem3 = DeterministicArgumentGrounder.ground("add", "In the year 2024, what is 116 plus 722?")
    assert mem3.status == "grounded"
    assert mem3.arguments == {"a": 116, "b": 722}
    span_a = (mem3.provenance["a"].start_idx, mem3.provenance["a"].end_idx)
    assert "In the year 2024, what is 116 plus 722?"[span_a[0] : span_a[1]] == "116"

    # Missing operand
    mem4 = DeterministicArgumentGrounder.ground("add", "What is 116 plus?")
    assert mem4.status == "rejected"


def test_grounder_compare_multiword_and_cycles() -> None:
    """Verify transitive comparison with multiword names and rejection of cycles."""
    # Multiword entity names
    q = (
        "If Mary Jane is taller than Uncle Bob and Uncle Bob is taller than Big Brother, "
        "who is taller, Mary Jane or Big Brother?"
    )
    mem = DeterministicArgumentGrounder.ground("compare", q)
    assert mem.status == "grounded"
    assert mem.arguments["entity_a"] == "Mary Jane"
    assert mem.arguments["entity_b"] == "Big Brother"
    assert "Mary Jane > Uncle Bob" in mem.arguments["relations"]
    assert "Uncle Bob > Big Brother" in mem.arguments["relations"]

    # Contradictory cycle
    q_cycle = "If Anna is taller than Ben and Ben is taller than Anna, who is taller, Anna or Ben?"
    mem_cycle = DeterministicArgumentGrounder.ground("compare", q_cycle)
    # Execution will reject cycle
    executor = RuleExecutor()
    res = executor.execute("compare", mem_cycle.arguments, verify=True)
    assert not res.success
    assert "cyclic" in (res.error or "").lower()


def test_grounder_average_speed_units_and_preconditions() -> None:
    """Verify average speed extraction across different unit expressions and precondition rejection."""
    # Variant units: kilometers and hrs
    q1 = "A car covers a distance of 150 kilometers in 3 hrs. What is its average speed?"
    mem1 = DeterministicArgumentGrounder.ground("average_speed", q1)
    assert mem1.status == "grounded"
    assert mem1.arguments == {"distance_km": 150.0, "time_hours": 3.0}

    # Changed ordering
    q2 = "Taking 2 h to cover 120 km, what is the vehicle's average speed?"
    mem2 = DeterministicArgumentGrounder.ground("average_speed", q2)
    assert mem2.status == "grounded"
    assert mem2.arguments == {"distance_km": 120.0, "time_hours": 2.0}

    # Precondition failure: time <= 0
    q_zero = "A car travels 120 km in 0 hours. Calculate average speed."
    mem_zero = DeterministicArgumentGrounder.ground("average_speed", q_zero)
    assert mem_zero.status == "rejected"
    assert "Invalid duration" in (mem_zero.error or "")


def test_span_copy_grounder() -> None:
    """Verify span-copy extraction reproduces exact substrings without token generation."""
    q_add = "Calculate the sum of 345 and 678."
    mem = SpanCopyArgumentGrounder.ground("add", q_add)
    assert mem.status == "grounded"
    assert mem.arguments == {"a": 345, "b": 678}
    assert mem.provenance["a"].extraction_method == "span_copy"

    q_rev = "Please reverse the characters of 'abcdef'."
    mem_rev = SpanCopyArgumentGrounder.ground("reverse", q_rev)
    assert mem_rev.status == "grounded"
    assert mem_rev.arguments == {"s": "abcdef"}


def test_reasoning_controller_grounded_end_to_end() -> None:
    """Verify end-to-end grounded reasoning execution with verified outcomes."""
    controller = ReasoningController()

    # 1. Addition
    res_add = controller.solve(
        question="What is 116 plus 722?",
        expected_rule="add",
        expected_answer="838",
        ground_truth_args={"a": 116, "b": 722},
        mode="deterministic",
    )
    assert res_add.status == "success"
    assert res_add.answer_correct
    assert res_add.final_answer == "838"
    assert res_add.working_memory is not None
    assert res_add.working_memory.verified

    # 2. Reverse
    res_rev = controller.solve(
        question="Write 'photon' backward.",
        expected_rule="reverse",
        expected_answer="notohp",
        ground_truth_args={"s": "photon"},
        mode="span_copy",
    )
    assert res_rev.status == "success"
    assert res_rev.answer_correct
    assert res_rev.final_answer == "notohp"
    assert res_rev.working_memory is not None
    assert res_rev.working_memory.provenance["s"].source_text == "photon"


def test_abstention_on_malformed_inputs() -> None:
    """Verify that malformed or contradictory inputs are safely rejected/abstained."""
    controller = ReasoningController()
    abstention_problems = generate_abstention_problems(num_samples=10, seed=42)
    assert len(abstention_problems) > 0

    for prob in abstention_problems:
        res = controller.solve(
            question=prob.question,
            expected_rule=prob.rule_name,
            expected_answer=prob.expected_answer,
            ground_truth_args=prob.args,
            mode="deterministic",
            should_abstain=True,
        )
        assert res.abstained is True
        assert res.abstention_correct is True
        assert res.status == "abstained"
