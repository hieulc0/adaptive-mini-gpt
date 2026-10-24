"""Unit tests for Milestone M79: Active Experimentation and Epistemic Recognition.

Strictly fast, CPU-only tests verifying logic and invariants without heavy GPU operations.
"""

from __future__ import annotations

import pytest

from adaptive_mini_gpt.active_experimentation.active_experiment_engine import (
    ActiveExperimentEngine,
    EpistemicState,
    ProbeSelectionStrategy,
    VerifiedTeacherInterface,
)
from adaptive_mini_gpt.active_experimentation.audit_m78 import (
    M78AuditReport,
    M78ForensicAuditor,
)
from adaptive_mini_gpt.active_experimentation.causal_active_benchmark import (
    CausalActiveBenchmark,
)
from adaptive_mini_gpt.neural_skill_composition.primitive_skills import PrimitiveSkillType


def test_m78_forensic_audit_mathematical_proof() -> None:
    """Verify that R o A is mathematically equivalent to A o R across Z_16."""
    is_equiv = M78ForensicAuditor.verify_functional_equivalence(vocab_size=16)
    assert is_equiv is True

    audit = M78ForensicAuditor.execute_audit()
    assert audit.status == "AUDIT_VERIFIED"
    assert audit.reconciled_metrics["M78_AUTONOMOUS_EXACT_MATCH"] == 0.8885
    assert audit.reconciled_metrics["M78_AUTONOMOUS_PLAN_ACCURACY"] == 0.8000
    assert audit.reconciled_metrics["M78_AUTONOMOUS_FUNCTIONAL_ACCURACY"] == 0.9250
    assert audit.decomposition_80_tasks["functionally_equivalent_plan_count"] == 10
    assert "Q.E.D." in audit.mathematical_proof_functional_equivalence


def test_epistemic_state_classification() -> None:
    """Test classification of the learner's epistemic confidence states."""
    # 1. High likelihood, high margin -> Confident
    state_conf = ActiveExperimentEngine.evaluate_epistemic_state(
        best_ll=-0.50,
        best_token_acc=0.95,
        evidence_margin=0.25,
        top_confidence=0.92,
    )
    assert state_conf == EpistemicState.KNOWN_CONFIDENT

    # 2. High likelihood, low margin (< 0.08) -> Ambiguous
    state_amb = ActiveExperimentEngine.evaluate_epistemic_state(
        best_ll=-0.60,
        best_token_acc=0.90,
        evidence_margin=0.03,
        top_confidence=0.55,
    )
    assert state_amb == EpistemicState.KNOWN_AMBIGUOUS

    # 3. Low top likelihood (< -1.65) -> Unsupported skill gap
    state_unsupp = ActiveExperimentEngine.evaluate_epistemic_state(
        best_ll=-2.80,
        best_token_acc=0.30,
        evidence_margin=0.01,
        top_confidence=0.20,
    )
    assert state_unsupp == EpistemicState.SKILL_GAP_UNSUPPORTED

    # 4. Low token accuracy (< 0.60) -> Unsupported skill gap
    state_low_acc = ActiveExperimentEngine.evaluate_epistemic_state(
        best_ll=-1.20,
        best_token_acc=0.45,
        evidence_margin=0.15,
        top_confidence=0.85,
    )
    assert state_low_acc == EpistemicState.SKILL_GAP_UNSUPPORTED


def test_verified_teacher_budget_enforcement() -> None:
    """Test that the teacher interface tracks queries and enforces budget bounds."""
    mock_fn = lambda seq: tuple((s + 1) % 16 for s in seq)
    teacher = VerifiedTeacherInterface(ground_truth_map_fn=mock_fn, max_budget=2)

    assert teacher.queries_used == 0

    # Query 1
    out1 = teacher.query_oracle((1, 2, 3))
    assert out1 == (2, 3, 4)
    assert teacher.queries_used == 1

    # Query 2
    out2 = teacher.query_oracle((5, 6, 7))
    assert out2 == (6, 7, 8)
    assert teacher.queries_used == 2

    # Query 3 should exceed budget
    with pytest.raises(RuntimeError, match="Teacher query budget exceeded"):
        teacher.query_oracle((8, 9, 10))


def test_functional_plan_equivalence_checker() -> None:
    """Test detection of functional equivalence between (R, A) and (A, R)."""
    r = PrimitiveSkillType.COMPLEMENT_REFLECT.value
    a = PrimitiveSkillType.AFFINE_TRANSFORM.value
    s = PrimitiveSkillType.CAESAR_SHIFT_MOD3.value

    # Identity
    assert CausalActiveBenchmark.is_functionally_equivalent_plan((r, s), (r, s)) is True

    # Functional Equivalence: (R, A) == (A, R)
    assert CausalActiveBenchmark.is_functionally_equivalent_plan((r, a), (a, r)) is True
    assert CausalActiveBenchmark.is_functionally_equivalent_plan((a, r), (r, a)) is True

    # Non-commutative pairs: (R, S) != (S, R)
    assert CausalActiveBenchmark.is_functionally_equivalent_plan((r, s), (s, r)) is False

    # Distinct pairs
    assert CausalActiveBenchmark.is_functionally_equivalent_plan((r, s), (a, s)) is False
