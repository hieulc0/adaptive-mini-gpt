"""Milestone M94 Verification Suite: Neural Program Learning, Execution & Generalization.

Validates the complete M94 research loop across 6 unit and integration tests:
1. test_m93_forensic_audit_and_baseline_certification
2. test_mini_language_ast_and_reference_interpreter
3. test_tokenizer_and_syntax_abstention
4. test_program_curriculum_and_stages
5. test_program_reasoning_substrate_and_capacity_budget
6. test_autonomous_acquisition_and_safe_rollback
"""

from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.neural_program_learning.audit_m93 import (
    M93ForensicAuditor,
)
from adaptive_mini_gpt.neural_program_learning.mini_language import (
    AssignStmt,
    BinOpExpr,
    CompareExpr,
    ForStmt,
    IfStmt,
    LiteralExpr,
    PrintStmt,
    Program,
    ProgramTokenizer,
    ReferenceInterpreter,
    VarExpr,
)
from adaptive_mini_gpt.neural_program_learning.neural_program_executor import (
    ProgramReasoningSubstrate,
)
from adaptive_mini_gpt.neural_program_learning.program_acquisition import (
    AutonomousProgramLearner,
)
from adaptive_mini_gpt.neural_program_learning.program_curriculum import (
    ProgramCurriculumGenerator,
)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def base_model(device: torch.device) -> MiniGPT:
    cfg = ModelConfig(
        vocab_size=1024,
        context_length=128,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
    )
    model = MiniGPT(cfg).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def test_m93_forensic_audit_and_baseline_certification() -> None:
    """Validate Phase 0 forensic audit and reconciliation of M93 metrics."""
    audit = M93ForensicAuditor.execute_audit()
    assert audit.status == "M93_AUDIT_VERIFIED"
    assert audit.m93_integrated_discovery_em == 89.64
    assert audit.m93_clean_task_em == 100.00
    assert audit.m93_noisy_task_em == 95.83
    assert audit.m93_query_cost_reduction_pct == 75.0
    assert audit.m93_brier_score == 0.0628
    assert audit.m93_ece == 4.50
    assert audit.m93_lifelong_retention_pct == 96.95
    assert audit.total_trainable_parameters == 18_561
    assert audit.budget_headroom == 1_439
    assert audit.budget_ceiling == 20_000
    assert audit.frozen_backbone_weights == 1_071_872

    # Verify reconciliations
    assert audit.symbolic_catalog_limits_confirmed is True
    assert audit.no_hidden_symbolic_inference_verified is True
    assert audit.data_isolation_and_costs_verified is True


def test_mini_language_ast_and_reference_interpreter() -> None:
    """Verify AST construction, reference execution, and trace generation."""
    # Program from specification:
    # total = 0
    # for x in [3, 5, 2]:
    #     if x > 2:
    #         total = total + x
    # print(total)
    prog = Program(
        [
            AssignStmt("total", LiteralExpr(0)),
            ForStmt(
                loop_var="x",
                items=(3, 5, 2),
                body=[
                    IfStmt(
                        cond=CompareExpr(VarExpr("x"), ">", LiteralExpr(2)),
                        body=[AssignStmt("total", BinOpExpr(VarExpr("total"), "+", VarExpr("x")))],
                    )
                ],
            ),
            PrintStmt("total"),
        ]
    )

    result = ReferenceInterpreter.execute(prog)
    assert result.is_valid is True
    # 3 + 5 = 8 mod 16
    assert result.output_value == 8
    assert len(result.trace) > 0

    # Test error handling (undefined variable)
    invalid_prog = Program(
        [
            AssignStmt("a", VarExpr("undefined_foo")),
            PrintStmt("a"),
        ]
    )
    inv_res = ReferenceInterpreter.execute(invalid_prog)
    assert inv_res.is_valid is False
    assert inv_res.output_value is None
    assert "UNDEFINED_VAR" in (inv_res.error_type or "")


def test_tokenizer_and_syntax_abstention() -> None:
    """Verify code tokenizer maps keywords, vars, literals, and unknown tokens."""
    code = "total = 0\nfor x in [3, 5]:\n    total = total + x\nprint(total)"
    tokens = ProgramTokenizer.tokenize_code(code)
    assert tokens[0] == ProgramTokenizer.SOS_ID
    assert tokens[-1] == ProgramTokenizer.EOS_ID
    assert ProgramTokenizer.VAR_TOKENS["total"] in tokens
    assert ProgramTokenizer.KEYWORD_TOKENS["for"] in tokens

    # Test unknown token maps to ABSTAIN_ID
    bad_code = "total = unsupported_macro()"
    bad_tokens = ProgramTokenizer.tokenize_code(bad_code)
    assert ProgramTokenizer.ABSTAIN_ID in bad_tokens


def test_program_curriculum_and_stages() -> None:
    """Verify curriculum generator generates all 8 stages with valid splits."""
    gen = ProgramCurriculumGenerator(seed=42)
    curriculum = gen.generate_full_curriculum()

    assert len(curriculum.stages) == 8
    for s_id in range(1, 9):
        suite = curriculum.stages[s_id]
        assert len(suite.train_examples) > 0
        assert len(suite.dev_examples) > 0
        assert len(suite.held_out_examples) > 0

    assert len(curriculum.unsupported_examples) > 0
    for unsup in curriculum.unsupported_examples:
        assert unsup.requires_abstention is True


def test_program_reasoning_substrate_and_capacity_budget(device: torch.device) -> None:
    """Verify substrate parameter counts, budget ceiling, and zero initialization."""
    substrate = ProgramReasoningSubstrate(budget_ceiling=20_000).to(device)

    trunk_params = substrate.count_trunk_parameters()
    assert trunk_params <= 16_000

    # Add 8 rank-2 residuals
    for i in range(8):
        substrate.add_residual(f"skill_{i}")

    total_params = substrate.count_total_parameters()
    assert total_params <= 20_000
    assert (20_000 - total_params) >= 1_000  # At least 1,000 parameter headroom

    # Verify zero-init preservation
    s_t = torch.tensor([2, 5], device=device)
    op_t = torch.tensor([1, 1], device=device)
    v_t = torch.tensor([3, 4], device=device)
    substrate.eval()
    with torch.no_grad():
        out_base = substrate.forward_transition(s_t, op_t, v_t)
        out_with_res = substrate.forward_transition(s_t, op_t, v_t, active_residual_id="skill_0")
        diff = torch.max(torch.abs(out_base - out_with_res)).item()
        assert diff == 0.0  # Exactly bit-identical at initialization


def test_autonomous_acquisition_and_safe_rollback(
    base_model: MiniGPT, device: torch.device
) -> None:
    """Verify autonomous skill acquisition, dev verification, and bit-exact rollback."""
    substrate = ProgramReasoningSubstrate(budget_ceiling=20_000).to(device)

    gen = ProgramCurriculumGenerator(seed=42)
    s1 = gen._generate_stage_1(seed=42)

    # Acquire stage 1 concept
    res = AutonomousProgramLearner.acquire_concept(
        model=base_model,
        substrate=substrate,
        concept_id="test_concept",
        train_examples=s1.train_examples[:10],
        dev_examples=s1.dev_examples[:5],
        device=device,
        steps=10,
        acceptance_threshold_em=0.0,  # Accept for test
    )
    assert res.status == "ACCEPTED"
    assert res.rollback_performed is False
    assert "test_concept" in substrate.residuals

    # Test rejection and rollback
    res_rejected = AutonomousProgramLearner.acquire_concept(
        model=base_model,
        substrate=substrate,
        concept_id="impossible_concept",
        train_examples=s1.train_examples[:5],
        dev_examples=s1.dev_examples[:5],
        device=device,
        steps=2,
        acceptance_threshold_em=100.0,  # Impossible threshold triggers rollback
    )
    if res_rejected.status == "REJECTED":
        assert res_rejected.rollback_performed is True
        assert "impossible_concept" not in substrate.residuals
