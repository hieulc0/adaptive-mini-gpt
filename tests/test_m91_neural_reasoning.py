"""Unit and Integration Tests for Milestone M91: Autonomous Multi-Step Neural Reasoning.

Verifies:
1. Phase 0 forensic audit and parameter reconciliation (10,304 vs 10,624).
2. Persistent substrate parameter budget accounting (13,400 <= 20,000 ceiling).
3. Zero-initialization function-preservation of compressed modular residuals.
4. Multi-step grammar, step traces, and ground-truth evaluator.
5. Curriculum generator partitions (train, dev, qual, test disjoint sets).
6. Autonomous autoregressive execution and step corruption divergence.
7. End-to-end benchmark smoke test and artifact generation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.concept_instruction.instruction_grammar import Opcode, ParaphraseOpcode
from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.neural_reasoning.audit_m90 import (
    M90ForensicAuditor,
)
from adaptive_mini_gpt.neural_reasoning.benchmark_m91 import (
    M91ReasoningBenchmark,
)
from adaptive_mini_gpt.neural_reasoning.persistent_substrate import (
    CompressedResidual,
    PersistentReasoningSubstrate,
)
from adaptive_mini_gpt.neural_reasoning.reasoning_curriculum import (
    M91ReasoningCurriculumGenerator,
)
from adaptive_mini_gpt.neural_reasoning.reasoning_engine import (
    ReasoningEngine,
)
from adaptive_mini_gpt.neural_reasoning.reasoning_grammar import (
    MultiStepEvaluator,
    MultiStepProblem,
    StepRecord,
)


def test_m90_forensic_audit_and_parameter_reconciliation() -> None:
    """Verify Phase 0 forensic audit resolves the 10,304 vs 10,624 ambiguity."""
    audit = M90ForensicAuditor.execute_audit()
    assert audit.status == "M90_AUDIT_VERIFIED"
    assert audit.reconciled_pure_trunk_params == 10_304
    assert audit.reconciled_task_embedding_trunk_params == 10_624
    assert audit.task_embedding_overhead_params == 320
    assert audit.frozen_backbone_weights == 1_071_872

    # Verify hybrid planned capacity
    assert audit.hybrid_capacity_planned_params == 13_400
    assert audit.hybrid_capacity_headroom == 6_600
    assert audit.hybrid_capacity_planned_params <= audit.budget_ceiling
    assert audit.parameter_reconciliation_verified is True


def test_persistent_substrate_budget_and_function_preservation() -> None:
    """Verify hybrid substrate adheres to capacity ceiling and achieves zero-init function preservation."""
    substrate = PersistentReasoningSubstrate(
        d_model=128,
        d_shared=32,
        num_symbols=16,
        rank_residual=2,
        budget_ceiling=20_000,
    )

    # Pure trunk params
    assert substrate.count_trunk_parameters() == 10_304
    assert substrate.count_residual_parameters() == 0
    assert substrate.count_total_parameters() == 10_304

    # Add single residual
    substrate.add_residual("task_0")
    res = substrate.residuals["task_0"]
    assert isinstance(res, CompressedResidual)
    # rank 2 residual: down 128*2=256, norm 4, up 2*128=256 -> 516
    assert res.count_parameters() == 516
    assert substrate.count_residual_parameters() == 516
    assert substrate.count_total_parameters() == 10_304 + 516

    # Test function preservation at init (up projection initialized to zeros)
    dummy_h = torch.randn(2, 4, 128)
    out_trunk = substrate(dummy_h, task_id=None)
    out_with_res = substrate(dummy_h, task_id="task_0")
    max_diff = torch.max(torch.abs(out_trunk - out_with_res)).item()
    assert max_diff == 0.0, f"Expected exact function preservation at init, got max diff {max_diff}"

    # Add 5 more residuals (total 6 skills)
    for i in range(1, 6):
        substrate.add_residual(f"task_{i}")
    assert len(substrate.residuals) == 6
    assert substrate.count_residual_parameters() == 6 * 516  # 3,096
    assert substrate.count_total_parameters() == 13_400  # 10,304 + 3,096
    assert substrate.count_total_parameters() <= 20_000

    # Test freeze trunk behavior
    substrate.freeze_trunk()
    for p in substrate.down_proj.parameters():
        assert p.requires_grad is False
    for p in substrate.residuals["task_0"].parameters():
        assert p.requires_grad is True


def test_multi_step_grammar_and_evaluator() -> None:
    """Verify ground-truth multi-step evaluator and trace generation."""
    # Test step-by-step evaluation
    assert MultiStepEvaluator.evaluate_step(Opcode.ADD, 3, 5) == (5 + 3) % 16  # 8
    assert MultiStepEvaluator.evaluate_step(Opcode.SUB, 4, 2) == (2 - 4) % 16  # 14
    assert MultiStepEvaluator.evaluate_step(Opcode.SCALE, 2, 7) == (7 * 2) % 16  # 14
    assert MultiStepEvaluator.evaluate_step(Opcode.XOR, 5, 12) == 12 ^ 5  # 9

    # Test multi-step chain: 7 -> Add 5 -> Scale 2 -> Sub 4
    chain = ((Opcode.ADD, 5), (Opcode.SCALE, 2), (Opcode.SUB, 4))
    steps, res = MultiStepEvaluator.evaluate_chain(7, chain)
    # 7 + 5 = 12
    # 12 * 2 = 24 % 16 = 8
    # 8 - 4 = 4
    assert res == 4
    assert len(steps) == 3
    assert steps[0].output_value == 12
    assert steps[1].output_value == 8
    assert steps[2].output_value == 4

    # Test conditional step
    # if val >= 8, val ^ k, else (val + k) % 16
    cond_res_high = MultiStepEvaluator.evaluate_step(Opcode.COND, 3, 10)
    assert cond_res_high == (10 ^ 3) % 16  # 9
    cond_res_low = MultiStepEvaluator.evaluate_step(Opcode.COND, 3, 5)
    assert cond_res_low == (5 + 3) % 16  # 8


def test_reasoning_curriculum_partitions() -> None:
    """Verify reasoning curriculum partitions have correct sizes and valid metadata."""
    generator = M91ReasoningCurriculumGenerator()
    suite = generator.generate_suite(seed=42)

    # Verify partition sizes across depths
    suites = [
        suite.one_step_suite,
        suite.two_step_suite,
        suite.three_step_suite,
        suite.four_step_suite,
        suite.five_step_suite,
        suite.conditional_suite,
        suite.held_out_operand_suite,
        suite.held_out_paraphrase_suite,
    ]
    for s in suites:
        assert len(s.train_problems) == 48
        assert len(s.dev_problems) == 16
        assert len(s.qual_problems) == 16
        assert len(s.test_problems) == 48

    # Verify chain lengths in suites
    assert suite.one_step_suite.test_problems[0].chain_length == 1
    assert suite.two_step_suite.test_problems[0].chain_length == 2
    assert suite.three_step_suite.test_problems[0].chain_length == 3
    assert suite.four_step_suite.test_problems[0].chain_length == 4
    assert suite.five_step_suite.test_problems[0].chain_length == 5
    assert suite.conditional_suite.test_problems[0].is_conditional is True


def test_autonomous_execution_and_causal_attribution() -> None:
    """Verify autonomous execution, teacher forcing, and causal perturbation."""
    cfg = ModelConfig(
        vocab_size=1024,
        context_length=128,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
    )
    model = MiniGPT(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    device = torch.device("cpu")
    substrate = PersistentReasoningSubstrate(
        d_model=128,
        d_shared=32,
        num_symbols=16,
        rank_residual=2,
    ).to(device)

    generator = M91ReasoningCurriculumGenerator()
    suite = generator.generate_suite(seed=42)
    sample_problem = suite.two_step_suite.test_problems[0]

    # Test autonomous execution returns valid ChainExecutionResult
    auto_res = ReasoningEngine.execute_autonomous(model, substrate, sample_problem, device)
    assert auto_res.chain_length == 2
    assert len(auto_res.predicted_steps) == 2
    assert len(auto_res.intermediate_accuracies) == 2

    # Test teacher forced execution
    tf_res = ReasoningEngine.execute_teacher_forced(model, substrate, sample_problem, device)
    assert tf_res.chain_length == 2
    assert len(tf_res.predicted_steps) == 2

    # Test corrupted step execution
    _, diverged = ReasoningEngine.execute_corrupted_step(
        model, substrate, sample_problem, corrupt_step_idx=0, noise=1, device=device
    )
    assert isinstance(diverged, bool)


def test_m91_benchmark_smoke_and_artifact_generation(tmp_path: Path) -> None:
    """Run an end-to-end smoke test of M91ReasoningBenchmark and verify artifact files."""
    cfg = ModelConfig(
        vocab_size=1024,
        context_length=128,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
    )
    model = MiniGPT(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    device = torch.device("cpu")
    out_dir = tmp_path / "m91_smoke"

    res = M91ReasoningBenchmark.run_benchmark(
        model=model,
        device=device,
        seeds=[42],
        output_dir=out_dir,
        epochs_trunk=2,
        lr_trunk=0.01,
    )

    assert res["milestone"] == "M91"
    assert res["budget_compliant"] is True
    assert res["trainable_parameters"] == 13_400
    assert res["budget_ceiling"] == 20_000

    # Verify generated report files
    json_path = out_dir / "m91_report.json"
    md_path = out_dir / "m91_report.md"
    assert json_path.exists(), "m91_report.json was not generated"
    assert md_path.exists(), "m91_report.md was not generated"

    content = md_path.read_text()
    assert "Milestone M91 Scientific Report" in content
    assert "Phase 0 — M90 Forensic Audit & Parameter Reconciliation" in content
