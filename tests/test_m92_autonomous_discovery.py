"""Unit and Integration Tests for Milestone M92: Autonomous Rule Discovery & Neural Acquisition.

Verifies:
1. Phase 0 forensic audit of Milestone M91 baseline.
2. Unknown rule environment task families, query budget, and partition isolation.
3. Active discovery engine hypothesis filtering, disagreement entropy, and calibrated abstention.
4. Neural skill acquisition, verification, and causal weight attribution.
5. Persistent knowledge retention and compositional multi-step transfer.
6. End-to-end benchmark execution and scientific report artifact generation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.autonomous_discovery.active_discovery_engine import (
    ActiveDiscoveryEngine,
    HypothesisSpace,
)
from adaptive_mini_gpt.autonomous_discovery.audit_m91 import (
    M91ForensicAuditor,
)
from adaptive_mini_gpt.autonomous_discovery.benchmark_m92 import (
    M92DiscoveryBenchmark,
)
from adaptive_mini_gpt.autonomous_discovery.neural_rule_learner import (
    NeuralRuleLearner,
)
from adaptive_mini_gpt.autonomous_discovery.persistent_knowledge_manager import (
    PersistentKnowledgeManager,
)
from adaptive_mini_gpt.autonomous_discovery.unknown_rule_environment import (
    ProceduralRule,
    RuleObservation,
    UnknownRuleEnvironment,
    UnknownRuleTask,
)
from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.neural_reasoning.persistent_substrate import (
    PersistentReasoningSubstrate,
)


def test_m91_forensic_audit_and_baseline_certification() -> None:
    """Verify Phase 0 forensic audit certifies M91 baselines."""
    audit = M91ForensicAuditor.execute_audit()
    assert audit.status == "M91_AUDIT_VERIFIED"
    assert audit.m91_two_step_em == 98.75
    assert audit.m91_three_step_em == 98.75
    assert audit.m91_five_step_em == 94.58
    assert audit.m91_continual_retention_em == 91.32
    assert audit.pure_trunk_params == 10_304
    assert audit.modular_residuals_params == 3_096
    assert audit.combined_trainable_params == 13_400
    assert audit.budget_headroom == 6_600
    assert audit.frozen_backbone_weights == 1_071_872
    assert audit.instruction_conditioning_distinction_verified is True
    assert audit.autonomous_discovery_imperative_verified is True


def test_unknown_rule_environment_and_partitions() -> None:
    """Verify unfamiliar task families, query budget, and partition isolation."""
    # Quadratic rule: y = (x^2 + 3) % 16
    r_quad = UnknownRuleEnvironment.create_quadratic_rule("test_quad", 1, 0, 3)
    assert r_quad.apply(0) == 3
    assert r_quad.apply(2) == (4 + 3) % 16  # 7
    assert r_quad.apply(4) == (16 + 3) % 16  # 3

    # Piecewise rule
    r_piece = UnknownRuleEnvironment.create_piecewise_rule("test_piece", thresh=8, m1=2, c1=1, m2=1, c2=0)
    assert r_piece.apply(3) == (2 * 3 + 1) % 16  # 7 (since 3 < 8)
    assert r_piece.apply(10) == 10 % 16  # 10 (since 10 >= 8)

    # Task creation and query budget
    task = UnknownRuleEnvironment.generate_task_from_rule("task_test", r_quad, seed=42)
    assert len(task.initial_observations) == 3
    assert len(task.dev_observations) == 3
    assert len(task.test_observations) == 10
    assert task.remaining_budget == 3

    # Query execution
    obs1 = task.oracle_query(5)
    assert obs1.x == 5
    assert obs1.y == r_quad.apply(5)
    assert task.queries_used == 1
    assert task.remaining_budget == 2

    # Query until budget exhausted
    task.oracle_query(6)
    task.oracle_query(7)
    assert task.remaining_budget == 0

    with pytest.raises(RuntimeError):
        task.oracle_query(8)


def test_active_discovery_engine_disagreement_and_abstention() -> None:
    """Verify active query selection resolves ambiguity and abstains on contradictions."""
    engine = ActiveDiscoveryEngine()

    # Create disambiguation pair
    r_a, r_b, inits = UnknownRuleEnvironment.create_disambiguation_pair("disambig_test")
    task_a = UnknownRuleEnvironment.generate_task_from_rule("task_da", r_a, fixed_inits=inits, seed=42)

    # Before query, both r_a and r_b are consistent on initial points
    candidates = engine.filter_consistent(engine.hypothesis_space.hypotheses, task_a.initial_observations)
    cand_ids = [c.rule_id for c in candidates]
    # Check that query selection finds point of disagreement
    probe_x, score = engine.select_informative_query(candidates, {obs.x for obs in task_a.initial_observations})
    assert score > 0.0

    # Run active discovery
    outcome = engine.discover_rule(task_a, strategy="active")
    assert outcome.decision == "EXECUTE"
    assert outcome.is_supported is True
    assert outcome.queries_used <= 3
    assert outcome.selected_rule is not None
    assert outcome.confidence > 0.50

    # Test calibrated abstention on contradictory task
    unsupported_task = UnknownRuleEnvironment.create_unsupported_task("unsupported_ctrl", seed=99)
    u_outcome = engine.discover_rule(unsupported_task, strategy="active")
    assert u_outcome.decision == "ABSTAIN"
    assert u_outcome.is_supported is False


def test_neural_rule_acquisition_and_causal_attribution() -> None:
    """Verify neural skill acquisition into modular residual and causal attribution."""
    cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = MiniGPT(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    device = torch.device("cpu")
    substrate = PersistentReasoningSubstrate(d_model=128, d_shared=32, num_symbols=16, rank_residual=2).to(device)

    rule = UnknownRuleEnvironment.create_quadratic_rule("quad_test", a=1, b=0, c=3)
    task = UnknownRuleEnvironment.generate_task_from_rule("task_acquire", rule, seed=42)

    # Perform queries to collect observations
    task.oracle_query(5)
    task.oracle_query(8)

    # Acquire skill with dev_threshold=0.0 to verify acceptance and attribution
    res = NeuralRuleLearner.acquire_skill(model, substrate, task, device=device, steps=20, dev_threshold=0.0)
    assert res.task_id == "task_acquire"
    assert res.status == "ACCEPTED"
    assert res.rollback_performed is False
    assert isinstance(res.causal_attribution_drop, float)
    assert substrate.count_total_parameters() == 10_304 + 516

    # Test rollback path when dev_threshold is unreachable (101.0)
    res_fail = NeuralRuleLearner.acquire_skill(model, substrate, task, device=device, steps=2, dev_threshold=101.0)
    assert res_fail.status == "REJECTED_DEV_FAILURE"
    assert res_fail.rollback_performed is True
    assert substrate.count_total_parameters() == 10_304 + 516  # previous residual preserved, rejected one removed


def test_persistent_knowledge_retention_and_compositional_transfer() -> None:
    """Verify multi-step compositional transfer and lifelong retention evaluation."""
    cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = MiniGPT(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    device = torch.device("cpu")
    substrate = PersistentReasoningSubstrate(d_model=128, d_shared=32, num_symbols=16, rank_residual=2).to(device)

    rule = UnknownRuleEnvironment.create_quadratic_rule("quad_comp", a=1, b=0, c=3)
    task = UnknownRuleEnvironment.generate_task_from_rule("task_comp", rule, seed=42)
    substrate.add_residual(task.task_id)

    # Test compositional transfer (discovered -> ADD_3)
    ct_res = PersistentKnowledgeManager.evaluate_compositional_transfer(
        model=model,
        substrate=substrate,
        task=task,
        downstream_operand=3,
        device=device,
    )
    assert ct_res.task_id == task.task_id
    assert ct_res.evaluated_cases == len(task.test_observations)
    assert isinstance(ct_res.compositional_exact_match, float)

    # Test retention report
    ret_report = PersistentKnowledgeManager.evaluate_lifelong_retention(
        model=model,
        substrate=substrate,
        admitted_tasks=[task],
        device=device,
        target_retention_pct=90.0,
    )
    assert ret_report.evaluated_skills_count == 1
    assert task.task_id in ret_report.individual_skill_accuracies


def test_m92_benchmark_smoke_and_artifact_generation(tmp_path: Path) -> None:
    """Run an end-to-end smoke test of M92DiscoveryBenchmark and verify generated artifacts."""
    cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = MiniGPT(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    device = torch.device("cpu")
    out_dir = tmp_path / "m92_smoke"

    res = M92DiscoveryBenchmark.run_benchmark(
        model=model,
        device=device,
        seeds=[42],
        output_dir=out_dir,
        epochs_trunk=2,
        lr_trunk=0.01,
    )

    assert res["milestone"] == "M92"
    assert res["budget_compliant"] is True
    assert res["budget_ceiling"] == 20_000

    # Verify generated report files
    json_path = out_dir / "m92_report.json"
    md_path = out_dir / "m92_report.md"
    assert json_path.exists(), "m92_report.json was not generated"
    assert md_path.exists(), "m92_report.md was not generated"

    content = md_path.read_text()
    assert "Milestone M92 Scientific Report" in content
    assert "Phase 0 — M91 Forensic Audit" in content
    assert "Milestone M92 Acceptance Gate Scorecard" in content
