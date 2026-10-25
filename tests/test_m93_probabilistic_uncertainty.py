"""Milestone M93 Verification Suite: Probabilistic Uncertainty & Robust Neural Rule Discovery.

Validates the complete M93 research loop across 6 unit and integration tests:
1. test_m92_forensic_audit_and_baseline_certification
2. test_noisy_environment_and_partitions
3. test_neural_uncertainty_and_temperature_calibration
4. test_cost_aware_experimenter_and_soft_likelihood
5. test_robust_neural_rule_acquisition_and_attribution
6. test_m93_benchmark_smoke_and_artifact_generation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.neural_reasoning.persistent_substrate import (
    PersistentReasoningSubstrate,
)
from adaptive_mini_gpt.probabilistic_uncertainty.audit_m92 import (
    M92ForensicAuditor,
)
from adaptive_mini_gpt.probabilistic_uncertainty.benchmark_m93 import (
    M93UncertaintyBenchmark,
)
from adaptive_mini_gpt.probabilistic_uncertainty.cost_aware_experimenter import (
    CostAwareExperimenter,
)
from adaptive_mini_gpt.probabilistic_uncertainty.neural_uncertainty import (
    NeuralUncertaintyEvaluator,
    TemperatureScaler,
)
from adaptive_mini_gpt.probabilistic_uncertainty.noisy_environment import (
    NoisyObservation,
    NoisyRuleEnvironment,
)
from adaptive_mini_gpt.probabilistic_uncertainty.robust_rule_learner import (
    RobustRuleLearner,
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


def test_m92_forensic_audit_and_baseline_certification() -> None:
    """Validate Phase 0 forensic audit and reconciliation of M92 metrics."""
    audit = M92ForensicAuditor.execute_audit()
    assert audit.status == "M92_AUDIT_VERIFIED"
    assert audit.m92_integrated_discovery_em == 92.29
    assert audit.m92_lifelong_retention_pct == 94.76
    assert audit.total_trainable_parameters == 18_560
    assert audit.budget_headroom == 1_440
    assert audit.budget_ceiling == 20_000
    assert audit.frozen_backbone_weights == 1_071_872

    # Verify reconciliation of the two passive comparisons
    assert audit.neural_vs_adapt_gain_pct == 62.76  # Arm 6 vs Arm 5
    assert audit.neural_vs_passive_gain_pct == 12.54  # Arm 6 vs Arm 2
    assert audit.symbolic_vs_neural_gap_pct == 7.71  # Arm 4 (100%) vs Arm 6 (92.29%)
    assert audit.noisy_feedback_imperative_verified is True


def test_noisy_environment_and_partitions() -> None:
    """Verify NoisyRuleEnvironment supports clean, noisy, input-dependent, and contradictory tasks."""
    suite = NoisyRuleEnvironment.generate_benchmark_suite(seed=42)
    assert len(suite) == 9

    clean_tasks = [t for t in suite if t.noise_rate == 0.0 and t.is_learnable]
    noisy_tasks = [t for t in suite if t.noise_rate > 0.0 and t.is_learnable]
    unsupported = [t for t in suite if not t.is_learnable]

    assert len(clean_tasks) == 2
    assert len(noisy_tasks) == 6
    assert len(unsupported) == 1

    # Verify query budget and cost accounting
    t_clean = clean_tasks[0]
    assert t_clean.remaining_budget == 4
    obs = t_clean.oracle_query(5)
    assert isinstance(obs, NoisyObservation)
    assert obs.x == 5
    assert t_clean.queries_used == 1
    assert t_clean.total_cost_incurred == 1.0

    # Verify contradiction task
    u_task = unsupported[0]
    contradicting = [obs.x for obs in u_task.initial_observations if obs.x == 2]
    assert len(contradicting) == 2


def test_neural_uncertainty_and_temperature_calibration() -> None:
    """Test TemperatureScaler, NLL, Brier score, ECE, and selective prediction."""
    torch.manual_seed(42)
    scaler = TemperatureScaler(initial_temp=1.0)
    assert scaler.temperature == 1.0

    # Synthesize overconfident uncalibrated logits
    logits = torch.randn(20, 16) * 5.0
    labels = torch.randint(0, 16, (20,))

    # Fit temperature
    opt_temp = scaler.fit(logits, labels, max_iter=30)
    assert opt_temp > 0.1

    calibrated_probs = torch.softmax(scaler(logits), dim=-1)
    metrics = NeuralUncertaintyEvaluator.compute_calibration_metrics(
        calibrated_probs, labels, temperature=scaler.temperature
    )
    assert metrics.brier_score >= 0.0
    assert metrics.ece >= 0.0
    assert metrics.mean_entropy >= 0.0

    # Verify selective prediction profile
    sel_pred = NeuralUncertaintyEvaluator.compute_selective_prediction(calibrated_probs, labels)
    assert len(sel_pred.coverages) == 4
    assert len(sel_pred.error_rates) == 4


def test_cost_aware_experimenter_and_soft_likelihood() -> None:
    """Test Bayesian soft-likelihood belief updating, EIG probe selection, and early stopping."""
    experimenter = CostAwareExperimenter()
    suite = NoisyRuleEnvironment.generate_benchmark_suite(seed=42)
    clean_task = suite[0]  # clean_quad_1

    outcome = experimenter.run_experimentation(clean_task, strategy="cost_aware")
    assert outcome.decision == "EXECUTE"
    assert outcome.selected_rule is not None
    assert outcome.selected_rule.family == "quadratic"
    assert outcome.queries_used <= 4
    assert outcome.final_confidence >= 0.70

    # Test contradiction task triggering abstention
    u_task = suite[-1]  # unsupported_contradiction
    u_outcome = experimenter.run_experimentation(u_task, strategy="cost_aware")
    assert u_outcome.decision == "ABSTAIN"


def test_robust_neural_rule_acquisition_and_attribution(base_model: MiniGPT, device: torch.device) -> None:
    """Test robust residual acquisition, dev verification, temperature scaling, and weight shuffling."""
    substrate = PersistentReasoningSubstrate(
        d_model=128,
        d_shared=32,
        num_symbols=16,
        rank_residual=4,
        budget_ceiling=20_000,
    ).to(device)

    suite = NoisyRuleEnvironment.generate_benchmark_suite(seed=42)
    task = suite[0]  # clean_quad_1

    experimenter = CostAwareExperimenter()
    outcome = experimenter.run_experimentation(task, strategy="cost_aware")
    assert outcome.decision == "EXECUTE" and outcome.selected_rule is not None

    acq_res = RobustRuleLearner.acquire_skill(
        model=base_model,
        substrate=substrate,
        task=task,
        device=device,
        discovered_rule=outcome.selected_rule,
        steps=30,
        lr=0.03,
    )
    assert acq_res.status == "ACCEPTED"
    assert acq_res.dev_exact_match >= 80.0
    assert acq_res.held_out_test_em >= 80.0
    assert acq_res.causal_attribution_drop >= 50.0  # Shuffled weights drop significantly
    assert acq_res.calibration_metrics.brier_score <= 0.30
    assert substrate.count_total_parameters() == 10_304 + 1_032


def test_m93_benchmark_smoke_and_artifact_generation(base_model: MiniGPT, device: torch.device) -> None:
    """Fast smoke test of Milestone M93 benchmark runner across 1 seed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        report = M93UncertaintyBenchmark.run_benchmark(
            model=base_model,
            device=device,
            seeds=[42],
            output_dir=tmp_dir,
            epochs_trunk=5,
            lr_trunk=0.02,
        )
        assert report["milestone"] == "M93"
        assert report["budget_compliant"] is True
        assert report["trainable_parameters"] <= 20_000
        assert (Path(tmp_dir) / "m93_report.json").exists()
        assert (Path(tmp_dir) / "m93_report.md").exists()
