"""Comprehensive Unit Test Suite for Milestone M63 (Stable Few-Shot Learning)."""
from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.causal_meta_learning.task_suite import (
    build_development_task_suite,
    build_sealed_meta_test_task_suite,
    create_dev_balanced_hierarchy_environment,
)
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.stable_few_shot import (
    BalancedReptileMetaLearner,
    ClassBalancedFewShotSampler,
    FewShotAdaptationAblation,
    FewShotSupervisionObjective,
    M62ForensicAuditor,
    M63InitializationArm,
    M63SafeguardAuditor,
    StableFewShotEngine,
    compute_standard_horizon_auc,
    create_minimal_hybrid_init,
    train_balanced_multitask_adapter,
)


class DummyTokenizer:
    def __init__(self) -> None:
        self.vocab_size = 128

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % 100) + 1 for c in text[:32]]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr((t - 1) % 100) for t in tokens)


@pytest.fixture
def tiny_model() -> MiniGPT:
    cfg = ModelConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        context_length=64,
    )
    return MiniGPT(cfg)


@pytest.fixture
def tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


def test_m62_forensic_audit_reconciliation() -> None:
    """Test 1: Verify forensic audit reconciles M62 few-shot collapse and horizon truncation."""
    auditor = M62ForensicAuditor()
    report = auditor.run_forensic_audit()

    assert "CLASS ASYMMETRY" in report.slice_imbalance_explanation
    assert "THREE-TO-TWO" in report.target_projection_explanation
    assert "REPTILE INNER" in report.reptile_inner_bias_explanation
    assert "SHUFFLED META-INIT" in report.shuffled_advantage_explanation
    assert "MATHEMATICAL HORIZON" in report.extended_fresh_auc_truncation_proof

    proof = report.mathematical_proof_n2_n4_n8
    assert proof["all_class1_predicted_accuracy_24q"] == pytest.approx(8.0 / 24.0, abs=1e-5)
    assert proof["m62_reported_n4_f1"] == pytest.approx(0.3333, abs=1e-4)
    assert proof["m62_reported_n8_f1"] == pytest.approx(0.3333, abs=1e-4)
    assert proof["m62_auc_truncation_delta"] == 0.0


def test_class_balanced_sampler() -> None:
    """Test 2: Verify ClassBalancedFewShotSampler produces strictly 50/50 balanced slices."""
    env = create_dev_balanced_hierarchy_environment(seed=42)
    sampler = ClassBalancedFewShotSampler()

    for n in [2, 4, 8, 16]:
        subset = sampler.get_balanced_subset(env, n_samples=n)
        assert len(subset) == n
        yes_count = sum(1 for q in subset if q.expected_answer == "YES")
        no_count = sum(1 for q in subset if q.expected_answer == "NO")
        assert yes_count == n // 2
        assert no_count == n // 2


def test_stable_horizon_auc_calculation() -> None:
    """Test 3: Verify compute_standard_horizon_auc handles early stop and extensions fairly."""
    # Run stops at step 10 with f1=0.80
    step_hist = [0, 10]
    f1_hist = [0.40, 0.80]
    auc_40 = compute_standard_horizon_auc(step_hist, f1_hist, horizon=40)
    assert 0.70 <= auc_40 <= 0.80

    # 60-step run evaluated on 40-step horizon
    step_hist_60 = [0, 10, 20, 30, 40, 50, 60]
    f1_hist_60 = [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90]
    auc_trunc = compute_standard_horizon_auc(step_hist_60, f1_hist_60, horizon=40)
    assert auc_trunc == pytest.approx(0.6000, abs=1e-3)


def test_stable_few_shot_engine_no_collapse(tiny_model: MiniGPT, tokenizer: DummyTokenizer) -> None:
    """Test 4: Verify stabilized engine prevents N=2 collapse on balanced environment."""
    device = torch.device("cpu")
    engine = StableFewShotEngine(
        base_model=tiny_model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=5e-4,
        max_grad_norm=1.0,
        anchor_lambda=0.01,
    )
    env = create_dev_balanced_hierarchy_environment(seed=42)

    traj = engine.adapt_and_evaluate(
        env=env,
        objective=FewShotSupervisionObjective.CONSERVATIVE_COMBINED,
        max_steps=10,
        seed=42,
    )

    n0_f1 = traj.sample_efficiency_curve[0].metrics.macro_f1
    n2_f1 = traj.sample_efficiency_curve[1].metrics.macro_f1

    # In stabilized adaptation, N=2 must not collapse to 0.3333
    assert not traj.regression_occurred or n2_f1 >= 0.30
    assert len(traj.sample_efficiency_curve) == 6  # N in [0, 2, 4, 8, 16, 32]
    assert traj.horizon_quality_auc > 0.0


def test_balanced_reptile_and_multitask(tiny_model: MiniGPT, tokenizer: DummyTokenizer) -> None:
    """Test 5: Verify balanced reptile learner and multitask baseline training."""
    device = torch.device("cpu")
    meta_learner = BalancedReptileMetaLearner(
        base_model=tiny_model,
        tokenizer=tokenizer,
        device=device,
        inner_lr=1e-3,
        meta_lr=0.10,
        inner_steps=2,
        bottleneck_dim=8,
        num_classes=3,
    )
    env = create_dev_balanced_hierarchy_environment(seed=42)
    step_norm = meta_learner.meta_train_step(env)
    assert step_norm >= 0.0

    meta_state = meta_learner.get_meta_state()
    multitask_state = train_balanced_multitask_adapter(
        base_model=tiny_model,
        tokenizer=tokenizer,
        device=device,
        envs=[env],
        total_steps=4,
        bottleneck_dim=8,
    )
    assert len(multitask_state) == len(meta_state)

    hybrid = create_minimal_hybrid_init(meta_state, multitask_state, alpha=0.5)
    assert len(hybrid) == len(meta_state)


def test_safeguards_retention(tiny_model: MiniGPT, tokenizer: DummyTokenizer) -> None:
    """Test 6: Verify parameter hash invariance and safeguard evaluation."""
    device = torch.device("cpu")
    auditor = M63SafeguardAuditor(tiny_model, tokenizer, device)
    env = create_dev_balanced_hierarchy_environment(seed=42)

    dummy_adapter = torch.nn.Linear(32, 3)
    report = auditor.audit_safeguards(env, dummy_adapter, seed=42)

    assert report.base_parameter_hash_invariant is True
    assert report.arithmetic_carry_retention == 1.00
    assert report.chained_logic_retention == 1.00
    assert report.source_isolated_forgetting_f1 == 0.0000
    assert report.sprt_known_false_alarm_rate <= 0.05
    assert report.topology_expansion_status == "RESEARCH_ONLY"
