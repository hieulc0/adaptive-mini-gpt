"""Comprehensive Unit Test Suite for Milestone M62 (Neural Meta-Initialization)."""
from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.causal_meta_learning.task_suite import (
    build_development_task_suite,
    build_sealed_meta_test_task_suite,
    create_dev_balanced_hierarchy_environment,
    create_meta_long_chain_aerospace,
)
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.neural_meta_learning import (
    CrossFamilyGeneralizationAuditor,
    M61MeasurementAuditor,
    M62AdaptationEvaluator,
    M62InitializationArm,
    M62SafeguardAuditor,
    ReptileMetaLearner,
    compute_horizon_quality_auc,
    train_multitask_pretrained_adapter,
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


def test_m61_measurement_protocol_audit() -> None:
    """Test 1: Verify forensic audit of M61 measurement protocols."""
    auditor = M61MeasurementAuditor()
    report = auditor.run_measurement_audit()

    assert "TRUNCATED HORIZON" in report.flawed_auc_integral_mechanism
    assert "RESET PROTOCOL" in report.reconciliation_root_cause
    assert report.reset_rate_pct == 60.0
    assert report.experience_memory_incremental_gain == 0.0000
    assert "AUDIT CONFIRMED" in report.audit_verdict


def test_compute_horizon_quality_auc_carrying_forward() -> None:
    """Test 2: Verify Horizon Quality AUC carries forward final performance across horizon."""
    # Run stops at step 10 with f1=0.80
    step_hist = [0, 10]
    f1_hist = [0.40, 0.80]

    # Without carrying forward: integral [0..0.25] yields ~0.15
    # With carrying forward: [0..0.25] is (0.4+0.8)/2 * 0.25 = 0.15; [0.25..1.0] is 0.80 * 0.75 = 0.60; Total = 0.75
    auc = compute_horizon_quality_auc(step_hist, f1_hist, horizon=40)
    assert 0.70 <= auc <= 0.80


def test_reptile_meta_learner_step(tiny_model: MiniGPT, tokenizer: DummyTokenizer) -> None:
    """Test 3: Verify ReptileMetaLearner performs inner adaptation and outer interpolation."""
    device = torch.device("cpu")
    meta_learner = ReptileMetaLearner(
        base_model=tiny_model,
        tokenizer=tokenizer,
        device=device,
        inner_lr=5e-3,
        meta_lr=0.20,
        inner_steps=3,
        bottleneck_dim=8,
    )

    env = create_dev_balanced_hierarchy_environment(seed=42)
    init_state = meta_learner.get_meta_initialization_state()
    rec = meta_learner.meta_train_step(env, iteration=1)

    assert rec.iteration == 1
    assert rec.outer_step_norm > 0.0
    new_state = meta_learner.get_meta_initialization_state()

    # Verify parameters updated
    any_diff = False
    for k in init_state:
        if not torch.allclose(init_state[k], new_state[k]):
            any_diff = True
            break
    assert any_diff is True


def test_multitask_pretrained_adapter(tiny_model: MiniGPT, tokenizer: DummyTokenizer) -> None:
    """Test 4: Verify ordinary multitask pretraining pools meta-training tasks."""
    device = torch.device("cpu")
    env1 = create_dev_balanced_hierarchy_environment(seed=42)
    env2 = create_meta_long_chain_aerospace(seed=100)

    adapter = train_multitask_pretrained_adapter(
        base_model=tiny_model,
        tokenizer=tokenizer,
        device=device,
        meta_training_envs=[env1, env2],
        total_steps=10,
    )

    assert adapter is not None
    assert sum(p.numel() for p in adapter.parameters()) > 0


def test_m62_adaptation_evaluator(tiny_model: MiniGPT, tokenizer: DummyTokenizer) -> None:
    """Test 5: Verify M62AdaptationEvaluator produces sample efficiency curve and Horizon AUC."""
    device = torch.device("cpu")
    evaluator = M62AdaptationEvaluator(
        base_model=tiny_model,
        tokenizer=tokenizer,
        device=device,
    )
    env = create_dev_balanced_hierarchy_environment(seed=42)

    res = evaluator.evaluate_adaptation_trajectory(
        env=env,
        arm=M62InitializationArm.ARM_A_FRESH_RANDOM_INIT,
        seed=42,
    )

    assert res.actual_steps_taken == 40
    assert len(res.sample_efficiency_curve) == 6  # N in [0, 2, 4, 8, 16, 32]
    assert res.horizon_quality_auc >= 0.0
    assert res.compute_adjusted_efficiency > 0.0


def test_cross_family_generalization_auditor() -> None:
    """Test 6: Verify cross-family generalization auditor categorizes transfer boundaries."""
    auditor = CrossFamilyGeneralizationAuditor()
    mock_evals = {
        "dev_deep_chain_dag": {
            "Arm_D_Reptile_Meta_Learned_Init": {"horizon_quality_auc": 0.55},
            "Arm_A_Fresh_Random_Init": {"horizon_quality_auc": 0.45},
        },
        "dev_dense_cluster_exceptions": {
            "Arm_D_Reptile_Meta_Learned_Init": {"horizon_quality_auc": 0.52},
            "Arm_A_Fresh_Random_Init": {"horizon_quality_auc": 0.48},
        },
        "dev_incompatible_parity": {
            "Arm_D_Reptile_Meta_Learned_Init": {"horizon_quality_auc": 0.35},
            "Arm_A_Fresh_Random_Init": {"horizon_quality_auc": 0.35},
        },
    }

    report = auditor.audit_family_generalization(
        task_arm_evals=mock_evals,
        meta_arm_name="Arm_D_Reptile_Meta_Learned_Init",
        baseline_arm_name="Arm_A_Fresh_Random_Init",
    )

    assert report.in_family_transfer_advantage > 0.0
    assert "BROAD_NEURAL_TRANSFER" in report.generalization_verdict


def test_m62_safeguard_auditor(tiny_model: MiniGPT, tokenizer: DummyTokenizer) -> None:
    """Test 7: Verify parameter invariance and retention safeguards."""
    device = torch.device("cpu")
    auditor = M62SafeguardAuditor(tiny_model, tokenizer, device)
    d_model = getattr(tiny_model.config, "d_model", getattr(tiny_model.config, "n_embd", 32))
    src_adapter = PersistentCandidateAdapter(d_model=d_model, bottleneck_dim=8, num_classes=2)
    src_env = create_dev_balanced_hierarchy_environment(seed=42)

    rep = auditor.audit_safeguards(src_env, src_adapter, seed=42)
    assert rep.base_parameter_hash_invariant is True
    assert rep.arithmetic_carry_retention == 1.00
    assert rep.chained_logic_retention == 1.00
    assert rep.all_safeguards_passed is True
