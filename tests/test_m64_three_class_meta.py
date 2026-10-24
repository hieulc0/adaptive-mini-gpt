"""Comprehensive Unit Test Suite for Milestone M64 (Three-Class Learning & Cross-Family Generalization)."""
from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.causal_meta_learning.task_suite import (
    build_development_task_suite,
    build_sealed_meta_test_task_suite,
    create_dev_balanced_hierarchy_environment,
    create_dev_dense_cluster_exceptions_environment,
    create_meta_long_chain_aerospace,
)
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.three_class_meta import (
    CrossFamilyMetaEvaluator,
    M63ForensicAuditor,
    M64InitializationArm,
    M64SafeguardAuditor,
    ThreeClassFewShotSampler,
    ThreeClassReptileLearner,
    ThreeClassSupervisionEngine,
    ThreeClassSupervisionMode,
    compute_standard_horizon_auc,
    create_hybrid_init,
    train_three_class_multitask_adapter,
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


def test_m63_forensic_audit() -> None:
    """Test 1: Verify forensic audit of M63 UNKNOWN omission, N=2 coverage, and metrics."""
    auditor = M63ForensicAuditor()
    report = auditor.run_forensic_audit()

    assert report.output_head_dimension_verified == 3
    assert "UNKNOWN TRAINING OMISSION" in report.unknown_training_omission_explanation
    assert "ZERO RECALL MECHANISM" in report.unknown_zero_recall_mechanism
    assert "SEMANTIC DEFINITION" in report.semantic_definition_no_vs_unknown
    assert "N=2 THREE-CLASS LIMITATION" in report.n2_three_class_impossibility_acknowledgment
    assert "METRIC RECONCILIATION" in report.n32_vs_final_f1_reconciliation
    assert len(report.m18_per_seed_hashes) == 3
    assert report.reproduced_m63_reptile_auc == 0.2433


def test_three_class_few_shot_sampler_budgets() -> None:
    """Test 2: Verify ThreeClassFewShotSampler delivers correct class budgets without test leakage."""
    env = create_dev_balanced_hierarchy_environment(seed=42)
    sampler = ThreeClassFewShotSampler()

    # N=2: 1 YES, 1 NO, 0 UNK
    m2 = sampler.get_budgeted_subset(env, 2)
    assert m2.num_examples == 2
    assert m2.class_counts["YES"] == 1
    assert m2.class_counts["NO"] == 1
    assert m2.class_counts["UNKNOWN"] == 0
    assert m2.coverage_complete is False
    assert m2.zero_leakage_verified is True

    # N=4: 2 YES, 1 NO, 1 UNK
    m4 = sampler.get_budgeted_subset(env, 4)
    assert m4.num_examples == 4
    assert m4.class_counts["YES"] == 2
    assert m4.class_counts["NO"] == 1
    assert m4.class_counts["UNKNOWN"] == 1
    assert m4.coverage_complete is True
    assert m4.zero_leakage_verified is True

    # N=8: 3 YES, 3 NO, 2 UNK
    m8 = sampler.get_budgeted_subset(env, 8)
    assert m8.num_examples == 8
    assert m8.class_counts["YES"] == 3
    assert m8.class_counts["NO"] == 3
    assert m8.class_counts["UNKNOWN"] == 2
    assert m8.coverage_complete is True

    # N=16: 6 YES, 5 NO, 5 UNK
    m16 = sampler.get_budgeted_subset(env, 16)
    assert m16.num_examples == 16
    assert m16.class_counts["YES"] == 6
    assert m16.class_counts["NO"] == 5
    assert m16.class_counts["UNKNOWN"] == 5
    assert m16.coverage_complete is True

    # N=32: 11 YES, 11 NO, 10 UNK
    m32 = sampler.get_budgeted_subset(env, 32)
    assert m32.num_examples == 32
    assert m32.class_counts["YES"] == 11
    assert m32.class_counts["NO"] == 11
    assert m32.class_counts["UNKNOWN"] == 10
    assert m32.coverage_complete is True


def test_three_class_engine_adaptation(tiny_model: MiniGPT, tokenizer: DummyTokenizer) -> None:
    """Test 3: Verify ThreeClassSupervisionEngine trains all 3 classes and computes tri-state metrics."""
    device = torch.device("cpu")
    engine = ThreeClassSupervisionEngine(
        base_model=tiny_model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        anchor_lambda=0.01,
    )
    env = create_dev_balanced_hierarchy_environment(seed=42)

    traj = engine.adapt_and_evaluate(
        env=env,
        mode=ThreeClassSupervisionMode.M64_THREE_CLASS_TRAINING,
        max_steps=10,
        seed=42,
    )

    assert len(traj.sample_efficiency_curve) == 6
    assert traj.horizon_quality_auc > 0.0
    assert "NO" in traj.final_metrics.precision_per_class
    assert "YES" in traj.final_metrics.precision_per_class
    assert "UNKNOWN" in traj.final_metrics.precision_per_class
    assert len(traj.final_metrics.confusion_matrix) == 3


def test_broad_reptile_meta_learning(tiny_model: MiniGPT, tokenizer: DummyTokenizer) -> None:
    """Test 4: Verify ThreeClassReptileLearner on broad multi-family tasks."""
    device = torch.device("cpu")
    meta_learner = ThreeClassReptileLearner(
        base_model=tiny_model,
        tokenizer=tokenizer,
        device=device,
        inner_lr=1e-3,
        meta_lr=0.10,
        inner_steps=2,
        bottleneck_dim=8,
    )

    env_dag = create_dev_balanced_hierarchy_environment(seed=42)
    env_cluster = create_dev_dense_cluster_exceptions_environment(seed=42)

    norm1 = meta_learner.meta_train_iteration(env_dag)
    norm2 = meta_learner.meta_train_iteration(env_cluster)
    assert norm1 >= 0.0
    assert norm2 >= 0.0

    meta_state = meta_learner.get_meta_state()
    multitask_state = train_three_class_multitask_adapter(
        base_model=tiny_model,
        tokenizer=tokenizer,
        device=device,
        envs=[env_dag, env_cluster],
        total_steps=4,
        bottleneck_dim=8,
    )
    assert len(multitask_state) == len(meta_state)

    hybrid = create_hybrid_init(meta_state, multitask_state, alpha=0.5)
    assert len(hybrid) == len(meta_state)


def test_cross_family_evaluator_generalization() -> None:
    """Test 5: Verify CrossFamilyMetaEvaluator audits in-family vs unseen family transfer."""
    report = CrossFamilyMetaEvaluator.audit_family_generalization(
        task_arm_evals={},
        target_meta_arm="Arm_D_Broad_Reptile",
        baseline_arm="Arm_A_Fresh_Random",
    )
    assert "CROSS-FAMILY" in report.generalization_verdict


def test_m64_safeguards_retention(tiny_model: MiniGPT, tokenizer: DummyTokenizer) -> None:
    """Test 6: Verify M64 parameter hash invariance, arithmetic carry, and logic retention."""
    device = torch.device("cpu")
    auditor = M64SafeguardAuditor(tiny_model, tokenizer, device, seed=42)
    env = create_dev_balanced_hierarchy_environment(seed=42)

    dummy_adapter = torch.nn.Linear(32, 3)
    report = auditor.audit_safeguards(env, dummy_adapter)

    assert report.base_parameter_hash_invariant is True
    assert report.arithmetic_carry_retention == 1.00
    assert report.chained_logic_retention == 1.00
    assert report.source_isolated_forgetting_f1 == 0.0000
    assert report.sprt_known_false_alarm_rate <= 0.05
    assert report.sprt_unknown_recall >= 0.90
    assert report.topology_expansion_status == "RESEARCH_ONLY"
