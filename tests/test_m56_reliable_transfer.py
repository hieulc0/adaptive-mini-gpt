"""Unit tests for M56: Reliable Transfer and Noise-Aware Epistemic Learning."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.model.modular_adapter import compute_base_model_hash
from adaptive_mini_gpt.model.transformer import MiniGPT, MiniGPTConfig
from adaptive_mini_gpt.reliable_transfer.active_probing import (
    ControlledActiveProbingAuditor,
)
from adaptive_mini_gpt.reliable_transfer.audit import (
    M55ScientificAuditor,
    M55AuditSummary,
)
from adaptive_mini_gpt.reliable_transfer.composition_models import (
    GatedResidualSkillAdapter,
    PartiallyTrainableSkillAdapter,
    ResidualSkillAdapter,
)
from adaptive_mini_gpt.reliable_transfer.intervention import (
    RepresentationDriftInterventionAnalyzer,
)
from adaptive_mini_gpt.reliable_transfer.noise_aware_belief import (
    BaselineFixedThresholdTracker,
    NoiseAwareEpistemicAuditor,
    NoiseAwarePredictiveTracker,
    SequentialProbabilityRatioTest,
)
from adaptive_mini_gpt.reliable_transfer.retention_auditor import (
    ReliableRetentionAuditor,
)
from adaptive_mini_gpt.reliable_transfer.target_environments import (
    create_metabolic_bypass_environment,
    create_reporting_hierarchy_environment,
    create_social_coalition_environment,
    create_supply_chain_pipeline_environment,
)
from adaptive_mini_gpt.reliable_transfer.transfer_engine import (
    ReliableTransferEngine,
)


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [hash(w) % 100 for w in text.split()] or [0]


@pytest.fixture
def base_model_and_tokenizer():
    config = MiniGPTConfig(
        vocab_size=1024,
        block_size=128,
        n_layer=2,
        n_head=2,
        d_model=128,
        d_ff=256,
        dropout=0.0,
    )
    model = MiniGPT(config)
    tokenizer = DummyTokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device


def test_m55_scientific_audit_reconciliation():
    """Verify reproduction of M55 baseline metrics and resolution of retention queries."""
    summary = M55ScientificAuditor.audit_m55_execution()
    assert summary.reported_fresh_auc == pytest.approx(0.4514, abs=0.01)
    assert summary.reported_frozen_residual_auc == pytest.approx(0.4730, abs=0.01)
    assert summary.reporting_hierarchy_residual_gain == pytest.approx(0.0216, abs=0.01)

    # Verify recovered supply chain metrics
    assert summary.supply_chain_gated_auc > summary.supply_chain_residual_auc
    assert summary.supply_chain_gated_auc > summary.supply_chain_fresh_auc

    # Verify retention mechanics
    assert summary.source_isolated_forgetting_f1 == 0.0000
    assert summary.composite_residual_was_active_on_source is True
    assert summary.source_composite_active_residual_interference_f1 > 0.0
    assert summary.param_reduction_pct == pytest.approx(47.1, abs=0.5)


def test_composite_adapter_models():
    """Verify parameter counts, freezing, and gradient flow across adapter architectures."""
    source_adapter = PersistentCandidateAdapter(d_model=128)

    # 1. ResidualSkillAdapter
    res_model = ResidualSkillAdapter(source_adapter, d_model=128)
    res_counts = res_model.count_parameters()
    assert res_counts["trainable"] == 2371  # 128*16 + 16 + 16*3 + 3 = 2048 + 16 + 48 + 3 = 2115 + layer params = 2371
    assert res_counts["frozen"] == 4483
    for p in res_model.source_adapter.parameters():
        assert not p.requires_grad

    # Test forward and source_only passes
    dummy_h = torch.randn(1, 10, 128)
    out_full = res_model(dummy_h)
    out_src = res_model.evaluate_source_only(dummy_h)
    assert out_full.shape == (1, 3)
    assert out_src.shape == (1, 3)

    # 2. GatedResidualSkillAdapter
    gated_model = GatedResidualSkillAdapter(source_adapter, d_model=128)
    gated_counts = gated_model.count_parameters()
    assert gated_counts["trainable"] == 2371 + 3  # residual head + gate_logit (3) -> 2374/3014
    assert gated_model.gate_logit.requires_grad is True

    # 3. PartiallyTrainableSkillAdapter
    pt_frozen = PartiallyTrainableSkillAdapter(source_adapter, trainability_alpha=0.0, d_model=128)
    assert pt_frozen.count_parameters()["trainable"] == 2371

    pt_trainable = PartiallyTrainableSkillAdapter(source_adapter, trainability_alpha=1.0, d_model=128)
    assert pt_trainable.count_parameters()["trainable"] == 2371 + 4483


def test_multi_world_environments():
    """Verify target environments have valid reachability graphs and balanced queries."""
    env_rep = create_reporting_hierarchy_environment(seed=42)
    assert len(env_rep.ground_truth.nodes) == 6
    assert len(env_rep.held_out_test_queries) >= 16
    assert len(env_rep.held_out_minimal_pairs) >= 6

    env_sc = create_supply_chain_pipeline_environment(seed=42)
    assert len(env_sc.ground_truth.nodes) == 8
    assert len(env_sc.held_out_test_queries) >= 16

    env_met = create_metabolic_bypass_environment(seed=42)
    assert len(env_met.ground_truth.nodes) == 6

    env_soc = create_social_coalition_environment(seed=42)
    assert len(env_soc.ground_truth.nodes) == 8


def test_noise_aware_epistemic_detectors():
    """Verify SPRT and posterior-predictive tracker vs baseline on clean and noisy episodes."""
    baseline = BaselineFixedThresholdTracker(threshold=0.50)
    noise_aware = NoiseAwarePredictiveTracker()
    sprt = SequentialProbabilityRatioTest(assumed_noise_floor=0.15)

    # Case 1: Known world with 15% noise (2 mismatches in 12 queries)
    noisy_known_obs = [("YES", "YES")] * 10 + [("YES", "NO")] * 2
    d_base = baseline.evaluate_episode(noisy_known_obs, budget=12)
    d_na = noise_aware.evaluate_episode(noisy_known_obs, budget=12)
    d_sprt = sprt.evaluate_episode(noisy_known_obs, budget=12)

    # SPRT should correctly stay below upper threshold or cross lower threshold
    assert d_sprt.is_unknown_declared is False
    assert d_sprt.log_likelihood_ratio < sprt.threshold_A

    # Case 2: Genuinely unknown world with structural contradiction (9 mismatches in 12 queries)
    unknown_obs = [("YES", "NO")] * 9 + [("YES", "YES")] * 3
    d_sprt_unk = sprt.evaluate_episode(unknown_obs, budget=12)
    assert d_sprt_unk.is_unknown_declared is True
    assert d_sprt_unk.log_likelihood_ratio > 0.0

    # Test multi-regime evaluation
    regime_results = NoiseAwareEpistemicAuditor.evaluate_noise_regimes(
        noise_levels=[0.0, 0.15],
        episodes_per_regime=5,
        budget=12,
        seed=42,
    )
    assert len(regime_results) == 2
    for r in regime_results:
        assert r.sprt_recall >= 0.80
        assert r.sprt_false_alarm_rate <= 0.20


def test_controlled_active_probing():
    """Verify active probing query generation and tracking under fixed budget."""
    auditor = ControlledActiveProbingAuditor(budget=12, noise_rate=0.15, seed=42)
    env = create_reporting_hierarchy_environment(seed=42)
    sprt = SequentialProbabilityRatioTest(assumed_noise_floor=0.15)

    ep_random = auditor.run_probing_episode("random_valid", env, sprt)
    assert ep_random.queries_consumed <= 12
    assert ep_random.repeated_queries_count >= 0

    ep_targeted = auditor.run_probing_episode("targeted_peripheral_bridge", env, sprt)
    assert ep_targeted.queries_consumed <= 12
    assert ep_targeted.useful_evidence_count >= 0


def test_reliable_retention_auditor(base_model_and_tokenizer):
    """Verify retention audit evaluating both isolated and active-residual composite pathways."""
    model, tokenizer, device = base_model_and_tokenizer
    auditor = ReliableRetentionAuditor(model, tokenizer, device)

    source_adapter = PersistentCandidateAdapter(d_model=128).to(device)
    composite_adapter = ResidualSkillAdapter(source_adapter, d_model=128).to(device)

    env_src = create_reporting_hierarchy_environment(seed=42)
    env_tgt = create_supply_chain_pipeline_environment(seed=43)

    result = auditor.audit_full_retention(
        source_task_id="reporting_hierarchy",
        target_task_id="supply_chain_pipeline",
        initial_source_adapter=source_adapter,
        trained_composite_module=composite_adapter,
        source_env=env_src,
        target_env=env_tgt,
    )

    # Isolated branch must have exactly 0 forgetting since weights are identical
    assert result.source_isolated_forgetting_f1 == pytest.approx(0.0000, abs=1e-5)
    assert result.arithmetic_carry_retention == 1.0
    assert result.chained_logic_retention == 1.0
    assert result.base_parameter_hash_invariant is True
    assert result.rollback_triggered is False
