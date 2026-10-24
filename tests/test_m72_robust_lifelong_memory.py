"""Deterministic Unit & Integration Tests for Milestone M72: Robust Lifelong Neural Memory."""
from __future__ import annotations

import pytest

from adaptive_mini_gpt.data.bpe_scaling import train_scaled_bpe_tokenizer
from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.modular_skill_learning.repertoire import Rank16ResidualAdapter
from adaptive_mini_gpt.robust_lifelong_memory import (
    CompleteModelCapacityTracker,
    M71ScientificAuditor,
    RobustSkillRegistry,
    SkillState,
    build_m72_lifelong_stream_sequence,
    create_anti_transitive_dag_env,
    create_modular_arithmetic_relational_env,
)


@pytest.fixture
def tiny_model_and_tokenizer():
    """Create deterministic tiny test model."""
    tok = train_scaled_bpe_tokenizer("A B C 0 1 2 YES NO UNKNOWN Evidence [A->B]", target_vocab_size=64)
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        context_length=64,
    )
    model = MiniGPT(cfg)
    model.tokenizer = tok
    model.eval()
    return model, tok


def test_m71_forensic_audit():
    """Test 1: Verify M71 forensic audit reconciles all 6 anomalies."""
    report = M71ScientificAuditor.audit_m71_results()
    assert report.audit_passed is True
    assert len(report.anomalies) == 6
    assert report.isolation_verified is True
    assert report.gradient_training_verified is True

    anomaly_ids = [a.anomaly_id for a in report.anomalies]
    assert "M71-A01-ROUTING-ARM-EQUIVALENCE" in anomaly_ids
    assert "M71-A02-LIMITED-STREAM-HORIZON" in anomaly_ids
    assert "M71-A03-PARAMETER-ACCOUNTING-CONFUSION" in anomaly_ids
    assert "M71-A04-HINDSIGHT-ORACLE-GAP" in anomaly_ids
    assert "M71-A05-ALWAYS-FRESH-RAW-F1" in anomaly_ids
    assert "M71-A06-GRADIENT-TRAINING-CONFIRMATION" in anomaly_ids


def test_capacity_accounting(tiny_model_and_tokenizer):
    """Test 2: Verify exact integer parameter allocation tracking."""
    model, _ = tiny_model_and_tokenizer
    reg = RobustSkillRegistry(d_model=128, bottleneck_dim=16, num_classes=3)

    # Register 3 source adapters
    for i in range(3):
        ad = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
        reg.register_skill(f"source_skill_{i}", "SRC_FAM", "src_env", ad, 0.60, 0.60)

    # Register 1 acquired candidate adapter
    cand = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    reg.register_skill("cand_test", "CAND_FAM", "cand_env", cand, 0.70, 0.70, initial_state=SkillState.ACTIVE)

    audit = CompleteModelCapacityTracker.audit_capacity(
        backbone_model=model,
        registry=reg,
        routed_skill_id="cand_test",
        candidate_adapter=cand,
    )

    backbone_p = sum(p.numel() for p in model.parameters())
    assert audit.backbone_params == backbone_p
    assert audit.source_adapter_params == 3 * 2371  # 7,113
    assert audit.total_base_params == backbone_p + 7113
    assert audit.acquired_adapter_params == 2371
    assert audit.total_stored_adapter_params == 4 * 2371  # 9,484
    assert audit.total_stored_model_params == backbone_p + 9484
    assert audit.active_inference_params == backbone_p + 2371  # Exactly 1 active adapter
    assert audit.dormant_adapter_params == 3 * 2371  # Stored minus 1 routed
    assert audit.trainable_adaptation_params == 2371
    assert audit.active_adapter_count == 4
    assert audit.dormant_adapter_count == 0


def test_robust_skill_lifecycle():
    """Test 3: Verify skill lifecycle, reversible state transitions, and pruning."""
    reg = RobustSkillRegistry(d_model=128, bottleneck_dim=16, num_classes=3)
    ad1 = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    rec1 = reg.register_skill("skill_1", "FAM_A", "env_1", ad1, 0.75, 0.75, initial_state=SkillState.ACTIVE)

    h_init = rec1.compute_hash()

    # Deactivate -> DORMANT
    reg.deactivate_skill("skill_1")
    assert reg.get_skill("skill_1").state == SkillState.DORMANT
    assert len(reg.list_active_skills()) == 0
    assert len(reg.list_dormant_skills()) == 1
    assert rec1.compute_hash() == h_init  # Weights unaltered!

    # Reactivate -> ACTIVE
    reg.activate_skill("skill_1")
    assert reg.get_skill("skill_1").state == SkillState.ACTIVE
    assert len(reg.list_active_skills()) == 1
    assert rec1.compute_hash() == h_init

    # Mark prunable -> OPTIONALLY_PRUNABLE
    reg.mark_prunable("skill_1", reason="No marginal benefit")
    assert reg.get_skill("skill_1").state == SkillState.OPTIONALLY_PRUNABLE

    # Prune redundant skills -> moves prunable to DORMANT without deleting weights
    pruned = reg.prune_redundant_skills()
    assert pruned == ["skill_1"]
    assert reg.get_skill("skill_1").state == SkillState.DORMANT
    assert rec1.compute_hash() == h_init


def test_20_episode_stream_generation():
    """Test 4: Verify 20-episode task stream generation and property coverage."""
    sequence = build_m72_lifelong_stream_sequence()
    assert len(sequence) == 20

    worlds = {ep["world_id"] for ep in sequence}
    assert "world_alpha" in worlds
    assert "world_beta" in worlds
    assert "world_gamma" in worlds
    assert "world_delta" in worlds

    families = {ep["rule_family"] for ep in sequence}
    assert "PARITY_CYCLE" in families
    assert "CHROMATIN_LATTICE" in families
    assert "FAULT_CASCADE" in families
    assert "TRANSITIVE_DAG" in families
    assert "ANTI_TRANSITIVE_DAG" in families
    assert "INVERTED_PARITY_CYCLE" in families
    assert "MODULAR_CONGRUENCE_MOD_3" in families

    repeated_episodes = [ep for ep in sequence if ep["is_repeated_encounter"]]
    assert len(repeated_episodes) == 7  # 7 returning/repeat encounters

    adversarial_episodes = [ep for ep in sequence if ep["is_adversarial"]]
    assert len(adversarial_episodes) == 3  # Anti-DAG, Inverted Rail, Contradictory Lattice


def test_adversarial_and_nongraph_environments():
    """Test 5: Verify anti-transitive and non-graph modular environments."""
    anti_env = create_anti_transitive_dag_env("test_anti")
    assert len(anti_env.held_out_test_queries) >= 8

    mod_env = create_modular_arithmetic_relational_env("test_mod")
    assert len(mod_env.held_out_test_queries) >= 8
    # Test query format contains modular difference note
    assert any("is_congruent_mod_3_with" in q.question_text for q in mod_env.held_out_test_queries)
