"""Unit tests for Milestone M54: Verified Skill Reuse and Adaptive Transfer."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import pytest
import torch
import numpy as np

from adaptive_mini_gpt.data.bpe_scaling import train_scaled_bpe_tokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.skill_reuse import (
    M53ScientificAuditor,
    M53ScientificAuditReport,
    SkillRepertoireRegistry,
    RegisteredSkill,
    compute_adapter_hash,
    build_m54_environment_suite,
    SkillBenchmarkEnvironments,
    SkillCompatibilityEvaluator,
    NeuralSkillSelectorHead,
    AdaptiveTransferEngine,
    RetentionEvaluationEngine,
    EpistemicGeneralizationAuditor,
)


def _make_dummy_model_and_tok() -> tuple[MiniGPT, Any]:
    text = (
        "GeneLocusAlpha activates_transcription_of PromoterSiteBeta. EnhancerUnitGamma. "
        "TranscriptionFactorDelta RNAIntermediateEpsilon RibosomeTargetZeta. "
        "QuantumSourceQ0 routes_quantum_information_to RepeaterCouplerQ1. "
        "BridgeDecoherenceNode RepeaterCouplerQ2 QuantumSinkQ3 BypassAuxQ4. "
        "MeshRegistry in_same_security_zone GatewayCluster PaymentServiceNode. "
        "AuthServicePod AnalyticsNode LoggingDaemon. "
        "DomainArchaea taxonomically_subsumes PhylumMethano ClassHalobacteria. "
        "OrderThermococci FamilyArchae GenusPyro. "
        "JuniorAnalystJ1 reports_transitively_to SeniorAssociateS2. TeamLeadT3. "
        "EngineeringDirectorD4 VicePresidentV5 ChiefExecutiveC6. "
        "PrecursorSubstrateAlpha catalyzes_synthesis_of IntermediateEnzymeBeta. "
        "InhibitedChokePointGamma KinaseFactorDelta DownstreamMetaboliteEpsilon AlternativeShuntOmega. "
        "SovereignStateNorth holds_mutual_pact_with AlliancePartnerEast. "
        "FederationSouth TreatyStateWest NeutralObservatory IsolatedFaction. "
        "FeedstockStage0 directly_feeds_stage RefiningStage1 SynthesisStage2. "
        "PackagingStage3 InspectionStage4 DispatchStage5. "
        "RingNodeA shares_parity_state_with RingNodeB. RingNodeC RingNodeD RingNodeE RingNodeF. "
        "PartitionLeft1 cross_partition_connects PartitionRight1 PartitionLeft2 PartitionRight2 PartitionLeft3 PartitionRight3. "
        "CoreMember1 strongly_interlinks_with CoreMember2 CoreMember3 PeripheryNode1 PeripheryNode2 PeripheryNode3. "
        "NoisyGeneA NoisyGeneB NoisyGeneC NoisyGeneD NoisyGeneE NoisyGeneF. "
        "RootTaxonA SubTaxonB SubTaxonC LeafTaxonD LeafTaxonE LeafTaxonF. "
        "YES NO UNKNOWN."
    )
    tok = train_scaled_bpe_tokenizer(text, target_vocab_size=512)
    cfg = ModelConfig(
        vocab_size=512,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        context_length=128,
    )
    model = MiniGPT(cfg)
    return model, tok


def test_m53_scientific_audit_and_lineage_reconciliation():
    rep = M53ScientificAuditor.audit_lineage_and_metrics()
    assert rep.lineage_reconciled is True
    assert rep.audit_verdict == "PASS"
    assert len(rep.lineage_reconciliations) == 3

    for rec in rep.lineage_reconciliations:
        assert rec.is_disk_file_identical is True
        assert rec.is_base_weights_frozen_post_expansion is True
        assert len(rec.raw_disk_sha256) == 64
        assert len(rec.pre_expansion_param_hash_vocab400) == 16
        assert len(rec.post_expansion_param_hash_vocab1024) == 16
        assert rec.pre_expansion_param_hash_vocab400 != rec.post_expansion_param_hash_vocab1024

    # Sign convention check
    assert "(BWT) = Task1_Score_Final - Task1_Score_Initial" in rep.retention_convention.bwt_definition
    assert "(BF) = Task1_Score_Initial - Task1_Score_Final" in rep.retention_convention.bf_definition


def test_skill_repertoire_registry_and_serialization(tmp_path: Path):
    d_model = 32
    adapter = PersistentCandidateAdapter(d_model=d_model, bottleneck_dim=8, num_classes=3)
    registry = SkillRepertoireRegistry()

    meta = registry.register_skill(
        skill_id="transitive_reachability",
        adapter=adapter,
        base_model_hash="test_base_hash_123",
        training_lineage={"seed": 42, "epochs": 10},
        applicability="Directed transitive relations",
        limitations="Fails on symmetric clusters",
        validation_performance={"accuracy": 0.85, "macro_f1": 0.82},
        retention_measurements={"initial_f1": 0.82, "retained_f1": 0.82},
    )

    assert meta.skill_id == "transitive_reachability"
    assert registry.list_skills() == ["transitive_reachability"]
    assert registry.verify_integrity("transitive_reachability") is True

    # Persist and reload
    save_dir = tmp_path / "skill_repo"
    registry.save_to_dir(save_dir)

    loaded_reg = SkillRepertoireRegistry.load_from_dir(
        save_dir,
        d_model=d_model,
        bottleneck_dim=8,
        num_classes=3,
    )
    assert loaded_reg.list_skills() == ["transitive_reachability"]
    assert loaded_reg.verify_integrity("transitive_reachability") is True
    loaded_meta = loaded_reg.get_metadata("transitive_reachability")
    assert loaded_meta.applicability == "Directed transitive relations"


def test_m54_environment_suite_structure():
    suite = build_m54_environment_suite()
    assert isinstance(suite, SkillBenchmarkEnvironments)

    # 4 Source Skills
    assert "transitive_reachability" in suite.source_skills
    assert "blocked_bridge" in suite.source_skills
    assert "symmetric_equivalence" in suite.source_skills
    assert "hierarchical_boundary" in suite.source_skills

    # 4 Target Tasks
    assert "reporting_hierarchy" in suite.target_tasks
    assert "metabolic_bypass" in suite.target_tasks
    assert "social_coalition" in suite.target_tasks
    assert "alternating_dependency" in suite.target_tasks

    # 5 Epistemic Diagnostics
    assert "parity_cycle" in suite.epistemic_diagnostics
    assert "bipartite_coloring" in suite.epistemic_diagnostics
    assert "dense_clique" in suite.epistemic_diagnostics
    assert "noisy_known" in suite.epistemic_diagnostics
    assert "rare_exception" in suite.epistemic_diagnostics

    # Validate balanced queries across environments
    for env in list(suite.source_skills.values()) + list(suite.target_tasks.values()):
        assert len(env.held_out_test_queries) == 24
        counts = {"YES": 0, "NO": 0, "UNKNOWN": 0}
        for q in env.held_out_test_queries:
            counts[q.expected_answer] += 1
        assert counts["YES"] == 8
        assert counts["NO"] == 8
        assert counts["UNKNOWN"] == 8
        assert len(env.minimal_pairs) > 0


def test_compatibility_evaluation_and_selection_arms():
    model, tok = _make_dummy_model_and_tok()
    suite = build_m54_environment_suite()
    d_model = 32

    registry = SkillRepertoireRegistry()
    for sid in suite.source_skills.keys():
        ad = PersistentCandidateAdapter(d_model=d_model, bottleneck_dim=8, num_classes=3)
        registry.register_skill(
            skill_id=sid,
            adapter=ad,
            base_model_hash="dummy_hash",
            training_lineage={"seed": 42},
            applicability=f"Applicability for {sid}",
            limitations=f"Limitations for {sid}",
            validation_performance={"macro_f1": 0.50},
        )

    neural_head = NeuralSkillSelectorHead(d_model=d_model, num_skills=len(registry.list_skills()), hidden_dim=16)
    evaluator = SkillCompatibilityEvaluator(
        base_model=model,
        tokenizer=tok,
        registry=registry,
        neural_selector=neural_head,
        device=torch.device("cpu"),
    )

    env = suite.target_tasks["reporting_hierarchy"]
    observations = evaluator.sample_verified_observations(env, budget=6, seed=42)
    assert len(observations) == 6

    # Test all 5 selection arms
    for arm in ["fresh", "oracle", "deterministic_evidence", "neural_evidence", "incompatible_control"]:
        dec = evaluator.select_skill(
            arm_name=arm,
            env=env,
            oracle_skill_id="transitive_reachability",
            incompatible_skill_id="symmetric_equivalence",
            budget=6,
            seed=42,
        )
        assert dec.arm_name == arm
        assert dec.selected_skill_id in (["fresh"] + registry.list_skills())
        assert len(dec.scores_by_skill) == len(registry.list_skills())


def test_adaptive_transfer_engine_toy():
    model, tok = _make_dummy_model_and_tok()
    suite = build_m54_environment_suite()
    d_model = 32

    engine = AdaptiveTransferEngine(
        base_model=model,
        tokenizer=tok,
        device=torch.device("cpu"),
        seed=42,
    )
    env = suite.target_tasks["reporting_hierarchy"]
    adapter = PersistentCandidateAdapter(d_model=d_model, bottleneck_dim=8, num_classes=3)

    eval_step = engine.evaluate_adapter(adapter, env)
    assert 0.0 <= eval_step.accuracy <= 1.0
    assert 0.0 <= eval_step.macro_f1 <= 1.0
    assert 0.0 <= eval_step.paired_correctness <= 1.0


def test_retention_isolation_and_replay():
    model, tok = _make_dummy_model_and_tok()
    suite = build_m54_environment_suite()
    d_model = 32

    ret_engine = RetentionEvaluationEngine(
        base_model=model,
        tokenizer=tok,
        device=torch.device("cpu"),
        seed=42,
    )

    src_env = suite.source_skills["transitive_reachability"]
    tgt_env = suite.target_tasks["reporting_hierarchy"]
    adapter = PersistentCandidateAdapter(d_model=d_model, bottleneck_dim=8, num_classes=3)

    target_obs = [(q.source, q.target, q.expected_answer) for q in tgt_env.held_out_test_queries[:4]]
    src_rehearsal = [(q.source, q.target, q.expected_answer) for q in src_env.held_out_test_queries[:2]]

    report = ret_engine.evaluate_retention_regimes(
        source_task_id="transitive_reachability",
        target_task_id="reporting_hierarchy",
        source_adapter=adapter,
        source_env=src_env,
        target_env=tgt_env,
        target_examples=target_obs,
        source_rehearsal_examples=src_rehearsal,
    )

    assert "isolation" in report.strategies
    assert "compatible_replay" in report.strategies
    assert "shared_updating" in report.strategies

    # Isolation guarantees 0.0 forgetting
    iso = report.strategies["isolation"]
    assert iso.backward_forgetting_f1 == 0.0
    assert iso.rollback_triggered is False
    assert report.arithmetic_carry_retention == 1.0
    assert report.chained_logic_retention == 1.0


def test_epistemic_generalization_auditor_toy():
    suite = build_m54_environment_suite()
    rep = EpistemicGeneralizationAuditor.run_epistemic_benchmark(
        epistemic_envs=suite.epistemic_diagnostics,
        frozen_threshold=0.50,
        query_budget=6,
        num_seeds_per_mech=2,
    )
    assert 0.0 <= rep.overall_unseen_recall <= 1.0
    assert 0.0 <= rep.overall_false_alarm_rate <= 1.0
    assert 0.0 <= rep.overall_precision <= 1.0
    assert 0.0 <= rep.overall_f1 <= 1.0
    assert rep.mean_oracle_query_cost > 0.0
