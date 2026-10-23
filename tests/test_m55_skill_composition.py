"""Unit tests for Milestone M55: Representation-Preserving Skill Composition."""
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
from adaptive_mini_gpt.skill_composition import (
    M54ScientificAuditor,
    M54ScientificAuditReport,
    RepresentationDriftAnalyzer,
    RepresentationDriftMeasurement,
    ResidualSkillAdapter,
    GatedResidualSkillAdapter,
    build_m55_environment_suite,
    CompositionBenchmarkEnvironments,
    CompositionTransferEngine,
    CompositionRetentionAuditor,
    EpistemicDenseDiagnosticAuditor,
    DenseCliqueDiagnosticReport,
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
        "SupplierTier1 ships_freight_downstream_to PortTerminalA2 CustomsStationB3 RegionalHubC4. "
        "LocalDepotD5 DistributionCenterE6 RetailOutletF7 CustomerEndpointG8. "
        "PrecursorSubstrateAlpha catalyzes_synthesis_of IntermediateEnzymeBeta. "
        "InhibitedChokePointGamma KinaseFactorDelta DownstreamMetaboliteEpsilon AlternativeShuntOmega. "
        "SovereignStateNorth holds_mutual_pact_with AlliancePartnerEast. "
        "FederationSouth TreatyStateWest NeutralObservatory IsolatedFaction. "
        "FeedstockStage0 directly_feeds_stage RefiningStage1 SynthesisStage2. "
        "PackagingStage3 InspectionStage4 DispatchStage5. "
        "CoreMember1 strongly_interlinks_with CoreMember2 CoreMember3 PeripheryNode1 PeripheryNode2 PeripheryNode3. "
        "DenseNode1 in_same_security_zone DenseNode2 DenseNode3 DenseNode4 DenseNode5 DenseNode6. "
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


def test_m54_scientific_audit_and_oracle_reconciliation():
    rep = M54ScientificAuditor.audit_m54_findings()
    assert rep.audit_verdict == "PASS"
    assert rep.lineage_verified is True
    assert rep.base_parameters_frozen is True
    assert rep.metrics.fresh_auc == 0.4477
    assert rep.metrics.deterministic_evidence_auc == 0.4425
    assert rep.metrics.deterministic_jumpstart_gain_pp == 19.24
    assert "empirical neural transfer utility" in rep.oracle_analysis.structural_compatibility_vs_empirical_utility


def test_representation_drift_analyzer():
    model, tok = _make_dummy_model_and_tok()
    suite = build_m55_environment_suite()
    d_model = 32

    src_adapter = PersistentCandidateAdapter(d_model=d_model, bottleneck_dim=8, num_classes=3)
    adapted_adapter = PersistentCandidateAdapter(d_model=d_model, bottleneck_dim=8, num_classes=3)

    analyzer = RepresentationDriftAnalyzer(
        base_model=model,
        tokenizer=tok,
        device=torch.device("cpu"),
    )

    l2_dist, rel_pct = analyzer.compute_parameter_displacement(src_adapter, adapted_adapter)
    assert l2_dist >= 0.0
    assert rel_pct >= 0.0

    src_env = suite.source_skills["transitive_reachability"]
    tgt_env = suite.target_tasks["reporting_hierarchy"]

    drift = analyzer.analyze_drift(
        source_env=src_env,
        target_env=tgt_env,
        initial_source_adapter=src_adapter,
        adapted_source_adapter=adapted_adapter,
        target_jumpstart_f1=0.45,
        target_final_f1=0.44,
    )

    assert isinstance(drift, RepresentationDriftMeasurement)
    assert -1.0 <= drift.source_probes_cosine_similarity <= 1.0
    assert -1.0 <= drift.target_probes_cosine_similarity <= 1.0
    assert 0.0 <= drift.source_task_initial_f1 <= 1.0


def test_residual_and_gated_skill_adapter_architectures(tmp_path: Path):
    d_model = 32
    src_adapter = PersistentCandidateAdapter(d_model=d_model, bottleneck_dim=8, num_classes=3)

    # 1. ResidualSkillAdapter
    res_adapter = ResidualSkillAdapter(
        source_adapter=src_adapter,
        d_model=d_model,
        residual_bottleneck_dim=8,
        num_classes=3,
    )

    # Verify source weights are frozen
    for p in res_adapter.source_adapter.parameters():
        assert p.requires_grad is False

    # Verify residual weights are trainable
    trainable_p = [p for p in res_adapter.residual_head.parameters() if p.requires_grad]
    assert len(trainable_p) > 0
    assert res_adapter.num_trainable_parameters() < res_adapter.num_total_parameters()

    x = torch.randn(2, 5, d_model)
    out_res = res_adapter(x)
    assert out_res.shape == (2, 3)
    out_src_only = res_adapter.evaluate_source_only(x)
    assert out_src_only.shape == (2, 3)

    save_path = tmp_path / "residual.pt"
    h = res_adapter.save_checkpoint(save_path)
    assert len(h) == 16
    assert save_path.exists()

    # 2. GatedResidualSkillAdapter
    gated_adapter = GatedResidualSkillAdapter(
        source_adapter=src_adapter,
        d_model=d_model,
        residual_bottleneck_dim=8,
        num_classes=3,
    )

    for p in gated_adapter.source_adapter.parameters():
        assert p.requires_grad is False

    out_gated = gated_adapter(x)
    assert out_gated.shape == (2, 3)
    save_gated_path = tmp_path / "gated.pt"
    h_g = gated_adapter.save_checkpoint(save_gated_path)
    assert len(h_g) == 16


def test_m55_environment_suite_structure():
    suite = build_m55_environment_suite()
    assert isinstance(suite, CompositionBenchmarkEnvironments)

    # 4 Source Skills
    assert "transitive_reachability" in suite.source_skills
    assert "blocked_bridge" in suite.source_skills
    assert "symmetric_equivalence" in suite.source_skills
    assert "hierarchical_boundary" in suite.source_skills

    # 5 Target Tasks
    assert "reporting_hierarchy" in suite.target_tasks
    assert "supply_chain_pipeline" in suite.target_tasks
    assert "metabolic_bypass" in suite.target_tasks
    assert "social_coalition" in suite.target_tasks
    assert "alternating_dependency" in suite.target_tasks

    # Diagnostic Environments
    assert "dense_clique_with_exceptions" in suite.dense_clique_diagnostics
    assert "noisy_dense_known" in suite.dense_clique_diagnostics

    for env in list(suite.source_skills.values()) + list(suite.target_tasks.values()):
        assert len(env.held_out_test_queries) == 24
        counts = {"YES": 0, "NO": 0, "UNKNOWN": 0}
        for q in env.held_out_test_queries:
            counts[q.expected_answer] += 1
        assert counts["YES"] == 8
        assert counts["NO"] == 8
        assert counts["UNKNOWN"] == 8
        assert len(env.minimal_pairs) > 0


def test_composition_transfer_engine_toy():
    model, tok = _make_dummy_model_and_tok()
    suite = build_m55_environment_suite()
    d_model = 32

    engine = CompositionTransferEngine(
        base_model=model,
        tokenizer=tok,
        device=torch.device("cpu"),
        seed=42,
    )
    env = suite.target_tasks["reporting_hierarchy"]
    adapter = PersistentCandidateAdapter(d_model=d_model, bottleneck_dim=8, num_classes=3)

    step = engine.evaluate_model(adapter, env)
    assert 0.0 <= step.accuracy <= 1.0
    assert 0.0 <= step.macro_f1 <= 1.0


def test_composition_retention_auditor():
    model, tok = _make_dummy_model_and_tok()
    suite = build_m55_environment_suite()
    d_model = 32

    auditor = CompositionRetentionAuditor(
        base_model=model,
        tokenizer=tok,
        device=torch.device("cpu"),
    )

    src_adapter = PersistentCandidateAdapter(d_model=d_model, bottleneck_dim=8, num_classes=3)
    res_adapter = ResidualSkillAdapter(
        source_adapter=src_adapter,
        d_model=d_model,
        residual_bottleneck_dim=8,
        num_classes=3,
    )

    src_env = suite.source_skills["transitive_reachability"]
    tgt_env = suite.target_tasks["reporting_hierarchy"]

    res = auditor.audit_retention(
        source_task_id="transitive_reachability",
        target_task_id="reporting_hierarchy",
        initial_source_adapter=src_adapter,
        trained_composition_module=res_adapter,
        source_env=src_env,
        target_env=tgt_env,
        composition_type="residual",
    )

    # Frozen source guarantees zero backward forgetting
    assert res.backward_forgetting_f1 == 0.0
    assert res.rollback_triggered is False
    assert res.arithmetic_carry_retention == 1.0
    assert res.chained_logic_retention == 1.0
    assert res.base_parameter_hash_invariant is True


def test_dense_clique_diagnostic_auditor_toy():
    suite = build_m55_environment_suite()
    rep = EpistemicDenseDiagnosticAuditor.run_dense_diagnostic(
        dense_envs=suite.dense_clique_diagnostics,
        query_budget=6,
        frozen_threshold=0.50,
        num_episodes=2,
    )
    assert isinstance(rep, DenseCliqueDiagnosticReport)
    assert 0.0 <= rep.random_probing.unseen_dense_detection_rate <= 1.0
    assert 0.0 <= rep.targeted_boundary_probing.unseen_dense_detection_rate <= 1.0
    assert 0.0 <= rep.targeted_boundary_probing.noisy_control_false_alarm_rate <= 1.0
