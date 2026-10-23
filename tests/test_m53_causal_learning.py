"""Unit tests for M53: Causal Learning Audit and Transferable Skill Acquisition."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import pytest
import torch
import numpy as np

from adaptive_mini_gpt.data.bpe_scaling import train_scaled_bpe_tokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.causal_learning_audit import (
    M52ScientificAuditor,
    M52AuditReport,
    build_m53_environment_suite,
    CausalAuditEnvironmentSuite,
    PersistentCandidateAdapter,
    FactorialStudyEngine,
    FactorialCellResult,
    SkillTransferEngine,
    TransferEvaluationReport,
    EpistemicAuditEngine,
    EpistemicAuditReport,
)


def _make_dummy_model_and_tok() -> tuple[MiniGPT, Any]:
    text = (
        "GeneLocusAlpha activates_transcription_of PromoterSiteBeta. EnhancerUnitGamma. "
        "TranscriptionFactorDelta RNAIntermediateEpsilon RibosomeTargetZeta. "
        "SupplierAnchor treaty_partner_with CustomsPointA CustomsPointB. "
        "ConsortiumRelayX ConsortiumRelayY LocalWarehouse. "
        "MeshRegistry in_same_security_zone GatewayCluster PaymentServiceNode. "
        "DomainArchaea taxonomically_subsumes PhylumMethano ClassHalobacteria. "
        "ProtoRootX phonologically_evolved_into BranchDialectY ArchaicFormZ. "
        "MetaboliteSubstrate0 catalyzes_synthesis_of EnzymeComplex1. "
        "QuantumSourceQ0 routes_quantum_information_to RepeaterCouplerQ1. "
        "RingNodeA shares_parity_state_with RingNodeB. "
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


def test_m52_forensic_audit():
    auditor = M52ScientificAuditor()
    report = auditor.audit_report_file("runs/kaggle-m52/output/m52_report.json")

    assert report.is_reconciled is True
    assert report.code_trace_line == 302
    assert "if p == g" in report.corrected_code_statement
    assert "CRITICAL METRIC BUG CONFIRMED" in report.root_cause_diagnosis
    assert report.true_accuracy_lower_bound < report.true_accuracy_upper_bound


def test_environment_suite_construction():
    suite = build_m53_environment_suite()
    assert isinstance(suite, CausalAuditEnvironmentSuite)

    assert "env_train_genomics_dag" in suite.train_environments
    assert "env_dev_phylogenetics_hier" in suite.dev_environments
    assert "env_sealed_biochem_trans" in suite.sealed_test_environments
    assert "env_sealed_quantum_bridge" in suite.sealed_test_environments
    assert "env_sealed_social_parity" in suite.sealed_test_environments

    for env_id, env in suite.all_environments().items():
        assert len(env.entities) >= 5
        # 24 held-out test queries (8 YES, 8 NO, 8 UNKNOWN)
        assert len(env.held_out_test_queries) == 24
        ans_counts = {"YES": 0, "NO": 0, "UNKNOWN": 0}
        for q in env.held_out_test_queries:
            ans_counts[q.expected_answer] += 1
        assert ans_counts["YES"] == 8
        assert ans_counts["NO"] == 8
        assert ans_counts["UNKNOWN"] == 8

        # Verified minimal pairs
        assert len(env.minimal_pairs) > 0


def test_persistent_candidate_adapter_serialization(tmp_path: Path):
    d_model = 32
    adapter = PersistentCandidateAdapter(d_model=d_model, bottleneck_dim=8, num_classes=3)
    adapter.eval()
    x = torch.randn(2, 5, d_model)
    out1 = adapter(x)
    assert out1.shape == (2, 3)

    chk_path = tmp_path / "adapter.pt"
    h1 = adapter.save_checkpoint(chk_path)
    assert chk_path.exists()

    reloaded = PersistentCandidateAdapter.load_checkpoint(chk_path)
    reloaded.eval()
    h2 = reloaded.compute_parameter_hash()
    assert h1 == h2

    out2 = reloaded(x)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_factorial_cell_and_full_study():
    model, tok = _make_dummy_model_and_tok()
    suite = build_m53_environment_suite()
    env = suite.train_environments["env_train_genomics_dag"]

    engine = FactorialStudyEngine(model=model, tokenizer=tok, device=torch.device("cpu"), seed=42)

    # Test single cell
    cell_res = engine.evaluate_cell(env, objective="pointwise", policy="random", budget=4)
    assert isinstance(cell_res, FactorialCellResult)
    assert 0.0 <= cell_res.accuracy <= 1.0
    assert 0.0 <= cell_res.macro_f1 <= 1.0
    assert 0.0 <= cell_res.paired_correctness <= 1.0

    # Test full 3x4 factorial study
    report = engine.run_full_factorial_study(env, budget=4)
    assert len(report.cell_results) == 12
    assert "pointwise" in report.objective_means
    assert "paired" in report.objective_means
    assert "combined" in report.objective_means
    assert "random" in report.policy_means
    assert "eig" in report.policy_means
    assert "CAUSAL ATTRIBUTION RESOLVED" in report.causal_verdict


def test_skill_transfer_and_retention():
    model, tok = _make_dummy_model_and_tok()
    suite = build_m53_environment_suite()

    src_env = suite.train_environments["env_train_genomics_dag"]
    tgt_env = suite.sealed_test_environments["env_sealed_biochem_trans"]
    incomp_env = suite.sealed_test_environments["env_sealed_social_parity"]

    engine = SkillTransferEngine(model=model, tokenizer=tok, device=torch.device("cpu"), seed=42)
    report = engine.run_transfer_benchmark(
        source_env=src_env,
        target_env=tgt_env,
        incompatible_env=incomp_env,
        budget=4,
    )

    assert isinstance(report, TransferEvaluationReport)
    assert "fresh" in report.condition_results
    assert "compatible_transfer" in report.condition_results
    assert "replay" in report.condition_results
    assert "incompatible_control" in report.condition_results
    assert report.base_parameter_hash_invariant is True
    assert report.arithmetic_carry_retention == 1.00
    assert report.chained_logic_retention == 1.00


def test_epistemic_audit_engine():
    suite = build_m53_environment_suite()
    known_envs = [suite.train_environments["env_train_genomics_dag"]]
    ooh_envs = [suite.sealed_test_environments["env_sealed_social_parity"]]

    report = EpistemicAuditEngine.audit_epistemic_tracker(
        known_envs=known_envs,
        ooh_envs=ooh_envs,
        query_budget=6,
        threshold=0.50,
    )

    assert isinstance(report, EpistemicAuditReport)
    assert report.known_regime.num_episodes > 0
    assert report.noisy_regime.num_episodes > 0
    assert report.out_of_hypothesis_regime.num_episodes > 0
    assert 0.0 <= report.known_regime.false_positive_rate <= 1.0
    assert 0.0 <= report.out_of_hypothesis_regime.detection_rate_at_threshold <= 1.0
