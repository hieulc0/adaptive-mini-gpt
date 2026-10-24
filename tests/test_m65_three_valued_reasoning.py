"""Unit tests for Milestone M65 Three-Valued Reasoning & Representation Sufficiency."""
from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.three_valued_reasoning import (
    LinearFeatureProbe,
    M64ForensicAuditor,
    M65ArchitectureVariant,
    M65SafeguardAuditor,
    NonlinearResidualHead,
    Rank16ResidualAdapter,
    Rank32ResidualAdapter,
    ThreeClassLearningPathwayVerifier,
    ThreeValuedDatasetValidator,
    ThreeValuedReasoningEngine,
    build_m65_new_sealed_suite,
    create_model_for_arm,
)
from adaptive_mini_gpt.guided_experiment.environment import EnvironmentQuery


@pytest.fixture
def base_model_and_tokenizer():
    """Create lightweight base model and dummy tokenizer for unit tests."""
    class DummyTokenizer:
        vocab_size = 256
        def encode(self, text: str) -> list[int]:
            return [hash(w) % 256 for w in text.split()] or [0]
        def decode(self, toks: list[int]) -> str:
            return " ".join(str(t) for t in toks)

    config = ModelConfig(
        vocab_size=256,
        d_model=128,
        n_layers=2,
        n_heads=2,
        d_ff=256,
        context_length=128,
    )
    model = MiniGPT(config)
    return model, DummyTokenizer()


def test_m64_forensic_audit():
    """Verify Phase 1 forensic audit of M64."""
    auditor = M64ForensicAuditor()
    report = auditor.run_forensic_audit()

    # 1. Traces
    assert len(report.pathway_traces) == 6

    # 2. Confusion matrix arithmetic reconciliation
    # Row 0: [8, 0, 0] -> Rec=1.0, Prec=8/17, F1=16/25=0.6400
    # Row 1: [1, 7, 0] -> Rec=7/8, Prec=1.0, F1=14/15=0.9333
    # Row 2: [8, 0, 0] -> Rec=0.0, Prec=0.0, F1=0.0
    # Macro F1 = (0.64 + 0.933333 + 0.0) / 3 = 0.524444...
    rec = report.matrix_reconciliation
    assert abs(rec.macro_f1 - 0.524444) < 1e-4
    assert abs(rec.f1_per_class["NO"] - 0.64) < 1e-4
    assert abs(rec.f1_per_class["YES"] - 0.933333) < 1e-4
    assert rec.f1_per_class["UNKNOWN"] == 0.0

    # 3. Identifiability conflicts
    assert len(report.identifiability_conflicts) >= 1
    assert any("AeroModuleM0" in c.query_text for c in report.identifiability_conflicts)


def test_dataset_validator():
    """Verify Phase 2 dataset validator detects conflicting labels."""
    clean_queries = [
        EnvironmentQuery("q1", "A", "B", "T", "Does A link B?", "YES"),
        EnvironmentQuery("q2", "B", "A", "T", "Does B link A?", "NO"),
        EnvironmentQuery("q3", "X", "Y", "T", "Does X link Y?", "UNKNOWN"),
    ]
    rep_clean = ThreeValuedDatasetValidator.validate_query_set(clean_queries, "clean")
    assert rep_clean.validation_passed
    assert rep_clean.identifiability_score == 1.0

    conflicting_queries = [
        EnvironmentQuery("q1", "A", "B", "T", "Does A link B?", "YES"),
        EnvironmentQuery("q2", "A", "B", "T", "Does A link B?", "UNKNOWN"),  # Identical input, conflicting label!
    ]
    rep_conflict = ThreeValuedDatasetValidator.validate_query_set(conflicting_queries, "conflict")
    assert not rep_conflict.validation_passed
    assert len(rep_conflict.conflicts_detected) == 1


def test_three_class_smoke_test(base_model_and_tokenizer):
    """Verify Phase 3 smoke test checks gradient flow and overfitting on separable data."""
    model, tokenizer = base_model_and_tokenizer
    device = torch.device("cpu")
    verifier = ThreeClassLearningPathwayVerifier(
        base_model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=5e-3,
    )
    # Run short smoke test (15 epochs)
    report = verifier.run_smoke_test(num_epochs=15)
    assert report.unknown_gradient_norm > 0.0
    assert report.final_overfit_loss < report.initial_train_loss


def test_candidate_architectures():
    """Verify Phase 4 parameter counts and architecture factory."""
    # Rank 16: 2,371 parameters
    r16 = create_model_for_arm(M65ArchitectureVariant.ARM_A_RANK16_BASELINE, d_model=128)
    assert r16.count_parameters() == 2371

    # Rank 32: 4,483 parameters
    r32 = create_model_for_arm(M65ArchitectureVariant.ARM_B_RANK32_EXPANDED, d_model=128)
    assert r32.count_parameters() == 4483

    # Nonlinear head: 2,643 parameters
    nonlin = create_model_for_arm(M65ArchitectureVariant.ARM_C_NONLINEAR_HEAD, d_model=128)
    assert nonlin.count_parameters() == 2643

    # Linear probe: 387 parameters (128*3 + 3)
    probe = create_model_for_arm(M65ArchitectureVariant.ARM_E_LINEAR_FEATURE_PROBE, d_model=128)
    assert probe.count_parameters() == 387


def test_new_sealed_suite():
    """Verify Phase 7 new independent sealed environments validate with 0% conflicts."""
    suite = build_m65_new_sealed_suite(use_evidence_prefix=True)
    assert len(suite) == 5
    for env in suite:
        assert len(env.held_out_test_queries) == 24
        rep = ThreeValuedDatasetValidator.validate_query_set(env.held_out_test_queries, env.env_id)
        assert rep.validation_passed


def test_m65_safeguards(base_model_and_tokenizer):
    """Verify Phase 8 base parameter invariance and epistemic decoupling."""
    model, tokenizer = base_model_and_tokenizer
    device = torch.device("cpu")
    auditor = M65SafeguardAuditor(model, tokenizer, device, seed=42)
    rep = auditor.audit_safeguards()
    assert rep.base_parameter_hash_invariant
    assert rep.arithmetic_carry_retention == 1.0
    assert rep.chained_logic_retention == 1.0
    assert rep.relation_vs_rule_decoupled
    assert rep.all_safeguards_passed
