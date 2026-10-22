"""Unit tests for M45: Epistemic Uncertainty and Transferable Learning."""
from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.uncertainty.audit import M44AttributionAuditor
from adaptive_mini_gpt.uncertainty.epistemic_benchmark import (
    build_epistemic_world,
    get_m45_epistemic_worlds,
)
from adaptive_mini_gpt.uncertainty.tri_state_classifier import (
    EpistemicUncertaintyEvaluator,
    TriStateEpistemicHead,
)
from adaptive_mini_gpt.uncertainty.learned_transfer import (
    LearnedStructuralTransferEvaluator,
    NeuralSchemaPredictorHead,
)
from adaptive_mini_gpt.uncertainty.corrective_learning import CorrectiveSelfLearningRunner
from adaptive_mini_gpt.uncertainty.learning_efficiency import EpistemicEfficiencyBenchmarkRunner


@pytest.fixture
def dummy_epistemic_setup() -> tuple[MiniGPT, BPETokenizer, torch.device]:
    device = torch.device("cpu")
    tok = BPETokenizer()
    corpus = (
        "YES NO UNKNOWN [PREMISES] [QUESTION] [ANSWER] [EXEMPLAR] [NEURAL_SCHEMA] [CORRECTED_SCHEMA] Pattern Rule "
        "ExecutiveBoard ManagingDirector DivisionHead TeamLead ExternalAuditor HamletDistrict ProvincialZone "
        "NationalTerritory ContinentalPlate OceanTrench ApexPredator MesoCarnivore SmallHerbivore PrimaryProducer "
        "DeepSeaBenthos ContemporaryTreatise MidCenturyTheory ClassicalMonograph AncientManuscript UnpublishedDraft "
        "BinaryPulsarA BinaryPulsarB NeutronCompanion AccretingSingularity RogueComet FederationAlpha RepublicBeta "
        "KingdomGamma UnionDelta NeutralEnclave supervises supervised_by situated_within contains_region preys_upon "
        "preyed_on_by cites_foundational_work cited_by_work gravitationally_coupled_with allied_with "
        "Does Is it verified that PhantomNodeOmega UnseenEntityAlpha VoidGhostNode PhantomAgentX DarkNebula "
        "QuantumVoid UnobservedStateZ"
    )
    tok.train(corpus, target_vocab_size=380)

    config = ModelConfig(
        vocab_size=tok.vocab_size,
        context_length=128,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        dropout=0.0,
    )
    model = MiniGPT(config).to(device)
    return model, tok, device


def test_m44_attribution_audit() -> None:
    rep = M44AttributionAuditor.audit_m44_results()
    assert len(rep.per_seed_results) == 3
    assert rep.attribution_breakdown.oracle_schema_contribution > 0.30
    assert "confound" in rep.positive_control_artifact_diagnosis.lower() or "oracle" in rep.positive_control_artifact_diagnosis.lower()
    assert "retention" in rep.sequential_retention_reconciliation.lower()


def test_epistemic_benchmark_generation() -> None:
    worlds = get_m45_epistemic_worlds()
    assert "epistemological_citation" in worlds
    assert "astrophysical_gravitation" in worlds
    assert "diplomatic_alliance" in worlds

    target_world = worlds["epistemological_citation"]
    test_tasks = target_world.held_out_test_pool
    assert len(test_tasks) == 24

    yes_count = sum(1 for t in test_tasks if t.expected_verdict == "YES")
    no_count = sum(1 for t in test_tasks if t.expected_verdict == "NO")
    unk_count = sum(1 for t in test_tasks if t.expected_verdict == "UNKNOWN")

    # Balanced 3-class distribution
    assert yes_count == 8
    assert no_count == 8
    assert unk_count == 8


def test_tri_state_classifier_arms(dummy_epistemic_setup: tuple[MiniGPT, BPETokenizer, torch.device]) -> None:
    model, tok, device = dummy_epistemic_setup
    worlds = get_m45_epistemic_worlds()
    target_world = worlds["epistemological_citation"]
    source_worlds = [worlds["corporate_governance"], worlds["geospatial_containment"]]

    evaluator = EpistemicUncertaintyEvaluator(model, tok, device)

    # 1. Constant Baseline
    base_yes = evaluator.evaluate_constant_baseline(target_world.held_out_test_pool, "YES")
    assert base_yes.overall_accuracy == pytest.approx(8 / 24, abs=1e-3)
    assert base_yes.yes_recall == 1.0
    assert base_yes.no_recall == 0.0
    assert base_yes.unknown_recall == 0.0

    # 2. External Solver (Arm D)
    rep_d = evaluator.evaluate_external_solver(target_world)
    assert rep_d.overall_accuracy == 1.0
    assert rep_d.macro_f1 == 1.0

    # 3. Arm B: Balanced Tri-State Head
    rep_b = evaluator.evaluate_arm_b_tri_state_head(target_world, practice_budget=4)
    assert rep_b.overall_accuracy >= 0.0
    assert rep_b.expected_calibration_error >= 0.0
    assert rep_b.brier_score >= 0.0

    # 4. Arm C: Analogy Supervised Head
    rep_c = evaluator.evaluate_arm_c_analogy_supervised_head(target_world, source_worlds, practice_budget=4)
    assert rep_c.overall_accuracy >= 0.0


def test_learned_structural_transfer_ablation(dummy_epistemic_setup: tuple[MiniGPT, BPETokenizer, torch.device]) -> None:
    model, tok, device = dummy_epistemic_setup
    worlds = get_m45_epistemic_worlds()
    target_world = worlds["epistemological_citation"]
    source_worlds = [worlds["corporate_governance"]]

    evaluator = LearnedStructuralTransferEvaluator(model, tok, device, memory_bank=[])
    report = evaluator.evaluate_all_regimes(target_world, source_worlds, practice_budget=4)

    assert "REGIME_1_RAW_UNFAMILIAR" in report.results_by_regime
    assert "REGIME_3_DETERMINISTIC_ORACLE_SCHEMA" in report.results_by_regime
    assert "REGIME_4_NEURAL_PREDICTED_SCHEMA" in report.results_by_regime
    assert report.results_by_regime["REGIME_4_NEURAL_PREDICTED_SCHEMA"].schema_prediction_accuracy is not None


def test_closed_loop_corrective_learning(dummy_epistemic_setup: tuple[MiniGPT, BPETokenizer, torch.device]) -> None:
    model, tok, device = dummy_epistemic_setup
    worlds = get_m45_epistemic_worlds()
    target_world = worlds["epistemological_citation"]

    runner = CorrectiveSelfLearningRunner(model, tok, device)
    comp_rep = runner.run_closed_loop_comparison(target_world, total_episodes=4, budget_per_episode=2)

    assert comp_rep.closed_loop_corrective.episodes_executed == 4
    assert comp_rep.static_schedule.canonical_skills_preserved is True
    assert comp_rep.closed_loop_corrective.canonical_skills_preserved is True


def test_epistemic_sample_efficiency_curves(dummy_epistemic_setup: tuple[MiniGPT, BPETokenizer, torch.device]) -> None:
    model, tok, device = dummy_epistemic_setup
    worlds = get_m45_epistemic_worlds()
    target_world = worlds["epistemological_citation"]

    runner = EpistemicEfficiencyBenchmarkRunner(model, tok, device)
    rep = runner.run_strategy_evaluation("FRESH_NEURAL_LEARNER", target_world, budgets=[0, 2])

    assert len(rep.curve) == 2
    assert rep.curve[0].budget == 0
    assert rep.curve[1].budget == 2
    assert rep.area_under_curve >= 0.0
