"""Unit tests for M44: Structural Analogy, Memory Transfer, and Learning Diagnostics."""
from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.analogy.audit import M43DiagnosticAuditor
from adaptive_mini_gpt.analogy.positive_control import MemoryPositiveControlEvaluator
from adaptive_mini_gpt.analogy.structural_schema import (
    CanonicalGraphSignature,
    DeterministicStructuralMatcher,
    RelationProperties,
    StructuralAnalogyExperience,
)
from adaptive_mini_gpt.analogy.analogy_experiments import (
    StructuralAnalogyExperimentRunner,
    build_analogy_task_family,
    get_m44_benchmark_families,
)
from adaptive_mini_gpt.analogy.sequential_retention import SequentialRetentionEvaluator
from adaptive_mini_gpt.analogy.language_diagnostic import LanguageDiagnosticEvaluator
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.meta.benchmark_suite import MetaTask
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT


@pytest.fixture
def dummy_analogy_setup() -> tuple[MiniGPT, BPETokenizer, torch.device]:
    device = torch.device("cpu")
    tok = BPETokenizer()
    corpus = (
        "YES NO UNKNOWN [ANALOGY_SCHEMA] [EXEMPLAR] [PREMISES] [QUESTION] [ANSWER] Pattern Rule Ex "
        "JuniorDev TechLead EngineeringVP ChiefExec LocalCluster SectorRegion DomainSphere UniversalMacro "
        "AcidReagent BaseSolution VolatileSolvent InertBuffer ProtoRootX ArchaicFormY DialectZ ModernLexemeW "
        "EnzymeAlpha SubstrateBeta ProductGamma MetaboliteDelta GuildSolar FactionLunar AllianceNova OrderZenith "
        "reports_to manages subsumed_by encloses incompatible_with derives_from gives_rise_to catalyzes "
        "inhibited_by rival_of Does Is it verified that GhostEntity UnknownParticleX NonExistentElement "
        "AlphaPhantom BetaPhantom"
    )
    tok.train(corpus, target_vocab_size=360)

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


def test_m43_root_cause_forensic_audit(dummy_analogy_setup: tuple[MiniGPT, BPETokenizer, torch.device]) -> None:
    model, tok, device = dummy_analogy_setup
    families = get_m44_benchmark_families()
    eval_tasks = families["linguistic_etymology"][0].held_out_test_pool

    rep = M43DiagnosticAuditor.audit_m43_pipeline(model, tok, device, eval_tasks)
    assert rep.adapter_gradient_active is True
    assert rep.trainable_weights_change is True
    assert rep.positive_control.is_overfitted_successfully is True
    assert rep.reproduced_constant_accuracy == 0.3500
    assert rep.constant_prediction_class == "YES"
    assert len(rep.budget_confusion_matrices) == 6
    assert rep.readiness_for_m44_analogy is True


def test_canonical_graph_signature_and_structural_matcher() -> None:
    rel_props = RelationProperties("reports_to", is_transitive=True, is_symmetric=False)

    # 1. Chain 2-hop task
    t_2hop = MetaTask(
        "t1", "corp", "test",
        ["JuniorDev reports_to TechLead", "TechLead reports_to EngineeringVP"],
        "Does JuniorDev reports_to EngineeringVP?",
        "YES", "two_hop", ["JuniorDev", "EngineeringVP"]
    )
    sig_2hop = DeterministicStructuralMatcher.extract_signature(t_2hop, rel_props)
    assert sig_2hop.pattern_type == "CHAIN_2HOP"
    assert sig_2hop.path_length == 2
    assert sig_2hop.is_grounded is True

    # 2. Asymmetric reversal
    t_rev = MetaTask(
        "t2", "corp", "test",
        ["JuniorDev reports_to TechLead"],
        "Does TechLead reports_to JuniorDev?",
        "NO", "asymmetric", ["TechLead", "JuniorDev"]
    )
    sig_rev = DeterministicStructuralMatcher.extract_signature(t_rev, rel_props)
    assert sig_rev.pattern_type == "ASYMMETRIC_REVERSAL"
    assert sig_rev.involves_inversion is True

    # 3. Ungrounded task
    t_unk = MetaTask(
        "t3", "corp", "test",
        ["JuniorDev reports_to TechLead"],
        "Does JuniorDev reports_to PhantomGhost?",
        "UNKNOWN", "ungrounded", ["JuniorDev", "PhantomGhost"]
    )
    sig_unk = DeterministicStructuralMatcher.extract_signature(t_unk, rel_props)
    assert sig_unk.pattern_type == "UNGROUNDED"
    assert sig_unk.is_grounded is False

    # 4. Cross-domain structural analogy match
    exp = StructuralAnalogyExperience(
        experience_id="exp_test",
        original_family="corporate_hierarchy",
        relation_name="reports_to",
        relation_properties=rel_props,
        canonical_signature=sig_2hop,
        abstract_rule="Transitive 2-hop chain",
        exemplar_premises=["JuniorDev reports_to TechLead", "TechLead reports_to EngineeringVP"],
        exemplar_question="Does JuniorDev reports_to EngineeringVP?",
        verified_verdict="YES",
    )

    # Completely different domain: metabolic cascade (0% word overlap)
    rel_metabolic = RelationProperties("catalyzes", is_transitive=True, is_symmetric=False)
    t_metabolic = MetaTask(
        "tm1", "metabolic", "test",
        ["EnzymeAlpha catalyzes SubstrateBeta", "SubstrateBeta catalyzes ProductGamma"],
        "Does EnzymeAlpha catalyzes ProductGamma?",
        "YES", "two_hop", ["EnzymeAlpha", "ProductGamma"]
    )
    score = DeterministicStructuralMatcher.match_analogy(t_metabolic, rel_metabolic, exp)
    assert score >= 0.8  # High structural analogy score despite completely disjoint entities and words


def test_positive_control_scenarios() -> None:
    evaluator = MemoryPositiveControlEvaluator(max_exemplars=2)
    rep = evaluator.run_all_positive_controls()

    # Scenario A (Same Domain): Lexical recall should be high
    assert rep.scenario_a_same_family["lexical"].recall >= 0.5
    assert rep.scenario_a_same_family["structure_aware"].recall >= 0.5

    # Scenario B (Cross-Domain Analogy): Lexical recall MUST be 0 (0% word overlap) while Structure-aware recall is 1.0!
    assert rep.scenario_b_cross_domain_analogy["lexical"].recall == 0.0
    assert rep.scenario_b_cross_domain_analogy["structure_aware"].recall == 1.0
    assert rep.overall_structure_aware_advantage == 1.0

    # Scenario C (Distractor Rejection): Structure-aware correctly rejects incompatible distractors
    assert rep.scenario_c_distractor_rejection["structure_aware"].correct_rejection_rate >= 0.5


def test_analogy_approaches_and_learning_curves(dummy_analogy_setup: tuple[MiniGPT, BPETokenizer, torch.device]) -> None:
    model, tok, device = dummy_analogy_setup
    families = get_m44_benchmark_families()
    target_fam, target_props = families["linguistic_etymology"]

    exp = StructuralAnalogyExperience(
        experience_id="exp_corp_1",
        original_family="corporate_hierarchy",
        relation_name="reports_to",
        relation_properties=RelationProperties("reports_to", is_transitive=True, is_symmetric=False),
        canonical_signature=CanonicalGraphSignature("CHAIN_2HOP", 3, 2, 2, False, True),
        abstract_rule="Transitive 2-hop hierarchy",
        exemplar_premises=["JuniorDev reports_to TechLead", "TechLead reports_to EngineeringVP"],
        exemplar_question="Does JuniorDev reports_to EngineeringVP?",
        verified_verdict="YES",
    )
    runner = StructuralAnalogyExperimentRunner(model, tok, device, [exp])

    # Run quick evaluation on budgets [0, 2]
    rep_fresh = runner.run_approach("APPROACH_A_FRESH", target_fam, target_props, budgets=[0, 2])
    assert len(rep_fresh.curve) == 2
    assert rep_fresh.curve[0].budget == 0
    assert rep_fresh.curve[1].budget == 2
    assert rep_fresh.area_under_curve >= 0.0

    rep_struct = runner.run_approach("APPROACH_C_STRUCTURE_MATCHED", target_fam, target_props, budgets=[0, 2])
    assert len(rep_struct.curve) == 2


def test_multi_episode_sequential_retention(dummy_analogy_setup: tuple[MiniGPT, BPETokenizer, torch.device]) -> None:
    model, tok, device = dummy_analogy_setup
    families = get_m44_benchmark_families()
    episodes = [
        families["corporate_hierarchy"][0],
        families["containment_taxonomy"][0],
    ]
    evaluator = SequentialRetentionEvaluator(model, tok, device)
    rep = evaluator.run_benchmark(episodes, practice_budget=2)

    assert rep.base_parameter_hash_invariant is True
    assert rep.canonical_carry_intact is True
    assert rep.canonical_logic_intact is True
    assert rep.modular_isolation.backward_transfer == 0.0


def test_language_representation_diagnostic(dummy_analogy_setup: tuple[MiniGPT, BPETokenizer, torch.device]) -> None:
    model, tok, device = dummy_analogy_setup
    families = get_m44_benchmark_families()
    test_fam, _ = families["linguistic_etymology"]

    evaluator = LanguageDiagnosticEvaluator(model, tok, device)
    rep = evaluator.run_diagnostic(test_fam)
    assert rep.diagnostic_verdict == "STRUCTURAL_ANALOGY_INDEPENDENT_OF_LANGUAGE_SCALE"
    assert rep.base_foundation.parameter_count == rep.language_enhanced_foundation.parameter_count
