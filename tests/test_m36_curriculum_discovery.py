"""Unit tests for M36: Self-Directed Capability-Gap Discovery and Curriculum Learning."""

import pytest
import torch

from adaptive_mini_gpt.curriculum_discovery.curriculum_generator import (
    VerifiedCurriculumGenerator,
)
from adaptive_mini_gpt.curriculum_discovery.gap_analyzer import (
    CapabilityGapAnalyzer,
)
from adaptive_mini_gpt.curriculum_discovery.multi_episode_learner import (
    MultiEpisodeLearner,
)
from adaptive_mini_gpt.curriculum_discovery.objective import (
    HypothesizedCause,
    LearningObjective,
)
from adaptive_mini_gpt.curriculum_discovery.objective_selector import (
    LargestGapObjectiveSelector,
    NeuralAssistedObjectiveSelector,
    RoundRobinObjectiveSelector,
)
from adaptive_mini_gpt.curriculum_discovery.structural_evolution import (
    StructuralEvolutionGate,
    StructuralModificationType,
    StructuralProposal,
)
from adaptive_mini_gpt.curriculum_discovery.uncertainty import (
    EnhancedMetacognitivePolicy,
    EvidenceFeatures,
    extract_evidence_features,
)
from adaptive_mini_gpt.evaluation.multi_domain import DomainType
from adaptive_mini_gpt.interaction.knowledge_gap import (
    ErrorCategory,
    KnowledgeGapRecord,
    KnowledgeGapTracker,
)
from adaptive_mini_gpt.metacognition.action_space import LearningAction


def test_learning_objective_validation() -> None:
    """Test LearningObjective validation, serialization, and roundtrip deserialization."""
    obj = LearningObjective.create(
        skill_or_concept="4-Digit Arithmetic Carry",
        evidence_of_repeated_failure=["Carry overflow failed on 4829 + 1782"],
        hypothesized_cause=HypothesizedCause.PROCEDURAL_GAP,
        prerequisite_knowledge=["single_digit_addition"],
        proposed_intervention=LearningAction.TRAIN_CANDIDATE,
        expected_benefit=0.85,
        estimated_resource_cost=1.00,
        independent_success_measurement="Accuracy >= 85%",
    )
    assert obj.is_valid()
    assert obj.objective_id is not None

    d = obj.to_dict()
    assert d["hypothesized_cause"] == "procedural_gap"
    assert d["proposed_intervention"] == "TRAIN_CANDIDATE"

    restored = LearningObjective.from_dict(d)
    assert restored.objective_id == obj.objective_id
    assert restored.skill_or_concept == obj.skill_or_concept
    assert restored.is_valid()


def test_evidence_features_extraction_and_uncertainty() -> None:
    """Test extraction of explicit evidence features on clear vs contradictory context."""
    # Contradictory context
    ctx_contra = "Fact A is true. However, fact A is completely false and contradicts reality."
    q_contra = "Is Fact A true?"
    feats_contra = extract_evidence_features(q_contra, ctx_contra, predictive_entropy=0.82)
    assert feats_contra.has_evidence == 1.0
    assert feats_contra.contradiction_score == 1.0
    assert feats_contra.predictive_entropy == 0.82

    tensor = feats_contra.to_tensor()
    assert tensor.shape == (8,)

    # Missing context on factual query
    q_fact = "When was the telescope invented?"
    feats_missing = extract_evidence_features(q_fact, None, predictive_entropy=0.91)
    assert feats_missing.has_evidence == 0.0
    assert feats_missing.evidence_absent_flag == 1.0


def test_enhanced_policy_forward() -> None:
    """Test EnhancedMetacognitivePolicy forward pass and shape consistency."""
    policy = EnhancedMetacognitivePolicy(d_model=64, d_features=8, d_hidden=32, num_actions=7)
    pooled = torch.randn(2, 64)
    ev_feats = torch.randn(2, 8)

    logits = policy(pooled, ev_feats)
    assert logits.shape == (2, 7)


def test_capability_gap_analyzer_proposes_valid_objectives() -> None:
    """Test that analyzer clusters errors and proposes structured LearningObjectives."""
    tracker = KnowledgeGapTracker()
    tracker.record_interaction(
        domain=DomainType.ARITHMETIC,
        question="Add 4819 + 3128",
        predicted_answer="7937",
        ground_truth="7947",
        confidence_score=0.88,
        is_verified=False,
        verification_status="FAIL",
    )
    tracker.record_interaction(
        domain=DomainType.ARITHMETIC,
        question="Add 9128 + 2381",
        predicted_answer="11409",
        ground_truth="11509",
        confidence_score=0.92,
        is_verified=False,
        verification_status="FAIL",
    )
    tracker.record_interaction(
        domain=DomainType.LANGUAGE,
        question="What is the capital of Mars?",
        predicted_answer="Olympus",
        ground_truth=None,
        confidence_score=0.95,
        is_verified=False,
        verification_status="UNSUPPORTED",
    )

    analyzer = CapabilityGapAnalyzer()
    objectives = analyzer.analyze_gaps(tracker)
    assert len(objectives) >= 2
    for obj in objectives:
        assert obj.is_valid()
        assert obj.proposed_intervention in list(LearningAction)


def test_objective_selection_policies() -> None:
    """Test comparison of Round-Robin, Largest-Gap, and Neural-Assisted selectors."""
    objs = [
        LearningObjective.create(
            skill_or_concept="Basic Math",
            evidence_of_repeated_failure=["fail 1"],
            hypothesized_cause=HypothesizedCause.PROCEDURAL_GAP,
            prerequisite_knowledge=[],
            proposed_intervention=LearningAction.PRACTICE,
            expected_benefit=0.40,
            estimated_resource_cost=0.25,
            independent_success_measurement="Math acc >= 80%",
        ),
        LearningObjective.create(
            skill_or_concept="Advanced Multi-Hop Logic",
            evidence_of_repeated_failure=["fail a", "fail b", "fail c"],
            hypothesized_cause=HypothesizedCause.PROCEDURAL_GAP,
            prerequisite_knowledge=["1_hop_modus_ponens"],
            proposed_intervention=LearningAction.TRAIN_CANDIDATE,
            expected_benefit=0.80,
            estimated_resource_cost=1.00,
            independent_success_measurement="Logic acc >= 75%",
        ),
    ]

    rr = RoundRobinObjectiveSelector()
    sel_rr1 = rr.select_objective(objs, completed_skills=set(), remaining_budget=2.0)
    sel_rr2 = rr.select_objective(objs, completed_skills=set(), remaining_budget=2.0)
    assert sel_rr1.objective_id != sel_rr2.objective_id

    lg = LargestGapObjectiveSelector()
    sel_lg = lg.select_objective(objs, completed_skills=set(), remaining_budget=2.0)
    # Picks the one with 3 evidence items
    assert sel_lg.skill_or_concept == "Advanced Multi-Hop Logic"

    na = NeuralAssistedObjectiveSelector()
    # When prereq is not met, Basic Math has higher ROI
    sel_na1 = na.select_objective(objs, completed_skills=set(), remaining_budget=2.0)
    assert sel_na1.skill_or_concept == "Basic Math"


def test_verified_curriculum_generator() -> None:
    """Test programmatic verified curriculum generation for arithmetic, logic, and factual QA."""
    gen = VerifiedCurriculumGenerator(seed=99)
    arith = gen.generate_arithmetic_practice(num_samples=5, digits=3)
    assert len(arith) == 5
    for item in arith:
        assert item.domain == DomainType.ARITHMETIC
        assert len(item.intermediate_steps) == 3

    logic = gen.generate_logic_practice(num_samples=3, num_hops=3)
    assert len(logic) == 3
    for item in logic:
        assert item.domain == DomainType.LOGIC
        assert "Rules:" in item.prompt

    factual = gen.generate_factual_practice(num_samples=4)
    assert len(factual) == 4
    for item in factual:
        assert item.domain == DomainType.LANGUAGE
        assert item.context is not None


def test_multi_episode_learner_simulation() -> None:
    """Test simulated multi-episode bounded self-learning with qualification gating."""
    selector = NeuralAssistedObjectiveSelector()
    learner = MultiEpisodeLearner("neural_assisted", selector, budget_per_episode=1.5, seed=42)

    rep1 = learner.run_episode(
        episode_idx=1,
        prompt="Add 5291 + 4812",
        domain=DomainType.ARITHMETIC,
        ground_truth="10103",
        initial_accuracy=0.20,
    )
    assert rep1.episode_index == 1
    assert rep1.compute_cost > 0.0

    rep2 = learner.run_episode(
        episode_idx=2,
        prompt="Context: A is True. But A is False. Question: Is A True?",
        domain=DomainType.LANGUAGE,
        ground_truth="insufficient information.",
        initial_accuracy=0.10,
        is_unanswerable=True,
    )
    assert rep2.episode_index == 2
    assert len(learner.history) == 2


def test_structural_evolution_gate() -> None:
    """Test external gate validation for future structural evolution proposals."""
    gate = StructuralEvolutionGate(max_parameter_growth_ratio=0.15, min_expected_gain=0.10)
    proposal_good = StructuralProposal.create(
        modification_type=StructuralModificationType.ADD_UPDATE_ADAPTER,
        target_module="block_3_adapter",
        evidence_of_need="Repeated procedural saturation on 8-digit addition",
        expected_gain=0.20,
        parameter_delta=73728,
        compute_overhead_ratio=0.07,
        current_checkpoint_hash="sha256_mock_hash",
    )
    approved, msg = gate.evaluate_proposal(proposal_good, base_parameter_count=1089280)
    assert approved is True

    proposal_bad = StructuralProposal.create(
        modification_type=StructuralModificationType.EXPAND_DIMENSIONS,
        target_module="backbone_wte",
        evidence_of_need="Speculative capacity increase",
        expected_gain=0.05,  # Below 0.10
        parameter_delta=500000,  # Exceeds 15%
        compute_overhead_ratio=0.50,
        current_checkpoint_hash="sha256_mock_hash",
    )
    approved_bad, msg_bad = gate.evaluate_proposal(proposal_bad, base_parameter_count=1089280)
    assert approved_bad is False
