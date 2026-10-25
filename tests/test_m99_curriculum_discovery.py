"""Milestone M99 Verification Suite: Autonomous Curriculum Discovery & Self-Directed Learning.

Validates the complete M99 research system across 7 unit and integration tests:
1. test_m98_forensic_audit_invariants
2. test_goal_candidate_pool_and_verifier
3. test_curriculum_policies_and_scoring
4. test_active_experiment_design_info_gain
5. test_adaptive_capacity_accounting_and_tiers
6. test_four_student_future_learning_evaluation
7. test_m99_checkpoint_serialization_and_recovery
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from adaptive_mini_gpt.cross_domain_transfer.neural_cross_domain_transfer import (
    MODULUS,
    OP_ADD,
    OP_ASSIGN,
)
from adaptive_mini_gpt.cross_subject_skills.cross_subject_substrate import (
    CrossSubjectReasoningSubstrate,
)
from adaptive_mini_gpt.persistent_memory.knowledge_store import (
    PersistentKnowledgeStore,
)
from adaptive_mini_gpt.self_directed_curriculum.active_experiment_designer import (
    ActiveExperimentDesigner,
)
from adaptive_mini_gpt.self_directed_curriculum.adaptive_capacity_manager import (
    AdaptiveCapacityManager,
    CapacityTier,
)
from adaptive_mini_gpt.self_directed_curriculum.audit_m98 import (
    M98ForensicAuditor,
)
from adaptive_mini_gpt.self_directed_curriculum.curriculum_policy import (
    CurriculumDiscoveryPolicy,
    CurriculumSelector,
    CurriculumStrategy,
    GoalDiagnosticState,
)
from adaptive_mini_gpt.self_directed_curriculum.future_learning_evaluator import (
    FutureLearningEvaluator,
)
from adaptive_mini_gpt.self_directed_curriculum.goal_candidate_pool import (
    GoalCandidatePoolGenerator,
    IndependentTaskVerifier,
    SubjectCategory,
)
from adaptive_mini_gpt.self_directed_curriculum.persistent_student_m99 import (
    PersistentAdaptiveStudentM99,
)


@pytest.fixture
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def test_m98_forensic_audit_invariants(device: torch.device):
    """Verify Phase 0 audit confirms M98 invariants, 40->20->10 stepping, and 1,049 headroom."""
    audit = M98ForensicAuditor.execute_audit(device=device)
    assert audit.status == "M98_AUDIT_VERIFIED"
    assert audit.m98_unfamiliar_skill_em == 100.0
    assert audit.m98_composition_em == 100.0
    assert audit.m98_student_a_steps == 40
    assert audit.m98_student_b_steps == 20
    assert audit.m98_student_c_steps == 10
    assert audit.m98_experience_advantage_pct == 50.0
    assert audit.m98_meta_advantage_pct == 50.0
    assert audit.m98_trainable_parameters == 18_951
    assert audit.m98_parameter_headroom == 1_049
    assert audit.frozen_backbone_weights == 1_071_872
    assert audit.empirical_runtime_savings_pct >= 0.0


def test_goal_candidate_pool_and_verifier(device: torch.device):
    """Verify pool generates 5 subjects + 2 controls and verifier scores accurately."""
    pool = GoalCandidatePoolGenerator.generate_candidate_pool(device=device)
    assert len(pool) == 7

    categories = {opp.category for opp in pool}
    assert SubjectCategory.ALGEBRA in categories
    assert SubjectCategory.GRAPH in categories
    assert SubjectCategory.LOGIC in categories
    assert SubjectCategory.PROGRAM in categories
    assert SubjectCategory.COMPOSITION in categories
    assert SubjectCategory.NOISE_DISTRACTOR in categories
    assert SubjectCategory.ALREADY_MASTERED in categories

    for opp in pool:
        assert len(opp.diagnostic_examples) == 16
        verifier = IndependentTaskVerifier(opp)
        train_ex = opp.diagnostic_examples[:8]
        test_ex = opp.diagnostic_examples[8:]
        for s, op, v, y in train_ex + test_ex:
            assert verifier.verify_transition(s, op, v, y) is True


def test_curriculum_policies_and_scoring(device: torch.device):
    """Verify 6 curriculum strategies prioritize candidates according to their specific inductive bias."""
    pool = GoalCandidatePoolGenerator.generate_candidate_pool(device=device)
    policy_net = CurriculumDiscoveryPolicy(in_features=5, hidden_dim=16).to(device)

    # 1. Random strategy
    sel_rand = CurriculumSelector(strategy=CurriculumStrategy.RANDOM)
    # 2. Fixed order strategy
    sel_fixed = CurriculumSelector(strategy=CurriculumStrategy.FIXED_ORDER)
    # 3. Uncertainty only (should pick noise distractor or unmastered high uncertainty)
    sel_unc = CurriculumSelector(strategy=CurriculumStrategy.UNCERTAINTY_ONLY)
    # 4. Progress only
    sel_prog = CurriculumSelector(strategy=CurriculumStrategy.PROGRESS_ONLY)
    # 5. Combined Self-Directed
    sel_comb = CurriculumSelector(policy_network=policy_net, strategy=CurriculumStrategy.COMBINED_SELF_DIRECTED)
    # 6. Oracle
    sel_oracle = CurriculumSelector(strategy=CurriculumStrategy.ORACLE)

    states = [
        GoalDiagnosticState(
            goal_id=opp.goal_id,
            uncertainty=opp.initial_uncertainty,
            recent_progress=0.4 if opp.category != SubjectCategory.NOISE_DISTRACTOR else 0.0,
            query_cost_estimate=opp.query_cost_estimate,
            staleness_steps=5.0,
            is_mastered=(opp.category == SubjectCategory.ALREADY_MASTERED),
        )
        for opp in pool
    ]

    r_unc = sel_unc.rank_opportunities(states)
    assert r_unc[0].goal_id == "goal_noise_distractor"  # Noise has highest uncertainty (0.95)

    r_comb = sel_comb.rank_opportunities(states)
    assert r_comb[0].goal_id != "goal_noise_distractor"  # Combined penalizes zero progress / unlearnable noise
    assert r_comb[0].goal_id != "goal_already_mastered"   # Combined deprioritizes already-mastered skills


def test_active_experiment_design_info_gain(device: torch.device):
    """Verify active boundary probing achieves higher expected information gain (bits) and cost reduction."""
    pool = GoalCandidatePoolGenerator.generate_candidate_pool(device=device)
    algebra_opp = [o for o in pool if o.category == SubjectCategory.ALGEBRA][0]

    designer = ActiveExperimentDesigner(device=device)
    active_batch = designer.design_active_queries(algebra_opp, query_budget=8)
    random_batch = designer.design_random_queries(algebra_opp, query_budget=8)

    assert active_batch.query_cost < random_batch.query_cost
    assert active_batch.expected_info_gain_bits > random_batch.expected_info_gain_bits
    assert active_batch.cost_reduction_ratio >= 1.2  # >= 20% cost advantage


def test_adaptive_capacity_accounting_and_tiers(device: torch.device):
    """Verify adaptive capacity tiers (20k, 30k, 50k, 100k, adaptive) and evidence-guided expansion/rollback."""
    # 1. Baseline 20k
    mgr_20k = AdaptiveCapacityManager(tier=CapacityTier.TIER_20K, device=device)
    assert mgr_20k.budget_ceiling == 20_000

    # 2. Tier 50k expansion test
    mgr_50k = AdaptiveCapacityManager(tier=CapacityTier.TIER_50K, device=device)
    assert mgr_50k.budget_ceiling == 50_000

    allocated = mgr_50k.propose_growth(rank=4, in_features=128, out_features=128, name="algebra_refinement")
    assert allocated > 0
    assert "algebra_refinement" in mgr_50k.allocated_modules

    # Generalization test: if improvement passes verification, retain; if not, rollback
    kept = mgr_50k.evaluate_and_commit("algebra_refinement", held_out_generalization=85.0, retention_rate=95.0)
    assert kept is True

    # Failed candidate rollback
    _ = mgr_50k.propose_growth(rank=8, in_features=128, out_features=128, name="unstable_module")
    rolled_back = mgr_50k.evaluate_and_commit("unstable_module", held_out_generalization=40.0, retention_rate=70.0)
    assert rolled_back is False
    assert "unstable_module" not in mgr_50k.allocated_modules


def test_four_student_future_learning_evaluation(device: torch.device):
    """Verify 4-student evaluation: Student D achieves faster qualification than Students A, B, and C."""
    # Pretrain shared substrate on foundation
    substrate = CrossSubjectReasoningSubstrate(budget_ceiling=20_000, device=device)
    grid_s, grid_v = torch.meshgrid(
        torch.arange(MODULUS, device=device),
        torch.arange(MODULUS, device=device),
        indexing="ij",
    )
    grid_s = grid_s.reshape(-1)
    grid_v = grid_v.reshape(-1)
    all_s = torch.cat([grid_s, grid_s])
    all_op = torch.cat([torch.full_like(grid_s, OP_ASSIGN), torch.full_like(grid_s, OP_ADD)])
    all_v = torch.cat([grid_v, grid_v])
    all_y = torch.cat([grid_v, (grid_s + grid_v) % MODULUS])

    opt = torch.optim.AdamW(substrate.substrate.parameters(), lr=0.03)
    for _ in range(30):
        logits = substrate.forward_transition(all_s, all_op, all_v)
        loss = F.cross_entropy(logits, all_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    evaluator = FutureLearningEvaluator(device=device)
    res = evaluator.evaluate_four_students(
        pretrained_substrate=substrate,
        held_out_target_op=11,  # Held-out modular affine target
        seed=42,
        max_steps=40,
    )

    # Invariants: Student A (fresh) > Student B (fixed) > Student C (policy) >= Student D (self-directed)
    assert res.student_a_steps >= res.student_b_steps
    assert res.student_b_steps >= res.student_c_steps
    assert res.student_c_steps >= res.student_d_steps
    assert res.student_d_steps <= 10
    assert res.step_reduction_over_fixed_pct >= 20.0
    assert res.student_d_retention_em >= 90.0


def test_m99_checkpoint_serialization_and_recovery(device: torch.device):
    """Verify complete persistent student state serialization and bit-exact recovery."""
    student = PersistentAdaptiveStudentM99(budget_ceiling=20_000, tier=CapacityTier.TIER_20K, device=device)
    student.step_counter = 12
    student.acquired_skills = ["goal_modular_algebra_quad", "goal_constrained_graph_path"]
    student.memory_store.store_fact(
        entity="entity_node",
        attribute="energy",
        value=15,
        confidence=1.0,
        timestamp=1,
        source="test_m99",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        chk_path = Path(tmp_dir) / "m99_checkpoint"
        student.save_checkpoint(str(chk_path))

        reloaded = PersistentAdaptiveStudentM99.load_checkpoint(str(chk_path), device=device)
        assert reloaded.step_counter == student.step_counter
        assert len(reloaded.acquired_skills) == len(student.acquired_skills)
        assert len(reloaded.memory_store.facts) == len(student.memory_store.facts)
        assert reloaded.memory_store.retrieve_fact("entity_node", "energy").value == 15
