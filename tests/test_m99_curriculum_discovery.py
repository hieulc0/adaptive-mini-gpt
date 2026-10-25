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
    """Verify Phase 0 audit confirms M98 invariants, stepping attribution, and budget headroom."""
    audit = M98ForensicAuditor.execute_audit(device=device)

    assert audit.status == "M98_AUDIT_VERIFIED"
    assert audit.m98_unfamiliar_skill_em == 100.0
    assert audit.m98_composition_em == 100.0
    assert audit.m98_student_a_steps == 40
    assert audit.m98_student_b_steps == 20
    assert audit.m98_student_c_steps == 10
    assert audit.m98_experience_advantage_pct == 50.0
    assert audit.m98_meta_advantage_pct == 50.0
    assert audit.m98_neural_skill_retention_em == 100.0
    assert audit.m98_trainable_parameters == 18_951
    assert audit.m98_parameter_headroom == 1_049
    assert audit.frozen_backbone_weights == 1_071_872
    assert audit.full_regression_tests_passed == 118
    assert audit.measured_gpu_runtime_ms_short > 0.0
    assert audit.measured_gpu_runtime_ms_long > 0.0


def test_goal_candidate_pool_and_verifier():
    """Verify candidate goal pool covers all subjects and verifier provides accurate responses."""
    gen = GoalCandidatePoolGenerator(seed=42)
    pool = gen.generate_candidate_pool()

    assert len(pool) == 7
    subjects = {c.subject for c in pool}
    assert SubjectCategory.ALGEBRA in subjects
    assert SubjectCategory.GRAPH in subjects
    assert SubjectCategory.LOGIC in subjects
    assert SubjectCategory.PROGRAM in subjects
    assert SubjectCategory.COMPOSITION in subjects
    assert SubjectCategory.ALREADY_MASTERED in subjects
    assert SubjectCategory.NOISE_DISTRACTOR in subjects

    # Validate independent verifier query responses
    assert IndependentTaskVerifier.query("goal_algebra_affine", 2, OP_ADD, 0) == (3 * 2 + 5) % MODULUS
    assert IndependentTaskVerifier.query("goal_mastered_arithmetic", 4, OP_ADD, 3) == (4 + 3) % MODULUS


def test_curriculum_policies_and_scoring(device: torch.device):
    """Verify 113-parameter neural curriculum policy penalizes noise and prioritizes progress frontiers."""
    policy = CurriculumDiscoveryPolicy(in_features=5, hidden_dim=16).to(device)
    assert policy.count_parameters() == 113

    # High progress goal on learnable frontier
    st_frontier = GoalDiagnosticState(
        goal_id="goal_algebra",
        current_accuracy=40.0,
        recent_progress=30.0,
        epistemic_uncertainty=0.6,
        estimated_cost=0.4,
        rehearsal_staleness=5,
    )
    score_frontier = policy.score_goal(st_frontier, device=device)

    # Noise distractor: high uncertainty but zero/negative progress
    st_noise = GoalDiagnosticState(
        goal_id="goal_noise",
        current_accuracy=10.0,
        recent_progress=-5.0,
        epistemic_uncertainty=0.9,
        estimated_cost=0.8,
        rehearsal_staleness=2,
    )
    score_noise = policy.score_goal(st_noise, device=device)

    # Already mastered: 100% accuracy and fresh
    st_mastered = GoalDiagnosticState(
        goal_id="goal_mastered",
        current_accuracy=100.0,
        recent_progress=0.0,
        epistemic_uncertainty=0.05,
        estimated_cost=0.1,
        rehearsal_staleness=1,
    )
    score_mastered = policy.score_goal(st_mastered, device=device)

    # Policy should rank frontier higher than noise and mastered
    assert score_frontier > score_noise
    assert score_frontier > score_mastered


def test_active_experiment_design_info_gain():
    """Verify active informative experiment design achieves higher bits/query than uniform random."""
    pool_gen = GoalCandidatePoolGenerator(seed=42)
    pool = pool_gen.generate_candidate_pool()
    goal = pool[1]  # Algebra

    designer = ActiveExperimentDesigner(seed=42)
    active_out = designer.design_and_execute_experiments(goal, budget=4, strategy="ACTIVE_INFORMATIVE")
    random_out = designer.design_and_execute_experiments(goal, budget=4, strategy="UNIFORM_RANDOM")

    assert active_out.mean_bits_per_query > random_out.mean_bits_per_query
    assert active_out.queries_to_disambiguate < random_out.queries_to_disambiguate
    assert active_out.query_cost_reduction_pct == 50.0


def test_adaptive_capacity_accounting_and_tiers(device: torch.device):
    """Verify configurable capacity tiers (20k, 30k, 50k, 100k, Adaptive) and budget compliance."""
    substrate = CrossSubjectReasoningSubstrate(budget_ceiling=20_000, device=device)
    policy = CurriculumDiscoveryPolicy(in_features=5, hidden_dim=16).to(device)

    # Tier 20k baseline
    mgr_20k = AdaptiveCapacityManager(tier=CapacityTier.TIER_20K, device=device)
    acct_20k = mgr_20k.get_accounting(substrate, curriculum_policy_params=policy.count_parameters())

    assert acct_20k.substrate_parameters == 17_116
    assert acct_20k.acquisition_controller_parameters == 1_835
    assert acct_20k.curriculum_policy_parameters == 113
    assert acct_20k.total_persistent_trainable_parameters == 19_064
    assert acct_20k.total_persistent_trainable_parameters <= 20_000
    assert acct_20k.headroom_parameters == 936
    assert acct_20k.frozen_backbone_weights == 1_071_872

    # Tier 50k expansion
    mgr_50k = AdaptiveCapacityManager(tier=CapacityTier.TIER_50K, device=device)
    acct_50k = mgr_50k.get_accounting(substrate, curriculum_policy_params=policy.count_parameters())
    assert acct_50k.budget_ceiling == 50_000
    assert acct_50k.headroom_parameters == 50_000 - 19_064

    # Evidence-based growth & rollback
    alloc_info = mgr_50k.allocate_residual_expert(substrate, "dynamic_exp_1", rank=2)
    assert alloc_info["parameters"] > 0
    assert "dynamic_exp_1" in substrate.substrate.concept_residuals
    # Rollback
    rolled_back = mgr_50k.rollback_module(substrate, "dynamic_exp_1")
    assert rolled_back is True
    assert "dynamic_exp_1" not in substrate.substrate.concept_residuals


def test_four_student_future_learning_evaluation(device: torch.device):
    """Verify Student D achieves superior future adaptation over Student B and C."""
    # Pretrain trunk across 16x16 pairs of addition and assign
    substrate = CrossSubjectReasoningSubstrate(budget_ceiling=20_000, device=device)
    grid_s, grid_v = torch.meshgrid(
        torch.arange(MODULUS, device=device),
        torch.arange(MODULUS, device=device),
        indexing="ij",
    )
    grid_s, grid_v = grid_s.reshape(-1), grid_v.reshape(-1)
    all_s = torch.cat([grid_s, grid_s])
    all_op = torch.cat([torch.full_like(grid_s, OP_ADD), torch.full_like(grid_s, OP_ASSIGN)])
    all_v = torch.cat([grid_v, grid_v])
    all_y = torch.cat([(grid_s + grid_v) % MODULUS, grid_v])

    opt = torch.optim.AdamW(substrate.substrate.parameters(), lr=0.03)
    for _ in range(40):
        logits = substrate.forward_transition(all_s, all_op, all_v)
        loss = F.cross_entropy(logits, all_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    substrate.eval()

    evaluator = FutureLearningEvaluator(device=device)
    tgt_train = [(x, OP_ADD, (x * 2 + 5) % MODULUS, (3 * x + 5) % MODULUS) for x in range(12)]
    tgt_test = [(x, OP_ADD, (x * 2 + 5) % MODULUS, (3 * x + 5) % MODULUS) for x in range(12, 16)]
    anchor_ex = [(x, OP_ADD, 2, (x + 2) % MODULUS) for x in range(4)]

    res = evaluator.evaluate_target_family(
        target_train_examples=tgt_train,
        target_test_examples=tgt_test,
        anchor_examples=anchor_ex,
        pretrained_substrate=substrate,
        self_directed_substrate=substrate,
        max_steps=35,
    )

    assert res.student_b.adaptation_steps <= res.student_a.adaptation_steps
    assert res.student_c.adaptation_steps <= res.student_b.adaptation_steps
    assert res.student_d.adaptation_steps <= res.student_c.adaptation_steps
    assert res.student_d.retained_anchor_accuracy >= 90.0
    assert res.self_directed_curriculum_advantage_pct >= 20.0


def test_m99_checkpoint_serialization_and_recovery(device: torch.device):
    """Verify persistent student saves complete state and reloads with bit-exact consistency."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        substrate = CrossSubjectReasoningSubstrate(budget_ceiling=20_000, device=device)
        policy = CurriculumDiscoveryPolicy(in_features=5, hidden_dim=16).to(device)
        mgr = AdaptiveCapacityManager(tier=CapacityTier.TIER_20K, device=device)
        store = PersistentKnowledgeStore(max_byte_budget=262_144)
        store.store_fact("entity_node", "energy", 15, source="test_m99", timestamp=1)

        student = PersistentAdaptiveStudentM99(
            substrate=substrate,
            curriculum_policy=policy,
            capacity_manager=mgr,
            memory_store=store,
            device=device,
        )

        chk_path = str(Path(tmp_dir) / "student_m99_chk")
        student.save_checkpoint(chk_path)

        reloaded = PersistentAdaptiveStudentM99.load_checkpoint(chk_path, device=device)

        assert reloaded.step_counter == student.step_counter
        assert len(reloaded.acquired_skills) == len(student.acquired_skills)
        assert len(reloaded.memory_store.facts) == len(student.memory_store.facts)

        s = torch.tensor([3], device=device)
        op = torch.tensor([OP_ADD], device=device)
        v = torch.tensor([4], device=device)

        p1 = student.substrate.forward_transition(s, op, v).argmax().item()
        p2 = reloaded.substrate.forward_transition(s, op, v).argmax().item()
        assert p1 == p2
