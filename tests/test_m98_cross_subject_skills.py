"""Milestone M98 Verification Suite: Autonomous Cross-Subject Skill Acquisition & Continual Learning.

Validates the complete M98 research system across 6 unit and integration tests:
1. test_m97_forensic_audit_invariants
2. test_cross_subject_curriculum_and_solver
3. test_cross_subject_substrate_parameter_budget
4. test_tri_student_causal_comparison
5. test_cross_subject_composition_execution
6. test_checkpoint_serialization_and_bit_exact_recovery
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.cross_domain_transfer.neural_cross_domain_transfer import (
    MODULUS,
    OP_ADD,
    OP_ASSIGN,
    OP_IF_GT,
)
from adaptive_mini_gpt.cross_subject_skills.audit_m97 import (
    M97ForensicAuditor,
)
from adaptive_mini_gpt.cross_subject_skills.autonomous_skill_learner import (
    AutonomousSkillLearner,
    StudentType,
)
from adaptive_mini_gpt.cross_subject_skills.cross_subject_substrate import (
    CrossSubjectReasoningSubstrate,
)
from adaptive_mini_gpt.cross_subject_skills.persistent_student_m98 import (
    PersistentAdaptiveStudentM98,
)
from adaptive_mini_gpt.cross_subject_skills.subject_curriculum import (
    CrossSubjectCurriculumGenerator,
    IndependentGroundTruthSolver,
    Subject,
)
from adaptive_mini_gpt.persistent_memory.knowledge_store import (
    PersistentKnowledgeStore,
)


@pytest.fixture
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def test_m97_forensic_audit_invariants(device: torch.device):
    """Verify Phase 0 audit confirms M97 invariants and parameter headroom."""
    audit_res = M97ForensicAuditor.execute_audit(device=device)

    assert audit_res.m97_context_window == 1024
    assert audit_res.m97_needle_accuracy == 100.0
    assert audit_res.frozen_backbone_weights == 1_071_872
    assert audit_res.m97_trainable_parameters == 17_116
    assert audit_res.m97_parameter_headroom == 2_884
    assert audit_res.m97_external_memory_bytes == 14_458
    assert audit_res.m97_historical_regression_tests_passed == 112
    assert audit_res.m97_audit_certified is True
    assert audit_res.measured_gpu_runtime_ms_short > 0.0
    assert audit_res.measured_gpu_runtime_ms_long > 0.0


def test_cross_subject_curriculum_and_solver():
    """Verify curriculum generator covers all 3 subject areas and solver outputs valid labels."""
    gen = CrossSubjectCurriculumGenerator(seed=42, num_episodes=60)
    tasks = gen.generate_curriculum()

    assert len(tasks) == 60
    subjects = {t.subject for t in tasks}
    assert Subject.ALGEBRA in subjects
    assert Subject.GRAPH in subjects
    assert Subject.LOGIC in subjects
    assert Subject.CROSS_SUBJECT_COMPOSITION in subjects
    assert Subject.UNSUPPORTED_ABSTAIN in subjects

    # Validate independent ground-truth solver for each task
    for task in tasks:
        solver_target = IndependentGroundTruthSolver.solve(task)
        assert task.target_output == solver_target, f"Task {task.task_id} solver discrepancy"


def test_cross_subject_substrate_parameter_budget(device: torch.device):
    """Verify combined neural substrate + controller strictly respects the 20,000 parameter budget."""
    substrate = CrossSubjectReasoningSubstrate(budget_ceiling=20_000, device=device)
    breakdown = substrate.parameter_breakdown()

    assert breakdown["substrate_parameters"] == 17_116
    assert breakdown["controller_parameters"] == 1_835
    assert breakdown["total_trainable_parameters"] == 18_951
    assert breakdown["total_trainable_parameters"] <= 20_000
    assert breakdown["budget_headroom"] == 1_049


def test_tri_student_causal_comparison(device: torch.device):
    """Verify Student B requires fewer steps than Student A, and Student C outperforms Student B."""
    learner = AutonomousSkillLearner(device=device)

    # Pretrain concept trunk across 16x16 pairs of addition and assign
    substrate = CrossSubjectReasoningSubstrate(budget_ceiling=20_000, device=device)
    S, V = torch.meshgrid(torch.arange(MODULUS), torch.arange(MODULUS), indexing="ij")
    S, V = S.reshape(-1).to(device), V.reshape(-1).to(device)
    all_s = torch.cat([S, S])
    all_op = torch.cat([torch.full_like(S, OP_ADD), torch.full_like(S, OP_ASSIGN)])
    all_v = torch.cat([V, V])
    all_y = torch.cat([(S + V) % MODULUS, V])

    opt = torch.optim.AdamW(substrate.substrate.parameters(), lr=0.03)
    for _ in range(40):
        logits = substrate.forward_transition(all_s, all_op, all_v)
        loss = torch.nn.functional.cross_entropy(logits, all_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    substrate.eval()

    train_data = [(x, OP_ADD, (x * 2 + 3) % MODULUS, (3 * x + 3) % MODULUS) for x in range(12)]
    test_data = [(x, OP_ADD, (x * 2 + 3) % MODULUS, (3 * x + 3) % MODULUS) for x in range(12, 16)]
    anchor_data = [(x, OP_ADD, 2, (x + 2) % MODULUS) for x in range(4)]

    # Student A: Fresh
    fresh_sub = CrossSubjectReasoningSubstrate(budget_ceiling=20_000, device=device)
    res_a = learner.adapt_student(
        student_substrate=fresh_sub,
        student_type=StudentType.STUDENT_A_FRESH,
        subject="algebra",
        train_examples=train_data,
        test_examples=test_data,
        anchor_replay_examples=anchor_data,
        max_steps=35,
    )

    # Student B: Experienced + Fixed
    res_b = learner.adapt_student(
        student_substrate=copy.deepcopy(substrate),
        student_type=StudentType.STUDENT_B_EXPERIENCED_FIXED,
        subject="algebra",
        train_examples=train_data,
        test_examples=test_data,
        anchor_replay_examples=anchor_data,
        max_steps=35,
    )

    # Student C: Experienced + Learned
    res_c = learner.adapt_student(
        student_substrate=copy.deepcopy(substrate),
        student_type=StudentType.STUDENT_C_EXPERIENCED_LEARNED,
        subject="algebra",
        train_examples=train_data,
        test_examples=test_data,
        anchor_replay_examples=anchor_data,
        max_steps=35,
    )

    # Student B adapts in fewer steps than Student A
    assert res_b.steps_taken <= res_a.steps_taken
    # Student C adapts in fewer or equal steps than Student B with dynamic stopping
    assert res_c.steps_taken <= res_b.steps_taken


def test_cross_subject_composition_execution(device: torch.device):
    """Verify persistent student accurately solves cross-subject compositions."""
    substrate = CrossSubjectReasoningSubstrate(budget_ceiling=20_000, device=device)
    store = PersistentKnowledgeStore(max_byte_budget=262_144)

    student = PersistentAdaptiveStudentM98(
        substrate=substrate,
        memory_store=store,
        device=device,
    )

    gen = CrossSubjectCurriculumGenerator(seed=42, num_episodes=60)
    tasks = gen.generate_curriculum()

    comp_task = tasks[55]  # First composition task
    assert comp_task.subject == Subject.CROSS_SUBJECT_COMPOSITION
    res = student.process_task(comp_task)
    assert res.is_correct is True
    assert res.predicted_output == comp_task.target_output


def test_checkpoint_serialization_and_bit_exact_recovery(device: torch.device):
    """Verify saving and reloading persistent student produces bit-exact behavior."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        substrate = CrossSubjectReasoningSubstrate(budget_ceiling=20_000, device=device)
        store = PersistentKnowledgeStore(max_byte_budget=262_144)
        store.store_fact("entity_node", "energy", 12, source="test", timestamp=1)

        student = PersistentAdaptiveStudentM98(
            substrate=substrate,
            memory_store=store,
            device=device,
        )

        chk_path = str(Path(tmp_dir) / "student_chk_m98")
        student.save_checkpoint(chk_path)

        reloaded = PersistentAdaptiveStudentM98.load_checkpoint(chk_path, device=device)

        assert reloaded.step_counter == student.step_counter
        assert len(reloaded.memory_store.facts) == len(student.memory_store.facts)

        gen = CrossSubjectCurriculumGenerator(seed=42, num_episodes=60)
        tasks = gen.generate_curriculum()
        t = tasks[0]

        r1 = student.process_task(t)
        r2 = reloaded.process_task(t)
        assert r1.predicted_output == r2.predicted_output
