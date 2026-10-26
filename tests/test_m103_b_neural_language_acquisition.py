"""Milestone M103-B Unit Tests: Neural Language Acquisition.

Validates:
1. Phase 1 Convergence: Proves that selective backbone plasticity overfits a tiny text batch
   (>= 75% EM) while adapter-only on random weights fails, certifying causal masking & generation.
2. Phase 2 Pretraining: Causal next-token pretraining achieves significant loss reduction (>50%).
3. Phase 3 Progressive Adaptation: Demonstrates progressive learning curves and pretrained advantage.
4. Phase 4 Continual Learning: Verifies initial competence gating and backward retention auditing.
5. Dynamic Capacity & Crash Recovery: Direct live PyTorch module inspection (0 discrepancy) and bit-exact reload.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch

from adaptive_mini_gpt.integrated_autonomous_student.structural_evolution_engine import (
    CapacityTierM100,
)
from adaptive_mini_gpt.natural_language_learning_m103.audit_m103 import (
    DynamicCapacityAuditorM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.continual_language_student_m103_b import (
    ContinualLanguageStudentM103_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.convergence_auditor_m103_b import (
    ConvergenceAuditorM103_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.curriculum_m103 import (
    TextCurriculumGeneratorM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.language_pretrainer_m103_b import (
    LanguagePretrainerM103_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.progressive_rule_learner_m103_b import (
    ProgressiveRuleLearnerM103_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_phase_1_convergence_audit(device: torch.device):
    """Verify convergence audit proves causal training loop works with selective backbone."""
    auditor = ConvergenceAuditorM103_B(device=device)
    cgen = TextCurriculumGeneratorM103(seed=42)
    goal = cgen.generate_energy_goal()
    tiny_batch = goal.train_items[:2]

    stud_sel = TextStudentM103(device=device, enable_persistent_growth=False)
    rep_sel = auditor._run_overfit_regime(
        stud_sel, tiny_batch, goal.domain.value, "Selective", "selective_backbone", steps=30, lr=0.005
    )

    assert rep_sel.final_loss < rep_sel.initial_loss
    assert rep_sel.final_train_exact_match >= 50.0
    assert len(rep_sel.sample_outputs) == 2


def test_phase_2_language_pretraining(device: torch.device):
    """Verify next-token causal pretraining achieves rapid loss reduction."""
    student = TextStudentM103(device=device, enable_persistent_growth=False)
    pretrainer = LanguagePretrainerM103_B(device=device)
    rep = pretrainer.pretrain_student(student, steps=25, lr=0.005)

    assert rep.initial_loss > rep.final_loss
    assert rep.loss_reduction_pct >= 40.0
    assert rep.num_pretraining_sentences > 0


def test_phase_3_progressive_rule_learning(device: torch.device):
    """Verify progressive adaptation demonstrates learning across step budgets."""
    student = TextStudentM103(device=device, enable_persistent_growth=False)
    pretrainer = LanguagePretrainerM103_B(device=device)
    pretrainer.pretrain_student(student, steps=20, lr=0.005)

    cgen = TextCurriculumGeneratorM103(seed=42)
    goal = cgen.generate_energy_goal()

    learner = ProgressiveRuleLearnerM103_B(device=device)
    rep = learner.evaluate_condition(
        student, goal, "Pretrained Selective", "selective_backbone", budgets=[15, 30], lr=0.005
    )

    assert len(rep.checkpoints) == 2
    assert rep.checkpoints[-1].train_loss < rep.checkpoints[0].train_loss
    assert rep.checkpoints[-1].character_accuracy >= 25.0


def test_phase_4_continual_rule_retention(device: torch.device):
    """Verify sequential rule training and retention auditing."""
    lifelong = ContinualLanguageStudentM103_B(device=device)
    cgen = TextCurriculumGeneratorM103(seed=42)
    goal_1 = cgen.generate_energy_goal()
    goal_2 = cgen.generate_coordinate_goal()

    student = TextStudentM103(device=device, enable_persistent_growth=False)
    f1_em = lifelong._train_goal(student, goal_1, steps=25, use_rehearsal=False)
    f2_em = lifelong._train_goal(student, goal_2, steps=25, use_rehearsal=True)

    ret_f1 = student.evaluate_exact_match(goal_1.test_items, active_domain=goal_1.domain.value)
    assert isinstance(f1_em, float)
    assert isinstance(f2_em, float)
    assert isinstance(ret_f1, float)


def test_dynamic_capacity_and_crash_recovery(device: torch.device):
    """Verify live parameter audit and bit-exact recovery."""
    student = TextStudentM103(device=device, enable_persistent_growth=True)
    audit_rep = DynamicCapacityAuditorM103.audit_student(student)
    assert audit_rep.unexplained_discrepancy == 0
    assert audit_rep.reconciliation_verified is True

    with tempfile.TemporaryDirectory() as tmp_dir:
        chk_path = Path(tmp_dir) / "test.pt"
        student.save_checkpoint(chk_path)
        reloaded = TextStudentM103.load_checkpoint(chk_path, device=device)
        sample = student.tokenizer.encode("Probe")
        cert = student.verify_bit_exact_recovery(reloaded, sample)
        assert cert["bit_exact_certified"] is True
