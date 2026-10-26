"""Milestone M103 Unit Tests: Natural-Language Learning Foundation.

Validates:
1. ByteTextTokenizerM103: Deterministic byte-level encoding, decoding, prompt formatting, pad batching.
2. Curriculum Partition Disjointness: Strictly 0% token overlap between train, val, test, and novel parameterization.
3. Routing Diagnosis: Empirical isolation of M102.2 router collapse and verification of rehearsal supervision remediation.
4. Dynamic Capacity Audit: Live PyTorch module inspection with 0 discrepancy under Permanent Architecture Principle.
5. TextStudent Autoregressive LM: Causal attention, prompt masking, and token generation.
6. Bit-Exact Crash Recovery: Continuous logit match (< 1e-6) and exact argmax across checkpoint save/reload.
7. Multi-Mode Comparator: Valid execution across Neural, In-Context, and Baseline learning modes.
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
from adaptive_mini_gpt.natural_language_learning_m103.comparator_m103 import (
    NaturalLanguageComparatorM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.curriculum_m103 import (
    TextCurriculumGeneratorM103,
    TextRuleDomainM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.routing_diagnostic_m103 import (
    RoutingDiagnosticM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_byte_tokenizer_encoding_decoding_roundtrip():
    """Verify byte-level tokenizer encoding and decoding roundtrip accurately."""
    tok = ByteTextTokenizerM103()
    sample = "In System-Alpha, a blue orb gains 2 energy units."

    encoded = tok.encode(sample, add_bos=True, add_eos=True)
    assert encoded[0] == tok.BOS_ID
    assert encoded[-1] == tok.EOS_ID

    decoded = tok.decode(encoded, skip_special_tokens=True)
    assert decoded == sample

    # QA formatting
    full_toks, p_len = tok.format_qa_sequence(
        question="What is final energy?",
        answer="6 energy",
        explanation="Blue gains 2.",
    )
    assert p_len > 0
    assert tok.RULE_ID in full_toks
    assert tok.QUESTION_ID in full_toks
    assert tok.ANSWER_ID in full_toks
    assert full_toks[-1] == tok.EOS_ID


def test_curriculum_disjointness_zero_leakage():
    """Verify strictly 0% token leakage between train, val, test, and novel partitions."""
    gen = TextCurriculumGeneratorM103(seed=42)
    goals = gen.generate_full_curriculum()

    for g in goals:
        tr_toks = {tuple(it.full_tokens) for it in g.train_items}
        val_toks = {tuple(it.full_tokens) for it in g.val_items}
        test_toks = {tuple(it.full_tokens) for it in g.test_items}
        novel_toks = {tuple(it.full_tokens) for it in g.novel_items}

        assert len(tr_toks.intersection(val_toks)) == 0, f"Train and Val overlap in {g.domain}"
        assert len(tr_toks.intersection(test_toks)) == 0, f"Train and Test overlap in {g.domain}"
        assert len(val_toks.intersection(test_toks)) == 0, f"Val and Test overlap in {g.domain}"
        assert len(tr_toks.intersection(novel_toks)) == 0, f"Train and Novel overlap in {g.domain}"


def test_routing_diagnosis_rehearsal_remediation(device: torch.device):
    """Verify diagnosis isolates M102.2 router collapse and proves rehearsal supervision fix."""
    diag = RoutingDiagnosticM103(d_model=64, device=device)
    report = diag.run_isolated_diagnosis(seed=42, episodes_per_domain=4, steps_per_episode=15)

    assert report.num_domains == 9
    assert report.rehearsal_supervised_routing_accuracy > report.unbalanced_m102_2_routing_accuracy
    assert report.rehearsal_supervised_routing_accuracy >= 75.0
    assert report.remediation_verified is True
    assert "catastrophic forgetting" in report.diagnosed_root_cause


def test_dynamic_capacity_audit_and_reconciliation(device: torch.device):
    """Verify live parameter inspection reconciles exact persistent parameters with 0 discrepancy."""
    # 1. Base student
    student_base = TextStudentM103(
        tier=CapacityTierM100.TIER_20K,
        device=device,
        enable_persistent_growth=False,
    )
    rep_base = DynamicCapacityAuditorM103.audit_student(student_base)
    assert rep_base.unexplained_discrepancy == 0
    assert rep_base.reconciliation_verified is True

    # 2. Grown student
    student_grown = TextStudentM103(
        tier=CapacityTierM100.TIER_20K,
        device=device,
        enable_persistent_growth=True,
    )
    rep_grown = DynamicCapacityAuditorM103.audit_student(student_grown)
    assert rep_grown.persistent_growth_active is True
    assert rep_grown.unexplained_discrepancy == 0
    assert rep_grown.reconciliation_verified is True


def test_text_student_autoregressive_generation_and_loss(device: torch.device):
    """Verify causal autoregressive forward pass, loss with prompt masking, and generation."""
    student = TextStudentM103(tier=CapacityTierM100.TIER_20K, device=device)
    curric = TextCurriculumGeneratorM103(seed=42)
    goal = curric.generate_energy_goal()

    # Compute loss on batch of QA items
    loss = student.compute_qa_loss(goal.train_items[:4], active_domain=goal.domain.value)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    assert loss.requires_grad is True

    # Test greedy generation
    prompt_toks = goal.train_items[0].full_tokens[:goal.train_items[0].prompt_len]
    gen_ids = student.generate(prompt_toks, max_new_tokens=10, active_domain=goal.domain.value)
    assert isinstance(gen_ids, list)
    assert len(gen_ids) <= 10


def test_bit_exact_crash_recovery_state_logits(device: torch.device):
    """Verify bit-exact crash recovery compares state dicts, continuous logits, and predictions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        student = TextStudentM103(tier=CapacityTierM100.TIER_20K, device=device)
        student.step_counter = 42

        chk_path = Path(tmp_dir) / "student_m103.pt"
        student.save_checkpoint(chk_path)

        reloaded = TextStudentM103.load_checkpoint(chk_path, device=device)
        assert reloaded.step_counter == 42

        sample_tokens = student.tokenizer.encode("Verification probe string")
        cert = student.verify_bit_exact_recovery(reloaded, sample_tokens)

        assert cert["bit_exact_certified"] is True
        assert cert["parameter_match"] is True
        assert cert["max_logit_difference"] < 1e-6
        assert cert["argmax_predictions_match"] is True


def test_multi_mode_comparator(device: torch.device):
    """Verify comparative evaluation executes across all 4 learning modes."""
    student = TextStudentM103(tier=CapacityTierM100.TIER_20K, device=device)
    curric = TextCurriculumGeneratorM103(seed=42)
    goal = curric.generate_energy_goal()

    comp = NaturalLanguageComparatorM103(device=device)
    rep = comp.evaluate_modes(student, goal, max_steps=2)

    assert rep.neural_learning.mode_name == "Neural Learning (Adapters)"
    assert rep.in_context_learning.mode_name == "In-Context Learning (Prompted)"
    assert rep.retrieval_memory.mode_name == "Retrieval Memory Baseline"
    assert rep.fresh_student.mode_name == "Fresh Student Baseline"
    assert rep.neural_learning.adaptation_steps > 0
    assert rep.in_context_learning.adaptation_steps == 0
