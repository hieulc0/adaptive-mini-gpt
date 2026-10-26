"""Unit Tests for Milestone M103-D: Genuine Concept Acquisition & Generalization."""

import pytest
import torch

from adaptive_mini_gpt.natural_language_learning_m103.audit_m103 import (
    DynamicCapacityAuditorM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m103_d import (
    ConceptLearnerM103_D,
)
from adaptive_mini_gpt.natural_language_learning_m103.number_tokenizer_m103_d import (
    NumberAwareTokenizerM103_D,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)


def test_number_tokenizer_roundtrip_and_atomic_mapping():
    """Verify NumberAwareTokenizerM103_D handles byte digits, atomic number tokens, and scratchpads."""
    tok = NumberAwareTokenizerM103_D()

    # 1. Standard byte encoding roundtrip
    text = "An orb starts with 6 energy."
    toks_byte = tok.encode(text, use_number_tokens=False)
    decoded_byte = tok.decode(toks_byte)
    assert decoded_byte == text

    # 2. Structured number tokens mapping
    assert tok.is_number_token(916) is True
    assert tok.number_token_to_int(916) == 16
    assert tok.int_to_number_token(16) == 916

    # 3. Structured number encoding roundtrip
    toks_num = tok.encode("16 energy", use_number_tokens=True)
    assert 916 in toks_num
    decoded_num = tok.decode(toks_num)
    assert decoded_num == "16 energy"

    # 4. Scratchpad sequence formatting
    seq, p_len = tok.format_qa_sequence(
        question="What is final energy?",
        answer="8 energy",
        explanation="Blue adds 2.",
        scratchpad="s=6,+2=8",
    )
    assert p_len > 0
    assert len(seq) > p_len
    assert tok.SCRATCHPAD_ID in seq
    assert seq[-1] == tok.EOS_ID


def test_dataset_partitioning_interpolation_vs_extrapolation():
    """Verify that M103-D strictly partitions interpolation (in-range odd) from extrapolation (out-of-range)."""
    learner = ConceptLearnerM103_D()
    datasets = learner.build_m103_d_datasets(mode="raw_bytes")

    train_items = datasets["train"]
    val_interp = datasets["val_interpolation"]
    test_extrap = datasets["test_extrapolation"]
    test_para = datasets["test_paraphrase"]
    test_hypo = datasets["test_hypothetical"]

    assert len(train_items) > 0
    assert len(val_interp) > 0
    assert len(test_extrap) > 0
    assert len(test_para) > 0
    assert len(test_hypo) > 0

    # Extract starting numbers from questions
    train_nums = {learner.extract_numeric_value(it.question) for it in train_items}
    interp_nums = {learner.extract_numeric_value(it.question) for it in val_interp}
    extrap_nums = {learner.extract_numeric_value(it.question) for it in test_extrap}

    # Verify train starting values are all EVEN: {2, 4, 6, 8, 10, 12, 14}
    assert all(n % 2 == 0 for n in train_nums if n is not None)
    # Verify interpolation starting values are all ODD within the same range: {3, 5, 7, 9, 11, 13}
    assert all(n % 2 == 1 for n in interp_nums if n is not None)
    assert max(n for n in interp_nums if n is not None) <= 14
    # Zero overlap between train and interpolation starting numbers
    assert not train_nums.intersection(interp_nums)

    # Verify extrapolation starting values are all > 14
    assert all(n > 14 for n in extrap_nums if n is not None)
    assert not train_nums.intersection(extrap_nums)


def test_parameter_accounting_reconciliation_zero_discrepancy():
    """Verify exact parameter breakdown: 1,088,256 base + 20,970 routers + 2,048 adapters + 198,272 growth = 1,309,546."""
    student = TextStudentM103(enable_persistent_growth=True, device=torch.device("cpu"))
    audit = DynamicCapacityAuditorM103.audit_student(student)

    assert audit.base_backbone_params == 1088256
    assert audit.layer_routers_params == 20970
    assert audit.domain_adapters_params == 2048
    assert audit.growth_block_params == 198272
    assert audit.total_measured_persistent_params == 1309546
    assert audit.unexplained_discrepancy == 0
    assert audit.reconciliation_verified is True


def test_loss_computation_across_representation_modes():
    """Verify that causal autoregressive loss executes cleanly across all three modes."""
    learner = ConceptLearnerM103_D(device=torch.device("cpu"))
    student = TextStudentM103(enable_persistent_growth=True, device=torch.device("cpu"))
    learner.configure_selective_backbone(student)

    for mode in ["raw_bytes", "structured_tokens", "scratchpad"]:
        ds = learner.build_m103_d_datasets(mode=mode)
        loss = learner._compute_batch_loss(student, ds["train"][:4])
        assert not torch.isnan(loss)
        assert loss.item() > 0.0


def test_hypothetical_predictions_and_corrective_structures():
    """Verify Phase D3 simulation and Phase D4 corrective structures."""
    learner = ConceptLearnerM103_D(device=torch.device("cpu"))
    student = TextStudentM103(enable_persistent_growth=True, device=torch.device("cpu"))
    learner.configure_selective_backbone(student)

    ds = learner.build_m103_d_datasets(mode="raw_bytes")

    # Smoke test Phase D3 evaluation
    d3_rep = learner.run_phase_d3_hypothetical_predictions(student, ds["test_hypothetical"])
    assert hasattr(d3_rep, "counterfactual_em")
    assert hasattr(d3_rep, "novel_entity_em")
    assert hasattr(d3_rep, "multistep_em")

    # Smoke test Phase D4 corrective learning
    d4_rep = learner.run_phase_d4_corrective_learning(
        student=student,
        original_train_items=ds["train"][:10],
        corrective_steps=2,
    )
    assert hasattr(d4_rep, "feedback_adaptation_gain_pct")
    assert hasattr(d4_rep, "backward_retention_original_rule_pct")
