"""Unit Tests for M27 Multi-Domain Learning Quality and Capacity Scaling.

Covers:
  - Phase 1: Replay and Retention disjointness audit (0.0% sentence/paragraph overlap).
  - Phase 2: Language dataset isolation, prompt formatting, and exact/grounded verifiers.
  - Phase 4: Capacity scaling parameter counts and function-preserving weight migration.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from adaptive_mini_gpt.data.language_data import (
    ABSTENTION_TARGET,
    BenchmarkSplit,
    DocumentPassage,
    TEST_DOCUMENTS,
    TRAIN_DOCUMENTS,
    VAL_DOCUMENTS,
    build_language_benchmark_samples,
    format_qa_prompt,
    normalize_text,
    parse_language_completion,
    verify_language_exact,
    verify_language_grounded,
)
from adaptive_mini_gpt.data.retention_data import (
    EXPANDED_RETENTION_CORPUS,
    INDEPENDENT_REPLAY_CORPUS,
    verify_corpora_disjointness,
)
from adaptive_mini_gpt.evaluation.multi_domain import BenchmarkSample, DomainType
from adaptive_mini_gpt.model.capacity_migration import (
    count_parameters,
    create_1m_control_config,
    create_3m_scaled_config,
    estimate_training_flops_per_token,
    migrate_1m_to_3m,
)
from adaptive_mini_gpt.model.transformer import MiniGPT


def test_retention_replay_strict_disjointness() -> None:
    """Audit: Guarantee 0.0% sentence overlap between evaluation retention and training replay."""
    overlap_rate = verify_corpora_disjointness()
    assert overlap_rate == 0.0, f"Expected 0.0% overlap between retention and replay corpora, got {overlap_rate}"

    # Also verify non-empty text
    assert len(EXPANDED_RETENTION_CORPUS) >= 4
    assert len(INDEPENDENT_REPLAY_CORPUS) >= 4


def test_language_document_partition_isolation() -> None:
    """Phase 2: Verify strict partition isolation between train, val, and test documents."""
    train_ids = {d.doc_id for d in TRAIN_DOCUMENTS}
    val_ids = {d.doc_id for d in VAL_DOCUMENTS}
    test_ids = {d.doc_id for d in TEST_DOCUMENTS}

    # Verify zero document ID overlap
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val document IDs overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test document IDs overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test document IDs overlap!"

    # Verify zero text overlap
    train_texts = {d.text.strip().lower() for d in TRAIN_DOCUMENTS}
    val_texts = {d.text.strip().lower() for d in VAL_DOCUMENTS}
    test_texts = {d.text.strip().lower() for d in TEST_DOCUMENTS}

    assert len(train_texts.intersection(val_texts)) == 0, "Train and Val text content overlaps!"
    assert len(train_texts.intersection(test_texts)) == 0, "Train and Test text content overlaps!"
    assert len(val_texts.intersection(test_texts)) == 0, "Val and Test text content overlaps!"


def test_language_verifiers_and_normalization() -> None:
    """Phase 2: Verify text normalization and exact vs grounded answer verification."""
    assert normalize_text("The Chloroplasts.") == "chloroplasts"
    assert normalize_text("a Great Red Spot!") == "great red spot"
    assert normalize_text("An Event Horizon") == "event horizon"

    sample_ans = BenchmarkSample(
        sample_id="test_s1",
        domain=DomainType.LANGUAGE,
        split=BenchmarkSplit.TEST,
        prompt="Sample Prompt",
        expected_answer="Chloroplasts",
        allowed_answers=["in chloroplasts", "chloroplast"],
        is_unanswerable=False,
    )

    # Exact match
    assert verify_language_exact("chloroplasts", sample_ans)
    assert verify_language_exact("The chloroplasts", sample_ans)
    assert verify_language_exact("Chloroplast", sample_ans)
    assert not verify_language_exact("Cellular chloroplasts within plant cells", sample_ans)

    # Grounded extraction match
    assert verify_language_grounded("chloroplasts", sample_ans)
    assert verify_language_grounded("in chloroplasts", sample_ans)
    assert verify_language_grounded("Photosynthesis occurs in chloroplasts", sample_ans)
    # Stopword protection: trivial stopword must NOT falsely match
    assert not verify_language_grounded("in", sample_ans)
    assert not verify_language_grounded("the", sample_ans)

    # Safe abstention verification
    sample_unans = BenchmarkSample(
        sample_id="test_s2",
        domain=DomainType.LANGUAGE,
        split=BenchmarkSplit.TEST,
        prompt="Sample Prompt",
        expected_answer=ABSTENTION_TARGET,
        allowed_answers=["insufficient information", "unanswerable"],
        is_unanswerable=True,
    )

    assert verify_language_exact("Insufficient information.", sample_unans)
    assert verify_language_grounded("This is unanswerable from the passage.", sample_unans)
    assert not verify_language_exact("Chloroplasts", sample_unans)
    assert not verify_language_grounded("Chloroplasts", sample_unans)

    # False abstention rejection on answerable sample
    assert not verify_language_exact("Insufficient information.", sample_ans)
    assert not verify_language_grounded("Insufficient information.", sample_ans)


def test_capacity_scaling_parameters() -> None:
    """Phase 4: Verify architecture parameter counts and scaling ratios."""
    cfg_1m = create_1m_control_config()
    model_1m = MiniGPT(cfg_1m)
    params_1m = count_parameters(model_1m)
    # 1M parameter model with tied/untied weights
    assert 850_000 <= params_1m <= 950_000, f"Expected ~1M params, got {params_1m}"

    cfg_3m = create_3m_scaled_config()
    model_3m = MiniGPT(cfg_3m)
    params_3m = count_parameters(model_3m)
    assert 3_300_000 <= params_3m <= 3_550_000, f"Expected ~3.43M params, got {params_3m}"

    flops_1m = estimate_training_flops_per_token(params_1m)
    flops_3m = estimate_training_flops_per_token(params_3m)
    assert flops_3m > 3.5 * flops_1m


def test_capacity_migration_weight_transfer() -> None:
    """Phase 4: Test weight-preserving dimension migration from 1M to 3M architecture."""
    cfg_1m = create_1m_control_config()
    model_1m = MiniGPT(cfg_1m)

    # Initialize random non-zero weights
    for p in model_1m.parameters():
        nn.init.normal_(p, mean=0.0, std=0.02)

    model_3m = migrate_1m_to_3m(model_1m)

    # Verify embeddings migrated
    assert torch.equal(model_3m.wte.weight[:, :128], model_1m.wte.weight)
    assert torch.all(model_3m.wte.weight[:, 128:] == 0.0)

    # Verify block 0 attention weights migrated
    b0_1m = model_1m.blocks[0]
    b0_3m = model_3m.blocks[0]

    # Q projection migrated
    assert torch.equal(b0_3m.attn.c_attn.weight[:128, :128], b0_1m.attn.c_attn.weight[:128, :128])
    # K projection migrated
    assert torch.equal(b0_3m.attn.c_attn.weight[256:384, :128], b0_1m.attn.c_attn.weight[128:256, :128])
    # V projection migrated
    assert torch.equal(b0_3m.attn.c_attn.weight[512:640, :128], b0_1m.attn.c_attn.weight[256:384, :128])

    # Unmigrated dimensions are zero
    assert torch.all(b0_3m.attn.c_attn.weight[:, 128:] == 0.0)
