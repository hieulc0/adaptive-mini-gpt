"""Unit and Integration Tests for M28 Language Foundation Benchmark."""

from __future__ import annotations

import pytest

from adaptive_mini_gpt.data.bpe_scaling import (
    compare_tokenizers,
    train_scaled_bpe_tokenizer,
)
from adaptive_mini_gpt.data.language_corpus import (
    ABSTENTION_TARGET,
    PRETRAIN_PROSE_CORPUS,
    SEALED_TEST_QA_DOCUMENTS,
    TRAIN_QA_DOCUMENTS,
    VAL_DOCUMENTS,
    compute_corpus_fingerprint,
    format_qa_completion,
    format_qa_prompt,
    get_corpus_statistics,
    normalize_text,
    parse_language_completion,
    verify_all_language_splits_disjoint,
    verify_format_compliance,
)
from adaptive_mini_gpt.data.language_data import (
    BenchmarkSample,
    BenchmarkSplit,
    DomainType,
    build_language_benchmark_samples,
    verify_language_exact,
    verify_language_grounded,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer


def test_corpus_statistics_and_fingerprints() -> None:
    """Validate corpus statistics and deterministic SHA-256 fingerprints."""
    stats = get_corpus_statistics()
    assert stats["pretrain_passages"] >= 50
    assert stats["train_documents"] == 16
    assert stats["val_documents"] == 6
    assert stats["test_documents"] == 8
    assert stats["train_qa_samples"] >= 80
    assert stats["val_qa_samples"] >= 30
    assert stats["test_qa_samples"] >= 40
    assert len(stats["pretrain_fingerprint"]) == 16
    assert len(stats["train_fingerprint"]) == 16
    assert len(stats["val_fingerprint"]) == 16
    assert len(stats["test_fingerprint"]) == 16


def test_corpus_disjointness_audit() -> None:
    """Verify strictly 0.0% leakage across train, val, and sealed test partitions."""
    audit = verify_all_language_splits_disjoint()
    assert audit["status"] == "PASSED"
    assert audit["question_overlap_rate"] == 0.0
    assert audit["document_overlap_rate"] == 0.0
    assert audit["sentence_overlap_rate"] == 0.0


def test_scaled_bpe_tokenizer_training() -> None:
    """Verify BPE scaling to 1024 tokens and roundtrip decoding."""
    prose_sample = " ".join(PRETRAIN_PROSE_CORPUS[:15])
    tok = train_scaled_bpe_tokenizer(prose_sample, target_vocab_size=512)

    assert tok.vocab_size == 512
    # Base is 7 special + 256 bytes = 263. 512 - 263 = 249 merges.
    assert len(tok.merges) == 249

    test_sentence = "Photosynthesis inside chloroplasts produces glucose and oxygen gas."
    encoded = tok.encode(test_sentence)
    assert len(encoded) > 0
    decoded = tok.decode(encoded)
    assert decoded == test_sentence


def test_bpe_tokenizer_compression_and_fragmentation() -> None:
    """Verify 1024-vocab tokenizer reduces fragmentation and sequence length vs 400 baseline."""
    all_prose = " ".join(PRETRAIN_PROSE_CORPUS[:25])
    tok400 = BPETokenizer()
    tok400.train(all_prose, target_vocab_size=400)

    tok1024 = train_scaled_bpe_tokenizer(all_prose, target_vocab_size=1024)

    comp = compare_tokenizers(tok400, tok1024, PRETRAIN_PROSE_CORPUS[:10])
    assert comp["scaled_chars_per_token"] > comp["baseline_chars_per_token"]
    assert comp["scaled_subwords_per_word"] < comp["baseline_subwords_per_word"]
    assert comp["sequence_length_reduction_pct"] > 15.0


def test_language_parsing_and_verifiers() -> None:
    """Verify completion parsing, format compliance, exact match, and safe abstention."""
    sample_answerable = BenchmarkSample(
        sample_id="test_ans",
        domain=DomainType.LANGUAGE,
        split=BenchmarkSplit.TEST,
        prompt="Sample prompt",
        expected_answer="Chloroplasts",
        allowed_answers=["in chloroplasts", "chloroplast"],
        is_unanswerable=False,
    )

    sample_unanswerable = BenchmarkSample(
        sample_id="test_unans",
        domain=DomainType.LANGUAGE,
        split=BenchmarkSplit.TEST,
        prompt="Sample prompt",
        expected_answer=ABSTENTION_TARGET,
        allowed_answers=["insufficient information", "unanswerable"],
        is_unanswerable=True,
    )

    # 1. Exact answer
    assert verify_language_exact("Chloroplasts<|eos|>\n", sample_answerable) is True
    assert verify_language_exact("chloroplast<|eos|>\n", sample_answerable) is True
    assert verify_language_exact("Mitochondria<|eos|>\n", sample_answerable) is False

    # 2. Grounded answer
    assert verify_language_grounded("They occur inside chloroplasts.<|eos|>\n", sample_answerable) is True
    assert verify_language_grounded("Wrong answer<|eos|>\n", sample_answerable) is False

    # 3. Unanswerable question: safe abstention vs hallucination
    assert verify_language_exact("Insufficient information.<|eos|>\n", sample_unanswerable) is True
    assert verify_language_exact("unanswerable<|eos|>\n", sample_unanswerable) is True
    assert verify_language_exact("42 degrees Celsius<|eos|>\n", sample_unanswerable) is False

    # 4. Format compliance
    assert verify_format_compliance("Short concise answer<|eos|>\n") is True
    assert verify_format_compliance("") is False


def test_vocabulary_expansion_torch() -> None:
    """Verify function-preserving weight expansion when PyTorch is available."""
    try:
        import torch
        from adaptive_mini_gpt.data.bpe_scaling import expand_model_vocabulary
        from adaptive_mini_gpt.model.capacity_migration import create_1m_control_config
        from adaptive_mini_gpt.model.transformer import MiniGPT
    except ImportError:
        pytest.skip("PyTorch not installed in local environment; runs exclusively on Tesla T4 GPU.")

    tok_old = BPETokenizer()
    tok_old.train("Small baseline corpus text.", target_vocab_size=400)

    tok_new = train_scaled_bpe_tokenizer("Expanded corpus with diverse English text.", target_vocab_size=1024)

    cfg = create_1m_control_config(vocab_size=tok_old.vocab_size)
    model = MiniGPT(cfg)

    expanded = expand_model_vocabulary(model, tok_old, tok_new)
    assert expanded.config.vocab_size == 1024
    assert expanded.wte.weight.shape == (1024, 128)
    assert expanded.lm_head.weight.shape == (1024, 128)
    # Check that original 400 weights were perfectly preserved
    assert torch.equal(expanded.wte.weight[:400], model.wte.weight)
    assert torch.equal(expanded.lm_head.weight[:400], model.lm_head.weight)
