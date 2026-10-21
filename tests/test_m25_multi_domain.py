"""Unit tests for M25 Multi-Domain Learning: Benchmark Interface, Language, and Logic.

Tests:
  1. MultiDomainBenchmark interface and reporting.
  2. Strict dataset partition isolation (0% fact or variable leakage).
  3. Language comprehension verifiers, answer extraction, and safe abstention.
  4. Symbolic logic verifiers, boolean operations, comparisons, chained deductions, and contradictions.
  5. Dataset tokenization, sequence alignment, and prompt loss masking.
"""

import pytest
import torch

from adaptive_mini_gpt.data.language_data import (
    ABSTENTION_TARGET,
    TEST_DOCUMENTS,
    TRAIN_DOCUMENTS,
    VAL_DOCUMENTS,
    LanguageComprehensionDataset,
    build_language_benchmark_samples,
    collate_language_batch,
    parse_language_completion,
    verify_language_answer,
)
from adaptive_mini_gpt.data.logic_data import (
    CONTRADICTION_TARGET,
    LOGIC_ABSTENTION_TARGET,
    TEST_ENTITIES,
    TEST_PROPERTIES,
    TEST_VARS,
    TRAIN_ENTITIES,
    TRAIN_PROPERTIES,
    TRAIN_VARS,
    VAL_ENTITIES,
    VAL_PROPERTIES,
    VAL_VARS,
    LogicReasoningDataset,
    generate_logic_samples_for_split,
    parse_logic_completion,
    verify_logic_answer,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.evaluation.multi_domain import (
    BenchmarkSample,
    BenchmarkSplit,
    DomainResult,
    DomainType,
    MultiDomainBenchmark,
    MultiDomainReport,
)


@pytest.fixture
def mock_tokenizer() -> BPETokenizer:
    tok = BPETokenizer()
    corpus = (
        "<|user|> <|assistant|> <|eos|> <|system|> <pad> <unk> "
        "Passage: Question: Answer: Premises: Conclude: True False Yes No Undetermined Contradiction "
        "Photosynthesis Jupiter Gutenberg Lift Lithosphere Lens Hydrothermal Webb Babbage "
        "Alpha Beta Gamma Delta Epsilon Zeta Theta Lambda Sigma Omega X Y Z "
        "orb cube cylinder prism token monolith vessel matrix conduit relic "
        "red blue metallic heavy warm bright dense resonant inert fragile "
        "The quick brown fox jumps over the lazy dog. 0 1 2 3 4 5 6 7 8 9"
    )
    tok.train(corpus, target_vocab_size=200)
    return tok


# =====================================================================
# 1. MultiDomainBenchmark Interface Tests
# =====================================================================


def test_benchmark_interface_registration():
    bench = MultiDomainBenchmark(version="m25-test")
    sample = BenchmarkSample(
        sample_id="test_s1",
        domain=DomainType.LANGUAGE,
        split=BenchmarkSplit.TEST,
        prompt="Sample prompt",
        expected_answer="Sample answer",
    )
    bench.register_sample(sample)

    samples = bench.get_samples(DomainType.LANGUAGE, BenchmarkSplit.TEST)
    assert len(samples) == 1
    assert samples[0].sample_id == "test_s1"

    # Ensure other splits and domains remain isolated
    assert len(bench.get_samples(DomainType.LANGUAGE, BenchmarkSplit.TRAIN)) == 0
    assert len(bench.get_samples(DomainType.LOGIC, BenchmarkSplit.TEST)) == 0


def test_report_serialization():
    dom_res = DomainResult(
        domain="language",
        split="test",
        total_samples=10,
        exact_accuracy=0.8,
        abstention_accuracy=1.0,
        format_compliance_rate=1.0,
        mean_latency_ms=12.5,
        sub_metrics={"reading_comp": 0.85},
    )
    report = MultiDomainReport(
        benchmark_version="m25-v1",
        model_id="test_model",
        checkpoint_sha256="abc123sha",
        tokenizer_fingerprint="tok_fp_1",
        param_count=895744,
        device="cuda",
        timestamp="2026-10-21T00:00:00",
        domain_results={"language": dom_res},
        retention_drift={"prose_original": 0.05},
        summary={"status": "QUALIFIED"},
    )
    data = report.to_dict()
    assert data["benchmark_version"] == "m25-v1"
    assert data["domain_results"]["language"]["exact_accuracy"] == 0.8
    assert data["retention_drift"]["prose_original"] == 0.05


# =====================================================================
# 2. Language Comprehension Partition Isolation and Verifier Tests
# =====================================================================


def test_language_document_partition_isolation():
    train_ids = {d.doc_id for d in TRAIN_DOCUMENTS}
    val_ids = {d.doc_id for d in VAL_DOCUMENTS}
    test_ids = {d.doc_id for d in TEST_DOCUMENTS}

    # Zero overlap in document identifiers
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

    # Zero overlap in passage texts
    train_texts = {d.text for d in TRAIN_DOCUMENTS}
    val_texts = {d.text for d in VAL_DOCUMENTS}
    test_texts = {d.text for d in TEST_DOCUMENTS}
    assert len(train_texts.intersection(val_texts)) == 0
    assert len(train_texts.intersection(test_texts)) == 0
    assert len(val_texts.intersection(test_texts)) == 0


def test_language_answer_verifiers():
    # 1. Answerable sample
    ans_sample = BenchmarkSample(
        sample_id="s1",
        domain=DomainType.LANGUAGE,
        split=BenchmarkSplit.TEST,
        prompt="What gas do plants release?",
        expected_answer="Oxygen",
        allowed_answers=["oxygen gas", "O2"],
        is_unanswerable=False,
    )

    # Exact and alias matches
    assert verify_language_answer("Oxygen", ans_sample)
    assert verify_language_answer("Oxygen gas", ans_sample)
    assert verify_language_answer("plants release oxygen.", ans_sample)

    # Reject incorrect answers and incorrect abstentions
    assert not verify_language_answer("Carbon dioxide", ans_sample)
    assert not verify_language_answer("Insufficient information", ans_sample)
    assert not verify_language_answer("", ans_sample)

    # 2. Unanswerable sample (safe abstention battery)
    unans_sample = BenchmarkSample(
        sample_id="s2",
        domain=DomainType.LANGUAGE,
        split=BenchmarkSplit.TEST,
        prompt="What was the construction cost?",
        expected_answer=ABSTENTION_TARGET,
        allowed_answers=["insufficient information", "unanswerable", "unknown"],
        is_unanswerable=True,
    )

    # Correct abstention
    assert verify_language_answer("Insufficient information.", unans_sample)
    assert verify_language_answer("It is unanswerable from the passage.", unans_sample)
    assert verify_language_answer("Unknown", unans_sample)

    # Hallucinated answer must fail
    assert not verify_language_answer("10 billion dollars", unans_sample)
    assert not verify_language_answer("It cost 5 million.", unans_sample)


def test_language_dataset_masking(mock_tokenizer: BPETokenizer):
    samples = build_language_benchmark_samples(BenchmarkSplit.TEST)
    assert len(samples) > 0

    ds = LanguageComprehensionDataset(samples[:5], mock_tokenizer, context_length=256, ignore_index=-1)
    assert len(ds) == 5

    x, y = ds[0]
    assert x.size(0) == y.size(0)
    # The prompt prefix tokens must be masked with -1
    assert y[0].item() == -1
    # Assistant response tokens must have valid token targets
    assert (y != -1).any()

    # Test batch collation
    batch_x, batch_y = collate_language_batch([ds[0], ds[1]], pad_token_id=0, ignore_index=-1)
    assert batch_x.size(0) == 2
    assert batch_x.size(1) == max(ds[0][0].size(0), ds[1][0].size(0))


# =====================================================================
# 3. Symbolic Logic Partition Isolation and Verifier Tests
# =====================================================================


def test_logic_partition_isolation():
    # Disjoint variable names
    train_v = set(TRAIN_VARS)
    val_v = set(VAL_VARS)
    test_v = set(TEST_VARS)
    assert len(train_v.intersection(val_v)) == 0
    assert len(train_v.intersection(test_v)) == 0
    assert len(val_v.intersection(test_v)) == 0

    # Disjoint entity classes
    train_e = set(TRAIN_ENTITIES)
    val_e = set(VAL_ENTITIES)
    test_e = set(TEST_ENTITIES)
    assert len(train_e.intersection(val_e)) == 0
    assert len(train_e.intersection(test_e)) == 0
    assert len(val_e.intersection(test_e)) == 0

    # Disjoint properties
    train_p = set(TRAIN_PROPERTIES)
    val_p = set(VAL_PROPERTIES)
    test_p = set(TEST_PROPERTIES)
    assert len(train_p.intersection(val_p)) == 0
    assert len(train_p.intersection(test_p)) == 0
    assert len(val_p.intersection(test_p)) == 0


def test_logic_verifiers():
    # 1. Boolean logic
    bool_sample = BenchmarkSample(
        sample_id="b1",
        domain=DomainType.LOGIC,
        split=BenchmarkSplit.TEST,
        prompt="Prompt...",
        expected_answer="True",
    )
    assert verify_logic_answer("True", bool_sample)
    assert verify_logic_answer("The value is True.", bool_sample)
    assert not verify_logic_answer("False", bool_sample)
    assert not verify_logic_answer("Undetermined", bool_sample)

    # 2. Transitive comparison
    comp_sample = BenchmarkSample(
        sample_id="c1",
        domain=DomainType.LOGIC,
        split=BenchmarkSplit.TEST,
        prompt="Prompt...",
        expected_answer="Yes",
    )
    assert verify_logic_answer("Yes", comp_sample)
    assert not verify_logic_answer("No", comp_sample)
    assert not verify_logic_answer("Undetermined", comp_sample)

    # 3. Contradiction abstention
    contra_sample = BenchmarkSample(
        sample_id="k1",
        domain=DomainType.LOGIC,
        split=BenchmarkSplit.TEST,
        prompt="Prompt...",
        expected_answer=CONTRADICTION_TARGET,
        is_unanswerable=True,
    )
    assert verify_logic_answer("Contradiction detected.", contra_sample)
    assert verify_logic_answer("There is a contradiction in premises.", contra_sample)
    assert not verify_logic_answer("Yes", contra_sample)
    assert not verify_logic_answer("True", contra_sample)


def test_logic_sample_generation_and_masking(mock_tokenizer: BPETokenizer):
    test_samples = generate_logic_samples_for_split(BenchmarkSplit.TEST, count_per_category=5, seed=42)
    # 5 categories x 5 samples = 25 samples
    assert len(test_samples) == 25

    # Check that test samples ONLY use test variables and entities
    for s in test_samples:
        for tv in TRAIN_VARS:
            # Word boundary check to ensure no train variable appears
            assert f" {tv} " not in s.prompt and f"Item {tv} " not in s.prompt

    ds = LogicReasoningDataset(test_samples, mock_tokenizer, context_length=256, ignore_index=-1)
    assert len(ds) == 25

    x, y = ds[0]
    assert x.size(0) == y.size(0)
    assert y[0].item() == -1
    assert (y != -1).any()
