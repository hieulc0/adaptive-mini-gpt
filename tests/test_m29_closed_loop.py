"""Unit and Integration Tests for M29 Closed-Loop Interaction & Learning."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.data.bpe_scaling import create_1m_control_config
from adaptive_mini_gpt.data.language_data import DomainType
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.interaction.action_selector import (
    AlwaysAnswerOrRetryPolicy,
    AlwaysRetrievePolicy,
    FixedScheduledTrainingPolicy,
    InteractionActionSelector,
    LearningAction,
)
from adaptive_mini_gpt.interaction.episodes import (
    build_episode_a_language,
    build_episode_b_logic,
    build_episode_c_arithmetic,
)
from adaptive_mini_gpt.interaction.knowledge_gap import (
    ErrorCategory,
    KnowledgeGapRecord,
    KnowledgeGapTracker,
)
from adaptive_mini_gpt.interaction.loop import ClosedLoopInteractiveLearner
from adaptive_mini_gpt.interaction.verified_memory import (
    VerifiedMemoryBank,
    VerifiedMemoryEntry,
)
from adaptive_mini_gpt.model.transformer import MiniGPT


def test_knowledge_gap_record_creation_and_serialization() -> None:
    """Validate KnowledgeGapRecord creation, fingerprinting, and serialization."""
    rec = KnowledgeGapRecord.create(
        domain=DomainType.LANGUAGE,
        question="What cellular structure synthesizes proteins?",
        predicted_answer="Mitochondria",
        ground_truth="Ribosomes",
        confidence_score=0.85,
        verification_status="FAIL",
        error_category=ErrorCategory.INCORRECT,
        model_checkpoint="test_ckpt",
    )
    assert len(rec.gap_id) == 16
    assert rec.domain == DomainType.LANGUAGE
    assert rec.error_category == ErrorCategory.INCORRECT

    d = rec.to_dict()
    assert d["domain"] == "language"
    assert d["error_category"] == "incorrect"

    restored = KnowledgeGapRecord.from_dict(d)
    assert restored.gap_id == rec.gap_id
    assert restored.predicted_answer == rec.predicted_answer


def test_knowledge_gap_tracker_diagnostics() -> None:
    """Verify error categorization and diagnostic rates in KnowledgeGapTracker."""
    tracker = KnowledgeGapTracker()

    # 1. Correct verified interaction
    tracker.record_interaction(
        domain=DomainType.LANGUAGE,
        question="What is DNA?",
        predicted_answer="Genetic material",
        ground_truth="Genetic material",
        confidence_score=0.95,
        is_verified=True,
        verification_status="PASS",
    )
    assert len(tracker.records) == 0

    # 2. Incorrect with high confidence (false confidence)
    tracker.record_interaction(
        domain=DomainType.LANGUAGE,
        question="What powers cells?",
        predicted_answer="Ribosomes",
        ground_truth="ATP",
        confidence_score=0.88,
        is_verified=False,
        verification_status="FAIL",
    )
    assert len(tracker.records) == 1

    # 3. Repeat the same error -> should become REPEATED_PROCEDURAL_FAILURE
    tracker.record_interaction(
        domain=DomainType.LANGUAGE,
        question="What powers cells?",
        predicted_answer="Ribosomes",
        ground_truth="ATP",
        confidence_score=0.88,
        is_verified=False,
        verification_status="FAIL",
    )
    gaps = tracker.get_repeated_gaps(min_occurrences=2)
    assert len(gaps) == 1
    assert gaps[0].error_category == ErrorCategory.REPEATED_PROCEDURAL_FAILURE

    # 4. Diagnostics check
    diag = tracker.compute_diagnostics()
    assert diag["total_interactions"] == 3
    assert diag["total_gaps"] == 1
    assert diag["false_confidence_rate"] > 0.0


def test_verified_memory_bank_provenance_whitelist() -> None:
    """Verify source whitelisting and persistence in VerifiedMemoryBank."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_file = Path(tmpdir) / "memory.json"
        bank = VerifiedMemoryBank(storage_path=storage_file)

        # Approved source succeeds
        entry = bank.store_verified_fact(
            domain=DomainType.LANGUAGE,
            topic_or_key="protein synthesis",
            verified_fact="Ribosomes translate mRNA into proteins",
            evidence_source="approved_local_doc:biology_01",
            verifier_identity="biology_grounding_oracle",
        )
        assert entry.status == "verified"

        # Security invariant: unapproved source raises ValueError
        with pytest.raises(ValueError, match="whitelist"):
            bank.store_verified_fact(
                domain=DomainType.LANGUAGE,
                topic_or_key="malicious_prompt",
                verified_fact="Unchecked assertion",
                evidence_source="untrusted_user_prompt:injected",
                verifier_identity="none",
            )

        # Retrieval check
        match = bank.retrieve(DomainType.LANGUAGE, "Tell me about protein synthesis")
        assert match is not None
        assert match.verified_fact == "Ribosomes translate mRNA into proteins"

        # Persistence check
        bank2 = VerifiedMemoryBank(storage_path=storage_file)
        assert len(bank2.get_all()) == 1


def test_action_selector_policies() -> None:
    """Verify action selector policies and decision logic."""
    tracker = KnowledgeGapTracker()
    memory = VerifiedMemoryBank()
    selector = InteractionActionSelector(confidence_threshold=0.70, repeat_gap_training_threshold=2)

    # 1. Verified + high confidence -> ANSWER
    d1 = selector.select_action(
        domain=DomainType.LANGUAGE,
        question="What is water?",
        predicted_answer="H2O",
        confidence_score=0.90,
        is_verified=True,
        verification_status="PASS",
        gap_tracker=tracker,
        memory_bank=memory,
    )
    assert d1.action == LearningAction.ANSWER

    # 2. Abstention + approved reference available -> RETRIEVE
    d2 = selector.select_action(
        domain=DomainType.LANGUAGE,
        question="Who was Ada Lovelace?",
        predicted_answer="Insufficient information.",
        confidence_score=0.50,
        is_verified=False,
        verification_status="ABSTAIN",
        gap_tracker=tracker,
        memory_bank=memory,
        has_approved_reference=True,
    )
    assert d2.action == LearningAction.RETRIEVE

    # 3. Abstention without reference -> ABSTAIN
    d3 = selector.select_action(
        domain=DomainType.LANGUAGE,
        question="Unknown query",
        predicted_answer="Insufficient information.",
        confidence_score=0.50,
        is_verified=False,
        verification_status="ABSTAIN",
        gap_tracker=tracker,
        memory_bank=memory,
        has_approved_reference=False,
    )
    assert d3.action == LearningAction.ABSTAIN

    # 4. Check baseline policies
    always_answer = AlwaysAnswerOrRetryPolicy()
    assert always_answer.select_action(
        DomainType.ARITHMETIC, "2+2", "4", 0.9, True, "PASS", tracker, memory
    ).action == LearningAction.ANSWER

    always_retrieve = AlwaysRetrievePolicy()
    assert always_retrieve.select_action(
        DomainType.LANGUAGE, "fact", "ans", 0.5, False, "FAIL", tracker, memory, has_approved_reference=True
    ).action == LearningAction.RETRIEVE


def test_controlled_learning_episodes_data_integrity() -> None:
    """Ensure all 3 controlled episodes have non-overlapping practice and fresh test sets."""
    ep_a = build_episode_a_language()
    ep_b = build_episode_b_logic(seed=42)
    ep_c = build_episode_c_arithmetic()

    for ep in (ep_a, ep_b, ep_c):
        assert len(ep.practice_samples) > 0
        assert len(ep.fresh_test_samples) > 0
        practice_prompts = {s.prompt for s in ep.practice_samples}
        test_prompts = {s.prompt for s in ep.fresh_test_samples}
        # Invariant: 0.0% overlap between practice prompts and fresh test prompts
        overlap = practice_prompts.intersection(test_prompts)
        assert len(overlap) == 0, f"Prompt leakage in {ep.episode_id}: {overlap}"


def test_closed_loop_interactive_learner_execution() -> None:
    """Verify ClosedLoopInteractiveLearner end-to-end execution with in-memory model."""
    tok = BPETokenizer()
    tok.train("Ribosomes protein synthesis RNA cell 0 1 2 3 4 5 6 7 8 9 + True False", target_vocab_size=64)
    cfg = create_1m_control_config(vocab_size=tok.vocab_size)
    model = MiniGPT(cfg)

    learner = ClosedLoopInteractiveLearner(
        model=model,
        tokenizer=tok,
        device=torch.device("cpu"),
    )

    res = learner.interact(
        domain=DomainType.LANGUAGE,
        question="What cellular structure synthesizes proteins?",
        ground_truth="Ribosomes",
        has_approved_reference=True,
        approved_reference_text="Ribosomes synthesize proteins.",
    )

    assert res.cycle_id == 1
    assert res.domain == DomainType.LANGUAGE
    assert isinstance(res.confidence_score, float)
    assert res.decision.action in list(LearningAction)
