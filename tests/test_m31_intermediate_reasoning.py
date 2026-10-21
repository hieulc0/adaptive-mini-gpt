"""Unit and Integration Tests for M31 Verified Intermediate Reasoning."""

from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.data.bpe_scaling import train_scaled_bpe_tokenizer
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.capacity_migration import create_1m_control_config
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.reasoning.dataset import (
    MultiRepresentationReasoningDataset,
    collate_reasoning_batch,
)
from adaptive_mini_gpt.reasoning.scratchpad import (
    ApproachType,
    ReasoningDomain,
    build_arithmetic_reasoning_sample,
    build_language_reasoning_sample,
    build_logic_reasoning_sample,
    parse_reasoning_completion,
)


def test_arithmetic_scratchpad_formatting_and_parsing() -> None:
    """Verify arithmetic column-by-column trace formatting and parsing."""
    sample = build_arithmetic_reasoning_sample(487, 596)
    assert sample.expected_final_answer == "1083"
    assert "col 0: 7 + 6 + 0 = 13 -> write 3, carry 1" in sample.target_scratchpad
    assert "col 1: 8 + 9 + 1 = 18 -> write 8, carry 1" in sample.target_scratchpad
    assert "col 2: 4 + 5 + 1 = 10 -> write 0, carry 1" in sample.target_scratchpad
    assert "col 3: 0 + 0 + 1 = 1 -> write 1, carry 0" in sample.target_scratchpad
    assert "Final Answer: 1083" in sample.target_scratchpad

    # Verify parser
    parsed = parse_reasoning_completion(sample.target_scratchpad, ApproachType.SCRATCHPAD, ReasoningDomain.ARITHMETIC)
    assert parsed.is_valid_syntax
    assert parsed.has_final_answer_tag
    assert parsed.parsed_final_answer == "1083"
    assert len(parsed.intermediate_steps) == 4


def test_logic_scratchpad_formatting_and_parsing() -> None:
    """Verify symbolic logic step-by-step deduction formatting and parsing."""
    sample = build_logic_reasoning_sample(
        var_chain=["A", "B", "C"],
        negations=[False, True],
        query_var="C",
        initial_fact=("A", True),
    )
    # A=True, If A then B -> B=True
    # B=True, If B then not C -> C=False
    assert sample.expected_final_answer == "False"
    assert "Step 1: A is True and (If A then B) -> B is True" in sample.target_scratchpad
    assert "Step 2: B is True and (If B then not C) -> C is False" in sample.target_scratchpad
    assert "Final Answer: False" in sample.target_scratchpad

    parsed = parse_reasoning_completion(sample.target_scratchpad, ApproachType.SCRATCHPAD, ReasoningDomain.LOGIC)
    assert parsed.is_valid_syntax
    assert parsed.parsed_final_answer == "False"
    assert len(parsed.intermediate_steps) == 2


def test_language_evidence_span_scratchpad() -> None:
    """Verify language evidence extraction scratchpad formatting and parsing."""
    sample = build_language_reasoning_sample(
        sample_id="test_lang",
        context="Ribosomes are macromolecular machines found within all living cells that perform biological protein synthesis.",
        question="What is the function of ribosomes?",
        evidence_span="perform biological protein synthesis.",
        extracted_fact="Ribosomes perform biological protein synthesis",
        expected_answer="protein synthesis",
    )
    assert 'Evidence: "perform biological protein synthesis."' in sample.target_scratchpad
    assert "Extracted Fact: Ribosomes perform biological protein synthesis" in sample.target_scratchpad
    assert "Final Answer: protein synthesis" in sample.target_scratchpad

    parsed = parse_reasoning_completion(sample.target_scratchpad, ApproachType.SCRATCHPAD, ReasoningDomain.LANGUAGE)
    assert parsed.is_valid_syntax
    assert parsed.parsed_final_answer == "protein synthesis"
    assert len(parsed.intermediate_steps) == 2


def test_recurrent_arithmetic_steps_structure() -> None:
    """Verify recurrent single-column step generation."""
    sample = build_arithmetic_reasoning_sample(48, 75)
    # 8 + 5 = 13 (write 3, carry 1)
    # 4 + 7 + 1 = 12 (write 2, carry 1)
    # 0 + 0 + 1 = 1 (write 1, carry 0) -> 123
    assert len(sample.recurrent_steps) == 3
    p0, t0 = sample.recurrent_steps[0]
    assert "Col: 8 + 5 + 0" in p0
    assert "13 > 3 : 1" in t0

    p1, t1 = sample.recurrent_steps[1]
    assert "Col: 4 + 7 + 1" in p1
    assert "12 > 2 : 1" in t1

    p2, t2 = sample.recurrent_steps[2]
    assert "Col: 0 + 0 + 1" in p2
    assert "1 > 1 : 0 | HALT" in t2


def test_dataset_prompt_loss_masking() -> None:
    """Verify that prompt tokens are masked with ignore_index=-100 in the dataset."""
    tok = train_scaled_bpe_tokenizer("Add: 12 + 34 = 46 col write carry Step Final Answer", target_vocab_size=32)

    sample = build_arithmetic_reasoning_sample(12, 34)
    ds = MultiRepresentationReasoningDataset(
        samples=[sample],
        tokenizer=tok,
        approach=ApproachType.DIRECT,
        context_length=64,
        ignore_index=-100,
    )
    assert len(ds) == 1
    item = ds[0]
    labels = item["labels"].tolist()
    # At least some prompt tokens must be masked with -100
    assert -100 in labels
    # Target tokens must not be masked
    assert any(l != -100 for l in labels)
