"""Unit tests for M24 Reliable Neural Arithmetic Transitions and Deep Extrapolation."""

import re
import pytest
import torch

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.rules.arithmetic_dataset import ArithmeticProblem
from adaptive_mini_gpt.rules.reliable_transition import (
    ElementaryTransition,
    ElementaryTransitionDataset,
    audit_elementary_transition_space,
    collate_elementary_batch,
    execute_reliable_recurrent_addition,
    format_elementary_prompt_and_target,
    get_all_200_transitions,
    parse_elementary_completion,
)


@pytest.fixture
def trained_tokenizer() -> BPETokenizer:
    tok = BPETokenizer()
    corpus = (
        "[IN] [OUT] [ADD] a= b= c= c_in= d= C_OUT: DIGIT: Col: + > : \n "
        "0 1 2 3 4 5 6 7 8 9 = | "
        "The quick brown fox jumps over the lazy dog. Photosynthesis Kepler Amazon plate"
    )
    tok.train(corpus, target_vocab_size=150)
    return tok


def test_elementary_transition_space_enumeration():
    all_200 = get_all_200_transitions()
    assert len(all_200) == 200

    # Ensure all (a, b, c_in) triples are unique
    triples = {(t.a, t.b, t.c_in) for t in all_200}
    assert len(triples) == 200

    # Test classification
    tr_gen = ElementaryTransition.create(7, 8, 0)  # 7+8=15 >= 10 -> generate
    assert tr_gen.carry_category == "generate"
    assert tr_gen.carry_out == 1
    assert tr_gen.write_digit == 5

    tr_prop = ElementaryTransition.create(4, 5, 1)  # 4+5=9 < 10, 4+5+1=10 >= 10 -> propagate
    assert tr_prop.carry_category == "propagate"
    assert tr_prop.carry_out == 1
    assert tr_prop.write_digit == 0

    tr_none = ElementaryTransition.create(2, 3, 1)  # 2+3+1=6 < 10 -> none
    assert tr_none.carry_category == "none"
    assert tr_none.carry_out == 0
    assert tr_none.write_digit == 6

    # Test symmetry
    tr_sym = ElementaryTransition.create(4, 4, 1)
    assert tr_sym.is_symmetric
    tr_asym = ElementaryTransition.create(4, 5, 1)
    assert not tr_asym.is_symmetric


def test_format_and_parse_across_formats():
    tr = ElementaryTransition.create(7, 8, 1)  # 7 + 8 + 1 = 16 -> d=6, c=1

    # 1. Canonical State
    p_can, t_can = format_elementary_prompt_and_target(tr, format_type="canonical_state")
    assert "[IN] a=7 b=8 c=1" in p_can
    assert "[OUT] d=6 c=1" in t_can
    parsed_can = parse_elementary_completion(t_can, format_type="canonical_state")
    assert parsed_can["is_valid"]
    assert parsed_can["write_digit"] == 6
    assert parsed_can["carry_out"] == 1

    # 2. Separate Supervision (Carry First)
    p_sep, t_sep = format_elementary_prompt_and_target(tr, format_type="separate_supervision")
    assert "[ADD] a=7 b=8 c_in=1" in p_sep
    assert "C_OUT: 1 | DIGIT: 6" in t_sep
    parsed_sep = parse_elementary_completion(t_sep, format_type="separate_supervision")
    assert parsed_sep["is_valid"]
    assert parsed_sep["write_digit"] == 6
    assert parsed_sep["carry_out"] == 1

    # 3. M23 Baseline
    p_m23, t_m23 = format_elementary_prompt_and_target(tr, format_type="m23_baseline")
    assert "Col: 7 + 8 + 1" in p_m23
    assert "16 > 6 : 1" in t_m23
    parsed_m23 = parse_elementary_completion(t_m23, format_type="m23_baseline")
    assert parsed_m23["is_valid"]
    assert parsed_m23["write_digit"] == 6
    assert parsed_m23["carry_out"] == 1


def test_elementary_transition_dataset_and_collate(trained_tokenizer: BPETokenizer):
    all_200 = get_all_200_transitions()
    ds = ElementaryTransitionDataset(
        transitions=all_200[:20],
        tokenizer=trained_tokenizer,
        format_type="canonical_state",
        context_length=64,
        replay_texts=["Photosynthesis converts solar light into chemical energy."],
        replay_ratio=0.1,
    )
    assert len(ds) >= 20
    inp, lab = ds[0]
    assert inp.shape == lab.shape
    assert (lab == -100).any()
    assert (lab != -100).any()

    batch = [ds[0], ds[1]]
    b_in, b_lab = collate_elementary_batch(batch, pad_token_id=0)
    assert b_in.shape[0] == 2
    assert b_lab.shape[0] == 2


class MockTransitionOracle(torch.nn.Module):
    """Mock model that returns exact oracle transition tokens for test simulation."""

    def __init__(self, tokenizer: BPETokenizer, format_type: str = "canonical_state"):
        super().__init__()
        self.tokenizer = tokenizer
        self.format_type = format_type

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, None]:
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.tokenizer.vocab_size)
        text = self.tokenizer.decode(input_ids[0].tolist())

        # Extract a, b, c from prompt
        m = re.search(r"a=(\d+)\s+b=(\d+)\s+c(?:_in)?=(\d+)", text)
        if not m:
            m = re.search(r"Col:\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)", text)

        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            c = int(m.group(3))
            tr = ElementaryTransition.create(a, b, c)
            _, target_str = format_elementary_prompt_and_target(tr, format_type=self.format_type)

            target_toks = self.tokenizer.encode(target_str)
            assistant_part = text.split("<|assistant|>\n")[-1]
            gen_toks = self.tokenizer.encode(assistant_part)
            if len(gen_toks) < len(target_toks):
                logits[0, -1, target_toks[len(gen_toks)]] = 100.0
                return logits, None

        logits[0, -1, self.tokenizer.eos_id] = 100.0
        return logits, None


def test_audit_transition_space(trained_tokenizer: BPETokenizer):
    mock = MockTransitionOracle(trained_tokenizer, format_type="canonical_state")
    held_out = {(7, 8, 1), (9, 9, 0), (3, 4, 1)}
    audit = audit_elementary_transition_space(
        model=mock,
        tokenizer=trained_tokenizer,
        device=torch.device("cpu"),
        format_type="canonical_state",
        held_out_triples=held_out,
    )
    assert audit["overall_exact_acc"] == 100.0
    assert audit["overall_digit_acc"] == 100.0
    assert audit["overall_carry_acc"] == 100.0
    assert audit["by_split"]["held_out"]["total"] == 3
    assert audit["by_split"]["held_out"]["exact_acc"] == 100.0


def test_execute_reliable_addition_deep_lengths(trained_tokenizer: BPETokenizer):
    mock = MockTransitionOracle(trained_tokenizer, format_type="canonical_state")
    device = torch.device("cpu")

    # 1. 2-digit problem
    p2 = ArithmeticProblem.create(47, 85)
    r2 = execute_reliable_recurrent_addition(mock, trained_tokenizer, p2, device, format_type="canonical_state")
    assert r2["is_exact"]
    assert r2["predicted_result"] == 132
    assert r2["first_failure_col"] is None

    # 2. 8-digit problem
    p8 = ArithmeticProblem.create(98765432, 12345678)
    r8 = execute_reliable_recurrent_addition(mock, trained_tokenizer, p8, device, format_type="canonical_state")
    assert r8["is_exact"]
    assert r8["predicted_result"] == 111111110

    # 3. 16-digit problem (Deep Extrapolation)
    a16 = 1234567890123456
    b16 = 9876543210987654
    p16 = ArithmeticProblem.create(a16, b16)
    r16 = execute_reliable_recurrent_addition(mock, trained_tokenizer, p16, device, format_type="canonical_state")
    assert r16["is_exact"]
    assert r16["predicted_result"] == a16 + b16

    # 4. 32-digit problem (Extreme Extrapolation)
    a32 = int("9" * 32)
    b32 = 1
    p32 = ArithmeticProblem.create(a32, b32)
    r32 = execute_reliable_recurrent_addition(mock, trained_tokenizer, p32, device, format_type="canonical_state")
    assert r32["is_exact"]
    assert r32["predicted_result"] == 10**32
