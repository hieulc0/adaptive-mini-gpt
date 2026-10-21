"""Unit tests for M23 Recurrent Neural Arithmetic Transition Model."""

import pytest
import torch

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.rules.arithmetic_dataset import ArithmeticProblem
from adaptive_mini_gpt.rules.recurrent_arithmetic import (
    TransitionDataset,
    TransitionTriple,
    collate_transition_batch,
    execute_recurrent_addition,
    format_transition_problem,
    generate_all_transition_triples,
    parse_transition_completion,
)


@pytest.fixture
def trained_tokenizer() -> BPETokenizer:
    tok = BPETokenizer()
    corpus = (
        "Col: 0 1 2 3 4 5 6 7 8 9 + = > : | HALT CONT [LAST] \n "
        "The quick brown fox jumps over the lazy dog. Paris London Tokyo science energy"
    )
    tok.train(corpus, target_vocab_size=120)
    return tok


def test_transition_triple_creation():
    # 7 + 6 + 0 = 13 -> write 3, carry 1, not halt
    tr = TransitionTriple.create(7, 6, 0, is_last=False)
    assert tr.total_sum == 13
    assert tr.write_digit == 3
    assert tr.carry_out == 1
    assert not tr.is_last
    assert not tr.is_halt

    # 1 + 3 + 1 = 5 -> write 5, carry 0. If is_last=True -> is_halt=True
    tr2 = TransitionTriple.create(1, 3, 1, is_last=True)
    assert tr2.total_sum == 5
    assert tr2.write_digit == 5
    assert tr2.carry_out == 0
    assert tr2.is_last
    assert tr2.is_halt

    # 9 + 1 + 0 = 10 -> write 0, carry 1. If is_last=True -> is_halt=False (overflow carry needed!)
    tr3 = TransitionTriple.create(9, 1, 0, is_last=True)
    assert tr3.total_sum == 10
    assert tr3.write_digit == 0
    assert tr3.carry_out == 1
    assert tr3.is_last
    assert not tr3.is_halt


def test_format_transition_problem():
    tr = TransitionTriple.create(7, 6, 0, is_last=False)
    p, t = format_transition_problem(tr, include_halt_token=False)
    assert "Col: 7 + 6 + 0" in p
    assert "[LAST]" not in p
    assert "13 > 3 : 1<|eos|>" in t

    # Arm D with halt
    tr_halt = TransitionTriple.create(1, 3, 1, is_last=True)
    p2, t2 = format_transition_problem(tr_halt, include_halt_token=True)
    assert "Col: 1 + 3 + 1 [LAST]" in p2
    assert "5 > 5 : 0 | HALT<|eos|>" in t2

    # Arm D without halt (overflow carry)
    tr_cont = TransitionTriple.create(9, 1, 0, is_last=True)
    p3, t3 = format_transition_problem(tr_cont, include_halt_token=True)
    assert "Col: 9 + 1 + 0 [LAST]" in p3
    assert "10 > 0 : 1 | CONT<|eos|>" in t3


def test_parse_transition_completion():
    # Standard format: "13 > 3 : 1"
    res1 = parse_transition_completion("13 > 3 : 1\n")
    assert res1["is_valid"]
    assert res1["total_sum"] == 13
    assert res1["write_digit"] == 3
    assert res1["carry_out"] == 1
    assert not res1["is_halt"]

    # Short format: "3 : 1"
    res2 = parse_transition_completion("3 : 1")
    assert res2["is_valid"]
    assert res2["write_digit"] == 3
    assert res2["carry_out"] == 1

    # Format with HALT
    res3 = parse_transition_completion("5 > 5 : 0 | HALT<|eos|>")
    assert res3["is_valid"]
    assert res3["write_digit"] == 5
    assert res3["carry_out"] == 0
    assert res3["is_halt"]

    # Format with CONT
    res4 = parse_transition_completion("10 > 0 : 1 | CONT")
    assert res4["is_valid"]
    assert res4["write_digit"] == 0
    assert res4["carry_out"] == 1
    assert not res4["is_halt"]


def test_generate_all_transition_triples_partition():
    train_triples, test_triples = generate_all_transition_triples(include_last_flag=False)
    assert len(train_triples) == 160
    assert len(test_triples) == 40
    assert len(train_triples) + len(test_triples) == 200

    # Ensure complete disjointness
    train_keys = {(t.a, t.b, t.c_in) for t in train_triples}
    test_keys = {(t.a, t.b, t.c_in) for t in test_triples}
    assert len(train_keys.intersection(test_keys)) == 0

    # Test with include_last_flag=True
    train_triples_d, test_triples_d = generate_all_transition_triples(include_last_flag=True)
    assert len(train_triples_d) == 320
    assert len(test_triples_d) == 80


def test_transition_dataset_and_collate(trained_tokenizer: BPETokenizer):
    train_triples, _ = generate_all_transition_triples()
    ds = TransitionDataset(
        triples=train_triples[:20],
        tokenizer=trained_tokenizer,
        context_length=64,
        replay_texts=["The quick brown fox jumps over the lazy dog."],
        replay_ratio=0.1,
    )
    assert len(ds) >= 20
    inp, lab = ds[0]
    assert inp.shape == lab.shape
    # Ensure target-only loss: first tokens are -100
    assert (lab == -100).any()
    assert (lab != -100).any()

    # Test batch collate
    batch = [ds[0], ds[1]]
    b_in, b_lab = collate_transition_batch(batch, pad_token_id=0)
    assert b_in.shape[0] == 2
    assert b_lab.shape[0] == 2


class MockOracleTransitionModel(torch.nn.Module):
    """Mock model that generates exact token strings for column transition."""

    def __init__(self, tokenizer: BPETokenizer, learned_termination: bool = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.learned_termination = learned_termination

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, None]:
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.tokenizer.vocab_size)

        # Decode context so far
        text = self.tokenizer.decode(input_ids[0].tolist())
        import re
        m = re.search(r"Col:\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)", text)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            c = int(m.group(3))
            s = a + b + c
            w = s % 10
            c_out = s // 10
            is_last = "[LAST]" in text
            is_halt = bool(is_last and c_out == 0)

            if self.learned_termination:
                target_str = f"{s} > {w} : {c_out} | {'HALT' if is_halt else 'CONT'}\n"
            else:
                target_str = f"{s} > {w} : {c_out}\n"

"""Unit tests for M23 Recurrent Neural Arithmetic Transition Model."""

import re
import pytest
import torch

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.rules.arithmetic_dataset import ArithmeticProblem
from adaptive_mini_gpt.rules.recurrent_arithmetic import (
    TransitionDataset,
    TransitionTriple,
    collate_transition_batch,
    execute_recurrent_addition,
    format_transition_problem,
    generate_all_transition_triples,
    parse_transition_completion,
)


@pytest.fixture
def trained_tokenizer() -> BPETokenizer:
    tok = BPETokenizer()
    corpus = (
        "Col: 0 1 2 3 4 5 6 7 8 9 + = > : | HALT CONT [LAST] \n "
        "The quick brown fox jumps over the lazy dog. Paris London Tokyo science energy"
    )
    tok.train(corpus, target_vocab_size=120)
    return tok


def test_transition_triple_creation():
    # 7 + 6 + 0 = 13 -> write 3, carry 1, not halt
    tr = TransitionTriple.create(7, 6, 0, is_last=False)
    assert tr.total_sum == 13
    assert tr.write_digit == 3
    assert tr.carry_out == 1
    assert not tr.is_last
    assert not tr.is_halt

    # 1 + 3 + 1 = 5 -> write 5, carry 0. If is_last=True -> is_halt=True
    tr2 = TransitionTriple.create(1, 3, 1, is_last=True)
    assert tr2.total_sum == 5
    assert tr2.write_digit == 5
    assert tr2.carry_out == 0
    assert tr2.is_last
    assert tr2.is_halt

    # 9 + 1 + 0 = 10 -> write 0, carry 1. If is_last=True -> is_halt=False (overflow carry needed!)
    tr3 = TransitionTriple.create(9, 1, 0, is_last=True)
    assert tr3.total_sum == 10
    assert tr3.write_digit == 0
    assert tr3.carry_out == 1
    assert tr3.is_last
    assert not tr3.is_halt


def test_format_transition_problem():
    tr = TransitionTriple.create(7, 6, 0, is_last=False)
    p, t = format_transition_problem(tr, include_halt_token=False)
    assert "Col: 7 + 6 + 0" in p
    assert "[LAST]" not in p
    assert "13 > 3 : 1<|eos|>" in t

    # Arm D with halt
    tr_halt = TransitionTriple.create(1, 3, 1, is_last=True)
    p2, t2 = format_transition_problem(tr_halt, include_halt_token=True)
    assert "Col: 1 + 3 + 1 [LAST]" in p2
    assert "5 > 5 : 0 | HALT<|eos|>" in t2

    # Arm D without halt (overflow carry)
    tr_cont = TransitionTriple.create(9, 1, 0, is_last=True)
    p3, t3 = format_transition_problem(tr_cont, include_halt_token=True)
    assert "Col: 9 + 1 + 0 [LAST]" in p3
    assert "10 > 0 : 1 | CONT<|eos|>" in t3


def test_parse_transition_completion():
    # Standard format: "13 > 3 : 1"
    res1 = parse_transition_completion("13 > 3 : 1\n")
    assert res1["is_valid"]
    assert res1["total_sum"] == 13
    assert res1["write_digit"] == 3
    assert res1["carry_out"] == 1
    assert not res1["is_halt"]

    # Short format: "3 : 1"
    res2 = parse_transition_completion("3 : 1")
    assert res2["is_valid"]
    assert res2["write_digit"] == 3
    assert res2["carry_out"] == 1

    # Format with HALT
    res3 = parse_transition_completion("5 > 5 : 0 | HALT<|eos|>")
    assert res3["is_valid"]
    assert res3["write_digit"] == 5
    assert res3["carry_out"] == 0
    assert res3["is_halt"]

    # Format with CONT
    res4 = parse_transition_completion("10 > 0 : 1 | CONT")
    assert res4["is_valid"]
    assert res4["write_digit"] == 0
    assert res4["carry_out"] == 1
    assert not res4["is_halt"]


def test_generate_all_transition_triples_partition():
    train_triples, test_triples = generate_all_transition_triples(include_last_flag=False)
    assert len(train_triples) == 160
    assert len(test_triples) == 40
    assert len(train_triples) + len(test_triples) == 200

    # Ensure complete disjointness
    train_keys = {(t.a, t.b, t.c_in) for t in train_triples}
    test_keys = {(t.a, t.b, t.c_in) for t in test_triples}
    assert len(train_keys.intersection(test_keys)) == 0

    # Test with include_last_flag=True
    train_triples_d, test_triples_d = generate_all_transition_triples(include_last_flag=True)
    assert len(train_triples_d) == 320
    assert len(test_triples_d) == 80


def test_transition_dataset_and_collate(trained_tokenizer: BPETokenizer):
    train_triples, _ = generate_all_transition_triples()
    ds = TransitionDataset(
        triples=train_triples[:20],
        tokenizer=trained_tokenizer,
        context_length=64,
        replay_texts=["The quick brown fox jumps over the lazy dog."],
        replay_ratio=0.1,
    )
    assert len(ds) >= 20
    inp, lab = ds[0]
    assert inp.shape == lab.shape
    # Ensure target-only loss: first tokens are -100
    assert (lab == -100).any()
    assert (lab != -100).any()

    # Test batch collate
    batch = [ds[0], ds[1]]
    b_in, b_lab = collate_transition_batch(batch, pad_token_id=0)
    assert b_in.shape[0] == 2
    assert b_lab.shape[0] == 2


class MockOracleTransitionModel(torch.nn.Module):
    """Mock model that generates exact token strings for column transition."""

    def __init__(self, tokenizer: BPETokenizer, learned_termination: bool = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.learned_termination = learned_termination

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, None]:
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.tokenizer.vocab_size)

        text = self.tokenizer.decode(input_ids[0].tolist())
        m = re.search(r"Col:\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)", text)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            c = int(m.group(3))
            s = a + b + c
            w = s % 10
            c_out = s // 10
            is_last = "[LAST]" in text
            is_halt = bool(is_last and c_out == 0)

            if self.learned_termination:
                target_str = f"{s} > {w} : {c_out} | {'HALT' if is_halt else 'CONT'}\n"
            else:
                target_str = f"{s} > {w} : {c_out}\n"

            target_toks = self.tokenizer.encode(target_str)
            assistant_part = text.split("<|assistant|>\n")[-1]
            gen_toks = self.tokenizer.encode(assistant_part)
            if len(gen_toks) < len(target_toks):
                logits[0, -1, target_toks[len(gen_toks)]] = 100.0
                return logits, None

        logits[0, -1, self.tokenizer.eos_id] = 100.0
        return logits, None


def test_execute_recurrent_addition_mock_lengths(trained_tokenizer: BPETokenizer):
    mock_model = MockOracleTransitionModel(trained_tokenizer, learned_termination=False)
    device = torch.device("cpu")

    # Test 2-digit problem: 47 + 85 = 132
    p2 = ArithmeticProblem.create(47, 85)
    res2 = execute_recurrent_addition(mock_model, trained_tokenizer, p2, device, closed_loop=True, learned_termination=False)
    assert res2["is_exact"]
    assert res2["predicted_result"] == 132
    assert res2["all_transitions_correct"]
    assert res2["all_carries_correct"]

    # Test 4-digit problem: 4827 + 8596 = 13423
    p4 = ArithmeticProblem.create(4827, 8596)
    res4 = execute_recurrent_addition(mock_model, trained_tokenizer, p4, device, closed_loop=True, learned_termination=False)
    assert res4["is_exact"]
    assert res4["predicted_result"] == 13423

    # Test 8-digit problem: 98765432 + 12345678 = 111111110
    p8 = ArithmeticProblem.create(98765432, 12345678)
    res8 = execute_recurrent_addition(mock_model, trained_tokenizer, p8, device, closed_loop=True, learned_termination=False)
    assert res8["is_exact"]
    assert res8["predicted_result"] == 111111110


def test_execute_recurrent_addition_learned_termination(trained_tokenizer: BPETokenizer):
    mock_model = MockOracleTransitionModel(trained_tokenizer, learned_termination=True)
    device = torch.device("cpu")

    # Test 3-digit problem without overflow: 123 + 234 = 357
    p3 = ArithmeticProblem.create(123, 234)
    res3 = execute_recurrent_addition(mock_model, trained_tokenizer, p3, device, closed_loop=True, learned_termination=True)
    assert res3["is_exact"]
    assert res3["predicted_result"] == 357
    assert res3["stopping_correct"]

    # Test 3-digit problem with overflow: 999 + 1 = 1000
    p_ovf = ArithmeticProblem.create(999, 1)
    res_ovf = execute_recurrent_addition(mock_model, trained_tokenizer, p_ovf, device, closed_loop=True, learned_termination=True)
    assert res_ovf["is_exact"]
    assert res_ovf["predicted_result"] == 1000
    assert res_ovf["stopping_correct"]
