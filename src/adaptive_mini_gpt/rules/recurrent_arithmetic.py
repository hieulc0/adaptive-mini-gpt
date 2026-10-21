"""M23: Recurrent Neural Arithmetic Transition Model and Length Extrapolation Harness.

Implements a position-independent local column-addition transition:
    f_theta(a_i, b_i, c_i) -> (d_i, c_{i+1})
and unrolls it recurrently across arbitrary operand lengths (2 to 8+ digits)
without external symbolic calculators.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.arithmetic_dataset import ArithmeticProblem

logger = logging.getLogger("recurrent_arithmetic")


@dataclass(frozen=True)
class TransitionTriple:
    """Elementary one-column addition transition state."""

    a: int
    b: int
    c_in: int
    total_sum: int
    write_digit: int
    carry_out: int
    is_last: bool = False
    is_halt: bool = False

    @classmethod
    def create(
        cls,
        a: int,
        b: int,
        c_in: int,
        is_last: bool = False,
    ) -> TransitionTriple:
        s = a + b + c_in
        w = s % 10
        c_out = s // 10
        # Halts if this is marked as last AND there is no outgoing carry
        is_halt = bool(is_last and c_out == 0)
        return cls(
            a=a,
            b=b,
            c_in=c_in,
            total_sum=s,
            write_digit=w,
            carry_out=c_out,
            is_last=is_last,
            is_halt=is_halt,
        )


def format_transition_problem(
    triple: TransitionTriple,
    include_halt_token: bool = False,
) -> tuple[str, str]:
    """Format single-step column transition prompt and target."""
    last_tag = " [LAST]" if (include_halt_token and triple.is_last) else ""
    prompt = f"<|user|>\nCol: {triple.a} + {triple.b} + {triple.c_in}{last_tag}\n<|assistant|>\n"

    if include_halt_token:
        stop_tag = " | HALT" if triple.is_halt else " | CONT"
        target = f"{triple.total_sum} > {triple.write_digit} : {triple.carry_out}{stop_tag}<|eos|>\n"
    else:
        target = f"{triple.total_sum} > {triple.write_digit} : {triple.carry_out}<|eos|>\n"

    return prompt, target


def parse_transition_completion(raw_text: str) -> dict[str, Any]:
    """Parse output from local transition step."""
    clean = raw_text.strip()
    write_digit: int | None = None
    carry_out: int | None = None
    total_sum: int | None = None
    is_halt: bool = False
    is_valid = False

    # Check for HALT / STOP marker
    if "HALT" in clean or "STOP" in clean:
        is_halt = True

    # Look for sum > write : carry pattern: e.g. "13 > 3 : 1" or "3 : 1"
    m = re.search(r"(?:(\d+)\s*>\s*)?(\d+)\s*:\s*(\d+)", clean)
    if m:
        if m.group(1) is not None:
            total_sum = int(m.group(1))
        write_digit = int(m.group(2))
        carry_out = int(m.group(3))
        is_valid = True
    else:
        # Fallback: look for two numbers separated by colon or space
        m2 = re.search(r"(\d+)\s*[:,\s]\s*(\d+)", clean)
        if m2:
            write_digit = int(m2.group(1)) % 10
            carry_out = int(m2.group(2))
            is_valid = True

    return {
        "is_valid": is_valid,
        "write_digit": write_digit,
        "carry_out": carry_out,
        "total_sum": total_sum,
        "is_halt": is_halt,
        "raw_text": raw_text[:80],
    }


class TransitionDataset(Dataset):
    """PyTorch Dataset for training local transition model with target-only loss and replay."""

    def __init__(
        self,
        triples: list[TransitionTriple],
        tokenizer: BPETokenizer,
        context_length: int = 128,
        include_halt_token: bool = False,
        replay_texts: list[str] | None = None,
        replay_ratio: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.triples = triples
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.include_halt_token = include_halt_token

        self.input_ids_list: list[Tensor] = []
        self.labels_list: list[Tensor] = []

        for tr in self.triples:
            p_str, t_str = format_transition_problem(tr, include_halt_token=self.include_halt_token)
            p_toks = self.tokenizer.encode(p_str)
            t_toks = self.tokenizer.encode(t_str)

            full_toks = p_toks + t_toks
            if len(full_toks) > self.context_length:
                full_toks = full_toks[: self.context_length]
                p_len = min(len(p_toks), self.context_length)
            else:
                p_len = len(p_toks)

            labels = [-100] * p_len + full_toks[p_len:]
            self.input_ids_list.append(torch.tensor(full_toks, dtype=torch.long))
            self.labels_list.append(torch.tensor(labels, dtype=torch.long))

        # Replay mixing with independent corpus
        if replay_texts and replay_ratio > 0.0:
            rng = random.Random(seed)
            n_rep = int(len(self.triples) * replay_ratio)
            for _ in range(n_rep):
                txt = rng.choice(replay_texts)
                toks = self.tokenizer.encode(txt)
                if len(toks) > self.context_length:
                    toks = toks[: self.context_length]
                if len(toks) > 1:
                    self.input_ids_list.append(torch.tensor(toks, dtype=torch.long))
                    self.labels_list.append(torch.tensor(list(toks), dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids_list)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.input_ids_list[idx], self.labels_list[idx]


def collate_transition_batch(
    batch: list[tuple[Tensor, Tensor]],
    pad_token_id: int = 0,
) -> tuple[Tensor, Tensor]:
    """Collate variable length transition sequences."""
    max_len = max(len(x[0]) for x in batch)
    b_inputs = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    b_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (inp, lab) in enumerate(batch):
        b_inputs[i, : len(inp)] = inp
        b_labels[i, : len(lab)] = lab

    return b_inputs, b_labels


def generate_all_transition_triples(
    include_last_flag: bool = False,
) -> tuple[list[TransitionTriple], list[TransitionTriple]]:
    """Partition all 200 possible (a, b, c_in) triples into train (160) and held-out test (40).

    If include_last_flag is True, each (a, b, c_in) is also paired with is_last=True,
    yielding 400 total combinations (320 train, 80 test).
    """
    base_triples: list[tuple[int, int, int]] = []
    for a in range(10):
        for b in range(10):
            for c in range(2):
                base_triples.append((a, b, c))

    # Deterministic split: seed 42
    rng = random.Random(42)
    shuffled_bases = list(base_triples)
    rng.shuffle(shuffled_bases)

    # 80% train (160), 20% test (40)
    train_bases = shuffled_bases[:160]
    test_bases = shuffled_bases[160:]

    train_triples: list[TransitionTriple] = []
    test_triples: list[TransitionTriple] = []

    for a, b, c in train_bases:
        train_triples.append(TransitionTriple.create(a, b, c, is_last=False))
        if include_last_flag:
            train_triples.append(TransitionTriple.create(a, b, c, is_last=True))

    for a, b, c in test_bases:
        test_triples.append(TransitionTriple.create(a, b, c, is_last=False))
        if include_last_flag:
            test_triples.append(TransitionTriple.create(a, b, c, is_last=True))

    return train_triples, test_triples


def build_transition_training_dataset(
    train_triples: list[TransitionTriple],
    num_samples: int = 1200,
    seed: int = 1023,
) -> list[TransitionTriple]:
    """Sample diverse training transitions matching balanced carry proportions."""
    rng = random.Random(seed)
    samples: list[TransitionTriple] = []
    while len(samples) < num_samples:
        samples.append(rng.choice(train_triples))
    return samples


def evaluate_transition_accuracy(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    triples: list[TransitionTriple],
    device: torch.device,
    include_halt_token: bool = False,
) -> dict[str, float]:
    """Evaluate exact step accuracy on a suite of transition triples."""
    model.eval()
    correct_write = 0
    correct_carry = 0
    correct_halt = 0
    correct_exact = 0
    total = len(triples)

    with torch.no_grad():
        for tr in triples:
            prompt_str, _ = format_transition_problem(tr, include_halt_token=include_halt_token)
            p_toks = tokenizer.encode(prompt_str)

            gen_tokens = list(p_toks)
            for _ in range(24):
                curr_ids = torch.tensor([gen_tokens], dtype=torch.long, device=device)
                logits, _ = model(curr_ids)
                next_tok = int(torch.argmax(logits[0, -1, :]).item())
                gen_tokens.append(next_tok)
                if next_tok == tokenizer.eos_id:
                    break
                decoded = tokenizer.decode(gen_tokens[len(p_toks) :])
                if "\n" in decoded:
                    break

            raw_out = tokenizer.decode(gen_tokens[len(p_toks) :])
            parsed = parse_transition_completion(raw_out)

            w_ok = (parsed["write_digit"] == tr.write_digit)
            c_ok = (parsed["carry_out"] == tr.carry_out)
            h_ok = (parsed["is_halt"] == tr.is_halt) if include_halt_token else True
            exact_ok = w_ok and c_ok and h_ok

            if w_ok:
                correct_write += 1
            if c_ok:
                correct_carry += 1
            if h_ok:
                correct_halt += 1
            if exact_ok:
                correct_exact += 1

    return {
        "transition_exact_acc": round(correct_exact / max(1, total) * 100.0, 2),
        "write_digit_acc": round(correct_write / max(1, total) * 100.0, 2),
        "carry_out_acc": round(correct_carry / max(1, total) * 100.0, 2),
        "halt_acc": round(correct_halt / max(1, total) * 100.0, 2),
        "total_evaluated": total,
    }


def execute_recurrent_addition(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    prob: ArithmeticProblem,
    device: torch.device,
    closed_loop: bool = True,
    learned_termination: bool = False,
    max_steps: int = 15,
) -> dict[str, Any]:
    """Execute addition by applying the shared neural transition sequentially across columns.

    Scientific Principle:
    Zero Python arithmetic is performed during inference:
    Every digit and carry is predicted by model(Col: a + b + c).
    Python solely performs indexing and output assembly.
    """
    model.eval()
    str_a = str(prob.a)
    str_b = str(prob.b)

    # Digits from least significant (units) to most significant
    digits_a = [int(d) for d in reversed(str_a)]
    digits_b = [int(d) for d in reversed(str_b)]
    num_cols = max(len(digits_a), len(digits_b))

    # Expected stopping column: if overflow occurs, stops at num_cols, else at num_cols - 1
    # Check ground truth column sum at final column to know expected stop step
    gt_has_overflow = False
    temp_c = 0
    for i in range(num_cols):
        da = digits_a[i] if i < len(digits_a) else 0
        db = digits_b[i] if i < len(digits_b) else 0
        s = da + db + temp_c
        temp_c = s // 10
    if temp_c > 0:
        gt_has_overflow = True
    expected_stop_step = num_cols if gt_has_overflow else (num_cols - 1)

    curr_carry = 0
    emitted_digits: list[int] = []
    transition_trace: list[dict[str, Any]] = []

    all_transitions_correct = True
    all_carries_correct = True
    stopped_at_step: int | None = None
    stopping_correct = False

    t0 = time.perf_counter()
    step_count = 0

    if not learned_termination:
        # Arm C: Host-controlled loop over operand columns + explicit overflow check
        for i in range(num_cols):
            step_count += 1
            d_a = digits_a[i] if i < len(digits_a) else 0
            d_b = digits_b[i] if i < len(digits_b) else 0

            # Construct single-step prompt
            prompt_str = f"<|user|>\nCol: {d_a} + {d_b} + {curr_carry}\n<|assistant|>\n"
            prompt_tokens = tokenizer.encode(prompt_str)

            gen_tokens = list(prompt_tokens)
            with torch.no_grad():
                for _ in range(24):
                    curr_ids = torch.tensor([gen_tokens], dtype=torch.long, device=device)
                    logits, _ = model(curr_ids)
                    next_tok = int(torch.argmax(logits[0, -1, :]).item())
                    gen_tokens.append(next_tok)
                    if next_tok == tokenizer.eos_id:
                        break
                    decoded_text = tokenizer.decode(gen_tokens[len(prompt_tokens) :])
                    if "\n" in decoded_text:
                        break

            completion = tokenizer.decode(gen_tokens[len(prompt_tokens) :])
            parsed = parse_transition_completion(completion)

            pred_w = parsed["write_digit"]
            pred_c = parsed["carry_out"]

            # Ground truth step values
            gt_sum = d_a + d_b + curr_carry
            gt_w = gt_sum % 10
            gt_c = gt_sum // 10

            step_correct = (pred_w == gt_w)
            carry_correct = (pred_c == gt_c)

            if not step_correct:
                all_transitions_correct = False
            if not carry_correct:
                all_carries_correct = False

            emitted_digits.append(pred_w if pred_w is not None else 0)

            transition_trace.append({
                "col": i,
                "d_a": d_a,
                "d_b": d_b,
                "c_in": curr_carry,
                "pred_w": pred_w,
                "pred_c": pred_c,
                "gt_w": gt_w,
                "gt_c": gt_c,
                "step_correct": step_correct,
                "carry_correct": carry_correct,
                "completion": completion.strip(),
            })

            if closed_loop:
                curr_carry = pred_c if pred_c is not None else 0
            else:
                curr_carry = gt_c

        # Overflow column check
        if curr_carry > 0 and step_count < max_steps:
            step_count += 1
            prompt_str = f"<|user|>\nCol: 0 + 0 + {curr_carry}\n<|assistant|>\n"
            prompt_tokens = tokenizer.encode(prompt_str)

            gen_tokens = list(prompt_tokens)
            with torch.no_grad():
                for _ in range(24):
                    curr_ids = torch.tensor([gen_tokens], dtype=torch.long, device=device)
                    logits, _ = model(curr_ids)
                    next_tok = int(torch.argmax(logits[0, -1, :]).item())
                    gen_tokens.append(next_tok)
                    if next_tok == tokenizer.eos_id:
                        break
                    decoded_text = tokenizer.decode(gen_tokens[len(prompt_tokens) :])
                    if "\n" in decoded_text:
                        break

            completion = tokenizer.decode(gen_tokens[len(prompt_tokens) :])
            parsed = parse_transition_completion(completion)
            pred_w = parsed["write_digit"]
            pred_c = parsed["carry_out"]

            gt_sum = curr_carry
            gt_w = gt_sum % 10
            gt_c = 0

            step_correct = (pred_w == gt_w)
            carry_correct = (pred_c == gt_c)

            if not step_correct:
                all_transitions_correct = False
            if not carry_correct:
                all_carries_correct = False

            emitted_digits.append(pred_w if pred_w is not None else 0)

            transition_trace.append({
                "col": num_cols,
                "d_a": 0,
                "d_b": 0,
                "c_in": curr_carry,
                "pred_w": pred_w,
                "pred_c": pred_c,
                "gt_w": gt_w,
                "gt_c": gt_c,
                "step_correct": step_correct,
                "carry_correct": carry_correct,
                "completion": completion.strip(),
            })

        stopping_correct = True
        stopped_at_step = step_count - 1

    else:
        # Arm D: Learned termination
        for i in range(max_steps):
            step_count += 1
            if i < num_cols:
                d_a = digits_a[i] if i < len(digits_a) else 0
                d_b = digits_b[i] if i < len(digits_b) else 0
                is_last = (i == num_cols - 1)
            else:
                d_a = 0
                d_b = 0
                is_last = True

            last_tag = " [LAST]" if is_last else ""
            prompt_str = f"<|user|>\nCol: {d_a} + {d_b} + {curr_carry}{last_tag}\n<|assistant|>\n"
            prompt_tokens = tokenizer.encode(prompt_str)

            gen_tokens = list(prompt_tokens)
            with torch.no_grad():
                for _ in range(24):
                    curr_ids = torch.tensor([gen_tokens], dtype=torch.long, device=device)
                    logits, _ = model(curr_ids)
                    next_tok = int(torch.argmax(logits[0, -1, :]).item())
                    gen_tokens.append(next_tok)
                    if next_tok == tokenizer.eos_id:
                        break
                    decoded_text = tokenizer.decode(gen_tokens[len(prompt_tokens) :])
                    if "\n" in decoded_text:
                        break

            completion = tokenizer.decode(gen_tokens[len(prompt_tokens) :])
            parsed = parse_transition_completion(completion)

            pred_w = parsed["write_digit"]
            pred_c = parsed["carry_out"]
            is_halt = parsed["is_halt"]

            # Ground truth
            gt_sum = d_a + d_b + curr_carry
            gt_w = gt_sum % 10
            gt_c = gt_sum // 10

            step_correct = (pred_w == gt_w)
            carry_correct = (pred_c == gt_c)

            if not step_correct:
                all_transitions_correct = False
            if not carry_correct:
                all_carries_correct = False

            emitted_digits.append(pred_w if pred_w is not None else 0)

            transition_trace.append({
                "col": i,
                "d_a": d_a,
                "d_b": d_b,
                "c_in": curr_carry,
                "is_last": is_last,
                "pred_w": pred_w,
                "pred_c": pred_c,
                "is_halt": is_halt,
                "gt_w": gt_w,
                "gt_c": gt_c,
                "step_correct": step_correct,
                "carry_correct": carry_correct,
                "completion": completion.strip(),
            })

            if closed_loop:
                curr_carry = pred_c if pred_c is not None else 0
            else:
                curr_carry = gt_c

            if is_halt:
                stopped_at_step = i
                break

        if stopped_at_step is None:
            stopped_at_step = max_steps - 1

        stopping_correct = (stopped_at_step == expected_stop_step)

    duration_ms = (time.perf_counter() - t0) * 1000.0

    # Reconstruct final multi-digit number (reverse digits to MSD)
    res_str = "".join(str(d) for d in reversed(emitted_digits))
    pred_res = int(res_str) if res_str.isdigit() else None
    is_exact = (pred_res is not None and pred_res == prob.result)
    if learned_termination and not stopping_correct:
        is_exact = False

    return {
        "problem_a": prob.a,
        "problem_b": prob.b,
        "expected_result": prob.result,
        "predicted_result": pred_res,
        "is_exact": is_exact,
        "all_transitions_correct": all_transitions_correct,
        "all_carries_correct": all_carries_correct,
        "stopping_correct": stopping_correct,
        "stopped_at_step": stopped_at_step,
        "expected_stop_step": expected_stop_step,
        "total_columns": step_count,
        "duration_ms": round(duration_ms, 2),
        "latency_per_digit_ms": round(duration_ms / max(1, step_count), 2),
        "trace": transition_trace,
    }
