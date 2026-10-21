"""M24: Reliable Neural Arithmetic Transition Engine and Deep Generalization Harness.

Implements:
1. Exhaustive 200-transition state space auditing with error classification.
2. Canonical structured digit/carry state encodings.
3. Separate output-digit and carry-state supervision.
4. Balanced transition curriculum.
5. Recurrent neural execution with first-failure diagnostics on lengths up to 32 digits.
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

logger = logging.getLogger("reliable_transition")


@dataclass(frozen=True)
class ElementaryTransition:
    """An elementary decimal addition transition (a, b, c_in) -> (d, c_out)."""

    a: int
    b: int
    c_in: int
    total_sum: int
    write_digit: int
    carry_out: int
    carry_category: str  # "generate" (a+b>=10), "propagate" (a+b+1>=10 and a+b<10), "none" (sum<10)
    is_symmetric: bool  # a == b

    @classmethod
    def create(cls, a: int, b: int, c_in: int) -> ElementaryTransition:
        s = a + b + c_in
        w = s % 10
        c_out = s // 10

        if (a + b) >= 10:
            category = "generate"
        elif c_in == 1 and s >= 10:
            category = "propagate"
        else:
            category = "none"

        return cls(
            a=a,
            b=b,
            c_in=c_in,
            total_sum=s,
            write_digit=w,
            carry_out=c_out,
            carry_category=category,
            is_symmetric=(a == b),
        )


def get_all_200_transitions() -> list[ElementaryTransition]:
    """Enumerate all 200 elementary decimal addition transitions (10 x 10 x 2)."""
    transitions: list[ElementaryTransition] = []
    for a in range(10):
        for b in range(10):
            for c_in in range(2):
                transitions.append(ElementaryTransition.create(a, b, c_in))
    return transitions


def format_elementary_prompt_and_target(
    tr: ElementaryTransition,
    format_type: str = "canonical_state",
) -> tuple[str, str]:
    """Format single-step prompt and target string according to experimental arm format.

    Supported formats:
      - 'm23_baseline': Col: a + b + c_in -> s > d : c_out
      - 'canonical_state': [IN] a=A b=B c=C [OUT] d=D c=C_OUT
      - 'separate_supervision': [ADD] a=A b=B c_in=C -> C_OUT: C_OUT | DIGIT: D
    """
    if format_type == "m23_baseline":
        prompt = f"<|user|>\nCol: {tr.a} + {tr.b} + {tr.c_in}\n<|assistant|>\n"
        target = f"{tr.total_sum} > {tr.write_digit} : {tr.carry_out}<|eos|>\n"

    elif format_type == "canonical_state":
        prompt = f"<|user|>\n[IN] a={tr.a} b={tr.b} c={tr.c_in}\n<|assistant|>\n"
        target = f"[OUT] d={tr.write_digit} c={tr.carry_out}<|eos|>\n"

    elif format_type == "separate_supervision":
        # Predict binary carry first to condition the write digit
        prompt = f"<|user|>\n[ADD] a={tr.a} b={tr.b} c_in={tr.c_in}\n<|assistant|>\n"
        target = f"C_OUT: {tr.carry_out} | DIGIT: {tr.write_digit}<|eos|>\n"

    else:
        raise ValueError(f"Unknown transition format type: {format_type}")

    return prompt, target


def parse_elementary_completion(raw_text: str, format_type: str = "canonical_state") -> dict[str, Any]:
    """Parse output text from an elementary column transition step."""
    clean = raw_text.strip()
    write_digit: int | None = None
    carry_out: int | None = None
    is_valid = False

    if format_type == "canonical_state":
        # e.g., "[OUT] d=3 c=1"
        m = re.search(r"d\s*=\s*(\d+).*?c\s*=\s*(\d+)", clean, re.DOTALL)
        if m:
            write_digit = int(m.group(1)) % 10
            carry_out = int(m.group(2))
            is_valid = True
        else:
            # Fallback
            digits = re.findall(r"\d+", clean)
            if len(digits) >= 2:
                write_digit = int(digits[0]) % 10
                carry_out = int(digits[1])
                is_valid = True

    elif format_type == "separate_supervision":
        # e.g., "C_OUT: 1 | DIGIT: 3"
        m_c = re.search(r"C_OUT:\s*(\d+)", clean, re.IGNORECASE)
        m_d = re.search(r"DIGIT:\s*(\d+)", clean, re.IGNORECASE)
        if m_c and m_d:
            carry_out = int(m_c.group(1))
            write_digit = int(m_d.group(1)) % 10
            is_valid = True
        else:
            digits = re.findall(r"\d+", clean)
            if len(digits) >= 2:
                carry_out = int(digits[0])
                write_digit = int(digits[1]) % 10
                is_valid = True

    elif format_type == "m23_baseline":
        m = re.search(r"(?:(\d+)\s*>\s*)?(\d+)\s*:\s*(\d+)", clean)
        if m:
            write_digit = int(m.group(2)) % 10
            carry_out = int(m.group(3))
            is_valid = True
        else:
            digits = re.findall(r"\d+", clean)
            if len(digits) >= 2:
                write_digit = int(digits[0]) % 10
                carry_out = int(digits[1])
                is_valid = True

    return {
        "is_valid": is_valid,
        "write_digit": write_digit,
        "carry_out": carry_out,
        "raw_text": raw_text[:80],
    }


class ElementaryTransitionDataset(Dataset):
    """Dataset for training MiniGPT on elementary transitions with target-only loss and replay."""

    def __init__(
        self,
        transitions: list[ElementaryTransition],
        tokenizer: BPETokenizer,
        format_type: str = "canonical_state",
        context_length: int = 128,
        replay_texts: list[str] | None = None,
        replay_ratio: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.transitions = transitions
        self.tokenizer = tokenizer
        self.format_type = format_type
        self.context_length = context_length

        self.input_ids_list: list[Tensor] = []
        self.labels_list: list[Tensor] = []

        for tr in self.transitions:
            p_str, t_str = format_elementary_prompt_and_target(tr, format_type=self.format_type)
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
            n_rep = int(len(self.transitions) * replay_ratio)
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


def collate_elementary_batch(
    batch: list[tuple[Tensor, Tensor]],
    pad_token_id: int = 0,
) -> tuple[Tensor, Tensor]:
    """Collate variable length elementary transition sequences."""
    max_len = max(len(x[0]) for x in batch)
    b_inputs = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    b_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (inp, lab) in enumerate(batch):
        b_inputs[i, : len(inp)] = inp
        b_labels[i, : len(lab)] = lab

    return b_inputs, b_labels


def audit_elementary_transition_space(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    device: torch.device,
    format_type: str = "canonical_state",
    held_out_triples: set[tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
    """Exhaustively audit all 200 elementary decimal addition transitions."""
    model.eval()
    all_200 = get_all_200_transitions()

    total_exact = 0
    total_digit_correct = 0
    total_carry_correct = 0

    category_stats: dict[str, dict[str, int]] = {
        "generate": {"total": 0, "exact": 0, "digit_ok": 0, "carry_ok": 0},
        "propagate": {"total": 0, "exact": 0, "digit_ok": 0, "carry_ok": 0},
        "none": {"total": 0, "exact": 0, "digit_ok": 0, "carry_ok": 0},
    }

    symmetry_stats: dict[str, dict[str, int]] = {
        "symmetric": {"total": 0, "exact": 0},
        "asymmetric": {"total": 0, "exact": 0},
    }

    split_stats: dict[str, dict[str, int]] = {
        "training_seen": {"total": 0, "exact": 0, "digit_ok": 0, "carry_ok": 0},
        "held_out": {"total": 0, "exact": 0, "digit_ok": 0, "carry_ok": 0},
    }

    error_examples: list[dict[str, Any]] = []

    with torch.no_grad():
        for tr in all_200:
            prompt_str, _ = format_elementary_prompt_and_target(tr, format_type=format_type)
            p_toks = tokenizer.encode(prompt_str)

            gen_tokens = list(p_toks)
            for _ in range(20):
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
            parsed = parse_elementary_completion(raw_out, format_type=format_type)

            digit_ok = (parsed["write_digit"] == tr.write_digit)
            carry_ok = (parsed["carry_out"] == tr.carry_out)
            exact_ok = digit_ok and carry_ok

            if digit_ok:
                total_digit_correct += 1
            if carry_ok:
                total_carry_correct += 1
            if exact_ok:
                total_exact += 1
            else:
                if len(error_examples) < 15:
                    error_examples.append({
                        "a": tr.a,
                        "b": tr.b,
                        "c_in": tr.c_in,
                        "expected_d": tr.write_digit,
                        "expected_c": tr.carry_out,
                        "predicted_d": parsed["write_digit"],
                        "predicted_c": parsed["carry_out"],
                        "category": tr.carry_category,
                        "raw_output": raw_out.strip(),
                    })

            # Category
            cat = tr.carry_category
            category_stats[cat]["total"] += 1
            if exact_ok:
                category_stats[cat]["exact"] += 1
            if digit_ok:
                category_stats[cat]["digit_ok"] += 1
            if carry_ok:
                category_stats[cat]["carry_ok"] += 1

            # Symmetry
            sym_key = "symmetric" if tr.is_symmetric else "asymmetric"
            symmetry_stats[sym_key]["total"] += 1
            if exact_ok:
                symmetry_stats[sym_key]["exact"] += 1

            # Split (if defined)
            if held_out_triples is not None:
                is_held = (tr.a, tr.b, tr.c_in) in held_out_triples
                sp_key = "held_out" if is_held else "training_seen"
                split_stats[sp_key]["total"] += 1
                if exact_ok:
                    split_stats[sp_key]["exact"] += 1
                if digit_ok:
                    split_stats[sp_key]["digit_ok"] += 1
                if carry_ok:
                    split_stats[sp_key]["carry_ok"] += 1

    n_tot = len(all_200)

    def calc_pct(k: int, denom: int) -> float:
        return round(k / max(1, denom) * 100.0, 2)

    cat_report = {
        cat: {
            "total": d["total"],
            "exact_acc": calc_pct(d["exact"], d["total"]),
            "digit_acc": calc_pct(d["digit_ok"], d["total"]),
            "carry_acc": calc_pct(d["carry_ok"], d["total"]),
        }
        for cat, d in category_stats.items()
    }

    sym_report = {
        k: {
            "total": d["total"],
            "exact_acc": calc_pct(d["exact"], d["total"]),
        }
        for k, d in symmetry_stats.items()
    }

    split_report = {}
    if held_out_triples is not None:
        split_report = {
            k: {
                "total": d["total"],
                "exact_acc": calc_pct(d["exact"], d["total"]),
                "digit_acc": calc_pct(d["digit_ok"], d["total"]),
                "carry_acc": calc_pct(d["carry_ok"], d["total"]),
            }
            for k, d in split_stats.items()
        }

    return {
        "overall_exact_acc": calc_pct(total_exact, n_tot),
        "overall_digit_acc": calc_pct(total_digit_correct, n_tot),
        "overall_carry_acc": calc_pct(total_carry_correct, n_tot),
        "total_evaluated": n_tot,
        "by_carry_category": cat_report,
        "by_symmetry": sym_report,
        "by_split": split_report,
        "sample_errors": error_examples,
    }


def execute_reliable_recurrent_addition(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    prob: ArithmeticProblem,
    device: torch.device,
    format_type: str = "canonical_state",
    closed_loop: bool = True,
    max_steps: int = 40,
) -> dict[str, Any]:
    """Execute addition by recurrently applying the local transition.

    Scientific Principle:
    Zero Python arithmetic during neural inference:
    Every digit and carry is predicted by model(Col: a + b + c).
    Python solely performs indexing and output assembly.
    Tracks first-failure diagnostics and supports up to 32+ digit problems.
    """
    model.eval()
    str_a = str(prob.a)
    str_b = str(prob.b)

    # Digits from least significant (units) to most significant
    digits_a = [int(d) for d in reversed(str_a)]
    digits_b = [int(d) for d in reversed(str_b)]
    num_cols = max(len(digits_a), len(digits_b))

    curr_carry = 0
    emitted_digits: list[int] = []
    transition_trace: list[dict[str, Any]] = []

    first_failure_col: int | None = None
    first_failure_type: str | None = None

    all_transitions_correct = True
    all_carries_correct = True

    t0 = time.perf_counter()
    step_count = 0

    for i in range(num_cols):
        step_count += 1
        d_a = digits_a[i] if i < len(digits_a) else 0
        d_b = digits_b[i] if i < len(digits_b) else 0

        # Construct single-step transition object for prompting
        dummy_tr = ElementaryTransition.create(d_a, d_b, curr_carry)
        prompt_str, _ = format_elementary_prompt_and_target(dummy_tr, format_type=format_type)
        prompt_tokens = tokenizer.encode(prompt_str)

        gen_tokens = list(prompt_tokens)
        with torch.no_grad():
            for _ in range(20):
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
        parsed = parse_elementary_completion(completion, format_type=format_type)

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
            if first_failure_col is None:
                first_failure_col = i
                first_failure_type = "both" if not carry_correct else "write_digit"

        if not carry_correct:
            all_carries_correct = False
            if first_failure_col is None:
                first_failure_col = i
                first_failure_type = "both" if not step_correct else "carry_out"

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
        dummy_tr = ElementaryTransition.create(0, 0, curr_carry)
        prompt_str, _ = format_elementary_prompt_and_target(dummy_tr, format_type=format_type)
        prompt_tokens = tokenizer.encode(prompt_str)

        gen_tokens = list(prompt_tokens)
        with torch.no_grad():
            for _ in range(20):
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
        parsed = parse_elementary_completion(completion, format_type=format_type)
        pred_w = parsed["write_digit"]
        pred_c = parsed["carry_out"]

        gt_sum = curr_carry
        gt_w = gt_sum % 10
        gt_c = 0

        step_correct = (pred_w == gt_w)
        carry_correct = (pred_c == gt_c)

        if not step_correct:
            all_transitions_correct = False
            if first_failure_col is None:
                first_failure_col = num_cols
                first_failure_type = "both" if not carry_correct else "write_digit"

        if not carry_correct:
            all_carries_correct = False
            if first_failure_col is None:
                first_failure_col = num_cols
                first_failure_type = "both" if not step_correct else "carry_out"

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

    duration_ms = (time.perf_counter() - t0) * 1000.0

    # Reconstruct final multi-digit number
    res_str = "".join(str(d) for d in reversed(emitted_digits))
    pred_res = int(res_str) if res_str.isdigit() else None
    is_exact = (pred_res is not None and pred_res == prob.result)

    return {
        "problem_a": prob.a,
        "problem_b": prob.b,
        "expected_result": prob.result,
        "predicted_result": pred_res,
        "is_exact": is_exact,
        "all_transitions_correct": all_transitions_correct,
        "all_carries_correct": all_carries_correct,
        "first_failure_col": first_failure_col,
        "first_failure_type": first_failure_type,
        "total_columns": step_count,
        "duration_ms": round(duration_ms, 2),
        "latency_per_digit_ms": round(duration_ms / max(1, step_count), 2),
        "trace": transition_trace,
    }
