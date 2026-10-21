"""M21: Digit-Level Neural Arithmetic and Procedural Scratchpad Dataset Generator.

Generates verified addition problems with controlled operand lengths and carry counts:
- Arm A: Standard text (e.g. "127 + 386 = 513")
- Arm B: Digit-level MSD (e.g. "1 2 7 + 3 8 6 = 5 1 3")
- Arm C: Column-by-column procedural scratchpad with intermediate carries
- Arm D: Digit-level LSD (e.g. "7 2 1 + 6 8 3 = 3 1 5", least-significant-digit first)

Features:
- Controlled carry distributions: 0 carries, 1 carry, 2 carries, 3+ carries
- Disjoint train, validation, and sealed-test sets
- Controlled length extrapolation partitions (2-3 digits train, 4-5 digits test)
- Target-only cross-entropy loss masking (-100 for prompt tokens)
- Cryptographic SHA-256 dataset fingerprinting
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer


@dataclass
class ColumnCalculation:
    """Detailed intermediate state for a single column in column addition."""

    col_index: int
    name: str  # e.g. "units", "tens", "hundreds"
    digit_a: int
    digit_b: int
    carry_in: int
    total_sum: int
    write_digit: int
    carry_out: int


@dataclass(frozen=True)
class ArithmeticProblem:
    """An exact addition problem with complete column-by-column metadata."""

    a: int
    b: int
    result: int
    length_a: int
    length_b: int
    max_length: int
    num_carries: int
    columns: tuple[ColumnCalculation, ...]
    split: str

    @classmethod
    def create(cls, a: int, b: int, split: str = "train") -> ArithmeticProblem:
        """Construct problem and compute column-by-column addition procedure."""
        res = a + b
        str_a = str(a)
        str_b = str(b)
        len_a = len(str_a)
        len_b = len(str_b)
        max_len = max(len_a, len_b)

        # Pad with leading zeros from the left so columns align from right (units = col 0)
        pad_a = str_a.zfill(max_len)
        pad_b = str_b.zfill(max_len)

        col_names = ["units", "tens", "hundreds", "thousands", "ten_thousands", "hundred_thousands"]
        columns: list[ColumnCalculation] = []
        carry = 0
        total_carries = 0

        # Process from right (units) to left
        for i in range(max_len):
            col_name = col_names[i] if i < len(col_names) else f"col_{i}"
            idx_from_right = max_len - 1 - i
            d_a = int(pad_a[idx_from_right])
            d_b = int(pad_b[idx_from_right])
            col_sum = d_a + d_b + carry
            w_digit = col_sum % 10
            next_carry = col_sum // 10

            if next_carry > 0:
                total_carries += 1

            columns.append(
                ColumnCalculation(
                    col_index=i,
                    name=col_name,
                    digit_a=d_a,
                    digit_b=d_b,
                    carry_in=carry,
                    total_sum=col_sum,
                    write_digit=w_digit,
                    carry_out=next_carry,
                )
            )
            carry = next_carry

        # If there is a trailing carry beyond max_len (e.g. 99 + 1 = 100)
        if carry > 0:
            col_name = col_names[max_len] if max_len < len(col_names) else f"col_{max_len}"
            columns.append(
                ColumnCalculation(
                    col_index=max_len,
                    name=col_name,
                    digit_a=0,
                    digit_b=0,
                    carry_in=carry,
                    total_sum=carry,
                    write_digit=carry,
                    carry_out=0,
                )
            )

        return cls(
            a=a,
            b=b,
            result=res,
            length_a=len_a,
            length_b=len_b,
            max_length=max_len,
            num_carries=total_carries,
            columns=tuple(columns),
            split=split,
        )


def format_problem(prob: ArithmeticProblem, arm: str) -> tuple[str, str]:
    """Format prompt and target strings for a given experimental arm.
    
    Returns:
        (prompt_text, target_text)
    """
    str_a = str(prob.a)
    str_b = str(prob.b)
    str_res = str(prob.result)

    if arm == "standard":
        prompt = f"<|user|>\nAdd: {str_a} + {str_b}\n<|assistant|>\nAnswer: "
        target = f"{str_res}<|eos|>\n"

    elif arm == "digit_msd":
        spaced_a = " ".join(str_a)
        spaced_b = " ".join(str_b)
        spaced_res = " ".join(str_res)
        prompt = f"<|user|>\nAdd: {spaced_a} + {spaced_b}\n<|assistant|>\nAnswer: "
        target = f"{spaced_res}<|eos|>\n"

    elif arm == "scratchpad":
        spaced_a = " ".join(str_a)
        spaced_b = " ".join(str_b)
        spaced_res = " ".join(str_res)
        prompt = f"<|user|>\nAdd: {spaced_a} + {spaced_b}\n<|assistant|>\nScratchpad:\n"

        lines: list[str] = []
        for col in prob.columns:
            # e.g. "col 0 (units): 7 + 6 + 0 = 13 | write 3 | carry 1"
            lines.append(
                f"col {col.col_index} ({col.name}): {col.digit_a} + {col.digit_b} + {col.carry_in} = {col.total_sum} | write {col.write_digit} | carry {col.carry_out}"
            )
        lines.append(f"Answer: {spaced_res}")
        target = "\n".join(lines) + "<|eos|>\n"

    elif arm == "digit_lsd":
        # Least Significant Digit first (reversed digit sequence)
        rev_a = " ".join(reversed(str_a))
        rev_b = " ".join(reversed(str_b))
        rev_res = " ".join(reversed(str_res))
        prompt = f"<|user|>\nAdd (LSD): {rev_a} + {rev_b}\n<|assistant|>\nAnswer: "
        target = f"{rev_res}<|eos|>\n"

    else:
        raise ValueError(f"Unknown experimental arm: {arm}")

    return prompt, target


class ArithmeticDataset(Dataset):
    """PyTorch Dataset for causal language modeling of arithmetic with target-only loss."""

    def __init__(
        self,
        problems: list[ArithmeticProblem],
        tokenizer: BPETokenizer,
        arm: str = "digit_msd",
        context_length: int = 256,
    ) -> None:
        self.problems = problems
        self.tokenizer = tokenizer
        self.arm = arm
        self.context_length = context_length

        self.input_ids_list: list[Tensor] = []
        self.labels_list: list[Tensor] = []
        self._encode_all()

    def _encode_all(self) -> None:
        for prob in self.problems:
            prompt_str, target_str = format_problem(prob, self.arm)
            prompt_toks = self.tokenizer.encode(prompt_str)
            target_toks = self.tokenizer.encode(target_str)

            full_toks = prompt_toks + target_toks
            if len(full_toks) > self.context_length:
                full_toks = full_toks[: self.context_length]
                p_len = min(len(prompt_toks), self.context_length)
            else:
                p_len = len(prompt_toks)

            # Target-only loss: mask prompt tokens with -100
            labels = [-100] * p_len + full_toks[p_len:]

            self.input_ids_list.append(torch.tensor(full_toks, dtype=torch.long))
            self.labels_list.append(torch.tensor(labels, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids_list)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.input_ids_list[idx], self.labels_list[idx]


def collate_arithmetic_batch(
    batch: list[tuple[Tensor, Tensor]],
    pad_token_id: int = 0,
) -> tuple[Tensor, Tensor]:
    """Collate variable length arithmetic sequences."""
    max_len = max(len(x[0]) for x in batch)
    b_inputs = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    b_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (inp, lab) in enumerate(batch):
        b_inputs[i, : len(inp)] = inp
        b_labels[i, : len(lab)] = lab

    return b_inputs, b_labels


def generate_controlled_addition_dataset(
    seed: int = 42,
    num_samples: int = 1000,
    lengths: tuple[int, ...] = (2, 3),
    split: str = "train",
    target_carry_distribution: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> list[ArithmeticProblem]:
    """Generate verified, non-overlapping addition problems with controlled carry distributions.
    
    target_carry_distribution corresponds to proportions for [0 carries, 1 carry, 2 carries, 3+ carries].
    """
    rng = random.Random(seed)
    seen_pairs: set[tuple[int, int]] = set()
    problems: list[ArithmeticProblem] = []

    # Map carry count to target bucket index: 0 -> 0, 1 -> 1, 2 -> 2, 3+ -> 3
    bucket_targets = [int(num_samples * p) for p in target_carry_distribution]
    # Adjust remainder to sum exactly to num_samples
    bucket_targets[0] += num_samples - sum(bucket_targets)
    bucket_counts = [0, 0, 0, 0]

    max_attempts = num_samples * 300
    attempts = 0

    while len(problems) < num_samples and attempts < max_attempts:
        attempts += 1
        len_a = rng.choice(lengths)
        len_b = rng.choice(lengths)

        min_a = 10 ** (len_a - 1)
        max_a = (10**len_a) - 1
        min_b = 10 ** (len_b - 1)
        max_b = (10**len_b) - 1

        a = rng.randint(min_a, max_a)
        b = rng.randint(min_b, max_b)

        # Enforce canonical ordering (a <= b) to prevent permutation duplicates
        pair = (min(a, b), max(a, b))
        if pair in seen_pairs:
            continue

        prob = ArithmeticProblem.create(pair[0], pair[1], split=split)
        bucket_idx = min(prob.num_carries, 3)

        if bucket_counts[bucket_idx] < bucket_targets[bucket_idx] or len(problems) > int(num_samples * 0.9):
            seen_pairs.add(pair)
            problems.append(prob)
            bucket_counts[bucket_idx] += 1

    return problems


def compute_dataset_fingerprint(problems: list[ArithmeticProblem]) -> str:
    """Compute deterministic SHA-256 fingerprint of problem pairs."""
    serialized = json.dumps(
        sorted([{"a": p.a, "b": p.b, "res": p.result, "carries": p.num_carries} for p in problems], key=lambda x: (x["a"], x["b"])),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def parse_arithmetic_completion(raw_text: str, arm: str) -> dict[str, Any]:
    """Parse raw completion text into structured prediction metrics."""
    cleaned = raw_text.strip()
    predicted_answer_str: str | None = None
    predicted_carries: list[int] = []
    is_valid = False

    if arm in ("standard", "digit_msd", "digit_lsd"):
        # Pattern: look for Answer: <digits>
        m = re.search(r"Answer:\s*([0-9\s]+)", cleaned, re.IGNORECASE)
        if m:
            digits_raw = m.group(1).strip()
            # Remove whitespace
            digits_clean = re.sub(r"\s+", "", digits_raw)
            if digits_clean.isdigit():
                if arm == "digit_lsd":
                    # Reverse digits back to most significant first
                    predicted_answer_str = digits_clean[::-1]
                else:
                    predicted_answer_str = digits_clean
                is_valid = True

    elif arm == "scratchpad":
        # Extract intermediate carries from scratchpad lines: "carry (\d+)"
        carry_matches = re.findall(r"carry\s+(\d+)", cleaned, re.IGNORECASE)
        predicted_carries = [int(c) for c in carry_matches]

        # Extract final answer line: "Answer: <digits>"
        m = re.search(r"Answer:\s*([0-9\s]+)", cleaned, re.IGNORECASE)
        if m:
            digits_raw = m.group(1).strip()
            digits_clean = re.sub(r"\s+", "", digits_raw)
            if digits_clean.isdigit():
                predicted_answer_str = digits_clean
                is_valid = True

    return {
        "predicted_answer": predicted_answer_str,
        "predicted_carries": predicted_carries,
        "is_valid": is_valid,
        "raw_text": raw_text[:200],
    }
