"""Milestone M89: Causally-Consistent Foundational Curriculum Generator.

Constructs controlled multi-domain tasks with:
1. Explicit task-conditioning prefixes (instruction tokens) so that identical numerical sequences
   are not asked to map to conflicting target procedures without task context.
2. Causally-computable procedures (respecting lower-triangular attention so positions do not depend
   on unobserved future tokens).
3. 6 Core Foundational Domains:
   - Place-Value Reasoning (positional magnitude scaling, digit extraction).
   - Multi-Digit Arithmetic (modular affine addition, subtraction).
   - Boolean & Conditional Logic (bitwise AND, OR, XOR, conditional parity).
   - Sequence Manipulation (modular quadratic polynomials, reflection).
   - Compositional Procedures (two-stage chaining of arithmetic and logic).
   - Instruction Semantics (opcode-guided execution).
4. Strictly disjoint partitions: train (48), dev (16), qualification (16), sealed test (48).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

from adaptive_mini_gpt.model import MiniGPT


class RetentionDomain(Enum):
    PLACE_VALUE = "PLACE_VALUE"
    ARITHMETIC = "ARITHMETIC"
    LOGIC = "LOGIC"
    SEQUENCE = "SEQUENCE"
    COMPOSITION = "COMPOSITION"
    INSTRUCTION = "INSTRUCTION"


@dataclass(frozen=True)
class RetentionExample:
    """Single input-target sequence pair with optional task opcode token."""

    input_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    task_id: str
    opcode: int


@dataclass(frozen=True)
class RetentionDataset:
    """Four mutually disjoint partitions for a task."""

    task_id: str
    domain: RetentionDomain
    train_examples: list[RetentionExample]
    dev_examples: list[RetentionExample]
    qualification_examples: list[RetentionExample]
    sealed_test_examples: list[RetentionExample]


class M89CurriculumGenerator:
    """Generates structured, causally-consistent foundational tasks with task conditioning."""

    def __init__(self, vocab_size: int = 16, seq_len: int = 4, opcode_offset: int = 16) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.opcode_offset = opcode_offset

    def generate_task(
        self,
        task_id: str,
        domain: RetentionDomain,
        rule_name: str,
        rule_params: dict[str, Any],
        opcode: int,
        seed: int = 42,
    ) -> RetentionDataset:
        """Generate a complete task with four strictly disjoint partitions."""
        rng = random.Random(seed)
        L = self.seq_len

        # Pool of unique numerical candidate sequences
        all_sequences: set[tuple[int, ...]] = set()
        target_count = 128  # 48 train + 16 dev + 16 qual + 48 test

        attempts = 0
        while len(all_sequences) < target_count and attempts < 10_000:
            attempts += 1
            seq = tuple(rng.randint(0, self.vocab_size - 1) for _ in range(L))
            all_sequences.add(seq)

        seq_list = sorted(all_sequences)
        rng.shuffle(seq_list)

        train_seqs = seq_list[:48]
        dev_seqs = seq_list[48:64]
        qual_seqs = seq_list[64:80]
        test_seqs = seq_list[80:128]

        def _compute_target(seq: tuple[int, ...]) -> tuple[int, ...]:
            if domain == RetentionDomain.PLACE_VALUE:
                # Place extraction / scaling: each digit scaled and biased
                scale = rule_params.get("scale", 3)
                offset = rule_params.get("offset", 2)
                return tuple((s * scale + offset) % self.vocab_size for s in seq)

            elif domain == RetentionDomain.ARITHMETIC:
                # Modular affine addition / subtraction
                addend = rule_params.get("addend", 5)
                subtrahend = rule_params.get("subtrahend", 0)
                return tuple((s + addend - subtrahend) % self.vocab_size for s in seq)

            elif domain == RetentionDomain.LOGIC:
                # Boolean conditional XOR
                mask = rule_params.get("mask", 6)
                thresh = rule_params.get("thresh", 8)
                return tuple(
                    (s ^ mask) % self.vocab_size if s >= thresh else (s ^ (mask >> 1)) % self.vocab_size
                    for s in seq
                )

            elif domain == RetentionDomain.SEQUENCE:
                # Modular quadratic transformation: a*s^2 + b*s + c mod V
                a = rule_params.get("quad_a", 1)
                b = rule_params.get("quad_b", 2)
                c = rule_params.get("quad_c", 3)
                return tuple((a * s * s + b * s + c) % self.vocab_size for s in seq)

            elif domain == RetentionDomain.COMPOSITION:
                # Two-stage composition: arithmetic addition then bitwise XOR
                add_k = rule_params.get("stage1_add", 3)
                xor_m = rule_params.get("stage2_xor", 5)
                return tuple((((s + add_k) % self.vocab_size) ^ xor_m) % self.vocab_size for s in seq)

            elif domain == RetentionDomain.INSTRUCTION:
                # Instruction execution based on opcode mode
                op_mode = rule_params.get("mode", 1)
                if op_mode == 0:
                    return tuple((s + 4) % self.vocab_size for s in seq)
                elif op_mode == 1:
                    return tuple((s ^ 7) % self.vocab_size for s in seq)
                else:
                    return tuple((s * 3) % self.vocab_size for s in seq)

            return seq

        def _make_examples(sequences: list[tuple[int, ...]]) -> list[RetentionExample]:
            return [
                RetentionExample(
                    input_tokens=s,
                    target_tokens=_compute_target(s),
                    task_id=task_id,
                    opcode=opcode,
                )
                for s in sequences
            ]

        return RetentionDataset(
            task_id=task_id,
            domain=domain,
            train_examples=_make_examples(train_seqs),
            dev_examples=_make_examples(dev_seqs),
            qualification_examples=_make_examples(qual_seqs),
            sealed_test_examples=_make_examples(test_seqs),
        )

    def generate_curriculum(self, num_tasks: int = 10, seed: int = 42) -> list[RetentionDataset]:
        """Generate a sequence of N structured foundational tasks across domains."""
        domains = [
            (RetentionDomain.PLACE_VALUE, "place_scale", lambda i: {"scale": 3, "offset": 2 + (i % 3)}),
            (RetentionDomain.ARITHMETIC, "affine_add", lambda i: {"addend": 3 + (i % 5), "subtrahend": 0}),
            (RetentionDomain.LOGIC, "bool_xor", lambda i: {"mask": 6, "thresh": 8}),
            (RetentionDomain.ARITHMETIC, "affine_sub", lambda i: {"addend": 0, "subtrahend": 5 + (i % 4)}),
            (RetentionDomain.COMPOSITION, "comp_add_xor", lambda i: {"stage1_add": 3, "stage2_xor": 5}),
            (RetentionDomain.INSTRUCTION, "inst_opcode", lambda i: {"mode": i % 3}),
        ]

        tasks = []
        for i in range(num_tasks):
            domain, rule_prefix, param_fn = domains[i % len(domains)]
            rule_params = param_fn(i)
            t = self.generate_task(
                task_id=f"m89_{rule_prefix}_{i}",
                domain=domain,
                rule_name=f"{rule_prefix}_{i}",
                rule_params=rule_params,
                opcode=self.opcode_offset + i,
                seed=seed + i * 100,
            )
            tasks.append(t)
        return tasks


def extract_retention_features(
    model: MiniGPT,
    examples: list[RetentionExample],
    device: torch.device,
    base_offset: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pass examples through frozen MiniGPT backbone and extract hidden states for symbol positions."""
    input_ids_list = []
    target_ids_list = []

    for ex in examples:
        in_seq = [base_offset + s for s in ex.input_tokens]
        tgt_seq = list(ex.target_tokens)
        input_ids_list.append(in_seq)
        target_ids_list.append(tgt_seq)

    x = torch.tensor(input_ids_list, dtype=torch.long, device=device)
    y = torch.tensor(target_ids_list, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        tok_emb = model.wte(x)
        if model.wpe is not None:
            pos = torch.arange(0, x.size(1), dtype=torch.long, device=device)
            h = model.drop(tok_emb + model.wpe(pos))
        else:
            h = model.drop(tok_emb)
        for block in model.blocks:
            h, _ = block(h)
        h = model.ln_f(h)

    return h, y
