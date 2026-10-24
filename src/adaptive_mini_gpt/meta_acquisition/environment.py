"""Milestone M82 Phase 1 & 2: Multi-Family Procedural Environment with Strict Disjoint Splits.

Constructs disjoint task-family partitions:
1. Meta-Training Families:
   - ARITHMETIC_AFFINE: (a * s + b) % 16
   - PARITY_LOGIC: if s % 2 == 1 -> s ^ mask; else -> (s + offset) % 16
   - CYCLIC_PROCEDURE: reverse(seq) + cyclic shift
2. Meta-Validation Families:
   - MODULAR_NONLINEAR: (s * 3 + (s ^ 5)) % 16
   - BLOCK_TRANSPOSITION: pairwise adjacent symbol swapping
3. Sealed Meta-Test Families (strictly held-out, unseen during policy training):
   - PALINDROME_REFLECT: sequence reflection with alternating bitwise inversion
   - TWO_STAGE_CASCADE: affine map followed by parity branching
   - PERMUTATION_INVERSION: permutation inverse with modular offset

Each task instance features strictly disjoint partitions:
- train (48): adaptation examples
- dev (16): validation & probe observations
- qualification (16): admission gate (EM >= 0.85)
- sealed_test (48): final scientific generalization evaluation
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class TaskFamily(StrEnum):
    """Categorical family classification for procedural tasks."""

    # Meta-Training Families
    ARITHMETIC_AFFINE = "ARITHMETIC_AFFINE"
    PARITY_LOGIC = "PARITY_LOGIC"
    CYCLIC_PROCEDURE = "CYCLIC_PROCEDURE"

    # Meta-Validation Families
    MODULAR_NONLINEAR = "MODULAR_NONLINEAR"
    MODULAR_QUADRATIC = "MODULAR_QUADRATIC"
    BLOCK_TRANSPOSITION = "BLOCK_TRANSPOSITION"

    # Sealed Meta-Test Families (unseen during policy training)
    PALINDROME_REFLECT = "PALINDROME_REFLECT"
    TWO_STAGE_CASCADE = "TWO_STAGE_CASCADE"
    PERMUTATION_INVERSION = "PERMUTATION_INVERSION"


class SplitType(StrEnum):
    """Regime split for meta-learning curriculum."""

    META_TRAIN = "META_TRAIN"
    META_VAL = "META_VAL"
    META_TEST = "META_TEST"


@dataclass(frozen=True)
class TaskExample:
    """Single input/target sequence pair with immutable metadata."""

    example_id: str
    family: TaskFamily
    rule_instance_id: str
    input_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    input_text: str
    target_text: str


@dataclass
class MetaTaskDataset:
    """Disjoint dataset instance with isolated partitions."""

    task_id: str
    family: TaskFamily
    split: SplitType
    rule_instance_id: str
    rule_params: dict[str, Any]
    seed: int
    train_examples: list[TaskExample]
    dev_examples: list[TaskExample]
    qualification_examples: list[TaskExample]
    sealed_test_examples: list[TaskExample]

    def all_input_tuples(self) -> set[tuple[int, ...]]:
        all_ex = (
            self.train_examples
            + self.dev_examples
            + self.qualification_examples
            + self.sealed_test_examples
        )
        return {ex.input_tokens for ex in all_ex}


def apply_procedural_rule(
    family: TaskFamily,
    rule_params: dict[str, Any],
    seq: tuple[int, ...],
    vocab_size: int = 16,
) -> tuple[int, ...]:
    """Deterministically execute parametric procedural rule on integer token sequence."""
    if family == TaskFamily.ARITHMETIC_AFFINE:
        a = rule_params.get("a", 3)
        b = rule_params.get("b", 5)
        return tuple((a * s + b) % vocab_size for s in seq)

    elif family == TaskFamily.PARITY_LOGIC:
        mask = rule_params.get("mask", 6)
        offset = rule_params.get("offset", 4)
        return tuple(
            ((s ^ mask) % vocab_size) if (s % 2 == 1) else ((s + offset) % vocab_size)
            for s in seq
        )

    elif family == TaskFamily.CYCLIC_PROCEDURE:
        shift = rule_params.get("shift", 3)
        reversed_seq = tuple(reversed(seq))
        return tuple((s + shift) % vocab_size for s in reversed_seq)

    elif family == TaskFamily.MODULAR_NONLINEAR:
        mult = rule_params.get("mult", 3)
        mask = rule_params.get("mask", 5)
        return tuple(((s * mult) + (s ^ mask)) % vocab_size for s in seq)

    elif family == TaskFamily.MODULAR_QUADRATIC:
        a = rule_params.get("a", 1)
        b = rule_params.get("b", 3)
        c = rule_params.get("c", 1)
        return tuple((a * s * s + b * s + c) % vocab_size for s in seq)

    elif family == TaskFamily.BLOCK_TRANSPOSITION:
        # Swap adjacent pairs: (s0, s1, s2, s3) -> (s1, s0, s3, s2)
        out = list(seq)
        for i in range(0, len(out) - 1, 2):
            out[i], out[i + 1] = out[i + 1], out[i]
        offset = rule_params.get("offset", 1)
        return tuple((s + offset) % vocab_size for s in out)

    elif family == TaskFamily.PALINDROME_REFLECT:
        # Reverse sequence and apply alternating mask
        alt_mask = rule_params.get("alt_mask", 7)
        rev = tuple(reversed(seq))
        return tuple(
            (s ^ alt_mask) % vocab_size if (i % 2 == 0) else s
            for i, s in enumerate(rev)
        )

    elif family == TaskFamily.TWO_STAGE_CASCADE:
        a = rule_params.get("a", 3)
        b = rule_params.get("b", 1)
        mask = rule_params.get("mask", 6)
        offset = rule_params.get("offset", 2)
        inter = tuple((a * s + b) % vocab_size for s in seq)
        return tuple(
            ((s ^ mask) % vocab_size) if (s % 2 == 1) else ((s + offset) % vocab_size)
            for s in inter
        )

    elif family == TaskFamily.PERMUTATION_INVERSION:
        shift = rule_params.get("shift", 5)
        # Permutation inversion: value-to-index mapping modulo vocab_size
        return tuple((vocab_size - 1 - s + shift) % vocab_size for s in seq)

    else:
        raise ValueError(f"Unsupported task family: {family}")


class MetaEnvironmentGenerator:
    """Generates procedural tasks partitioned across meta-train, meta-val, and sealed meta-test."""

    def __init__(self, vocab_size: int = 16, seq_len: int = 4) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        # Canonical space of sequences of length 4 from vocab [0..15]
        # Total universe = 16^4 = 65,536 sequences
        # Each task requires 48 + 16 + 16 + 48 = 128 mutually disjoint sequences

    def generate_task(
        self,
        task_id: str,
        family: TaskFamily,
        split: SplitType,
        rule_params: dict[str, Any],
        rule_instance_id: str,
        seed: int = 42,
    ) -> MetaTaskDataset:
        """Create a reproducible task dataset with isolated 48/16/16/48 partitions."""
        rng = random.Random(seed)

        # Sample 128 unique sequences
        needed = 48 + 16 + 16 + 48  # 128
        sampled_inputs: set[tuple[int, ...]] = set()

        while len(sampled_inputs) < needed:
            candidate = tuple(rng.randrange(0, self.vocab_size) for _ in range(self.seq_len))
            # Require distinct symbols or valid pattern for high informational content
            sampled_inputs.add(candidate)

        input_list = sorted(list(sampled_inputs))
        rng.shuffle(input_list)

        train_in = input_list[0:48]
        dev_in = input_list[48:64]
        qual_in = input_list[64:80]
        test_in = input_list[80:128]

        def _make_examples(inputs: list[tuple[int, ...]], prefix: str) -> list[TaskExample]:
            exs = []
            for idx, inp in enumerate(inputs):
                tgt = apply_procedural_rule(family, rule_params, inp, self.vocab_size)
                inp_str = " ".join(str(x) for x in inp)
                tgt_str = " ".join(str(x) for x in tgt)
                exs.append(
                    TaskExample(
                        example_id=f"{task_id}_{prefix}_{idx:03d}",
                        family=family,
                        rule_instance_id=rule_instance_id,
                        input_tokens=inp,
                        target_tokens=tgt,
                        input_text=inp_str,
                        target_text=tgt_str,
                    )
                )
            return exs

        train_ex = _make_examples(train_in, "train")
        dev_ex = _make_examples(dev_in, "dev")
        qual_ex = _make_examples(qual_in, "qual")
        test_ex = _make_examples(test_in, "test")

        # Verify strict disjointness
        assert len(set(train_in) & set(dev_in)) == 0
        assert len(set(train_in) & set(qual_in)) == 0
        assert len(set(train_in) & set(test_in)) == 0
        assert len(set(dev_in) & set(qual_in)) == 0
        assert len(set(dev_in) & set(test_in)) == 0
        assert len(set(qual_in) & set(test_in)) == 0

        return MetaTaskDataset(
            task_id=task_id,
            family=family,
            split=split,
            rule_instance_id=rule_instance_id,
            rule_params=rule_params,
            seed=seed,
            train_examples=train_ex,
            dev_examples=dev_ex,
            qualification_examples=qual_ex,
            sealed_test_examples=test_ex,
        )
