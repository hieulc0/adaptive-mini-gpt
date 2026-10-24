"""Milestone M81 Phase 1: Continual Skill Learning Curriculum with Unseen Rule Instances.

Implements structured curriculum across four progressive procedural families:
1. Arithmetic Transformations (parametric modular affine maps with unseen coefficients).
2. Conditional & Boolean Logic (parity-dependent bitwise masking and branching).
3. Symbolic Sequence Procedures (sequence reversals, cyclic shifts, and bijective transpositions).
4. Compositions of Operations (two-stage compound transformations).

Crucial distinction: generates task families with unseen rule instances,
ensuring the student is evaluated on learning new rules rather than memorizing a single fixed rule.
Strictly isolated disjoint partitions: train (48), dev (16), qualification (16), sealed_test (48).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from enum import StrEnum


class CurriculumFamily(StrEnum):
    """Procedural task families in the M81 continual learning curriculum."""

    ARITHMETIC = "ARITHMETIC"
    CONDITIONAL_LOGIC = "CONDITIONAL_LOGIC"
    SYMBOLIC_PROCEDURE = "SYMBOLIC_PROCEDURE"
    COMPOSITION = "COMPOSITION"


@dataclass(frozen=True)
class CurriculumExample:
    """A single input/output sequence transformation pair."""

    example_id: str
    family: CurriculumFamily
    rule_instance_id: str
    input_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    input_text: str
    target_text: str


@dataclass
class CurriculumTaskDataset:
    """Disjoint dataset split for a curriculum task instance."""

    task_id: str
    family: CurriculumFamily
    rule_instance_id: str
    seed: int
    train_examples: list[CurriculumExample]
    dev_examples: list[CurriculumExample]
    qualification_examples: list[CurriculumExample]
    sealed_test_examples: list[CurriculumExample]

    def all_input_tuples(self) -> set[tuple[int, ...]]:
        all_ex = (
            self.train_examples
            + self.dev_examples
            + self.qualification_examples
            + self.sealed_test_examples
        )
        return {ex.input_tokens for ex in all_ex}


def apply_curriculum_rule(
    family: CurriculumFamily,
    rule_instance: dict[str, int],
    seq: tuple[int, ...],
    vocab_size: int = 16,
) -> tuple[int, ...]:
    """Deterministically apply a parametric rule instance to a sequence of symbol tokens."""
    if family == CurriculumFamily.ARITHMETIC:
        # Affine map: (a * s + b) % vocab_size
        a = rule_instance.get("a", 3)
        b = rule_instance.get("b", 1)
        return tuple((a * s + b) % vocab_size for s in seq)

    elif family == CurriculumFamily.CONDITIONAL_LOGIC:
        # Branching logic: if s % 2 == 1 -> s ^ mask; else -> (s + offset) % vocab_size
        mask = rule_instance.get("mask", 6)
        offset = rule_instance.get("offset", 4)
        return tuple(
            ((s ^ mask) % vocab_size) if (s % 2 == 1) else ((s + offset) % vocab_size)
            for s in seq
        )

    elif family == CurriculumFamily.SYMBOLIC_PROCEDURE:
        # Sequence-level procedure: reverse + cyclic shift
        shift = rule_instance.get("shift", 3)
        reversed_seq = tuple(reversed(seq))
        return tuple((s + shift) % vocab_size for s in reversed_seq)

    elif family == CurriculumFamily.COMPOSITION:
        # Two-stage compound rule: Stage 1 arithmetic -> Stage 2 conditional
        a = rule_instance.get("a", 3)
        b = rule_instance.get("b", 5)
        mask = rule_instance.get("mask", 6)
        offset = rule_instance.get("offset", 2)
        inter = tuple((a * s + b) % vocab_size for s in seq)
        return tuple(
            ((s ^ mask) % vocab_size) if (s % 2 == 1) else ((s + offset) % vocab_size)
            for s in inter
        )

    else:
        raise ValueError(f"Unknown curriculum family: {family}")


class CurriculumTaskGenerator:
    """Generates procedural sequence tasks across families with strictly disjoint partitions."""

    def __init__(self, vocab_size: int = 16) -> None:
        self.vocab_size = vocab_size

    def generate_task(
        self,
        task_id: str,
        family: CurriculumFamily,
        rule_instance: dict[str, int],
        rule_instance_id: str,
        seed: int = 42,
        seq_len: int = 3,
        n_train: int = 48,
        n_dev: int = 16,
        n_qual: int = 16,
        n_test: int = 48,
    ) -> CurriculumTaskDataset:
        """Generate a task instance with strict split isolation."""
        rng = random.Random(seed)
        symbols = list(range(self.vocab_size))
        all_possible = [
            seq for seq in itertools.product(symbols, repeat=seq_len) if len(set(seq)) > 1
        ]
        rng.shuffle(all_possible)

        total_needed = n_train + n_dev + n_qual + n_test
        if len(all_possible) < total_needed:
            raise ValueError(
                f"Vocabulary {self.vocab_size} with length {seq_len} provides {len(all_possible)}, need {total_needed}"
            )

        selected = all_possible[:total_needed]

        def make_examples(
            seq_list: list[tuple[int, ...]], prefix: str
        ) -> list[CurriculumExample]:
            exs = []
            for idx, seq in enumerate(seq_list):
                tgt = apply_curriculum_rule(family, rule_instance, seq, self.vocab_size)
                in_str = " ".join(f"S{s}" for s in seq)
                tgt_str = " ".join(f"S{s}" for s in tgt)
                exs.append(
                    CurriculumExample(
                        example_id=f"{prefix}_{idx}",
                        family=family,
                        rule_instance_id=rule_instance_id,
                        input_tokens=seq,
                        target_tokens=tgt,
                        input_text=f"Family: {family.value} | Rule: {rule_instance_id} | In: {in_str} -> Out:",
                        target_text=tgt_str,
                    )
                )
            return exs

        train_seqs = selected[:n_train]
        dev_seqs = selected[n_train : n_train + n_dev]
        qual_seqs = selected[n_train + n_dev : n_train + n_dev + n_qual]
        test_seqs = selected[n_train + n_dev + n_qual :]

        # Verify strict partition isolation
        assert len(set(train_seqs) & set(dev_seqs)) == 0, "Train and Dev must be disjoint"
        assert len(set(train_seqs) & set(qual_seqs)) == 0, "Train and Qual must be disjoint"
        assert len(set(train_seqs) & set(test_seqs)) == 0, "Train and Test must be disjoint"
        assert len(set(dev_seqs) & set(qual_seqs)) == 0, "Dev and Qual must be disjoint"
        assert len(set(dev_seqs) & set(test_seqs)) == 0, "Dev and Test must be disjoint"
        assert len(set(qual_seqs) & set(test_seqs)) == 0, "Qual and Test must be disjoint"

        return CurriculumTaskDataset(
            task_id=task_id,
            family=family,
            rule_instance_id=rule_instance_id,
            seed=seed,
            train_examples=make_examples(train_seqs, "train"),
            dev_examples=make_examples(dev_seqs, "dev"),
            qualification_examples=make_examples(qual_seqs, "qual"),
            sealed_test_examples=make_examples(test_seqs, "test"),
        )
