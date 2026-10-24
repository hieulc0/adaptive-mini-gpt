"""Phase 4 & 5: High-Information Procedural Sequence Transformation Tasks.

Implements:
- Controlled symbolic sequence transformations with 16 discrete symbols (S0..S15).
- Sequence rules:
    * CAESAR_SHIFT_MOD3: si -> (si + 3) % 16
    * CAESAR_SHIFT_MOD5: si -> (si + 5) % 16
    * AFFINE_TRANSFORM: si -> (3 * si + 1) % 16
    * COMPLEMENT_REFLECT: si -> 15 - si
    * NONLINEAR_SBOX: bijective non-linear permutation pi(si)
    * BITWISE_XOR_MASK: si -> (si ^ 6) % 16
    * MODULAR_DOUBLING: si -> (2 * si) % 16
- Four isolated, disjoint partitions per task instance:
    * train (practice/adaptation examples)
    * dev (router observable examples)
    * qualification (held-out candidate qualification)
    * sealed_test (strictly isolated evaluation)
- Exact-match sequence correctness and token accuracy metrics.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from enum import StrEnum

SBOX_MAP = (7, 12, 14, 9, 2, 1, 5, 15, 11, 6, 13, 0, 4, 8, 10, 3)


class ProceduralRuleFamily(StrEnum):
    """Supported deterministic procedural transformation families."""

    CAESAR_SHIFT_MOD3 = "CAESAR_SHIFT_MOD3"
    CAESAR_SHIFT_MOD5 = "CAESAR_SHIFT_MOD5"
    AFFINE_TRANSFORM = "AFFINE_TRANSFORM"
    COMPLEMENT_REFLECT = "COMPLEMENT_REFLECT"
    NONLINEAR_SBOX = "NONLINEAR_SBOX"
    BITWISE_XOR_MASK = "BITWISE_XOR_MASK"
    MODULAR_DOUBLING = "MODULAR_DOUBLING"


@dataclass(frozen=True)
class ProceduralExample:
    """A single input/output sequence transformation pair."""

    example_id: str
    rule_family: str
    input_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    input_text: str
    target_text: str


@dataclass
class ProceduralTaskDataset:
    """Disjoint dataset split for a procedural task instance."""

    task_id: str
    rule_family: ProceduralRuleFamily
    seed: int
    train_examples: list[ProceduralExample]
    dev_examples: list[ProceduralExample]
    qualification_examples: list[ProceduralExample]
    sealed_test_examples: list[ProceduralExample]

    def all_input_tuples(self) -> set[tuple[int, ...]]:
        all_ex = (
            self.train_examples
            + self.dev_examples
            + self.qualification_examples
            + self.sealed_test_examples
        )
        return {ex.input_tokens for ex in all_ex}


def apply_procedural_rule(
    rule_family: ProceduralRuleFamily | str, seq: tuple[int, ...], vocab_size: int = 16
) -> tuple[int, ...]:
    """Deterministically map input sequence to target sequence under the rule."""
    rf_str = rule_family.value if hasattr(rule_family, "value") else str(rule_family)
    if rf_str == ProceduralRuleFamily.CAESAR_SHIFT_MOD3:
        return tuple((s + 3) % vocab_size for s in seq)
    elif rf_str == ProceduralRuleFamily.CAESAR_SHIFT_MOD5:
        return tuple((s + 5) % vocab_size for s in seq)
    elif rf_str == ProceduralRuleFamily.AFFINE_TRANSFORM:
        return tuple((3 * s + 1) % vocab_size for s in seq)
    elif rf_str == ProceduralRuleFamily.COMPLEMENT_REFLECT:
        return tuple((vocab_size - 1 - s) for s in seq)
    elif rf_str == ProceduralRuleFamily.NONLINEAR_SBOX:
        return tuple(SBOX_MAP[s % len(SBOX_MAP)] for s in seq)
    elif rf_str == ProceduralRuleFamily.BITWISE_XOR_MASK:
        return tuple((s ^ 6) % vocab_size for s in seq)
    elif rf_str == ProceduralRuleFamily.MODULAR_DOUBLING:
        return tuple((2 * s) % vocab_size for s in seq)
    else:
        raise ValueError(f"Unknown rule family: {rule_family}")


class ProceduralTaskGenerator:
    """Deterministic generator of procedural sequence tasks with strict split isolation."""

    def __init__(self, vocab_size: int = 16) -> None:
        self.vocab_size = vocab_size

    def generate_task(
        self,
        task_id: str,
        rule_family: ProceduralRuleFamily,
        seed: int = 42,
        seq_len: int = 3,
        n_train: int = 48,
        n_dev: int = 16,
        n_qual: int = 16,
        n_test: int = 32,
    ) -> ProceduralTaskDataset:
        """Generate a procedural task instance with strictly disjoint partitions."""
        rng = random.Random(seed)

        symbols = list(range(self.vocab_size))
        all_possible = [
            seq for seq in itertools.product(symbols, repeat=seq_len) if len(set(seq)) > 1
        ]
        rng.shuffle(all_possible)

        total_needed = n_train + n_dev + n_qual + n_test
        if len(all_possible) < total_needed:
            raise ValueError(
                f"Vocabulary size {self.vocab_size} with length {seq_len} provides {len(all_possible)} combinations, need {total_needed}"
            )

        selected = all_possible[:total_needed]

        def make_examples(seq_list: list[tuple[int, ...]], prefix: str) -> list[ProceduralExample]:
            exs = []
            for idx, seq in enumerate(seq_list):
                tgt = apply_procedural_rule(rule_family, seq, self.vocab_size)
                in_str = " ".join(f"S{s}" for s in seq)
                tgt_str = " ".join(f"S{s}" for s in tgt)
                exs.append(
                    ProceduralExample(
                        example_id=f"{prefix}_{idx}",
                        rule_family=str(rule_family.value if hasattr(rule_family, "value") else rule_family),
                        input_tokens=seq,
                        target_tokens=tgt,
                        input_text=f"Rule: {rule_family.value if hasattr(rule_family, 'value') else rule_family} | Input: {in_str} -> Output:",
                        target_text=tgt_str,
                    )
                )
            return exs

        train_seqs = selected[:n_train]
        dev_seqs = selected[n_train : n_train + n_dev]
        qual_seqs = selected[n_train + n_dev : n_train + n_dev + n_qual]
        test_seqs = selected[n_train + n_dev + n_qual :]

        # Verify disjointness
        assert len(set(train_seqs) & set(dev_seqs)) == 0
        assert len(set(train_seqs) & set(qual_seqs)) == 0
        assert len(set(train_seqs) & set(test_seqs)) == 0
        assert len(set(dev_seqs) & set(qual_seqs)) == 0
        assert len(set(dev_seqs) & set(test_seqs)) == 0
        assert len(set(qual_seqs) & set(test_seqs)) == 0

        return ProceduralTaskDataset(
            task_id=task_id,
            rule_family=rule_family,
            seed=seed,
            train_examples=make_examples(train_seqs, "train"),
            dev_examples=make_examples(dev_seqs, "dev"),
            qualification_examples=make_examples(qual_seqs, "qual"),
            sealed_test_examples=make_examples(test_seqs, "test"),
        )
