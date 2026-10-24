"""Phase 5: High-Information Procedural Task Streams and Generalization Taxonomy.

Implements:
- Generalization categories:
    1. NEW_INPUTS_KNOWN_RULE: Familiar rule with completely disjoint input sequences.
    2. NEW_PARAMETERIZATION: Known family with novel algebraic parameters (e.g., Affine slope/intercept, Caesar mod 7).
    3. COMPOSITIONAL_PROCEDURE: Novel composition of known procedural primitives (e.g. Reflect then Shift).
    4. UNFAMILIAR_FAMILY: Genuinely unseen algorithmic rules (Bitwise XOR mask, Deceptive parity inversion).
- Strict 4-way split isolation per task:
    * train: adaptation/training examples.
    * dev: observable development examples for zero-shot probing and feature extraction.
    * qualification: held-out candidate qualification split.
    * sealed_test: strictly isolated sealed evaluation split.
- Fully decoupled target scrambling for rigorous negative controls without permutation leakage.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from enum import StrEnum

from adaptive_mini_gpt.experience_guided_learning.experience_records import (
    ObservableTaskDescriptors,
)

SBOX_MAP = (7, 12, 14, 9, 2, 1, 5, 15, 11, 6, 13, 0, 4, 8, 10, 3)


class GeneralizationCategory(StrEnum):
    """Taxonomy of procedural generalization."""

    NEW_INPUTS_KNOWN_RULE = "NEW_INPUTS_KNOWN_RULE"
    NEW_PARAMETERIZATION = "NEW_PARAMETERIZATION"
    COMPOSITIONAL_PROCEDURE = "COMPOSITIONAL_PROCEDURE"
    UNFAMILIAR_FAMILY = "UNFAMILIAR_FAMILY"


class ProceduralRuleFamily(StrEnum):
    """Supported deterministic procedural transformation families."""

    # M74 Baseline families
    CAESAR_SHIFT_MOD3 = "CAESAR_SHIFT_MOD3"
    CAESAR_SHIFT_MOD5 = "CAESAR_SHIFT_MOD5"
    AFFINE_TRANSFORM = "AFFINE_TRANSFORM"
    COMPLEMENT_REFLECT = "COMPLEMENT_REFLECT"
    NONLINEAR_SBOX = "NONLINEAR_SBOX"

    # M75 New parameterizations & unfamiliar families
    AFFINE_PARAM_5_3 = "AFFINE_PARAM_5_3"
    CAESAR_SHIFT_MOD7 = "CAESAR_SHIFT_MOD7"
    COMPOSITE_SHIFT_REFLECT = "COMPOSITE_SHIFT_REFLECT"
    BITWISE_XOR_MASK = "BITWISE_XOR_MASK"
    DECEPTIVE_INVERSION = "DECEPTIVE_INVERSION"


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
    generalization_category: GeneralizationCategory
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

    def compute_observable_descriptors(
        self,
        zero_shot_dev_em: float = 0.0,
        zero_shot_dev_loss: float = 3.0,
        best_repertoire_em: float = 0.0,
        best_repertoire_loss: float = 3.0,
    ) -> ObservableTaskDescriptors:
        """Extract pre-training task descriptors observable from input sequences and dev set."""
        all_inputs = [ex.input_tokens for ex in self.dev_examples]
        flat_tokens = [tok for seq in all_inputs for tok in seq]

        sample_count = len(self.dev_examples)
        seq_length = len(all_inputs[0]) if all_inputs else 3
        unique_symbols = len(set(flat_tokens))

        # Input token variance
        if flat_tokens:
            mean_tok = sum(flat_tokens) / len(flat_tokens)
            variance = sum((t - mean_tok) ** 2 for t in flat_tokens) / len(flat_tokens)
        else:
            variance = 0.0

        # Input Shannon entropy
        counts: dict[int, int] = {}
        for t in flat_tokens:
            counts[t] = counts.get(t, 0) + 1
        entropy = 0.0
        n_tok = len(flat_tokens)
        if n_tok > 0:
            for count in counts.values():
                p = count / n_tok
                if p > 0.0:
                    entropy -= p * math.log2(p)

        return ObservableTaskDescriptors(
            sample_count=sample_count,
            seq_length=seq_length,
            unique_symbols_count=unique_symbols,
            input_variance=round(variance, 4),
            input_entropy=round(entropy, 4),
            zero_shot_dev_em=round(zero_shot_dev_em, 4),
            zero_shot_dev_loss=round(zero_shot_dev_loss, 4),
            best_repertoire_em=round(best_repertoire_em, 4),
            best_repertoire_loss=round(best_repertoire_loss, 4),
        )


def apply_procedural_rule(
    rule_family: ProceduralRuleFamily, seq: tuple[int, ...], vocab_size: int = 16
) -> tuple[int, ...]:
    """Deterministically map input sequence to target sequence under the rule."""
    if rule_family == ProceduralRuleFamily.CAESAR_SHIFT_MOD3:
        return tuple((s + 3) % vocab_size for s in seq)
    elif rule_family == ProceduralRuleFamily.CAESAR_SHIFT_MOD5:
        return tuple((s + 5) % vocab_size for s in seq)
    elif rule_family == ProceduralRuleFamily.CAESAR_SHIFT_MOD7:
        return tuple((s + 7) % vocab_size for s in seq)
    elif rule_family == ProceduralRuleFamily.AFFINE_TRANSFORM:
        return tuple((3 * s + 1) % vocab_size for s in seq)
    elif rule_family == ProceduralRuleFamily.AFFINE_PARAM_5_3:
        return tuple((5 * s + 3) % vocab_size for s in seq)
    elif rule_family == ProceduralRuleFamily.COMPLEMENT_REFLECT:
        return tuple((vocab_size - 1 - s) for s in seq)
    elif rule_family == ProceduralRuleFamily.NONLINEAR_SBOX:
        return tuple(SBOX_MAP[s % len(SBOX_MAP)] for s in seq)
    elif rule_family == ProceduralRuleFamily.COMPOSITE_SHIFT_REFLECT:
        # Reflect then shift mod 16
        return tuple((vocab_size - 1 - s + 4) % vocab_size for s in seq)
    elif rule_family == ProceduralRuleFamily.BITWISE_XOR_MASK:
        # Bitwise XOR with mask 9 (0b1001) mod 16
        return tuple((s ^ 9) % vocab_size for s in seq)
    elif rule_family == ProceduralRuleFamily.DECEPTIVE_INVERSION:
        # Deceptive parity inversion: position 0, 2 shift mod 3, position 1 inverts
        return tuple(
            ((s + 3) % vocab_size if idx % 2 == 0 else (vocab_size - 1 - s) % vocab_size)
            for idx, s in enumerate(seq)
        )
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
        seed: int,
        seq_length: int = 3,
        n_train: int = 48,
        n_dev: int = 16,
        n_qual: int = 16,
        n_sealed_test: int = 48,
        generalization_category: GeneralizationCategory = GeneralizationCategory.NEW_INPUTS_KNOWN_RULE,
    ) -> ProceduralTaskDataset:
        """Generate a task instance with mathematically disjoint input splits."""
        all_possible = list(
            itertools.permutations(range(self.vocab_size), seq_length)
        )
        rng = random.Random(seed)
        rng.shuffle(all_possible)

        total_needed = n_train + n_dev + n_qual + n_sealed_test
        if len(all_possible) < total_needed:
            raise ValueError(
                f"Requested {total_needed} examples, but only {len(all_possible)} available."
            )

        train_inputs = all_possible[:n_train]
        dev_inputs = all_possible[n_train : n_train + n_dev]
        qual_inputs = all_possible[n_train + n_dev : n_train + n_dev + n_qual]
        test_inputs = all_possible[
            n_train + n_dev + n_qual : total_needed
        ]

        def build_examples(
            inputs: list[tuple[int, ...]], prefix: str
        ) -> list[ProceduralExample]:
            exs = []
            for i, inp in enumerate(inputs):
                tgt = apply_procedural_rule(rule_family, inp, self.vocab_size)
                inp_str = " ".join(f"S{s}" for s in inp)
                tgt_str = " ".join(f"S{s}" for s in tgt)
                exs.append(
                    ProceduralExample(
                        example_id=f"{task_id}_{prefix}_{i}",
                        rule_family=rule_family.value,
                        input_tokens=inp,
                        target_tokens=tgt,
                        input_text=inp_str,
                        target_text=tgt_str,
                    )
                )
            return exs

        return ProceduralTaskDataset(
            task_id=task_id,
            rule_family=rule_family,
            generalization_category=generalization_category,
            seed=seed,
            train_examples=build_examples(train_inputs, "train"),
            dev_examples=build_examples(dev_inputs, "dev"),
            qualification_examples=build_examples(qual_inputs, "qual"),
            sealed_test_examples=build_examples(test_inputs, "test"),
        )

    def generate_decoupled_scrambled_task(
        self,
        task: ProceduralTaskDataset,
        seed: int,
    ) -> ProceduralTaskDataset:
        """Produce fully decoupled random target sequences breaking all correlations.

        Addresses M74 forensic audit: uniformly samples new random symbols from the 16-symbol
        vocabulary rather than permuting the true 3 target tokens, eliminating any identity
        or co-occurrence leakage.
        """
        rng = random.Random(seed)
        shuffled_train = []
        for ex in task.train_examples:
            # Independent uniform random symbols from entire alphabet
            random_tgt = tuple(rng.randrange(self.vocab_size) for _ in ex.input_tokens)
            shuffled_train.append(
                ProceduralExample(
                    example_id=f"decoupled_{ex.example_id}",
                    rule_family="DECOUPLED_RANDOM_TARGETS",
                    input_tokens=ex.input_tokens,
                    target_tokens=random_tgt,
                    input_text=ex.input_text,
                    target_text=" ".join(f"S{s}" for s in random_tgt),
                )
            )

        return ProceduralTaskDataset(
            task_id=f"scrambled_{task.task_id}",
            rule_family=task.rule_family,
            generalization_category=task.generalization_category,
            seed=seed,
            train_examples=shuffled_train,
            dev_examples=task.dev_examples,
            qualification_examples=task.qualification_examples,
            sealed_test_examples=task.sealed_test_examples,
        )
