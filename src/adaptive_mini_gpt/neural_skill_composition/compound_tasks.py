"""Phase 3 & 6: Noncommutative Compound Procedural Sequence Tasks.

Implements:
- Two-stage procedural composition tasks over discrete 16-symbol alphabet:
    1. REFLECT_THEN_SHIFT3: s -> 15 - s -> (15 - s + 3) mod 16 = (2 - s) mod 16
    2. SHIFT3_THEN_REFLECT: s -> (s + 3) mod 16 -> 15 - ((s + 3) mod 16) = (12 - s) mod 16
       (Strictly noncommutative: REFLECT_THEN_SHIFT3 != SHIFT3_THEN_REFLECT)
    3. REFLECT_THEN_AFFINE: s -> 15 - s -> (3 * (15 - s) + 1) mod 16 = (14 - 3*s) mod 16
    4. AFFINE_THEN_SHIFT3: s -> (3*s + 1) mod 16 -> ((3*s + 1) + 3) mod 16 = (3*s + 4) mod 16
- Tracking of ground-truth intermediate tokens for exact error diagnosis (Stage 1 vs Stage 2 vs Propagation).
- Strict partition isolation enforcing disjoint input sequences across train, dev, qualification, and sealed_test.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from enum import StrEnum

from adaptive_mini_gpt.neural_skill_composition.primitive_skills import PrimitiveSkillType


class CompoundTaskType(StrEnum):
    """Supported two-stage compound procedural tasks."""

    REFLECT_THEN_SHIFT3 = "REFLECT_THEN_SHIFT3"
    SHIFT3_THEN_REFLECT = "SHIFT3_THEN_REFLECT"
    REFLECT_THEN_AFFINE = "REFLECT_THEN_AFFINE"
    AFFINE_THEN_SHIFT3 = "AFFINE_THEN_SHIFT3"


@dataclass(frozen=True)
class CompoundProceduralExample:
    """A compound sequence example tracking input, intermediate, and final tokens."""

    example_id: str
    compound_type: str
    stage1_primitive: str
    stage2_primitive: str
    input_tokens: tuple[int, ...]
    intermediate_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    input_text: str
    intermediate_text: str
    target_text: str


@dataclass
class CompoundTaskDataset:
    """Disjoint dataset split for a compound procedural task."""

    task_id: str
    compound_type: CompoundTaskType
    stage1_primitive: PrimitiveSkillType
    stage2_primitive: PrimitiveSkillType
    seed: int
    train_examples: list[CompoundProceduralExample]
    dev_examples: list[CompoundProceduralExample]
    qualification_examples: list[CompoundProceduralExample]
    sealed_test_examples: list[CompoundProceduralExample]


def apply_primitive_mapping(
    skill_type: PrimitiveSkillType, seq: tuple[int, ...], vocab_size: int = 16
) -> tuple[int, ...]:
    """Deterministically map sequence under a single primitive rule."""
    if skill_type == PrimitiveSkillType.COMPLEMENT_REFLECT:
        return tuple((vocab_size - 1 - s) for s in seq)
    elif skill_type == PrimitiveSkillType.CAESAR_SHIFT_MOD3:
        return tuple((s + 3) % vocab_size for s in seq)
    elif skill_type == PrimitiveSkillType.AFFINE_TRANSFORM:
        return tuple((3 * s + 1) % vocab_size for s in seq)
    else:
        raise ValueError(f"Unknown primitive skill type: {skill_type}")


class CompoundTaskGenerator:
    """Generates two-stage compound procedural tasks with isolated partitions."""

    def __init__(self, vocab_size: int = 16) -> None:
        self.vocab_size = vocab_size

    def generate_compound_task(
        self,
        task_id: str,
        compound_type: CompoundTaskType,
        seed: int,
        seq_length: int = 3,
        n_train: int = 48,
        n_dev: int = 16,
        n_qual: int = 16,
        n_sealed_test: int = 48,
    ) -> CompoundTaskDataset:
        """Generate compound task instance with ground-truth intermediate tracking."""
        # Determine primitive chain
        if compound_type == CompoundTaskType.REFLECT_THEN_SHIFT3:
            stage1_prim = PrimitiveSkillType.COMPLEMENT_REFLECT
            stage2_prim = PrimitiveSkillType.CAESAR_SHIFT_MOD3
        elif compound_type == CompoundTaskType.SHIFT3_THEN_REFLECT:
            stage1_prim = PrimitiveSkillType.CAESAR_SHIFT_MOD3
            stage2_prim = PrimitiveSkillType.COMPLEMENT_REFLECT
        elif compound_type == CompoundTaskType.REFLECT_THEN_AFFINE:
            stage1_prim = PrimitiveSkillType.COMPLEMENT_REFLECT
            stage2_prim = PrimitiveSkillType.AFFINE_TRANSFORM
        elif compound_type == CompoundTaskType.AFFINE_THEN_SHIFT3:
            stage1_prim = PrimitiveSkillType.AFFINE_TRANSFORM
            stage2_prim = PrimitiveSkillType.CAESAR_SHIFT_MOD3
        else:
            raise ValueError(f"Unknown compound type: {compound_type}")

        all_possible = list(itertools.permutations(range(self.vocab_size), seq_length))
        rng = random.Random(seed)
        rng.shuffle(all_possible)

        total_needed = n_train + n_dev + n_qual + n_sealed_test
        if len(all_possible) < total_needed:
            raise ValueError(f"Need {total_needed} examples, but only {len(all_possible)} available.")

        train_inputs = all_possible[:n_train]
        dev_inputs = all_possible[n_train : n_train + n_dev]
        qual_inputs = all_possible[n_train + n_dev : n_train + n_dev + n_qual]
        test_inputs = all_possible[n_train + n_dev + n_qual : total_needed]

        def build_examples(
            inputs: list[tuple[int, ...]], prefix: str
        ) -> list[CompoundProceduralExample]:
            exs = []
            for i, inp in enumerate(inputs):
                inter = apply_primitive_mapping(stage1_prim, inp, self.vocab_size)
                final = apply_primitive_mapping(stage2_prim, inter, self.vocab_size)
                exs.append(
                    CompoundProceduralExample(
                        example_id=f"{task_id}_{prefix}_{i}",
                        compound_type=compound_type.value,
                        stage1_primitive=stage1_prim.value,
                        stage2_primitive=stage2_prim.value,
                        input_tokens=inp,
                        intermediate_tokens=inter,
                        target_tokens=final,
                        input_text=" ".join(f"S{s}" for s in inp),
                        intermediate_text=" ".join(f"S{s}" for s in inter),
                        target_text=" ".join(f"S{s}" for s in final),
                    )
                )
            return exs

        return CompoundTaskDataset(
            task_id=task_id,
            compound_type=compound_type,
            stage1_primitive=stage1_prim,
            stage2_primitive=stage2_prim,
            seed=seed,
            train_examples=build_examples(train_inputs, "train"),
            dev_examples=build_examples(dev_inputs, "dev"),
            qualification_examples=build_examples(qual_inputs, "qual"),
            sealed_test_examples=build_examples(test_inputs, "test"),
        )
