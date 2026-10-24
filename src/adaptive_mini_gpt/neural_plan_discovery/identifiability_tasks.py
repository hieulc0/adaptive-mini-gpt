"""Phase 3 & 7: Plan Identifiability Benchmark & Procedural Tasks Generator.

Generates tasks with controlled demonstration budgets (N in {0, 2, 4, 8}),
including:
1. Composable noncommutative tasks (Reflect -> Shift, Shift -> Reflect, Reflect -> Affine, Affine -> Shift)
2. Single-primitive tasks (Reflect, Shift, Affine)
3. Unfamiliar / uncomposable procedural families (Bitwise XOR, Deceptive Parity Inversion)
4. Demonstration sets, development sets, and sealed test sets with strict partition isolation.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from adaptive_mini_gpt.neural_skill_composition.compound_tasks import (
    CompoundProceduralExample,
    CompoundTaskType,
)
from adaptive_mini_gpt.neural_skill_composition.primitive_skills import PrimitiveSkillType


class TaskFamilyClass(StrEnum):
    """Broad class of procedural task."""

    PRIMITIVE_ATOMIC = "PRIMITIVE_ATOMIC"
    COMPOUND_COMPOSABLE = "COMPOUND_COMPOSABLE"
    UNFAMILIAR_UNCOMPOSABLE = "UNFAMILIAR_UNCOMPOSABLE"


@dataclass
class IdentifiableTaskDataset:
    """A procedural task with observable demonstrations and sealed test set."""

    task_id: str
    family_class: TaskFamilyClass
    ground_truth_plan_stages: tuple[str, ...]
    demonstrations: list[CompoundProceduralExample]
    development_examples: list[CompoundProceduralExample]
    sealed_test_examples: list[CompoundProceduralExample]
    vocab_size: int = 16
    seq_len: int = 3


def apply_unfamiliar_rule(rule_name: str, seq: Sequence[int], vocab_size: int = 16) -> tuple[int, ...]:
    """Execute an uncomposable procedural rule (cannot be decomposed into Reflect/Shift3/Affine)."""
    if rule_name == "BITWISE_XOR_MASK":
        # Nonlinear bitwise XOR with 9 mod vocab_size
        return tuple((s ^ 9) % vocab_size for s in seq)
    elif rule_name == "DECEPTIVE_INVERSION":
        # Conditional non-affine rule: reflect even symbols, modular shift odd symbols
        return tuple((15 - s) if (s % 2 == 0) else ((s + 7) % vocab_size) for s in seq)
    else:
        raise ValueError(f"Unknown unfamiliar rule: {rule_name}")


class PlanIdentifiabilityGenerator:
    """Generates identifiable procedural tasks with varying demonstration budgets."""

    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size

    def generate_task(
        self,
        task_id: str,
        compound_type: CompoundTaskType | None = None,
        primitive_type: PrimitiveSkillType | None = None,
        unfamiliar_rule: str | None = None,
        seed: int = 42,
        seq_len: int = 3,
        n_demo: int = 4,
        n_dev: int = 16,
        n_sealed_test: int = 48,
    ) -> IdentifiableTaskDataset:
        """Generate a task instance with strict partition isolation."""
        rng = random.Random(seed)
        symbols = list(range(self.vocab_size))
        all_possible = [seq for seq in itertools.product(symbols, repeat=seq_len) if len(set(seq)) > 1]
        rng.shuffle(all_possible)

        total_needed = 8 + n_dev + n_sealed_test  # Max 8 demonstrations
        if len(all_possible) < total_needed:
            raise ValueError(f"Need {total_needed} sequences, got {len(all_possible)}")

        selected = all_possible[:total_needed]
        max_demo_seqs = selected[:8]
        dev_seqs = selected[8 : 8 + n_dev]
        test_seqs = selected[8 + n_dev : 8 + n_dev + n_sealed_test]

        demo_seqs = max_demo_seqs[:n_demo]

        # Determine target mapping and ground truth plan
        if compound_type is not None:
            family_class = TaskFamilyClass.COMPOUND_COMPOSABLE
            if compound_type == CompoundTaskType.REFLECT_THEN_SHIFT3:
                gt_stages = (PrimitiveSkillType.COMPLEMENT_REFLECT.value, PrimitiveSkillType.CAESAR_SHIFT_MOD3.value)
                map_fn = lambda seq: tuple((15 - s + 3) % self.vocab_size for s in seq)
                mid_fn = lambda seq: tuple((15 - s) % self.vocab_size for s in seq)
            elif compound_type == CompoundTaskType.SHIFT3_THEN_REFLECT:
                gt_stages = (PrimitiveSkillType.CAESAR_SHIFT_MOD3.value, PrimitiveSkillType.COMPLEMENT_REFLECT.value)
                map_fn = lambda seq: tuple((15 - ((s + 3) % self.vocab_size)) % self.vocab_size for s in seq)
                mid_fn = lambda seq: tuple((s + 3) % self.vocab_size for s in seq)
            elif compound_type == CompoundTaskType.REFLECT_THEN_AFFINE:
                gt_stages = (PrimitiveSkillType.COMPLEMENT_REFLECT.value, PrimitiveSkillType.AFFINE_TRANSFORM.value)
                map_fn = lambda seq: tuple((3 * (15 - s) + 1) % self.vocab_size for s in seq)
                mid_fn = lambda seq: tuple((15 - s) % self.vocab_size for s in seq)
            elif compound_type == CompoundTaskType.AFFINE_THEN_SHIFT3:
                gt_stages = (PrimitiveSkillType.AFFINE_TRANSFORM.value, PrimitiveSkillType.CAESAR_SHIFT_MOD3.value)
                map_fn = lambda seq: tuple(((3 * s + 1) + 3) % self.vocab_size for s in seq)
                mid_fn = lambda seq: tuple((3 * s + 1) % self.vocab_size for s in seq)
            else:
                raise ValueError(f"Unsupported compound type: {compound_type}")

        elif primitive_type is not None:
            family_class = TaskFamilyClass.PRIMITIVE_ATOMIC
            gt_stages = (primitive_type.value,)
            if primitive_type == PrimitiveSkillType.COMPLEMENT_REFLECT:
                map_fn = lambda seq: tuple((15 - s) % self.vocab_size for s in seq)
            elif primitive_type == PrimitiveSkillType.CAESAR_SHIFT_MOD3:
                map_fn = lambda seq: tuple((s + 3) % self.vocab_size for s in seq)
            elif primitive_type == PrimitiveSkillType.AFFINE_TRANSFORM:
                map_fn = lambda seq: tuple((3 * s + 1) % self.vocab_size for s in seq)
            else:
                raise ValueError(f"Unsupported primitive type: {primitive_type}")
            mid_fn = map_fn

        elif unfamiliar_rule is not None:
            family_class = TaskFamilyClass.UNFAMILIAR_UNCOMPOSABLE
            gt_stages = ()  # Incompatible / no valid composition in repertoire
            map_fn = lambda seq: apply_unfamiliar_rule(unfamiliar_rule, seq, self.vocab_size)
            mid_fn = map_fn
        else:
            raise ValueError("Must specify compound_type, primitive_type, or unfamiliar_rule")

        def make_examples(seq_list: list[tuple[int, ...]], prefix: str) -> list[CompoundProceduralExample]:
            exs = []
            for idx, seq in enumerate(seq_list):
                tgt = map_fn(seq)
                mid = mid_fn(seq)
                in_str = " ".join(f"S{s}" for s in seq)
                tgt_str = " ".join(f"S{s}" for s in tgt)
                mid_str = " ".join(f"S{s}" for s in mid)
                exs.append(
                    CompoundProceduralExample(
                        example_id=f"{prefix}_{idx}",
                        compound_type=compound_type.value if compound_type else (primitive_type.value if primitive_type else unfamiliar_rule),
                        stage1_primitive=gt_stages[0] if len(gt_stages) >= 1 else "NONE",
                        stage2_primitive=gt_stages[1] if len(gt_stages) >= 2 else "NONE",
                        input_tokens=seq,
                        intermediate_tokens=mid,
                        target_tokens=tgt,
                        input_text=f"Task: {task_id} | Input: {in_str} -> Output:",
                        intermediate_text=mid_str,
                        target_text=tgt_str,
                    )
                )
            return exs

        demos = make_examples(demo_seqs, "demo")
        devs = make_examples(dev_seqs, "dev")
        tests = make_examples(test_seqs, "sealed_test")

        return IdentifiableTaskDataset(
            task_id=task_id,
            family_class=family_class,
            ground_truth_plan_stages=gt_stages,
            demonstrations=demos,
            development_examples=devs,
            sealed_test_examples=tests,
            vocab_size=self.vocab_size,
            seq_len=seq_len,
        )

    @classmethod
    def create_shuffled_demonstrations(
        cls,
        demonstrations: list[CompoundProceduralExample],
        seed: int = 999,
    ) -> list[CompoundProceduralExample]:
        """Scramble target tokens across demonstration examples, breaking input-output pairing."""
        if len(demonstrations) <= 1:
            return list(demonstrations)

        rng = random.Random(seed)
        shuffled_targets = [d.target_tokens for d in demonstrations]
        rng.shuffle(shuffled_targets)

        # Ensure no identical assignment if possible
        if shuffled_targets == [d.target_tokens for d in demonstrations]:
            shuffled_targets = shuffled_targets[1:] + shuffled_targets[:1]

        scrambled: list[CompoundProceduralExample] = []
        for d, new_tgt in zip(demonstrations, shuffled_targets):
            tgt_str = " ".join(f"S{s}" for s in new_tgt)
            scrambled.append(
                CompoundProceduralExample(
                    example_id=f"shuffled_{d.example_id}",
                    compound_type=d.compound_type,
                    stage1_primitive=d.stage1_primitive,
                    stage2_primitive=d.stage2_primitive,
                    input_tokens=d.input_tokens,
                    intermediate_tokens=d.intermediate_tokens,
                    target_tokens=new_tgt,
                    input_text=d.input_text,
                    intermediate_text=d.intermediate_text,
                    target_text=tgt_str,
                )
            )
        return scrambled
