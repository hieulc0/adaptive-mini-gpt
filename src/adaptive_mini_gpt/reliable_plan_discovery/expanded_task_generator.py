"""Phase 3: Expanded Task Generator and Identifiability Suite.

Provides balanced evaluation suites for both:
1. Supported composable tasks across noncommutative compound pairs
2. Diverse unsupported / unfamiliar procedural families (Bitwise XOR, Deceptive Parity, Modular Subtraction, Nonlinear S-Box)
With strict partition isolation between demonstrations, dev, and sealed test sets.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from adaptive_mini_gpt.neural_plan_discovery.identifiability_tasks import (
    IdentifiableTaskDataset,
    TaskFamilyClass,
)
from adaptive_mini_gpt.neural_skill_composition.compound_tasks import (
    CompoundProceduralExample,
    CompoundTaskType,
)
from adaptive_mini_gpt.neural_skill_composition.primitive_skills import PrimitiveSkillType


# Non-linear substitution box for cryptographic/unfamiliar procedural transformation
SBOX_16: tuple[int, ...] = (7, 12, 14, 9, 2, 1, 5, 15, 11, 6, 13, 0, 4, 8, 10, 3)


def apply_expanded_unfamiliar_rule(
    rule_name: str, seq: Sequence[int], vocab_size: int = 16
) -> tuple[int, ...]:
    """Execute an uncomposable procedural rule outside the primitive repertoire."""
    if rule_name == "BITWISE_XOR_MASK":
        return tuple((s ^ 9) % vocab_size for s in seq)
    elif rule_name == "DECEPTIVE_INVERSION":
        return tuple((15 - s) if (s % 2 == 0) else ((s + 7) % vocab_size) for s in seq)
    elif rule_name == "MODULAR_SUBTRACTION":
        return tuple((11 - 2 * s) % vocab_size for s in seq)
    elif rule_name == "NONLINEAR_SBOX":
        return tuple(SBOX_16[s % 16] for s in seq)
    else:
        raise ValueError(f"Unknown unfamiliar rule: {rule_name}")


class ExpandedPlanTaskGenerator:
    """Generates balanced, partition-isolated task suites for M78 evaluation."""

    UNFAMILIAR_RULES: list[str] = [
        "BITWISE_XOR_MASK",
        "DECEPTIVE_INVERSION",
        "MODULAR_SUBTRACTION",
        "NONLINEAR_SBOX",
    ]

    COMPOSABLE_TYPES: list[CompoundTaskType] = [
        CompoundTaskType.REFLECT_THEN_SHIFT3,
        CompoundTaskType.SHIFT3_THEN_REFLECT,
        CompoundTaskType.REFLECT_THEN_AFFINE,
        CompoundTaskType.AFFINE_THEN_SHIFT3,
    ]

    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size

    def generate_single_task(
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
        """Generate a single task instance with strictly partitioned splits."""
        rng = random.Random(seed)
        symbols = list(range(self.vocab_size))
        all_possible = [seq for seq in itertools.product(symbols, repeat=seq_len) if len(set(seq)) > 1]
        rng.shuffle(all_possible)

        total_needed = 8 + n_dev + n_sealed_test
        if len(all_possible) < total_needed:
            raise ValueError(f"Need {total_needed} sequences, got {len(all_possible)}")

        selected = all_possible[:total_needed]
        max_demo_seqs = selected[:8]
        dev_seqs = selected[8 : 8 + n_dev]
        test_seqs = selected[8 + n_dev : 8 + n_dev + n_sealed_test]

        demo_seqs = max_demo_seqs[:n_demo]

        if compound_type is not None:
            family_class = TaskFamilyClass.COMPOUND_COMPOSABLE
            if compound_type == CompoundTaskType.REFLECT_THEN_SHIFT3:
                gt_stages = (
                    PrimitiveSkillType.COMPLEMENT_REFLECT.value,
                    PrimitiveSkillType.CAESAR_SHIFT_MOD3.value,
                )
                map_fn = lambda seq: tuple((15 - s + 3) % self.vocab_size for s in seq)
                mid_fn = lambda seq: tuple((15 - s) % self.vocab_size for s in seq)
            elif compound_type == CompoundTaskType.SHIFT3_THEN_REFLECT:
                gt_stages = (
                    PrimitiveSkillType.CAESAR_SHIFT_MOD3.value,
                    PrimitiveSkillType.COMPLEMENT_REFLECT.value,
                )
                map_fn = lambda seq: tuple((15 - ((s + 3) % self.vocab_size)) % self.vocab_size for s in seq)
                mid_fn = lambda seq: tuple((s + 3) % self.vocab_size for s in seq)
            elif compound_type == CompoundTaskType.REFLECT_THEN_AFFINE:
                gt_stages = (
                    PrimitiveSkillType.COMPLEMENT_REFLECT.value,
                    PrimitiveSkillType.AFFINE_TRANSFORM.value,
                )
                map_fn = lambda seq: tuple((3 * (15 - s) + 1) % self.vocab_size for s in seq)
                mid_fn = lambda seq: tuple((15 - s) % self.vocab_size for s in seq)
            elif compound_type == CompoundTaskType.AFFINE_THEN_SHIFT3:
                gt_stages = (
                    PrimitiveSkillType.AFFINE_TRANSFORM.value,
                    PrimitiveSkillType.CAESAR_SHIFT_MOD3.value,
                )
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
            gt_stages = ()
            map_fn = lambda seq: apply_expanded_unfamiliar_rule(unfamiliar_rule, seq, self.vocab_size)
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
                        compound_type=compound_type.value if compound_type else (primitive_type.value if primitive_type else str(unfamiliar_rule)),
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

        demos = make_examples(demo_seqs, f"{task_id}_demo")
        devs = make_examples(dev_seqs, f"{task_id}_dev")
        tests = make_examples(test_seqs, f"{task_id}_test")

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

    def generate_balanced_suite(
        self,
        seed: int = 42,
        n_demo: int = 4,
        composable_replications: int = 4,
        uncomposable_replications: int = 2,
    ) -> list[IdentifiableTaskDataset]:
        """Generate a balanced task suite for a seed.

        Returns 16 composable tasks (4 compound types x 4 replications)
        and 8 uncomposable tasks (4 unfamiliar rules x 2 replications).
        Total: 24 tasks per seed.
        """
        suite: list[IdentifiableTaskDataset] = []

        # 1. Composable tasks
        for comp_type in self.COMPOSABLE_TYPES:
            for rep in range(composable_replications):
                task_id = f"comp_{comp_type.value}_rep{rep}_s{seed}"
                task = self.generate_single_task(
                    task_id=task_id,
                    compound_type=comp_type,
                    seed=seed * 1000 + hash(comp_type.value) % 500 + rep,
                    n_demo=n_demo,
                )
                suite.append(task)

        # 2. Uncomposable tasks
        for rule_name in self.UNFAMILIAR_RULES:
            for rep in range(uncomposable_replications):
                task_id = f"uncomp_{rule_name.lower()}_rep{rep}_s{seed}"
                task = self.generate_single_task(
                    task_id=task_id,
                    unfamiliar_rule=rule_name,
                    seed=seed * 2000 + hash(rule_name) % 500 + rep,
                    n_demo=n_demo,
                )
                suite.append(task)

        return suite
