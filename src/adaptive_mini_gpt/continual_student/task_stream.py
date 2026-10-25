"""Milestone M86: Continual Task Stream Generator.

Generates a controlled, diverse sequence of continual learning tasks:
1. Novel Primitive Skills (requiring initial acquisition).
2. Supported / Duplicate Skills (testing zero-shot reuse).
3. Compound Compositional Skills (testing multi-skill execution).
4. Capacity-Demanding Skills (testing dynamic rank expansion).
5. Overlapping Skills (testing knowledge consolidation).
6. Conflicting Skills (testing conflict detection and trial rollback).
7. Simple Compressible Skills (testing rank shrinkage and parameter recovery).
8. Long-Horizon Revisit Skills (testing backward retention / catastrophic forgetting).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.meta_acquisition.environment import (
    MetaEnvironmentGenerator,
    MetaTaskDataset,
    SplitType,
    TaskFamily,
)


@dataclass
class ContinualTaskEpisode:
    """Specification of an individual task episode in the lifelong stream."""

    episode_index: int
    task_id: str
    task_family: TaskFamily
    dataset: MetaTaskDataset
    intended_regime: str
    expected_optimal_action: str
    target_em: float = 0.85


class ContinualTaskStreamGenerator:
    """Generates reproducible, deterministic multi-task continual learning streams."""

    @classmethod
    def generate_stream(
        cls,
        seed: int = 42,
        num_episodes: int = 10,
        vocab_size: int = 16,
        seq_len: int = 4,
    ) -> list[ContinualTaskEpisode]:
        """Construct a structured multi-regime continual learning stream."""
        env_gen = MetaEnvironmentGenerator(vocab_size=vocab_size, seq_len=seq_len)
        episodes: list[ContinualTaskEpisode] = []

        # Episode 0: Novel Primitive 1 - Modular Nonlinear
        t0 = env_gen.generate_task(
            task_id="modular_nonlinear_1",
            family=TaskFamily.MODULAR_NONLINEAR,
            split=SplitType.META_TRAIN,
            rule_params={"a": 3, "b": 5, "c": 2},
            rule_instance_id="NONLINEAR_3s5_2",
            seed=seed + 100,
        )
        episodes.append(
            ContinualTaskEpisode(
                episode_index=0,
                task_id="modular_nonlinear_1",
                task_family=TaskFamily.MODULAR_NONLINEAR,
                dataset=t0,
                intended_regime="NOVEL_PRIMITIVE_ACQUISITION",
                expected_optimal_action="ACQUIRE_NEW_SKILL",
            )
        )

        # Episode 1: Novel Primitive 2 - Parity Logic (mask=6)
        t1 = env_gen.generate_task(
            task_id="parity_prim_2",
            family=TaskFamily.PARITY_LOGIC,
            split=SplitType.META_TRAIN,
            rule_params={"mask": 6},
            rule_instance_id="PARITY_M6",
            seed=seed + 101,
        )
        episodes.append(
            ContinualTaskEpisode(
                episode_index=1,
                task_id="parity_prim_2",
                task_family=TaskFamily.PARITY_LOGIC,
                dataset=t1,
                intended_regime="NOVEL_PRIMITIVE_ACQUISITION",
                expected_optimal_action="ACQUIRE_NEW_SKILL",
            )
        )

        # Episode 2: Supported Skill - Duplicate of Episode 0 (different seed)
        t2 = env_gen.generate_task(
            task_id="modular_nonlinear_1_dup",
            family=TaskFamily.MODULAR_NONLINEAR,
            split=SplitType.META_VAL,
            rule_params={"a": 3, "b": 5, "c": 2},
            rule_instance_id="NONLINEAR_3s5_2",
            seed=seed + 102,
        )
        episodes.append(
            ContinualTaskEpisode(
                episode_index=2,
                task_id="modular_nonlinear_1_dup",
                task_family=TaskFamily.MODULAR_NONLINEAR,
                dataset=t2,
                intended_regime="ZERO_SHOT_REUSE",
                expected_optimal_action="REUSE_OR_COMPOSE",
            )
        )

        # Episode 3: Capacity-Demanding Skill - Complex Two-Stage Cascade
        t3 = env_gen.generate_task(
            task_id="cascade_complex",
            family=TaskFamily.TWO_STAGE_CASCADE,
            split=SplitType.META_TRAIN,
            rule_params={"a": 3, "b": 7, "mask": 6, "offset": 2},
            rule_instance_id="CASCADE_3s7_M6",
            seed=seed + 103,
        )
        episodes.append(
            ContinualTaskEpisode(
                episode_index=3,
                task_id="cascade_complex",
                task_family=TaskFamily.TWO_STAGE_CASCADE,
                dataset=t3,
                intended_regime="CAPACITY_EXPANSION",
                expected_optimal_action="EXPAND_CAPACITY",
            )
        )

        # Episode 4: Compressible Skill - Nonlinear Polynomial with Low Complexity
        t4 = env_gen.generate_task(
            task_id="modular_nonlinear_simple",
            family=TaskFamily.MODULAR_NONLINEAR,
            split=SplitType.META_TRAIN,
            rule_params={"a": 3, "b": 5, "c": 2},
            rule_instance_id="NONLINEAR_3s5_2",
            seed=seed + 104,
        )
        episodes.append(
            ContinualTaskEpisode(
                episode_index=4,
                task_id="modular_nonlinear_simple",
                task_family=TaskFamily.MODULAR_NONLINEAR,
                dataset=t4,
                intended_regime="CAPACITY_SHRINKAGE",
                expected_optimal_action="CONSOLIDATE_SHRINK",
            )
        )

        # Episode 5: Overlapping Related Skill - Shared slope, different offset
        t5 = env_gen.generate_task(
            task_id="affine_prim_related",
            family=TaskFamily.ARITHMETIC_AFFINE,
            split=SplitType.META_TRAIN,
            rule_params={"a": 3, "b": 9},
            rule_instance_id="AFFINE_3s9",
            seed=seed + 105,
        )
        episodes.append(
            ContinualTaskEpisode(
                episode_index=5,
                task_id="affine_prim_related",
                task_family=TaskFamily.ARITHMETIC_AFFINE,
                dataset=t5,
                intended_regime="KNOWLEDGE_CONSOLIDATION",
                expected_optimal_action="CONSOLIDATE_SHRINK",
            )
        )

        # Episode 6: Conflicting Opposing Skill - Incompatible branching rule
        t6 = env_gen.generate_task(
            task_id="parity_opposing_conflict",
            family=TaskFamily.PARITY_LOGIC,
            split=SplitType.META_VAL,
            rule_params={"mask": 1},
            rule_instance_id="PARITY_M1_OPP",
            seed=seed + 106,
        )
        episodes.append(
            ContinualTaskEpisode(
                episode_index=6,
                task_id="parity_opposing_conflict",
                task_family=TaskFamily.PARITY_LOGIC,
                dataset=t6,
                intended_regime="CONFLICT_REJECTION_ROLLBACK",
                expected_optimal_action="ACQUIRE_NEW_SKILL",
            )
        )

        # Episode 7: Long-Horizon Revisit - Revisit Episode 0 Skill to test forgetting
        t7 = env_gen.generate_task(
            task_id="modular_nonlinear_1_revisit",
            family=TaskFamily.MODULAR_NONLINEAR,
            split=SplitType.META_TEST,
            rule_params={"a": 3, "b": 5, "c": 2},
            rule_instance_id="NONLINEAR_3s5_2",
            seed=seed + 107,
        )
        episodes.append(
            ContinualTaskEpisode(
                episode_index=7,
                task_id="modular_nonlinear_1_revisit",
                task_family=TaskFamily.MODULAR_NONLINEAR,
                dataset=t7,
                intended_regime="BACKWARD_RETENTION_PROBE",
                expected_optimal_action="REUSE_OR_COMPOSE",
            )
        )

        # Episode 8: Long-Horizon Revisit - Revisit Episode 1 Skill to test forgetting
        t8 = env_gen.generate_task(
            task_id="parity_prim_2_revisit",
            family=TaskFamily.PARITY_LOGIC,
            split=SplitType.META_TEST,
            rule_params={"mask": 6},
            rule_instance_id="PARITY_M6",
            seed=seed + 108,
        )
        episodes.append(
            ContinualTaskEpisode(
                episode_index=8,
                task_id="parity_prim_2_revisit",
                task_family=TaskFamily.PARITY_LOGIC,
                dataset=t8,
                intended_regime="BACKWARD_RETENTION_PROBE",
                expected_optimal_action="REUSE_OR_COMPOSE",
            )
        )

        # Episode 9: Novel Unseen Transfer Task - Permutation Inversion
        t9 = env_gen.generate_task(
            task_id="permutation_transfer",
            family=TaskFamily.PERMUTATION_INVERSION,
            split=SplitType.META_TEST,
            rule_params={"shift": 3},
            rule_instance_id="PERM_INV_K3",
            seed=seed + 109,
        )
        episodes.append(
            ContinualTaskEpisode(
                episode_index=9,
                task_id="permutation_transfer",
                task_family=TaskFamily.PERMUTATION_INVERSION,
                dataset=t9,
                intended_regime="FORWARD_TRANSFER_EVALUATION",
                expected_optimal_action="ACQUIRE_NEW_SKILL",
            )
        )

        return episodes[:num_episodes]
