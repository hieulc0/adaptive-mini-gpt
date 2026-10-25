"""Milestone M87: 100-Episode Continual Learning Task Stream Generator.

Constructs an extended, reproducible, 100-episode curriculum divided into 4 phases:
- Phase 1 (Episodes 0–24): Foundational procedural skill acquisition across 9 families.
- Phase 2 (Episodes 25–49): Knowledge reuse, exact duplicates, and multi-stage cascades.
- Phase 3 (Episodes 50–74): Opposing conflicts, dynamic rank expansion/shrinkage, and capacity pressure.
- Phase 4 (Episodes 75–99): Long-delayed revisits (delay >= 50 episodes), autonomous retirement, and lifelong plasticity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.continual_student.task_stream import ContinualTaskEpisode
from adaptive_mini_gpt.meta_acquisition.environment import (
    MetaEnvironmentGenerator,
    MetaTaskDataset,
    SplitType,
    TaskFamily,
)


@dataclass
class StreamPhaseInfo:
    """Metadata describing each 25-episode phase of the curriculum."""

    phase_index: int
    name: str
    episode_range: tuple[int, int]
    core_objectives: list[str]


class ContinualTaskStream100Generator:
    """Generates a frozen 100-episode continual task curriculum across independent seeds."""

    PHASE_DEFINITIONS = [
        StreamPhaseInfo(
            phase_index=1,
            name="Foundational Skill Acquisition",
            episode_range=(0, 24),
            core_objectives=[
                "Acquire diverse procedural primitives across arithmetic, logic, and permutation",
                "Establish baseline repertoire with rank-4/8 adapters",
            ],
        ),
        StreamPhaseInfo(
            phase_index=2,
            name="Reuse and Compound Composition",
            episode_range=(25, 49),
            core_objectives=[
                "Autonomously detect supported/duplicate tasks for zero-shot reuse",
                "Compose multi-stage rules without retraining existing components",
            ],
        ),
        StreamPhaseInfo(
            phase_index=3,
            name="Conflicts and Capacity Pressure",
            episode_range=(50, 74),
            core_objectives=[
                "Detect opposing contradictory rules and execute bit-exact rollback",
                "Dynamically expand rank on complex cascades and shrink on simple rules",
                "Handle capacity pressure approaching the 20,000 parameter budget",
            ],
        ),
        StreamPhaseInfo(
            phase_index=4,
            name="Lifelong Adaptation and Long-Delayed Revisits",
            episode_range=(75, 99),
            core_objectives=[
                "Measure backward transfer and retention on tasks revisited after >= 50 episodes",
                "Retire obsolete or redundant capacity to maintain budget compliance",
                "Acquire late-arriving skills under strict capacity constraints",
            ],
        ),
    ]

    @classmethod
    def generate_stream(
        cls,
        seed: int = 42,
        num_episodes: int = 100,
        vocab_size: int = 16,
        seq_len: int = 4,
    ) -> list[ContinualTaskEpisode]:
        """Construct the frozen 100-episode lifelong curriculum."""
        env_gen = MetaEnvironmentGenerator(vocab_size=vocab_size, seq_len=seq_len)
        episodes: list[ContinualTaskEpisode] = []

        # Cache of foundational task datasets for exact and near duplicate reuse
        foundational_cache: dict[str, MetaTaskDataset] = {}

        # ---------------------------------------------------------------------
        # PHASE 1: EPISODES 0–24 (Foundational Acquisition)
        # ---------------------------------------------------------------------
        # Families: AFFINE (5), PARITY (5), NONLINEAR (5), QUADRATIC (3),
        #           TRANSPOSITION (3), REFLECT (2), CASCADE (2) -> 25 tasks
        phase1_specs: list[tuple[str, TaskFamily, dict[str, Any], str]] = [
            # Affine family (5)
            ("affine_a3_b1", TaskFamily.ARITHMETIC_AFFINE, {"a": 3, "b": 1}, "AFF_3_1"),
            ("affine_a5_b2", TaskFamily.ARITHMETIC_AFFINE, {"a": 5, "b": 2}, "AFF_5_2"),
            ("affine_a7_b3", TaskFamily.ARITHMETIC_AFFINE, {"a": 7, "b": 3}, "AFF_7_3"),
            ("affine_a3_b5", TaskFamily.ARITHMETIC_AFFINE, {"a": 3, "b": 5}, "AFF_3_5"),
            ("affine_a5_b7", TaskFamily.ARITHMETIC_AFFINE, {"a": 5, "b": 7}, "AFF_5_7"),
            # Parity logic family (5)
            ("parity_m2_o1", TaskFamily.PARITY_LOGIC, {"mask": 2, "offset": 1}, "PAR_2_1"),
            ("parity_m4_o3", TaskFamily.PARITY_LOGIC, {"mask": 4, "offset": 3}, "PAR_4_3"),
            ("parity_m6_o5", TaskFamily.PARITY_LOGIC, {"mask": 6, "offset": 5}, "PAR_6_5"),
            ("parity_m3_o2", TaskFamily.PARITY_LOGIC, {"mask": 3, "offset": 2}, "PAR_3_2"),
            ("parity_m5_o4", TaskFamily.PARITY_LOGIC, {"mask": 5, "offset": 4}, "PAR_5_4"),
            # Modular nonlinear family (5)
            ("nonlin_m3_k2", TaskFamily.MODULAR_NONLINEAR, {"mult": 3, "mask": 2}, "NL_3_2"),
            ("nonlin_m5_k3", TaskFamily.MODULAR_NONLINEAR, {"mult": 5, "mask": 3}, "NL_5_3"),
            ("nonlin_m3_k5", TaskFamily.MODULAR_NONLINEAR, {"mult": 3, "mask": 5}, "NL_3_5"),
            ("nonlin_m7_k2", TaskFamily.MODULAR_NONLINEAR, {"mult": 7, "mask": 2}, "NL_7_2"),
            ("nonlin_m5_k7", TaskFamily.MODULAR_NONLINEAR, {"mult": 5, "mask": 7}, "NL_5_7"),
            # Modular quadratic family (3)
            ("quad_a1_b2_c1", TaskFamily.MODULAR_QUADRATIC, {"a": 1, "b": 2, "c": 1}, "QD_1_2_1"),
            ("quad_a2_b1_c3", TaskFamily.MODULAR_QUADRATIC, {"a": 2, "b": 1, "c": 3}, "QD_2_1_3"),
            ("quad_a1_b3_c2", TaskFamily.MODULAR_QUADRATIC, {"a": 1, "b": 3, "c": 2}, "QD_1_3_2"),
            # Block transposition family (3)
            ("transp_o1", TaskFamily.BLOCK_TRANSPOSITION, {"offset": 1}, "TR_1"),
            ("transp_o3", TaskFamily.BLOCK_TRANSPOSITION, {"offset": 3}, "TR_3"),
            ("transp_o5", TaskFamily.BLOCK_TRANSPOSITION, {"offset": 5}, "TR_5"),
            # Palindrome reflection family (2)
            ("reflect_o0", TaskFamily.PALINDROME_REFLECT, {"offset": 0}, "REF_0"),
            ("reflect_o2", TaskFamily.PALINDROME_REFLECT, {"offset": 2}, "REF_2"),
            # Two-stage cascade family (2)
            ("cascade_aff_par", TaskFamily.TWO_STAGE_CASCADE, {"step1": "AFF_3_1", "step2": "PAR_2_1"}, "CAS_1"),
            ("cascade_nl_transp", TaskFamily.TWO_STAGE_CASCADE, {"step1": "NL_3_2", "step2": "TR_1"}, "CAS_2"),
        ]

        for ep_idx, (t_id, fam, params, r_id) in enumerate(phase1_specs):
            ds = env_gen.generate_task(
                task_id=t_id,
                family=fam,
                split=SplitType.META_TRAIN,
                rule_params=params,
                rule_instance_id=r_id,
                seed=seed + ep_idx * 13,
            )
            foundational_cache[t_id] = ds
            episodes.append(
                ContinualTaskEpisode(
                    episode_index=ep_idx,
                    task_id=t_id,
                    task_family=fam,
                    dataset=ds,
                    intended_regime="FOUNDATIONAL_ACQUISITION",
                    expected_optimal_action="ACQUIRE_NEW_SKILL",
                )
            )

        # ---------------------------------------------------------------------
        # PHASE 2: EPISODES 25–49 (Reuse & Compound Composition)
        # ---------------------------------------------------------------------
        phase1_keys = list(foundational_cache.keys())
        for ep_idx in range(25, 50):
            # Alternate between Exact Duplicate Reuse, Cascades, and Related Variants
            pattern = ep_idx % 3
            if pattern == 0:
                # Exact Duplicate of a foundational task (different random sample seed)
                source_key = phase1_keys[(ep_idx - 25) % len(phase1_keys)]
                orig_ds = foundational_cache[source_key]
                dup_ds = env_gen.generate_task(
                    task_id=f"{source_key}_dup_ep{ep_idx}",
                    family=orig_ds.family,
                    split=SplitType.META_VAL,
                    rule_params=orig_ds.rule_params,
                    rule_instance_id=orig_ds.rule_instance_id,
                    seed=seed + ep_idx * 17,
                )
                episodes.append(
                    ContinualTaskEpisode(
                        episode_index=ep_idx,
                        task_id=f"{source_key}_dup_ep{ep_idx}",
                        task_family=orig_ds.family,
                        dataset=dup_ds,
                        intended_regime="ZERO_SHOT_REUSE",
                        expected_optimal_action="REUSE_OR_COMPOSE",
                    )
                )
            elif pattern == 1:
                # Novel multi-stage cascade composition
                cas_id = f"cascade_compound_{ep_idx}"
                cas_ds = env_gen.generate_task(
                    task_id=cas_id,
                    family=TaskFamily.TWO_STAGE_CASCADE,
                    split=SplitType.META_VAL,
                    rule_params={"step1": "AFF_5_2", "step2": "NL_3_2"},
                    rule_instance_id=f"CAS_CMP_{ep_idx}",
                    seed=seed + ep_idx * 19,
                )
                episodes.append(
                    ContinualTaskEpisode(
                        episode_index=ep_idx,
                        task_id=cas_id,
                        task_family=TaskFamily.TWO_STAGE_CASCADE,
                        dataset=cas_ds,
                        intended_regime="COMPOUND_COMPOSITION",
                        expected_optimal_action="ACQUIRE_NEW_SKILL",
                    )
                )
            else:
                # Related procedural variant with parameter shift
                var_id = f"affine_variant_{ep_idx}"
                var_ds = env_gen.generate_task(
                    task_id=var_id,
                    family=TaskFamily.ARITHMETIC_AFFINE,
                    split=SplitType.META_VAL,
                    rule_params={"a": 3, "b": (ep_idx % 7) + 1},
                    rule_instance_id=f"AFF_VAR_{ep_idx}",
                    seed=seed + ep_idx * 23,
                )
                episodes.append(
                    ContinualTaskEpisode(
                        episode_index=ep_idx,
                        task_id=var_id,
                        task_family=TaskFamily.ARITHMETIC_AFFINE,
                        dataset=var_ds,
                        intended_regime="RELATED_VARIANT_ADAPTATION",
                        expected_optimal_action="ACQUIRE_NEW_SKILL",
                    )
                )

        # ---------------------------------------------------------------------
        # PHASE 3: EPISODES 50–74 (Conflicts & Capacity Pressure)
        # ---------------------------------------------------------------------
        for ep_idx in range(50, 75):
            sub_type = ep_idx % 4
            if sub_type == 0:
                # Opposing conflict task (inverted/opposing logic designed to trigger rollback)
                conf_id = f"opposing_conflict_{ep_idx}"
                conf_ds = env_gen.generate_task(
                    task_id=conf_id,
                    family=TaskFamily.PARITY_LOGIC,
                    split=SplitType.META_TEST,
                    rule_params={"mask": 15, "offset": 9, "conflict": True},
                    rule_instance_id=f"CONFLICT_{ep_idx}",
                    seed=seed + ep_idx * 29,
                )
                episodes.append(
                    ContinualTaskEpisode(
                        episode_index=ep_idx,
                        task_id=conf_id,
                        task_family=TaskFamily.PARITY_LOGIC,
                        dataset=conf_ds,
                        intended_regime="CONFLICT_REJECTION_ROLLBACK",
                        expected_optimal_action="REJECT_ROLLBACK",
                    )
                )
            elif sub_type == 1:
                # High complexity quadratic task requiring rank expansion
                exp_id = f"complex_quadratic_{ep_idx}"
                exp_ds = env_gen.generate_task(
                    task_id=exp_id,
                    family=TaskFamily.MODULAR_QUADRATIC,
                    split=SplitType.META_TEST,
                    rule_params={"a": 3, "b": 5, "c": 7},
                    rule_instance_id=f"QD_EXP_{ep_idx}",
                    seed=seed + ep_idx * 31,
                )
                episodes.append(
                    ContinualTaskEpisode(
                        episode_index=ep_idx,
                        task_id=exp_id,
                        task_family=TaskFamily.MODULAR_QUADRATIC,
                        dataset=exp_ds,
                        intended_regime="CAPACITY_EXPANSION",
                        expected_optimal_action="EXPAND_CAPACITY",
                    )
                )
            elif sub_type == 2:
                # Simple permutation task allowing rank shrinkage
                shrk_id = f"simple_transp_{ep_idx}"
                shrk_ds = env_gen.generate_task(
                    task_id=shrk_id,
                    family=TaskFamily.BLOCK_TRANSPOSITION,
                    split=SplitType.META_TEST,
                    rule_params={"offset": 2},
                    rule_instance_id=f"TR_SHR_{ep_idx}",
                    seed=seed + ep_idx * 37,
                )
                episodes.append(
                    ContinualTaskEpisode(
                        episode_index=ep_idx,
                        task_id=shrk_id,
                        task_family=TaskFamily.BLOCK_TRANSPOSITION,
                        dataset=shrk_ds,
                        intended_regime="CAPACITY_SHRINKAGE",
                        expected_optimal_action="CONSOLIDATE_SHRINK",
                    )
                )
            else:
                # Duplicate reuse under capacity pressure
                source_key = phase1_keys[ep_idx % len(phase1_keys)]
                orig_ds = foundational_cache[source_key]
                dup_ds = env_gen.generate_task(
                    task_id=f"{source_key}_dup_ep{ep_idx}",
                    family=orig_ds.family,
                    split=SplitType.META_TEST,
                    rule_params=orig_ds.rule_params,
                    rule_instance_id=orig_ds.rule_instance_id,
                    seed=seed + ep_idx * 41,
                )
                episodes.append(
                    ContinualTaskEpisode(
                        episode_index=ep_idx,
                        task_id=f"{source_key}_dup_ep{ep_idx}",
                        task_family=orig_ds.family,
                        dataset=dup_ds,
                        intended_regime="ZERO_SHOT_REUSE",
                        expected_optimal_action="REUSE_OR_COMPOSE",
                    )
                )

        # ---------------------------------------------------------------------
        # PHASE 4: EPISODES 75–99 (Lifelong Adaptation, Delayed Revisits & Retirement)
        # ---------------------------------------------------------------------
        for ep_idx in range(75, 100):
            pt = ep_idx % 3
            if pt == 0:
                # LONG-DELAYED REVISIT to foundational skills (delay >= 50 episodes!)
                # Select a task from Episodes 0–10
                base_idx = (ep_idx - 75) % 10
                source_key = phase1_specs[base_idx][0]
                orig_ds = foundational_cache[source_key]
                revisit_ds = env_gen.generate_task(
                    task_id=f"{source_key}_revisit_ep{ep_idx}",
                    family=orig_ds.family,
                    split=SplitType.META_TEST,
                    rule_params=orig_ds.rule_params,
                    rule_instance_id=orig_ds.rule_instance_id,
                    seed=seed + ep_idx * 43,
                )
                episodes.append(
                    ContinualTaskEpisode(
                        episode_index=ep_idx,
                        task_id=f"{source_key}_revisit_ep{ep_idx}",
                        task_family=orig_ds.family,
                        dataset=revisit_ds,
                        intended_regime="LONG_DELAYED_REVISIT",
                        expected_optimal_action="REUSE_OR_COMPOSE",
                    )
                )
            elif pt == 1:
                # Novel permutation inversion rule under tight capacity
                perm_id = f"perm_inversion_{ep_idx}"
                perm_ds = env_gen.generate_task(
                    task_id=perm_id,
                    family=TaskFamily.PERMUTATION_INVERSION,
                    split=SplitType.META_TEST,
                    rule_params={"offset": 3},
                    rule_instance_id=f"PERM_INV_{ep_idx}",
                    seed=seed + ep_idx * 47,
                )
                episodes.append(
                    ContinualTaskEpisode(
                        episode_index=ep_idx,
                        task_id=perm_id,
                        task_family=TaskFamily.PERMUTATION_INVERSION,
                        dataset=perm_ds,
                        intended_regime="LATE_STAGE_PLASTICITY",
                        expected_optimal_action="ACQUIRE_NEW_SKILL",
                    )
                )
            else:
                # Cyclic procedure novel rule
                cyc_id = f"cyclic_proc_{ep_idx}"
                cyc_ds = env_gen.generate_task(
                    task_id=cyc_id,
                    family=TaskFamily.CYCLIC_PROCEDURE,
                    split=SplitType.META_TEST,
                    rule_params={"shift": (ep_idx % 5) + 1},
                    rule_instance_id=f"CYC_{ep_idx}",
                    seed=seed + ep_idx * 53,
                )
                episodes.append(
                    ContinualTaskEpisode(
                        episode_index=ep_idx,
                        task_id=cyc_id,
                        task_family=TaskFamily.CYCLIC_PROCEDURE,
                        dataset=cyc_ds,
                        intended_regime="LATE_STAGE_PLASTICITY",
                        expected_optimal_action="ACQUIRE_NEW_SKILL",
                    )
                )

        return episodes[:num_episodes]
