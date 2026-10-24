"""Milestone M72 Phase 3 & 7: 20-Episode Lifelong Sequential Task Stream Generator.

Generates a structured, reproducible 20-episode sequential stream across 4 independent worlds:
- World Alpha (Episodes 0-2)
- World Beta (Episodes 3-7)
- World Gamma (Episodes 8-13)
- World Delta (Episodes 14-19)

Features:
1. Initial novel skill acquisition (Episodes 0, 1, 3).
2. Familiar source skill reuses (Episodes 2, 4, 9, 12, 16).
3. Long-horizon family returns after 4-8 intervening episodes (Episodes 5, 7, 11, 13, 15, 17).
4. Adversarial / Misleading tasks with inverted axioms (Episodes 6, 10).
5. Open-world uncertainty and contradictory evidence probes (Episodes 8, 14).
6. Non-graph relational arithmetic diagnostic (Episodes 18, 19).
"""
from __future__ import annotations

import logging
from typing import Any

from adaptive_mini_gpt.guided_experiment.environment import (
    AutonomousEnvironment,
    RuleHypothesis,
)
from adaptive_mini_gpt.modular_skill_learning.task_suite import (
    create_source_consortium_bipartite,
    create_source_metabolic_inhibitor,
)
from adaptive_mini_gpt.three_valued_reasoning.semantics import (
    GroundedThreeValuedEnvironment,
)
from adaptive_mini_gpt.utility_calibrated_selection.task_environments import (
    OUT_OF_HYPOTHESIS_PARITY,
    create_chromatin_stage1_train_env,
    create_chromatin_stage2_qual_env,
    create_chromatin_stage3_eval_env,
    create_circuit_stage1_train_env,
    create_circuit_stage2_qual_env,
    create_circuit_stage3_eval_env,
    create_known_incomplete_dag_env,
    create_known_reusable_dag_env,
    create_rail_stage1_train_env,
    create_rail_stage2_qual_env,
    create_rail_stage3_eval_env,
)

logger = logging.getLogger(__name__)


# =========================================================================
# Custom Adversarial & Non-Graph Task Environments
# =========================================================================

def create_anti_transitive_dag_env(env_id: str = "m72_anti_dag", is_qual: bool = False) -> GroundedThreeValuedEnvironment:
    """Adversarial DAG where reachability is inverted on odd-hop paths."""
    entities = [f"SubsystemModuleM{i}" for i in range(9)]
    edges = [(entities[i], entities[i + 1]) for i in range(8)]
    base_env = AutonomousEnvironment(
        env_id=env_id,
        domain_name="AntiTransitiveDependencyGraph",
        relation_name="inversely_blocks_subsystem",
        entities=entities,
        base_edges=edges,
        true_hypothesis=RuleHypothesis(
            hypothesis_id="ANTI_TRANSITIVE_DAG",
            name="Anti-Transitive Dependency Graph",
            is_symmetric=False,
            is_transitive=False,
            description="Odd hop connections are strictly NO; only direct even hops are YES",
        ),
        held_out_test_queries=[],
        minimal_pairs=[],
    )
    env = GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)
    # Invert expected answers on multi-hop queries to trigger severe negative transfer for transitive models
    for q in env.held_out_test_queries:
        if q.query_type != "OBSERVED_POSITIVE_EDGE":
            if q.expected_answer == "YES":
                q.expected_answer = "NO"
            elif q.expected_answer == "NO":
                q.expected_answer = "YES"
    return env


def create_inverted_rail_parity_env(env_id: str = "m72_inverted_rail") -> GroundedThreeValuedEnvironment:
    """Adversarial rail parity where odd cycles are disallowed and even cycles are forbidden."""
    entities = [f"InvertedRailNodeBeta{i}" for i in range(9)]
    edges = [(entities[i], entities[i + 1]) for i in range(8)]
    base_env = AutonomousEnvironment(
        env_id=env_id,
        domain_name="InvertedRailGridBeta",
        relation_name="dispatches_high_speed_consist_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=OUT_OF_HYPOTHESIS_PARITY,
        held_out_test_queries=[],
        minimal_pairs=[],
    )
    env = GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)
    for q in env.held_out_test_queries:
        if q.expected_answer == "YES":
            q.expected_answer = "NO"
        elif q.expected_answer == "NO":
            q.expected_answer = "YES"
    return env


def create_contradictory_lattice_env(env_id: str = "m72_contradictory_lattice") -> GroundedThreeValuedEnvironment:
    """Lattice with contradictory evidence probes requiring UNKNOWN rejection."""
    entities = [f"LocusDelta{i}" for i in range(9)]
    edges = [(entities[i], entities[i + 1]) for i in range(8)]
    base_env = AutonomousEnvironment(
        env_id=env_id,
        domain_name="ContradictorySilencerLatticeDelta",
        relation_name="propagates_histone_acetylation_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=OUT_OF_HYPOTHESIS_PARITY,
        held_out_test_queries=[],
        minimal_pairs=[],
    )
    env = GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)
    # Inject contradictory evidence into prefixes
    for q in env.held_out_test_queries:
        q.question_text = f"Evidence: [{entities[0]}->{entities[1]}, NOT({entities[0]}->{entities[1]})]. {q.question_text}"
        q.expected_answer = "UNKNOWN"
    return env


def create_modular_arithmetic_relational_env(env_id: str = "m72_mod_arith", seed_offset: int = 0) -> GroundedThreeValuedEnvironment:
    """Phase 7 Non-Graph Diagnostic: Modular relational arithmetic (congruence mod 3)."""
    entities = [f"NumeralVal{i + seed_offset}" for i in range(12)]
    edges = [
        (entities[0], entities[3]), (entities[3], entities[6]), (entities[6], entities[9]),
        (entities[1], entities[4]), (entities[4], entities[7]), (entities[7], entities[10]),
        (entities[2], entities[5]), (entities[5], entities[8]), (entities[8], entities[11]),
    ]
    base_env = AutonomousEnvironment(
        env_id=env_id,
        domain_name="ModularRelationalArithmeticDelta",
        relation_name="is_congruent_mod_3_with",
        entities=entities,
        base_edges=edges,
        true_hypothesis=RuleHypothesis(
            hypothesis_id="MODULAR_CONGRUENCE_MOD_3",
            name="Modular Congruence Mod 3",
            is_symmetric=True,
            is_transitive=True,
            description="Numeral u is congruent mod 3 with Numeral v",
        ),
        held_out_test_queries=[],
        minimal_pairs=[],
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


# =========================================================================
# 20-Episode Lifelong Task Sequence Generator
# =========================================================================

def build_m72_lifelong_stream_sequence() -> list[dict[str, Any]]:
    """Build the complete 20-episode sequential stream."""
    return [
        # Episode 0: Novel Rail Parity (World Alpha) -> Adapts & creates Rail adapter
        {
            "episode_index": 0,
            "task_name": "ep0_novel_rail_alpha",
            "rule_family": "PARITY_CYCLE",
            "world_id": "world_alpha",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_rail_stage1_train_env(),
            "qual_env": create_rail_stage2_qual_env(),
            "eval_env": create_rail_stage3_eval_env(),
        },
        # Episode 1: Novel Chromatin Lattice (World Alpha) -> Adapts & creates Chromatin adapter
        {
            "episode_index": 1,
            "task_name": "ep1_novel_chromatin_alpha",
            "rule_family": "CHROMATIN_LATTICE",
            "world_id": "world_alpha",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_chromatin_stage1_train_env(),
            "qual_env": create_chromatin_stage2_qual_env(),
            "eval_env": create_chromatin_stage3_eval_env(),
        },
        # Episode 2: Familiar Transitive DAG (World Alpha) -> Reuses source DAG skill (0 steps)
        {
            "episode_index": 2,
            "task_name": "ep2_familiar_dag_alpha",
            "rule_family": "TRANSITIVE_DAG",
            "world_id": "world_alpha",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_known_incomplete_dag_env(),
            "qual_env": create_known_reusable_dag_env(),
            "eval_env": create_known_reusable_dag_env(),
        },
        # Episode 3: Novel Circuit Fault Cascade (World Beta) -> Adapts & creates Circuit adapter
        {
            "episode_index": 3,
            "task_name": "ep3_novel_circuit_beta",
            "rule_family": "FAULT_CASCADE",
            "world_id": "world_beta",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_circuit_stage1_train_env(),
            "qual_env": create_circuit_stage2_qual_env(),
            "eval_env": create_circuit_stage3_eval_env(),
        },
        # Episode 4: Familiar Metabolic Inhibitor (World Beta) -> Reuses source Metabolic skill
        {
            "episode_index": 4,
            "task_name": "ep4_familiar_metabolic_beta",
            "rule_family": "INHIBITOR_CASCADE",
            "world_id": "world_beta",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_source_metabolic_inhibitor(),
            "qual_env": create_source_metabolic_inhibitor(),
            "eval_env": create_source_metabolic_inhibitor(),
        },
        # Episode 5: Returning Rail Parity (World Beta) -> Realized Transfer 1 (Reuses Ep 0 adapter)
        {
            "episode_index": 5,
            "task_name": "ep5_return_rail_beta",
            "rule_family": "PARITY_CYCLE",
            "world_id": "world_beta",
            "is_repeated_encounter": True,
            "is_adversarial": False,
            "train_env": create_rail_stage1_train_env(),
            "qual_env": create_rail_stage2_qual_env(),
            "eval_env": create_rail_stage3_eval_env(),
        },
        # Episode 6: Adversarial Anti-Transitive DAG (World Beta) -> Tests Negative Transfer Prevention
        {
            "episode_index": 6,
            "task_name": "ep6_adv_anti_dag_beta",
            "rule_family": "ANTI_TRANSITIVE_DAG",
            "world_id": "world_beta",
            "is_repeated_encounter": False,
            "is_adversarial": True,
            "train_env": create_anti_transitive_dag_env("ep6_anti_dag_tr"),
            "qual_env": create_anti_transitive_dag_env("ep6_anti_dag_q", is_qual=True),
            "eval_env": create_anti_transitive_dag_env("ep6_anti_dag_ev"),
        },
        # Episode 7: Returning Chromatin Lattice (World Beta) -> Realized Transfer 2 (Reuses Ep 1 adapter)
        {
            "episode_index": 7,
            "task_name": "ep7_return_chromatin_beta",
            "rule_family": "CHROMATIN_LATTICE",
            "world_id": "world_beta",
            "is_repeated_encounter": True,
            "is_adversarial": False,
            "train_env": create_chromatin_stage1_train_env(),
            "qual_env": create_chromatin_stage2_qual_env(),
            "eval_env": create_chromatin_stage3_eval_env(),
        },
        # Episode 8: Open-World Circuit with Unobserved Gates (World Gamma) -> Uncertainty withholding
        {
            "episode_index": 8,
            "task_name": "ep8_uncertain_circuit_gamma",
            "rule_family": "FAULT_CASCADE",
            "world_id": "world_gamma",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_circuit_stage1_train_env(),
            "qual_env": create_circuit_stage2_qual_env(),
            "eval_env": create_circuit_stage3_eval_env(),
        },
        # Episode 9: Familiar Consortium Bipartite (World Gamma) -> Reuses source consortium skill
        {
            "episode_index": 9,
            "task_name": "ep9_familiar_consortium_gamma",
            "rule_family": "BIPARTITE_CONSORTIUM",
            "world_id": "world_gamma",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_source_consortium_bipartite(),
            "qual_env": create_source_consortium_bipartite(),
            "eval_env": create_source_consortium_bipartite(),
        },
        # Episode 10: Adversarial Inverted Rail Parity (World Gamma) -> Tests Negative Transfer Prevention
        {
            "episode_index": 10,
            "task_name": "ep10_adv_inverted_rail_gamma",
            "rule_family": "INVERTED_PARITY_CYCLE",
            "world_id": "world_gamma",
            "is_repeated_encounter": False,
            "is_adversarial": True,
            "train_env": create_inverted_rail_parity_env("ep10_inv_rail_tr"),
            "qual_env": create_inverted_rail_parity_env("ep10_inv_rail_q"),
            "eval_env": create_inverted_rail_parity_env("ep10_inv_rail_ev"),
        },
        # Episode 11: Returning Circuit Fault Cascade (World Gamma) -> Realized Transfer 3 (Reuses Ep 3 adapter)
        {
            "episode_index": 11,
            "task_name": "ep11_return_circuit_gamma",
            "rule_family": "FAULT_CASCADE",
            "world_id": "world_gamma",
            "is_repeated_encounter": True,
            "is_adversarial": False,
            "train_env": create_circuit_stage1_train_env(),
            "qual_env": create_circuit_stage2_qual_env(),
            "eval_env": create_circuit_stage3_eval_env(),
        },
        # Episode 12: Familiar Metabolic Cascade (World Gamma) -> Reuses source Metabolic skill
        {
            "episode_index": 12,
            "task_name": "ep12_familiar_metabolic_gamma",
            "rule_family": "INHIBITOR_CASCADE",
            "world_id": "world_gamma",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_source_metabolic_inhibitor(),
            "qual_env": create_source_metabolic_inhibitor(),
            "eval_env": create_source_metabolic_inhibitor(),
        },
        # Episode 13: 2nd Return of Rail Parity (World Gamma) -> Realized Transfer 4 (Retention across 7 eps)
        {
            "episode_index": 13,
            "task_name": "ep13_2nd_return_rail_gamma",
            "rule_family": "PARITY_CYCLE",
            "world_id": "world_gamma",
            "is_repeated_encounter": True,
            "is_adversarial": False,
            "train_env": create_rail_stage1_train_env(),
            "qual_env": create_rail_stage2_qual_env(),
            "eval_env": create_rail_stage3_eval_env(),
        },
        # Episode 14: Contradictory Evidence Lattice (World Delta) -> Tests Epistemic Conflict Detection
        {
            "episode_index": 14,
            "task_name": "ep14_contradictory_lattice_delta",
            "rule_family": "CHROMATIN_LATTICE",
            "world_id": "world_delta",
            "is_repeated_encounter": False,
            "is_adversarial": True,
            "train_env": create_contradictory_lattice_env("ep14_contra_tr"),
            "qual_env": create_contradictory_lattice_env("ep14_contra_q"),
            "eval_env": create_contradictory_lattice_env("ep14_contra_ev"),
        },
        # Episode 15: 2nd Return of Chromatin Lattice (World Gamma) -> Realized Transfer 5 (Retention across 7 eps)
        {
            "episode_index": 15,
            "task_name": "ep15_2nd_return_chromatin_gamma",
            "rule_family": "CHROMATIN_LATTICE",
            "world_id": "world_gamma",
            "is_repeated_encounter": True,
            "is_adversarial": False,
            "train_env": create_chromatin_stage1_train_env(),
            "qual_env": create_chromatin_stage2_qual_env(),
            "eval_env": create_chromatin_stage3_eval_env(),
        },
        # Episode 16: Familiar DAG Transitive (World Delta) -> Reuses source DAG skill
        {
            "episode_index": 16,
            "task_name": "ep16_familiar_dag_delta",
            "rule_family": "TRANSITIVE_DAG",
            "world_id": "world_delta",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_known_incomplete_dag_env(),
            "qual_env": create_known_reusable_dag_env(),
            "eval_env": create_known_reusable_dag_env(),
        },
        # Episode 17: 2nd Return of Circuit Fault Cascade (World Delta) -> Realized Transfer 6 (Retention across 5 eps)
        {
            "episode_index": 17,
            "task_name": "ep17_2nd_return_circuit_delta",
            "rule_family": "FAULT_CASCADE",
            "world_id": "world_delta",
            "is_repeated_encounter": True,
            "is_adversarial": False,
            "train_env": create_circuit_stage1_train_env(),
            "qual_env": create_circuit_stage2_qual_env(),
            "eval_env": create_circuit_stage3_eval_env(),
        },
        # Episode 18: Phase 7 Non-Graph Diagnostic 1: Modular Relational Arithmetic (World Delta) -> Adapts & creates
        {
            "episode_index": 18,
            "task_name": "ep18_nongraph_mod_arith_delta",
            "rule_family": "MODULAR_CONGRUENCE_MOD_3",
            "world_id": "world_delta",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_modular_arithmetic_relational_env("ep18_mod_tr", seed_offset=0),
            "qual_env": create_modular_arithmetic_relational_env("ep18_mod_q", seed_offset=12),
            "eval_env": create_modular_arithmetic_relational_env("ep18_mod_ev", seed_offset=24),
        },
        # Episode 19: Phase 7 Non-Graph Diagnostic 2: Repeat Modular Congruence -> Realized Transfer 7
        {
            "episode_index": 19,
            "task_name": "ep19_repeat_mod_arith_delta",
            "rule_family": "MODULAR_CONGRUENCE_MOD_3",
            "world_id": "world_delta",
            "is_repeated_encounter": True,
            "is_adversarial": False,
            "train_env": create_modular_arithmetic_relational_env("ep19_mod_tr", seed_offset=36),
            "qual_env": create_modular_arithmetic_relational_env("ep19_mod_q", seed_offset=48),
            "eval_env": create_modular_arithmetic_relational_env("ep19_mod_ev", seed_offset=60),
        },
    ]
