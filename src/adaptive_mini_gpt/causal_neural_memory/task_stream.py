"""Milestone M73 Phase 3 & 5: Sequential Lifelong Task Stream Generator.

Provides two 20-episode streams:
1. Canonical M72 Stream:
   Reproduces the exact 20-episode sequence from M72 across Worlds Alpha, Beta, Gamma, Delta
   to audit and compare retention, routing, and cost accounting.
2. Independent M73 Open-World Stream:
   Evaluates new independent task domains:
   - Hierarchical taxonomic tree subsumption (World Epsilon)
   - Modular arithmetic congruence modulo 5 (World Zeta)
   - Partial observation transitive graphs with contradictory evidence
   - Intervening unrelated episodes testing multi-encounter retention across 8+ episodes.
"""

from __future__ import annotations

import logging
from typing import Any

from adaptive_mini_gpt.guided_experiment.environment import (
    AutonomousEnvironment,
    RuleHypothesis,
)
from adaptive_mini_gpt.robust_lifelong_memory.lifelong_task_stream import (
    build_m72_lifelong_stream_sequence,
    create_anti_transitive_dag_env,
    create_contradictory_lattice_env,
    create_inverted_rail_parity_env,
    create_modular_arithmetic_relational_env,
)
from adaptive_mini_gpt.three_valued_reasoning.semantics import (
    GroundedThreeValuedEnvironment,
)
from adaptive_mini_gpt.utility_calibrated_selection.task_environments import (
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

__all__ = [
    "build_m72_lifelong_stream_sequence",
    "build_m73_open_world_stream_sequence",
    "create_hierarchical_subsumption_env",
    "create_modular_congruence_mod5_env",
]


def create_hierarchical_subsumption_env(
    env_id: str = "m73_subsumption",
) -> GroundedThreeValuedEnvironment:
    """Taxonomic tree ontology where ancestor categories subsume descendant subcategories."""
    entities = [
        "CategoryRoot",
        "SubTaxonA",
        "SubTaxonB",
        "LeafSubclassA1",
        "LeafSubclassA2",
        "LeafSubclassB1",
        "LeafSubclassB2",
        "SpecializedInstanceA1",
        "SpecializedInstanceB1",
    ]
    edges = [
        ("CategoryRoot", "SubTaxonA"),
        ("CategoryRoot", "SubTaxonB"),
        ("SubTaxonA", "LeafSubclassA1"),
        ("SubTaxonA", "LeafSubclassA2"),
        ("SubTaxonB", "LeafSubclassB1"),
        ("SubTaxonB", "LeafSubclassB2"),
        ("LeafSubclassA1", "SpecializedInstanceA1"),
        ("LeafSubclassB1", "SpecializedInstanceB1"),
    ]
    base_env = AutonomousEnvironment(
        env_id=env_id,
        domain_name="TaxonomicOntologyTree",
        relation_name="subsumes_category",
        entities=entities,
        base_edges=edges,
        true_hypothesis=RuleHypothesis(
            hypothesis_id="HIERARCHICAL_SUBSUMPTION",
            name="Hierarchical Taxonomic Subsumption",
            is_symmetric=False,
            is_transitive=True,
            description="Parent node strictly subsumes child and descendant subcategories",
        ),
        held_out_test_queries=[],
        minimal_pairs=[],
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


def create_modular_congruence_mod5_env(env_id: str = "m73_mod5") -> GroundedThreeValuedEnvironment:
    """Modular arithmetic congruence modulo 5."""
    entities = [f"Mod5NumeralVal{i}" for i in range(15)]
    edges = [
        (entities[0], entities[5]),
        (entities[5], entities[10]),
        (entities[1], entities[6]),
        (entities[6], entities[11]),
        (entities[2], entities[7]),
        (entities[7], entities[12]),
        (entities[3], entities[8]),
        (entities[8], entities[13]),
        (entities[4], entities[9]),
        (entities[9], entities[14]),
    ]
    base_env = AutonomousEnvironment(
        env_id=env_id,
        domain_name="ModularRelationalArithmeticMod5",
        relation_name="is_congruent_mod_5_with",
        entities=entities,
        base_edges=edges,
        true_hypothesis=RuleHypothesis(
            hypothesis_id="MODULAR_CONGRUENCE_MOD_5",
            name="Modular Congruence Mod 5",
            is_symmetric=True,
            is_transitive=True,
            description="Numeral u is congruent mod 5 with Numeral v",
        ),
        held_out_test_queries=[],
        minimal_pairs=[],
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


def build_m73_open_world_stream_sequence() -> list[dict[str, Any]]:
    """Build the new independent 20-episode task stream for M73."""
    # Build open-world sequence with 7 returns, 3 adversarial tasks, and non-graph arithmetic
    seq = []
    # Ep 0-4: World Epsilon
    seq.append(
        {
            "episode_index": 0,
            "task_name": "ep0_hierarchical_subsumption_eps",
            "rule_family": "HIERARCHICAL_SUBSUMPTION",
            "world_id": "world_epsilon",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_hierarchical_subsumption_env("m73_subsumption_tr"),
            "qual_env": create_hierarchical_subsumption_env("m73_subsumption_val"),
            "eval_env": create_hierarchical_subsumption_env("m73_subsumption_ev"),
        }
    )
    seq.append(
        {
            "episode_index": 1,
            "task_name": "ep1_circuit_cascade_eps",
            "rule_family": "FAULT_CASCADE",
            "world_id": "world_epsilon",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_circuit_stage1_train_env(),
            "qual_env": create_circuit_stage2_qual_env(),
            "eval_env": create_circuit_stage3_eval_env(),
        }
    )
    seq.append(
        {
            "episode_index": 2,
            "task_name": "ep2_familiar_dag_eps",
            "rule_family": "TRANSITIVE_DAG",
            "world_id": "world_epsilon",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_known_incomplete_dag_env(),
            "qual_env": create_known_reusable_dag_env(),
            "eval_env": create_known_reusable_dag_env(),
        }
    )
    seq.append(
        {
            "episode_index": 3,
            "task_name": "ep3_mod5_arithmetic_eps",
            "rule_family": "MODULAR_CONGRUENCE_MOD_5",
            "world_id": "world_epsilon",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_modular_congruence_mod5_env("m73_mod5_tr"),
            "qual_env": create_modular_congruence_mod5_env("m73_mod5_val"),
            "eval_env": create_modular_congruence_mod5_env("m73_mod5_ev"),
        }
    )
    seq.append(
        {
            "episode_index": 4,
            "task_name": "ep4_return_hierarchical_eps",
            "rule_family": "HIERARCHICAL_SUBSUMPTION",
            "world_id": "world_epsilon",
            "is_repeated_encounter": True,
            "is_adversarial": False,
            "train_env": create_hierarchical_subsumption_env("m73_subsumption_return_tr"),
            "qual_env": create_hierarchical_subsumption_env("m73_subsumption_return_val"),
            "eval_env": create_hierarchical_subsumption_env("m73_subsumption_return_ev"),
        }
    )

    # Ep 5-9: World Zeta
    seq.append(
        {
            "episode_index": 5,
            "task_name": "ep5_novel_chromatin_zeta",
            "rule_family": "CHROMATIN_LATTICE",
            "world_id": "world_zeta",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_chromatin_stage1_train_env(),
            "qual_env": create_chromatin_stage2_qual_env(),
            "eval_env": create_chromatin_stage3_eval_env(),
        }
    )
    seq.append(
        {
            "episode_index": 6,
            "task_name": "ep6_adv_anti_dag_zeta",
            "rule_family": "ANTI_TRANSITIVE_DAG",
            "world_id": "world_zeta",
            "is_repeated_encounter": False,
            "is_adversarial": True,
            "train_env": create_anti_transitive_dag_env("m73_anti_dag_tr"),
            "qual_env": create_anti_transitive_dag_env("m73_anti_dag_val"),
            "eval_env": create_anti_transitive_dag_env("m73_anti_dag_ev"),
        }
    )
    seq.append(
        {
            "episode_index": 7,
            "task_name": "ep7_return_mod5_zeta",
            "rule_family": "MODULAR_CONGRUENCE_MOD_5",
            "world_id": "world_zeta",
            "is_repeated_encounter": True,
            "is_adversarial": False,
            "train_env": create_modular_congruence_mod5_env("m73_mod5_return_tr"),
            "qual_env": create_modular_congruence_mod5_env("m73_mod5_return_val"),
            "eval_env": create_modular_congruence_mod5_env("m73_mod5_return_ev"),
        }
    )
    seq.append(
        {
            "episode_index": 8,
            "task_name": "ep8_novel_rail_zeta",
            "rule_family": "PARITY_CYCLE",
            "world_id": "world_zeta",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_rail_stage1_train_env(),
            "qual_env": create_rail_stage2_qual_env(),
            "eval_env": create_rail_stage3_eval_env(),
        }
    )
    seq.append(
        {
            "episode_index": 9,
            "task_name": "ep9_return_circuit_zeta",
            "rule_family": "FAULT_CASCADE",
            "world_id": "world_zeta",
            "is_repeated_encounter": True,
            "is_adversarial": False,
            "train_env": create_circuit_stage1_train_env(),
            "qual_env": create_circuit_stage2_qual_env(),
            "eval_env": create_circuit_stage3_eval_env(),
        }
    )

    # Ep 10-14: World Eta
    seq.append(
        {
            "episode_index": 10,
            "task_name": "ep10_adv_inverted_rail_eta",
            "rule_family": "INVERTED_PARITY_CYCLE",
            "world_id": "world_eta",
            "is_repeated_encounter": False,
            "is_adversarial": True,
            "train_env": create_inverted_rail_parity_env("m73_inv_rail_tr"),
            "qual_env": create_inverted_rail_parity_env("m73_inv_rail_val"),
            "eval_env": create_inverted_rail_parity_env("m73_inv_rail_ev"),
        }
    )
    seq.append(
        {
            "episode_index": 11,
            "task_name": "ep11_return_chromatin_eta",
            "rule_family": "CHROMATIN_LATTICE",
            "world_id": "world_eta",
            "is_repeated_encounter": True,
            "is_adversarial": False,
            "train_env": create_chromatin_stage1_train_env(),
            "qual_env": create_chromatin_stage2_qual_env(),
            "eval_env": create_chromatin_stage3_eval_env(),
        }
    )
    seq.append(
        {
            "episode_index": 12,
            "task_name": "ep12_2nd_return_hierarchical_eta",
            "rule_family": "HIERARCHICAL_SUBSUMPTION",
            "world_id": "world_eta",
            "is_repeated_encounter": True,
            "is_adversarial": False,
            "train_env": create_hierarchical_subsumption_env("m73_subsumption_2nd_tr"),
            "qual_env": create_hierarchical_subsumption_env("m73_subsumption_2nd_val"),
            "eval_env": create_hierarchical_subsumption_env("m73_subsumption_2nd_ev"),
        }
    )
    seq.append(
        {
            "episode_index": 13,
            "task_name": "ep13_return_rail_eta",
            "rule_family": "PARITY_CYCLE",
            "world_id": "world_eta",
            "is_repeated_encounter": True,
            "is_adversarial": False,
            "train_env": create_rail_stage1_train_env(),
            "qual_env": create_rail_stage2_qual_env(),
            "eval_env": create_rail_stage3_eval_env(),
        }
    )
    seq.append(
        {
            "episode_index": 14,
            "task_name": "ep14_adv_contradictory_lattice_eta",
            "rule_family": "CHROMATIN_LATTICE",
            "world_id": "world_eta",
            "is_repeated_encounter": False,
            "is_adversarial": True,
            "train_env": create_contradictory_lattice_env("m73_contra_tr"),
            "qual_env": create_contradictory_lattice_env("m73_contra_val"),
            "eval_env": create_contradictory_lattice_env("m73_contra_ev"),
        }
    )

    # Ep 15-19: World Theta
    seq.append(
        {
            "episode_index": 15,
            "task_name": "ep15_2nd_return_mod5_theta",
            "rule_family": "MODULAR_CONGRUENCE_MOD_5",
            "world_id": "world_theta",
            "is_repeated_encounter": True,
            "is_adversarial": False,
            "train_env": create_modular_congruence_mod5_env("m73_mod5_3rd_tr"),
            "qual_env": create_modular_congruence_mod5_env("m73_mod5_3rd_val"),
            "eval_env": create_modular_congruence_mod5_env("m73_mod5_3rd_ev"),
        }
    )
    seq.append(
        {
            "episode_index": 16,
            "task_name": "ep16_familiar_dag_theta",
            "rule_family": "TRANSITIVE_DAG",
            "world_id": "world_theta",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_known_incomplete_dag_env(),
            "qual_env": create_known_reusable_dag_env(),
            "eval_env": create_known_reusable_dag_env(),
        }
    )
    seq.append(
        {
            "episode_index": 17,
            "task_name": "ep17_2nd_return_circuit_theta",
            "rule_family": "FAULT_CASCADE",
            "world_id": "world_theta",
            "is_repeated_encounter": True,
            "is_adversarial": False,
            "train_env": create_circuit_stage1_train_env(),
            "qual_env": create_circuit_stage2_qual_env(),
            "eval_env": create_circuit_stage3_eval_env(),
        }
    )
    seq.append(
        {
            "episode_index": 18,
            "task_name": "ep18_nongraph_mod_arith_theta",
            "rule_family": "MODULAR_CONGRUENCE_MOD_3",
            "world_id": "world_theta",
            "is_repeated_encounter": False,
            "is_adversarial": False,
            "train_env": create_modular_arithmetic_relational_env(
                "m73_mod3_theta_tr", seed_offset=50
            ),
            "qual_env": create_modular_arithmetic_relational_env(
                "m73_mod3_theta_val", seed_offset=50
            ),
            "eval_env": create_modular_arithmetic_relational_env(
                "m73_mod3_theta_ev", seed_offset=50
            ),
        }
    )
    seq.append(
        {
            "episode_index": 19,
            "task_name": "ep19_repeat_mod_arith_theta",
            "rule_family": "MODULAR_CONGRUENCE_MOD_3",
            "world_id": "world_theta",
            "is_repeated_encounter": True,
            "is_adversarial": False,
            "train_env": create_modular_arithmetic_relational_env(
                "m73_mod3_repeat_tr", seed_offset=100
            ),
            "qual_env": create_modular_arithmetic_relational_env(
                "m73_mod3_repeat_val", seed_offset=100
            ),
            "eval_env": create_modular_arithmetic_relational_env(
                "m73_mod3_repeat_ev", seed_offset=100
            ),
        }
    )

    return seq
