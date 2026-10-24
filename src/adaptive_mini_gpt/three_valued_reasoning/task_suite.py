"""Phase 7: Brand-New Independent Sealed Evaluation Environments for M65."""
from __future__ import annotations

import logging
from typing import Any

from adaptive_mini_gpt.guided_experiment.environment import (
    EXPANDED_HYPOTHESES,
    OUT_OF_HYPOTHESIS_PARITY,
    AutonomousEnvironment,
    EnvironmentQuery,
    MinimalPairControl,
    RuleHypothesis,
)
from adaptive_mini_gpt.skill_reuse.environment import (
    OUT_OF_HYPOTHESIS_BIPARTITE,
    OUT_OF_HYPOTHESIS_DENSE_CLIQUE,
)
from adaptive_mini_gpt.three_valued_reasoning.semantics import (
    GroundedThreeValuedEnvironment,
    ThreeValuedDatasetValidator,
)

logger = logging.getLogger(__name__)


def create_new_sealed_quantum_teleportation_dag(seed: int = 501, use_evidence_prefix: bool = True) -> GroundedThreeValuedEnvironment:
    """New Task 1: 10-qubit quantum teleportation circuit DAG with tier boundary constraints."""
    entities = [f"QubitStateQ{i}" for i in range(10)]
    edges = [(entities[i], entities[i + 1]) for i in range(9)]
    # Add cross-tier teleportation skip
    edges.append((entities[0], entities[2]))
    edges.append((entities[4], entities[6]))
    levels = {e: i for i, e in enumerate(entities)}

    base_env = AutonomousEnvironment(
        env_id="m65_sealed_quantum_teleportation_dag",
        domain_name="QuantumInformationTeleportation",
        relation_name="teleports_quantum_state_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        held_out_test_queries=[],
        minimal_pairs=[],
        entity_levels=levels,
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=use_evidence_prefix)


def create_new_sealed_metabolic_inhibitor_tree(seed: int = 502, use_evidence_prefix: bool = True) -> GroundedThreeValuedEnvironment:
    """New Task 2: 10-enzyme branching metabolic flux cascade with active inhibitor nodes."""
    entities = [
        "PrecursorEnzymeE0",
        "KinasePathwayK1", "KinasePathwayK2", "KinasePathwayK3",
        "MetaboliteM11", "MetaboliteM12", "MetaboliteM21", "MetaboliteM22", "MetaboliteM31", "MetaboliteM32",
    ]
    edges = [
        ("PrecursorEnzymeE0", "KinasePathwayK1"),
        ("PrecursorEnzymeE0", "KinasePathwayK2"),
        ("PrecursorEnzymeE0", "KinasePathwayK3"),
        ("KinasePathwayK1", "MetaboliteM11"), ("KinasePathwayK1", "MetaboliteM12"),
        ("KinasePathwayK2", "MetaboliteM21"), ("KinasePathwayK2", "MetaboliteM22"),
        ("KinasePathwayK3", "MetaboliteM31"), ("KinasePathwayK3", "MetaboliteM32"),
    ]
    negated = {("KinasePathwayK1", "MetaboliteM12"), ("KinasePathwayK3", "MetaboliteM31")}

    base_env = AutonomousEnvironment(
        env_id="m65_sealed_metabolic_inhibitor_tree",
        domain_name="CellularMetabolicFluxCascade",
        relation_name="catalyzes_metabolic_cascade_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=EXPANDED_HYPOTHESES["EXPLICIT_REFUTATION"],
        held_out_test_queries=[],
        minimal_pairs=[],
        negated_edges=negated,
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=use_evidence_prefix)


def create_new_sealed_consortium_smart_contract(seed: int = 503, use_evidence_prefix: bool = True) -> GroundedThreeValuedEnvironment:
    """New Task 3: 8-node consortium blockchain partition (non-transitive bipartite protocol)."""
    entities = [
        "ValidatorAuthV0", "EscrowContractE1", "SettlementRelayS2", "ArbiterChamberA3",
        "ClearingHouseC4", "LiquidityPoolL5", "CollateralVaultV6", "AuditorRegistryR7",
    ]
    edges = [
        ("ValidatorAuthV0", "EscrowContractE1"),
        ("EscrowContractE1", "SettlementRelayS2"),
        ("SettlementRelayS2", "ArbiterChamberA3"),
        ("ArbiterChamberA3", "ClearingHouseC4"),
        ("ClearingHouseC4", "LiquidityPoolL5"),
        ("LiquidityPoolL5", "CollateralVaultV6"),
        ("CollateralVaultV6", "AuditorRegistryR7"),
    ]

    base_env = AutonomousEnvironment(
        env_id="m65_sealed_consortium_smart_contract",
        domain_name="ConsortiumBlockchainGovernance",
        relation_name="executes_cross_shard_transaction_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=OUT_OF_HYPOTHESIS_BIPARTITE,
        held_out_test_queries=[],
        minimal_pairs=[],
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=use_evidence_prefix)


def create_new_sealed_epigenetic_chromatin_lattice(seed: int = 504, use_evidence_prefix: bool = True) -> GroundedThreeValuedEnvironment:
    """New Task 4: 8-locus chromatin lattice with open-world incomplete reachability."""
    entities = [f"ChromatinLocusL{i}" for i in range(8)]
    edges = [
        (entities[0], entities[1]), (entities[0], entities[2]),
        (entities[1], entities[3]), (entities[2], entities[4]),
        (entities[3], entities[5]), (entities[4], entities[6]),
        (entities[5], entities[7]), (entities[6], entities[7]),
    ]

    base_env = AutonomousEnvironment(
        env_id="m65_sealed_epigenetic_chromatin_lattice",
        domain_name="EpigeneticChromatinRemodeling",
        relation_name="propagates_histone_acetylation_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=EXPANDED_HYPOTHESES["OPEN_WORLD_INCOMPLETE"],
        held_out_test_queries=[],
        minimal_pairs=[],
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=use_evidence_prefix)


def create_new_sealed_autonomous_rail_grid(seed: int = 505, use_evidence_prefix: bool = True) -> GroundedThreeValuedEnvironment:
    """New Task 5: 8-station high-speed rail grid with alternating track power parity constraints."""
    entities = [f"TerminalStationT{i}" for i in range(8)]
    edges = [(entities[i], entities[(i + 1) % 8]) for i in range(8)]

    base_env = AutonomousEnvironment(
        env_id="m65_sealed_autonomous_rail_grid",
        domain_name="AutonomousHighSpeedRailNetwork",
        relation_name="dispatches_high_speed_consist_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=OUT_OF_HYPOTHESIS_PARITY,
        held_out_test_queries=[],
        minimal_pairs=[],
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=use_evidence_prefix)


def build_m65_new_sealed_suite(use_evidence_prefix: bool = True) -> list[GroundedThreeValuedEnvironment]:
    """Construct complete brand-new sealed evaluation suite for M65 Phase 7."""
    suite = [
        create_new_sealed_quantum_teleportation_dag(use_evidence_prefix=use_evidence_prefix),
        create_new_sealed_metabolic_inhibitor_tree(use_evidence_prefix=use_evidence_prefix),
        create_new_sealed_consortium_smart_contract(use_evidence_prefix=use_evidence_prefix),
        create_new_sealed_epigenetic_chromatin_lattice(use_evidence_prefix=use_evidence_prefix),
        create_new_sealed_autonomous_rail_grid(use_evidence_prefix=use_evidence_prefix),
    ]

    # Run dataset validator across all 5 new environments
    for env in suite:
        rep = ThreeValuedDatasetValidator.validate_query_set(env.held_out_test_queries, env.env_id)
        if not rep.validation_passed:
            raise RuntimeError(f"New sealed environment {env.env_id} failed validation: {rep.diagnostics}")

    return suite
