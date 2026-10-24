"""Phase 8: Brand-New Independent Sealed Evaluation Environments for M66."""
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


def create_source_dag_transitive(seed: int = 601) -> GroundedThreeValuedEnvironment:
    """Source Skill 1: 10-node supply chain logistics hierarchy (DIR_TRANS)."""
    entities = [f"SupplyNodeS{i}" for i in range(10)]
    edges = [(entities[i], entities[i + 1]) for i in range(9)]
    edges.append((entities[0], entities[3]))
    edges.append((entities[2], entities[6]))
    levels = {e: i for i, e in enumerate(entities)}

    base_env = AutonomousEnvironment(
        env_id="m66_source_dag_transitive",
        domain_name="IndustrialSupplyChainLogistics",
        relation_name="ships_freight_cargo_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        held_out_test_queries=[],
        minimal_pairs=[],
        entity_levels=levels,
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


def create_source_metabolic_inhibitor(seed: int = 602) -> GroundedThreeValuedEnvironment:
    """Source Skill 2: 11-enzyme metabolic flux cascade with active inhibitors."""
    entities = [
        "MetabolicEnzymeA0",
        "KinaseEnzymeK1", "KinaseEnzymeK2", "KinaseEnzymeK3",
        "SubstrateS11", "SubstrateS12", "SubstrateS13",
        "SubstrateS21", "SubstrateS22",
        "SubstrateS31", "SubstrateS32",
    ]
    edges = [
        ("MetabolicEnzymeA0", "KinaseEnzymeK1"),
        ("MetabolicEnzymeA0", "KinaseEnzymeK2"),
        ("MetabolicEnzymeA0", "KinaseEnzymeK3"),
        ("KinaseEnzymeK1", "SubstrateS11"),
        ("KinaseEnzymeK1", "SubstrateS12"),
        ("KinaseEnzymeK1", "SubstrateS13"),
        ("KinaseEnzymeK2", "SubstrateS21"),
        ("KinaseEnzymeK2", "SubstrateS22"),
        ("KinaseEnzymeK3", "SubstrateS31"),
        ("KinaseEnzymeK3", "SubstrateS32"),
    ]
    negated = {("KinaseEnzymeK1", "SubstrateS12"), ("KinaseEnzymeK3", "SubstrateS31")}

    base_env = AutonomousEnvironment(
        env_id="m66_source_metabolic_inhibitor",
        domain_name="BiochemicalMetabolismNetwork",
        relation_name="phosphorylates_metabolite_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=EXPANDED_HYPOTHESES["EXPLICIT_REFUTATION"],
        held_out_test_queries=[],
        minimal_pairs=[],
        negated_edges=negated,
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


def create_source_consortium_bipartite(seed: int = 603) -> GroundedThreeValuedEnvironment:
    """Source Skill 3: 9-node blockchain shard governance (bipartite parity)."""
    entities = [
        "ShardValidatorV0", "EscrowBridgeB1", "StateRelayS2", "DisputeArbiterA3",
        "TreasuryVaultT4", "LiquidityPoolL5", "CollateralManagerC6", "AuditRegistryR7", "GuardianNodeG8",
    ]
    edges = [
        ("ShardValidatorV0", "EscrowBridgeB1"),
        ("EscrowBridgeB1", "StateRelayS2"),
        ("StateRelayS2", "DisputeArbiterA3"),
        ("DisputeArbiterA3", "TreasuryVaultT4"),
        ("TreasuryVaultT4", "LiquidityPoolL5"),
        ("LiquidityPoolL5", "CollateralManagerC6"),
        ("CollateralManagerC6", "AuditRegistryR7"),
        ("AuditRegistryR7", "GuardianNodeG8"),
    ]
    levels = {e: i for i, e in enumerate(entities)}

    base_env = AutonomousEnvironment(
        env_id="m66_source_consortium_bipartite",
        domain_name="BlockchainShardConsortium",
        relation_name="executes_cross_shard_call_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=OUT_OF_HYPOTHESIS_BIPARTITE,
        held_out_test_queries=[],
        minimal_pairs=[],
        entity_levels=levels,
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


# --- Sequential Episode Test Environments ---

def create_ep1_epigenetic_chromatin_lattice(seed: int = 604) -> GroundedThreeValuedEnvironment:
    """Episode 1: 8-locus chromatin lattice with open-world reachability and explicit silencers."""
    entities = [f"ChromatinLocusAlpha{i}" for i in range(8)]
    edges = [
        (entities[0], entities[1]), (entities[0], entities[2]),
        (entities[1], entities[3]), (entities[2], entities[4]),
        (entities[3], entities[5]), (entities[4], entities[6]),
        (entities[5], entities[7]), (entities[6], entities[7]),
    ]
    negated = {
        (entities[7], entities[0]), (entities[6], entities[1]),
        (entities[5], entities[2]), (entities[4], entities[3]),
        (entities[3], entities[4]), (entities[2], entities[5]),
        (entities[1], entities[6]), (entities[7], entities[1]),
    }

    hyp = RuleHypothesis(
        hypothesis_id="OPEN_WORLD_EXPLICIT_REFUTATION",
        name="Open-World Transitive with Explicit Negative Constraints",
        is_symmetric=False,
        is_transitive=True,
        has_explicit_refutation=True,
        is_open_world=True,
        description="Open-world lattice with explicit negative methylation silencing constraints.",
    )

    base_env = AutonomousEnvironment(
        env_id="m66_ep1_epigenetic_chromatin_lattice",
        domain_name="EpigeneticChromatinRemodelingAlpha",
        relation_name="propagates_histone_acetylation_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=hyp,
        held_out_test_queries=[],
        minimal_pairs=[],
        negated_edges=negated,
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


def create_ep2_epigenetic_chromatin_lattice(seed: int = 605) -> GroundedThreeValuedEnvironment:
    """Episode 2: 10-locus independent chromatin lattice with disjoint entities and topology."""
    entities = [f"ChromatinLocusBeta{i}" for i in range(10)]
    edges = [
        (entities[0], entities[1]), (entities[0], entities[2]),
        (entities[1], entities[3]), (entities[2], entities[4]),
        (entities[3], entities[5]), (entities[4], entities[6]),
        (entities[5], entities[7]), (entities[6], entities[8]),
        (entities[7], entities[9]), (entities[8], entities[9]),
    ]
    negated = {
        (entities[9], entities[0]), (entities[8], entities[1]),
        (entities[7], entities[2]), (entities[6], entities[3]),
        (entities[5], entities[4]), (entities[4], entities[5]),
        (entities[3], entities[6]), (entities[2], entities[7]),
    }

    hyp = RuleHypothesis(
        hypothesis_id="OPEN_WORLD_EXPLICIT_REFUTATION",
        name="Open-World Transitive with Explicit Negative Constraints",
        is_symmetric=False,
        is_transitive=True,
        has_explicit_refutation=True,
        is_open_world=True,
        description="Open-world lattice with explicit negative methylation silencing constraints.",
    )

    base_env = AutonomousEnvironment(
        env_id="m66_ep2_epigenetic_chromatin_lattice",
        domain_name="EpigeneticChromatinRemodelingBeta",
        relation_name="propagates_histone_acetylation_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=hyp,
        held_out_test_queries=[],
        minimal_pairs=[],
        negated_edges=negated,
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


def create_ep1_autonomous_rail_grid(seed: int = 606) -> GroundedThreeValuedEnvironment:
    """Episode 1: 8-station high-speed rail consist grid (alternating parity)."""
    entities = [f"RailTerminalNorth{i}" for i in range(8)]
    edges = [(entities[i], entities[(i + 1) % 8]) for i in range(8)]

    base_env = AutonomousEnvironment(
        env_id="m66_ep1_autonomous_rail_grid",
        domain_name="AutonomousRailGridNorth",
        relation_name="dispatches_high_speed_consist_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=OUT_OF_HYPOTHESIS_PARITY,
        held_out_test_queries=[],
        minimal_pairs=[],
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


def create_ep2_autonomous_rail_grid(seed: int = 607) -> GroundedThreeValuedEnvironment:
    """Episode 2: 10-station independent rail grid with disjoint entities and route network."""
    entities = [f"RailTerminalSouth{i}" for i in range(10)]
    edges = [(entities[i], entities[(i + 1) % 10]) for i in range(10)]

    base_env = AutonomousEnvironment(
        env_id="m66_ep2_autonomous_rail_grid",
        domain_name="AutonomousRailGridSouth",
        relation_name="dispatches_high_speed_consist_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=OUT_OF_HYPOTHESIS_PARITY,
        held_out_test_queries=[],
        minimal_pairs=[],
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


def create_unrelated_control_quantum_dag(seed: int = 608) -> GroundedThreeValuedEnvironment:
    """Unrelated Control Family: 10-qubit circuit DAG for negative-transfer testing."""
    entities = [f"QubitStateZ{i}" for i in range(10)]
    edges = [(entities[i], entities[i + 1]) for i in range(9)]
    edges.append((entities[0], entities[2]))
    edges.append((entities[5], entities[8]))
    levels = {e: i for i, e in enumerate(entities)}

    base_env = AutonomousEnvironment(
        env_id="m66_ctrl_quantum_teleportation",
        domain_name="QuantumInformationCircuits",
        relation_name="teleports_quantum_state_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        held_out_test_queries=[],
        minimal_pairs=[],
        entity_levels=levels,
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


def build_m66_complete_task_suite() -> dict[str, GroundedThreeValuedEnvironment]:
    """Construct and validate all M66 benchmark environments."""
    suite = {
        "source_dag": create_source_dag_transitive(),
        "source_metabolic": create_source_metabolic_inhibitor(),
        "source_bipartite": create_source_consortium_bipartite(),
        "ep1_chromatin": create_ep1_epigenetic_chromatin_lattice(),
        "ep2_chromatin": create_ep2_epigenetic_chromatin_lattice(),
        "ep1_rail": create_ep1_autonomous_rail_grid(),
        "ep2_rail": create_ep2_autonomous_rail_grid(),
        "ctrl_quantum": create_unrelated_control_quantum_dag(),
    }

    # Validate each environment
    for name, env in suite.items():
        rep = ThreeValuedDatasetValidator.validate_query_set(env.held_out_test_queries, env.env_id)
        if not rep.validation_passed:
            raise RuntimeError(f"Task suite env '{name}' ({env.env_id}) failed validation: {rep.diagnostics}")

    return suite
