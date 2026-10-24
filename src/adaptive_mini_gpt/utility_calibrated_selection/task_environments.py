"""Phase 3: Fully Isolated Sequential Environments with Strict 3-Stage Separation.

Enforces strict isolation across:
Stage 1: Episode 1 Candidate Acquisition (World G1, Vocab V1)
Stage 2: Independent Qualification (World G_val, Vocab V_val)
Stage 3: Episode 2 Future Sealed Transfer (World G2, Vocab V2)

Guarantees Episode 2 graph and queries are never accessed during candidate training,
qualification, or hyperparameter selection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.guided_experiment.environment import (
    EXPANDED_HYPOTHESES,
    OUT_OF_HYPOTHESIS_PARITY,
    AutonomousEnvironment,
    EnvironmentQuery,
    RuleHypothesis,
)
from adaptive_mini_gpt.skill_reuse.environment import OUT_OF_HYPOTHESIS_BIPARTITE
from adaptive_mini_gpt.three_valued_reasoning.semantics import (
    GroundedThreeValuedEnvironment,
    ThreeValuedDatasetValidator,
)

logger = logging.getLogger(__name__)


@dataclass
class ThreeStageFamilySuite:
    """A rule family packaged into strictly separated Stage 1, Stage 2, and Stage 3 environments."""
    family_name: str
    stage1_train_env: GroundedThreeValuedEnvironment
    stage2_qual_env: GroundedThreeValuedEnvironment
    stage3_eval_env: GroundedThreeValuedEnvironment


# =========================================================================
# 1. Family 1: Autonomous Rail Parity Grid (Alternating Power Cycles)
# =========================================================================

def create_rail_stage1_train_env(seed: int = 701) -> GroundedThreeValuedEnvironment:
    """Stage 1 (Acquisition): Station entities Alpha0..Alpha7, train graph G1."""
    entities = [f"RailStationAlpha{i}" for i in range(8)]
    edges = [(entities[i], entities[i + 1]) for i in range(7)]
    base_env = AutonomousEnvironment(
        env_id="m67_rail_stage1_train",
        domain_name="AutonomousRailParityGridAlpha",
        relation_name="dispatches_high_speed_consist_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=OUT_OF_HYPOTHESIS_PARITY,
        held_out_test_queries=[],
        minimal_pairs=[],
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


def create_rail_stage2_qual_env(seed: int = 702) -> GroundedThreeValuedEnvironment:
    """Stage 2 (Qualification): Intermediate separate graph G_val with Depot Beta0..Beta7."""
    entities = [f"RailDepotBeta{i}" for i in range(8)]
    edges = [(entities[i], entities[i + 1]) for i in range(7)]
    base_env = AutonomousEnvironment(
        env_id="m67_rail_stage2_qual",
        domain_name="AutonomousRailParityGridBeta",
        relation_name="dispatches_high_speed_consist_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=OUT_OF_HYPOTHESIS_PARITY,
        held_out_test_queries=[],
        minimal_pairs=[],
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


def create_rail_stage3_eval_env(seed: int = 703) -> GroundedThreeValuedEnvironment:
    """Stage 3 (Future Encounter): Sealed future encounter graph G2 with Terminal Gamma0..Gamma7."""
    entities = [f"TerminalStationGamma{i}" for i in range(8)]
    edges = [(entities[i], entities[i + 1]) for i in range(7)]
    base_env = AutonomousEnvironment(
        env_id="m67_rail_stage3_eval",
        domain_name="AutonomousRailParityGridGamma",
        relation_name="dispatches_high_speed_consist_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=OUT_OF_HYPOTHESIS_PARITY,
        held_out_test_queries=[],
        minimal_pairs=[],
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


# =========================================================================
# 2. Family 2: Epigenetic Chromatin Lattice (Open-World Silencer Lattice)
# =========================================================================

def create_chromatin_stage1_train_env(seed: int = 704) -> GroundedThreeValuedEnvironment:
    """Stage 1: Loci Alpha0..Alpha7 with explicit silencer constraints."""
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
        env_id="m67_chromatin_stage1_train",
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


def create_chromatin_stage2_qual_env(seed: int = 705) -> GroundedThreeValuedEnvironment:
    """Stage 2: Validation Loci Beta0..Beta8."""
    entities = [f"ChromatinLocusBeta{i}" for i in range(9)]
    edges = [
        (entities[0], entities[1]), (entities[0], entities[2]),
        (entities[1], entities[3]), (entities[2], entities[4]),
        (entities[3], entities[5]), (entities[4], entities[6]),
        (entities[5], entities[7]), (entities[6], entities[8]),
    ]
    negated = {
        (entities[8], entities[0]), (entities[7], entities[1]),
        (entities[6], entities[2]), (entities[5], entities[3]),
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
        env_id="m67_chromatin_stage2_qual",
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


def create_chromatin_stage3_eval_env(seed: int = 706) -> GroundedThreeValuedEnvironment:
    """Stage 3: Sealed Future Encounter with Target Loci Gamma0..Gamma9."""
    entities = [f"ChromatinLocusGamma{i}" for i in range(10)]
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
        env_id="m67_chromatin_stage3_eval",
        domain_name="EpigeneticChromatinRemodelingGamma",
        relation_name="propagates_histone_acetylation_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=hyp,
        held_out_test_queries=[],
        minimal_pairs=[],
        negated_edges=negated,
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


# =========================================================================
# 3. Family 3: Circuit Fault Cascade (Novel Cascading Inhibitor Trees)
# =========================================================================

def create_circuit_stage1_train_env(seed: int = 707) -> GroundedThreeValuedEnvironment:
    """Stage 1: Logic gates GateAlpha0..Alpha7 with cascading trip refutations."""
    entities = [f"CircuitGateAlpha{i}" for i in range(8)]
    edges = [
        (entities[0], entities[1]), (entities[0], entities[2]),
        (entities[1], entities[3]), (entities[1], entities[4]),
        (entities[2], entities[5]), (entities[2], entities[6]),
        (entities[3], entities[7]),
    ]
    negated = {(entities[2], entities[7]), (entities[4], entities[0])}
    hyp = RuleHypothesis(
        hypothesis_id="CASCADING_TRIP_INHIBITOR",
        name="Branching Circuit Overcurrent Cascade with Negative Trip Gates",
        is_symmetric=False,
        is_transitive=True,
        has_explicit_refutation=True,
        is_open_world=False,
        description="Cascading overcurrent propagation with active circuit breaker inhibitor gates.",
    )
    base_env = AutonomousEnvironment(
        env_id="m67_circuit_stage1_train",
        domain_name="CircuitFaultCascadeAlpha",
        relation_name="propagates_overcurrent_surge_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=hyp,
        held_out_test_queries=[],
        minimal_pairs=[],
        negated_edges=negated,
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


def create_circuit_stage2_qual_env(seed: int = 708) -> GroundedThreeValuedEnvironment:
    """Stage 2: Validation gates GateBeta0..Beta8."""
    entities = [f"CircuitGateBeta{i}" for i in range(9)]
    edges = [
        (entities[0], entities[1]), (entities[0], entities[2]),
        (entities[1], entities[3]), (entities[1], entities[4]),
        (entities[2], entities[5]), (entities[2], entities[6]),
        (entities[3], entities[7]), (entities[4], entities[8]),
    ]
    negated = {(entities[2], entities[7]), (entities[5], entities[0])}
    hyp = RuleHypothesis(
        hypothesis_id="CASCADING_TRIP_INHIBITOR",
        name="Branching Circuit Overcurrent Cascade with Negative Trip Gates",
        is_symmetric=False,
        is_transitive=True,
        has_explicit_refutation=True,
        is_open_world=False,
        description="Cascading overcurrent propagation with active circuit breaker inhibitor gates.",
    )
    base_env = AutonomousEnvironment(
        env_id="m67_circuit_stage2_qual",
        domain_name="CircuitFaultCascadeBeta",
        relation_name="propagates_overcurrent_surge_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=hyp,
        held_out_test_queries=[],
        minimal_pairs=[],
        negated_edges=negated,
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


def create_circuit_stage3_eval_env(seed: int = 709) -> GroundedThreeValuedEnvironment:
    """Stage 3: Sealed Future Encounter with Relays RelayGamma0..Gamma9."""
    entities = [f"CircuitRelayGamma{i}" for i in range(10)]
    edges = [
        (entities[0], entities[1]), (entities[0], entities[2]),
        (entities[1], entities[3]), (entities[1], entities[4]),
        (entities[2], entities[5]), (entities[2], entities[6]),
        (entities[3], entities[7]), (entities[4], entities[8]),
        (entities[5], entities[9]),
    ]
    negated = {(entities[2], entities[7]), (entities[6], entities[0])}
    hyp = RuleHypothesis(
        hypothesis_id="CASCADING_TRIP_INHIBITOR",
        name="Branching Circuit Overcurrent Cascade with Negative Trip Gates",
        is_symmetric=False,
        is_transitive=True,
        has_explicit_refutation=True,
        is_open_world=False,
        description="Cascading overcurrent propagation with active circuit breaker inhibitor gates.",
    )
    base_env = AutonomousEnvironment(
        env_id="m67_circuit_stage3_eval",
        domain_name="CircuitFaultCascadeGamma",
        relation_name="propagates_overcurrent_surge_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=hyp,
        held_out_test_queries=[],
        minimal_pairs=[],
        negated_edges=negated,
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)


# =========================================================================
# 4. Known Reusable, Incomplete, & Control Environments
# =========================================================================

def create_known_reusable_dag_env(seed: int = 601) -> GroundedThreeValuedEnvironment:
    """Known reusable family environment (source_dag_transitive domain)."""
    entities = [f"SupplyNodeS{i}" for i in range(10)]
    edges = [(entities[i], entities[i + 1]) for i in range(9)]
    edges.append((entities[0], entities[3]))
    edges.append((entities[2], entities[6]))
    levels = {e: i for i, e in enumerate(entities)}

    base_env = AutonomousEnvironment(
        env_id="m67_known_reusable_dag",
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


def create_known_incomplete_dag_env(seed: int = 601) -> GroundedThreeValuedEnvironment:
    """Known family with missing observations (should NOT trigger adapter allocation)."""
    env = create_known_reusable_dag_env(seed=seed)
    env.env_id = "m67_known_incomplete_dag"
    return env


def create_unrelated_control_quantum_dag(seed: int = 608) -> GroundedThreeValuedEnvironment:
    """Unrelated family for negative transfer auditing."""
    entities = [f"QubitStateQ{i}" for i in range(10)]
    edges = [(entities[i], entities[i + 1]) for i in range(9)]
    edges.append((entities[0], entities[4]))
    edges.append((entities[2], entities[7]))

    hyp = RuleHypothesis(
        hypothesis_id="CROSS_TIER_JUMP_DAG",
        name="Hierarchical Tier Teleportation with Cross-Tier Jump Constraint",
        is_symmetric=False,
        is_transitive=True,
        has_explicit_refutation=True,
        is_open_world=False,
        description="Quantum circuit state teleportation with explicit tier jump barriers.",
    )
    base_env = AutonomousEnvironment(
        env_id="m67_unrelated_quantum_teleportation_dag",
        domain_name="QuantumInformationCircuits",
        relation_name="teleports_quantum_state_to",
        entities=entities,
        base_edges=edges,
        true_hypothesis=hyp,
        held_out_test_queries=[],
        minimal_pairs=[],
        negated_edges={(entities[5], entities[0]), (entities[8], entities[1])},
    )
    return GroundedThreeValuedEnvironment(base_env, use_evidence_prefix=True)
