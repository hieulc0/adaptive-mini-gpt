"""Phase 2: Multi-Split Task Environment Suite & Independent Reachability Engine for M60.

Provides:
1. Strict segregation between:
   - Meta-Training Tasks: Used exclusively to generate historical learning traces.
   - Development Tasks: Used for pre-evaluation parameter calibration and diagnostic validation.
   - Sealed Meta-Test Tasks: Genuinely unfamiliar structures, vocabularies, and topologies.
2. Independent Graph Reachability Engine for zero-defect oracle verification.
3. Explicit topological complexity profiling.
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
    MinimalPairControl,
)
from adaptive_mini_gpt.skill_composition.environment import _build_balanced_env
from adaptive_mini_gpt.skill_reuse.environment import (
    OUT_OF_HYPOTHESIS_BIPARTITE,
    OUT_OF_HYPOTHESIS_DENSE_CLIQUE,
)

logger = logging.getLogger(__name__)


@dataclass
class ObservableTaskComplexity:
    """Observable structural complexity computed strictly from base observations."""
    env_id: str
    num_nodes: int
    num_edges: int
    graph_diameter: int
    graph_density: float
    mean_degree: float
    transitive_reachability_density: float
    structural_complexity_index: float


def compute_independent_graph_reachability(
    nodes: list[str], edges: list[tuple[str, str]]
) -> dict[tuple[str, str], bool]:
    """Compute all-pairs reachability using Floyd-Warshall transitive closure independently of environment classes."""
    node_set = list(dict.fromkeys(nodes))
    reachable = { (u, v): False for u in node_set for v in node_set }
    for u in node_set:
        reachable[(u, u)] = True
    for u, v in edges:
        if (u, v) in reachable:
            reachable[(u, v)] = True

    # Transitive closure
    for k in node_set:
        for i in node_set:
            for j in node_set:
                if reachable[(i, k)] and reachable[(k, j)]:
                    reachable[(i, j)] = True

    return reachable


def compute_observable_complexity(env: AutonomousEnvironment) -> ObservableTaskComplexity:
    """Extract observable topological complexity strictly from the observed base edges."""
    nodes = list(env.entities)
    n = len(nodes)
    edges = list(env.base_edges)
    m = len(edges)

    max_possible_edges = n * (n - 1) if n > 1 else 1
    density = float(m / max_possible_edges)

    degrees: dict[str, int] = {u: 0 for u in nodes}
    for u, v in edges:
        degrees[u] = degrees.get(u, 0) + 1
        degrees[v] = degrees.get(v, 0) + 1
    mean_deg = float(sum(degrees.values()) / (2 * n)) if n > 0 else 0.0

    dist: dict[tuple[str, str], float] = {}
    for u in nodes:
        for v in nodes:
            dist[(u, v)] = 0.0 if u == v else float("inf")
    for u, v in edges:
        dist[(u, v)] = 1.0

    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[(i, k)] + dist[(k, j)] < dist[(i, j)]:
                    dist[(i, j)] = dist[(i, k)] + dist[(k, j)]

    reachable_pairs = 0
    max_finite_dist = 0
    for i in nodes:
        for j in nodes:
            if i != j and dist[(i, j)] < float("inf"):
                reachable_pairs += 1
                if dist[(i, j)] > max_finite_dist:
                    max_finite_dist = int(dist[(i, j)])

    reachability_density = float(reachable_pairs / max_possible_edges)
    diameter = max_finite_dist

    # Structural complexity index: combines depth, size, and reachability density
    complexity_index = float((n * 0.4) + (diameter * 0.8) + (reachability_density * 2.0))

    return ObservableTaskComplexity(
        env_id=env.env_id,
        num_nodes=n,
        num_edges=m,
        graph_diameter=diameter,
        graph_density=density,
        mean_degree=mean_deg,
        transitive_reachability_density=reachability_density,
        structural_complexity_index=complexity_index,
    )


# =========================================================================
# 1. META-TRAINING WORLDS (For generating historical training traces)
# =========================================================================

def create_train_genomics_hierarchy(seed: int = 10) -> AutonomousEnvironment:
    """Meta-Train Task 1: 6-node genomic transcription hierarchy."""
    entities = ["GeneMasterA", "OperonUnitB", "PromoterRegionC", "RnaPolymeraseD", "ExonCodingE", "ProteinProductF"]
    edges = [
        ("GeneMasterA", "OperonUnitB"),
        ("OperonUnitB", "PromoterRegionC"),
        ("PromoterRegionC", "RnaPolymeraseD"),
        ("RnaPolymeraseD", "ExonCodingE"),
        ("ExonCodingE", "ProteinProductF"),
    ]
    levels = {e: i for i, e in enumerate(entities)}
    return _build_balanced_env(
        env_id="train_genomics_hierarchy",
        domain_name="CellularGenomics",
        relation_name="activates_transcription_of",
        entities=entities,
        edges=edges,
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        entity_levels=levels,
    )


def create_train_cluster_mesh(seed: int = 11) -> AutonomousEnvironment:
    """Meta-Train Task 2: 6-node symmetric cluster mesh."""
    entities = ["MeshNodeA1", "MeshNodeA2", "MeshNodeA3", "MeshNodeB1", "MeshNodeB2", "MeshNodeB3"]
    edges = [
        ("MeshNodeA1", "MeshNodeA2"), ("MeshNodeA2", "MeshNodeA3"), ("MeshNodeA3", "MeshNodeA1"),
        ("MeshNodeB1", "MeshNodeB2"), ("MeshNodeB2", "MeshNodeB3"), ("MeshNodeB3", "MeshNodeB1"),
    ]
    levels = {e: 1 for e in entities}
    return _build_balanced_env(
        env_id="train_cluster_mesh",
        domain_name="DistributedMeshGrid",
        relation_name="replicates_state_to",
        entities=entities,
        edges=edges,
        hypothesis=OUT_OF_HYPOTHESIS_DENSE_CLIQUE,
        entity_levels=levels,
    )


def create_train_deep_causal_chain(seed: int = 12) -> AutonomousEnvironment:
    """Meta-Train Task 3: 7-node causal DAG with diameter 6."""
    entities = [f"CausalFactorF{i}" for i in range(7)]
    edges = [(entities[i], entities[i + 1]) for i in range(6)]
    levels = {e: i for i, e in enumerate(entities)}
    return _build_balanced_env(
        env_id="train_deep_causal_chain",
        domain_name="EpidemiologicalCausality",
        relation_name="induces_pathological_change_in",
        entities=entities,
        edges=edges,
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        entity_levels=levels,
    )


# =========================================================================
# 2. DEVELOPMENT WORLDS (Calibration, Tuning, Diagnostic Checks)
# =========================================================================

def create_dev_deep_chain_environment(seed: int = 42) -> AutonomousEnvironment:
    """Dev Task 1: 8-node deep transitive DAG with diameter 7."""
    entities = [
        "ComponentStageC1", "AssemblyUnitA2", "SubsystemModuleS3", "CalibrationRigR4",
        "IntegrationBayI5", "AvionicsHousingH6", "PropulsionCouplerP7", "OrbitalPayloadO8",
    ]
    edges = [
        ("ComponentStageC1", "AssemblyUnitA2"),
        ("AssemblyUnitA2", "SubsystemModuleS3"),
        ("SubsystemModuleS3", "CalibrationRigR4"),
        ("CalibrationRigR4", "IntegrationBayI5"),
        ("IntegrationBayI5", "AvionicsHousingH6"),
        ("AvionicsHousingH6", "PropulsionCouplerP7"),
        ("PropulsionCouplerP7", "OrbitalPayloadO8"),
    ]
    levels = {e: i for i, e in enumerate(entities)}
    return _build_balanced_env(
        env_id="dev_deep_chain_dag",
        domain_name="AerospacePayloadPipeline",
        relation_name="feeds_downstream_stage_to",
        entities=entities,
        edges=edges,
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        entity_levels=levels,
    )


def create_dev_wide_branching_tree_environment(seed: int = 42) -> AutonomousEnvironment:
    """Dev Task 2: 8-node wide branching tree with diameter 2."""
    entities = [
        "ClusterRootR0",
        "SubClusterA1", "SubClusterB2", "SubClusterC3",
        "LeafWorkerA11", "LeafWorkerA12", "LeafWorkerB21", "LeafWorkerC31",
    ]
    edges = [
        ("ClusterRootR0", "SubClusterA1"),
        ("ClusterRootR0", "SubClusterB2"),
        ("ClusterRootR0", "SubClusterC3"),
        ("SubClusterA1", "LeafWorkerA11"),
        ("SubClusterA1", "LeafWorkerA12"),
        ("SubClusterB2", "LeafWorkerB21"),
        ("SubClusterC3", "LeafWorkerC31"),
    ]
    levels = {
        "ClusterRootR0": 0,
        "SubClusterA1": 1, "SubClusterB2": 1, "SubClusterC3": 1,
        "LeafWorkerA11": 2, "LeafWorkerA12": 2, "LeafWorkerB21": 2, "LeafWorkerC31": 2,
    }
    return _build_balanced_env(
        env_id="dev_wide_branching_tree",
        domain_name="CloudClusterDispatch",
        relation_name="delegates_workload_to",
        entities=entities,
        edges=edges,
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        entity_levels=levels,
    )


def create_dev_balanced_hierarchy_environment(seed: int = 42) -> AutonomousEnvironment:
    """Dev Task 3: 6-node hierarchical corporate structure with diameter 5 (frozen source anchor)."""
    entities = [
        "JuniorAnalystJ1", "SeniorAssociateS2", "TeamLeadT3",
        "EngineeringDirectorD4", "VicePresidentV5", "ChiefExecutiveC6",
    ]
    edges = [
        ("JuniorAnalystJ1", "SeniorAssociateS2"),
        ("SeniorAssociateS2", "TeamLeadT3"),
        ("TeamLeadT3", "EngineeringDirectorD4"),
        ("EngineeringDirectorD4", "VicePresidentV5"),
        ("VicePresidentV5", "ChiefExecutiveC6"),
    ]
    levels = {e: i + 1 for i, e in enumerate(entities)}
    return _build_balanced_env(
        env_id="dev_balanced_hierarchy",
        domain_name="CorporateGovernance",
        relation_name="reports_transitively_to",
        entities=entities,
        edges=edges,
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        entity_levels=levels,
    )


def create_dev_dense_cluster_exceptions_environment(seed: int = 42) -> AutonomousEnvironment:
    """Dev Task 4: 6-node dense security cluster with peripheral exceptions."""
    entities = [
        "CoreSecurityNode1", "CoreSecurityNode2", "CoreSecurityNode3", "CoreSecurityNode4",
        "PeripheralSensorNode5", "IsolatedGatewayNode6",
    ]
    edges = [
        ("CoreSecurityNode1", "CoreSecurityNode2"),
        ("CoreSecurityNode2", "CoreSecurityNode3"),
        ("CoreSecurityNode3", "CoreSecurityNode4"),
        ("CoreSecurityNode4", "CoreSecurityNode1"),
    ]
    levels = {
        "CoreSecurityNode1": 2, "CoreSecurityNode2": 2, "CoreSecurityNode3": 2, "CoreSecurityNode4": 2,
        "PeripheralSensorNode5": 0, "IsolatedGatewayNode6": 0,
    }
    return _build_balanced_env(
        env_id="dev_dense_cluster_exceptions",
        domain_name="SecureSubnetMesh",
        relation_name="in_same_security_zone",
        entities=entities,
        edges=edges,
        hypothesis=OUT_OF_HYPOTHESIS_DENSE_CLIQUE,
        entity_levels=levels,
    )


def create_dev_incompatible_parity_environment(seed: int = 42) -> AutonomousEnvironment:
    """Dev Task 5: 6-node parity constraint environment (negative transfer control)."""
    entities = [
        "StateAlpha1", "StateBeta2", "StateGamma3",
        "StateDelta4", "StateEpsilon5", "StateZeta6",
    ]
    edges = [
        ("StateAlpha1", "StateBeta2"),
        ("StateBeta2", "StateGamma3"),
        ("StateGamma3", "StateDelta4"),
        ("StateDelta4", "StateEpsilon5"),
        ("StateEpsilon5", "StateZeta6"),
        ("StateZeta6", "StateAlpha1"),
    ]
    return _build_balanced_env(
        env_id="dev_incompatible_parity",
        domain_name="GeopoliticalCoalition",
        relation_name="holds_mutual_pact_with",
        entities=entities,
        edges=edges,
        hypothesis=OUT_OF_HYPOTHESIS_PARITY,
    )


def create_dev_non_transitive_bipartite_environment(seed: int = 42) -> AutonomousEnvironment:
    """Dev Task 6: 6-node non-transitive bipartite partition environment where 2-hop relations fail."""
    entities = ["PartyNodeA0", "PartyNodeB1", "PartyNodeC2", "PartyNodeD3", "PartyNodeE4", "PartyNodeF5"]
    edges = [
        ("PartyNodeA0", "PartyNodeB1"),
        ("PartyNodeB1", "PartyNodeC2"),
        ("PartyNodeC2", "PartyNodeD3"),
        ("PartyNodeD3", "PartyNodeE4"),
        ("PartyNodeE4", "PartyNodeF5"),
    ]
    levels = {e: i for i, e in enumerate(entities)}
    return _build_balanced_env(
        env_id="dev_non_transitive_bipartite",
        domain_name="BipartitePoliticalCoalition",
        relation_name="opposes_ideological_bloc",
        entities=entities,
        edges=edges,
        hypothesis=OUT_OF_HYPOTHESIS_BIPARTITE,
        entity_levels=levels,
    )


# =========================================================================
# 3. SEALED FINAL META-TEST WORLDS (Strictly held-out evaluation)
# =========================================================================

def create_meta_long_chain_aerospace(seed: int = 100) -> AutonomousEnvironment:
    """Meta-Test Task 1: 10-node deep aerospace assembly DAG with diameter 9."""
    entities = [f"AeroModuleM{i}" for i in range(10)]
    edges = [(entities[i], entities[i + 1]) for i in range(9)]
    levels = {e: i for i, e in enumerate(entities)}
    return _build_balanced_env(
        env_id="meta_long_chain_aerospace",
        domain_name="AerospaceAvionicsChain",
        relation_name="assembles_into_subsystem",
        entities=entities,
        edges=edges,
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        entity_levels=levels,
    )


def create_meta_branching_pyramid_cloud(seed: int = 101) -> AutonomousEnvironment:
    """Meta-Test Task 2: 10-node branching cloud deployment hierarchy with diameter 3."""
    entities = [
        "CloudMasterNode0",
        "ZoneAlpha1", "ZoneBeta2", "ZoneGamma3",
        "PodA11", "PodA12", "PodB21", "PodB22", "PodC31", "PodC32",
    ]
    edges = [
        ("CloudMasterNode0", "ZoneAlpha1"),
        ("CloudMasterNode0", "ZoneBeta2"),
        ("CloudMasterNode0", "ZoneGamma3"),
        ("ZoneAlpha1", "PodA11"), ("ZoneAlpha1", "PodA12"),
        ("ZoneBeta2", "PodB21"), ("ZoneBeta2", "PodB22"),
        ("ZoneGamma3", "PodC31"), ("ZoneGamma3", "PodC32"),
    ]
    levels = {
        "CloudMasterNode0": 0,
        "ZoneAlpha1": 1, "ZoneBeta2": 1, "ZoneGamma3": 1,
        "PodA11": 2, "PodA12": 2, "PodB21": 2, "PodB22": 2, "PodC31": 2, "PodC32": 2,
    }
    return _build_balanced_env(
        env_id="meta_branching_pyramid_cloud",
        domain_name="KubernetesCloudDeployment",
        relation_name="orchestrates_container_pod",
        entities=entities,
        edges=edges,
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        entity_levels=levels,
    )


def create_meta_cyclic_parity_treaty(seed: int = 102) -> AutonomousEnvironment:
    """Meta-Test Task 3: 8-node cyclic treaty parity constraint network."""
    entities = [f"NationSovereignN{i}" for i in range(8)]
    edges = [(entities[i], entities[(i + 1) % 8]) for i in range(8)]
    levels = {e: 1 for e in entities}
    return _build_balanced_env(
        env_id="meta_cyclic_parity_treaty",
        domain_name="InternationalNonProliferation",
        relation_name="shares_nuclear_safeguard_pact",
        entities=entities,
        edges=edges,
        hypothesis=OUT_OF_HYPOTHESIS_PARITY,
        entity_levels=levels,
    )


def create_meta_modular_network_biomedical(seed: int = 103) -> AutonomousEnvironment:
    """Meta-Test Task 4: 8-node biomedical dense phosphorylation module with diameter 7."""
    entities = [f"ProteinKinaseK{i}" for i in range(8)]
    edges = [
        (entities[0], entities[1]), (entities[1], entities[2]), (entities[2], entities[3]), (entities[3], entities[0]),
        (entities[4], entities[5]), (entities[5], entities[6]), (entities[6], entities[7]), (entities[7], entities[4]),
        (entities[3], entities[4]),
    ]
    levels = {e: 2 for e in entities}
    return _build_balanced_env(
        env_id="meta_modular_network_biomedical",
        domain_name="KinasePhosphorylationNetwork",
        relation_name="catalyzes_phosphorylation_of",
        entities=entities,
        edges=edges,
        hypothesis=OUT_OF_HYPOTHESIS_DENSE_CLIQUE,
        entity_levels=levels,
    )


def create_meta_sparse_lattice_logistics(seed: int = 104) -> AutonomousEnvironment:
    """Meta-Test Task 5: 8-node sparse logistics grid lattice with diameter 4."""
    entities = [f"LogisticsHubH{i}" for i in range(8)]
    edges = [
        (entities[0], entities[1]), (entities[0], entities[2]),
        (entities[1], entities[3]), (entities[2], entities[4]),
        (entities[3], entities[5]), (entities[4], entities[6]),
        (entities[5], entities[7]), (entities[6], entities[7]),
    ]
    levels = {
        entities[0]: 0,
        entities[1]: 1, entities[2]: 1,
        entities[3]: 2, entities[4]: 2,
        entities[5]: 3, entities[6]: 3,
        entities[7]: 4,
    }
    return _build_balanced_env(
        env_id="meta_sparse_lattice_logistics",
        domain_name="IntermodalFreightLattice",
        relation_name="forwards_container_consignment_to",
        entities=entities,
        edges=edges,
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        entity_levels=levels,
    )


# =========================================================================
# Multi-Split Suite Builders
# =========================================================================

def build_meta_training_task_suite() -> list[AutonomousEnvironment]:
    """Return list of meta-training environments."""
    return [
        create_train_genomics_hierarchy(seed=10),
        create_train_cluster_mesh(seed=11),
        create_train_deep_causal_chain(seed=12),
    ]


def build_development_task_suite() -> list[AutonomousEnvironment]:
    """Return list of development target environments."""
    return [
        create_dev_deep_chain_environment(seed=42),
        create_dev_wide_branching_tree_environment(seed=42),
        create_dev_dense_cluster_exceptions_environment(seed=42),
        create_dev_incompatible_parity_environment(seed=42),
        create_dev_non_transitive_bipartite_environment(seed=42),
    ]


def build_sealed_meta_test_task_suite() -> list[AutonomousEnvironment]:
    """Return list of sealed meta-test target environments."""
    return [
        create_meta_long_chain_aerospace(seed=100),
        create_meta_branching_pyramid_cloud(seed=101),
        create_meta_cyclic_parity_treaty(seed=102),
        create_meta_modular_network_biomedical(seed=103),
        create_meta_sparse_lattice_logistics(seed=104),
    ]
