"""Phase 2: Task Complexity Environment Suite and Observable Topological Metrics.

Disentangles graph diameter (depth) from node count (size), relation types,
and structural exception density.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.guided_experiment.environment import (
    AutonomousEnvironment,
    EnvironmentQuery,
    EXPANDED_HYPOTHESES,
    MinimalPairControl,
    OUT_OF_HYPOTHESIS_PARITY,
)
from adaptive_mini_gpt.skill_composition.environment import _build_balanced_env
from adaptive_mini_gpt.skill_reuse.environment import (
    OUT_OF_HYPOTHESIS_BIPARTITE,
    OUT_OF_HYPOTHESIS_DENSE_CLIQUE,
)


@dataclass
class ObservableTaskComplexity:
    """Observable structural features of a relational task without peeking at test labels."""
    env_id: str
    num_nodes: int
    num_edges: int
    graph_diameter: int
    graph_density: float
    mean_degree: float
    transitive_reachability_density: float
    structural_complexity_index: float  # Combined normalized complexity score


def compute_observable_complexity(env: AutonomousEnvironment) -> ObservableTaskComplexity:
    """Compute observable topological complexity metrics directly from observed base edges."""
    nodes = list(env.entities)
    n = len(nodes)
    edges = list(env.base_edges)
    m = len(edges)

    # 1. Density
    max_possible_edges = n * (n - 1) if n > 1 else 1
    density = float(m / max_possible_edges)

    # 2. Mean degree
    degrees: dict[str, int] = {u: 0 for u in nodes}
    for u, v in edges:
        degrees[u] = degrees.get(u, 0) + 1
        degrees[v] = degrees.get(v, 0) + 1
    mean_deg = float(sum(degrees.values()) / (2 * n)) if n > 0 else 0.0

    # 3. Transitive Closure & Graph Diameter using Floyd-Warshall
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

    # Structural complexity index: combines depth, size, and reachability entropy
    # Higher for deep chains and dense exception graphs, lower for simple isomorphic trees
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


# ---------------------------------------------------------------------------
# Independent Target Environments (Disentangling Depth and Size)
# ---------------------------------------------------------------------------

def create_deep_chain_environment(seed: int = 42) -> AutonomousEnvironment:
    """Target World 1: 8 nodes, diameter d=7 (Deep Chain DAG).

    High diameter, sparse connectivity. Tests long-range composition.
    """
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
        env_id="target_deep_chain_dag",
        domain_name="AerospacePayloadPipeline",
        relation_name="feeds_downstream_stage_to",
        entities=entities,
        edges=edges,
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        entity_levels=levels,
    )


def create_wide_branching_tree_environment(seed: int = 42) -> AutonomousEnvironment:
    """Target World 2: 8 nodes, diameter d=3 (Wide Branching Tree).

    Same node count as Deep Chain (N=8), but shallow diameter (d=3).
    Tests whether N=8 difficulty is driven by node count or diameter depth.
    """
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
        env_id="target_wide_branching_tree",
        domain_name="CloudClusterDispatch",
        relation_name="delegates_workload_to",
        entities=entities,
        edges=edges,
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        entity_levels=levels,
    )


def create_balanced_hierarchy_environment(seed: int = 42) -> AutonomousEnvironment:
    """Target World 3: 6 nodes, diameter d=4 (Corporate Governance DAG).

    Isomorphic in size and diameter to the source Genomics pretraining task.
    """
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
        env_id="target_balanced_hierarchy",
        domain_name="CorporateGovernance",
        relation_name="reports_transitively_to",
        entities=entities,
        edges=edges,
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        entity_levels=levels,
    )


def create_dense_cluster_exceptions_environment(seed: int = 42) -> AutonomousEnvironment:
    """Target World 4: 6 nodes, diameter d=2 (Dense Clique with Peripheral Exceptions).

    Dense 4-node core with 2 peripheral exception nodes.
    """
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
        env_id="target_dense_cluster_exceptions",
        domain_name="SecureSubnetMesh",
        relation_name="in_same_security_zone",
        entities=entities,
        edges=edges,
        hypothesis=OUT_OF_HYPOTHESIS_DENSE_CLIQUE,
        entity_levels=levels,
    )


def create_incompatible_parity_environment(seed: int = 42) -> AutonomousEnvironment:
    """Target World 5: 6 nodes, diameter d=3 (Incompatible Parity Cycle).

    Negative control: mutual defense pact alternating cycle.
    """
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
        env_id="target_incompatible_parity",
        domain_name="GeopoliticalCoalition",
        relation_name="holds_mutual_pact_with",
        entities=entities,
        edges=edges,
        hypothesis=OUT_OF_HYPOTHESIS_PARITY,
    )
