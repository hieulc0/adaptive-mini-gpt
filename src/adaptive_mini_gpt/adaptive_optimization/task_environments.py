"""Phase 2: Task Environments and Observable Complexity for M58 Benchmark.

Separates:
1. Development Tasks: Used for controller thresholding, learning-curve slope tuning, and validation.
2. Sealed Final Meta-Test Tasks: Unfamiliar vocabularies, unseen graph topologies, and edge densities.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment, EnvironmentQuery, MinimalPairControl
from adaptive_mini_gpt.skill_composition.environment import _build_balanced_env

logger = logging.getLogger(__name__)


@dataclass
class ObservableTaskComplexity:
    """Observable structural complexity computed strictly from unlabelled relation observations."""
    env_id: str
    num_nodes: int
    num_edges: int
    graph_diameter: int
    graph_density: float
    mean_degree: float
    transitive_reachability_density: float
    structural_complexity_index: float


def compute_observable_complexity(env: AutonomousEnvironment) -> ObservableTaskComplexity:
    """Extract observable topological complexity strictly from the task relational structure."""
    nodes = set(env.nodes)
    n = len(nodes)
    if n == 0:
        return ObservableTaskComplexity(
            env_id=env.env_id, num_nodes=0, num_edges=0, graph_diameter=0,
            graph_density=0.0, mean_degree=0.0, transitive_reachability_density=0.0,
            structural_complexity_index=0.0,
        )

    edges = set(env.directed_edges)
    num_edges = len(edges)
    density = num_edges / (n * (n - 1)) if n > 1 else 0.0
    mean_deg = num_edges / n

    # Compute shortest path lengths using Floyd-Warshall
    node_list = sorted(list(nodes))
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)
    for u, v in edges:
        if u in node_to_idx and v in node_to_idx:
            dist[node_to_idx[u], node_to_idx[v]] = 1

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    finite_dists = dist[np.isfinite(dist) & (dist > 0)]
    diameter = int(np.max(finite_dists)) if len(finite_dists) > 0 else 0
    reachability_count = np.sum(np.isfinite(dist) & (dist > 0))
    reachability_density = reachability_count / (n * (n - 1)) if n > 1 else 0.0

    # Composite structural complexity index
    complexity_index = float(diameter + (density * 10.0) + (mean_deg * 2.0) + reachability_density)

    return ObservableTaskComplexity(
        env_id=env.env_id,
        num_nodes=n,
        num_edges=num_edges,
        graph_diameter=diameter,
        graph_density=density,
        mean_degree=mean_deg,
        transitive_reachability_density=reachability_density,
        structural_complexity_index=complexity_index,
    )


# =========================================================================
# DEVELOPMENT TASKS (Pre-exposure, parameter calibration, validation)
# =========================================================================

def create_dev_deep_chain_environment(seed: int = 42) -> AutonomousEnvironment:
    """Dev Task: 8-node deep transitive DAG with diameter 7."""
    nodes = [f"Stage{i}" for i in range(8)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(7)]
    return _build_balanced_env(
        env_id="dev_deep_chain_dag",
        nodes=nodes,
        directed_edges=edges,
        relation_name="feeds_downstream_stage_to",
        seed=seed,
    )


def create_dev_wide_branching_tree_environment(seed: int = 42) -> AutonomousEnvironment:
    """Dev Task: 8-node wide branching tree with diameter 2."""
    nodes = ["ClusterRootR0", "SubClusterA1", "SubClusterB2", "SubClusterC3",
             "LeafWorkerA11", "LeafWorkerA12", "LeafWorkerB21", "LeafWorkerC31"]
    edges = [
        ("ClusterRootR0", "SubClusterA1"), ("ClusterRootR0", "SubClusterB2"), ("ClusterRootR0", "SubClusterC3"),
        ("SubClusterA1", "LeafWorkerA11"), ("SubClusterA1", "LeafWorkerA12"),
        ("SubClusterB2", "LeafWorkerB21"), ("SubClusterC3", "LeafWorkerC31"),
    ]
    return _build_balanced_env(
        env_id="dev_wide_branching_tree",
        nodes=nodes,
        directed_edges=edges,
        relation_name="delegates_workload_to",
        seed=seed,
    )


def create_dev_balanced_hierarchy_environment(seed: int = 42) -> AutonomousEnvironment:
    """Dev Task: 6-node hierarchical corporate structure with diameter 5."""
    nodes = [f"CorpNode{i}" for i in range(6)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(5)]
    return _build_balanced_env(
        env_id="dev_balanced_hierarchy",
        nodes=nodes,
        directed_edges=edges,
        relation_name="reports_transitively_to",
        seed=seed,
    )


def create_dev_dense_cluster_exceptions_environment(seed: int = 42) -> AutonomousEnvironment:
    """Dev Task: 6-node dense security cluster with peripheral exceptions."""
    nodes = [f"SecurityNode{i}" for i in range(6)]
    edges = [
        (nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[3]),
        (nodes[3], nodes[0]), (nodes[0], nodes[2]), (nodes[1], nodes[3]),
        (nodes[4], nodes[5]),
    ]
    return _build_balanced_env(
        env_id="dev_dense_cluster_exceptions",
        nodes=nodes,
        directed_edges=edges,
        relation_name="in_same_security_zone",
        seed=seed,
    )


def create_dev_incompatible_parity_environment(seed: int = 42) -> AutonomousEnvironment:
    """Dev Task: 6-node parity constraint environment (negative transfer control)."""
    nodes = [f"PactState{i}" for i in range(6)]
    edges = [
        (nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0]),
        (nodes[3], nodes[4]), (nodes[4], nodes[5]), (nodes[5], nodes[3]),
    ]
    return _build_balanced_env(
        env_id="dev_incompatible_parity",
        nodes=nodes,
        directed_edges=edges,
        relation_name="holds_mutual_pact_with",
        seed=seed,
    )


# =========================================================================
# SEALED META-TEST TASKS (Unfamiliar vocabularies, unseen graph topologies)
# =========================================================================

def create_meta_long_chain_aerospace(seed: int = 100) -> AutonomousEnvironment:
    """Meta-Test Task 1: 10-node deep aerospace assembly DAG with diameter 9."""
    nodes = [f"AeroModuleM{i}" for i in range(10)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(9)]
    return _build_balanced_env(
        env_id="meta_long_chain_aerospace",
        nodes=nodes,
        directed_edges=edges,
        relation_name="assembles_into_subsystem",
        seed=seed,
    )


def create_meta_branching_pyramid_cloud(seed: int = 101) -> AutonomousEnvironment:
    """Meta-Test Task 2: 10-node cloud microservice pyramid with diameter 3."""
    nodes = ["GatewayGW", "AuthSvcA1", "BillingSvcB2", "CatalogSvcC3",
             "DbWorkerD1", "DbWorkerD2", "CacheWorkerC1", "CacheWorkerC2",
             "QueueWorkerQ1", "AuditWorkerA2"]
    edges = [
        ("GatewayGW", "AuthSvcA1"), ("GatewayGW", "BillingSvcB2"), ("GatewayGW", "CatalogSvcC3"),
        ("AuthSvcA1", "DbWorkerD1"), ("AuthSvcA1", "CacheWorkerC1"),
        ("BillingSvcB2", "DbWorkerD2"), ("BillingSvcB2", "AuditWorkerA2"),
        ("CatalogSvcC3", "CacheWorkerC2"), ("CatalogSvcC3", "QueueWorkerQ1"),
    ]
    return _build_balanced_env(
        env_id="meta_branching_pyramid_cloud",
        nodes=nodes,
        directed_edges=edges,
        relation_name="routes_traffic_payload_to",
        seed=seed,
    )


def create_meta_cyclic_parity_treaty(seed: int = 102) -> AutonomousEnvironment:
    """Meta-Test Task 3: 8-node parity alliance treaty environment."""
    nodes = [f"NationN{i}" for i in range(8)]
    edges = [
        (nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[3]), (nodes[3], nodes[0]),
        (nodes[4], nodes[5]), (nodes[5], nodes[6]), (nodes[6], nodes[7]), (nodes[7], nodes[4]),
    ]
    return _build_balanced_env(
        env_id="meta_cyclic_parity_treaty",
        nodes=nodes,
        directed_edges=edges,
        relation_name="ratifies_bilateral_treaty_with",
        seed=seed,
    )


def create_meta_modular_network_biomedical(seed: int = 103) -> AutonomousEnvironment:
    """Meta-Test Task 4: 8-node biomedical network with two clusters connected by a bottleneck bridge."""
    nodes = [f"ProteinP{i}" for i in range(8)]
    edges = [
        # Cluster 1: P0, P1, P2
        (nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0]),
        # Bottleneck bridge: P2 -> P3 -> P4
        (nodes[2], nodes[3]), (nodes[3], nodes[4]),
        # Cluster 2: P4, P5, P6, P7
        (nodes[4], nodes[5]), (nodes[5], nodes[6]), (nodes[6], nodes[7]), (nodes[7], nodes[4]),
    ]
    return _build_balanced_env(
        env_id="meta_modular_network_biomedical",
        nodes=nodes,
        directed_edges=edges,
        relation_name="catalyzes_phosphorylation_of",
        seed=seed,
    )
