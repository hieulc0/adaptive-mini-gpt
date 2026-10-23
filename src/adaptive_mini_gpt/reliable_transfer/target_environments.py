"""Multi-world benchmark environments for M56 skill transfer evaluation."""
from __future__ import annotations

import random
from typing import Any
from adaptive_mini_gpt.guided_experiment.environment import (
    AutonomousEnvironment,
    ExperimentQuery,
    GroundTruthGraph,
)


def create_reporting_hierarchy_environment(seed: int = 42) -> AutonomousEnvironment:
    """Corporate Reporting Hierarchy: 6-node Transitive DAG with unfamiliar corporate vocabulary."""
    nodes = [
        "ChiefExecutive", "VicePresidentOps", "DirectorLogistics",
        "WarehouseManager", "InventoryLead", "FloorSupervisor",
    ]
    edges = [
        ("ChiefExecutive", "VicePresidentOps"),
        ("VicePresidentOps", "DirectorLogistics"),
        ("DirectorLogistics", "WarehouseManager"),
        ("WarehouseManager", "InventoryLead"),
        ("InventoryLead", "FloorSupervisor"),
    ]
    gt = GroundTruthGraph(
        nodes=nodes,
        edges=edges,
        rule_family="transitive_reachability",
        domain="corporate_governance",
        seed=seed,
    )
    env = AutonomousEnvironment(
        ground_truth=gt,
        num_development_episodes=6,
        seed=seed,
    )
    _populate_minimal_pairs_and_heldout(env, nodes, edges, "transitive")
    return env


def create_supply_chain_pipeline_environment(seed: int = 42) -> AutonomousEnvironment:
    """Supply Chain Logistics Pipeline: 8-node Transitive Chain with deep path diameter."""
    nodes = [
        "SupplierAlpha", "SmeltingPlant", "ComponentForge", "AssemblyDepot",
        "QualityLab", "DistributionHub", "RetailPartner", "ConsumerPoint",
    ]
    edges = [
        ("SupplierAlpha", "SmeltingPlant"),
        ("SmeltingPlant", "ComponentForge"),
        ("ComponentForge", "AssemblyDepot"),
        ("AssemblyDepot", "QualityLab"),
        ("QualityLab", "DistributionHub"),
        ("DistributionHub", "RetailPartner"),
        ("RetailPartner", "ConsumerPoint"),
    ]
    gt = GroundTruthGraph(
        nodes=nodes,
        edges=edges,
        rule_family="transitive_reachability",
        domain="logistics_supply_chain",
        seed=seed,
    )
    env = AutonomousEnvironment(
        ground_truth=gt,
        num_development_episodes=6,
        seed=seed,
    )
    _populate_minimal_pairs_and_heldout(env, nodes, edges, "transitive")
    return env


def create_metabolic_bypass_environment(seed: int = 42) -> AutonomousEnvironment:
    """Metabolic Bypass: 6-node Diamond Graph with Bypass and blocked paths (partially compatible)."""
    nodes = [
        "GlucoseInput", "KinaseEnzyme", "AldolaseEnzyme",
        "BypassEnzyme", "PyruvateKinase", "ATPOutput",
    ]
    # Diamond with bypass: Glucose -> Kinase -> Aldolase -> Pyruvate -> ATP, plus Glucose -> Bypass -> Pyruvate
    edges = [
        ("GlucoseInput", "KinaseEnzyme"),
        ("KinaseEnzyme", "AldolaseEnzyme"),
        ("AldolaseEnzyme", "PyruvateKinase"),
        ("GlucoseInput", "BypassEnzyme"),
        ("BypassEnzyme", "PyruvateKinase"),
        ("PyruvateKinase", "ATPOutput"),
    ]
    gt = GroundTruthGraph(
        nodes=nodes,
        edges=edges,
        rule_family="diamond_bypass_reachability",
        domain="metabolic_biochemistry",
        seed=seed,
    )
    env = AutonomousEnvironment(
        ground_truth=gt,
        num_development_episodes=6,
        seed=seed,
    )
    _populate_minimal_pairs_and_heldout(env, nodes, edges, "transitive")
    return env


def create_social_coalition_environment(seed: int = 42) -> AutonomousEnvironment:
    """Social Coalition: 8-node Bipartite Partition / Parity graph (deliberately incompatible)."""
    nodes = [
        "FactionAlpha1", "FactionAlpha2", "FactionAlpha3", "FactionAlpha4",
        "FactionBeta1", "FactionBeta2", "FactionBeta3", "FactionBeta4",
    ]
    # Bipartite edges strictly between Alpha and Beta
    edges = [
        ("FactionAlpha1", "FactionBeta1"),
        ("FactionBeta1", "FactionAlpha2"),
        ("FactionAlpha2", "FactionBeta2"),
        ("FactionBeta2", "FactionAlpha3"),
        ("FactionAlpha3", "FactionBeta3"),
        ("FactionBeta3", "FactionAlpha4"),
        ("FactionAlpha4", "FactionBeta4"),
        ("FactionBeta4", "FactionAlpha1"),
    ]
    gt = GroundTruthGraph(
        nodes=nodes,
        edges=edges,
        rule_family="bipartite_parity_coalition",
        domain="social_dynamics",
        seed=seed,
    )
    env = AutonomousEnvironment(
        ground_truth=gt,
        num_development_episodes=6,
        seed=seed,
    )
    _populate_minimal_pairs_and_heldout(env, nodes, edges, "bipartite")
    return env


def _populate_minimal_pairs_and_heldout(
    env: AutonomousEnvironment,
    nodes: list[str],
    edges: list[tuple[str, str]],
    rule_type: str,
) -> None:
    """Generate balanced held-out test queries and structural minimal pairs."""
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for u, v in edges:
        adj[u].add(v)

    # Compute reachable pairs using BFS
    reachable: set[tuple[str, str]] = set()
    for start in nodes:
        visited = set()
        queue = [start]
        while queue:
            curr = queue.pop(0)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    reachable.add((start, neighbor))
                    queue.append(neighbor)

    test_queries: list[ExperimentQuery] = []
    minimal_pairs: list[tuple[ExperimentQuery, ExperimentQuery]] = []

    # 1. Connected pairs (YES)
    for u, v in list(reachable)[:8]:
        q = ExperimentQuery(
            query_id=f"q_{u}_{v}",
            source_entity=u,
            target_entity=v,
            question_text=f"Is {u} connected to {v}?",
            expected_answer="YES",
        )
        test_queries.append(q)

    # 2. Reverse directional queries (NO or UNKNOWN)
    for u, v in list(reachable)[:8]:
        expected = "YES" if (v, u) in reachable else "NO"
        q_rev = ExperimentQuery(
            query_id=f"q_rev_{v}_{u}",
            source_entity=v,
            target_entity=u,
            question_text=f"Is {v} connected to {u}?",
            expected_answer=expected,
        )
        test_queries.append(q_rev)

        # Build minimal contrastive pair
        q_orig = ExperimentQuery(
            query_id=f"q_fwd_{u}_{v}",
            source_entity=u,
            target_entity=v,
            question_text=f"Is {u} connected to {v}?",
            expected_answer="YES",
        )
        minimal_pairs.append((q_orig, q_rev))

    # 3. Disconnected / Unknown queries
    for u in nodes:
        for v in nodes:
            if u != v and (u, v) not in reachable and (v, u) not in reachable:
                q_unk = ExperimentQuery(
                    query_id=f"q_unk_{u}_{v}",
                    source_entity=u,
                    target_entity=v,
                    question_text=f"Is {u} connected to {v}?",
                    expected_answer="UNKNOWN",
                )
                test_queries.append(q_unk)
                if len(test_queries) >= 24:
                    break
        if len(test_queries) >= 24:
            break

    env.held_out_test_queries = test_queries
    env.held_out_minimal_pairs = minimal_pairs
