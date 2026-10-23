"""Phase 3: Multi-World Relational Benchmark Environment Suite for Causal Audit."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.guided_experiment.environment import (
    AutonomousEnvironment,
    EnvironmentQuery,
    EXPANDED_HYPOTHESES,
    MinimalPairControl,
    OUT_OF_HYPOTHESIS_PARITY,
    RuleHypothesis,
    UNKNOWN_RULE,
)
from adaptive_mini_gpt.learning_driven_experiment.minimal_pairs import MinimalPairEngine


@dataclass
class CausalAuditEnvironmentSuite:
    """Rigidly segregated multi-domain environment suite for M53 causal study."""
    train_environments: dict[str, AutonomousEnvironment]
    dev_environments: dict[str, AutonomousEnvironment]
    sealed_test_environments: dict[str, AutonomousEnvironment]

    def all_environments(self) -> dict[str, AutonomousEnvironment]:
        res: dict[str, AutonomousEnvironment] = {}
        res.update(self.train_environments)
        res.update(self.dev_environments)
        res.update(self.sealed_test_environments)
        return res


def _make_world(
    env_id: str,
    domain_name: str,
    relation_name: str,
    entities: list[str],
    edges: list[tuple[str, str]],
    hypothesis: RuleHypothesis,
    blocked_nodes: set[str] | None = None,
    negated_edges: set[tuple[str, str]] | None = None,
    entity_levels: dict[str, int] | None = None,
) -> AutonomousEnvironment:
    """Helper creating an autonomous relational environment with balanced held-out queries and minimal pairs."""
    blocked_nodes = blocked_nodes or set()
    negated_edges = negated_edges or set()
    entity_levels = entity_levels or {}

    all_pairs: list[tuple[str, str, str]] = []
    for u in entities:
        for v in entities:
            if u != v:
                ans = hypothesis.evaluate_query(
                    source=u,
                    target=v,
                    edges=edges,
                    entity_levels=entity_levels,
                    blocked_nodes=blocked_nodes,
                    negated_edges=negated_edges,
                )
                all_pairs.append((u, v, ans))

    yes_queries = [
        EnvironmentQuery(
            query_id=f"q_{u}_{v}",
            source=u,
            target=v,
            query_type="direct",
            question_text=f"Does {u} {relation_name} {v}?",
            expected_answer=ans,
        )
        for u, v, ans in all_pairs if ans == "YES"
    ]
    no_queries = [
        EnvironmentQuery(
            query_id=f"q_{u}_{v}",
            source=u,
            target=v,
            query_type="direct",
            question_text=f"Does {u} {relation_name} {v}?",
            expected_answer=ans,
        )
        for u, v, ans in all_pairs if ans == "NO"
    ]
    unk_queries = [
        EnvironmentQuery(
            query_id=f"q_{u}_{v}",
            source=u,
            target=v,
            query_type="direct",
            question_text=f"Does {u} {relation_name} {v}?",
            expected_answer=ans,
        )
        for u, v, ans in all_pairs if ans == "UNKNOWN"
    ]

    # Incomplete open-world queries if UNKNOWN is sparse
    if len(unk_queries) < 8:
        for i in range(len(unk_queries), 8):
            ext_u = f"ExtUnknown_{i}"
            ext_v = entities[i % len(entities)]
            unk_queries.append(
                EnvironmentQuery(
                    query_id=f"ext_{i}",
                    source=ext_u,
                    target=ext_v,
                    query_type="open_world",
                    question_text=f"Does ungrounded {ext_u} {relation_name} {ext_v}?",
                    expected_answer="UNKNOWN",
                )
            )

    # Strictly balanced held-out test queries (8 YES, 8 NO, 8 UNKNOWN = 24 queries)
    held_out = yes_queries[:8] + no_queries[:8] + unk_queries[:8]

    env = AutonomousEnvironment(
        env_id=env_id,
        domain_name=domain_name,
        relation_name=relation_name,
        entities=entities,
        base_edges=edges,
        true_hypothesis=hypothesis,
        entity_levels=entity_levels,
        blocked_nodes=blocked_nodes,
        negated_edges=negated_edges,
        held_out_test_queries=held_out,
        minimal_pairs=[],
    )

    # Attach verified minimal pairs
    env.minimal_pairs = MinimalPairEngine.generate_controlled_minimal_pairs(env, max_pairs=8)
    return env


def build_m53_environment_suite() -> CausalAuditEnvironmentSuite:
    """Construct multi-domain environment suite for M53 causal and transfer audit."""
    train_envs: dict[str, AutonomousEnvironment] = {}
    dev_envs: dict[str, AutonomousEnvironment] = {}
    sealed_envs: dict[str, AutonomousEnvironment] = {}

    # 1. Meta-Training Worlds
    # Domain A: Cellular Genomics (Directed Transitive Reachability)
    genom_entities = ["GeneLocusAlpha", "PromoterSiteBeta", "EnhancerUnitGamma", "TranscriptionFactorDelta", "RNAIntermediateEpsilon", "RibosomeTargetZeta"]
    genom_edges = [
        ("GeneLocusAlpha", "PromoterSiteBeta"),
        ("PromoterSiteBeta", "EnhancerUnitGamma"),
        ("EnhancerUnitGamma", "TranscriptionFactorDelta"),
        ("TranscriptionFactorDelta", "RNAIntermediateEpsilon"),
        ("RNAIntermediateEpsilon", "RibosomeTargetZeta"),
    ]
    train_envs["env_train_genomics_dag"] = _make_world(
        env_id="env_train_genomics_dag",
        domain_name="Genomics",
        relation_name="activates_transcription_of",
        entities=genom_entities,
        edges=genom_edges,
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
    )

    # Domain B: Supply Chain Logistics (Symmetric Non-Transitive)
    logistics_entities = ["SupplierAnchor", "CustomsPointA", "CustomsPointB", "ConsortiumRelayX", "ConsortiumRelayY", "LocalWarehouse"]
    logistics_edges = [
        ("SupplierAnchor", "CustomsPointA"),
        ("CustomsPointA", "ConsortiumRelayX"),
        ("CustomsPointB", "ConsortiumRelayY"),
        ("ConsortiumRelayX", "LocalWarehouse"),
    ]
    train_envs["env_train_logistics_lattice"] = _make_world(
        env_id="env_train_logistics_lattice",
        domain_name="SupplyChainLogistics",
        relation_name="treaty_partner_with",
        entities=logistics_entities,
        edges=logistics_edges,
        hypothesis=EXPANDED_HYPOTHESES["SYM_NONTRANS"],
    )

    # Domain C: Microservice Distributed Mesh (Symmetric Transitive)
    mesh_entities = ["MeshRegistry", "GatewayCluster", "PaymentServiceNode", "MetricsCollector", "DatabaseShard1", "DatabaseShard2"]
    mesh_edges = [
        ("MeshRegistry", "GatewayCluster"),
        ("GatewayCluster", "PaymentServiceNode"),
        ("PaymentServiceNode", "DatabaseShard1"),
        ("DatabaseShard1", "DatabaseShard2"),
    ]
    train_envs["env_train_mesh_cluster"] = _make_world(
        env_id="env_train_mesh_cluster",
        domain_name="DistributedMesh",
        relation_name="in_same_security_zone",
        entities=mesh_entities,
        edges=mesh_edges,
        hypothesis=EXPANDED_HYPOTHESES["SYM_TRANS"],
    )

    # 2. Development Worlds (Used for policy reward, fixed adaptation tuning)
    # Domain D: Phylogenetics (Hierarchical Dominance)
    phylo_entities = ["DomainArchaea", "PhylumMethano", "ClassHalobacteria", "OrderArchaeoglobales", "FamilyHalococcaceae", "GenusHalococcus"]
    phylo_edges = [
        ("DomainArchaea", "PhylumMethano"),
        ("PhylumMethano", "ClassHalobacteria"),
        ("ClassHalobacteria", "OrderArchaeoglobales"),
        ("OrderArchaeoglobales", "FamilyHalococcaceae"),
        ("FamilyHalococcaceae", "GenusHalococcus"),
    ]
    phylo_levels = {
        "DomainArchaea": 5, "PhylumMethano": 4, "ClassHalobacteria": 3,
        "OrderArchaeoglobales": 2, "FamilyHalococcaceae": 1, "GenusHalococcus": 0,
    }
    dev_envs["env_dev_phylogenetics_hier"] = _make_world(
        env_id="env_dev_phylogenetics_hier",
        domain_name="Phylogenetics",
        relation_name="taxonomically_subsumes",
        entities=phylo_entities,
        edges=phylo_edges,
        hypothesis=EXPANDED_HYPOTHESES["HIERARCHICAL_BOUNDARY"],
        entity_levels=phylo_levels,
    )

    # Domain E: Diachronic Linguistics (Open World Incomplete)
    ling_entities = ["ProtoRootX", "BranchDialectY", "ArchaicFormZ", "PhoneticShiftW", "ModernReflexV", "IsolatedCognateU"]
    ling_edges = [
        ("ProtoRootX", "BranchDialectY"),
        ("BranchDialectY", "ArchaicFormZ"),
        ("ArchaicFormZ", "PhoneticShiftW"),
    ]
    dev_envs["env_dev_linguistics_open"] = _make_world(
        env_id="env_dev_linguistics_open",
        domain_name="DiachronicLinguistics",
        relation_name="phonologically_evolved_into",
        entities=ling_entities,
        edges=ling_edges,
        hypothesis=EXPANDED_HYPOTHESES["OPEN_WORLD_INCOMPLETE"],
    )

    # 3. Sealed Meta-Test Worlds (Strictly isolated from training and dev)
    # Sealed World 1: Biochemical Metabolic Cascade (Directed Transitive)
    biochem_entities = ["MetaboliteSubstrate0", "EnzymeComplex1", "PhosphorylatedInter2", "CatalyticCofactor3", "EndProductTarget4", "EffluxCarrier5"]
    biochem_edges = [
        ("MetaboliteSubstrate0", "EnzymeComplex1"),
        ("EnzymeComplex1", "PhosphorylatedInter2"),
        ("PhosphorylatedInter2", "CatalyticCofactor3"),
        ("CatalyticCofactor3", "EndProductTarget4"),
        ("EndProductTarget4", "EffluxCarrier5"),
    ]
    sealed_envs["env_sealed_biochem_trans"] = _make_world(
        env_id="env_sealed_biochem_trans",
        domain_name="BiochemicalMetabolism",
        relation_name="catalyzes_synthesis_of",
        entities=biochem_entities,
        edges=biochem_edges,
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
    )

    # Sealed World 2: Quantum Teleportation Routing (Blocked Bridge Bottleneck)
    quantum_entities = ["QuantumSourceQ0", "RepeaterCouplerQ1", "BlockedCavityBusQ2", "AnalyzerGateQ3", "QubitReceiverQ4", "TeleportSinkQ5"]
    quantum_edges = [
        ("QuantumSourceQ0", "RepeaterCouplerQ1"),
        ("RepeaterCouplerQ1", "BlockedCavityBusQ2"),
        ("BlockedCavityBusQ2", "AnalyzerGateQ3"),
        ("AnalyzerGateQ3", "QubitReceiverQ4"),
        ("QubitReceiverQ4", "TeleportSinkQ5"),
    ]
    quantum_blocked = {"BlockedCavityBusQ2"}
    sealed_envs["env_sealed_quantum_bridge"] = _make_world(
        env_id="env_sealed_quantum_bridge",
        domain_name="QuantumTeleportation",
        relation_name="routes_quantum_information_to",
        entities=quantum_entities,
        edges=quantum_edges,
        hypothesis=EXPANDED_HYPOTHESES["BLOCKED_BRIDGE"],
        blocked_nodes=quantum_blocked,
    )

    # Sealed World 3: Social Network Parity Cycle (Truly Out-of-Hypothesis)
    parity_entities = ["RingNodeA", "RingNodeB", "RingNodeC", "RingNodeD", "RingNodeE", "RingNodeF"]
    parity_edges = [
        ("RingNodeA", "RingNodeB"),
        ("RingNodeB", "RingNodeC"),
        ("RingNodeC", "RingNodeD"),
        ("RingNodeD", "RingNodeE"),
        ("RingNodeE", "RingNodeF"),
        ("RingNodeF", "RingNodeA"),
    ]
    sealed_envs["env_sealed_social_parity"] = _make_world(
        env_id="env_sealed_social_parity",
        domain_name="SocialNetworkParity",
        relation_name="shares_parity_state_with",
        entities=parity_entities,
        edges=parity_edges,
        hypothesis=OUT_OF_HYPOTHESIS_PARITY,
    )

    return CausalAuditEnvironmentSuite(
        train_environments=train_envs,
        dev_environments=dev_envs,
        sealed_test_environments=sealed_envs,
    )
