"""Phase 4 & 7: Multi-Domain Target and Dense-Clique Diagnostic Environments."""
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
from adaptive_mini_gpt.skill_reuse.environment import (
    CustomCallableHypothesis,
    OUT_OF_HYPOTHESIS_BIPARTITE,
    OUT_OF_HYPOTHESIS_DENSE_CLIQUE,
)


def _build_balanced_env(
    env_id: str,
    domain_name: str,
    relation_name: str,
    entities: list[str],
    edges: list[tuple[str, str]],
    hypothesis: RuleHypothesis,
    blocked_nodes: set[str] | None = None,
    negated_edges: set[tuple[str, str]] | None = None,
    entity_levels: dict[str, int] | None = None,
    num_test_per_class: int = 8,
) -> AutonomousEnvironment:
    """Build an AutonomousEnvironment with strictly balanced held-out test queries and minimal pairs."""
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

    yes_q = [
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
    no_q = [
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
    unk_q = [
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

    while len(yes_q) < num_test_per_class:
        idx = len(yes_q)
        yes_q.append(
            EnvironmentQuery(
                query_id=f"q_syn_yes_{idx}",
                source=entities[idx % len(entities)],
                target=entities[(idx + 1) % len(entities)],
                query_type="synthetic_yes",
                question_text=f"Does {entities[idx % len(entities)]} {relation_name} {entities[(idx + 1) % len(entities)]}?",
                expected_answer="YES",
            )
        )
    while len(no_q) < num_test_per_class:
        idx = len(no_q)
        no_q.append(
            EnvironmentQuery(
                query_id=f"q_syn_no_{idx}",
                source=entities[idx % len(entities)],
                target=entities[(idx + 2) % len(entities)],
                query_type="synthetic_no",
                question_text=f"Does {entities[idx % len(entities)]} {relation_name} {entities[(idx + 2) % len(entities)]}?",
                expected_answer="NO",
            )
        )
    while len(unk_q) < num_test_per_class:
        idx = len(unk_q)
        unk_q.append(
            EnvironmentQuery(
                query_id=f"q_syn_unk_{idx}",
                source=entities[idx % len(entities)],
                target=entities[(idx + 3) % len(entities)],
                query_type="synthetic_unk",
                question_text=f"Does {entities[idx % len(entities)]} {relation_name} {entities[(idx + 3) % len(entities)]}?",
                expected_answer="UNKNOWN",
            )
        )

    held_out = yes_q[:num_test_per_class] + no_q[:num_test_per_class] + unk_q[:num_test_per_class]
    rng = random.Random(42)
    rng.shuffle(held_out)

    minimal_pairs: list[MinimalPairControl] = []
    pair_count = min(len(yes_q), len(no_q))
    for i in range(min(pair_count, 8)):
        pair = MinimalPairControl(
            pair_id=f"pair_{env_id}_{i}",
            base_query=yes_q[i],
            perturbed_query=no_q[i],
            perturbation_type="structural_edge_contrast",
        )
        minimal_pairs.append(pair)

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
        minimal_pairs=minimal_pairs,
    )
    return env


@dataclass
class CompositionBenchmarkEnvironments:
    """Suite of source, multi-world target transfer, and dense-clique diagnostic environments."""
    source_skills: dict[str, AutonomousEnvironment]
    target_tasks: dict[str, AutonomousEnvironment]
    dense_clique_diagnostics: dict[str, AutonomousEnvironment]


def build_m55_environment_suite() -> CompositionBenchmarkEnvironments:
    """Construct multi-world target environments and dense-clique diagnostics."""
    # 1. Source Skills (Pretrained Repertoire)
    env_source_transitive = _build_balanced_env(
        env_id="source_transitive",
        domain_name="GenomicsRegulation",
        relation_name="activates_transcription_of",
        entities=["GeneLocusAlpha", "PromoterSiteBeta", "EnhancerUnitGamma", "TranscriptionFactorDelta", "RNAIntermediateEpsilon", "RibosomeTargetZeta"],
        edges=[("GeneLocusAlpha", "PromoterSiteBeta"), ("PromoterSiteBeta", "EnhancerUnitGamma"), ("EnhancerUnitGamma", "TranscriptionFactorDelta"), ("TranscriptionFactorDelta", "RNAIntermediateEpsilon")],
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
    )
    env_source_bridge = _build_balanced_env(
        env_id="source_bridge",
        domain_name="QuantumRelay",
        relation_name="routes_quantum_information_to",
        entities=["QuantumSourceQ0", "RepeaterCouplerQ1", "BridgeDecoherenceNode", "RepeaterCouplerQ2", "QuantumSinkQ3", "BypassAuxQ4"],
        edges=[("QuantumSourceQ0", "RepeaterCouplerQ1"), ("RepeaterCouplerQ1", "BridgeDecoherenceNode"), ("BridgeDecoherenceNode", "RepeaterCouplerQ2"), ("RepeaterCouplerQ2", "QuantumSinkQ3"), ("RepeaterCouplerQ1", "BypassAuxQ4"), ("BypassAuxQ4", "RepeaterCouplerQ2")],
        hypothesis=EXPANDED_HYPOTHESES["BLOCKED_BRIDGE"],
        blocked_nodes={"BridgeDecoherenceNode"},
    )
    env_source_symmetric = _build_balanced_env(
        env_id="source_symmetric",
        domain_name="MeshSecurity",
        relation_name="in_same_security_zone",
        entities=["MeshRegistry", "GatewayCluster", "PaymentServiceNode", "AuthServicePod", "AnalyticsNode", "LoggingDaemon"],
        edges=[("MeshRegistry", "GatewayCluster"), ("GatewayCluster", "MeshRegistry"), ("GatewayCluster", "PaymentServiceNode"), ("PaymentServiceNode", "GatewayCluster"), ("PaymentServiceNode", "AuthServicePod"), ("AuthServicePod", "PaymentServiceNode")],
        hypothesis=EXPANDED_HYPOTHESES["SYM_TRANS"],
    )
    env_source_hierarchy = _build_balanced_env(
        env_id="source_hierarchy",
        domain_name="Taxonomy",
        relation_name="taxonomically_subsumes",
        entities=["DomainArchaea", "PhylumMethano", "ClassHalobacteria", "OrderThermococci", "FamilyArchae", "GenusPyro"],
        edges=[("DomainArchaea", "PhylumMethano"), ("PhylumMethano", "ClassHalobacteria"), ("ClassHalobacteria", "OrderThermococci"), ("OrderThermococci", "FamilyArchae"), ("FamilyArchae", "GenusPyro")],
        hypothesis=EXPANDED_HYPOTHESES["HIERARCHICAL_BOUNDARY"],
        entity_levels={"DomainArchaea": 0, "PhylumMethano": 1, "ClassHalobacteria": 2, "OrderThermococci": 3, "FamilyArchae": 4, "GenusPyro": 5},
    )

    source_skills = {
        "transitive_reachability": env_source_transitive,
        "blocked_bridge": env_source_bridge,
        "symmetric_equivalence": env_source_symmetric,
        "hierarchical_boundary": env_source_hierarchy,
    }

    # 2. Target Tasks
    # Target 1: Corporate Reporting Hierarchy (Transitive, 6 nodes)
    env_target_reporting = _build_balanced_env(
        env_id="target_reporting_hierarchy",
        domain_name="CorporateGovernance",
        relation_name="reports_transitively_to",
        entities=["JuniorAnalystJ1", "SeniorAssociateS2", "TeamLeadT3", "EngineeringDirectorD4", "VicePresidentV5", "ChiefExecutiveC6"],
        edges=[("JuniorAnalystJ1", "SeniorAssociateS2"), ("SeniorAssociateS2", "TeamLeadT3"), ("TeamLeadT3", "EngineeringDirectorD4"), ("EngineeringDirectorD4", "VicePresidentV5"), ("VicePresidentV5", "ChiefExecutiveC6")],
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        entity_levels={"JuniorAnalystJ1": 1, "SeniorAssociateS2": 2, "TeamLeadT3": 3, "EngineeringDirectorD4": 4, "VicePresidentV5": 5, "ChiefExecutiveC6": 6},
    )

    # Target 2: Multi-Tier Supply Chain (Transitive, larger 8-node DAG)
    env_target_supply = _build_balanced_env(
        env_id="target_supply_chain",
        domain_name="GlobalLogistics",
        relation_name="ships_freight_downstream_to",
        entities=["SupplierTier1", "PortTerminalA2", "CustomsStationB3", "RegionalHubC4", "LocalDepotD5", "DistributionCenterE6", "RetailOutletF7", "CustomerEndpointG8"],
        edges=[("SupplierTier1", "PortTerminalA2"), ("PortTerminalA2", "CustomsStationB3"), ("CustomsStationB3", "RegionalHubC4"), ("RegionalHubC4", "LocalDepotD5"), ("LocalDepotD5", "DistributionCenterE6"), ("DistributionCenterE6", "RetailOutletF7"), ("RetailOutletF7", "CustomerEndpointG8")],
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
    )

    # Target 3: Metabolic Bypass Cascade (Partially compatible with blocked bridge)
    env_target_metabolic = _build_balanced_env(
        env_id="target_metabolic_bypass",
        domain_name="MetabolicBiochemistry",
        relation_name="catalyzes_synthesis_of",
        entities=["PrecursorSubstrateAlpha", "IntermediateEnzymeBeta", "InhibitedChokePointGamma", "KinaseFactorDelta", "DownstreamMetaboliteEpsilon", "AlternativeShuntOmega"],
        edges=[("PrecursorSubstrateAlpha", "IntermediateEnzymeBeta"), ("IntermediateEnzymeBeta", "InhibitedChokePointGamma"), ("InhibitedChokePointGamma", "KinaseFactorDelta"), ("KinaseFactorDelta", "DownstreamMetaboliteEpsilon"), ("IntermediateEnzymeBeta", "AlternativeShuntOmega"), ("AlternativeShuntOmega", "KinaseFactorDelta")],
        hypothesis=EXPANDED_HYPOTHESES["BLOCKED_BRIDGE"],
        blocked_nodes={"InhibitedChokePointGamma"},
    )

    # Target 4: Social Coalition (Symmetric Cluster, incompatible with transitive)
    env_target_coalition = _build_balanced_env(
        env_id="target_social_coalition",
        domain_name="GeopoliticalSecurity",
        relation_name="holds_mutual_pact_with",
        entities=["SovereignStateNorth", "AlliancePartnerEast", "FederationSouth", "TreatyStateWest", "NeutralObservatory", "IsolatedFaction"],
        edges=[("SovereignStateNorth", "AlliancePartnerEast"), ("AlliancePartnerEast", "SovereignStateNorth"), ("AlliancePartnerEast", "FederationSouth"), ("FederationSouth", "AlliancePartnerEast"), ("FederationSouth", "TreatyStateWest"), ("TreatyStateWest", "FederationSouth")],
        hypothesis=EXPANDED_HYPOTHESES["SYM_TRANS"],
    )

    # Target 5: Alternating Industrial Dependency (Incompatible negative control)
    env_target_alternating = _build_balanced_env(
        env_id="target_alternating_dependency",
        domain_name="IndustrialWorkflow",
        relation_name="directly_feeds_stage",
        entities=["FeedstockStage0", "RefiningStage1", "SynthesisStage2", "PackagingStage3", "InspectionStage4", "DispatchStage5"],
        edges=[("FeedstockStage0", "RefiningStage1"), ("RefiningStage1", "SynthesisStage2"), ("SynthesisStage2", "PackagingStage3"), ("PackagingStage3", "InspectionStage4"), ("InspectionStage4", "DispatchStage5")],
        hypothesis=OUT_OF_HYPOTHESIS_BIPARTITE,
        entity_levels={"FeedstockStage0": 0, "RefiningStage1": 1, "SynthesisStage2": 2, "PackagingStage3": 3, "InspectionStage4": 4, "DispatchStage5": 5},
    )

    target_tasks = {
        "reporting_hierarchy": env_target_reporting,
        "supply_chain_pipeline": env_target_supply,
        "metabolic_bypass": env_target_metabolic,
        "social_coalition": env_target_coalition,
        "alternating_dependency": env_target_alternating,
    }

    # 3. Dense-Clique Diagnostic Environments (Phase 7)
    env_dense_clique = _build_balanced_env(
        env_id="diagnostic_dense_clique_exceptions",
        domain_name="ClusterTopology",
        relation_name="strongly_interlinks_with",
        entities=["CoreMember1", "CoreMember2", "CoreMember3", "PeripheryNode1", "PeripheryNode2", "PeripheryNode3"],
        edges=[("CoreMember1", "CoreMember2"), ("CoreMember2", "CoreMember3"), ("CoreMember3", "CoreMember1")],
        hypothesis=OUT_OF_HYPOTHESIS_DENSE_CLIQUE,
        entity_levels={"CoreMember1": 2, "CoreMember2": 2, "CoreMember3": 2, "PeripheryNode1": 0, "PeripheryNode2": 0, "PeripheryNode3": 0},
    )

    env_noisy_dense = _build_balanced_env(
        env_id="diagnostic_noisy_dense_known",
        domain_name="DenseKnownMesh",
        relation_name="in_same_security_zone",
        entities=["DenseNode1", "DenseNode2", "DenseNode3", "DenseNode4", "DenseNode5", "DenseNode6"],
        edges=[("DenseNode1", "DenseNode2"), ("DenseNode2", "DenseNode3"), ("DenseNode3", "DenseNode1"), ("DenseNode3", "DenseNode4"), ("DenseNode4", "DenseNode5"), ("DenseNode5", "DenseNode6")],
        hypothesis=EXPANDED_HYPOTHESES["SYM_TRANS"],
    )

    dense_clique_diagnostics = {
        "dense_clique_with_exceptions": env_dense_clique,
        "noisy_dense_known": env_noisy_dense,
    }

    return CompositionBenchmarkEnvironments(
        source_skills=source_skills,
        target_tasks=target_tasks,
        dense_clique_diagnostics=dense_clique_diagnostics,
    )
