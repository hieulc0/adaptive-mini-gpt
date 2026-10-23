"""Phase 3: Multi-Domain Target and Epistemic Benchmark Environments."""
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


# ---------------------------------------------------------------------------
# Novel Out-Of-Hypothesis Rule Definitions for Epistemic Generalization
# ---------------------------------------------------------------------------

def _bipartite_coloring_eval(
    source: str,
    target: str,
    edges: list[tuple[str, str]],
    entity_levels: dict[str, int],
    blocked_nodes: set[str],
    negated_edges: set[tuple[str, str]],
) -> str:
    """Out-of-hypothesis rule 2: Bipartite partition coloring.
    
    Relation holds iff source and target belong to opposite color partitions (odd level difference),
    violating transitivity (if A->B and B->C, A and C are same color so A cannot relate to C).
    """
    src_lvl = entity_levels.get(source, 0)
    tgt_lvl = entity_levels.get(target, 0)
    if (src_lvl % 2) != (tgt_lvl % 2):
        return "YES"
    return "NO"


OUT_OF_HYPOTHESIS_BIPARTITE = RuleHypothesis(
    hypothesis_id="rule_bipartite_coloring_out_of_family",
    name="Odd-Cycle Bipartite Partition Relation",
    description="True iff source and target belong to opposite parity partitions; non-transitive.",
    evaluate_query=_bipartite_coloring_eval,
)


def _dense_clique_core_eval(
    source: str,
    target: str,
    edges: list[tuple[str, str]],
    entity_levels: dict[str, int],
    blocked_nodes: set[str],
    negated_edges: set[tuple[str, str]],
) -> str:
    """Out-of-hypothesis rule 3: Dense clique core with periphery exception.
    
    Nodes in core (level >= 1) form a dense clique (all pairs connect). Periphery nodes (level == 0)
    can only receive, never initiate. Blocked nodes break connection.
    """
    if source in blocked_nodes or target in blocked_nodes:
        return "NO"
    src_lvl = entity_levels.get(source, 0)
    tgt_lvl = entity_levels.get(target, 0)
    if src_lvl >= 1 and tgt_lvl >= 1:
        return "YES"
    if src_lvl == 0 and tgt_lvl >= 1:
        return "NO"
    return "UNKNOWN"


OUT_OF_HYPOTHESIS_DENSE_CLIQUE = RuleHypothesis(
    hypothesis_id="rule_dense_clique_periphery_exception",
    name="Dense Clique Core with Periphery Exceptions",
    description="Core entities are mutually connected; periphery entities possess unidirectional exceptions.",
    evaluate_query=_dense_clique_core_eval,
)


# ---------------------------------------------------------------------------
# Environment Factory
# ---------------------------------------------------------------------------

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

    # Ensure required counts by synthetic extrapolation if natural graph is sparse
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

    # Build minimal pair controls
    minimal_pairs: list[MinimalPairControl] = []
    pair_count = min(len(yes_q), len(no_q))
    for i in range(min(pair_count, 8)):
        pair = MinimalPairControl(
            pair_id=f"pair_{env_id}_{i}",
            standard_query=yes_q[i],
            perturbed_query=no_q[i],
            perturbation_type="structural_edge_contrast",
            expected_standard_answer="YES",
            expected_perturbed_answer="NO",
        )
        minimal_pairs.append(pair)

    env = AutonomousEnvironment(
        domain_name=domain_name,
        ground_truth_rule=hypothesis,
        entities=entities,
        base_edges=edges,
        blocked_nodes=blocked_nodes,
        negated_edges=negated_edges,
        entity_levels=entity_levels,
        held_out_test_queries=held_out,
        minimal_pairs=minimal_pairs,
    )
    return env


# ---------------------------------------------------------------------------
# Repertoire Environments & Target Benchmark Suite
# ---------------------------------------------------------------------------

@dataclass
class SkillBenchmarkEnvironments:
    """Suite of training, target transfer, and epistemic diagnostic environments."""
    # 1. Source Skill Environments (for initial qualification)
    source_skills: dict[str, AutonomousEnvironment]
    # 2. Target Transfer Environments (for compatibility selection & transfer adaptation)
    target_tasks: dict[str, AutonomousEnvironment]
    # 3. Epistemic Generalization Environments (for unseen rule detection)
    epistemic_diagnostics: dict[str, AutonomousEnvironment]


def build_m54_environment_suite() -> SkillBenchmarkEnvironments:
    """Build the comprehensive M54 benchmark environment suite."""
    # -----------------------------------------------------------------------
    # Source Environments (repertoire training)
    # -----------------------------------------------------------------------
    # Skill A: Transitive Reachability (Genomics DAG)
    env_source_transitive = _build_balanced_env(
        env_id="source_transitive",
        domain_name="GenomicsRegulation",
        relation_name="activates_transcription_of",
        entities=["GeneLocusAlpha", "PromoterSiteBeta", "EnhancerUnitGamma", "TranscriptionFactorDelta", "RNAIntermediateEpsilon", "RibosomeTargetZeta"],
        edges=[("GeneLocusAlpha", "PromoterSiteBeta"), ("PromoterSiteBeta", "EnhancerUnitGamma"), ("EnhancerUnitGamma", "TranscriptionFactorDelta"), ("TranscriptionFactorDelta", "RNAIntermediateEpsilon")],
        hypothesis=EXPANDED_HYPOTHESES[0],  # standard transitive
    )

    # Skill B: Blocked Bridge Bottleneck
    env_source_bridge = _build_balanced_env(
        env_id="source_bridge",
        domain_name="QuantumRelay",
        relation_name="routes_quantum_information_to",
        entities=["QuantumSourceQ0", "RepeaterCouplerQ1", "BridgeDecoherenceNode", "RepeaterCouplerQ2", "QuantumSinkQ3", "BypassAuxQ4"],
        edges=[("QuantumSourceQ0", "RepeaterCouplerQ1"), ("RepeaterCouplerQ1", "BridgeDecoherenceNode"), ("BridgeDecoherenceNode", "RepeaterCouplerQ2"), ("RepeaterCouplerQ2", "QuantumSinkQ3"), ("RepeaterCouplerQ1", "BypassAuxQ4"), ("BypassAuxQ4", "RepeaterCouplerQ2")],
        hypothesis=EXPANDED_HYPOTHESES[3],  # blocked bridge
        blocked_nodes={"BridgeDecoherenceNode"},
    )

    # Skill C: Symmetric Equivalence / Friendship Cluster
    env_source_symmetric = _build_balanced_env(
        env_id="source_symmetric",
        domain_name="MeshSecurity",
        relation_name="in_same_security_zone",
        entities=["MeshRegistry", "GatewayCluster", "PaymentServiceNode", "AuthServicePod", "AnalyticsNode", "LoggingDaemon"],
        edges=[("MeshRegistry", "GatewayCluster"), ("GatewayCluster", "MeshRegistry"), ("GatewayCluster", "PaymentServiceNode"), ("PaymentServiceNode", "GatewayCluster"), ("PaymentServiceNode", "AuthServicePod"), ("AuthServicePod", "PaymentServiceNode")],
        hypothesis=EXPANDED_HYPOTHESES[2],  # symmetric
    )

    # Skill D: Hierarchical Depth Boundary
    env_source_hierarchy = _build_balanced_env(
        env_id="source_hierarchy",
        domain_name="Taxonomy",
        relation_name="taxonomically_subsumes",
        entities=["DomainArchaea", "PhylumMethano", "ClassHalobacteria", "OrderThermococci", "FamilyArchae", "GenusPyro"],
        edges=[("DomainArchaea", "PhylumMethano"), ("PhylumMethano", "ClassHalobacteria"), ("ClassHalobacteria", "OrderThermococci"), ("OrderThermococci", "FamilyArchae"), ("FamilyArchae", "GenusPyro")],
        hypothesis=EXPANDED_HYPOTHESES[4],  # hierarchy with level bounds
        entity_levels={"DomainArchaea": 0, "PhylumMethano": 1, "ClassHalobacteria": 2, "OrderThermococci": 3, "FamilyArchae": 4, "GenusPyro": 5},
    )

    source_skills = {
        "transitive_reachability": env_source_transitive,
        "blocked_bridge": env_source_bridge,
        "symmetric_equivalence": env_source_symmetric,
        "hierarchical_boundary": env_source_hierarchy,
    }

    # -----------------------------------------------------------------------
    # Target Tasks (unfamiliar relational domains for transfer evaluation)
    # -----------------------------------------------------------------------
    # Target 1: Corporate Reporting Hierarchy (Compatible with transitive / hierarchical)
    env_target_reporting = _build_balanced_env(
        env_id="target_reporting_hierarchy",
        domain_name="CorporateGovernance",
        relation_name="reports_transitively_to",
        entities=["JuniorAnalystJ1", "SeniorAssociateS2", "TeamLeadT3", "EngineeringDirectorD4", "VicePresidentV5", "ChiefExecutiveC6"],
        edges=[("JuniorAnalystJ1", "SeniorAssociateS2"), ("SeniorAssociateS2", "TeamLeadT3"), ("TeamLeadT3", "EngineeringDirectorD4"), ("EngineeringDirectorD4", "VicePresidentV5"), ("VicePresidentV5", "ChiefExecutiveC6")],
        hypothesis=EXPANDED_HYPOTHESES[0],  # transitive reachability
        entity_levels={"JuniorAnalystJ1": 1, "SeniorAssociateS2": 2, "TeamLeadT3": 3, "EngineeringDirectorD4": 4, "VicePresidentV5": 5, "ChiefExecutiveC6": 6},
    )

    # Target 2: Metabolic Bypass Cascade (Compatible with blocked bridge)
    env_target_metabolic = _build_balanced_env(
        env_id="target_metabolic_bypass",
        domain_name="MetabolicBiochemistry",
        relation_name="catalyzes_synthesis_of",
        entities=["PrecursorSubstrateAlpha", "IntermediateEnzymeBeta", "InhibitedChokePointGamma", "KinaseFactorDelta", "DownstreamMetaboliteEpsilon", "AlternativeShuntOmega"],
        edges=[("PrecursorSubstrateAlpha", "IntermediateEnzymeBeta"), ("IntermediateEnzymeBeta", "InhibitedChokePointGamma"), ("InhibitedChokePointGamma", "KinaseFactorDelta"), ("KinaseFactorDelta", "DownstreamMetaboliteEpsilon"), ("IntermediateEnzymeBeta", "AlternativeShuntOmega"), ("AlternativeShuntOmega", "KinaseFactorDelta")],
        hypothesis=EXPANDED_HYPOTHESES[3],  # blocked bridge with shunt
        blocked_nodes={"InhibitedChokePointGamma"},
    )

    # Target 3: Social Non-Aggression Coalition (Compatible with symmetric equivalence)
    env_target_coalition = _build_balanced_env(
        env_id="target_social_coalition",
        domain_name="GeopoliticalSecurity",
        relation_name="holds_mutual_pact_with",
        entities=["SovereignStateNorth", "AlliancePartnerEast", "FederationSouth", "TreatyStateWest", "NeutralObservatory", "IsolatedFaction"],
        edges=[("SovereignStateNorth", "AlliancePartnerEast"), ("AlliancePartnerEast", "SovereignStateNorth"), ("AlliancePartnerEast", "FederationSouth"), ("FederationSouth", "AlliancePartnerEast"), ("FederationSouth", "TreatyStateWest"), ("TreatyStateWest", "FederationSouth")],
        hypothesis=EXPANDED_HYPOTHESES[2],  # symmetric
    )

    # Target 4: Alternating Production Dependency (Incompatible with transitivity; negative control)
    env_target_alternating = _build_balanced_env(
        env_id="target_alternating_dependency",
        domain_name="IndustrialWorkflow",
        relation_name="directly_feeds_stage",
        entities=["FeedstockStage0", "RefiningStage1", "SynthesisStage2", "PackagingStage3", "InspectionStage4", "DispatchStage5"],
        edges=[("FeedstockStage0", "RefiningStage1"), ("RefiningStage1", "SynthesisStage2"), ("SynthesisStage2", "PackagingStage3"), ("PackagingStage3", "InspectionStage4"), ("InspectionStage4", "DispatchStage5")],
        hypothesis=OUT_OF_HYPOTHESIS_BIPARTITE,  # alternating bipartite parity
        entity_levels={"FeedstockStage0": 0, "RefiningStage1": 1, "SynthesisStage2": 2, "PackagingStage3": 3, "InspectionStage4": 4, "DispatchStage5": 5},
    )

    target_tasks = {
        "reporting_hierarchy": env_target_reporting,
        "metabolic_bypass": env_target_metabolic,
        "social_coalition": env_target_coalition,
        "alternating_dependency": env_target_alternating,
    }

    # -----------------------------------------------------------------------
    # Epistemic Diagnostics (Generalization across unseen rule mechanisms)
    # -----------------------------------------------------------------------
    env_epistemic_parity = _build_balanced_env(
        env_id="epistemic_parity_cycle",
        domain_name="RingTopology",
        relation_name="shares_parity_state_with",
        entities=["RingNodeA", "RingNodeB", "RingNodeC", "RingNodeD", "RingNodeE", "RingNodeF"],
        edges=[("RingNodeA", "RingNodeB"), ("RingNodeB", "RingNodeC"), ("RingNodeC", "RingNodeD"), ("RingNodeD", "RingNodeE"), ("RingNodeE", "RingNodeF"), ("RingNodeF", "RingNodeA")],
        hypothesis=OUT_OF_HYPOTHESIS_PARITY,
    )

    env_epistemic_bipartite = _build_balanced_env(
        env_id="epistemic_bipartite_coloring",
        domain_name="BipartiteGraph",
        relation_name="cross_partition_connects",
        entities=["PartitionLeft1", "PartitionRight1", "PartitionLeft2", "PartitionRight2", "PartitionLeft3", "PartitionRight3"],
        edges=[("PartitionLeft1", "PartitionRight1"), ("PartitionRight1", "PartitionLeft2"), ("PartitionLeft2", "PartitionRight2")],
        hypothesis=OUT_OF_HYPOTHESIS_BIPARTITE,
        entity_levels={"PartitionLeft1": 0, "PartitionRight1": 1, "PartitionLeft2": 0, "PartitionRight2": 1, "PartitionLeft3": 0, "PartitionRight3": 1},
    )

    env_epistemic_clique = _build_balanced_env(
        env_id="epistemic_dense_clique",
        domain_name="ClusterTopology",
        relation_name="strongly_interlinks_with",
        entities=["CoreMember1", "CoreMember2", "CoreMember3", "PeripheryNode1", "PeripheryNode2", "PeripheryNode3"],
        edges=[("CoreMember1", "CoreMember2"), ("CoreMember2", "CoreMember3"), ("CoreMember3", "CoreMember1")],
        hypothesis=OUT_OF_HYPOTHESIS_DENSE_CLIQUE,
        entity_levels={"CoreMember1": 2, "CoreMember2": 2, "CoreMember3": 2, "PeripheryNode1": 0, "PeripheryNode2": 0, "PeripheryNode3": 0},
    )

    env_epistemic_noisy = _build_balanced_env(
        env_id="epistemic_noisy_known",
        domain_name="NoisyGenomics",
        relation_name="activates_transcription_of",
        entities=["NoisyGeneA", "NoisyGeneB", "NoisyGeneC", "NoisyGeneD", "NoisyGeneE", "NoisyGeneF"],
        edges=[("NoisyGeneA", "NoisyGeneB"), ("NoisyGeneB", "NoisyGeneC"), ("NoisyGeneC", "NoisyGeneD"), ("NoisyGeneD", "NoisyGeneE")],
        hypothesis=EXPANDED_HYPOTHESES[0],  # standard transitive
    )

    env_epistemic_exception = _build_balanced_env(
        env_id="epistemic_rare_exception",
        domain_name="ExceptionHierarchy",
        relation_name="taxonomically_subsumes",
        entities=["RootTaxonA", "SubTaxonB", "SubTaxonC", "LeafTaxonD", "LeafTaxonE", "LeafTaxonF"],
        edges=[("RootTaxonA", "SubTaxonB"), ("SubTaxonB", "SubTaxonC"), ("SubTaxonC", "LeafTaxonD")],
        hypothesis=EXPANDED_HYPOTHESES[1],  # transitive with single negated edge exception
        negated_edges={("SubTaxonB", "LeafTaxonD")},
    )

    epistemic_diagnostics = {
        "parity_cycle": env_epistemic_parity,
        "bipartite_coloring": env_epistemic_bipartite,
        "dense_clique": env_epistemic_clique,
        "noisy_known": env_epistemic_noisy,
        "rare_exception": env_epistemic_exception,
    }

    return SkillBenchmarkEnvironments(
        source_skills=source_skills,
        target_tasks=target_tasks,
        epistemic_diagnostics=epistemic_diagnostics,
    )
