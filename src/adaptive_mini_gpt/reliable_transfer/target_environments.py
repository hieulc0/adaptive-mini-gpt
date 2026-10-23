"""Multi-world benchmark environments for M56 skill transfer evaluation."""
from __future__ import annotations

import copy
import random
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
from adaptive_mini_gpt.skill_reuse.environment import (
    CustomCallableHypothesis,
    OUT_OF_HYPOTHESIS_BIPARTITE,
    OUT_OF_HYPOTHESIS_DENSE_CLIQUE,
)
from adaptive_mini_gpt.skill_composition.environment import _build_balanced_env


def create_reporting_hierarchy_environment(seed: int = 42) -> AutonomousEnvironment:
    """Target 1: Corporate Reporting Hierarchy (Transitive, 6 nodes)."""
    return _build_balanced_env(
        env_id="target_reporting_hierarchy",
        domain_name="CorporateGovernance",
        relation_name="reports_transitively_to",
        entities=[
            "JuniorAnalystJ1", "SeniorAssociateS2", "TeamLeadT3",
            "EngineeringDirectorD4", "VicePresidentV5", "ChiefExecutiveC6",
        ],
        edges=[
            ("JuniorAnalystJ1", "SeniorAssociateS2"),
            ("SeniorAssociateS2", "TeamLeadT3"),
            ("TeamLeadT3", "EngineeringDirectorD4"),
            ("EngineeringDirectorD4", "VicePresidentV5"),
            ("VicePresidentV5", "ChiefExecutiveC6"),
        ],
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
        entity_levels={
            "JuniorAnalystJ1": 1, "SeniorAssociateS2": 2, "TeamLeadT3": 3,
            "EngineeringDirectorD4": 4, "VicePresidentV5": 5, "ChiefExecutiveC6": 6,
        },
    )


def create_supply_chain_pipeline_environment(seed: int = 42) -> AutonomousEnvironment:
    """Target 2: Multi-Tier Supply Chain (Transitive, 8-node DAG with deep diameter)."""
    return _build_balanced_env(
        env_id="target_supply_chain",
        domain_name="GlobalLogistics",
        relation_name="ships_freight_downstream_to",
        entities=[
            "SupplierTier1", "PortTerminalA2", "CustomsStationB3", "RegionalHubC4",
            "LocalDepotD5", "DistributionCenterE6", "RetailOutletF7", "CustomerEndpointG8",
        ],
        edges=[
            ("SupplierTier1", "PortTerminalA2"),
            ("PortTerminalA2", "CustomsStationB3"),
            ("CustomsStationB3", "RegionalHubC4"),
            ("RegionalHubC4", "LocalDepotD5"),
            ("LocalDepotD5", "DistributionCenterE6"),
            ("DistributionCenterE6", "RetailOutletF7"),
            ("RetailOutletF7", "CustomerEndpointG8"),
        ],
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
    )


def create_metabolic_bypass_environment(seed: int = 42) -> AutonomousEnvironment:
    """Target 3: Metabolic Pathway with Bypass (Partially Compatible Diamond)."""
    return _build_balanced_env(
        env_id="target_metabolic_bypass",
        domain_name="BiochemicalMetabolism",
        relation_name="catalyzes_synthesis_of",
        entities=[
            "PrecursorSubstrateAlpha", "IntermediateEnzymeBeta", "InhibitedChokePointGamma",
            "KinaseFactorDelta", "DownstreamMetaboliteEpsilon", "AlternativeShuntOmega",
        ],
        edges=[
            ("PrecursorSubstrateAlpha", "IntermediateEnzymeBeta"),
            ("IntermediateEnzymeBeta", "InhibitedChokePointGamma"),
            ("InhibitedChokePointGamma", "KinaseFactorDelta"),
            ("KinaseFactorDelta", "DownstreamMetaboliteEpsilon"),
            ("IntermediateEnzymeBeta", "AlternativeShuntOmega"),
            ("AlternativeShuntOmega", "KinaseFactorDelta"),
        ],
        hypothesis=EXPANDED_HYPOTHESES["BLOCKED_BRIDGE"],
        blocked_nodes={"InhibitedChokePointGamma"},
    )


def create_social_coalition_environment(seed: int = 42) -> AutonomousEnvironment:
    """Target 4: Incompatible Rule: Parity Cycle Alternating Coalition."""
    return _build_balanced_env(
        env_id="target_social_coalition",
        domain_name="GeopoliticalCoalition",
        relation_name="holds_mutual_pact_with",
        entities=[
            "SovereignStateNorth", "AlliancePartnerEast", "FederationSouth",
            "TreatyStateWest", "NeutralObservatory", "IsolatedFaction",
        ],
        edges=[
            ("SovereignStateNorth", "AlliancePartnerEast"),
            ("AlliancePartnerEast", "FederationSouth"),
            ("FederationSouth", "TreatyStateWest"),
            ("TreatyStateWest", "NeutralObservatory"),
            ("NeutralObservatory", "IsolatedFaction"),
            ("IsolatedFaction", "SovereignStateNorth"),
        ],
        hypothesis=OUT_OF_HYPOTHESIS_PARITY,
    )


def create_dense_clique_diagnostic_environment(seed: int = 42) -> AutonomousEnvironment:
    """Diagnostic environment: dense clique with peripheral exceptions (out of hypothesis)."""
    return _build_balanced_env(
        env_id=f"diagnostic_dense_clique_exceptions_{seed}",
        domain_name="ClusterTopology",
        relation_name="strongly_interlinks_with",
        entities=[
            "CoreMember1", "CoreMember2", "CoreMember3",
            "PeripheryNode1", "PeripheryNode2", "PeripheryNode3",
        ],
        edges=[
            ("CoreMember1", "CoreMember2"),
            ("CoreMember2", "CoreMember3"),
            ("CoreMember3", "CoreMember1"),
        ],
        hypothesis=OUT_OF_HYPOTHESIS_DENSE_CLIQUE,
        entity_levels={
            "CoreMember1": 2, "CoreMember2": 2, "CoreMember3": 2,
            "PeripheryNode1": 0, "PeripheryNode2": 0, "PeripheryNode3": 0,
        },
    )


def create_noisy_dense_control_environment(seed: int = 42) -> AutonomousEnvironment:
    """Diagnostic environment: dense known symmetric cluster with noise."""
    return _build_balanced_env(
        env_id=f"diagnostic_noisy_dense_known_{seed}",
        domain_name="DenseKnownMesh",
        relation_name="in_same_security_zone",
        entities=[
            "DenseNode1", "DenseNode2", "DenseNode3",
            "DenseNode4", "DenseNode5", "DenseNode6",
        ],
        edges=[
            ("DenseNode1", "DenseNode2"),
            ("DenseNode2", "DenseNode3"),
            ("DenseNode3", "DenseNode1"),
            ("DenseNode3", "DenseNode4"),
            ("DenseNode4", "DenseNode5"),
            ("DenseNode5", "DenseNode6"),
        ],
        hypothesis=EXPANDED_HYPOTHESES["SYM_TRANS"],
    )
