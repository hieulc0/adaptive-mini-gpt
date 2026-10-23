"""Phase 2: Segregated Multi-Environment Suite for Meta-Training, Dev, and Sealed Testing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import copy

from adaptive_mini_gpt.guided_experiment.environment import (
    AutonomousEnvironment,
    EnvironmentQuery,
    EXPANDED_HYPOTHESES,
    MinimalPairItem,
    OUT_OF_HYPOTHESIS_PARITY,
    RuleHypothesis,
    UNKNOWN_RULE,
)
from adaptive_mini_gpt.learning_driven_experiment.minimal_pairs import MinimalPairEngine


@dataclass
class SegregatedEnvironmentSuite:
    """Rigidly segregated suite preventing any evaluation contamination into training."""
    train_environments: dict[str, AutonomousEnvironment]
    dev_environments: dict[str, AutonomousEnvironment]
    sealed_test_environments: dict[str, AutonomousEnvironment]

    def all_environments(self) -> dict[str, AutonomousEnvironment]:
        combined = {}
        combined.update(self.train_environments)
        combined.update(self.dev_environments)
        combined.update(self.sealed_test_environments)
        return combined


def build_m52_environment_suite() -> SegregatedEnvironmentSuite:
    """Construct segregated environment suite enforcing strict separation between training, dev, and sealed tests."""

    # -------------------------------------------------------------
    # 1. META-TRAINING WORLDS (Policy updates allowed here)
    # -------------------------------------------------------------
    train_envs: dict[str, AutonomousEnvironment] = {}

    # World 1: Linguistics Transitive
    ling_entities = ["ProtoIndoEuropean", "ProtoItalic", "ProtoGermanic", "LatinArchaic", "OldNorse", "ModernFrench"]
    ling_edges = [
        ("ProtoIndoEuropean", "ProtoItalic"),
        ("ProtoIndoEuropean", "ProtoGermanic"),
        ("ProtoItalic", "LatinArchaic"),
        ("ProtoGermanic", "OldNorse"),
        ("LatinArchaic", "ModernFrench"),
    ]
    train_envs["env_train_ling_dir_trans"] = _create_environment(
        env_id="env_train_ling_dir_trans",
        domain_name="Linguistics",
        relation_name="derived_from",
        entities=ling_entities,
        edges=ling_edges,
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
    )

    # World 2: Legal Symmetric Non-Transitive
    legal_entities = ["FederalAuthority", "StateTribunalA", "StateTribunalB", "AppellateChamberX", "AppellateChamberY", "MunicipalMagistrate"]
    legal_edges = [
        ("FederalAuthority", "StateTribunalA"),
        ("StateTribunalA", "AppellateChamberX"),
        ("StateTribunalB", "AppellateChamberY"),
        ("AppellateChamberX", "MunicipalMagistrate"),
    ]
    train_envs["env_train_legal_sym_nontrans"] = _create_environment(
        env_id="env_train_legal_sym_nontrans",
        domain_name="Jurisprudence",
        relation_name="treaty_partner_with",
        entities=legal_entities,
        edges=legal_edges,
        hypothesis=EXPANDED_HYPOTHESES["SYM_NONTRANS"],
    )

    # World 3: Software Symmetric Transitive
    soft_entities = ["ServiceRegistry", "AuthGatewayNode", "BillingClusterNode", "TelemetryAgentNode", "DatabaseReplicaA", "DatabaseReplicaB"]
    soft_edges = [
        ("ServiceRegistry", "AuthGatewayNode"),
        ("AuthGatewayNode", "BillingClusterNode"),
        ("TelemetryAgentNode", "DatabaseReplicaA"),
        ("DatabaseReplicaA", "DatabaseReplicaB"),
    ]
    train_envs["env_train_soft_sym_trans"] = _create_environment(
        env_id="env_train_soft_sym_trans",
        domain_name="DistributedSoftware",
        relation_name="in_same_security_zone",
        entities=soft_entities,
        edges=soft_edges,
        hypothesis=EXPANDED_HYPOTHESES["SYM_TRANS"],
    )

    # World 4: Biological Taxonomy Hierarchical
    bio_entities = ["DomainEukarya", "KingdomMetazoa", "PhylumChordata", "ClassMammalia", "OrderPrimates", "SpeciesSapiens"]
    bio_edges = [
        ("DomainEukarya", "KingdomMetazoa"),
        ("KingdomMetazoa", "PhylumChordata"),
        ("PhylumChordata", "ClassMammalia"),
        ("ClassMammalia", "OrderPrimates"),
        ("OrderPrimates", "SpeciesSapiens"),
    ]
    bio_levels = {
        "DomainEukarya": 1,
        "KingdomMetazoa": 2,
        "PhylumChordata": 3,
        "ClassMammalia": 4,
        "OrderPrimates": 5,
        "SpeciesSapiens": 6,
    }
    train_envs["env_train_bio_hierarchical"] = _create_environment(
        env_id="env_train_bio_hierarchical",
        domain_name="BiologicalTaxonomy",
        relation_name="taxonomically_subsumes",
        entities=bio_entities,
        edges=bio_edges,
        hypothesis=EXPANDED_HYPOTHESES["HIERARCHICAL_BOUNDARY"],
        entity_levels=bio_levels,
    )

    # -------------------------------------------------------------
    # 2. DEVELOPMENT WORLDS (Reward computation & hyperparameter tuning ONLY)
    # -------------------------------------------------------------
    dev_envs: dict[str, AutonomousEnvironment] = {}

    # Dev World 1: Biochemistry Explicit Refutation
    chem_entities = ["SubstrateAlpha", "IntermediateBeta", "EnzymeGamma", "InhibitorDelta", "MetaboliteOmega", "CataboliteZeta"]
    chem_edges = [
        ("SubstrateAlpha", "IntermediateBeta"),
        ("IntermediateBeta", "EnzymeGamma"),
        ("EnzymeGamma", "MetaboliteOmega"),
    ]
    chem_negated = {("IntermediateBeta", "InhibitorDelta"), ("InhibitorDelta", "MetaboliteOmega")}
    dev_envs["env_dev_chem_refutation"] = _create_environment(
        env_id="env_dev_chem_refutation",
        domain_name="Biochemistry",
        relation_name="catalyzes_synthesis_of",
        entities=chem_entities,
        edges=chem_edges,
        hypothesis=EXPANDED_HYPOTHESES["EXPLICIT_REFUTATION"],
        negated_edges=chem_negated,
    )

    # Dev World 2: Geopolitics Directional Non-Transitive
    geo_entities = ["BlocAllianceNorth", "MemberStateAlpha", "MemberStateBeta", "DiplomaticMissionX", "ObserverConsulateY", "NeutralTerritory"]
    geo_edges = [
        ("BlocAllianceNorth", "MemberStateAlpha"),
        ("BlocAllianceNorth", "MemberStateBeta"),
        ("MemberStateAlpha", "DiplomaticMissionX"),
        ("MemberStateBeta", "ObserverConsulateY"),
    ]
    dev_envs["env_dev_geo_dir_nontrans"] = _create_environment(
        env_id="env_dev_geo_dir_nontrans",
        domain_name="Geopolitics",
        relation_name="commands_diplomatic_envoy",
        entities=geo_entities,
        edges=geo_edges,
        hypothesis=EXPANDED_HYPOTHESES["DIR_NONTRANS"],
    )

    # -------------------------------------------------------------
    # 3. SEALED META-TEST WORLDS (Strictly frozen evaluation ONLY)
    # -------------------------------------------------------------
    sealed_envs: dict[str, AutonomousEnvironment] = {}

    # Sealed Test 1: Cryptographic Blocked Bridge
    crypto_entities = ["RootAuthorityKey", "IntermediateCertA", "IntermediateCertB", "RevocationList", "LeafSignerX", "ValidationNode"]
    crypto_edges = [
        ("RootAuthorityKey", "IntermediateCertA"),
        ("IntermediateCertA", "LeafSignerX"),
        ("RootAuthorityKey", "IntermediateCertB"),
        ("IntermediateCertB", "RevocationList"),
        ("RevocationList", "ValidationNode"),
    ]
    crypto_blocked = {"IntermediateCertB"}
    sealed_envs["env_sealed_crypto_blocked_bridge"] = _create_environment(
        env_id="env_sealed_crypto_blocked_bridge",
        domain_name="CryptographicPKI",
        relation_name="authorizes_session_for",
        entities=crypto_entities,
        edges=crypto_edges,
        hypothesis=EXPANDED_HYPOTHESES["BLOCKED_BRIDGE"],
        blocked_nodes=crypto_blocked,
    )

    # Sealed Test 2: Quantum Teleportation Open World Incomplete
    quantum_entities = ["EntangledPhotonPairA", "QuantumRepeaterNode1", "QuantumRepeaterNode2", "BellStateDetector", "QubitReceiverX", "ClassicalChannel"]
    quantum_edges = [
        ("EntangledPhotonPairA", "QuantumRepeaterNode1"),
        ("QuantumRepeaterNode1", "QuantumRepeaterNode2"),
        ("QuantumRepeaterNode2", "BellStateDetector"),
    ]
    sealed_envs["env_sealed_quantum_open_world"] = _create_environment(
        env_id="env_sealed_quantum_open_world",
        domain_name="QuantumTeleportation",
        relation_name="maintains_entanglement_with",
        entities=quantum_entities,
        edges=quantum_edges,
        hypothesis=EXPANDED_HYPOTHESES["OPEN_WORLD_INCOMPLETE"],
    )

    # Sealed Test 3: Social Parity Cycle (Out of Candidate Hypotheses)
    social_entities = ["HubLeader1", "RelayMemberA", "RelayMemberB", "BridgeNodeC", "TargetFollowerD", "IsolateObserverE"]
    social_edges = [
        ("HubLeader1", "RelayMemberA"),
        ("RelayMemberA", "RelayMemberB"),
        ("RelayMemberB", "BridgeNodeC"),
        ("BridgeNodeC", "TargetFollowerD"),
    ]
    sealed_envs["env_sealed_social_parity_cycle"] = _create_environment(
        env_id="env_sealed_social_parity_cycle",
        domain_name="SocialNetworkParity",
        relation_name="shares_parity_influence_with",
        entities=social_entities,
        edges=social_edges,
        hypothesis=OUT_OF_HYPOTHESIS_PARITY,
    )

    return SegregatedEnvironmentSuite(
        train_environments=train_envs,
        dev_environments=dev_envs,
        sealed_test_environments=sealed_envs,
    )


def _create_environment(
    env_id: str,
    domain_name: str,
    relation_name: str,
    entities: list[str],
    edges: list[tuple[str, str]],
    hypothesis: RuleHypothesis,
    entity_levels: dict[str, int] | None = None,
    blocked_nodes: set[str] | None = None,
    negated_edges: set[tuple[str, str]] | None = None,
) -> AutonomousEnvironment:
    """Helper creating environment with verified held-out queries and minimal pairs."""
    entity_levels = entity_levels or {}
    blocked_nodes = blocked_nodes or set()
    negated_edges = negated_edges or set()

    # Create all pairwise queries
    all_pairs = []
    for u in entities:
        for v in entities:
            if u != v:
                ans = hypothesis.evaluate_query(u, v, edges, entity_levels, blocked_nodes, negated_edges)
                all_pairs.append((u, v, ans))

    yes_pool = [EnvironmentQuery(u, v, relation_name, ans, f"Does {u} {relation_name} {v}?") for u, v, ans in all_pairs if ans == "YES"]
    no_pool = [EnvironmentQuery(u, v, relation_name, ans, f"Does {u} {relation_name} {v}?") for u, v, ans in all_pairs if ans == "NO"]
    unk_pool = [EnvironmentQuery(u, v, relation_name, ans, f"Does {u} {relation_name} {v}?") for u, v, ans in all_pairs if ans == "UNKNOWN"]

    # Incomplete open-world queries if UNKNOWN is sparse
    if len(unk_pool) < 8:
        for i in range(len(unk_pool), 8):
            ext_u = f"ExtUnknown_{i}"
            ext_v = entities[i % len(entities)]
            unk_pool.append(EnvironmentQuery(
                source=ext_u,
                target=ext_v,
                relation_name=relation_name,
                expected_answer="UNKNOWN",
                question_text=f"Does ungrounded {ext_u} {relation_name} {ext_v}?",
            ))

    # Balanced held-out test queries (8 YES, 8 NO, 8 UNKNOWN = 24 queries)
    held_out = yes_pool[:8] + no_pool[:8] + unk_pool[:8]

    env = AutonomousEnvironment(
        env_id=env_id,
        domain_name=domain_name,
        relation_name=relation_name,
        entities=entities,
        base_edges=edges,
        hypothesis=hypothesis,
        entity_levels=entity_levels,
        blocked_nodes=blocked_nodes,
        negated_edges=negated_edges,
        held_out_test_queries=held_out,
        minimal_pairs=[],
    )

    # Attach verified minimal pairs
    env.minimal_pairs = MinimalPairEngine.generate_controlled_minimal_pairs(env, max_pairs=8)
    return env
