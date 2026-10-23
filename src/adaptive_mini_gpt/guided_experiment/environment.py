"""Multi-Environment Suite with Unseen-Rule Generalization and Out-of-Hypothesis Detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import copy


@dataclass(frozen=True)
class RuleHypothesis:
    """Specification of an abstract relational rule hypothesis."""
    hypothesis_id: str
    name: str
    is_symmetric: bool
    is_transitive: bool
    has_boundary_constraint: bool = False
    has_blocked_bridge: bool = False
    has_explicit_refutation: bool = False
    is_open_world: bool = False
    is_parity_constrained: bool = False
    description: str = ""

    def evaluate_query(
        self,
        source: str,
        target: str,
        edges: list[tuple[str, str]],
        entity_levels: dict[str, int] | None = None,
        blocked_nodes: set[str] | None = None,
        negated_edges: set[tuple[str, str]] | None = None,
    ) -> str:
        """Evaluate expected tri-state answer under this hypothesis."""
        if source == target:
            return "YES"

        # Check ungrounded or external unknown entity
        known_nodes = set()
        for u, v in edges:
            known_nodes.add(u)
            known_nodes.add(v)

        if (
            source.startswith("ExtUnknown")
            or target.startswith("ExtUnknown")
            or source.startswith("DeletedBridge")
            or target.startswith("DeletedBridge")
            or (known_nodes and (source not in known_nodes or target not in known_nodes))
        ):
            return "UNKNOWN"

        edge_set = set(edges)
        if self.is_symmetric:
            sym_edges = set()
            for u, v in edges:
                sym_edges.add((u, v))
                sym_edges.add((v, u))
            edge_set = sym_edges

        # Parity cycle check (out-of-hypothesis rule)
        if self.is_parity_constrained:
            # Breadth-first search tracking path length
            queue = [(source, 0)]
            visited = set()
            while queue:
                curr, length = queue.pop(0)
                if curr == target:
                    return "YES" if (length % 2 == 0) else "NO"
                if (curr, length % 2) in visited:
                    continue
                visited.add((curr, length % 2))
                for u, v in edge_set:
                    if u == curr:
                        queue.append((v, length + 1))
            return "UNKNOWN" if self.is_open_world else "NO"

        # Direct edge check
        if (source, target) in edge_set:
            if self.has_explicit_refutation and negated_edges and (source, target) in negated_edges:
                return "NO"
            return "YES"

        # Explicit refutation check
        if self.has_explicit_refutation and negated_edges and (source, target) in negated_edges:
            return "NO"

        # Non-transitive rule: only direct edges hold
        if not self.is_transitive:
            if self.is_open_world:
                return "UNKNOWN"
            return "NO"

        # Transitive rule: multi-hop reachability check
        visited = set()
        queue = [source]
        found = False

        while queue:
            curr = queue.pop(0)
            if curr == target:
                found = True
                break
            if curr in visited:
                continue
            visited.add(curr)

            # Blocked bridge constraint
            if self.has_blocked_bridge and blocked_nodes and curr in blocked_nodes and curr != source:
                continue

            for u, v in edge_set:
                if u == curr and v not in visited:
                    # Boundary rank constraint
                    if self.has_boundary_constraint and entity_levels is not None:
                        curr_level = entity_levels.get(u, 0)
                        next_level = entity_levels.get(v, 0)
                        if abs(next_level - curr_level) > 1:
                            continue
                    queue.append(v)

        if found:
            return "YES"

        if self.is_open_world:
            return "UNKNOWN"
        return "NO"


# Standard 8-Hypothesis Family plus UNKNOWN_RULE and Out-of-Hypothesis Parity Rule
EXPANDED_HYPOTHESES: dict[str, RuleHypothesis] = {
    "DIR_TRANS": RuleHypothesis(
        "DIR_TRANS", "Directed Transitive", is_symmetric=False, is_transitive=True,
        description="Asymmetric transitive DAG: reachability propagates unidirectionally."
    ),
    "DIR_NONTRANS": RuleHypothesis(
        "DIR_NONTRANS", "Directed Non-Transitive", is_symmetric=False, is_transitive=False,
        description="Asymmetric non-transitive graph: only direct oriented edges hold."
    ),
    "SYM_TRANS": RuleHypothesis(
        "SYM_TRANS", "Symmetric Transitive", is_symmetric=True, is_transitive=True,
        description="Equivalence relation: connected components form fully reachable cliques."
    ),
    "SYM_NONTRANS": RuleHypothesis(
        "SYM_NONTRANS", "Symmetric Non-Transitive", is_symmetric=True, is_transitive=False,
        description="Undirected adjacency graph: reciprocal direct edges hold, but paths do not propagate."
    ),
    "HIERARCHICAL_BOUNDARY": RuleHypothesis(
        "HIERARCHICAL_BOUNDARY", "Hierarchical Boundary Transitive", is_symmetric=False, is_transitive=True,
        has_boundary_constraint=True,
        description="Directional transitive hierarchy: propagation is restricted to adjacent tiers."
    ),
    "BLOCKED_BRIDGE": RuleHypothesis(
        "BLOCKED_BRIDGE", "Blocked Bridge Transitive", is_symmetric=False, is_transitive=True,
        has_blocked_bridge=True,
        description="Transitive reachability holds except across revoked or compromised bridge intermediaries."
    ),
    "EXPLICIT_REFUTATION": RuleHypothesis(
        "EXPLICIT_REFUTATION", "Explicitly Refuted Transitive", is_symmetric=False, is_transitive=True,
        has_explicit_refutation=True,
        description="Transitive propagation with active inhibitor nodes that explicitly refute conclusions."
    ),
    "OPEN_WORLD_INCOMPLETE": RuleHypothesis(
        "OPEN_WORLD_INCOMPLETE", "Open-World Incomplete Transitive", is_symmetric=False, is_transitive=True,
        is_open_world=True,
        description="Open-world relational graph: absent reachability evidence is UNKNOWN rather than refuted."
    ),
}

# Out-of-hypothesis rule family (present in sealed meta-test 3, but absent from candidate hypotheses)
OUT_OF_HYPOTHESIS_PARITY = RuleHypothesis(
    "PARITY_CYCLE", "Parity Cycle Alternating", is_symmetric=False, is_transitive=True,
    is_parity_constrained=True,
    description="Alternating parity relation: reachability holds if and only if path length is even."
)

UNKNOWN_RULE = RuleHypothesis(
    "UNKNOWN_RULE", "Unknown or Inadequate Rule", is_symmetric=False, is_transitive=False,
    description="All candidate hypotheses in H8 are disproven; evidence requires an unmodeled rule."
)


@dataclass
class EnvironmentQuery:
    query_id: str
    source: str
    target: str
    query_type: str
    question_text: str
    expected_answer: str


@dataclass
class MinimalPairControl:
    pair_id: str
    base_query: EnvironmentQuery
    perturbed_query: EnvironmentQuery
    perturbation_type: str


class AutonomousEnvironment:
    """An environment governed by an unfamiliar relational rule."""

    def __init__(
        self,
        env_id: str,
        domain_name: str,
        relation_name: str,
        entities: list[str],
        base_edges: list[tuple[str, str]],
        true_hypothesis: RuleHypothesis,
        held_out_test_queries: list[EnvironmentQuery],
        minimal_pairs: list[MinimalPairControl],
        entity_levels: dict[str, int] | None = None,
        blocked_nodes: set[str] | None = None,
        negated_edges: set[tuple[str, str]] | None = None,
    ) -> None:
        self.env_id = env_id
        self.domain_name = domain_name
        self.relation_name = relation_name
        self.entities = entities
        self.base_edges = base_edges
        self.true_hypothesis = true_hypothesis
        self.held_out_test_queries = held_out_test_queries
        self.minimal_pairs = minimal_pairs
        self.entity_levels = entity_levels or {}
        self.blocked_nodes = blocked_nodes or set()
        self.negated_edges = negated_edges or set()

        self.oracle_call_count: int = 0
        self.query_history: list[tuple[str, str, str]] = []

    def evaluate_ground_truth(self, source: str, target: str) -> str:
        return self.true_hypothesis.evaluate_query(
            source=source,
            target=target,
            edges=self.base_edges,
            entity_levels=self.entity_levels,
            blocked_nodes=self.blocked_nodes,
            negated_edges=self.negated_edges,
        )

    def ask_oracle(self, source: str, target: str) -> str:
        self.oracle_call_count += 1
        ans = self.evaluate_ground_truth(source, target)
        self.query_history.append((source, target, ans))
        return ans

    def oracle_query(self, source: str, target: str) -> str:
        return self.ask_oracle(source, target)

    def reset_oracle_budget(self) -> None:
        self.oracle_call_count = 0
        self.query_history = []


def build_m51_environment_suite() -> dict[str, AutonomousEnvironment]:
    """Construct multi-environment suite spanning training, dev, and 3 held-out sealed meta-test worlds."""
    suite: dict[str, AutonomousEnvironment] = {}

    # 1. Training Environment: Linguistics (DIR_TRANS)
    ling_entities = ["ProtoIndoEuropean", "ProtoItalic", "ProtoGermanic", "LatinArchaic", "OldNorse", "ModernFrench"]
    ling_edges = [
        ("ProtoIndoEuropean", "ProtoItalic"),
        ("ProtoIndoEuropean", "ProtoGermanic"),
        ("ProtoItalic", "LatinArchaic"),
        ("ProtoGermanic", "OldNorse"),
        ("LatinArchaic", "ModernFrench"),
    ]
    suite["env_ling_dir_trans"] = _create_standard_environment(
        env_id="env_ling_dir_trans",
        domain_name="Linguistics",
        relation_name="derived_from",
        entities=ling_entities,
        edges=ling_edges,
        hypothesis=EXPANDED_HYPOTHESES["DIR_TRANS"],
    )

    # 2. Training Environment: Legal (SYM_NONTRANS)
    legal_entities = ["FederalAuthority", "StateTribunalA", "StateTribunalB", "AppellateChamberX", "AppellateChamberY", "MunicipalMagistrate"]
    legal_edges = [
        ("FederalAuthority", "StateTribunalA"),
        ("StateTribunalA", "AppellateChamberX"),
        ("StateTribunalB", "AppellateChamberY"),
        ("AppellateChamberX", "MunicipalMagistrate"),
    ]
    suite["env_legal_sym_nontrans"] = _create_standard_environment(
        env_id="env_legal_sym_nontrans",
        domain_name="Jurisprudence",
        relation_name="treaty_partner_with",
        entities=legal_entities,
        edges=legal_edges,
        hypothesis=EXPANDED_HYPOTHESES["SYM_NONTRANS"],
    )

    # 3. Training Environment: Software Protocol (SYM_TRANS)
    soft_entities = ["ServiceRegistry", "AuthGatewayNode", "BillingClusterNode", "TelemetryAgentNode", "DatabaseReplicaA", "DatabaseReplicaB"]
    soft_edges = [
        ("ServiceRegistry", "AuthGatewayNode"),
        ("AuthGatewayNode", "BillingClusterNode"),
        ("TelemetryAgentNode", "DatabaseReplicaA"),
        ("DatabaseReplicaA", "DatabaseReplicaB"),
    ]
    suite["env_soft_sym_trans"] = _create_standard_environment(
        env_id="env_soft_sym_trans",
        domain_name="DistributedSoftware",
        relation_name="in_same_security_zone",
        entities=soft_entities,
        edges=soft_edges,
        hypothesis=EXPANDED_HYPOTHESES["SYM_TRANS"],
    )

    # 4. Training Environment: Bio Hierarchy (HIERARCHICAL_BOUNDARY)
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
    suite["env_bio_hierarchical"] = _create_standard_environment(
        env_id="env_bio_hierarchical",
        domain_name="BiologicalTaxonomy",
        relation_name="taxonomically_subsumes",
        entities=bio_entities,
        edges=bio_edges,
        hypothesis=EXPANDED_HYPOTHESES["HIERARCHICAL_BOUNDARY"],
        entity_levels=bio_levels,
    )

    # 5. Dev Environment: Chemical Pathways with Refutation (EXPLICIT_REFUTATION)
    chem_entities = ["SubstrateAlpha", "IntermediateBeta", "EnzymeGamma", "InhibitorDelta", "MetaboliteOmega", "CataboliteZeta"]
    chem_edges = [
        ("SubstrateAlpha", "IntermediateBeta"),
        ("IntermediateBeta", "EnzymeGamma"),
        ("EnzymeGamma", "MetaboliteOmega"),
    ]
    chem_negated = {("IntermediateBeta", "InhibitorDelta"), ("InhibitorDelta", "MetaboliteOmega")}
    suite["env_chem_refutation"] = _create_standard_environment(
        env_id="env_chem_refutation",
        domain_name="Biochemistry",
        relation_name="catalyzes_synthesis_of",
        entities=chem_entities,
        edges=chem_edges,
        hypothesis=EXPANDED_HYPOTHESES["EXPLICIT_REFUTATION"],
        negated_edges=chem_negated,
    )

    # 6. Dev Environment: Geopolitics (DIR_NONTRANS)
    geo_entities = ["BlocAllianceNorth", "MemberStateAlpha", "MemberStateBeta", "DiplomaticMissionX", "ObserverConsulateY", "NeutralTerritory"]
    geo_edges = [
        ("BlocAllianceNorth", "MemberStateAlpha"),
        ("BlocAllianceNorth", "MemberStateBeta"),
        ("MemberStateAlpha", "DiplomaticMissionX"),
        ("MemberStateBeta", "ObserverConsulateY"),
    ]
    suite["env_geo_dir_nontrans"] = _create_standard_environment(
        env_id="env_geo_dir_nontrans",
        domain_name="Geopolitics",
        relation_name="commands_diplomatic_envoy",
        entities=geo_entities,
        edges=geo_edges,
        hypothesis=EXPANDED_HYPOTHESES["DIR_NONTRANS"],
    )

    # 7. Sealed Meta-Test Environment 1: Cryptographic Blocked Bridge (BLOCKED_BRIDGE)
    crypto_entities = ["RootAuthorityKey", "IntermediateCertA", "IntermediateCertB", "RevocationList", "LeafSignerX", "ValidationNode"]
    crypto_edges = [
        ("RootAuthorityKey", "IntermediateCertA"),
        ("IntermediateCertA", "LeafSignerX"),
        ("RootAuthorityKey", "IntermediateCertB"),
        ("IntermediateCertB", "RevocationList"),
        ("RevocationList", "ValidationNode"),
    ]
    crypto_blocked = {"IntermediateCertB"}
    suite["env_crypto_blocked_bridge"] = _create_standard_environment(
        env_id="env_crypto_blocked_bridge",
        domain_name="CryptographicPKI",
        relation_name="authorizes_session_for",
        entities=crypto_entities,
        edges=crypto_edges,
        hypothesis=EXPANDED_HYPOTHESES["BLOCKED_BRIDGE"],
        blocked_nodes=crypto_blocked,
    )

    # 8. Sealed Meta-Test Environment 2: Quantum Lattice Open World (OPEN_WORLD_INCOMPLETE)
    quantum_entities = ["EntangledPhotonPairA", "QuantumRepeaterNode1", "QuantumRepeaterNode2", "BellStateDetector", "QubitReceiverX", "ClassicalChannel"]
    quantum_edges = [
        ("EntangledPhotonPairA", "QuantumRepeaterNode1"),
        ("QuantumRepeaterNode1", "QuantumRepeaterNode2"),
        ("QuantumRepeaterNode2", "BellStateDetector"),
    ]
    suite["env_quantum_open_world"] = _create_standard_environment(
        env_id="env_quantum_open_world",
        domain_name="QuantumTeleportation",
        relation_name="maintains_entanglement_with",
        entities=quantum_entities,
        edges=quantum_edges,
        hypothesis=EXPANDED_HYPOTHESES["OPEN_WORLD_INCOMPLETE"],
    )

    # 9. Sealed Meta-Test Environment 3: Social Parity Cycle (OUT-OF-HYPOTHESIS RULE)
    social_entities = ["HubLeader1", "RelayMemberA", "RelayMemberB", "BridgeNodeC", "TargetFollowerD", "IsolateObserverE"]
    social_edges = [
        ("HubLeader1", "RelayMemberA"),
        ("RelayMemberA", "RelayMemberB"),
        ("RelayMemberB", "BridgeNodeC"),
        ("BridgeNodeC", "TargetFollowerD"),
    ]
    suite["env_social_parity_cycle"] = _create_standard_environment(
        env_id="env_social_parity_cycle",
        domain_name="SocialNetworkParity",
        relation_name="shares_parity_influence_with",
        entities=social_entities,
        edges=social_edges,
        hypothesis=OUT_OF_HYPOTHESIS_PARITY,
    )

    return suite


def _create_standard_environment(
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
    entity_levels = entity_levels or {}
    blocked_nodes = blocked_nodes or set()
    negated_edges = negated_edges or set()

    all_pairs = []
    for u in entities:
        for v in entities:
            if u != v:
                ans = hypothesis.evaluate_query(u, v, edges, entity_levels, blocked_nodes, negated_edges)
                all_pairs.append((u, v, ans))

    yes_pairs = [p for p in all_pairs if p[2] == "YES"]
    no_pairs = [p for p in all_pairs if p[2] == "NO"]
    unk_pairs = [p for p in all_pairs if p[2] == "UNKNOWN"]

    test_queries: list[EnvironmentQuery] = []
    qid = 1

    for i in range(8):
        if i < len(yes_pairs):
            u, v, ans = yes_pairs[i]
        else:
            u, v = edges[i % len(edges)]
            ans = "YES"
        test_queries.append(EnvironmentQuery(
            query_id=f"t_{qid:02d}",
            source=u,
            target=v,
            query_type="DIRECT_OR_REACHABLE",
            question_text=f"Does {u} {relation_name} {v}?",
            expected_answer=ans,
        ))
        qid += 1

    for i in range(8):
        if i < len(no_pairs):
            u, v, ans = no_pairs[i]
        else:
            u = entities[-(i % len(entities)) - 1]
            v = entities[i % len(entities)]
            ans = "NO"
        test_queries.append(EnvironmentQuery(
            query_id=f"t_{qid:02d}",
            source=u,
            target=v,
            query_type="REFUTED_OR_DISCONNECTED",
            question_text=f"Does {u} {relation_name} {v}?",
            expected_answer=ans,
        ))
        qid += 1

    for i in range(8):
        if i < len(unk_pairs):
            u, v, ans = unk_pairs[i]
        else:
            u = entities[i % len(entities)]
            v = f"ExtUnknownNode_{i}"
            ans = "UNKNOWN"
        test_queries.append(EnvironmentQuery(
            query_id=f"t_{qid:02d}",
            source=u,
            target=v,
            query_type="OPEN_WORLD_OR_UNREACHABLE",
            question_text=f"Does {u} {relation_name} {v}?",
            expected_answer="UNKNOWN",
        ))
        qid += 1

    e0, e1 = edges[0]
    rev_ans = hypothesis.evaluate_query(e1, e0, edges, entity_levels, blocked_nodes, negated_edges)
    pair1 = MinimalPairControl(
        pair_id="min_pair_reversal",
        base_query=EnvironmentQuery("mp1_base", e0, e1, "DIRECT_FORWARD", f"Does {e0} {relation_name} {e1}?", "YES"),
        perturbed_query=EnvironmentQuery("mp1_pert", e1, e0, "DIRECTIONAL_REVERSAL", f"Does {e1} {relation_name} {e0}?", rev_ans),
        perturbation_type="DIRECTIONAL_REVERSAL",
    )

    bridge_src = edges[0][0]
    bridge_tgt = edges[-1][1] if len(edges) > 1 else entities[-1]
    bridge_ans = hypothesis.evaluate_query(bridge_src, bridge_tgt, edges, entity_levels, blocked_nodes, negated_edges)
    pair2 = MinimalPairControl(
        pair_id="min_pair_bridge",
        base_query=EnvironmentQuery("mp2_base", bridge_src, bridge_tgt, "MULTI_HOP_CHAIN", f"Does {bridge_src} {relation_name} {bridge_tgt}?", bridge_ans),
        perturbed_query=EnvironmentQuery("mp2_pert", bridge_src, f"DeletedBridge_{bridge_tgt}", "BRIDGE_DELETED", f"Does {bridge_src} {relation_name} DeletedBridge_{bridge_tgt}?", "UNKNOWN"),
        perturbation_type="BRIDGE_DELETION",
    )

    return AutonomousEnvironment(
        env_id=env_id,
        domain_name=domain_name,
        relation_name=relation_name,
        entities=entities,
        base_edges=edges,
        true_hypothesis=hypothesis,
        held_out_test_queries=test_queries,
        minimal_pairs=[pair1, pair2],
        entity_levels=entity_levels,
        blocked_nodes=blocked_nodes,
        negated_edges=negated_edges,
    )
