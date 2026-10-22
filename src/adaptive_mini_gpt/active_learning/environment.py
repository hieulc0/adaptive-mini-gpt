"""Phase 2: Controlled Active-Learning Environment and Ambiguous Rule Families."""
from __future__ import annotations

import collections
import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RuleHypothesis:
    name: str  # "DIR_TRANS", "DIR_NONTRANS", "SYM_TRANS", "SYM_NONTRANS"
    is_transitive: bool
    is_symmetric: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "is_transitive": self.is_transitive,
            "is_symmetric": self.is_symmetric,
            "description": self.description,
        }


HYPOTHESES: dict[str, RuleHypothesis] = {
    "DIR_TRANS": RuleHypothesis("DIR_TRANS", is_transitive=True, is_symmetric=False, description="Directional & Transitive"),
    "DIR_NONTRANS": RuleHypothesis("DIR_NONTRANS", is_transitive=False, is_symmetric=False, description="Directional & Non-transitive"),
    "SYM_TRANS": RuleHypothesis("SYM_TRANS", is_transitive=True, is_symmetric=True, description="Symmetric & Transitive (Equivalence)"),
    "SYM_NONTRANS": RuleHypothesis("SYM_NONTRANS", is_transitive=False, is_symmetric=True, description="Symmetric & Non-transitive"),
}


@dataclass
class CandidateQuery:
    query_id: str
    source: str
    target: str
    query_type: str  # "REVERSE_TEST", "TWO_HOP_BRIDGE", "THREE_HOP_CHAIN", "REFUTATION_CHECK", "DISCONNECTED_CHECK"
    question_text: str
    expected_answer: str  # "YES", "NO", "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "source": self.source,
            "target": self.target,
            "query_type": self.query_type,
            "question_text": self.question_text,
            "expected_answer": self.expected_answer,
        }


@dataclass
class ActiveLearningEnvironment:
    env_id: str
    domain_name: str
    relation_name: str
    neg_relation_name: str
    entities: list[str]
    true_hypothesis: RuleHypothesis
    premises: list[str]
    candidate_queries: list[CandidateQuery]
    held_out_test_queries: list[CandidateQuery]
    minimal_pairs: list[dict[str, Any]] = field(default_factory=list)
    oracle_call_count: int = 0

    def ask_oracle(self, query_id: str) -> str:
        """Simulates an oracle interaction returning ONLY the verified answer.

        Crucially: does NOT disclose the underlying rule hypothesis, graph structure,
        or symbolic formulation.
        """
        for q in self.candidate_queries:
            if q.query_id == query_id:
                self.oracle_call_count += 1
                return q.expected_answer
        raise KeyError(f"Query {query_id} not found in candidate pool.")

    def reset_oracle_counter(self) -> None:
        self.oracle_call_count = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "domain_name": self.domain_name,
            "relation_name": self.relation_name,
            "true_hypothesis": self.true_hypothesis.to_dict(),
            "premise_count": len(self.premises),
            "candidate_query_count": len(self.candidate_queries),
            "held_out_test_count": len(self.held_out_test_queries),
            "minimal_pairs_count": len(self.minimal_pairs),
        }


def build_active_environment(
    env_id: str,
    domain_name: str,
    relation_name: str,
    neg_relation_name: str,
    entities: list[str],
    true_hypothesis: RuleHypothesis,
) -> ActiveLearningEnvironment:
    """Constructs a controlled environment with ambiguous initial evidence and query pool."""
    assert len(entities) >= 10, "Requires at least 10 entities"
    e = entities
    rel = relation_name
    neg_rel = neg_relation_name
    is_trans = true_hypothesis.is_transitive
    is_sym = true_hypothesis.is_symmetric
    is_dir = not is_sym

    premises: list[str] = []
    adj: dict[str, set[str]] = collections.defaultdict(set)
    refutations: set[tuple[str, str]] = set()

    def add_edge(u: str, v: str) -> None:
        premises.append(f"{u} {rel} {v}")
        adj[u].add(v)
        if is_sym:
            adj[v].add(u)

    def add_refutation(u: str, v: str) -> None:
        premises.append(f"{u} {neg_rel} {v}")
        refutations.add((u, v))
        if is_sym:
            refutations.add((v, u))

    # Standardized scaffold: Chain + Branch + Isolated + Refuted
    # e0 -> e1 -> e2 -> e3 -> e4
    add_edge(e[0], e[1])
    add_edge(e[1], e[2])
    add_edge(e[2], e[3])
    add_edge(e[3], e[4])
    # Branch e1 -> e5 -> e6
    add_edge(e[1], e[5])
    add_edge(e[5], e[6])
    # Parallel edge e7 -> e8
    add_edge(e[7], e[8])
    # Refutation e2 -/- e7
    add_refutation(e[2], e[7])
    # e9 is completely disconnected (open-world ungrounded)

    def is_connected(u: str, v: str) -> bool:
        if not is_trans:
            return v in adj[u]
        visited = set()
        queue = collections.deque([u])
        while queue:
            curr = queue.popleft()
            if curr == v and curr != u:
                return True
            if curr not in visited:
                visited.add(curr)
                for nxt in adj[curr]:
                    queue.append(nxt)
        return False

    def get_verdict(u: str, v: str) -> str:
        if is_connected(u, v):
            return "YES"
        if (u, v) in refutations or (v, u) in refutations or (is_connected(v, u) and is_dir):
            return "NO"
        return "UNKNOWN"

    # Candidate Query Pool Q for active experimentation (12 informative queries)
    candidate_queries: list[CandidateQuery] = []
    # 1. Reverse test (e1 -> e0): tests symmetry
    candidate_queries.append(CandidateQuery("q_rev_01", e[1], e[0], "REVERSE_TEST", f"Does {e[1]} {rel} {e[0]}?", get_verdict(e[1], e[0])))
    candidate_queries.append(CandidateQuery("q_rev_12", e[2], e[1], "REVERSE_TEST", f"Does {e[2]} {rel} {e[1]}?", get_verdict(e[2], e[1])))
    # 2. Two-hop bridge test (e0 -> e2 via e1): tests transitivity
    candidate_queries.append(CandidateQuery("q_bridge_02", e[0], e[2], "TWO_HOP_BRIDGE", f"Does {e[0]} {rel} {e[2]}?", get_verdict(e[0], e[2])))
    candidate_queries.append(CandidateQuery("q_bridge_13", e[1], e[3], "TWO_HOP_BRIDGE", f"Does {e[1]} {rel} {e[3]}?", get_verdict(e[1], e[3])))
    candidate_queries.append(CandidateQuery("q_bridge_16", e[1], e[6], "TWO_HOP_BRIDGE", f"Does {e[1]} {rel} {e[6]}?", get_verdict(e[1], e[6])))
    # 3. Three-hop chain test (e0 -> e3, e0 -> e4)
    candidate_queries.append(CandidateQuery("q_chain_03", e[0], e[3], "THREE_HOP_CHAIN", f"Does {e[0]} {rel} {e[3]}?", get_verdict(e[0], e[3])))
    candidate_queries.append(CandidateQuery("q_chain_04", e[0], e[4], "THREE_HOP_CHAIN", f"Does {e[0]} {rel} {e[4]}?", get_verdict(e[0], e[4])))
    # 4. Refutation check
    candidate_queries.append(CandidateQuery("q_ref_27", e[2], e[7], "REFUTATION_CHECK", f"Does {e[2]} {rel} {e[7]}?", get_verdict(e[2], e[7])))
    candidate_queries.append(CandidateQuery("q_ref_07", e[0], e[7], "REFUTATION_CHECK", f"Does {e[0]} {rel} {e[7]}?", get_verdict(e[0], e[7])))
    # 5. Direct 1-hop checks
    candidate_queries.append(CandidateQuery("q_dir_01", e[0], e[1], "DIRECT_EDGE", f"Does {e[0]} {rel} {e[1]}?", get_verdict(e[0], e[1])))
    candidate_queries.append(CandidateQuery("q_dir_78", e[7], e[8], "DIRECT_EDGE", f"Does {e[7]} {rel} {e[8]}?", get_verdict(e[7], e[8])))
    # 6. Disconnected check
    candidate_queries.append(CandidateQuery("q_unk_09", e[0], e[9], "DISCONNECTED_CHECK", f"Does {e[0]} {rel} {e[9]}?", get_verdict(e[0], e[9])))

    # Generate held-out evaluation pool (balanced 24 queries: 8 YES, 8 NO, 8 UNKNOWN)
    raw_yes: list[CandidateQuery] = []
    raw_no: list[CandidateQuery] = []
    raw_unk: list[CandidateQuery] = []

    q_idx = 0
    for i in range(len(e)):
        for j in range(len(e)):
            if i == j:
                continue
            u, v = e[i], e[j]
            q_idx += 1
            verdict = get_verdict(u, v)
            cq = CandidateQuery(f"eval_{env_id}_{q_idx}", u, v, "HELD_OUT_EVAL", f"Does {u} {rel} {v}?", verdict)
            if verdict == "YES":
                raw_yes.append(cq)
            elif verdict == "NO":
                raw_no.append(cq)
            else:
                raw_unk.append(cq)

    held_out_test_queries: list[CandidateQuery] = []
    for k in range(8):
        held_out_test_queries.append(copy.deepcopy(raw_yes[k % len(raw_yes)]))
        held_out_test_queries.append(copy.deepcopy(raw_no[k % len(raw_no)]))
        held_out_test_queries.append(copy.deepcopy(raw_unk[k % len(raw_unk)]))

    # Minimal pairs with paired correctness control
    minimal_pairs: list[dict[str, Any]] = [
        {
            "pair_name": "DIRECTIONAL_REVERSAL_PAIR",
            "original_query": CandidateQuery("orig_dir", e[0], e[1], "DIRECT", f"Does {e[0]} {rel} {e[1]}?", "YES"),
            "perturbed_query": CandidateQuery("pert_dir", e[1], e[0], "REVERSED", f"Does {e[1]} {rel} {e[0]}?", "YES" if is_sym else "NO"),
        },
        {
            "pair_name": "BRIDGE_DELETION_PAIR",
            "original_query": CandidateQuery("orig_bridge", e[0], e[2], "CHAIN_CONNECTED", f"Does {e[0]} {rel} {e[2]}?", "YES" if is_trans else "UNKNOWN"),
            "perturbed_query": CandidateQuery("pert_bridge", e[0], e[2], "BRIDGE_DELETED", f"Does {e[0]} {rel} {e[2]}?", "UNKNOWN"),
        },
    ]

    return ActiveLearningEnvironment(
        env_id=env_id,
        domain_name=domain_name,
        relation_name=relation_name,
        neg_relation_name=neg_relation_name,
        entities=entities,
        true_hypothesis=true_hypothesis,
        premises=premises,
        candidate_queries=candidate_queries,
        held_out_test_queries=held_out_test_queries,
        minimal_pairs=minimal_pairs,
    )


def build_m49_environment_suite() -> dict[str, ActiveLearningEnvironment]:
    """Constructs the M49 suite separating meta-training, development, and sealed meta-test."""
    suite: dict[str, ActiveLearningEnvironment] = {}

    # Training Environment 1: Linguistics - Directional Transitive
    suite["env_ling_dir_trans"] = build_active_environment(
        env_id="env_ling_dir_trans",
        domain_name="linguistics",
        relation_name="derived_from",
        neg_relation_name="unrelated_to",
        entities=["ProtoIndoEuropean", "ProtoItalic", "ProtoGermanic", "LatinArchaic", "OldNorse", "VulgarLatin", "OldEnglish", "AngloSaxon", "OldFrench", "MiddleHighGerman"],
        true_hypothesis=HYPOTHESES["DIR_TRANS"],
    )

    # Training Environment 2: Legal - Symmetric Non-transitive
    suite["env_legal_sym_nontrans"] = build_active_environment(
        env_id="env_legal_sym_nontrans",
        domain_name="legal",
        relation_name="treaty_partner_with",
        neg_relation_name="embargos",
        entities=["FederalAuthority", "StateTribunalA", "StateTribunalB", "MaritimeCourt", "BankruptcyBench", "CustomsPanel", "ForeignTribunalA", "ForeignTribunalB", "RegionalAssize", "MunicipalBench"],
        true_hypothesis=HYPOTHESES["SYM_NONTRANS"],
    )

    # Training Environment 3: Cellular - Directional Non-transitive
    suite["env_cell_dir_nontrans"] = build_active_environment(
        env_id="env_cell_dir_nontrans",
        domain_name="cellular",
        relation_name="inhibits_immediately",
        neg_relation_name="activates",
        entities=["SubstrateEnzyme", "IntermediateKinaseA", "IntermediateKinaseB", "EffectorProtein", "TranscriptionFactor", "CellularReceptor", "InhibitorComplex", "NuclearSignal", "MembranePhospholipid", "CytosolicPhosphatase"],
        true_hypothesis=HYPOTHESES["DIR_NONTRANS"],
    )

    # Development Environment 1: Software - Symmetric Transitive (Equivalence)
    suite["env_soft_sym_trans"] = build_active_environment(
        env_id="env_soft_sym_trans",
        domain_name="software",
        relation_name="equivalent_protocol_to",
        neg_relation_name="incompatible_with",
        entities=["ClientMicroservice", "ApiGatewayWorkerA", "ApiGatewayWorkerB", "DataRouterQueue", "CacheClusterNode", "DatabaseShard", "TelemetryAgent", "DeadLetterQueue", "LoadBalancerNode", "IngressController"],
        true_hypothesis=HYPOTHESES["SYM_TRANS"],
    )

    # Development Environment 2: Geopolitics - Directional Transitive
    suite["env_geo_dir_trans"] = build_active_environment(
        env_id="env_geo_dir_trans",
        domain_name="geopolitics",
        relation_name="subsumes_protectorate_of",
        neg_relation_name="disputes_sovereignty_with",
        entities=["HegemonRepublic", "SovereignStateA", "SovereignStateB", "ClientTerritoryA", "ClientTerritoryB", "BufferZone", "NeutralFederation", "NonAlignedPact", "StrategicProtectorate", "IslandCoalition"],
        true_hypothesis=HYPOTHESES["DIR_TRANS"],
    )

    # Sealed Meta-Test: Cryptographic Trust Network (Ambiguous Initial State)
    # Target rule: DIR_NONTRANS (Cryptographic delegation where token signing is strictly 1-hop non-transitive & asymmetric)
    suite["env_crypto_sealed_meta_test"] = build_active_environment(
        env_id="env_crypto_sealed_meta_test",
        domain_name="cryptography",
        relation_name="authorizes_session_for",
        neg_relation_name="revokes_keys_of",
        entities=["RootAuthorityKey", "IntermediateCertA", "IntermediateCertB", "LeafSignerX", "LeafSignerY", "HardwareEnclave", "RevocationList", "ValidationNode", "EphemeralNonce", "ZeroKnowledgeProver"],
        true_hypothesis=HYPOTHESES["DIR_NONTRANS"],
    )

    return suite
