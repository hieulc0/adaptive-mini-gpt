"""Phase 2: Disentangled Factorial Relational Benchmark and Minimal Pair Perturbations.

Factorially crosses Domains x Topologies x Rule Axioms:
  - Topologies: CHAIN, TREE, DIAMOND, FORK across all domains.
  - Domains: Linguistics, Legal Precedence, Cellular Biology, Geopolitics, Software Dependencies.
  - Axioms: Transitive Directed, Non-transitive Directed, Symmetric, Equivalence, Refutation.
  - Fine-grained minimal pairs: exactly 1 directed edge reversed or deleted.
  - In-context demonstrations: k-shot rule exemplars without symbolic schema names.
"""
from __future__ import annotations

import collections
import copy
from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.uncertainty.epistemic_benchmark import EpistemicTask, RelationSemantics


@dataclass
class RuleDemonstration:
    premises: list[str]
    question: str
    answer: str  # "YES", "NO", "UNKNOWN"
    explanation: str

    def to_text(self) -> str:
        return f"[DEMO_PREMISES] {' '.join(self.premises)} [DEMO_QUERY] {self.question} [DEMO_VERDICT] {self.answer}"


@dataclass
class DisentangledWorldDataset:
    world_id: str
    domain: str
    topology: str
    rule_type: str  # "TRANSITIVE", "NONTRANSITIVE", "SYMMETRIC", "EQUIVALENCE"
    semantics: RelationSemantics
    premises: list[str]
    train_practice_pool: list[EpistemicTask]
    validation_pool: list[EpistemicTask]
    held_out_test_pool: list[EpistemicTask]
    demonstrations: list[RuleDemonstration] = field(default_factory=list)
    minimal_pair_probes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "domain": self.domain,
            "topology": self.topology,
            "rule_type": self.rule_type,
            "relation_name": self.semantics.name,
            "is_transitive": self.semantics.is_transitive,
            "is_symmetric": self.semantics.is_symmetric,
            "premise_count": len(self.premises),
            "train_size": len(self.train_practice_pool),
            "val_size": len(self.validation_pool),
            "test_size": len(self.held_out_test_pool),
            "demo_count": len(self.demonstrations),
            "minimal_pairs_count": len(self.minimal_pair_probes),
        }


def build_disentangled_graph(
    world_id: str,
    domain: str,
    topology: str,
    rule_type: str,
    rel: str,
    inv_rel: str,
    neg_rel: str,
    entities: list[str],
) -> DisentangledWorldDataset:
    """Builds a verified relational dataset with decoupled domain, topology, and rule axioms."""
    assert len(entities) >= 12, "Requires at least 12 entities"
    e = entities

    is_trans = rule_type in ("TRANSITIVE", "EQUIVALENCE")
    is_sym = rule_type in ("SYMMETRIC", "EQUIVALENCE")
    is_dir = not is_sym

    semantics = RelationSemantics(
        name=rel,
        inverse_name=inv_rel,
        is_transitive=is_trans,
        is_symmetric=is_sym,
        is_directional=is_dir,
        world_assumption="OPEN_WORLD",
    )

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

    if topology == "CHAIN":
        # Chain 1: e0 -> e1 -> e2 -> e3 -> e4 -> e5 -> e6
        add_edge(e[0], e[1])
        add_edge(e[1], e[2])
        add_edge(e[2], e[3])
        add_edge(e[3], e[4])
        add_edge(e[4], e[5])
        add_edge(e[5], e[6])
        # Chain 2: e7 -> e8 -> e9 -> e10 -> e11
        add_edge(e[7], e[8])
        add_edge(e[8], e[9])
        add_edge(e[9], e[10])
        add_edge(e[10], e[11])
        # Inter-chain links
        add_edge(e[2], e[7])
        add_edge(e[4], e[9])
        # Explicit refutations
        add_refutation(e[2], e[6])
        add_refutation(e[3], e[10])

    elif topology == "TREE":
        # Hierarchical tree rooted at e0
        add_edge(e[0], e[1])
        add_edge(e[0], e[2])
        add_edge(e[1], e[3])
        add_edge(e[1], e[4])
        add_edge(e[2], e[5])
        add_edge(e[2], e[6])
        add_edge(e[3], e[7])
        add_edge(e[3], e[8])
        add_edge(e[4], e[9])
        add_edge(e[5], e[10])
        add_edge(e[6], e[11])
        # Refutations
        add_refutation(e[3], e[4])
        add_refutation(e[7], e[8])

    elif topology == "DIAMOND":
        # Multi-stage diamond
        add_edge(e[0], e[1])
        add_edge(e[0], e[2])
        add_edge(e[1], e[3])
        add_edge(e[2], e[3])
        add_edge(e[3], e[4])
        add_edge(e[3], e[5])
        add_edge(e[4], e[6])
        add_edge(e[5], e[6])
        add_edge(e[6], e[7])
        add_edge(e[6], e[8])
        add_edge(e[7], e[9])
        add_edge(e[8], e[9])
        add_edge(e[10], e[11])
        # Refutations
        add_refutation(e[4], e[5])
        add_refutation(e[7], e[8])

    elif topology == "FORK":
        # Fork with hub at e0
        add_edge(e[1], e[0])
        add_edge(e[2], e[0])
        add_edge(e[8], e[0])
        add_edge(e[9], e[0])
        add_edge(e[0], e[3])
        add_edge(e[0], e[4])
        add_edge(e[0], e[5])
        add_edge(e[0], e[10])
        add_edge(e[0], e[11])
        add_edge(e[6], e[7])
        # Refutations
        add_refutation(e[1], e[2])
        add_refutation(e[8], e[9])
    else:
        raise ValueError(f"Unknown topology: {topology}")

    def is_connected(u: str, v: str) -> bool:
        if not is_trans:
            # Only 1-hop direct edges are supported
            return v in adj[u]
        # Multi-hop transitive BFS
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

    # Generate demonstrations for rule induction (without rule name!)
    demonstrations: list[RuleDemonstration] = []
    # Demo 1: Direct 1-hop
    demonstrations.append(
        RuleDemonstration(
            premises=[f"{e[0]} {rel} {e[1]}"],
            question=f"Does {e[0]} {rel} {e[1]}?",
            answer="YES",
            explanation="Direct connection observed.",
        )
    )
    # Demo 2: Transitive 2-hop or Non-transitive
    if is_trans:
        demonstrations.append(
            RuleDemonstration(
                premises=[f"{e[0]} {rel} {e[1]}", f"{e[1]} {rel} {e[2]}"],
                question=f"Does {e[0]} {rel} {e[2]}?",
                answer="YES",
                explanation="Sequential chain holds transitively.",
            )
        )
    else:
        demonstrations.append(
            RuleDemonstration(
                premises=[f"{e[0]} {rel} {e[1]}", f"{e[1]} {rel} {e[2]}"],
                question=f"Does {e[0]} {rel} {e[2]}?",
                answer="UNKNOWN",
                explanation="Multi-hop steps are not implied by this relation.",
            )
        )
    # Demo 3: Reverse query
    if is_sym:
        demonstrations.append(
            RuleDemonstration(
                premises=[f"{e[0]} {rel} {e[1]}"],
                question=f"Does {e[1]} {rel} {e[0]}?",
                answer="YES",
                explanation="Relation is bidirectional.",
            )
        )
    else:
        demonstrations.append(
            RuleDemonstration(
                premises=[f"{e[0]} {rel} {e[1]}"],
                question=f"Does {e[1]} {rel} {e[0]}?",
                answer="NO",
                explanation="Direction cannot be inverted.",
            )
        )

    # Minimal pair perturbations (changing exactly 1 edge)
    minimal_pairs: list[dict[str, Any]] = []
    # Pair A: Forward vs Reversed edge
    minimal_pairs.append({
        "pair_type": "DIRECTIONAL_REVERSAL",
        "original_premise": f"{e[0]} {rel} {e[1]}",
        "perturbed_premise": f"{e[1]} {rel} {e[0]}",
        "original_query": f"Does {e[0]} {rel} {e[1]}?",
        "perturbed_query": f"Does {e[1]} {rel} {e[0]}?",
        "original_verdict": "YES",
        "perturbed_verdict": "YES" if is_sym else "NO",
    })
    # Pair B: Bridge deletion (deleting e1 -> e2)
    minimal_pairs.append({
        "pair_type": "BRIDGE_DELETION",
        "original_premise": f"{e[1]} {rel} {e[2]}",
        "perturbed_premise": "NONE",
        "query": f"Does {e[0]} {rel} {e[2]}?",
        "original_verdict": "YES" if is_trans else "UNKNOWN",
        "perturbed_verdict": "UNKNOWN",
    })

    # Classify all entity pairs
    raw_yes: list[EpistemicTask] = []
    raw_no: list[EpistemicTask] = []
    raw_unk: list[EpistemicTask] = []
    task_idx = 0

    for i in range(len(e)):
        for j in range(len(e)):
            if i == j:
                continue
            u, v = e[i], e[j]
            task_idx += 1
            t_id = f"{world_id}_{task_idx}"
            q = f"Does {u} {rel} {v}?"

            conn_forward = is_connected(u, v)
            conn_reverse = is_connected(v, u)
            is_refuted = (u, v) in refutations or (v, u) in refutations

            if conn_forward:
                raw_yes.append(EpistemicTask(t_id, world_id, "test", premises, q, "YES", "supported", [u, v]))
            elif is_refuted or (conn_reverse and is_dir):
                raw_no.append(EpistemicTask(t_id, world_id, "test", premises, q, "NO", "refuted", [u, v]))
            else:
                raw_unk.append(EpistemicTask(t_id, world_id, "test", premises, q, "UNKNOWN", "disconnected", [u, v]))

    # Deduplicate
    def dedup(tasks: list[EpistemicTask]) -> list[EpistemicTask]:
        seen = set()
        out = []
        for t in tasks:
            key = (t.question, t.expected_verdict)
            if key not in seen:
                seen.add(key)
                out.append(t)
        return out

    raw_yes = dedup(raw_yes)
    raw_no = dedup(raw_no)
    raw_unk = dedup(raw_unk)

    # Strictly stratified test pool: 8 YES, 8 NO, 8 UNK = 24 queries
    test_pool: list[EpistemicTask] = []
    for k in range(8):
        test_pool.append(copy.deepcopy(raw_yes[k % len(raw_yes)]))
        test_pool.append(copy.deepcopy(raw_no[k % len(raw_no)]))
        test_pool.append(copy.deepcopy(raw_unk[k % len(raw_unk)]))

    # Validation: 4 each = 12 total
    val_pool: list[EpistemicTask] = []
    for k in range(4):
        val_pool.append(copy.deepcopy(raw_yes[(k + 8) % len(raw_yes)]))
        val_pool.append(copy.deepcopy(raw_no[(k + 8) % len(raw_no)]))
        val_pool.append(copy.deepcopy(raw_unk[(k + 8) % len(raw_unk)]))

    # Practice pool: interleaved up to 36 total (12 each)
    train_pool: list[EpistemicTask] = []
    for k in range(12):
        train_pool.append(copy.deepcopy(raw_yes[(k + 12) % len(raw_yes)]))
        train_pool.append(copy.deepcopy(raw_no[(k + 12) % len(raw_no)]))
        train_pool.append(copy.deepcopy(raw_unk[(k + 12) % len(raw_unk)]))

    return DisentangledWorldDataset(
        world_id=world_id,
        domain=domain,
        topology=topology,
        rule_type=rule_type,
        semantics=semantics,
        premises=premises,
        train_practice_pool=train_pool,
        validation_pool=val_pool,
        held_out_test_pool=test_pool,
        demonstrations=demonstrations,
        minimal_pair_probes=minimal_pairs,
    )


def get_m48_disentangled_suite() -> dict[str, DisentangledWorldDataset]:
    """Builds a balanced factorial suite where domains and topologies are completely decoupled."""
    suite: dict[str, DisentangledWorldDataset] = {}

    # Domain 1: Linguistics - TREE topology - TRANSITIVE
    suite["ling_tree_transitive"] = build_disentangled_graph(
        world_id="ling_tree_transitive",
        domain="linguistics",
        topology="TREE",
        rule_type="TRANSITIVE",
        rel="derived_from",
        inv_rel="originates_into",
        neg_rel="unrelated_to",
        entities=[
            "ProtoIndoEuropean", "ProtoItalic", "ProtoGermanic", "LatinArchaic", "OldNorse", "VulgarLatin",
            "OldEnglish", "AngloSaxon", "OldFrench", "MiddleHighGerman", "GothicBiblical", "EarlyModernEnglish"
        ],
    )

    # Domain 2: Linguistics - CHAIN topology - NONTRANSITIVE
    suite["ling_chain_nontransitive"] = build_disentangled_graph(
        world_id="ling_chain_nontransitive",
        domain="linguistics",
        topology="CHAIN",
        rule_type="NONTRANSITIVE",
        rel="immediately_follows",
        inv_rel="immediately_precedes",
        neg_rel="separated_from",
        entities=[
            "TokenRoot", "TokenPrefix", "TokenStem", "TokenInfix", "TokenSuffix", "TokenEnclitic",
            "TokenTerminal", "TokenPunct", "TokenPause", "TokenStop", "TokenNewline", "TokenEOF"
        ],
    )

    # Domain 3: Legal - CHAIN topology - TRANSITIVE
    suite["legal_chain_transitive"] = build_disentangled_graph(
        world_id="legal_chain_transitive",
        domain="legal",
        topology="CHAIN",
        rule_type="TRANSITIVE",
        rel="appeals_to",
        inv_rel="reviews",
        neg_rel="overrules",
        entities=[
            "MagistrateCourt", "CountyTribunal", "DistrictBench", "AppellateDivision", "SupremeJudicature", "ConstitutionalChamber",
            "AdHocPanel", "ArbitrationForum", "SpecialTribunal", "EnBancCourt", "PrivyCouncil", "InternationalCourt"
        ],
    )

    # Domain 4: Legal - FORK topology - SYMMETRIC
    suite["legal_fork_symmetric"] = build_disentangled_graph(
        world_id="legal_fork_symmetric",
        domain="legal",
        topology="FORK",
        rule_type="SYMMETRIC",
        rel="jurisdiction_reciprocal_with",
        inv_rel="jurisdiction_reciprocal_with",
        neg_rel="disputes_jurisdiction_with",
        entities=[
            "FederalAuthority", "StateTribunalA", "StateTribunalB", "MaritimeCourt", "BankruptcyBench", "CustomsPanel",
            "ForeignTribunalA", "ForeignTribunalB", "RegionalAssize", "MunicipalBench", "CommercialCourt", "LabourArbitration"
        ],
    )

    # Domain 5: Cellular - DIAMOND topology - TRANSITIVE
    suite["cellular_diamond_transitive"] = build_disentangled_graph(
        world_id="cellular_diamond_transitive",
        domain="cellular",
        topology="DIAMOND",
        rule_type="TRANSITIVE",
        rel="catalyzes_synthesis_of",
        inv_rel="synthesized_by",
        neg_rel="degrades",
        entities=[
            "SubstrateEnzyme", "IntermediateKinaseA", "IntermediateKinaseB", "EffectorProtein", "TranscriptionFactor", "CellularReceptor",
            "InhibitorComplex", "NuclearSignal", "MembranePhospholipid", "CytosolicPhosphatase", "RibosomalSubunit", "MessengerRNA"
        ],
    )

    # Domain 6: Geopolitics (Sealed Meta-Test) - FORK topology - NONTRANSITIVE
    suite["geopolitics_fork_nontransitive"] = build_disentangled_graph(
        world_id="geopolitics_fork_nontransitive",
        domain="geopolitics",
        topology="FORK",
        rule_type="NONTRANSITIVE",
        rel="sponsors_proxy_treaty_with",
        inv_rel="sponsored_by",
        neg_rel="embargos",
        entities=[
            "HegemonRepublic", "SovereignStateA", "SovereignStateB", "ClientTerritoryA", "ClientTerritoryB", "BufferZone",
            "NeutralFederation", "NonAlignedPact", "StrategicProtectorate", "IslandCoalition", "BorderPrincipality", "AutonomousEnclave"
        ],
    )

    # Domain 7: Software (Sealed Meta-Test) - DIAMOND topology - NONTRANSITIVE
    suite["software_diamond_nontransitive"] = build_disentangled_graph(
        world_id="software_diamond_nontransitive",
        domain="software",
        topology="DIAMOND",
        rule_type="NONTRANSITIVE",
        rel="calls_dynamically",
        inv_rel="invoked_by",
        neg_rel="sandboxes_from",
        entities=[
            "ClientMicroservice", "ApiGatewayWorkerA", "ApiGatewayWorkerB", "DataRouterQueue", "CacheClusterNode", "DatabaseShard",
            "TelemetryAgent", "DeadLetterQueue", "LoadBalancerNode", "IngressController", "DistributedLogStream", "MetricCollector"
        ],
    )

    return suite
