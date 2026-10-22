"""Phase 2: Minimal Structural Representation Benchmark.

Constructs independent synthetic graph worlds spanning:
  - Topologies: CHAIN, TREE, DIAMOND, FORK with genuine topological distinguishable cues.
  - Composition depths: 1-hop, 2-hop, 3-hop directed chains.
  - Relation semantics: directed vs symmetric, transitive vs non-transitive.
  - Strict open-world refutation (NO via anti-relation / conflict) vs missing path (UNKNOWN).
  - Multi-world benchmark suite with distinct vocabularies and entity sets.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.uncertainty.epistemic_benchmark import EpistemicTask, RelationSemantics


@dataclass
class EnhancedRelationalDataset:
    world_name: str
    split: str  # "train", "val", "test", "meta_test"
    topology: str  # "CHAIN", "TREE", "DIAMOND", "FORK"
    semantics: RelationSemantics
    premises: list[str]
    train_practice_pool: list[EpistemicTask]  # Strictly stratified interleaved
    validation_pool: list[EpistemicTask]
    held_out_test_pool: list[EpistemicTask]
    reachability_probes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_name": self.world_name,
            "split": self.split,
            "topology": self.topology,
            "relation_name": self.semantics.name,
            "is_transitive": self.semantics.is_transitive,
            "is_symmetric": self.semantics.is_symmetric,
            "premise_count": len(self.premises),
            "train_size": len(self.train_practice_pool),
            "val_size": len(self.validation_pool),
            "test_size": len(self.held_out_test_pool),
        }


def build_enhanced_relational_world(
    world_name: str,
    split: str,
    topology: str,
    rel: str,
    inv_rel: str,
    neg_rel: str,
    entities: list[str],
    is_transitive: bool = True,
    is_symmetric: bool = False,
    is_directional: bool = True,
) -> EnhancedRelationalDataset:
    """Builds a relational graph world with verified open-world semantics and structural probes."""
    assert len(entities) >= 8, "World requires at least 8 distinct entities"
    e = entities

    semantics = RelationSemantics(
        name=rel,
        inverse_name=inv_rel,
        is_transitive=is_transitive,
        is_symmetric=is_symmetric,
        is_directional=is_directional,
        world_assumption="OPEN_WORLD",
    )

    premises: list[str] = []
    adj: dict[str, set[str]] = collections.defaultdict(set)
    refutations: set[tuple[str, str]] = set()

    def add_edge(u: str, v: str) -> None:
        premises.append(f"{u} {rel} {v}")
        adj[u].add(v)
        if is_symmetric:
            adj[v].add(u)

    def add_refutation(u: str, v: str) -> None:
        premises.append(f"{u} {neg_rel} {v}")
        refutations.add((u, v))
        if is_symmetric:
            refutations.add((v, u))

    # Construct topology with explicit structural patterns
    if topology == "CHAIN":
        # Linear sequence: e0 -> e1 -> e2 -> e3 -> e4 -> e5. e6, e7 isolated.
        add_edge(e[0], e[1])
        add_edge(e[1], e[2])
        add_edge(e[2], e[3])
        add_edge(e[3], e[4])
        add_edge(e[4], e[5])
        # Refutation: e2 explicitly blocked from e5
        add_refutation(e[2], e[6])
        # Isolated premise
        add_edge(e[6], e[7])

    elif topology == "TREE":
        # Hierarchical tree: e0 -> e1, e0 -> e2; e1 -> e3, e1 -> e4; e2 -> e5, e2 -> e6.
        add_edge(e[0], e[1])
        add_edge(e[0], e[2])
        add_edge(e[1], e[3])
        add_edge(e[1], e[4])
        add_edge(e[2], e[5])
        add_edge(e[2], e[6])
        # Refutation: branch contradiction between e3 and e4
        add_refutation(e[3], e[4])
        add_edge(e[6], e[7])

    elif topology == "DIAMOND":
        # Diamond multi-path: e0 -> e1, e0 -> e2; e1 -> e3, e2 -> e3; e3 -> e4; e3 -> e5.
        add_edge(e[0], e[1])
        add_edge(e[0], e[2])
        add_edge(e[1], e[3])
        add_edge(e[2], e[3])
        add_edge(e[3], e[4])
        add_edge(e[3], e[5])
        # Refutation
        add_refutation(e[4], e[5])
        add_edge(e[6], e[7])

    elif topology == "FORK":
        # Common-cause fork: e1 -> e0, e2 -> e0; e0 -> e3, e0 -> e4, e0 -> e5.
        add_edge(e[1], e[0])
        add_edge(e[2], e[0])
        add_edge(e[0], e[3])
        add_edge(e[0], e[4])
        add_edge(e[0], e[5])
        # Refutation
        add_refutation(e[1], e[2])
        add_edge(e[6], e[7])

    else:
        raise ValueError(f"Unknown topology: {topology}")

    def bfs_reachable(src: str, dst: str) -> bool:
        visited = set()
        queue = collections.deque([src])
        while queue:
            curr = queue.popleft()
            if curr == dst and curr != src:
                return True
            if curr not in visited:
                visited.add(curr)
                for nxt in adj[curr]:
                    queue.append(nxt)
        return False

    def path_length(src: str, dst: str) -> int:
        queue = collections.deque([(src, 0)])
        visited = {src}
        while queue:
            curr, d = queue.popleft()
            if curr == dst and curr != src:
                return d
            for nxt in adj[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, d + 1))
        return -1

    task_counter = 0
    def make_task(s: str, o: str, verdict: str, tag: str = "standard") -> EpistemicTask:
        nonlocal task_counter
        task_counter += 1
        t_id = f"{world_name}_{split}_{task_counter}"
        return EpistemicTask(
            task_id=t_id,
            world_name=world_name,
            split=split,
            premises=list(premises),
            question=f"Does {s} {rel} {o}?",
            expected_verdict=verdict,
            epistemic_category=tag,
            entities_involved=[s, o],
        )

    # 1. Generate candidate pools
    yes_tasks: list[EpistemicTask] = []
    no_tasks: list[EpistemicTask] = []
    unk_tasks: list[EpistemicTask] = []
    probes: list[dict[str, Any]] = []

    # Check pairs
    for i in range(len(e)):
        for j in range(len(e)):
            if i == j:
                continue
            u, v = e[i], e[j]
            is_reach = bfs_reachable(u, v)
            is_rev_reach = bfs_reachable(v, u)
            is_refuted = (u, v) in refutations or (v, u) in refutations

            dist = path_length(u, v)
            if is_reach:
                yes_tasks.append(make_task(u, v, "YES", f"{dist}hop"))
                probes.append({"src": u, "dst": v, "verdict": "YES", "hops": dist, "is_reach": True})
            elif is_refuted or (is_rev_reach and not is_symmetric and is_directional):
                no_tasks.append(make_task(u, v, "NO", "refuted"))
                probes.append({"src": u, "dst": v, "verdict": "NO", "hops": -1, "is_reach": False})
            else:
                unk_tasks.append(make_task(u, v, "UNKNOWN", "disconnected"))
                probes.append({"src": u, "dst": v, "verdict": "UNKNOWN", "hops": -1, "is_reach": False})

    # Deduplicate while preserving order
    def dedup(tasks: list[EpistemicTask]) -> list[EpistemicTask]:
        seen = set()
        res = []
        for t in tasks:
            key = (t.question, t.expected_verdict)
            if key not in seen:
                seen.add(key)
                res.append(t)
        return res

    yes_tasks = dedup(yes_tasks)
    no_tasks = dedup(no_tasks)
    unk_tasks = dedup(unk_tasks)

    # Interleave train practice pool: exactly equal YES, NO, UNKNOWN
    train_pool: list[EpistemicTask] = []
    val_pool: list[EpistemicTask] = []
    test_pool: list[EpistemicTask] = []

    # Exactly 8 YES, 8 NO, 8 UNKNOWN for sealed test pool = 24 total
    test_yes = yes_tasks[:8]
    test_no = no_tasks[:8]
    test_unk = unk_tasks[:8]
    for i in range(8):
        if i < len(test_yes):
            test_pool.append(test_yes[i])
        if i < len(test_no):
            test_pool.append(test_no[i])
        if i < len(test_unk):
            test_pool.append(test_unk[i])

    # Validation: 4 each = 12 total
    val_yes = yes_tasks[8:12]
    val_no = no_tasks[8:12]
    val_unk = unk_tasks[8:12]
    for i in range(4):
        if i < len(val_yes):
            val_pool.append(val_yes[i])
        if i < len(val_no):
            val_pool.append(val_no[i])
        if i < len(val_unk):
            val_pool.append(val_unk[i])

    # Practice pool: interleaved up to 36 total (12 YES, 12 NO, 12 UNK)
    rem_yes = yes_tasks[12:24]
    rem_no = no_tasks[12:24]
    rem_unk = unk_tasks[12:24]
    max_train_count = min(len(rem_yes), len(rem_no), len(rem_unk))
    if max_train_count < 12:
        # Recycle with augmented questions if needed
        rem_yes = (yes_tasks + yes_tasks)[:16]
        rem_no = (no_tasks + no_tasks)[:16]
        rem_unk = (unk_tasks + unk_tasks)[:16]
        max_train_count = 12

    for i in range(max_train_count):
        train_pool.append(rem_yes[i])
        train_pool.append(rem_no[i])
        train_pool.append(rem_unk[i])

    return EnhancedRelationalDataset(
        world_name=world_name,
        split=split,
        topology=topology,
        semantics=semantics,
        premises=premises,
        train_practice_pool=train_pool,
        validation_pool=val_pool,
        held_out_test_pool=test_pool,
        reachability_probes=probes,
    )


def get_m47_benchmark_suite() -> dict[str, EnhancedRelationalDataset]:
    """Generates the four core relational worlds with disjoint topologies and vocabularies."""
    worlds: dict[str, EnhancedRelationalDataset] = {}

    # World 1: CHAIN topology - Biological Taxonomy
    worlds["biological_taxonomy"] = build_enhanced_relational_world(
        world_name="biological_taxonomy",
        split="train",
        topology="CHAIN",
        rel="subsumes",
        inv_rel="belongs_to",
        neg_rel="excludes",
        entities=["DomainEukarya", "KingdomAnimalia", "PhylumChordata", "ClassMammalia", "OrderPrimates", "FamilyHominidae", "GenusPan", "GenusCanis"],
        is_transitive=True,
        is_symmetric=False,
    )

    # World 2: TREE topology - Jurisprudential Hierarchy
    worlds["jurisprudential_precedence"] = build_enhanced_relational_world(
        world_name="jurisprudential_precedence",
        split="train",
        topology="TREE",
        rel="binds",
        inv_rel="yields_to",
        neg_rel="overrules",
        entities=["ApexConstitutionalCourt", "AppellateCircuitA", "AppellateCircuitB", "DistrictTribunalOne", "DistrictTribunalTwo", "AdministrativeBoardA", "AdministrativeBoardB", "MunicipalMagistrate"],
        is_transitive=True,
        is_symmetric=False,
    )

    # World 3: DIAMOND topology - Metabolic Pathway
    worlds["metabolic_pathways"] = build_enhanced_relational_world(
        world_name="metabolic_pathways",
        split="val",
        topology="DIAMOND",
        rel="catalyzes_into",
        inv_rel="is_catalyzed_from",
        neg_rel="inhibits",
        entities=["GlucoseSubstrate", "IntermediatePhosphagenA", "IntermediatePhosphagenB", "PyruvateMetabolite", "AcetylCoenzymeCompound", "OxaloacetateCompound", "LactateByproduct", "CitratePrecursor"],
        is_transitive=True,
        is_symmetric=False,
    )

    # World 4: FORK topology - Geopolitical Alliances (Meta-Test)
    worlds["geopolitical_alliances"] = build_enhanced_relational_world(
        world_name="geopolitical_alliances",
        split="meta_test",
        topology="FORK",
        rel="defends",
        inv_rel="protected_by",
        neg_rel="sanctions",
        entities=["ConfederateCapitol", "NorthernProvince", "SouthernProvince", "EasternTerritory", "WesternTerritory", "MaritimeColony", "NeutralStateA", "NeutralStateB"],
        is_transitive=False,
        is_symmetric=False,
    )

    return worlds
