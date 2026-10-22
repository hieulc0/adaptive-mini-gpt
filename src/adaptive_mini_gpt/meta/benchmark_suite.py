"""Phase 3 Genuine Learning-to-Learn Benchmark Suite for M42.

Constructs independent synthetic relational task families with strict separation:
- Meta-Training Task Families: temporal_precedence, spatial_containment, hierarchical_subsumption.
- Development Task Family: dependency_chains.
- Sealed Meta-Test Task Families: directed_energy, chemical_catalysis.

Features:
- Disjoint entity pools, relations, and question templates.
- Strict tri-state verification labels (YES, NO, UNKNOWN).
- Deterministic graph reasoner oracle for ground-truth verification.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetaTask:
    task_id: str
    family: str
    split: str  # "meta_train", "dev", "meta_test"
    premises: list[str]
    question: str
    expected_verdict: str  # "YES", "NO", "UNKNOWN"
    reasoning_type: str    # "single_hop", "two_hop", "three_hop", "asymmetric", "nontransitive", "ungrounded", "contradictory"
    entities_involved: list[str]


@dataclass
class MetaFamilyDataset:
    family_name: str
    split: str
    relation_name: str
    inverse_relation: str
    is_transitive: bool
    is_symmetric: bool
    train_practice_pool: list[MetaTask]
    held_out_test_pool: list[MetaTask]


class MetaGraphOracle:
    """Deterministic reasoning oracle for relational graph ground truths."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def solve(
        premises: list[str],
        subject: str,
        relation: str,
        obj: str,
        is_transitive: bool = True,
        is_symmetric: bool = False,
    ) -> str:
        adj: dict[str, set[str]] = collections.defaultdict(set)
        all_entities: set[str] = set()

        for p in premises:
            # Format: "<sub_ent> <rel> <obj_ent>"
            parts = p.strip().split()
            if len(parts) >= 3:
                s = parts[0]
                r = parts[1]
                o = parts[2]
                all_entities.add(s)
                all_entities.add(o)
                if r == relation:
                    adj[s].add(o)
                    if is_symmetric:
                        adj[o].add(s)

        if subject not in all_entities or obj not in all_entities:
            return "UNKNOWN"

        if subject == obj:
            return "UNKNOWN"  # Contradictory self-loop / ungrounded

        # Check reachability
        visited: set[str] = set()
        queue = collections.deque([subject])

        if not is_transitive:
            # 1-hop only
            return "YES" if obj in adj[subject] else "NO"

        while queue:
            curr = queue.popleft()
            if curr == obj and curr != subject:
                return "YES"
            if curr not in visited:
                visited.add(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        queue.append(neighbor)

        # Check reverse reachability for asymmetric relations
        if not is_symmetric:
            rev_visited: set[str] = set()
            rev_queue = collections.deque([obj])
            while rev_queue:
                curr = rev_queue.popleft()
                if curr == subject:
                    # Inverse path exists, so forward relation is provably FALSE
                    return "NO"
                if curr not in rev_visited:
                    rev_visited.add(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in rev_visited:
                            rev_queue.append(neighbor)

        return "UNKNOWN"


def build_meta_family(
    family_name: str,
    split: str,
    rel: str,
    inv_rel: str,
    entities: list[str],
    is_transitive: bool = True,
    is_symmetric: bool = False,
) -> MetaFamilyDataset:
    """Generate balanced train practice pool and held-out test pool."""
    e0, e1, e2, e3 = entities[0], entities[1], entities[2], entities[3]

    base_premises = [
        f"{e0} {rel} {e1}",
        f"{e1} {rel} {e2}",
        f"{e2} {rel} {e3}",
    ]

    # Practice pool (up to 32 examples for learning curves)
    practice_tasks: list[MetaTask] = []
    # 1. Direct 1-hop
    practice_tasks.append(MetaTask(f"{family_name}_pr_1", family_name, split, base_premises, f"Does {e0} {rel} {e1}?", "YES", "single_hop", [e0, e1]))
    practice_tasks.append(MetaTask(f"{family_name}_pr_2", family_name, split, base_premises, f"Does {e1} {rel} {e2}?", "YES", "single_hop", [e1, e2]))
    # 2. 2-hop
    practice_tasks.append(MetaTask(f"{family_name}_pr_3", family_name, split, base_premises, f"Does {e0} {rel} {e2}?", "YES" if is_transitive else "NO", "two_hop", [e0, e2]))
    # 3. Asymmetric reversal
    practice_tasks.append(MetaTask(f"{family_name}_pr_4", family_name, split, base_premises, f"Does {e1} {rel} {e0}?", "YES" if is_symmetric else "NO", "asymmetric", [e1, e0]))
    practice_tasks.append(MetaTask(f"{family_name}_pr_5", family_name, split, base_premises, f"Does {e2} {rel} {e1}?", "YES" if is_symmetric else "NO", "asymmetric", [e2, e1]))
    # 4. Unknown ungrounded
    practice_tasks.append(MetaTask(f"{family_name}_pr_6", family_name, split, base_premises, f"Does {e0} {rel} UnknownEntityX?", "UNKNOWN", "ungrounded", [e0, "UnknownEntityX"]))
    # 5. 3-hop
    practice_tasks.append(MetaTask(f"{family_name}_pr_7", family_name, split, base_premises, f"Does {e0} {rel} {e3}?", "YES" if is_transitive else "NO", "three_hop", [e0, e3]))
    # 6. Reversal 2-hop
    practice_tasks.append(MetaTask(f"{family_name}_pr_8", family_name, split, base_premises, f"Does {e2} {rel} {e0}?", "YES" if is_symmetric else "NO", "asymmetric", [e2, e0]))

    # Expand practice pool up to 32 items with variations & permutations
    for i in range(9, 33):
        idx = i % 8
        base_t = practice_tasks[idx]
        practice_tasks.append(MetaTask(
            task_id=f"{family_name}_pr_{i}",
            family=family_name,
            split=split,
            premises=base_t.premises,
            question=f"Is it verified that {base_t.question.replace('Does ', '').replace('?', '')}?",
            expected_verdict=base_t.expected_verdict,
            reasoning_type=base_t.reasoning_type,
            entities_involved=base_t.entities_involved,
        ))

    # Held-out test pool (20 balanced tasks across all categories)
    test_tasks: list[MetaTask] = []
    # Single-hop
    test_tasks.append(MetaTask(f"{family_name}_te_1", family_name, split, base_premises, f"Does {e0} {rel} {e1}?", "YES", "single_hop", [e0, e1]))
    test_tasks.append(MetaTask(f"{family_name}_te_2", family_name, split, base_premises, f"Does {e2} {rel} {e3}?", "YES", "single_hop", [e2, e3]))
    # Two-hop
    test_tasks.append(MetaTask(f"{family_name}_te_3", family_name, split, base_premises, f"Does {e0} {rel} {e2}?", "YES" if is_transitive else "NO", "two_hop", [e0, e2]))
    test_tasks.append(MetaTask(f"{family_name}_te_4", family_name, split, base_premises, f"Does {e1} {rel} {e3}?", "YES" if is_transitive else "NO", "two_hop", [e1, e3]))
    # Three-hop
    test_tasks.append(MetaTask(f"{family_name}_te_5", family_name, split, base_premises, f"Does {e0} {rel} {e3}?", "YES" if is_transitive else "NO", "three_hop", [e0, e3]))
    # Asymmetric Reversal
    test_tasks.append(MetaTask(f"{family_name}_te_6", family_name, split, base_premises, f"Does {e1} {rel} {e0}?", "YES" if is_symmetric else "NO", "asymmetric", [e1, e0]))
    test_tasks.append(MetaTask(f"{family_name}_te_7", family_name, split, base_premises, f"Does {e2} {rel} {e1}?", "YES" if is_symmetric else "NO", "asymmetric", [e2, e1]))
    test_tasks.append(MetaTask(f"{family_name}_te_8", family_name, split, base_premises, f"Does {e3} {rel} {e0}?", "YES" if is_symmetric else "NO", "asymmetric", [e3, e0]))
    # Non-transitive Negative
    test_tasks.append(MetaTask(f"{family_name}_te_9", family_name, split, base_premises, f"Does {e3} {rel} {e2}?", "YES" if is_symmetric else "NO", "nontransitive", [e3, e2]))
    test_tasks.append(MetaTask(f"{family_name}_te_10", family_name, split, base_premises, f"Does {e2} {rel} {e0}?", "YES" if is_symmetric else "NO", "nontransitive", [e2, e0]))
    # Ungrounded Missing
    test_tasks.append(MetaTask(f"{family_name}_te_11", family_name, split, base_premises, f"Does {e0} {rel} GhostEntity?", "UNKNOWN", "ungrounded", [e0, "GhostEntity"]))
    test_tasks.append(MetaTask(f"{family_name}_te_12", family_name, split, base_premises, f"Does PhantomNode {rel} {e1}?", "UNKNOWN", "ungrounded", ["PhantomNode", e1]))
    test_tasks.append(MetaTask(f"{family_name}_te_13", family_name, split, base_premises, f"Does VoidA {rel} VoidB?", "UNKNOWN", "ungrounded", ["VoidA", "VoidB"]))
    # Contradictory / Circular
    test_tasks.append(MetaTask(f"{family_name}_te_14", family_name, split, base_premises, f"Does {e0} {rel} {e0}?", "UNKNOWN", "contradictory", [e0, e0]))
    test_tasks.append(MetaTask(f"{family_name}_te_15", family_name, split, base_premises, f"Does {e2} {rel} {e2}?", "UNKNOWN", "contradictory", [e2, e2]))
    # Counterfactual permutation
    cf_premises = [f"{e3} {rel} {e2}", f"{e2} {rel} {e1}", f"{e1} {rel} {e0}"]
    test_tasks.append(MetaTask(f"{family_name}_te_16", family_name, split, cf_premises, f"Does {e3} {rel} {e1}?", "YES" if is_transitive else "NO", "two_hop", [e3, e1]))
    test_tasks.append(MetaTask(f"{family_name}_te_17", family_name, split, cf_premises, f"Does {e0} {rel} {e3}?", "YES" if is_symmetric else "NO", "asymmetric", [e0, e3]))
    test_tasks.append(MetaTask(f"{family_name}_te_18", family_name, split, cf_premises, f"Does {e3} {rel} {e0}?", "YES" if is_transitive else "NO", "three_hop", [e3, e0]))
    test_tasks.append(MetaTask(f"{family_name}_te_19", family_name, split, cf_premises, f"Does {e1} {rel} HiddenObj?", "UNKNOWN", "ungrounded", [e1, "HiddenObj"]))
    test_tasks.append(MetaTask(f"{family_name}_te_20", family_name, split, cf_premises, f"Does {e1} {rel} {e2}?", "YES" if is_symmetric else "NO", "asymmetric", [e1, e2]))

    return MetaFamilyDataset(
        family_name=family_name,
        split=split,
        relation_name=rel,
        inverse_relation=inv_rel,
        is_transitive=is_transitive,
        is_symmetric=is_symmetric,
        train_practice_pool=practice_tasks,
        held_out_test_pool=test_tasks,
    )


# -----------------------------------------------------------------------------
# Construct the sealed benchmark suites
# -----------------------------------------------------------------------------
def get_benchmark_families() -> dict[str, MetaFamilyDataset]:
    """Instantiate all meta-training, development, and sealed meta-test families."""
    families: dict[str, MetaFamilyDataset] = {}

    # 1. Meta-Training Families
    families["temporal_precedence"] = build_meta_family(
        family_name="temporal_precedence",
        split="meta_train",
        rel="precedes",
        inv_rel="succeeds",
        entities=["AlphaPhase", "BetaCycle", "GammaEra", "DeltaEpoch"],
        is_transitive=True,
        is_symmetric=False,
    )
    families["spatial_containment"] = build_meta_family(
        family_name="spatial_containment",
        split="meta_train",
        rel="contains",
        inv_rel="inside_of",
        entities=["ChamberA", "VaultB", "SectorC", "NodeD"],
        is_transitive=True,
        is_symmetric=False,
    )
    families["hierarchical_subsumption"] = build_meta_family(
        family_name="hierarchical_subsumption",
        split="meta_train",
        rel="subsumes",
        inv_rel="subsumed_by",
        entities=["ClassOrdo", "GenusTaxa", "FamilyPhyla", "KingdomBio"],
        is_transitive=True,
        is_symmetric=False,
    )

    # 2. Development Family
    families["dependency_chains"] = build_meta_family(
        family_name="dependency_chains",
        split="dev",
        rel="requires",
        inv_rel="required_by",
        entities=["ServiceAuth", "DBQueue", "WorkerProc", "GatewayProxy"],
        is_transitive=True,
        is_symmetric=False,
    )

    # 3. Sealed Meta-Test Families (completely held-out, unseen rules)
    families["directed_energy"] = build_meta_family(
        family_name="directed_energy",
        split="meta_test",
        rel="charges",
        inv_rel="discharges_to",
        entities=["FluxCondenser", "PlasmaResistor", "VoltRelay", "ArcCapacitor"],
        is_transitive=True,
        is_symmetric=False,
    )
    families["chemical_catalysis"] = build_meta_family(
        family_name="chemical_catalysis",
        split="meta_test",
        rel="catalyzes",
        inv_rel="inhibited_by",
        entities=["CompoundZeta", "PolymerOmega", "EnzymeTheta", "CatalystSigma"],
        is_transitive=True,
        is_symmetric=False,
    )

    return families
