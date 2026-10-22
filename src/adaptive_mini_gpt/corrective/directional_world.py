"""M41 Directional Relational World with Tri-State Truth Values (YES, NO, UNKNOWN).

Provides formal semantics for:
  1. Directional Asymmetry: A -> B implies not (B -> A).
  2. Transitivity: Multi-hop reachability along transitive relations.
  3. Non-transitivity: Intermediate links do not entail connection.
  4. Symmetry: Bidirectional entailment.
  5. Calibrated Epistemic Boundaries:
     - Distinguishes between provably FALSE (contradicted by asymmetric premise)
       and UNKNOWN (underdetermined by incomplete or disjoint premise graphs).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TriStateVerdict(str, Enum):
    """Tri-state truth values representing formal epistemic certainty."""

    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"


class RelationCategory(str, Enum):
    """Semantic properties governing directional and compositional deduction."""

    TRANSITIVE_ASYMMETRIC = "transitive_asymmetric"  # e.g. is_inside, precedes
    STRICTLY_ASYMMETRIC = "strictly_asymmetric"  # e.g. powers, directs (non-transitive)
    SYMMETRIC = "symmetric"  # e.g. adjacent_to, mirrors
    NON_TRANSITIVE = "non_transitive"  # e.g. connected_to


@dataclass(frozen=True)
class DirectionalFact:
    """A verified proposition in the directional relational universe."""

    subject: str
    relation: str
    object: str
    category: RelationCategory = RelationCategory.TRANSITIVE_ASYMMETRIC

    def to_natural_language(self) -> str:
        rel_str = self.relation.replace("_", " ")
        return f"{self.subject} {rel_str} {self.object}."


@dataclass
class TriStateDeduction:
    """Deduction result with formal verdict, explanation, and shortest path."""

    verdict: TriStateVerdict
    explanation: str
    reasoning_path: list[str] = field(default_factory=list)


class TriStateGraphReasoner:
    """Deterministic oracle for ground-truth verification and dataset generation."""

    def __init__(self) -> None:
        # relation -> subject -> list of (object, category)
        self.forward_edges: dict[str, dict[str, list[tuple[str, RelationCategory]]]] = {}
        # relation -> object -> list of (subject, category)
        self.reverse_edges: dict[str, dict[str, list[tuple[str, RelationCategory]]]] = {}
        self.all_entities: set[str] = set()

    def add_fact(self, fact: DirectionalFact) -> None:
        """Add a verified directional fact to the knowledge graph."""
        self.all_entities.add(fact.subject)
        self.all_entities.add(fact.object)

        rel = fact.relation
        if rel not in self.forward_edges:
            self.forward_edges[rel] = {}
            self.reverse_edges[rel] = {}

        if fact.subject not in self.forward_edges[rel]:
            self.forward_edges[rel][fact.subject] = []
        self.forward_edges[rel][fact.subject].append((fact.object, fact.category))

        if fact.object not in self.reverse_edges[rel]:
            self.reverse_edges[rel][fact.object] = []
        self.reverse_edges[rel][fact.object].append((fact.subject, fact.category))

        if fact.category == RelationCategory.SYMMETRIC:
            if fact.object not in self.forward_edges[rel]:
                self.forward_edges[rel][fact.object] = []
            self.forward_edges[rel][fact.object].append((fact.subject, fact.category))

    def evaluate_query(
        self,
        query_subject: str,
        relation: str,
        query_object: str,
        expected_category: RelationCategory = RelationCategory.TRANSITIVE_ASYMMETRIC,
    ) -> TriStateDeduction:
        """Determine whether query_subject relates to query_object under formal relation semantics."""
        # 1. Epistemic boundary check: ungrounded entities
        if query_subject not in self.all_entities or query_object not in self.all_entities:
            missing = []
            if query_subject not in self.all_entities:
                missing.append(f"'{query_subject}'")
            if query_object not in self.all_entities:
                missing.append(f"'{query_object}'")
            return TriStateDeduction(
                verdict=TriStateVerdict.UNKNOWN,
                explanation=f"Entity {', '.join(missing)} is ungrounded in premises; truth value is underdetermined.",
            )

        if relation not in self.forward_edges:
            return TriStateDeduction(
                verdict=TriStateVerdict.UNKNOWN,
                explanation=f"Relation '{relation}' has no assertions in premise graph.",
            )

        # 2. Check for explicit cycle contradiction in asymmetric relations
        if expected_category in (RelationCategory.TRANSITIVE_ASYMMETRIC, RelationCategory.STRICTLY_ASYMMETRIC):
            has_forward = any(obj == query_object for obj, _ in self.forward_edges[relation].get(query_subject, []))
            has_reverse = any(obj == query_subject for obj, _ in self.forward_edges[relation].get(query_object, []))
            if has_forward and has_reverse:
                return TriStateDeduction(
                    verdict=TriStateVerdict.UNKNOWN,
                    explanation=f"Premises contain mutual contradictory cycle for asymmetric relation '{relation}'.",
                )

        # 3. Direct forward edge check
        for nxt, cat in self.forward_edges[relation].get(query_subject, []):
            if nxt == query_object:
                return TriStateDeduction(
                    verdict=TriStateVerdict.YES,
                    explanation=f"Direct premise confirms {query_subject} {relation.replace('_', ' ')} {query_object}.",
                    reasoning_path=[query_subject, query_object],
                )

        # 4. Asymmetric reversal check: if query asks B -> A when A -> B exists and relation is asymmetric
        if expected_category in (RelationCategory.TRANSITIVE_ASYMMETRIC, RelationCategory.STRICTLY_ASYMMETRIC):
            for nxt, cat in self.forward_edges[relation].get(query_object, []):
                if nxt == query_subject:
                    return TriStateDeduction(
                        verdict=TriStateVerdict.NO,
                        explanation=(
                            f"Relation '{relation}' is strictly asymmetric. "
                            f"Premise confirms {query_object} relates to {query_subject}, so reverse is provably false."
                        ),
                        reasoning_path=[query_object, query_subject],
                    )

        # 5. Multi-hop transitive deduction (BFS)
        if expected_category == RelationCategory.TRANSITIVE_ASYMMETRIC:
            queue: deque[tuple[str, list[str]]] = deque([(query_subject, [query_subject])])
            visited = {query_subject}

            while queue:
                curr, path = queue.popleft()
                for neighbor, cat in self.forward_edges[relation].get(curr, []):
                    if neighbor == query_object:
                        full_path = path + [neighbor]
                        return TriStateDeduction(
                            verdict=TriStateVerdict.YES,
                            explanation=f"Transitive deduction holds across {len(full_path)-1} hops: {' -> '.join(full_path)}.",
                            reasoning_path=full_path,
                        )
                    if neighbor not in visited and cat == RelationCategory.TRANSITIVE_ASYMMETRIC:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

        # 6. Non-transitive intermediate check: if relation is non-transitive
        if expected_category in (RelationCategory.NON_TRANSITIVE, RelationCategory.STRICTLY_ASYMMETRIC):
            return TriStateDeduction(
                verdict=TriStateVerdict.NO,
                explanation=f"Relation '{relation}' is non-transitive; multi-hop connection does not entail direct link.",
            )

        # 7. Disconnected known entities: neither entailed nor contradicted
        return TriStateDeduction(
            verdict=TriStateVerdict.UNKNOWN,
            explanation=f"Entities '{query_subject}' and '{query_object}' are known but no relational chain connects them.",
        )
