"""Synthetic Relational World & Deterministic Graph Reasoner for M40.

Provides an independent synthetic relational universe with invented entities
and formal relation semantics (transitive, asymmetric, symmetric, non-transitive).
Generates multi-hop reasoning tasks, counterfactual transfer tasks, distractors,
and abstention queries with ground-truth deductions verified by graph algorithms.

The graph solver is strictly an external ground-truth oracle and dataset builder;
it is never injected into neural-only model evaluation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RelationSemantics(str, Enum):
    """Semantic properties governing relational deduction."""

    TRANSITIVE = "transitive"  # A -> B and B -> C implies A -> C
    ASYMMETRIC = "asymmetric"  # A -> B implies not (B -> A)
    SYMMETRIC = "symmetric"  # A -> B implies B -> A
    NON_TRANSITIVE = "non_transitive"  # A -> B and B -> C does NOT imply A -> C


@dataclass(frozen=True)
class RelationalFact:
    """A single relational proposition in the synthetic world."""

    subject: str
    relation: str
    object: str
    semantics: RelationSemantics = RelationSemantics.TRANSITIVE

    def to_natural_language(self) -> str:
        """Convert triplet into natural language prose."""
        rel_clean = self.relation.replace("_", " ")
        return f"{self.subject} {rel_clean} {self.object}."


@dataclass
class DeductionResult:
    """Outcome of a graph deduction with intermediate path trace."""

    verdict: str  # "yes", "no", "abstain"
    reasoning_steps: list[str] = field(default_factory=list)
    shortest_path: list[str] = field(default_factory=list)


class DeterministicGraphSolver:
    """External deterministic graph solver for ground-truth verification."""

    def __init__(self) -> None:
        self.edges: dict[str, dict[str, list[tuple[str, RelationSemantics]]]] = {}
        self.all_entities: set[str] = set()

    def add_fact(self, fact: RelationalFact) -> None:
        """Add a verified fact to the internal knowledge graph."""
        self.all_entities.add(fact.subject)
        self.all_entities.add(fact.object)

        if fact.relation not in self.edges:
            self.edges[fact.relation] = {}

        if fact.subject not in self.edges[fact.relation]:
            self.edges[fact.relation][fact.subject] = []

        self.edges[fact.relation][fact.subject].append((fact.object, fact.semantics))

        # If symmetric, add inverse edge
        if fact.semantics == RelationSemantics.SYMMETRIC:
            if fact.object not in self.edges[fact.relation]:
                self.edges[fact.relation][fact.object] = []
            self.edges[fact.relation][fact.object].append((fact.subject, fact.semantics))

    def evaluate_query(
        self,
        query_subject: str,
        relation: str,
        query_object: str,
        semantics: RelationSemantics = RelationSemantics.TRANSITIVE,
    ) -> DeductionResult:
        """Evaluate whether query_subject relates to query_object under semantic rules."""
        if query_subject not in self.all_entities or query_object not in self.all_entities:
            return DeductionResult(
                verdict="abstain",
                reasoning_steps=[f"Entity '{query_subject}' or '{query_object}' is unknown in the premise graph."],
            )

        if relation not in self.edges:
            return DeductionResult(
                verdict="abstain",
                reasoning_steps=[f"Relation '{relation}' is not present in premises."],
            )

        # 1. Direct edge check
        adj = self.edges[relation].get(query_subject, [])
        for nxt, sem in adj:
            if nxt == query_object:
                rel_clean = relation.replace("_", " ")
                return DeductionResult(
                    verdict="yes",
                    reasoning_steps=[f"Direct premise states {query_subject} {rel_clean} {query_object}."],
                    shortest_path=[query_subject, query_object],
                )

        # 2. Check asymmetry conflict (e.g. if query asks if B -> A when A -> B and relation is asymmetric)
        if semantics == RelationSemantics.ASYMMETRIC:
            reverse_adj = self.edges[relation].get(query_object, [])
            for nxt, sem in reverse_adj:
                if nxt == query_subject:
                    return DeductionResult(
                        verdict="no",
                        reasoning_steps=[
                            f"Relation '{relation}' is strictly asymmetric. "
                            f"Premise confirms {query_object} relates to {query_subject}, so reverse is false."
                        ],
                        shortest_path=[query_object, query_subject],
                    )

        # 3. Transitive path traversal (BFS)
        if semantics == RelationSemantics.TRANSITIVE:
            queue: deque[tuple[str, list[str]]] = deque([(query_subject, [query_subject])])
            visited = {query_subject}

            while queue:
                curr, path = queue.popleft()
                for neighbor, sem in self.edges[relation].get(curr, []):
                    if neighbor == query_object:
                        full_path = path + [neighbor]
                        steps = []
                        for i in range(len(full_path) - 1):
                            steps.append(
                                f"Step {i+1}: {full_path[i]} {relation.replace('_', ' ')} {full_path[i+1]}."
                            )
                        steps.append(
                            f"Conclusion: By transitive deduction across {len(full_path)-1} steps, "
                            f"{query_subject} {relation.replace('_', ' ')} {query_object}."
                        )
                        return DeductionResult(
                            verdict="yes",
                            reasoning_steps=steps,
                            shortest_path=full_path,
                        )
                    if neighbor not in visited and sem == RelationSemantics.TRANSITIVE:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

        # 4. If query is non-transitive, multi-hop does not apply
        if semantics == RelationSemantics.NON_TRANSITIVE:
            return DeductionResult(
                verdict="no",
                reasoning_steps=[
                    f"Relation '{relation}' is non-transitive; intermediate links do not entail direct connection."
                ],
            )

        # 5. No connecting path found between known entities
        return DeductionResult(
            verdict="no",
            reasoning_steps=[f"No valid relational deduction connects {query_subject} to {query_object}."],
        )


class SyntheticRelationalUniverse:
    """Generator for synthetic worlds with invented entities and arbitrary rules."""

    STANDARD_INVENTED_ENTITIES: list[str] = [
        "Zorblax",
        "Nexus-4",
        "Vortice",
        "Krypton-9",
        "Epsilon-3",
        "Pyron-1",
        "Celestia",
        "Quasar-5",
        "Nebula-8",
        "Astraea",
        "Solon-X",
        "Thalos",
    ]

    COUNTERFACTUAL_ENTITIES: list[str] = [
        "Glapton",
        "Xyphos-7",
        "Morvan-Prime",
        "Zephyr-12",
        "Oryx-9",
        "Bravon-3",
    ]

    @staticmethod
    def build_deduction_prompt(
        premises: list[str],
        question: str,
        include_reasoning: bool = False,
        intermediate_reasoning: str | None = None,
        expected_answer: str | None = None,
    ) -> str:
        """Format premises, question, and optional reasoning tokens into a canonical prompt."""
        premise_text = " ".join(premises)
        prompt = f"[PREMISES] {premise_text} [QUESTION] {question}"
        if include_reasoning and intermediate_reasoning and expected_answer:
            prompt += f" [THINK] {intermediate_reasoning} [ANSWER] {expected_answer}"
        elif expected_answer:
            prompt += f" [ANSWER] {expected_answer}"
        return prompt
