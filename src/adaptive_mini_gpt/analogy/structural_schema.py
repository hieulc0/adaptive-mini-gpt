"""Phase 3: Minimal Structural Analogy Representation and Relational Schema Transfer.

Represents verified experiences by their underlying relational structure independently of entity names:
  - Entity role abstraction: Subject (S), Intermediate (I), Object (O), Anchor (A).
  - Directed relation properties: is_transitive, is_symmetric, is_directional.
  - Canonical Graph Signatures:
      * CHAIN_1HOP: S -> O
      * CHAIN_2HOP: S -> I -> O entails S -> O
      * CHAIN_3HOP: S -> I1 -> I2 -> O entails S -> O
      * ASYMMETRIC_REVERSAL: S -> O entails NOT (O -> S)
      * NONTRANSITIVE_NEGATIVE: Non-reachable downstream node
      * UNGROUNDED_MISSING: Entity absent from premise graph
      * CONTRADICTORY_CYCLE: Self-loop or circular inconsistency
  - Deterministic Structural Matcher:
      Identifies structural isomorphism between tasks across completely disjoint vocabularies.
"""
from __future__ import annotations

import collections
import hashlib
from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.meta.benchmark_suite import MetaTask


@dataclass
class RelationProperties:
    name: str
    is_transitive: bool = True
    is_symmetric: bool = False
    is_directional: bool = True

    def is_compatible_with(self, other: RelationProperties) -> bool:
        return (
            self.is_transitive == other.is_transitive
            and self.is_symmetric == other.is_symmetric
            and self.is_directional == other.is_directional
        )


@dataclass
class CanonicalGraphSignature:
    pattern_type: str  # "CHAIN_1HOP", "CHAIN_2HOP", "CHAIN_3HOP", "ASYMMETRIC_REVERSAL", "UNGROUNDED", "CONTRADICTORY"
    num_nodes: int
    num_edges: int
    path_length: int
    involves_inversion: bool
    is_grounded: bool

    @property
    def signature_id(self) -> str:
        key = f"{self.pattern_type}::{self.num_nodes}::{self.num_edges}::{self.path_length}::{self.involves_inversion}::{self.is_grounded}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


@dataclass
class StructuralAnalogyExperience:
    """An episodic memory enriched with canonical structural graph signature."""

    experience_id: str
    original_family: str
    relation_name: str
    relation_properties: RelationProperties
    canonical_signature: CanonicalGraphSignature
    abstract_rule: str
    exemplar_premises: list[str]
    exemplar_question: str
    verified_verdict: str  # "YES", "NO", "UNKNOWN"
    usefulness_score: float = 1.0


class DeterministicStructuralMatcher:
    """Extracts structural roles and canonical graph signatures from relational tasks."""

    @staticmethod
    def extract_signature(task: MetaTask, rel_props: RelationProperties | None = None) -> CanonicalGraphSignature:
        # Parse graph from premises
        adj: dict[str, set[str]] = collections.defaultdict(set)
        all_nodes: set[str] = set()

        for p in task.premises:
            parts = p.strip().split()
            if len(parts) >= 3:
                s, r, o = parts[0], parts[1], parts[2]
                adj[s].add(o)
                all_nodes.add(s)
                all_nodes.add(o)

        # Parse question query entities
        # Assuming format "Does S rel O?" or similar
        words = task.question.replace("?", "").replace("Does ", "").replace("Is it verified that ", "").split()
        q_subj = words[0] if len(words) > 0 else ""
        q_obj = words[-1] if len(words) > 1 else ""

        # Check ungrounded
        if q_subj not in all_nodes or q_obj not in all_nodes:
            return CanonicalGraphSignature(
                pattern_type="UNGROUNDED",
                num_nodes=len(all_nodes),
                num_edges=sum(len(v) for v in adj.values()),
                path_length=0,
                involves_inversion=False,
                is_grounded=False,
            )

        # Check self-loop / contradictory
        if q_subj == q_obj:
            return CanonicalGraphSignature(
                pattern_type="CONTRADICTORY",
                num_nodes=len(all_nodes),
                num_edges=sum(len(v) for v in adj.values()),
                path_length=0,
                involves_inversion=False,
                is_grounded=True,
            )

        # Find shortest path from q_subj to q_obj
        visited: set[str] = set()
        queue = collections.deque([(q_subj, 0)])
        forward_dist = -1

        while queue:
            curr, d = queue.popleft()
            if curr == q_obj and d > 0:
                forward_dist = d
                break
            if curr not in visited:
                visited.add(curr)
                for nbr in adj[curr]:
                    queue.append((nbr, d + 1))

        # Check reverse reachability (for asymmetric reversal)
        rev_visited: set[str] = set()
        rev_queue = collections.deque([(q_obj, 0)])
        reverse_dist = -1
        while rev_queue:
            curr, d = rev_queue.popleft()
            if curr == q_subj and d > 0:
                reverse_dist = d
                break
            if curr not in rev_visited:
                rev_visited.add(curr)
                for nbr in adj[curr]:
                    rev_queue.append((nbr, d + 1))

        if forward_dist == 1:
            pat = "CHAIN_1HOP"
        elif forward_dist == 2:
            pat = "CHAIN_2HOP"
        elif forward_dist >= 3:
            pat = "CHAIN_3HOP"
        elif reverse_dist > 0:
            pat = "ASYMMETRIC_REVERSAL"
        else:
            pat = "NONTRANSITIVE_NEGATIVE"

        return CanonicalGraphSignature(
            pattern_type=pat,
            num_nodes=len(all_nodes),
            num_edges=sum(len(v) for v in adj.values()),
            path_length=max(forward_dist, reverse_dist, 0),
            involves_inversion=(reverse_dist > 0 and forward_dist < 0),
            is_grounded=True,
        )

    @staticmethod
    def match_analogy(
        query_task: MetaTask,
        query_props: RelationProperties,
        candidate_experience: StructuralAnalogyExperience,
    ) -> float:
        """Computes structural analogy score in [0.0, 1.0]."""
        query_sig = DeterministicStructuralMatcher.extract_signature(query_task, query_props)
        cand_sig = candidate_experience.canonical_signature

        # 1. Structural pattern compatibility
        if query_sig.pattern_type != cand_sig.pattern_type:
            return 0.0

        score = 0.5  # Base match for identical graph pattern

        # 2. Relational semantic property compatibility
        if query_props.is_compatible_with(candidate_experience.relation_properties):
            score += 0.3

        # 3. Path length match
        if query_sig.path_length == cand_sig.path_length:
            score += 0.2

        return round(score, 4)

    @staticmethod
    def format_analogical_prompt(
        query_task: MetaTask,
        analogous_experience: StructuralAnalogyExperience,
    ) -> str:
        """Formats in-context prompt mapping abstract structural analogy to the new task."""
        exp_prem = " ".join(analogous_experience.exemplar_premises)
        prompt = (
            f"[ANALOGY_SCHEMA] Pattern: {analogous_experience.canonical_signature.pattern_type} | "
            f"Rule: {analogous_experience.abstract_rule} | "
            f"Ex: [{exp_prem}] {analogous_experience.exemplar_question} -> {analogous_experience.verified_verdict} "
            f"[PREMISES] {' '.join(query_task.premises)} [QUESTION] {query_task.question} [ANSWER]"
        )
        return prompt
