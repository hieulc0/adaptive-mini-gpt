"""Phase 3: Memory Retrieval and Selective Policy Engine.

Implements and benchmarks memory retrieval mechanisms:
  - Method A: No Memory (Zero retrieval baseline).
  - Method B: Keyword / Exact-Pattern Retrieval (Lexical match on entities, relations).
  - Method C: Similarity-Based Retrieval (Token set / Jaccard overlap on task context).
  - Method D: Selective Multi-Factor Retrieval (Relevance + Usefulness + Recency + Quality).
  - Baseline Full: Uncurated dump up to context budget.
  - Baseline Recency: Most recent experiences only.

Includes a noise resistance benchmark testing retrieval against candidate pools containing
irrelevant distractors, contradictory statements, and outdated/superseded memories.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.memory.episodic_memory import EpisodicMemoryBank, LearningExperience


@dataclass
class RetrievalScoringWeights:
    w_relevance: float = 0.45
    w_usefulness: float = 0.25
    w_recency: float = 0.15
    w_quality: float = 0.15


@dataclass
class RetrievalBenchmarkResult:
    method_name: str
    total_queries: int
    precision: float
    recall: float
    noise_rejection_rate: float
    contradiction_rejection_rate: float
    mean_retrieved_count: float
    mean_scoring_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "total_queries": self.total_queries,
            "precision": self.precision,
            "recall": self.recall,
            "noise_rejection_rate": self.noise_rejection_rate,
            "contradiction_rejection_rate": self.contradiction_rejection_rate,
            "mean_retrieved_count": self.mean_retrieved_count,
            "mean_scoring_latency_ms": self.mean_scoring_latency_ms,
        }


class MemoryRetriever:
    """Retrieves and ranks relevant verified experiences from EpisodicMemoryBank."""

    def __init__(
        self,
        memory_bank: EpisodicMemoryBank,
        weights: RetrievalScoringWeights | None = None,
        max_exemplars: int = 2,
        max_context_chars: int = 256,
    ) -> None:
        self.memory_bank = memory_bank
        self.weights = weights or RetrievalScoringWeights()
        self.max_exemplars = max_exemplars
        self.max_context_chars = max_context_chars

    @staticmethod
    def _extract_tokens(text: str) -> set[str]:
        words = re.findall(r"\b[A-Za-z0-9_]+\b", text.lower())
        # Filter common stopwords
        stopwords = {"is", "a", "the", "in", "to", "of", "and", "or", "what", "does", "relate", "true", "false"}
        return {w for w in words if w not in stopwords}

    def _compute_relevance(self, query_tokens: set[str], exp: LearningExperience) -> float:
        """Compute lexical & semantic token overlap between query and experience."""
        exp_text = f"{' '.join(exp.premises)} {exp.question} {exp.skill_family}"
        exp_tokens = self._extract_tokens(exp_text)
        if not query_tokens or not exp_tokens:
            return 0.0
        intersection = query_tokens.intersection(exp_tokens)
        union = query_tokens.union(exp_tokens)
        return float(len(intersection)) / float(max(1, len(union)))

    def _compute_recency_score(self, exp: LearningExperience, current_episode: int) -> float:
        """Exponential decay based on episode distance."""
        delta = max(0, current_episode - exp.last_accessed_episode)
        return math.exp(-0.15 * delta)

    def _compute_quality_score(self, exp: LearningExperience) -> float:
        """Scores evidence source reliability and penalizes superseded records."""
        if exp.is_superseded:
            return 0.0
        if exp.is_archived:
            return 0.3
        if exp.evidence_source.startswith("verified_oracle:") or exp.evidence_source.startswith("approved_ground_truth:"):
            return 1.0
        return 0.7

    # -------------------------------------------------------------------------
    # Retrieval Methods
    # -------------------------------------------------------------------------

    def retrieve_none(self, query: str, premises: list[str]) -> list[LearningExperience]:
        """Method A: No Memory."""
        return []

    def retrieve_keyword(
        self,
        query: str,
        premises: list[str],
        candidate_pool: list[LearningExperience] | None = None,
    ) -> list[LearningExperience]:
        """Method B: Keyword / Exact-Pattern Retrieval."""
        pool = candidate_pool if candidate_pool is not None else self.memory_bank.get_all_active()
        q_tokens = self._extract_tokens(f"{' '.join(premises)} {query}")

        matches: list[tuple[int, LearningExperience]] = []
        for exp in pool:
            if exp.is_superseded:
                continue
            exp_text = f"{' '.join(exp.premises)} {exp.question}"
            exp_tokens = self._extract_tokens(exp_text)
            overlap = len(q_tokens.intersection(exp_tokens))
            if overlap > 0:
                matches.append((overlap, exp))

        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[: self.max_exemplars]]

    def retrieve_similarity(
        self,
        query: str,
        premises: list[str],
        candidate_pool: list[LearningExperience] | None = None,
    ) -> list[LearningExperience]:
        """Method C: Similarity-Based Retrieval (Jaccard similarity)."""
        pool = candidate_pool if candidate_pool is not None else self.memory_bank.get_all_active()
        q_tokens = self._extract_tokens(f"{' '.join(premises)} {query}")

        scored: list[tuple[float, LearningExperience]] = []
        for exp in pool:
            if exp.is_superseded:
                continue
            relevance = self._compute_relevance(q_tokens, exp)
            if relevance > 0.05:
                scored.append((relevance, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[: self.max_exemplars]]

    def retrieve_selective(
        self,
        query: str,
        premises: list[str],
        current_episode: int = 0,
        candidate_pool: list[LearningExperience] | None = None,
    ) -> list[LearningExperience]:
        """Method D: Selective Multi-Factor Retrieval."""
        pool = candidate_pool if candidate_pool is not None else self.memory_bank.get_all_active()
        q_tokens = self._extract_tokens(f"{' '.join(premises)} {query}")

        scored: list[tuple[float, LearningExperience]] = []
        w = self.weights

        for exp in pool:
            # Completely eliminate superseded memories from active consideration
            if exp.is_superseded:
                continue

            rel = self._compute_relevance(q_tokens, exp)
            # Minimum relevance threshold to avoid noise pollution
            if rel < 0.05:
                continue

            use = exp.usefulness_score
            rec = self._compute_recency_score(exp, current_episode)
            qual = self._compute_quality_score(exp)

            total_score = (
                w.w_relevance * rel
                + w.w_usefulness * use
                + w.w_recency * rec
                + w.w_quality * qual
            )
            scored.append((total_score, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[: self.max_exemplars]]

    def retrieve_full_baseline(
        self,
        query: str,
        premises: list[str],
        candidate_pool: list[LearningExperience] | None = None,
    ) -> list[LearningExperience]:
        """Baseline Full-Memory: Retrieve uncurated candidates up to capacity."""
        pool = candidate_pool if candidate_pool is not None else self.memory_bank.get_all_active()
        # Return all active experiences up to max_exemplars without filtering
        return [e for e in pool if not e.is_superseded][: self.max_exemplars]

    def retrieve_recency_baseline(
        self,
        query: str,
        premises: list[str],
        candidate_pool: list[LearningExperience] | None = None,
    ) -> list[LearningExperience]:
        """Baseline Recency: Retrieve strictly most recent memories regardless of relevance."""
        pool = candidate_pool if candidate_pool is not None else self.memory_bank.get_all_active()
        valid = [e for e in pool if not e.is_superseded]
        valid.sort(key=lambda e: (e.last_accessed_episode, e.timestamp), reverse=True)
        return valid[: self.max_exemplars]

    def format_in_context_prompt(
        self,
        query: str,
        premises: list[str],
        retrieved_experiences: list[LearningExperience],
    ) -> str:
        """Formats the unified prompt with retrieved exemplar context."""
        parts: list[str] = []
        if retrieved_experiences:
            exemplars_text = []
            for exp in retrieved_experiences:
                exp_prem = " ".join(exp.premises)
                exemplars_text.append(f"Ex: [{exp_prem}] Q: {exp.question} -> {exp.verified_answer}")
            parts.append(f"[MEMORY] {' | '.join(exemplars_text)}")

        parts.append(f"[PREMISES] {' '.join(premises)}")
        parts.append(f"[QUESTION] {query}")
        parts.append("[ANSWER]")
        prompt = " ".join(parts)

        # Enforce budget truncation if needed
        if len(prompt) > self.max_context_chars:
            prompt = prompt[: self.max_context_chars]
        return prompt
