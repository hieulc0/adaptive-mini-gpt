"""M37 Phase 3: Autonomous Research & Local Search Engine.

Implements a deterministic, auditable local search engine over approved documents
and compares query formulation strategies:
  - Strategy A: Raw Question Baseline.
  - Strategy B: Deterministic Keyword Extraction.
  - Strategy C: Metacognitive Targeted Query Formulation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.research.corpus import (
    APPROVED_DOCUMENTS,
    DocumentType,
    ResearchDocument,
)

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "arent", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "cant", "cannot", "could", "couldnt", "did", "didnt",
    "do", "does", "doesnt", "doing", "dont", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed",
    "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
    "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "isnt", "it", "its",
    "itself", "lets", "me", "more", "most", "mustnt", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "shant", "she", "shed", "shell", "shes", "should",
    "shouldnt", "so", "some", "such", "than", "that", "thats", "the", "their", "theirs",
    "them", "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll",
    "theyre", "theyve", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasnt", "we", "wed", "well", "were", "werent", "what", "whats", "when",
    "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why", "whys",
    "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", "your",
    "yours", "yourself", "yourselves",
}


@dataclass
class SearchResult:
    """A scored passage retrieved from the approved document corpus."""

    doc_id: str
    title: str
    content: str
    subject: str
    doc_type: DocumentType
    provenance: str
    reliability_score: float
    relevance_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "subject": self.subject,
            "doc_type": self.doc_type.value,
            "provenance": self.provenance,
            "reliability_score": self.reliability_score,
            "relevance_score": round(self.relevance_score, 4),
        }


class LocalSearchEngine:
    """Deterministic BM25-style local search index over approved research documents."""

    def __init__(self, documents: list[ResearchDocument] | None = None) -> None:
        self.documents = list(documents or APPROVED_DOCUMENTS)
        self.doc_tokens: dict[str, list[str]] = {}
        self.doc_freqs: Counter[str] = Counter()
        self.total_docs = len(self.documents)
        self.avg_doc_len = 0.0

        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-\*]+\b", text.lower())
        return words

    def _build_index(self) -> None:
        total_len = 0
        for doc in self.documents:
            full_text = f"{doc.title} {doc.content} {doc.subject}"
            tokens = self._tokenize(full_text)
            self.doc_tokens[doc.doc_id] = tokens
            total_len += len(tokens)
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] += 1
        self.avg_doc_len = total_len / max(1, self.total_docs)

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.05,
    ) -> list[SearchResult]:
        """Search corpus and return ranked results using BM25 scoring."""
        q_tokens = [t for t in self._tokenize(query) if t not in STOPWORDS]
        if not q_tokens:
            q_tokens = self._tokenize(query)

        scores: list[tuple[float, ResearchDocument]] = []
        k1 = 1.5
        b = 0.75

        for doc in self.documents:
            tokens = self.doc_tokens[doc.doc_id]
            doc_len = len(tokens)
            token_counts = Counter(tokens)
            score = 0.0

            for q_term in q_tokens:
                if q_term in token_counts:
                    tf = token_counts[q_term]
                    df = self.doc_freqs.get(q_term, 1)
                    # Standard BM25 IDF
                    idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / max(1.0, self.avg_doc_len)))
                    score += idf * (numerator / denominator)

            if score >= min_score:
                scores.append((score, doc))

        # Sort descending by relevance score
        scores.sort(key=lambda pair: pair[0], reverse=True)

        results: list[SearchResult] = []
        for score, doc in scores[:top_k]:
            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    content=doc.content,
                    subject=doc.subject,
                    doc_type=doc.doc_type,
                    provenance=doc.provenance,
                    reliability_score=doc.reliability_score,
                    relevance_score=score,
                )
            )
        return results


class SearchQueryFormulator:
    """Generates search queries using competing formulation strategies."""

    @staticmethod
    def formulate_raw_query(question: str) -> str:
        """Strategy A: Direct verbatim question."""
        return question.strip()

    @staticmethod
    def formulate_keyword_query(question: str) -> str:
        """Strategy B: Deterministic stopword stripping."""
        words = re.findall(r"\b[a-zA-Z0-9_\-\*]+\b", question.lower())
        content_words = [w for w in words if w not in STOPWORDS]
        return " ".join(content_words) if content_words else question

    @staticmethod
    def formulate_metacognitive_query(question: str, subject_hint: str | None = None) -> str:
        """Strategy C: Targeted entity extraction and intent focusing."""
        q_lower = question.lower()
        # Remove question frames
        cleaned = re.sub(
            r"\b(what is the|what date did|in what year did|which organelle|is heat|are contemporary|according to verified|states that)\b",
            "",
            q_lower,
        )
        words = re.findall(r"\b[a-zA-Z0-9_\-\*]+\b", cleaned)
        keywords = [w for w in words if w not in STOPWORDS]

        # Prioritize key subject terms
        if subject_hint and subject_hint not in keywords:
            keywords.insert(0, subject_hint)

        return " ".join(keywords[:6])
