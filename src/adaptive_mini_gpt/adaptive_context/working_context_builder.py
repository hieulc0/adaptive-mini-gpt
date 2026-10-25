"""Working Context Builder & Multi-Hop Memory Retrieval for Milestone M97.

Implements:
1. Temporal Isolation Gating: strictly filters out records created after current_episode.
2. Distractor Filtering & Relevance Scoring: filters irrelevant background noise.
3. Multi-Hop Relational Retrieval: chains multi-step factual queries (Hop 1 -> Hop 2).
4. Compact Working Context Formatting: formats verified retrieved facts into working context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.persistent_memory.knowledge_store import (
    FactualRecord,
    PersistentKnowledgeStore,
)


@dataclass(frozen=True)
class WorkingContextDocument:
    """Formatted working context ready for attention integration."""

    context_text: str
    token_count: int
    retrieved_facts: list[dict[str, Any]]
    distractors_filtered: int
    hops_resolved: int
    temporal_isolation_certified: bool


class WorkingContextBuilder:
    """Constructs focused working context from persistent memory with distractor filtering and multi-hop resolution."""

    def __init__(
        self,
        knowledge_store: PersistentKnowledgeStore,
        max_context_tokens: int = 256,
        relevance_threshold: float = 0.4,
    ) -> None:
        self.store = knowledge_store
        self.max_context_tokens = max_context_tokens
        self.relevance_threshold = relevance_threshold

    def build_working_context(
        self,
        query: str,
        current_episode: int,
        allow_multi_hop: bool = True,
    ) -> WorkingContextDocument:
        """Retrieve relevant facts, filter distractors, resolve multi-hop links, and build context."""
        # Extract targeted search query if preceded by long background text
        search_q = query
        if "query:" in query.lower():
            parts = re.split(r"query:", query, flags=re.IGNORECASE)
            search_q = parts[-1].strip()

        clean_q = search_q.replace("_", " ").replace("-", " ").lower()
        query_tokens = set(t.strip() for t in clean_q.split() if len(t.strip()) > 1)

        # 1. Candidate retrieval including inactive versions for historical temporal reconstruction
        candidates = self.store.search_facts(search_q, include_inactive=True)

        # 2. Strict Temporal Isolation: group by (entity, attribute) and select latest version <= current_episode
        history_map: dict[tuple[str, str], FactualRecord] = {}
        for r in candidates:
            if r.timestamp <= current_episode:
                key = (r.entity, r.attribute)
                if key not in history_map or r.timestamp > history_map[key].timestamp:
                    history_map[key] = r

        valid_temporal_records = list(history_map.values())

        # 3. Distractor filtering via token overlap & entity exact matching
        scored_records: list[tuple[float, FactualRecord]] = []
        distractors_filtered = 0

        for rec in valid_temporal_records:
            rec_text = f"{rec.entity} {rec.attribute} {rec.value}".replace("_", " ").lower()
            rec_tokens = set(t.strip() for t in rec_text.split() if len(t.strip()) > 1)
            intersection = query_tokens.intersection(rec_tokens)
            if not rec_tokens:
                continue
            sim = len(intersection) / len(query_tokens.union(rec_tokens))

            # Entity name exact match boost
            clean_entity = rec.entity.replace("_", " ").lower()
            if clean_entity in clean_q or rec.entity.lower() in clean_q:
                sim += 0.4

            if sim >= self.relevance_threshold:
                scored_records.append((sim, rec))
            else:
                distractors_filtered += 1

        # Sort by relevance descending
        scored_records.sort(key=lambda x: x[0], reverse=True)
        primary_facts = [r for _, r in scored_records]

        # 4. Multi-hop Relational Expansion (Hop 2)
        hops_resolved = 1 if primary_facts else 0
        final_facts = list(primary_facts)

        if allow_multi_hop and primary_facts:
            # Check if any fact's value is an entity with further relevant attributes
            for fact in primary_facts:
                val_str = str(fact.value).lower().strip()
                if "_" in val_str or any(char.isalpha() for char in val_str):
                    # Potential entity reference for Hop 2
                    hop2_candidates = self.store.search_facts(val_str, include_inactive=True)
                    # Filter temporally for Hop 2 as well
                    h2_map: dict[tuple[str, str], FactualRecord] = {}
                    for h2 in hop2_candidates:
                        if h2.timestamp <= current_episode:
                            k2 = (h2.entity, h2.attribute)
                            if k2 not in h2_map or h2.timestamp > h2_map[k2].timestamp:
                                h2_map[k2] = h2
                    for h2 in h2_map.values():
                        if h2.record_id not in [f.record_id for f in final_facts]:
                            clean_h2_attr = h2.attribute.replace("_", " ").lower()
                            if clean_h2_attr in clean_q or any(tok in clean_h2_attr for tok in query_tokens):
                                final_facts.insert(0, h2)
                            else:
                                final_facts.append(h2)
                            hops_resolved = max(hops_resolved, 2)

        # 5. Format compact working context text
        context_lines: list[str] = []
        retrieved_dict_list: list[dict[str, Any]] = []

        for fact in final_facts:
            line = f"Fact: {fact.entity}.{fact.attribute} = {fact.value}"
            context_lines.append(line)
            retrieved_dict_list.append({
                "id": fact.record_id,
                "entity": fact.entity,
                "attribute": fact.attribute,
                "value": fact.value,
                "timestamp": fact.timestamp,
            })

        context_text = "\n".join(context_lines)
        token_count = len(context_text.split())

        return WorkingContextDocument(
            context_text=context_text,
            token_count=token_count,
            retrieved_facts=retrieved_dict_list,
            distractors_filtered=distractors_filtered,
            hops_resolved=hops_resolved,
            temporal_isolation_certified=all(f["timestamp"] <= current_episode for f in retrieved_dict_list),
        )
