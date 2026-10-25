"""Milestone M96 Phase 1: Persistent Knowledge Store.

Implements an explicit, persistent multi-part memory system:
1. Factual Knowledge Store:
   - Structured key-value factual records: (entity, attribute) -> value.
   - Versioning and superseding pointers for fact correction.
   - Fast exact indexing + inverted keyword indexing for semantic recall.
2. Neural Skill Registry:
   - Persistent references to acquired neural weights/adapters.
   - Operational metadata (domain, concept, parameter counts, qualification history).
3. Episodic Memory:
   - Chronological log of learning experiences, queries, decisions, and outcomes.
4. Correction Ledger:
   - Audit trail of superseded facts and updates.
5. Strict Capacity Budget Accounting:
   - Explicit byte calculation against 256 KiB budget ceiling.
   - Serialization to atomic JSON files with SHA-256 integrity checksums.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FactualRecord:
    """Individual factual statement with provenance and update tracking."""

    record_id: str
    entity: str
    attribute: str
    value: Any
    confidence: float = 1.0
    timestamp: int = 0
    source: str = "direct_observation"
    is_active: bool = True
    superseded_by: str | None = None


@dataclass
class SkillReference:
    """Persistent catalog reference to a neural adapter/concept."""

    skill_id: str
    domain: str
    concept: str
    adapter_key: str
    parameter_count: int
    success_rate: float = 1.0
    timestamp: int = 0


@dataclass
class EpisodicRecord:
    """Historical record of an individual learning episode."""

    episode_id: int
    task_description: str
    recalled_fact_ids: list[str]
    recalled_skill_ids: list[str]
    predicted_output: Any
    ground_truth: Any
    is_success: bool
    compute_steps: int
    timestamp: int


@dataclass
class CorrectionRecord:
    """Audit entry for a corrected or updated fact."""

    entity: str
    attribute: str
    old_value: Any
    new_value: Any
    timestamp: int
    reason: str


class PersistentKnowledgeStore:
    """Structured persistent memory decoupled from neural parameter weights."""

    def __init__(self, max_byte_budget: int = 262_144) -> None:  # 256 KiB
        self.max_byte_budget = max_byte_budget
        # Core storage
        self.facts: dict[str, FactualRecord] = {}  # record_id -> record
        self.fact_index: dict[tuple[str, str], str] = {}  # (entity, attribute) -> record_id
        self.inverted_index: dict[str, set[str]] = {}  # token -> set of record_ids

        self.skills: dict[str, SkillReference] = {}  # skill_id -> reference
        self.episodes: list[EpisodicRecord] = []
        self.corrections: list[CorrectionRecord] = []

    def store_fact(
        self,
        entity: str,
        attribute: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "observation",
        timestamp: int = 0,
    ) -> str:
        """Store a new verified factual statement."""
        clean_ent = entity.strip().lower()
        clean_attr = attribute.strip().lower()
        key = (clean_ent, clean_attr)

        # Check if already present
        if key in self.fact_index:
            old_id = self.fact_index[key]
            old_rec = self.facts[old_id]
            if old_rec.value == value and old_rec.is_active:
                return old_id  # Idempotent storage
            # Auto-update if value differs
            return self.update_fact(
                entity=clean_ent,
                attribute=clean_attr,
                new_value=value,
                confidence=confidence,
                reason="updated_observation",
                timestamp=timestamp,
            )

        rec_id = f"fact_{clean_ent}_{clean_attr}_{timestamp}"
        rec = FactualRecord(
            record_id=rec_id,
            entity=clean_ent,
            attribute=clean_attr,
            value=value,
            confidence=confidence,
            timestamp=timestamp,
            source=source,
            is_active=True,
        )
        self.facts[rec_id] = rec
        self.fact_index[key] = rec_id
        self._index_tokens(rec_id, f"{clean_ent} {clean_attr}")
        self.enforce_budget()
        return rec_id

    def update_fact(
        self,
        entity: str,
        attribute: str,
        new_value: Any,
        confidence: float = 1.0,
        reason: str = "correction",
        timestamp: int = 0,
    ) -> str:
        """Update an existing factual entry, creating a correction ledger entry."""
        clean_ent = entity.strip().lower()
        clean_attr = attribute.strip().lower()
        key = (clean_ent, clean_attr)

        old_val = None
        if key in self.fact_index:
            old_id = self.fact_index[key]
            old_rec = self.facts[old_id]
            old_val = old_rec.value
            old_rec.is_active = False

        new_id = f"fact_{clean_ent}_{clean_attr}_{timestamp}_v2"
        new_rec = FactualRecord(
            record_id=new_id,
            entity=clean_ent,
            attribute=clean_attr,
            value=new_value,
            confidence=confidence,
            timestamp=timestamp,
            source="verified_correction",
            is_active=True,
        )

        if key in self.fact_index:
            old_id = self.fact_index[key]
            self.facts[old_id].superseded_by = new_id

        self.facts[new_id] = new_rec
        self.fact_index[key] = new_id
        self._index_tokens(new_id, f"{clean_ent} {clean_attr}")

        # Record in correction ledger
        self.corrections.append(
            CorrectionRecord(
                entity=clean_ent,
                attribute=clean_attr,
                old_value=old_val,
                new_value=new_value,
                timestamp=timestamp,
                reason=reason,
            )
        )
        self.enforce_budget()
        return new_id

    def retrieve_fact(self, entity: str, attribute: str) -> FactualRecord | None:
        """Retrieve the current active factual statement for an entity and attribute."""
        key = (entity.strip().lower(), attribute.strip().lower())
        if key in self.fact_index:
            rec_id = self.fact_index[key]
            rec = self.facts.get(rec_id)
            if rec and rec.is_active:
                return rec
        return None

    def search_facts(self, query: str) -> list[FactualRecord]:
        """Find active factual records matching query tokens."""
        tokens = [t.strip().lower() for t in query.split() if len(t.strip()) > 1]
        matching_ids: set[str] = set()
        for tok in tokens:
            if tok in self.inverted_index:
                matching_ids.update(self.inverted_index[tok])

        results = []
        for rid in matching_ids:
            rec = self.facts.get(rid)
            if rec and rec.is_active:
                results.append(rec)
        return results

    def register_skill(
        self,
        skill_id: str,
        domain: str,
        concept: str,
        adapter_key: str,
        parameter_count: int,
        success_rate: float = 1.0,
        timestamp: int = 0,
    ) -> None:
        """Register a qualified neural skill into the persistent catalog."""
        self.skills[skill_id] = SkillReference(
            skill_id=skill_id,
            domain=domain,
            concept=concept,
            adapter_key=adapter_key,
            parameter_count=parameter_count,
            success_rate=success_rate,
            timestamp=timestamp,
        )

    def record_episode(
        self,
        episode_id: int,
        task_description: str,
        recalled_fact_ids: list[str],
        recalled_skill_ids: list[str],
        predicted_output: Any,
        ground_truth: Any,
        is_success: bool,
        compute_steps: int = 1,
        timestamp: int = 0,
    ) -> None:
        """Log an executed learning episode."""
        self.episodes.append(
            EpisodicRecord(
                episode_id=episode_id,
                task_description=task_description,
                recalled_fact_ids=recalled_fact_ids,
                recalled_skill_ids=recalled_skill_ids,
                predicted_output=predicted_output,
                ground_truth=ground_truth,
                is_success=is_success,
                compute_steps=compute_steps,
                timestamp=timestamp,
            )
        )
        self.enforce_budget()

    def _index_tokens(self, record_id: str, text: str) -> None:
        for word in text.split():
            clean = word.strip().lower()
            if len(clean) > 1:
                if clean not in self.inverted_index:
                    self.inverted_index[clean] = set()
                self.inverted_index[clean].add(record_id)

    def calculate_byte_size(self) -> int:
        """Calculate the total serialized byte footprint of all persistent memory."""
        serialized = self.to_json()
        return len(serialized.encode("utf-8"))

    def enforce_budget(self) -> None:
        """Prune oldest superseded facts or oldest episodes if byte budget exceeded."""
        cur_bytes = self.calculate_byte_size()
        if cur_bytes <= self.max_byte_budget:
            return

        # Phase 1: Prune superseded inactive facts
        inactive_ids = [rid for rid, r in self.facts.items() if not r.is_active]
        inactive_ids.sort(key=lambda rid: self.facts[rid].timestamp)
        for rid in inactive_ids:
            if self.calculate_byte_size() <= self.max_byte_budget:
                return
            del self.facts[rid]

        # Phase 2: Prune oldest episodic logs if still above budget
        while self.episodes and self.calculate_byte_size() > self.max_byte_budget:
            self.episodes.pop(0)

    def to_json(self) -> str:
        """Serialize memory store to formatted JSON."""
        data = {
            "max_byte_budget": self.max_byte_budget,
            "facts": {k: asdict(v) for k, v in self.facts.items()},
            "skills": {k: asdict(v) for k, v in self.skills.items()},
            "episodes": [asdict(e) for e in self.episodes],
            "corrections": [asdict(c) for c in self.corrections],
        }
        return json.dumps(data, indent=2, sort_keys=True)

    def compute_checksum(self) -> str:
        """Compute SHA-256 hash of the memory store."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def save_to_file(self, filepath: Path | str) -> None:
        """Atomic write to disk."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        tmp.replace(p)

    @classmethod
    def load_from_file(cls, filepath: Path | str) -> PersistentKnowledgeStore:
        """Restore memory store from file."""
        p = Path(filepath)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        store = cls(max_byte_budget=data.get("max_byte_budget", 262_144))
        for rid, fd in data.get("facts", {}).items():
            rec = FactualRecord(**fd)
            store.facts[rid] = rec
            if rec.is_active:
                store.fact_index[(rec.entity, rec.attribute)] = rid
                store._index_tokens(rid, f"{rec.entity} {rec.attribute}")

        for sid, sd in data.get("skills", {}).items():
            store.skills[sid] = SkillReference(**sd)

        for ed in data.get("episodes", []):
            store.episodes.append(EpisodicRecord(**ed))

        for cd in data.get("corrections", []):
            store.corrections.append(CorrectionRecord(**cd))

        return store
