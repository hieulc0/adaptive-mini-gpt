"""Phase 2: Minimal Episodic Memory for Verified Learning Experiences.

Provides a lightweight, persistent, provenance-aware store for verified learning experiences.
No external database service is required.

Invariants:
- Never treats unverified model-generated answers as ground truth.
- Enforces strict provenance whitelisting on evidence sources.
- Supports lifecycle operations: add, inspect, retrieve, revise, archive, restore, and consolidate.
- Tracks access, retrieval, and empirical usefulness statistics.
- Preserves full lineage/provenance when experiences are consolidated or superseded.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("episodic_memory")


@dataclass
class LearningExperience:
    """A single verified learning experience with provenance and usefulness metadata."""

    experience_id: str
    task_id: str
    family: str
    premises: list[str]
    question: str
    model_prediction: str
    verified_answer: str
    error_category: str
    corrected_procedure: str
    skill_family: str
    evidence_source: str
    episode_order: int
    timestamp: str
    access_count: int = 0
    success_count: int = 0
    last_accessed_episode: int = 0
    is_archived: bool = False
    is_superseded: bool = False
    superseded_by: str | None = None
    consolidated_from: list[str] = field(default_factory=list)

    @property
    def usefulness_score(self) -> float:
        """Laplace-smoothed empirical usefulness ratio."""
        # Returns score in (0, 1]
        return round((self.success_count + 1.0) / (self.access_count + 2.0), 4)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearningExperience:
        return cls(**data)


class EpisodicMemoryBank:
    """Lightweight, persistent memory bank for verified learning experiences."""

    APPROVED_PROVENANCE_PREFIXES = (
        "verified_oracle:",
        "approved_ground_truth:",
        "expert_curriculum:",
        "deterministic_verifier:",
        "approved_relational_oracle:",
    )

    def __init__(self, storage_path: Path | None = None, max_active_capacity: int = 128) -> None:
        self.storage_path = storage_path
        self.max_active_capacity = max_active_capacity
        self.active_experiences: dict[str, LearningExperience] = {}
        self.archived_experiences: dict[str, LearningExperience] = {}

        if storage_path and storage_path.exists():
            self.load(storage_path)

    @staticmethod
    def generate_experience_id(task_id: str, premises: list[str], question: str) -> str:
        key = f"{task_id}::{'|'.join(sorted(premises))}::{question.strip().lower()}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def add_experience(
        self,
        task_id: str,
        family: str,
        premises: list[str],
        question: str,
        model_prediction: str,
        verified_answer: str,
        error_category: str,
        corrected_procedure: str,
        skill_family: str,
        evidence_source: str,
        episode_order: int = 0,
        mark_success_if_correct: bool = True,
    ) -> LearningExperience:
        """Adds a verified experience to memory. Enforces provenance security check."""
        is_approved = any(evidence_source.startswith(p) for p in self.APPROVED_PROVENANCE_PREFIXES)
        if not is_approved:
            raise ValueError(
                f"Security violation: evidence_source '{evidence_source}' is not approved. "
                f"Must begin with one of {self.APPROVED_PROVENANCE_PREFIXES}. "
                "Never store unverified model predictions as ground truth!"
            )

        exp_id = self.generate_experience_id(task_id, premises, question)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # If already exists, update record rather than blindly duplicating
        if exp_id in self.active_experiences:
            exp = self.active_experiences[exp_id]
            exp.model_prediction = model_prediction
            exp.verified_answer = verified_answer
            exp.error_category = error_category
            exp.corrected_procedure = corrected_procedure
            exp.access_count += 1
            if mark_success_if_correct and (model_prediction == verified_answer):
                exp.success_count += 1
            exp.last_accessed_episode = episode_order
            return exp

        exp = LearningExperience(
            experience_id=exp_id,
            task_id=task_id,
            family=family,
            premises=list(premises),
            question=question.strip(),
            model_prediction=model_prediction,
            verified_answer=verified_answer,
            error_category=error_category,
            corrected_procedure=corrected_procedure,
            skill_family=skill_family,
            evidence_source=evidence_source,
            episode_order=episode_order,
            timestamp=now,
            access_count=1,
            success_count=1 if (mark_success_if_correct and model_prediction == verified_answer) else 0,
            last_accessed_episode=episode_order,
            is_archived=False,
            is_superseded=False,
        )

        self.active_experiences[exp_id] = exp
        self._enforce_capacity_limit()

        if self.storage_path:
            self.save(self.storage_path)

        return exp

    def inspect_experience(self, experience_id: str) -> LearningExperience | None:
        """Inspects an experience in active or archived memory."""
        if experience_id in self.active_experiences:
            return copy.deepcopy(self.active_experiences[experience_id])
        if experience_id in self.archived_experiences:
            return copy.deepcopy(self.archived_experiences[experience_id])
        return None

    def archive_experience(self, experience_id: str) -> bool:
        """Moves an active experience to the archive pool."""
        if experience_id in self.active_experiences:
            exp = self.active_experiences.pop(experience_id)
            exp.is_archived = True
            self.archived_experiences[experience_id] = exp
            if self.storage_path:
                self.save(self.storage_path)
            return True
        return False

    def restore_experience(self, experience_id: str) -> bool:
        """Restores an archived experience back to active memory."""
        if experience_id in self.archived_experiences:
            exp = self.archived_experiences.pop(experience_id)
            exp.is_archived = False
            self.active_experiences[experience_id] = exp
            self._enforce_capacity_limit()
            if self.storage_path:
                self.save(self.storage_path)
            return True
        return False

    def revise_experience(
        self,
        old_experience_id: str,
        new_verified_answer: str,
        new_procedure: str,
        new_evidence_source: str,
        episode_order: int = 0,
    ) -> LearningExperience:
        """Supersedes an outdated or contradicted memory and stores the corrected replacement."""
        is_approved = any(new_evidence_source.startswith(p) for p in self.APPROVED_PROVENANCE_PREFIXES)
        if not is_approved:
            raise ValueError(f"Revision security violation: unapproved evidence_source '{new_evidence_source}'")

        target = self.active_experiences.get(old_experience_id) or self.archived_experiences.get(old_experience_id)
        if target is None:
            raise KeyError(f"Experience '{old_experience_id}' not found for revision")

        target.is_superseded = True
        # Archive the superseded memory for auditable provenance
        if old_experience_id in self.active_experiences:
            self.archive_experience(old_experience_id)

        # Create updated experience
        new_exp = self.add_experience(
            task_id=f"{target.task_id}_revised",
            family=target.family,
            premises=target.premises,
            question=target.question,
            model_prediction=target.model_prediction,
            verified_answer=new_verified_answer,
            error_category="corrected_superseded",
            corrected_procedure=new_procedure,
            skill_family=target.skill_family,
            evidence_source=new_evidence_source,
            episode_order=episode_order,
        )
        target.superseded_by = new_exp.experience_id
        new_exp.consolidated_from.append(old_experience_id)

        if self.storage_path:
            self.save(self.storage_path)
        return new_exp

    def consolidate_redundant_experiences(self) -> int:
        """Consolidates duplicate or identical relational patterns while preserving complete provenance."""
        seen_patterns: dict[str, str] = {}
        redundant_ids: list[tuple[str, str]] = []  # (duplicate_id, canonical_id)

        for exp_id, exp in list(self.active_experiences.items()):
            # Canonical signature: sorted premises + question + verified answer
            pattern_key = f"{'|'.join(sorted(exp.premises))}->{exp.question.lower()}->{exp.verified_answer}"
            if pattern_key in seen_patterns:
                canonical_id = seen_patterns[pattern_key]
                redundant_ids.append((exp_id, canonical_id))
            else:
                seen_patterns[pattern_key] = exp_id

        consolidated_count = 0
        for dup_id, can_id in redundant_ids:
            dup_exp = self.active_experiences.pop(dup_id)
            can_exp = self.active_experiences[can_id]

            # Merge usage statistics
            can_exp.access_count += dup_exp.access_count
            can_exp.success_count += dup_exp.success_count
            can_exp.last_accessed_episode = max(can_exp.last_accessed_episode, dup_exp.last_accessed_episode)

            # Preserve provenance
            if dup_id not in can_exp.consolidated_from:
                can_exp.consolidated_from.append(dup_id)
            for prior in dup_exp.consolidated_from:
                if prior not in can_exp.consolidated_from:
                    can_exp.consolidated_from.append(prior)

            # Keep duplicate in archive for auditing
            dup_exp.is_archived = True
            dup_exp.is_superseded = True
            dup_exp.superseded_by = can_id
            self.archived_experiences[dup_id] = dup_exp
            consolidated_count += 1

        if consolidated_count > 0 and self.storage_path:
            self.save(self.storage_path)

        return consolidated_count

    def _enforce_capacity_limit(self) -> None:
        """If active experiences exceed capacity, archive lowest usefulness/recency items."""
        if len(self.active_experiences) <= self.max_active_capacity:
            return

        excess = len(self.active_experiences) - self.max_active_capacity
        # Sort by usefulness score ascending, then by recency ascending
        sorted_candidates = sorted(
            self.active_experiences.values(),
            key=lambda e: (e.usefulness_score, e.last_accessed_episode),
        )

        for i in range(excess):
            self.archive_experience(sorted_candidates[i].experience_id)

    def record_access(self, experience_id: str, success: bool, current_episode: int) -> None:
        """Update empirical usage statistics upon retrieval and utilization."""
        exp = self.active_experiences.get(experience_id) or self.archived_experiences.get(experience_id)
        if exp:
            exp.access_count += 1
            if success:
                exp.success_count += 1
            exp.last_accessed_episode = current_episode

    def get_all_active(self) -> list[LearningExperience]:
        return list(self.active_experiences.values())

    def get_all_archived(self) -> list[LearningExperience]:
        return list(self.archived_experiences.values())

    def save(self, file_path: Path) -> None:
        data = {
            "version": "1.0.0",
            "active_experiences": {k: v.to_dict() for k, v in self.active_experiences.items()},
            "archived_experiences": {k: v.to_dict() for k, v in self.archived_experiences.items()},
        }
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(data, indent=2))

    def load(self, file_path: Path) -> None:
        if not file_path.exists():
            return
        data = json.loads(file_path.read_text())
        self.active_experiences = {
            k: LearningExperience.from_dict(v) for k, v in data.get("active_experiences", {}).items()
        }
        self.archived_experiences = {
            k: LearningExperience.from_dict(v) for k, v in data.get("archived_experiences", {}).items()
        }
