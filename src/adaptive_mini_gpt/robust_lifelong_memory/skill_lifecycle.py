"""Milestone M72 Phase 6: Robust Skill Lifecycle and Reversible Memory Management.

Implements:
1. SkillState: ACTIVE, DORMANT, REJECTED, OPTIONALLY_PRUNABLE.
2. RobustSkillRecord: Full metadata, state, execution statistics, and SHA-256 parameter digest.
3. RobustSkillRegistry: Manages skill lifecycle, reversible state transitions, active/dormant segregation,
   and optional non-destructive capacity pruning.
"""
from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import dataclass, field
from enum import StrEnum

from adaptive_mini_gpt.modular_skill_learning.repertoire import (
    ModularSkillRepertoire,
    Rank16ResidualAdapter,
)

logger = logging.getLogger(__name__)


class SkillState(StrEnum):
    """Lifecycle state of a modular neural skill."""
    ACTIVE = "ACTIVE"                       # Qualified skill participating in active inference routing
    DORMANT = "DORMANT"                     # Retained in storage archive; excluded from active routing overhead
    REJECTED = "REJECTED"                   # Rolled back bit-exactly after failing development qualification
    OPTIONALLY_PRUNABLE = "OPTIONALLY_PRUNABLE"  # Retained skill evaluated as redundant without marginal utility


@dataclass
class RobustSkillRecord:
    """Archival record for a managed neural skill adapter in M72."""
    skill_id: str
    rule_family: str
    source_env_id: str
    state: SkillState
    adapter: Rank16ResidualAdapter
    macro_f1: float
    accuracy: float
    classwise_recall: dict[str, float] = field(default_factory=dict)
    created_episode: int = 0
    reuse_count: int = 0
    negative_transfer_count: int = 0
    last_evaluated_f1: float = 0.0
    sha256_hash: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 digest of adapter parameters."""
        hasher = hashlib.sha256()
        for k, tensor in sorted(self.adapter.state_dict().items()):
            hasher.update(k.encode("utf-8"))
            hasher.update(tensor.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]


class RobustSkillRegistry:
    """Registry providing reversible activation, deactivation, and non-destructive pruning."""

    def __init__(self, d_model: int = 128, bottleneck_dim: int = 16, num_classes: int = 3) -> None:
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes
        self._records: dict[str, RobustSkillRecord] = {}

    def register_skill(
        self,
        skill_id: str,
        rule_family: str,
        source_env_id: str,
        adapter: Rank16ResidualAdapter,
        macro_f1: float,
        accuracy: float,
        classwise_recall: dict[str, float] | None = None,
        initial_state: SkillState = SkillState.ACTIVE,
        created_episode: int = 0,
    ) -> RobustSkillRecord:
        """Register a skill into the managed registry."""
        rec = RobustSkillRecord(
            skill_id=skill_id,
            rule_family=rule_family,
            source_env_id=source_env_id,
            state=initial_state,
            adapter=copy.deepcopy(adapter),
            macro_f1=macro_f1,
            accuracy=accuracy,
            classwise_recall=classwise_recall or {},
            created_episode=created_episode,
            reuse_count=0,
            negative_transfer_count=0,
            last_evaluated_f1=macro_f1,
        )
        rec.sha256_hash = rec.compute_hash()
        self._records[skill_id] = rec
        logger.info(
            "Registered skill '%s' (family: %s, state: %s, hash: %s, F1: %.4f)",
            skill_id,
            rule_family,
            initial_state.value,
            rec.sha256_hash,
            macro_f1,
        )
        return rec

    def transition_state(self, skill_id: str, new_state: SkillState, reason: str = "") -> None:
        """Reversibly transition a skill's state without altering its weights."""
        if skill_id not in self._records:
            raise KeyError(f"Skill '{skill_id}' not found in registry.")
        old_state = self._records[skill_id].state
        self._records[skill_id].state = new_state
        logger.info(
            "Transitioned skill '%s' from %s to %s (reason: %s)",
            skill_id,
            old_state.value,
            new_state.value,
            reason or "unspecified",
        )

    def deactivate_skill(self, skill_id: str, reason: str = "Deactivated to prevent routing interference") -> None:
        """Move an active skill to DORMANT state, eliminating execution cost while preserving weights."""
        self.transition_state(skill_id, SkillState.DORMANT, reason=reason)

    def activate_skill(self, skill_id: str, reason: str = "Activated for inference") -> None:
        """Activate a dormant skill for active inference routing."""
        self.transition_state(skill_id, SkillState.ACTIVE, reason=reason)

    def mark_prunable(self, skill_id: str, reason: str = "Marked as prunable due to redundancy") -> None:
        """Flag a skill as OPTIONALLY_PRUNABLE when empirical evidence shows no marginal utility."""
        self.transition_state(skill_id, SkillState.OPTIONALLY_PRUNABLE, reason=reason)

    def rollback_candidate(self, skill_id: str, reason: str = "Candidate qualification failed") -> None:
        """Roll back an unaccepted candidate skill bit-exactly."""
        if skill_id in self._records:
            self.transition_state(skill_id, SkillState.REJECTED, reason=reason)

    def prune_redundant_skills(self) -> list[str]:
        """Reversibly prune all OPTIONALLY_PRUNABLE skills by archiving them to DORMANT."""
        pruned = []
        for sk_id, rec in self._records.items():
            if rec.state == SkillState.OPTIONALLY_PRUNABLE:
                self.transition_state(sk_id, SkillState.DORMANT, reason="Reversible capacity consolidation")
                pruned.append(sk_id)
        return pruned

    def get_skill(self, skill_id: str) -> RobustSkillRecord:
        """Retrieve record for a skill."""
        return self._records[skill_id]

    def list_skills(self, state_filter: SkillState | None = None) -> list[str]:
        """List skill IDs optionally filtered by state."""
        if state_filter is None:
            return sorted(self._records.keys())
        return sorted([k for k, v in self._records.items() if v.state == state_filter])

    def list_active_skills(self) -> list[str]:
        """List currently ACTIVE skill IDs."""
        return self.list_skills(SkillState.ACTIVE)

    def list_dormant_skills(self) -> list[str]:
        """List DORMANT skill IDs."""
        return self.list_skills(SkillState.DORMANT)

    def total_stored_parameters(self) -> int:
        """Calculate total parameters in non-rejected storage."""
        total = 0
        for rec in self._records.values():
            if rec.state != SkillState.REJECTED:
                total += sum(p.numel() for p in rec.adapter.parameters())
        return total

    def total_active_parameters(self) -> int:
        """Calculate total parameters in active routing pool."""
        total = 0
        for rec in self._records.values():
            if rec.state == SkillState.ACTIVE:
                total += sum(p.numel() for p in rec.adapter.parameters())
        return total

    def export_to_modular_repertoire(self, active_only: bool = True) -> ModularSkillRepertoire:
        """Export registry as a standard ModularSkillRepertoire."""
        rep = ModularSkillRepertoire(
            d_model=self.d_model,
            bottleneck_dim=self.bottleneck_dim,
            num_classes=self.num_classes,
        )
        for sk_id, rec in self._records.items():
            if active_only and rec.state != SkillState.ACTIVE:
                continue
            if rec.state == SkillState.REJECTED:
                continue
            rep.register_skill(
                skill_id=sk_id,
                rule_family=rec.rule_family,
                source_env_id=rec.source_env_id,
                adapter=rec.adapter,
                macro_f1=rec.macro_f1,
                accuracy=rec.accuracy,
                classwise_recall=rec.classwise_recall,
            )
        return rep
