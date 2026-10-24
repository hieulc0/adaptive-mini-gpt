"""Phase 3: Minimal Selective Neural Memory and Reversible Skill States.

Defines:
1. SkillState: ACTIVE, DORMANT, REJECTED, OPTIONALLY_PRUNABLE.
2. ManagedSkillRecord: Metadata, weights, state, usage statistics, and SHA-256 fingerprint.
3. SelectiveSkillMemoryRegistry: Manages active vs dormant skills, state transitions,
   and deterministic routing sets without parameter mutation.
"""
from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.modular_skill_learning.repertoire import (
    ModularSkillRepertoire,
    Rank16ResidualAdapter,
    extract_model_hidden,
)

logger = logging.getLogger(__name__)


class SkillState(str, Enum):
    """Lifecycle state of a modular neural skill."""
    ACTIVE = "ACTIVE"                       # Qualified skill participating in active inference routing
    DORMANT = "DORMANT"                     # Preserved in archive; excluded from default routing to avoid interference
    REJECTED = "REJECTED"                   # Discarded / rolled back after failing development qualification
    OPTIONALLY_PRUNABLE = "OPTIONALLY_PRUNABLE"  # Retained skill evaluated as redundant without marginal utility


@dataclass
class ManagedSkillRecord:
    """Archival record for a managed neural skill adapter."""
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
    last_evaluated_f1: float = 0.0
    sha256_hash: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 digest of adapter parameters."""
        hasher = hashlib.sha256()
        for k, tensor in sorted(self.adapter.state_dict().items()):
            hasher.update(k.encode("utf-8"))
            hasher.update(tensor.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]


class SelectiveSkillMemoryRegistry:
    """Skill memory registry providing selective activation, deactivation, and isolation."""

    def __init__(self, d_model: int = 128, bottleneck_dim: int = 16, num_classes: int = 3) -> None:
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes
        self._records: dict[str, ManagedSkillRecord] = {}

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
    ) -> ManagedSkillRecord:
        """Register a skill with a specific lifecycle state."""
        rec = ManagedSkillRecord(
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

    def get_skill(self, skill_id: str) -> ManagedSkillRecord:
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
        """Calculate total stored parameters across non-rejected skills."""
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
        """Export registry as a standard ModularSkillRepertoire for backward compatibility."""
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
