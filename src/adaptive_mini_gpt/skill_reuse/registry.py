"""Phase 2: Minimal Verified Neural Skill Repertoire Registry."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter

logger = logging.getLogger(__name__)


def compute_adapter_hash(adapter: nn.Module) -> str:
    """Compute deterministic SHA-256 hash of adapter parameter bytes."""
    hasher = hashlib.sha256()
    for name, param in sorted(adapter.named_parameters()):
        hasher.update(name.encode("utf-8"))
        tensor = param.detach().cpu().contiguous()
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()[:16]


@dataclass
class RegisteredSkill:
    """Metadata describing a qualified, reusable neural skill adapter."""
    skill_id: str
    checkpoint_hash: str
    base_model_hash: str
    training_lineage: dict[str, Any]
    applicability: str
    limitations: str
    validation_performance: dict[str, float]
    retention_measurements: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegisteredSkill:
        return cls(**data)


class SkillRepertoireRegistry:
    """In-memory and filesystem registry for qualified modular skill adapters."""

    def __init__(self, registry_dir: str | Path | None = None) -> None:
        self.registry_dir = Path(registry_dir) if registry_dir else None
        self._skills: dict[str, PersistentCandidateAdapter] = {}
        self._metadata: dict[str, RegisteredSkill] = {}

    def register_skill(
        self,
        skill_id: str,
        adapter: PersistentCandidateAdapter,
        base_model_hash: str,
        training_lineage: dict[str, Any],
        applicability: str,
        limitations: str,
        validation_performance: dict[str, float],
        retention_measurements: dict[str, float] | None = None,
    ) -> RegisteredSkill:
        """Register a new qualified neural skill adapter."""
        chk_hash = compute_adapter_hash(adapter)
        meta = RegisteredSkill(
            skill_id=skill_id,
            checkpoint_hash=chk_hash,
            base_model_hash=base_model_hash,
            training_lineage=copy.deepcopy(training_lineage),
            applicability=applicability,
            limitations=limitations,
            validation_performance=copy.deepcopy(validation_performance),
            retention_measurements=copy.deepcopy(retention_measurements or {}),
        )
        self._skills[skill_id] = copy.deepcopy(adapter)
        self._metadata[skill_id] = meta
        logger.info(f"Registered skill '{skill_id}' (hash: {chk_hash})")
        return meta

    def get_adapter(self, skill_id: str) -> PersistentCandidateAdapter:
        """Retrieve deep-copied adapter instance for a skill."""
        if skill_id not in self._skills:
            raise KeyError(f"Skill '{skill_id}' not found in registry. Registered: {list(self._skills.keys())}")
        return copy.deepcopy(self._skills[skill_id])

    def get_metadata(self, skill_id: str) -> RegisteredSkill:
        """Retrieve metadata for a skill."""
        if skill_id not in self._metadata:
            raise KeyError(f"Skill '{skill_id}' not found in registry.")
        return copy.deepcopy(self._metadata[skill_id])

    def list_skills(self) -> list[str]:
        """List all registered skill IDs."""
        return sorted(list(self._skills.keys()))

    def verify_integrity(self, skill_id: str) -> bool:
        """Verify that current adapter parameter hash matches recorded metadata hash."""
        if skill_id not in self._skills or skill_id not in self._metadata:
            return False
        adapter = self._skills[skill_id]
        meta = self._metadata[skill_id]
        current_hash = compute_adapter_hash(adapter)
        return current_hash == meta.checkpoint_hash

    def save_to_dir(self, target_dir: str | Path) -> None:
        """Persist registry metadata and adapter checkpoints to directory."""
        p = Path(target_dir)
        p.mkdir(parents=True, exist_ok=True)
        manifest = {
            "skills": {sid: meta.to_dict() for sid, meta in self._metadata.items()}
        }
        with open(p / "skill_registry_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        for sid, adapter in self._skills.items():
            adapter_path = p / f"{sid}.pt"
            adapter.save_checkpoint(adapter_path)

    @classmethod
    def load_from_dir(
        cls,
        target_dir: str | Path,
        d_model: int = 128,
        bottleneck_dim: int = 32,
        num_classes: int = 3,
    ) -> SkillRepertoireRegistry:
        """Load registry metadata and adapter checkpoints from directory."""
        p = Path(target_dir)
        manifest_path = p / "skill_registry_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        registry = cls(registry_dir=p)
        for sid, meta_dict in manifest.get("skills", {}).items():
            adapter_path = p / f"{sid}.pt"
            if not adapter_path.exists():
                raise FileNotFoundError(f"Adapter checkpoint missing for skill {sid} at {adapter_path}")
            adapter = PersistentCandidateAdapter.load_checkpoint(adapter_path)
            meta = RegisteredSkill.from_dict(meta_dict)
            current_hash = compute_adapter_hash(adapter)
            if current_hash != meta.checkpoint_hash:
                raise ValueError(
                    f"Integrity check failed for loaded skill {sid}: "
                    f"computed {current_hash} != expected {meta.checkpoint_hash}"
                )
            registry._skills[sid] = adapter
            registry._metadata[sid] = meta

        return registry
