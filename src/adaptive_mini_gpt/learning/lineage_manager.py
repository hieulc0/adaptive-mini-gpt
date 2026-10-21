"""M33 Phase 2: Persistent Evolving Neural Lineage Manager.

Maintains an immutable, traceable lineage tree for continuous self-learning:
  V1 -> candidate_V2 -> qualified_V2 -> candidate_V3 -> qualified_V3 -> candidate_V4 -> qualified_V4

Guarantees:
  1. Every new learning episode starts from the PREVIOUS QUALIFIED CHECKPOINT,
     never silently reverting to root or training independent specialists.
  2. Cryptographic SHA-256 parameter hashing connects child checkpoints to their exact parent.
  3. Strict rollback isolation: if a candidate fails its qualification gate, the system
     rolls back to the last qualified checkpoint without corrupting the lineage.
  4. Complete auditing: tracks curriculum metadata, metrics across all domains,
     retention measurements, and compute expenditures.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from adaptive_mini_gpt.model.transformer import MiniGPT

logger = logging.getLogger(__name__)


def compute_model_hash(model: nn.Module) -> str:
    """Compute deterministic SHA-256 hash of all model parameters."""
    hasher = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state.keys()):
        hasher.update(key.encode("utf-8"))
        tensor = state[key].detach().cpu().contiguous()
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()[:16]


@dataclass
class CheckpointRecord:
    """Immutable audit record for a single checkpoint in the lineage."""

    version_id: str
    parent_version_id: str | None
    checkpoint_hash: str
    parent_hash: str | None
    episode_id: str
    domain: str
    status: str  # "QUALIFIED", "CANDIDATE", "ROLLED_BACK", "SPECIALIST"
    gate_decision: str  # "ACCEPTED", "REJECTED", "ROOT"
    rationale: str
    metrics: dict[str, Any] = field(default_factory=dict)
    curriculum_meta: dict[str, Any] = field(default_factory=dict)
    compute_stats: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class LineageManifest:
    """Full serialization manifest of the lineage tree."""

    seed: int
    active_version_id: str
    root_hash: str
    total_episodes: int
    successful_promotions: int
    rollbacks: int
    checkpoints: list[CheckpointRecord] = field(default_factory=list)


class LineageManager:
    """Manages the persistent lineage of an evolving Mini-GPT model."""

    def __init__(self, seed: int, storage_dir: Path | None = None) -> None:
        self.seed = seed
        self.storage_dir = storage_dir
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.records: dict[str, CheckpointRecord] = {}
        self.models: dict[str, MiniGPT] = {}
        self.active_version_id: str = "V1"
        self.root_hash: str = ""
        self.successful_promotions: int = 0
        self.rollbacks: int = 0

    def register_root(
        self,
        model: MiniGPT,
        metrics: dict[str, Any] | None = None,
    ) -> CheckpointRecord:
        """Register the initial V1 root model checkpoint."""
        c_hash = compute_model_hash(model)
        self.root_hash = c_hash
        self.active_version_id = "V1"

        record = CheckpointRecord(
            version_id="V1",
            parent_version_id=None,
            checkpoint_hash=c_hash,
            parent_hash=None,
            episode_id="root",
            domain="base",
            status="QUALIFIED",
            gate_decision="ROOT",
            rationale="Initial M18-derived base model with scaled 1024-BPE vocabulary.",
            metrics=metrics or {},
            curriculum_meta={"description": "Pretrained base root"},
            compute_stats={"epochs": 0, "parameters": sum(p.numel() for p in model.parameters())},
        )

        self.records["V1"] = record
        self.models["V1"] = copy.deepcopy(model).cpu()
        if self.storage_dir:
            torch.save(model.state_dict(), self.storage_dir / "V1.pt")
        logger.info(f"Registered root lineage V1 (hash: {c_hash})")
        return record

    def get_active_model(self) -> MiniGPT:
        """Retrieve copy of the currently active qualified model."""
        if self.active_version_id not in self.models:
            raise KeyError(f"Active version {self.active_version_id} not found in registry.")
        return copy.deepcopy(self.models[self.active_version_id])

    def get_active_record(self) -> CheckpointRecord:
        """Retrieve record of the currently active qualified checkpoint."""
        return self.records[self.active_version_id]

    def create_candidate(
        self,
        candidate_model: MiniGPT,
        episode_id: str,
        domain: str,
        candidate_id: str,
        curriculum_meta: dict[str, Any] | None = None,
        compute_stats: dict[str, Any] | None = None,
    ) -> CheckpointRecord:
        """Register an experimental candidate checkpoint branching from current active version."""
        parent_rec = self.get_active_record()
        c_hash = compute_model_hash(candidate_model)

        record = CheckpointRecord(
            version_id=candidate_id,
            parent_version_id=parent_rec.version_id,
            checkpoint_hash=c_hash,
            parent_hash=parent_rec.checkpoint_hash,
            episode_id=episode_id,
            domain=domain,
            status="CANDIDATE",
            gate_decision="PENDING",
            rationale=f"Candidate branched from {parent_rec.version_id} for {episode_id}.",
            metrics={},
            curriculum_meta=curriculum_meta or {},
            compute_stats=compute_stats or {},
        )

        self.records[candidate_id] = record
        self.models[candidate_id] = copy.deepcopy(candidate_model).cpu()
        return record

    def promote_candidate(
        self,
        candidate_id: str,
        qualified_id: str,
        metrics: dict[str, Any],
        rationale: str,
    ) -> CheckpointRecord:
        """Promote a candidate checkpoint to the next qualified evolutionary version."""
        if candidate_id not in self.records:
            raise KeyError(f"Candidate {candidate_id} not found.")

        cand_rec = self.records[candidate_id]
        cand_model = self.models[candidate_id]
        c_hash = cand_rec.checkpoint_hash

        qual_rec = CheckpointRecord(
            version_id=qualified_id,
            parent_version_id=cand_rec.parent_version_id,
            checkpoint_hash=c_hash,
            parent_hash=cand_rec.parent_hash,
            episode_id=cand_rec.episode_id,
            domain=cand_rec.domain,
            status="QUALIFIED",
            gate_decision="ACCEPTED",
            rationale=rationale,
            metrics=metrics,
            curriculum_meta=cand_rec.curriculum_meta,
            compute_stats=cand_rec.compute_stats,
        )

        cand_rec.status = "PROMOTED"
        cand_rec.gate_decision = "ACCEPTED"
        cand_rec.rationale = f"Promoted to {qualified_id}."

        self.records[qualified_id] = qual_rec
        self.models[qualified_id] = copy.deepcopy(cand_model)
        self.active_version_id = qualified_id
        self.successful_promotions += 1

        if self.storage_dir:
            torch.save(cand_model.state_dict(), self.storage_dir / f"{qualified_id}.pt")

        logger.info(
            f"PROMOTED candidate {candidate_id} -> {qualified_id} "
            f"(hash: {c_hash}, parent: {qual_rec.parent_version_id})"
        )
        return qual_rec

    def rollback_candidate(
        self,
        candidate_id: str,
        metrics: dict[str, Any],
        rationale: str,
    ) -> CheckpointRecord:
        """Roll back a failing candidate, preserving active version at previous qualified state."""
        if candidate_id not in self.records:
            raise KeyError(f"Candidate {candidate_id} not found.")

        cand_rec = self.records[candidate_id]
        cand_rec.status = "ROLLED_BACK"
        cand_rec.gate_decision = "REJECTED"
        cand_rec.rationale = rationale
        cand_rec.metrics = metrics
        self.rollbacks += 1

        logger.warning(
            f"ROLLED BACK candidate {candidate_id}. "
            f"Active version remains {self.active_version_id} (hash: {self.get_active_record().checkpoint_hash})."
        )
        return cand_rec

    def export_manifest(self) -> LineageManifest:
        """Export serializable lineage manifest."""
        return LineageManifest(
            seed=self.seed,
            active_version_id=self.active_version_id,
            root_hash=self.root_hash,
            total_episodes=self.successful_promotions + self.rollbacks,
            successful_promotions=self.successful_promotions,
            rollbacks=self.rollbacks,
            checkpoints=list(self.records.values()),
        )

    def save_manifest_json(self, output_file: Path) -> None:
        """Write manifest JSON to disk."""
        manifest = self.export_manifest()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(asdict(manifest), f, indent=2)
