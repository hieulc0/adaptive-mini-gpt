"""M34 Phase 2 & 4: Modular Checkpoint and Adapter Lineage Manager.

Tracks persistent evolving lineages for modular systems:
  - Base model parameter hash (verified invariant).
  - Registry of qualified domain adapters (e.g. "arithmetic", "logic", "language").
  - Independent adapter versioning, state saving, promotion, and rollback.
  - Multi-domain manifest export with complete lineage audit trail.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from adaptive_mini_gpt.model.modular_adapter import (
    compute_base_model_hash,
    get_adapter_state_dict,
    load_adapter_state_dict,
)
from adaptive_mini_gpt.model.transformer import MiniGPT

logger = logging.getLogger(__name__)


def compute_adapter_hash(adapter_state: dict[str, torch.Tensor]) -> str:
    """Compute deterministic SHA-256 hash of an adapter's parameter weights."""
    hasher = hashlib.sha256()
    for key in sorted(adapter_state.keys()):
        hasher.update(key.encode("utf-8"))
        tensor = adapter_state[key].detach().cpu().contiguous()
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()[:16]


@dataclass
class ModularCheckpointRecord:
    """Immutable record for a checkpoint in the modular lineage."""

    version_id: str
    parent_version_id: str | None
    base_model_hash: str
    active_adapters: dict[str, str]  # {domain_name: adapter_hash}
    episode_id: str
    domain: str
    status: str  # "QUALIFIED", "CANDIDATE", "ROLLED_BACK", "ROOT"
    gate_decision: str  # "ACCEPTED", "REJECTED", "ROOT"
    rationale: str
    metrics: dict[str, Any] = field(default_factory=dict)
    curriculum_meta: dict[str, Any] = field(default_factory=dict)
    compute_stats: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ModularLineageManifest:
    """Full serialization manifest of the modular lineage tree."""

    seed: int
    active_version_id: str
    base_model_hash: str
    qualified_adapters: dict[str, str]
    total_episodes: int
    successful_promotions: int
    rollbacks: int
    checkpoints: list[ModularCheckpointRecord] = field(default_factory=list)


class ModularLineageManager:
    """Manages the persistent lineage of a modular Mini-GPT (base + adapters)."""

    def __init__(self, seed: int, storage_dir: Path | None = None) -> None:
        self.seed = seed
        self.storage_dir = storage_dir
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.records: dict[str, ModularCheckpointRecord] = {}
        self.adapter_store: dict[str, dict[str, dict[str, torch.Tensor]]] = {}  # {version_id: {domain: state_dict}}
        self.active_version_id: str = "V1"
        self.base_model_hash: str = ""
        self.successful_promotions: int = 0
        self.rollbacks: int = 0

    def register_foundation(
        self,
        base_model: MiniGPT,
        foundation_id: str = "V1",
        metrics: dict[str, Any] | None = None,
        description: str = "Initial foundation checkpoint",
    ) -> ModularCheckpointRecord:
        """Register the starting foundation checkpoint."""
        b_hash = compute_base_model_hash(base_model)
        self.base_model_hash = b_hash
        self.active_version_id = foundation_id

        record = ModularCheckpointRecord(
            version_id=foundation_id,
            parent_version_id=None,
            base_model_hash=b_hash,
            active_adapters={},
            episode_id="foundation",
            domain="base",
            status="QUALIFIED",
            gate_decision="ROOT",
            rationale=description,
            metrics=metrics or {},
            curriculum_meta={"description": description},
            compute_stats={"epochs": 0, "parameters": sum(p.numel() for p in base_model.parameters())},
        )

        self.records[foundation_id] = record
        self.adapter_store[foundation_id] = {}
        logger.info(f"Registered foundation checkpoint {foundation_id} (base_hash: {b_hash})")
        return record

    def get_active_record(self) -> ModularCheckpointRecord:
        """Retrieve record of currently active qualified version."""
        return self.records[self.active_version_id]

    def create_candidate(
        self,
        candidate_id: str,
        episode_id: str,
        domain: str,
        adapter_state: dict[str, torch.Tensor],
        base_model: MiniGPT,
        curriculum_meta: dict[str, Any] | None = None,
        compute_stats: dict[str, Any] | None = None,
    ) -> ModularCheckpointRecord:
        """Register an experimental candidate adapter branching from the active version."""
        parent_rec = self.get_active_record()
        current_base_hash = compute_base_model_hash(base_model)
        if current_base_hash != self.base_model_hash:
            raise ValueError(
                f"Base parameter hash mismatch! Expected {self.base_model_hash}, got {current_base_hash}. "
                "Base parameters must remain strictly immutable when training adapters."
            )

        adapt_hash = compute_adapter_hash(adapter_state)
        new_active_adapters = dict(parent_rec.active_adapters)
        new_active_adapters[domain] = adapt_hash

        record = ModularCheckpointRecord(
            version_id=candidate_id,
            parent_version_id=parent_rec.version_id,
            base_model_hash=self.base_model_hash,
            active_adapters=new_active_adapters,
            episode_id=episode_id,
            domain=domain,
            status="CANDIDATE",
            gate_decision="PENDING",
            rationale=f"Candidate adapter '{domain}' branched from {parent_rec.version_id}.",
            metrics={},
            curriculum_meta=curriculum_meta or {},
            compute_stats=compute_stats or {},
        )

        self.records[candidate_id] = record
        # Store adapter states for candidate
        parent_adapters = copy.deepcopy(self.adapter_store.get(parent_rec.version_id, {}))
        parent_adapters[domain] = copy.deepcopy(adapter_state)
        self.adapter_store[candidate_id] = parent_adapters

        return record

    def promote_candidate(
        self,
        candidate_id: str,
        qualified_id: str,
        metrics: dict[str, Any],
        rationale: str,
    ) -> ModularCheckpointRecord:
        """Promote a candidate adapter to next qualified evolutionary checkpoint."""
        if candidate_id not in self.records:
            raise KeyError(f"Candidate {candidate_id} not found.")

        cand_rec = self.records[candidate_id]
        cand_adapters = self.adapter_store[candidate_id]

        qual_rec = ModularCheckpointRecord(
            version_id=qualified_id,
            parent_version_id=cand_rec.parent_version_id,
            base_model_hash=cand_rec.base_model_hash,
            active_adapters=dict(cand_rec.active_adapters),
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

        self.records[qualified_id] = qual_rec
        self.adapter_store[qualified_id] = copy.deepcopy(cand_adapters)
        self.active_version_id = qualified_id
        self.successful_promotions += 1

        if self.storage_dir:
            torch.save(cand_adapters, self.storage_dir / f"{qualified_id}_adapters.pt")

        logger.info(
            f"PROMOTED candidate {candidate_id} -> {qualified_id} "
            f"(active_adapters: {list(qual_rec.active_adapters.keys())})"
        )
        return qual_rec

    def rollback_candidate(
        self,
        candidate_id: str,
        metrics: dict[str, Any],
        rationale: str,
    ) -> ModularCheckpointRecord:
        """Roll back a failing candidate adapter, leaving active qualified version unchanged."""
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
            f"Active version remains {self.active_version_id}."
        )
        return cand_rec

    def export_manifest(self) -> ModularLineageManifest:
        """Export serializable lineage manifest."""
        active_rec = self.get_active_record()
        return ModularLineageManifest(
            seed=self.seed,
            active_version_id=self.active_version_id,
            base_model_hash=self.base_model_hash,
            qualified_adapters=dict(active_rec.active_adapters),
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
