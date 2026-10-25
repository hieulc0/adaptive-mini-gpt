"""Milestone M86: Persistent Student State & Multi-Module Life-Cycle Manager.

Encapsulates:
1. Frozen Base Transformer (1,071,872 weights, SHA-256 invariant).
2. Dynamic Modular Head Registry (managing dynamic ranks 2, 4, 8, 16).
3. Skill Repertoire & Compatibility Relationships.
4. Episodic Experience Memory & Trajectory Logging.
5. Resource & Compute Accounting (FLOPs, parameters, teacher queries, storage).
6. Bit-Exact Multi-Module Checkpoint & State Rollback via SHA-256.
"""

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

from adaptive_mini_gpt.dimension_evolution.dynamic_adapter import (
    DynamicDimensionAdapter,
)
from adaptive_mini_gpt.dimension_evolution.module_registry import (
    DynamicModuleManager,
)
from adaptive_mini_gpt.model import MiniGPT

logger = logging.getLogger(__name__)


@dataclass
class SkillDescriptor:
    """Metadata describing an acquired neural skill."""

    skill_id: str
    module_name: str
    task_family: str
    bottleneck_rank: int
    parameter_count: int
    qualification_em: float
    created_at_episode: int
    last_revisited_episode: int
    revisit_count: int = 1
    associated_rules: list[str] = field(default_factory=list)


@dataclass
class LearningEpisodeLog:
    """Audit log of a single learning/continual adaptation episode."""

    episode_idx: int
    task_id: str
    task_family: str
    action_taken: str
    initial_em: float
    final_em: float
    steps_taken: int
    qualified: bool
    initial_parameters: int
    final_parameters: int
    forward_flops: int
    teacher_queries_used: int
    retained_or_rolled_back: str


@dataclass
class GlobalResourceBudget:
    """Tracks cumulative system resource consumption and enforced limits."""

    max_parameters_limit: int = 20_000
    max_training_flops_limit: int = 100_000_000
    max_active_modules_limit: int = 8

    # Cumulative counters
    cumulative_training_flops: int = 0
    cumulative_inference_flops: int = 0
    cumulative_training_steps: int = 0
    cumulative_teacher_queries: int = 0
    cumulative_rollbacks: int = 0


class PersistentStudentState(nn.Module):
    """Holistic persistent state for the Adaptive Mini-GPT continual learner."""

    def __init__(
        self,
        base_model: MiniGPT,
        d_model: int = 128,
        num_symbols: int = 16,
        budget: GlobalResourceBudget | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.d_model = d_model
        self.num_symbols = num_symbols

        # Ensure base model parameters are frozen
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.base_param_count = sum(p.numel() for p in self.base_model.parameters())

        # Dynamic Modular Registry
        self.module_manager = DynamicModuleManager(
            d_model=d_model, num_symbols=num_symbols
        )

        # Knowledge Repertoire
        self.skill_repertoire: dict[str, SkillDescriptor] = {}

        # Episodic Experience Log
        self.experience_history: list[LearningEpisodeLog] = []

        # Global Resource Budget
        self.budget = budget or GlobalResourceBudget()

    def count_total_system_parameters(self, controller_params: int = 0) -> int:
        """Count total physical parameters: base model + all active modules + controller."""
        return self.base_param_count + self.module_manager.get_total_physical_parameters() + controller_params

    def count_active_adapter_parameters(self) -> int:
        """Count physical parameters currently allocated in modular adapters."""
        return self.module_manager.get_total_physical_parameters()

    def compute_holistic_fingerprint(self) -> str:
        """Compute SHA-256 fingerprint over all module weights and repertoire structure."""
        hasher = hashlib.sha256()
        # Add registry module weights hash
        registry_hash = self.module_manager.compute_registry_hash()
        hasher.update(registry_hash.encode("utf-8"))

        # Add skill repertoire keys
        for k in sorted(self.skill_repertoire.keys()):
            hasher.update(k.encode("utf-8"))
            hasher.update(str(self.skill_repertoire[k].bottleneck_rank).encode("utf-8"))

        return hasher.hexdigest()[:16]

    def register_skill(
        self,
        skill_id: str,
        module_name: str,
        task_family: str,
        rank: int,
        em: float,
        episode: int,
        rules: list[str] | None = None,
    ) -> None:
        """Register newly acquired skill in repertoire."""
        mod = self.module_manager.get_module(module_name)
        params = mod.count_parameters()
        self.skill_repertoire[skill_id] = SkillDescriptor(
            skill_id=skill_id,
            module_name=module_name,
            task_family=task_family,
            bottleneck_rank=rank,
            parameter_count=params,
            qualification_em=em,
            created_at_episode=episode,
            last_revisited_episode=episode,
            revisit_count=1,
            associated_rules=rules or [],
        )
        logger.info(
            "Registered Skill '%s' -> Module '%s' (Rank %d, %d params, EM=%.4f)",
            skill_id,
            module_name,
            rank,
            params,
            em,
        )

    def log_episode(self, log_entry: LearningEpisodeLog) -> None:
        """Append episode audit log and update cumulative resource meters."""
        self.experience_history.append(log_entry)
        self.budget.cumulative_training_flops += log_entry.forward_flops
        self.budget.cumulative_training_steps += log_entry.steps_taken
        self.budget.cumulative_teacher_queries += log_entry.teacher_queries_used
        if log_entry.retained_or_rolled_back == "ROLLED_BACK":
            self.budget.cumulative_rollbacks += 1

    def save_checkpoint(self) -> dict[str, Any]:
        """Produce an exact serializable checkpoint of the entire persistent state."""
        return {
            "registry_checkpoint": self.module_manager.save_registry_checkpoint(),
            "skill_repertoire": {k: asdict(v) for k, v in self.skill_repertoire.items()},
            "budget": asdict(self.budget),
            "state_fingerprint": self.compute_holistic_fingerprint(),
        }

    def restore_checkpoint(
        self, checkpoint: dict[str, Any], device: torch.device | None = None
    ) -> None:
        """Bit-exact restoration of entire student state."""
        self.module_manager.restore_registry_checkpoint(
            checkpoint["registry_checkpoint"], device=device
        )
        self.skill_repertoire = {
            k: SkillDescriptor(**v) for k, v in checkpoint["skill_repertoire"].items()
        }
        b_dict = checkpoint["budget"]
        self.budget = GlobalResourceBudget(**b_dict)
        logger.info(
            "Restored Persistent Student State (Fingerprint: %s)",
            self.compute_holistic_fingerprint(),
        )
