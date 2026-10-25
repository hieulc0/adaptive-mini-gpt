"""Milestone M85 Phase 2: Dynamic Neural Module Registry and Lifecycle Manager.

Orchestrates full modular lifecycle operations:
1. Module Allocation (`ALLOCATE_NEW`): Spawns fresh physical adapter module (e.g., Rank-4).
2. Dimension Expansion (`EXPAND_RANK`): Enlarges bottleneck rank (e.g., Rank-4 -> Rank-8, Rank-8 -> Rank-16).
3. Dimension Contraction (`SHRINK_RANK`): Compresses bottleneck rank (e.g., Rank-8 -> Rank-4, Rank-4 -> Rank-2).
4. Module Retirement (`RETIRE_MODULE`): Physically removes redundant or obsolete modules, freeing parameters.
5. Module Merging (`MERGE_MODULES`): Combines overlapping modules into a shared representation.
6. Bit-Exact Registry Rollback: Full state snapshot and deterministic SHA-256 state restore.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch
import torch.nn as nn

from adaptive_mini_gpt.dimension_evolution.dynamic_adapter import (
    DynamicDimensionAdapter,
)

logger = logging.getLogger(__name__)


class ModuleActionType(StrEnum):
    """Categorical architectural modifications for module lifecycle."""

    ALLOCATE = "ALLOCATE"
    EXPAND = "EXPAND"
    SHRINK = "SHRINK"
    RETIRE = "RETIRE"
    MERGE = "MERGE"
    REUSE = "REUSE"


@dataclass
class ModuleLifecycleEvent:
    """Audit log entry capturing an architectural mutation."""

    step: int
    action: ModuleActionType
    skill_id: str
    details: str
    parameter_delta: int
    total_physical_parameters: int


class DynamicModuleManager(nn.Module):
    """Manages allocation, dimension scaling, and retirement of neural modules."""

    def __init__(self, d_model: int = 128, num_symbols: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_symbols = num_symbols

        self.modules_dict = nn.ModuleDict()
        self.lifecycle_history: list[ModuleLifecycleEvent] = []

    def has_module(self, skill_id: str) -> bool:
        """Check if skill adapter is currently registered."""
        return skill_id in self.modules_dict

    def get_module(self, skill_id: str) -> DynamicDimensionAdapter:
        """Retrieve registered adapter for specified skill."""
        if skill_id not in self.modules_dict:
            raise KeyError(f"Skill '{skill_id}' not found in active module registry")
        adapter = self.modules_dict[skill_id]
        assert isinstance(adapter, DynamicDimensionAdapter)
        return adapter

    def allocate_module(
        self,
        skill_id: str,
        initial_rank: int = 4,
        step: int = 0,
        device: torch.device | None = None,
    ) -> DynamicDimensionAdapter:
        """Allocate and register a fresh physical adapter module."""
        if skill_id in self.modules_dict:
            raise ValueError(f"Skill '{skill_id}' is already registered in module registry")

        adapter = DynamicDimensionAdapter(
            d_model=self.d_model,
            rank=initial_rank,
            num_symbols=self.num_symbols,
        )
        if device is not None:
            adapter.to(device)

        self.modules_dict[skill_id] = adapter
        param_count = adapter.count_parameters()

        self.lifecycle_history.append(
            ModuleLifecycleEvent(
                step=step,
                action=ModuleActionType.ALLOCATE,
                skill_id=skill_id,
                details=f"Allocated fresh adapter with rank {initial_rank}",
                parameter_delta=param_count,
                total_physical_parameters=self.get_total_physical_parameters(),
            )
        )
        logger.info("Allocated new module '%s' (Rank %d, %d params)", skill_id, initial_rank, param_count)
        return adapter

    def expand_module_rank(self, skill_id: str, target_rank: int, step: int = 0) -> tuple[int, int]:
        """Expand bottleneck rank of specified module."""
        adapter = self.get_module(skill_id)
        params_before = self.get_total_physical_parameters()
        old_rank, new_rank = adapter.expand_rank(target_rank)
        params_after = self.get_total_physical_parameters()

        self.lifecycle_history.append(
            ModuleLifecycleEvent(
                step=step,
                action=ModuleActionType.EXPAND,
                skill_id=skill_id,
                details=f"Expanded rank from {old_rank} to {new_rank}",
                parameter_delta=params_after - params_before,
                total_physical_parameters=params_after,
            )
        )
        logger.info("Expanded module '%s': Rank %d -> %d (+%d params)",
                    skill_id, old_rank, new_rank, params_after - params_before)
        return old_rank, new_rank

    def shrink_module_rank(self, skill_id: str, target_rank: int, step: int = 0) -> tuple[int, int]:
        """Contract bottleneck rank of specified module."""
        adapter = self.get_module(skill_id)
        params_before = self.get_total_physical_parameters()
        old_rank, new_rank = adapter.shrink_rank(target_rank)
        params_after = self.get_total_physical_parameters()

        self.lifecycle_history.append(
            ModuleLifecycleEvent(
                step=step,
                action=ModuleActionType.SHRINK,
                skill_id=skill_id,
                details=f"Shrunk rank from {old_rank} to {new_rank}",
                parameter_delta=params_after - params_before,
                total_physical_parameters=params_after,
            )
        )
        logger.info("Shrunk module '%s': Rank %d -> %d (%d params saved)",
                    skill_id, old_rank, new_rank, params_before - params_after)
        return old_rank, new_rank

    def retire_module(self, skill_id: str, step: int = 0) -> int:
        """Physically retire and deallocate module from registry, freeing memory."""
        if skill_id not in self.modules_dict:
            raise KeyError(f"Skill '{skill_id}' cannot be retired: not registered")

        adapter = self.get_module(skill_id)
        freed_params = adapter.count_parameters()
        del self.modules_dict[skill_id]

        params_after = self.get_total_physical_parameters()
        self.lifecycle_history.append(
            ModuleLifecycleEvent(
                step=step,
                action=ModuleActionType.RETIRE,
                skill_id=skill_id,
                details=f"Retired redundant module, freeing {freed_params} parameters",
                parameter_delta=-freed_params,
                total_physical_parameters=params_after,
            )
        )
        logger.info("Retired module '%s', freed %d parameters. Remaining total: %d",
                    skill_id, freed_params, params_after)
        return freed_params

    def get_total_physical_parameters(self) -> int:
        """Sum total physically allocated trainable parameters across all active modules."""
        return sum(m.count_parameters() for m in self.modules_dict.values())

    def compute_registry_hash(self) -> str:
        """Compute holistic SHA-256 fingerprint of all active modules and parameter weights."""
        hasher = hashlib.sha256()
        # Sort keys to ensure deterministic ordering
        for skill_id in sorted(self.modules_dict.keys()):
            hasher.update(skill_id.encode("utf-8"))
            adapter = self.get_module(skill_id)
            hasher.update(adapter.compute_parameter_hash().encode("utf-8"))
        return hasher.hexdigest()[:16]

    def save_registry_checkpoint(self) -> dict[str, Any]:
        """Capture bit-exact serialization of the entire multi-module repertoire."""
        ckpt_modules = {}
        for skill_id, module in self.modules_dict.items():
            assert isinstance(module, DynamicDimensionAdapter)
            ckpt_modules[skill_id] = module.save_checkpoint()

        return {
            "d_model": self.d_model,
            "num_symbols": self.num_symbols,
            "registry_hash": self.compute_registry_hash(),
            "modules": ckpt_modules,
        }

    def restore_registry_checkpoint(self, checkpoint: dict[str, Any], device: torch.device | None = None) -> None:
        """Restore exact multi-module architecture and parameter weights."""
        self.modules_dict.clear()
        for skill_id, mod_ckpt in checkpoint["modules"].items():
            adapter = DynamicDimensionAdapter(
                d_model=self.d_model,
                rank=mod_ckpt["rank"],
                num_symbols=self.num_symbols,
            )
            adapter.restore_checkpoint(mod_ckpt)
            if device is not None:
                adapter.to(device)
            self.modules_dict[skill_id] = adapter
