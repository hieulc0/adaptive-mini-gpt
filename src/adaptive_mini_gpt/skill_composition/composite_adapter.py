"""Phase 3: Representation-Preserving Skill Composition Architectures."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter

logger = logging.getLogger(__name__) if "logging" in locals() else None


class ResidualSkillAdapter(nn.Module):
    """Skill composition arm: strictly frozen source adapter combined with a small trainable residual head."""

    def __init__(
        self,
        source_adapter: PersistentCandidateAdapter,
        d_model: int = 128,
        residual_bottleneck_dim: int = 16,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.residual_bottleneck_dim = residual_bottleneck_dim
        self.num_classes = num_classes

        # Freeze source adapter completely
        self.source_adapter = copy.deepcopy(source_adapter)
        self.source_adapter.eval()
        for p in self.source_adapter.parameters():
            p.requires_grad = False

        # Trainable residual branch
        self.residual_head = PersistentCandidateAdapter(
            d_model=d_model,
            bottleneck_dim=residual_bottleneck_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compose frozen source skill representation with additive trainable residual."""
        with torch.no_grad():
            self.source_adapter.eval()
            source_logits = self.source_adapter(hidden_states)
        residual_logits = self.residual_head(hidden_states)
        return source_logits + residual_logits

    def evaluate_source_only(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Evaluate only the underlying frozen source adapter."""
        with torch.no_grad():
            self.source_adapter.eval()
            return self.source_adapter(hidden_states)

    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_trainable_hash(self) -> str:
        hasher = hashlib.sha256()
        for p in self.residual_head.parameters():
            hasher.update(p.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]

    def save_checkpoint(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        param_hash = self.compute_trainable_hash()
        payload = {
            "source_state_dict": self.source_adapter.state_dict(),
            "residual_state_dict": self.residual_head.state_dict(),
            "d_model": self.d_model,
            "residual_bottleneck_dim": self.residual_bottleneck_dim,
            "num_classes": self.num_classes,
            "param_hash": param_hash,
            "composition_type": "residual",
        }
        torch.save(payload, p)
        return param_hash


class GatedResidualSkillAdapter(nn.Module):
    """Skill composition arm: strictly frozen source adapter with a trainable gating mechanism and residual head."""

    def __init__(
        self,
        source_adapter: PersistentCandidateAdapter,
        d_model: int = 128,
        residual_bottleneck_dim: int = 16,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.residual_bottleneck_dim = residual_bottleneck_dim
        self.num_classes = num_classes

        # Freeze source adapter completely
        self.source_adapter = copy.deepcopy(source_adapter)
        self.source_adapter.eval()
        for p in self.source_adapter.parameters():
            p.requires_grad = False

        # Trainable residual branch
        self.residual_head = PersistentCandidateAdapter(
            d_model=d_model,
            bottleneck_dim=residual_bottleneck_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Trainable gating layer (initialized with negative bias to favor source skill initially)
        self.gate_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
            nn.Sigmoid(),
        )
        # Initialize gate bias to -1.0 so initial alpha ~ 0.27 (relies on source adapter)
        with torch.no_grad():
            nn.init.constant_(self.gate_layer[1].bias, -1.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compose frozen source skill and residual using dynamic feature gating."""
        last_hidden = hidden_states[:, -1, :]  # (B, D)
        alpha = self.gate_layer(last_hidden)    # (B, num_classes)

        with torch.no_grad():
            self.source_adapter.eval()
            source_logits = self.source_adapter(hidden_states)

        residual_logits = self.residual_head(hidden_states)
        return (1.0 - alpha) * source_logits + alpha * residual_logits

    def evaluate_source_only(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Evaluate only the underlying frozen source adapter."""
        with torch.no_grad():
            self.source_adapter.eval()
            return self.source_adapter(hidden_states)

    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_trainable_hash(self) -> str:
        hasher = hashlib.sha256()
        for p in self.residual_head.parameters():
            hasher.update(p.detach().cpu().numpy().tobytes())
        for p in self.gate_layer.parameters():
            hasher.update(p.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]

    def save_checkpoint(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        param_hash = self.compute_trainable_hash()
        payload = {
            "source_state_dict": self.source_adapter.state_dict(),
            "residual_state_dict": self.residual_head.state_dict(),
            "gate_state_dict": self.gate_layer.state_dict(),
            "d_model": self.d_model,
            "residual_bottleneck_dim": self.residual_bottleneck_dim,
            "num_classes": self.num_classes,
            "param_hash": param_hash,
            "composition_type": "gated_residual",
        }
        torch.save(payload, p)
        return param_hash
