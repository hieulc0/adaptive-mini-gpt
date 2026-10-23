"""Adapter composition and intervention architectures for M56."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter


class ResidualSkillAdapter(nn.Module):
    """Frozen Source Adapter + Lightweight Trainable Residual Correction."""

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

        # Freeze all source adapter parameters
        self.source_adapter = copy.deepcopy(source_adapter)
        for p in self.source_adapter.parameters():
            p.requires_grad = False
        self.source_adapter.eval()

        # Small trainable residual head (2,371 parameters)
        self.residual_head = PersistentCandidateAdapter(
            d_model=d_model,
            bottleneck_dim=residual_bottleneck_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Full composite forward pass: frozen source + active trainable residual."""
        with torch.no_grad():
            self.source_adapter.eval()
            source_logits = self.source_adapter(hidden_states)
        residual_logits = self.residual_head(hidden_states)
        return source_logits + residual_logits

    def evaluate_source_only(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Isolated evaluation through frozen source adapter only (residual deactivated)."""
        with torch.no_grad():
            self.source_adapter.eval()
            return self.source_adapter(hidden_states)

    def evaluate_residual_only(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Isolated evaluation through residual head only."""
        return self.residual_head(hidden_states)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


class GatedResidualSkillAdapter(nn.Module):
    """Frozen Source Adapter + Learnable Feature Gate + Trainable Residual."""

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

        self.source_adapter = copy.deepcopy(source_adapter)
        for p in self.source_adapter.parameters():
            p.requires_grad = False
        self.source_adapter.eval()

        self.residual_head = PersistentCandidateAdapter(
            d_model=d_model,
            bottleneck_dim=residual_bottleneck_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Dynamic gating layer: LayerNorm + Linear(d_model, num_classes) + Sigmoid (643 params)
        self.gate_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
            nn.Sigmoid(),
        )
        with torch.no_grad():
            nn.init.constant_(self.gate_layer[1].bias, -1.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        last_hidden = hidden_states[:, -1, :]  # (B, D)
        alpha = self.gate_layer(last_hidden)    # (B, num_classes)

        with torch.no_grad():
            self.source_adapter.eval()
            source_logits = self.source_adapter(hidden_states)

        residual_logits = self.residual_head(hidden_states)
        return (1.0 - alpha) * source_logits + alpha * residual_logits

    def evaluate_source_only(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.source_adapter.eval()
            return self.source_adapter(hidden_states)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


class PartiallyTrainableSkillAdapter(nn.Module):
    """Controlled intervention model for Phase 3: scales source adapter gradient flow alpha in [0, 1]."""

    def __init__(
        self,
        source_adapter: PersistentCandidateAdapter,
        trainability_alpha: float = 0.0,
        d_model: int = 128,
        residual_bottleneck_dim: int = 16,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.source_adapter = copy.deepcopy(source_adapter)
        self.trainability_alpha = float(trainability_alpha)

        # Configure source gradients according to trainability alpha
        for p in self.source_adapter.parameters():
            p.requires_grad = (self.trainability_alpha > 0.0)

        self.residual_head = PersistentCandidateAdapter(
            d_model=d_model,
            bottleneck_dim=residual_bottleneck_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.trainability_alpha == 0.0:
            with torch.no_grad():
                source_logits = self.source_adapter(hidden_states)
        else:
            source_logits = self.source_adapter(hidden_states)
            if self.training and self.trainability_alpha < 1.0:
                source_logits = (1.0 - self.trainability_alpha) * source_logits.detach() + self.trainability_alpha * source_logits

        residual_logits = self.residual_head(hidden_states)
        return source_logits + residual_logits

    def evaluate_source_only(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.source_adapter(hidden_states)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
