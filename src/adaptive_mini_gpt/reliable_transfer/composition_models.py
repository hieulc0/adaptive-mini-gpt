"""Adapter composition and intervention architectures for M56."""
from __future__ import annotations

import copy
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
        self.source_adapter = copy.deepcopy(source_adapter)
        # Freeze all source adapter parameters
        for p in self.source_adapter.parameters():
            p.requires_grad = False
        self.source_adapter.eval()

        # Small trainable residual head (2,371 parameters)
        self.residual_head = nn.Sequential(
            nn.Linear(d_model, residual_bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_bottleneck_dim, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Full composite forward pass: frozen source + active trainable residual."""
        with torch.no_grad():
            source_logits = self.source_adapter(h)
        residual_logits = self.residual_head(h)
        return source_logits + residual_logits

    def evaluate_source_only(self, h: torch.Tensor) -> torch.Tensor:
        """Isolated evaluation through frozen source adapter only (residual deactivated)."""
        with torch.no_grad():
            return self.source_adapter(h)

    def evaluate_residual_only(self, h: torch.Tensor) -> torch.Tensor:
        """Isolated evaluation through residual head only."""
        return self.residual_head(h)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


class GatedResidualSkillAdapter(nn.Module):
    """Frozen Source Adapter + Learnable Per-Dimension Gate + Trainable Residual."""

    def __init__(
        self,
        source_adapter: PersistentCandidateAdapter,
        d_model: int = 128,
        residual_bottleneck_dim: int = 16,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.source_adapter = copy.deepcopy(source_adapter)
        for p in self.source_adapter.parameters():
            p.requires_grad = False
        self.source_adapter.eval()

        self.residual_head = nn.Sequential(
            nn.Linear(d_model, residual_bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_bottleneck_dim, num_classes),
        )
        # Dynamic gating vector over class logits
        self.gate_logit = nn.Parameter(torch.zeros(num_classes))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            source_logits = self.source_adapter(h)
        residual_logits = self.residual_head(h)
        alpha = torch.sigmoid(self.gate_logit)
        return (1.0 - alpha) * source_logits + alpha * residual_logits

    def evaluate_source_only(self, h: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.source_adapter(h)

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

        self.residual_head = nn.Sequential(
            nn.Linear(d_model, residual_bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_bottleneck_dim, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.trainability_alpha == 0.0:
            with torch.no_grad():
                source_logits = self.source_adapter(h)
        else:
            source_logits = self.source_adapter(h)
            # Scale gradient if partially trainable
            if self.training and self.trainability_alpha < 1.0:
                source_logits = (1.0 - self.trainability_alpha) * source_logits.detach() + self.trainability_alpha * source_logits

        residual_logits = self.residual_head(h)
        return source_logits + residual_logits

    def evaluate_source_only(self, h: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.source_adapter(h)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
