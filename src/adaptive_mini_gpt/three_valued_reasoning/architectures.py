"""Phase 4: Candidate Adapter Architectures & Diagnostic Probes for Representation Sufficiency."""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any
import torch
import torch.nn as nn


class M65ArchitectureVariant(str, Enum):
    """Architectural and evidence arms evaluated for representation sufficiency."""
    ARM_A_RANK16_BASELINE = "Arm_A_Rank16_Baseline"
    ARM_B_RANK32_EXPANDED = "Arm_B_Rank32_Expanded"
    ARM_C_NONLINEAR_HEAD = "Arm_C_Nonlinear_Head"
    ARM_D_EVIDENCE_GROUNDED_RANK16 = "Arm_D_Evidence_Grounded_Rank16"
    ARM_E_LINEAR_FEATURE_PROBE = "Arm_E_Linear_Feature_Probe"


class Rank16ResidualAdapter(nn.Module):
    """Canonical compact 2,371-parameter rank-16 residual bottleneck adapter."""

    def __init__(self, d_model: int = 128, bottleneck_dim: int = 16, num_classes: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, num_classes),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        last_hidden = hidden_states[:, -1, :]
        return self.net(last_hidden)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_parameter_hash(self) -> str:
        hasher = hashlib.sha256()
        for p in self.parameters():
            hasher.update(p.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]


class Rank32ResidualAdapter(nn.Module):
    """Expanded 4,483-parameter rank-32 residual bottleneck adapter."""

    def __init__(self, d_model: int = 128, bottleneck_dim: int = 32, num_classes: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, num_classes),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        last_hidden = hidden_states[:, -1, :]
        return self.net(last_hidden)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_parameter_hash(self) -> str:
        hasher = hashlib.sha256()
        for p in self.parameters():
            hasher.update(p.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]


class NonlinearResidualHead(nn.Module):
    """Minimal 2-layer nonlinear residual head (2,643 parameters) with intermediate projection."""

    def __init__(self, d_model: int = 128, bottleneck_dim: int = 16, num_classes: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, bottleneck_dim)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.act2 = nn.GELU()
        self.out = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        last_hidden = hidden_states[:, -1, :]
        x = self.ln(last_hidden)
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(self.drop(h1))) + h1  # residual skip within bottleneck
        return self.out(h2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_parameter_hash(self) -> str:
        hasher = hashlib.sha256()
        for p in self.parameters():
            hasher.update(p.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]


class LinearFeatureProbe(nn.Module):
    """Direct diagnostic linear probe (387 parameters) on frozen backbone representations."""

    def __init__(self, d_model: int = 128, num_classes: int = 3) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.probe = nn.Linear(d_model, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        last_hidden = hidden_states[:, -1, :]
        return self.probe(last_hidden)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_parameter_hash(self) -> str:
        hasher = hashlib.sha256()
        for p in self.parameters():
            hasher.update(p.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]


def create_model_for_arm(
    arm: M65ArchitectureVariant,
    d_model: int = 128,
    num_classes: int = 3,
) -> nn.Module:
    """Factory to instantiate models for M65 representation sufficiency benchmark."""
    if arm in (M65ArchitectureVariant.ARM_A_RANK16_BASELINE, M65ArchitectureVariant.ARM_D_EVIDENCE_GROUNDED_RANK16):
        return Rank16ResidualAdapter(d_model=d_model, bottleneck_dim=16, num_classes=num_classes)
    elif arm == M65ArchitectureVariant.ARM_B_RANK32_EXPANDED:
        return Rank32ResidualAdapter(d_model=d_model, bottleneck_dim=32, num_classes=num_classes)
    elif arm == M65ArchitectureVariant.ARM_C_NONLINEAR_HEAD:
        return NonlinearResidualHead(d_model=d_model, bottleneck_dim=16, num_classes=num_classes)
    elif arm == M65ArchitectureVariant.ARM_E_LINEAR_FEATURE_PROBE:
        return LinearFeatureProbe(d_model=d_model, num_classes=num_classes)
    raise ValueError(f"Unknown architecture arm: {arm}")
