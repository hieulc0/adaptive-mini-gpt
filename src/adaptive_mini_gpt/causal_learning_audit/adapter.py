"""Phase 4: Persistent Isolated Neural Candidate Adapter."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn


class PersistentCandidateAdapter(nn.Module):
    """Isolated neural adapter head trained on verified observations and minimal pairs.

    Evaluated strictly neural-only without any external symbolic reasoning,
    memory databases, or oracle access at inference time.
    """

    def __init__(
        self,
        d_model: int = 64,
        bottleneck_dim: int = 16,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
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
        """Map sequence hidden states (B, T, D) to last-token class logits (B, num_classes)."""
        last_hidden = hidden_states[:, -1, :]  # (B, D)
        return self.net(last_hidden)

    def compute_parameter_hash(self) -> str:
        """Compute SHA-256 fingerprint of adapter parameters."""
        hasher = hashlib.sha256()
        for p in self.parameters():
            hasher.update(p.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]

    def save_checkpoint(self, path: str | Path) -> str:
        """Persist adapter weights to disk and return parameter hash."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        param_hash = self.compute_parameter_hash()
        payload = {
            "state_dict": self.state_dict(),
            "d_model": self.d_model,
            "bottleneck_dim": self.bottleneck_dim,
            "num_classes": self.num_classes,
            "param_hash": param_hash,
        }
        torch.save(payload, p)
        return param_hash

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: torch.device | None = None) -> PersistentCandidateAdapter:
        """Reload adapter from disk in a fresh instance and verify integrity."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Adapter checkpoint not found: {path}")

        payload = torch.load(p, map_location=device or torch.device("cpu"))
        adapter = cls(
            d_model=payload["d_model"],
            bottleneck_dim=payload["bottleneck_dim"],
            num_classes=payload["num_classes"],
        )
        adapter.load_state_dict(payload["state_dict"])
        if device is not None:
            adapter.to(device)

        reloaded_hash = adapter.compute_parameter_hash()
        expected_hash = payload.get("param_hash")
        if expected_hash and reloaded_hash != expected_hash:
            raise RuntimeError(f"Adapter parameter hash mismatch: {reloaded_hash} vs expected {expected_hash}")

        return adapter
