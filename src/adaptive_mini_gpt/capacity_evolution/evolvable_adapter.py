"""Phase 5: Evolvable Residual Skill Adapter with Zero-Initialized Reversible Expansion."""
from __future__ import annotations

import copy
import logging
from typing import Any
import torch
import torch.nn as nn

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter

logger = logging.getLogger(__name__)


class EvolvableResidualSkillAdapter(nn.Module):
    """Residual skill adapter with reversible, zero-initialized capacity expansion.

    Combines a frozen source adapter with a dynamically expandable residual head:
        y = f_source(h) + g_res(h)

    Bottleneck dimensions:
        - Compact (Base): d_res = 16 (2,371 trainable parameters)
        - Expanded:       d_res = 32 (4,483 trainable parameters)
        - Wide:           d_res = 64 (8,707 trainable parameters)

    Zero-Initialization Property:
        When expanding from d_old to d_new, newly added columns in the output
        projection are initialized strictly to zero:
            W_out[:, d_old:d_new] = 0.0
        ensuring that at the instant of expansion, f_expanded(h) == f_compact(h).
    """

    def __init__(
        self,
        source_adapter: PersistentCandidateAdapter,
        d_model: int = 128,
        initial_bottleneck_dim: int = 16,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.dropout = dropout

        # Frozen source adapter (immutable)
        self.source_adapter = copy.deepcopy(source_adapter)
        for p in self.source_adapter.parameters():
            p.requires_grad = False
        self.source_adapter.eval()

        # Evolvable residual head
        self.current_bottleneck_dim = initial_bottleneck_dim
        self.residual_head = PersistentCandidateAdapter(
            d_model=d_model,
            bottleneck_dim=initial_bottleneck_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Snapshot storage for complete reversibility / rollback
        self._snapshot_state: dict[str, Any] | None = None
        self.save_snapshot()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through frozen source plus evolvable residual."""
        with torch.no_grad():
            self.source_adapter.eval()
            source_logits = self.source_adapter(hidden_states)

        residual_logits = self.residual_head(hidden_states)
        return source_logits + residual_logits

    def evaluate_source_only(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Evaluate strictly through the frozen source pathway."""
        with torch.no_grad():
            self.source_adapter.eval()
            return self.source_adapter(hidden_states)

    def count_parameters(self) -> dict[str, int]:
        """Report total, trainable, and frozen parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def save_snapshot(self) -> None:
        """Capture deep snapshot of current residual head for reversible rollback."""
        self._snapshot_state = {
            "bottleneck_dim": self.current_bottleneck_dim,
            "residual_state_dict": {
                k: v.clone().detach() for k, v in self.residual_head.state_dict().items()
            },
        }

    def rollback_to_snapshot(self) -> None:
        """Revert residual head to the saved snapshot configuration with 100% fidelity."""
        if self._snapshot_state is None:
            raise RuntimeError("No snapshot available for rollback.")

        saved_dim = self._snapshot_state["bottleneck_dim"]
        if saved_dim != self.current_bottleneck_dim:
            self.residual_head = PersistentCandidateAdapter(
                d_model=self.d_model,
                bottleneck_dim=saved_dim,
                num_classes=self.num_classes,
                dropout=self.dropout,
            ).to(next(self.parameters()).device)
            self.residual_head.train(self.training)
            self.current_bottleneck_dim = saved_dim

        self.residual_head.load_state_dict(self._snapshot_state["residual_state_dict"])
        logger.info("Rolled back EvolvableResidualSkillAdapter to bottleneck dim %d", saved_dim)

    def expand_capacity(self, target_bottleneck_dim: int) -> int:
        """Expand residual bottleneck dimension with zero-initialization on new channels.

        Returns:
            The new trainable parameter count.
        """
        old_dim = self.current_bottleneck_dim
        if target_bottleneck_dim <= old_dim:
            logger.info("Target bottleneck %d <= current %d; no expansion performed", target_bottleneck_dim, old_dim)
            return self.count_parameters()["trainable"]

        device = next(self.parameters()).device
        self.save_snapshot()

        new_head = PersistentCandidateAdapter(
            d_model=self.d_model,
            bottleneck_dim=target_bottleneck_dim,
            num_classes=self.num_classes,
            dropout=self.dropout,
        ).to(device)
        new_head.train(self.training)

        old_net = self.residual_head.net
        new_net = new_head.net

        with torch.no_grad():
            # 1. LayerNorm: copy directly
            new_net[0].weight.copy_(old_net[0].weight)
            new_net[0].bias.copy_(old_net[0].bias)

            # 2. Linear down (d_model -> target_bottleneck_dim)
            new_net[1].weight[:old_dim, :].copy_(old_net[1].weight)
            new_net[1].bias[:old_dim].copy_(old_net[1].bias)
            # Initialize newly added channels with small variance
            nn.init.normal_(new_net[1].weight[old_dim:, :], mean=0.0, std=0.02)
            nn.init.zeros_(new_net[1].bias[old_dim:])

            # 3. Linear up (target_bottleneck_dim -> num_classes)
            new_net[4].weight[:, :old_dim].copy_(old_net[4].weight)
            new_net[4].bias.copy_(old_net[4].bias)
            # ZERO-INIT: Zero out newly added input dimensions in the output layer
            nn.init.zeros_(new_net[4].weight[:, old_dim:])

        self.residual_head = new_head
        self.current_bottleneck_dim = target_bottleneck_dim

        new_trainable = self.count_parameters()["trainable"]
        logger.info(
            "Expanded EvolvableResidualSkillAdapter from %d to %d (trainable params: %d)",
            old_dim, target_bottleneck_dim, new_trainable,
        )
        return new_trainable
