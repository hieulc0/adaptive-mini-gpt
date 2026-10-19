"""Mixture-of-Depths (M5): Dynamic token-level compute allocation & layer skipping."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MixtureOfDepthsRouter(nn.Module):
    """Token-level router that dynamically skips FFN/MoE computation on easy tokens."""

    def __init__(self, d_model: int, capacity_factor: float = 0.5, bias: bool = True) -> None:
        super().__init__()
        if not (0.0 < capacity_factor <= 1.0):
            raise ValueError(f"capacity_factor must be in (0.0, 1.0], got {capacity_factor}")
        self.d_model = d_model
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(d_model, 1, bias=bias)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)

    def forward_sublayer(
        self,
        x: Tensor,
        sublayer_fn: Callable[[Tensor], tuple[Tensor, dict[str, Any] | None]],
    ) -> tuple[Tensor, dict[str, Any]]:
        """Route only selected tokens through `sublayer_fn`, bypassing the rest.

        Args:
            x: Post-attention normed hidden states (batch_size, seq_len, d_model)
            sublayer_fn: Callable taking (N_selected, 1, d_model) and returning (out, aux_info)

        Returns:
            Tuple of (residual_update_tensor, mod_telemetry_dict)
        """
        b, t, c = x.shape
        logits = self.gate(x).squeeze(-1).float()  # (B, T)
        weights = torch.sigmoid(logits).to(x.dtype)  # (B, T)

        k = max(1, min(t, int(round(t * self.capacity_factor))))

        if self.training:
            # Top-K token selection per sequence + auxiliary BCE predictor loss
            _, topk_idx = torch.topk(logits, k=k, dim=-1)  # (B, k)
            target_mask = torch.zeros_like(logits)
            target_mask.scatter_(1, topk_idx, 1.0)
            # Train router logits to predict top-k membership around threshold 0.0
            mod_aux_loss = F.binary_cross_entropy_with_logits(logits, target_mask.detach())
            active_mask = target_mask.bool()
        else:
            # Causal inference mode: strictly token-local threshold (logits >= 0.0) so future tokens
            # never affect past tokens' routing decisions!
            active_mask = logits >= 0.0
            mod_aux_loss = torch.tensor(0.0, device=x.device)

        flat_x = x.view(-1, c)
        flat_mask = active_mask.view(-1)
        flat_weights = weights.view(-1, 1)

        selected_indices = torch.where(flat_mask)[0]
        actual_fraction = selected_indices.numel() / max(1, b * t)

        if selected_indices.numel() == 0:
            out = torch.zeros_like(flat_x)
            return out.view(b, t, c), {
                "mod_aux_loss": mod_aux_loss,
                "mod_active_token_fraction": 0.0,
            }

        # Gather only selected tokens (shape: [N_active, 1, C]) -> true FLOP reduction!
        selected_x = flat_x[selected_indices].unsqueeze(1)
        selected_out, sub_aux = sublayer_fn(selected_x)
        selected_out = selected_out.squeeze(1)  # (N_active, C)

        # Multiply by continuous router gate so gradients flow into self.gate!
        gated_selected = (selected_out * flat_weights[selected_indices]).to(flat_x.dtype)

        out = torch.zeros_like(flat_x)
        out.index_copy_(0, selected_indices, gated_selected)

        mod_info: dict[str, Any] = {
            "mod_aux_loss": mod_aux_loss,
            "mod_active_token_fraction": round(actual_fraction, 4),
        }
        if sub_aux is not None:
            mod_info.update(sub_aux)

        return out.view(b, t, c), mod_info
