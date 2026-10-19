"""Routing mechanisms and auxiliary load-balancing losses for Adaptive Mini-GPT."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TopKRouter(nn.Module):
    """Router supporting learned Top-K token routing and deterministic task routing."""

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 2,
        routing_mode: str = "learned_topk",
        noise_std: float = 0.01,
    ) -> None:
        super().__init__()
        if n_experts <= 0:
            raise ValueError(f"n_experts must be positive, got {n_experts}")
        if not (1 <= top_k <= n_experts):
            raise ValueError(f"top_k ({top_k}) must be in [1, n_experts ({n_experts})]")
        if routing_mode not in ("learned_topk", "task"):
            raise ValueError(f"Unknown routing_mode: {routing_mode}")

        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.routing_mode = routing_mode
        self.noise_std = noise_std

        self.gate = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: Tensor,
        task_id: int | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """Compute routing weights and indices for input tokens.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            task_id: Optional discrete skill/task index for task-conditioned routing.

        Returns:
            Tuple of:
              - routing_weights: (batch_size, seq_len, top_k_effective)
              - selected_experts: (batch_size, seq_len, top_k_effective)
              - aux_info: dict with 'balance_loss', 'z_loss', 'utilization', 'entropy'
        """
        b, t, _ = x.shape
        device = x.device

        # Mode 1: Deterministic task-conditioned routing
        if self.routing_mode == "task":
            tid = int(task_id) if task_id is not None else 0
            expert_idx = tid % self.n_experts
            selected_experts = torch.full((b, t, 1), expert_idx, dtype=torch.long, device=device)
            routing_weights = torch.ones((b, t, 1), dtype=x.dtype, device=device)
            utilization = torch.zeros(self.n_experts, device=device)
            utilization[expert_idx] = 1.0
            return (
                routing_weights,
                selected_experts,
                {
                    "balance_loss": torch.tensor(0.0, device=device),
                    "z_loss": torch.tensor(0.0, device=device),
                    "utilization": utilization.detach(),
                    "entropy": 0.0,
                },
            )

        # Mode 2: Learned Top-K token routing
        logits = self.gate(x.float())  # Compute router logits in FP32 for numerical stability

        # Router Z-loss: penalizes large logits to prevent FP16 overflow
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = torch.mean(log_z**2)

        if self.training and self.noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        probs = F.softmax(logits, dim=-1)  # (B, T, E)
        top_k_probs, selected_experts = torch.topk(probs, k=self.top_k, dim=-1)

        # Normalize probabilities across the chosen top-k experts
        routing_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)
        routing_weights = routing_weights.to(x.dtype)

        # Switch-Transformer Load Balancing Loss: E * sum_i (f_i * P_i)
        # f_i: fraction of tokens routed to expert i
        one_hot = F.one_hot(selected_experts, num_classes=self.n_experts).float()  # (B, T, k, E)
        tokens_per_expert = one_hot.sum(dim=2).mean(dim=(0, 1))  # (E,)
        mean_prob_per_expert = probs.mean(dim=(0, 1))  # (E,)
        balance_loss = self.n_experts * torch.sum(tokens_per_expert * mean_prob_per_expert)

        # Compute normalized routing entropy (1.0 = perfectly uniform across experts)
        avg_probs = mean_prob_per_expert.detach().clamp(min=1e-9)
        raw_entropy = -torch.sum(avg_probs * torch.log(avg_probs)).item()
        max_entropy = math.log(max(self.n_experts, 2))
        norm_entropy = raw_entropy / max_entropy

        return (
            routing_weights,
            selected_experts,
            {
                "balance_loss": balance_loss,
                "z_loss": z_loss,
                "utilization": (tokens_per_expert / self.top_k).detach(),
                "entropy": norm_entropy,
            },
        )
