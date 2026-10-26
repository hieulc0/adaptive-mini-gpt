"""Bottleneck residual experts and Mixture-of-Experts layer."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from adaptive_mini_gpt.adaptive.router import TopKRouter


class BottleneckExpert(nn.Module):
    """Low-rank bottleneck MLP expert: d_model -> d_expert -> d_model."""

    def __init__(
        self,
        d_model: int,
        d_expert: int,
        dropout: float = 0.1,
        bias: bool = True,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        self.w_up = nn.Linear(d_model, d_expert, bias=bias)
        self.act = nn.GELU()
        self.w_down = nn.Linear(d_expert, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.w_up.weight, mean=0.0, std=0.02)
        if self.w_up.bias is not None:
            nn.init.zeros_(self.w_up.bias)

        if zero_init_output:
            nn.init.zeros_(self.w_down.weight)
        else:
            nn.init.normal_(self.w_down.weight, mean=0.0, std=0.02)
        if self.w_down.bias is not None:
            nn.init.zeros_(self.w_down.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_down(self.act(self.w_up(x))))


class AdaptiveMoELayer(nn.Module):
    """Modular residual expert layer with Top-K or Task-conditioned routing."""

    def __init__(
        self,
        d_model: int,
        d_expert: int,
        n_experts: int = 4,
        top_k: int = 2,
        routing_mode: str = "learned_topk",
        dropout: float = 0.1,
        bias: bool = True,
        zero_init_experts: bool = False,
        cohort_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_expert = d_expert
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.routing_mode = routing_mode
        self.dropout = dropout
        self.bias = bias

        self.router = TopKRouter(
            d_model=d_model,
            n_experts=n_experts,
            top_k=self.top_k,
            routing_mode=routing_mode,
            cohort_sizes=cohort_sizes,
        )
        self.experts = nn.ModuleList(
            [
                BottleneckExpert(
                    d_model=d_model,
                    d_expert=d_expert,
                    dropout=dropout,
                    bias=bias,
                    zero_init_output=zero_init_experts,
                )
                for _ in range(n_experts)
            ]
        )

    def forward(
        self,
        x: Tensor,
        task_id: int | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Dispatch tokens to selected experts and merge weighted residual outputs."""
        b, t, c = x.shape
        flat_x = x.view(-1, c)  # (B*T, C)

        routing_weights, selected_experts, aux_info = self.router(x, task_id=task_id)
        k_eff = selected_experts.size(-1)
        flat_weights = routing_weights.view(-1, k_eff)  # (B*T, k_eff)
        flat_experts = selected_experts.view(-1, k_eff)  # (B*T, k_eff)

        out = torch.zeros_like(flat_x)

        # Dispatch per expert (clean, vectorized over active tokens per expert)
        for expert_idx, expert in enumerate(self.experts):
            mask = flat_experts == expert_idx  # (B*T, k_eff)
            if not mask.any():
                continue
            token_indices, slot_indices = torch.where(mask)
            expert_input = flat_x[token_indices]
            expert_output = expert(expert_input)
            weights = flat_weights[token_indices, slot_indices].unsqueeze(-1)
            out.index_add_(0, token_indices, (expert_output * weights).to(flat_x.dtype))

        return out.view(b, t, c), aux_info

    @torch.no_grad()
    def add_experts(self, num_new: int, freeze_existing: bool = False) -> int:
        """Append an independently normalized zero-output residual cohort.

        Existing cohort probabilities remain unchanged, even for negative logits.
        Eval t0 preservation does not imply future retention after training.
        """
        if self.routing_mode == "task":
            raise NotImplementedError(
                "Task expert expansion requires explicit stable task-ID migration"
            )
        if num_new <= 0:
            raise ValueError(f"num_new must be positive, got {num_new}")

        if freeze_existing:
            for exp in self.experts:
                for p in exp.parameters():
                    p.requires_grad = False

        device = self.router.gate.weight.device
        dtype = self.router.gate.weight.dtype

        for _ in range(num_new):
            new_exp = BottleneckExpert(
                d_model=self.d_model,
                d_expert=self.d_expert,
                dropout=self.dropout,
                bias=self.bias,
                zero_init_output=True,  # Zero new outputs alone do not preserve learned routing
            ).to(device=device, dtype=dtype)
            self.experts.append(new_exp)

        old_n = self.n_experts
        new_n = old_n + num_new
        self.n_experts = new_n
        self.router.n_experts = new_n

        # Expand router gate while preserving existing rows
        old_weight = self.router.gate.weight.data
        new_gate = nn.Linear(self.d_model, new_n, bias=False).to(device=device, dtype=dtype)
        new_gate.weight.data[:old_n].copy_(old_weight)
        # Old relative logits stay fixed, but selected sets/normalization can change.
        nn.init.zeros_(new_gate.weight.data[old_n:])
        self.router.gate = new_gate
        self.router.cohort_sizes.append(num_new)
        self.router.cohort_sizes_tensor = self.router.cohort_sizes_tensor.new_tensor(
            self.router.cohort_sizes
        )

        return new_n
