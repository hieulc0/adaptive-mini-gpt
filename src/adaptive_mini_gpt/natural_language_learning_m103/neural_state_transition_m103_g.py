"""Milestone M103-G Neural State Transition Module.

Implements the learned neural state transformation function:
    z_{t+1} = f_theta(z_t, a_t)

Features:
- Purely neural: No hardcoded arithmetic or transition rules in the forward pass.
- Decouples representation learning and state transformation from surface string decoding.
- Supports both continuous metric representation (sinusoidal / normalized) and discrete embeddings.
- Evaluates Phase G1 arithmetic state transformations (+2, +3, -1) across varied starting states.
- Evaluates Phase G2 non-numerical causal transformations.
"""

from __future__ import annotations

import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousStateEncoderM103_G(nn.Module):
    """Encodes continuous numeric scalar s into dense representation z_t."""

    def __init__(self, hidden_dim: int = 128, max_val: float = 40.0, num_frequencies: int = 8) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_val = max_val
        self.num_frequencies = num_frequencies

        # 1 normalized feature + 2 * num_frequencies (sin, cos)
        in_dim = 1 + 2 * num_frequencies
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Transforms tensor of shape (B,) of numeric states into (B, hidden_dim)."""
        # Ensure float
        v = values.float().unsqueeze(-1)  # (B, 1)
        norm_v = v / self.max_val

        features = [norm_v]
        for k in range(self.num_frequencies):
            freq = (2.0 ** k) * math.pi / self.max_val
            features.append(torch.sin(v * freq))
            features.append(torch.cos(v * freq))

        phi = torch.cat(features, dim=-1)  # (B, in_dim)
        return self.proj(phi)


class NeuralStateTransitionModuleM103_G(nn.Module):
    """Learned neural state transition dynamics: z_{t+1} = f_theta(z_t, a_t).

    Architecture:
    - State encoder: either continuous metric encoder or discrete lookup table.
    - Action encoder: learnable embedding table for discrete actions (+2, +3, -1, etc.).
    - Transition network: 2-layer residual MLP with GELU activations.
    - State prediction head: linear projection to target state space (0..max_states).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_states: int = 45,
        num_actions: int = 8,
        mode: str = "continuous_metric",  # "continuous_metric" or "discrete_embedding"
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_states = num_states
        self.num_actions = num_actions
        self.mode = mode

        if mode == "continuous_metric":
            self.state_encoder = ContinuousStateEncoderM103_G(hidden_dim=hidden_dim, max_val=float(num_states))
            self.discrete_state_emb = None
        else:
            self.state_encoder = None
            self.discrete_state_emb = nn.Embedding(num_states, hidden_dim)

        self.action_emb = nn.Embedding(num_actions, hidden_dim)

        # Transition function f_theta: [z_t; a_t] -> delta_z
        self.transition_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.post_norm = nn.LayerNorm(hidden_dim)

        # State classification head (logits over possible target states)
        self.state_head = nn.Linear(hidden_dim, num_states)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting next state.

        Args:
            states: (B,) integer state values.
            actions: (B,) integer action IDs.

        Returns:
            logits: (B, num_states) classification logits for next state.
            z_next: (B, hidden_dim) predicted next state embedding.
        """
        if self.mode == "continuous_metric":
            z_t = self.state_encoder(states)
        else:
            z_t = self.discrete_state_emb(states)

        a_t = self.action_emb(actions)

        # Concatenate state and action representations
        za = torch.cat([z_t, a_t], dim=-1)

        # Residual update: z_{t+1} = z_t + f_theta(z_t, a_t)
        delta_z = self.transition_net(za)
        z_next = self.post_norm(z_t + delta_z)

        logits = self.state_head(z_next)
        return logits, z_next

    def predict_next_state(self, state: int, action_id: int) -> int:
        """Runs greedy inference without any hardcoded arithmetic."""
        self.eval()
        with torch.no_grad():
            s_tensor = torch.tensor([state], dtype=torch.long, device=next(self.parameters()).device)
            a_tensor = torch.tensor([action_id], dtype=torch.long, device=next(self.parameters()).device)
            logits, _ = self.forward(s_tensor, a_tensor)
            return int(torch.argmax(logits, dim=-1).item())


class CausalLatentPredictorM103_G(nn.Module):
    """Latent state transition predictor for non-numerical causal concepts (Phase G2).

    Encodes object attributes and actions into latent space, applies neural
    transition f_theta, and predicts discrete outcome state.
    """

    def __init__(
        self,
        num_attributes: int = 12,
        num_actions: int = 6,
        num_states: int = 15,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attr_emb = nn.EmbeddingBag(num_attributes, hidden_dim, mode="mean")
        self.action_emb = nn.Embedding(num_actions, hidden_dim)

        self.transition_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_states)

    def forward(
        self,
        attr_indices: torch.Tensor,
        attr_offsets: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts causal state outcome."""
        obj_rep = self.attr_emb(attr_indices, attr_offsets)
        act_rep = self.action_emb(action_ids)

        comb = torch.cat([obj_rep, act_rep], dim=-1)
        z_out = self.post_norm(obj_rep + self.transition_net(comb))
        return self.classifier(z_out)
