"""Causal self-attention module for decoder-only Transformer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from adaptive_mini_gpt.model.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with scaled dot-product attention."""

    causal_mask: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.context_length = config.context_length

        # Combined projection for query, key, value
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular matrix, future positions are 0
        mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        ).view(1, 1, config.context_length, config.context_length)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        b, t, c = x.size()
        if t > self.context_length:
            raise ValueError(
                f"Sequence length {t} exceeds maximum context length {self.context_length}"
            )

        # Project and separate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        # Mask future tokens
        att = att.masked_fill(~self.causal_mask[:, :, :t, :t], float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, n_heads, T, head_dim)
        # Re-assemble all head outputs side-by-side
        y = y.transpose(1, 2).contiguous().view(b, t, c)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
