"""Causal self-attention module for decoder-only Transformer with RoPE and ALiBi support."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from adaptive_mini_gpt.model.config import ModelConfig


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dimensions of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embedding to query or key tensor."""
    return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with fused PyTorch SDPA, RoPE, and ALiBi."""

    causal_mask: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.context_length = config.context_length
        self.dropout_p = config.dropout
        self.pos_encoding = getattr(config, "pos_encoding", "absolute")
        self.position_offset: int = 0

        # Combined projection for query, key, value
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Retained for backward-compatibility with mask inspection tests
        mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        ).view(1, 1, config.context_length, config.context_length)
        self.register_buffer("causal_mask", mask, persistent=False)

        # RoPE inverse frequencies
        if self.pos_encoding == "rope":
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        # ALiBi head slopes: 2^(-8 * (h+1) / H)
        if self.pos_encoding == "alibi":
            slopes = torch.tensor(
                [2.0 ** (-8.0 * (i + 1) / self.n_heads) for i in range(self.n_heads)],
                dtype=torch.float32,
            ).view(1, self.n_heads, 1, 1)
            self.register_buffer("alibi_slopes", slopes, persistent=False)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Forward pass for causal self-attention using fused SDPA."""
        b, t, c = x.size()
        # For absolute position encoding, assert length bounds
        if self.pos_encoding == "absolute" and t > self.context_length:
            raise ValueError(
                f"Sequence length {t} exceeds maximum context length {self.context_length}"
            )

        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        if self.pos_encoding == "rope":
            assert isinstance(self.inv_freq, torch.Tensor)
            t_pos = torch.arange(t, device=x.device, dtype=torch.float32) + getattr(self, "position_offset", 0)
            freqs = torch.outer(t_pos, self.inv_freq.to(device=x.device))
            cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1).view(1, 1, t, self.head_dim).to(dtype=q.dtype)
            sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1).view(1, 1, t, self.head_dim).to(dtype=q.dtype)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        if self.pos_encoding == "alibi":
            assert isinstance(self.alibi_slopes, torch.Tensor)
            pos = torch.arange(t, device=x.device)
            dist = (pos.view(1, 1, 1, t) - pos.view(1, 1, t, 1)).to(dtype=q.dtype)
            alibi_bias = dist * self.alibi_slopes.to(device=x.device, dtype=q.dtype)
            causal_triu = torch.triu(
                torch.full((t, t), float("-inf"), device=x.device, dtype=q.dtype), diagonal=1
            ).view(1, 1, t, t)
            alibi_mask = causal_triu + alibi_bias
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    alibi_mask = alibi_mask.masked_fill(~attn_mask, float("-inf"))
                else:
                    alibi_mask = alibi_mask + attn_mask

            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=alibi_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            )
        else:
            if attn_mask is not None:
                causal = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
                if attn_mask.dtype == torch.bool:
                    attn_mask = attn_mask & causal
                else:
                    attn_mask = attn_mask.masked_fill(~causal, float("-inf"))
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=False,
                )
            else:
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=True,
                )

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_dropout(self.c_proj(y))
        return y
