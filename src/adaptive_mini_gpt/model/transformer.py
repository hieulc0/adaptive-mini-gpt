"""Decoder-only Transformer implementation for Adaptive Mini-GPT."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from adaptive_mini_gpt.model.attention import CausalSelfAttention
from adaptive_mini_gpt.model.config import ModelConfig


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm Transformer decoder block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        # Pre-norm residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.context_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.wte.weight

        # Weight initialization
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layers) ** 0.5)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass through the Transformer.

        Args:
            idx: Token indices tensor of shape (batch_size, seq_len)
            targets: Optional target token indices tensor of shape (batch_size, seq_len)

        Returns:
            Tuple of (logits, loss) where loss is None if targets is None.
        """
        b, t = idx.size()
        if t > self.config.context_length:
            raise ValueError(
                f"Cannot forward sequence of length {t}, block size is {self.config.context_length}"
            )

        device = idx.device
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.wte(idx)  # (b, t, d_model)
        pos_emb = self.wpe(pos)  # (t, d_model)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (b, t, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> Tensor:
        """Autoregressively generate new tokens from conditioning prefix idx.

        Args:
            idx: Conditioning tokens tensor of shape (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (1.0 = standard, 0.0 = greedy)
            top_k: Optional top-k filtering
            eos_id: Optional EOS token id to stop generation early

        Returns:
            Tensor of shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to context length if needed
            if idx.size(1) <= self.config.context_length:
                idx_cond = idx
            else:
                idx_cond = idx[:, -self.config.context_length :]
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (b, vocab_size)

            if temperature <= 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            if eos_id is not None and (idx_next == eos_id).all():
                break

        return idx

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
            if not self.config.tie_word_embeddings:
                n_params -= self.wte.weight.numel()
        return n_params

    def inspect_architecture(self) -> dict[str, Any]:
        """Return architectural details and parameter counts."""
        layer_details: list[dict[str, Any]] = []
        for i, block in enumerate(self.blocks):
            layer_params = sum(p.numel() for p in block.parameters())
            layer_details.append(
                {
                    "layer": i,
                    "params": layer_params,
                }
            )

        return {
            "architecture": self.config.architecture,
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "head_dim": self.config.head_dim,
            "d_ff": self.config.d_ff,
            "context_length": self.config.context_length,
            "vocab_size": self.config.vocab_size,
            "tie_word_embeddings": self.config.tie_word_embeddings,
            "total_params": self.get_num_params(non_embedding=False),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "non_embedding_params": self.get_num_params(non_embedding=True),
            "layers": layer_details,
        }
