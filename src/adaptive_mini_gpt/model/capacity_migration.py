"""Capacity Scaling and Function-Preserving Model Migration (M27 Phase 4).

Provides standardized capacity definitions and weight migration utilities:
  - 1M Control Configuration (895k / 928k parameters)
  - 3M Scaled Configuration (3.43M parameters)
  - Weight-preserving dimension expansion from 1M to 3M Transformer backbones.
  - FLOPs and compute estimation metrics.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from torch import Tensor

from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT


def create_1m_control_config(
    vocab_size: int = 400,
    context_length: int = 256,
    pos_encoding: str = "rope",
) -> ModelConfig:
    """Instantiate standard 1M control model configuration (matches M18-M26 baseline)."""
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        context_length=context_length,
        dropout=0.1,
        bias=True,
        pos_encoding=pos_encoding,
        tie_word_embeddings=False,
    )


def create_3m_scaled_config(
    vocab_size: int = 400,
    context_length: int = 256,
    pos_encoding: str = "rope",
) -> ModelConfig:
    """Instantiate standard 3M primary scaling model configuration (~3.43M params)."""
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
        context_length=context_length,
        dropout=0.1,
        bias=True,
        pos_encoding=pos_encoding,
        tie_word_embeddings=False,
    )


def count_parameters(model: nn.Module) -> int:
    """Compute total number of trainable and non-trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def estimate_training_flops_per_token(param_count: int) -> float:
    """Standard forward + backward FLOPs estimate (~6 * N_params per token)."""
    return 6.0 * float(param_count)


def migrate_1m_to_3m(model_1m: MiniGPT) -> MiniGPT:
    """Migrate trained ~1M MiniGPT weights into an expanded ~3M MiniGPT architecture.

    Copies existing attention, feedforward, and embedding projections into the primary
    subspaces, initializing the expanded dimensions to zero to provide a continuous,
    warm-started capacity growth pathway.
    """
    cfg_3m = create_3m_scaled_config(
        vocab_size=model_1m.config.vocab_size,
        context_length=model_1m.config.context_length,
        pos_encoding=model_1m.config.pos_encoding,
    )
    model_3m = MiniGPT(cfg_3m)

    d_1m = model_1m.config.d_model  # 128
    d_3m = cfg_3m.d_model           # 256
    ff_1m = model_1m.config.d_ff    # 512
    ff_3m = cfg_3m.d_ff             # 1024

    with torch.no_grad():
        # 1. Token Embeddings
        model_3m.wte.weight.zero_()
        model_3m.wte.weight.data[:, :d_1m] = model_1m.wte.weight.data

        # Positional Embeddings if absolute
        if model_1m.wpe is not None and model_3m.wpe is not None:
            model_3m.wpe.weight.zero_()
            model_3m.wpe.weight.data[:, :d_1m] = model_1m.wpe.weight.data

        # 2. Transformer Decoder Blocks
        for l_idx in range(len(model_1m.blocks)):
            b_1m = model_1m.blocks[l_idx]
            b_3m = model_3m.blocks[l_idx]

            # LayerNorm 1
            b_3m.ln_1.weight.fill_(1.0)
            b_3m.ln_1.weight.data[:d_1m] = b_1m.ln_1.weight.data
            if b_1m.ln_1.bias is not None and b_3m.ln_1.bias is not None:
                b_3m.ln_1.bias.zero_()
                b_3m.ln_1.bias.data[:d_1m] = b_1m.ln_1.bias.data

            # Attention c_attn (Q, K, V combined)
            b_3m.attn.c_attn.weight.zero_()
            # Q projection
            b_3m.attn.c_attn.weight.data[0:d_1m, :d_1m] = b_1m.attn.c_attn.weight.data[0:d_1m, :d_1m]
            # K projection
            b_3m.attn.c_attn.weight.data[d_3m : d_3m + d_1m, :d_1m] = b_1m.attn.c_attn.weight.data[d_1m : 2 * d_1m, :d_1m]
            # V projection
            b_3m.attn.c_attn.weight.data[2 * d_3m : 2 * d_3m + d_1m, :d_1m] = b_1m.attn.c_attn.weight.data[2 * d_1m : 3 * d_1m, :d_1m]

            if b_1m.attn.c_attn.bias is not None and b_3m.attn.c_attn.bias is not None:
                b_3m.attn.c_attn.bias.zero_()
                b_3m.attn.c_attn.bias.data[0:d_1m] = b_1m.attn.c_attn.bias.data[0:d_1m]
                b_3m.attn.c_attn.bias.data[d_3m : d_3m + d_1m] = b_1m.attn.c_attn.bias.data[d_1m : 2 * d_1m]
                b_3m.attn.c_attn.bias.data[2 * d_3m : 2 * d_3m + d_1m] = b_1m.attn.c_attn.bias.data[2 * d_1m : 3 * d_1m]

            # Attention c_proj
            b_3m.attn.c_proj.weight.zero_()
            b_3m.attn.c_proj.weight.data[:d_1m, :d_1m] = b_1m.attn.c_proj.weight.data
            if b_1m.attn.c_proj.bias is not None and b_3m.attn.c_proj.bias is not None:
                b_3m.attn.c_proj.bias.zero_()
                b_3m.attn.c_proj.bias.data[:d_1m] = b_1m.attn.c_proj.bias.data

            # LayerNorm 2
            b_3m.ln_2.weight.fill_(1.0)
            b_3m.ln_2.weight.data[:d_1m] = b_1m.ln_2.weight.data
            if b_1m.ln_2.bias is not None and b_3m.ln_2.bias is not None:
                b_3m.ln_2.bias.zero_()
                b_3m.ln_2.bias.data[:d_1m] = b_1m.ln_2.bias.data

            # FeedForward c_fc
            b_3m.mlp.c_fc.weight.zero_()
            b_3m.mlp.c_fc.weight.data[:ff_1m, :d_1m] = b_1m.mlp.c_fc.weight.data
            if b_1m.mlp.c_fc.bias is not None and b_3m.mlp.c_fc.bias is not None:
                b_3m.mlp.c_fc.bias.zero_()
                b_3m.mlp.c_fc.bias.data[:ff_1m] = b_1m.mlp.c_fc.bias.data

            # FeedForward c_proj
            b_3m.mlp.c_proj.weight.zero_()
            b_3m.mlp.c_proj.weight.data[:d_1m, :ff_1m] = b_1m.mlp.c_proj.weight.data
            if b_1m.mlp.c_proj.bias is not None and b_3m.mlp.c_proj.bias is not None:
                b_3m.mlp.c_proj.bias.zero_()
                b_3m.mlp.c_proj.bias.data[:d_1m] = b_1m.mlp.c_proj.bias.data

        # 3. Final LayerNorm
        model_3m.ln_f.weight.fill_(1.0)
        model_3m.ln_f.weight.data[:d_1m] = model_1m.ln_f.weight.data
        if model_1m.ln_f.bias is not None and model_3m.ln_f.bias is not None:
            model_3m.ln_f.bias.zero_()
            model_3m.ln_f.bias.data[:d_1m] = model_1m.ln_f.bias.data

        # 4. Final Language Modeling Head
        model_3m.lm_head.weight.zero_()
        model_3m.lm_head.weight.data[:, :d_1m] = model_1m.lm_head.weight.data

    return model_3m
