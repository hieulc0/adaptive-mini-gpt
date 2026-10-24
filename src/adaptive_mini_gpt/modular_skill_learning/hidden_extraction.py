"""Robust hidden representation extractor supporting MiniGPT and mock models."""
from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn


def extract_model_hidden(model: Any, inp: torch.Tensor) -> torch.Tensor:
    """Extract final layer hidden representation of dimension (B, T, d_model)."""
    if hasattr(model, "blocks") and hasattr(model, "ln_f") and hasattr(model, "wte"):
        b, t = inp.size()
        device = inp.device
        tok_emb = model.wte(inp)
        if getattr(model, "wpe", None) is not None:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            x = model.drop(tok_emb + model.wpe(pos))
        else:
            x = model.drop(tok_emb)
        for block in model.blocks:
            x, _ = block(x)
        return model.ln_f(x)
    else:
        out = model(inp)
        if hasattr(out, "hidden_states"):
            return out.hidden_states
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        if isinstance(out, tuple):
            return out[0]
        return out
