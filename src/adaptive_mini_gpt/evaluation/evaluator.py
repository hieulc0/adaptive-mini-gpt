"""Evaluation and generation utilities for Adaptive Mini-GPT."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from adaptive_mini_gpt.data.dataset import TokenDataset
from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer
from adaptive_mini_gpt.model.transformer import MiniGPT


@torch.no_grad()
def evaluate_dataset(
    model: MiniGPT,
    dataset: TokenDataset,
    batch_size: int,
    device: torch.device | str = "cpu",
    max_batches: int = 50,
) -> dict[str, float]:
    """Evaluate model loss, perplexity, and bits-per-character over a dataset."""
    model.eval()
    total_loss = 0.0
    batches_run = 0

    for i in range(max_batches):
        start_idx = i * batch_size * model.config.context_length
        x, y, _ = dataset.get_sequential_batch(
            start_idx=start_idx,
            batch_size=batch_size,
            device=device,
        )
        _, loss = model(x, y)
        if loss is not None:
            total_loss += loss.item()
            batches_run += 1

    mean_loss = total_loss / batches_run if batches_run > 0 else 0.0
    perplexity = math.exp(min(mean_loss, 20.0))
    # Bits per character (or bits per token): cross-entropy loss divided by ln(2)
    bpc = mean_loss / math.log(2)

    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "bits_per_character": bpc,
        "batches_evaluated": batches_run,
    }


@torch.no_grad()
def generate_sample(
    model: MiniGPT,
    tokenizer: CharacterTokenizer,
    prompt: str = "The ",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Generate text completion from a prompt using the model and tokenizer."""
    model.eval()
    encoded = tokenizer.encode(prompt)
    if not encoded:
        # If prompt is empty, use first vocab token
        encoded = [0]

    input_tokens = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    output_tokens: Tensor = model.generate(
        idx=input_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    generated_text = tokenizer.decode(output_tokens[0])
    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "total_tokens": output_tokens.size(1),
        "new_tokens": max_new_tokens,
    }
