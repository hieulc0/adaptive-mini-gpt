"""Language Pretraining Dataset and Perplexity Evaluator (M28 Phase 3).

Provides:
  - Causal language modeling dataset for autoregressive next-token pretraining.
  - Efficient sequence packing and padding collator.
  - Cross-entropy loss and perplexity evaluation on held-out validation prose.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset
    from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
    from adaptive_mini_gpt.model.transformer import MiniGPT
else:
    try:
        import torch
        from torch import Tensor
        from torch.utils.data import Dataset
    except ImportError:
        torch = None
        Tensor = None
        Dataset = object


class PretrainLanguageDataset(Dataset):
    """Autoregressive next-token prediction dataset for language pretraining."""

    def __init__(
        self,
        texts: list[str],
        tokenizer: BPETokenizer,
        context_length: int = 256,
    ) -> None:
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.samples: list[Tensor] = []

        if torch is None:
            return

        pad_id = getattr(tokenizer, "pad_id", 0)

        for text in texts:
            toks = tokenizer.encode(text, add_bos=True, add_eos=True)
            if len(toks) < 2:
                continue

            # Slice into chunks of up to context_length
            for i in range(0, len(toks) - 1, context_length - 1):
                chunk = toks[i : i + context_length]
                if len(chunk) >= 2:
                    self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        seq = self.samples[idx]
        x = seq[:-1]
        y = seq[1:]
        return x, y


def collate_pretrain_batch(
    batch: list[tuple[Tensor, Tensor]],
    pad_token_id: int = 0,
) -> tuple[Tensor, Tensor]:
    """Collate variable-length pretrain sequences with right-padding."""
    max_len = max(x.size(0) for x, _ in batch)
    batch_size = len(batch)

    x_padded = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    y_padded = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)

    for i, (x, y) in enumerate(batch):
        seq_len = x.size(0)
        x_padded[i, :seq_len] = x
        y_padded[i, :seq_len] = y

    return x_padded, y_padded


def evaluate_language_modeling(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, float]:
    """Compute cross-entropy loss and perplexity on prose texts."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    dataset = PretrainLanguageDataset(
        texts=texts,
        tokenizer=tokenizer,
        context_length=model.config.context_length,
    )

    if len(dataset) == 0:
        return {"loss": 0.0, "perplexity": 1.0}

    pad_id = getattr(tokenizer, "pad_id", 0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_pretrain_batch(b, pad_token_id=pad_id),
    )

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            # Compute loss ignoring pad tokens
            loss = F.cross_entropy(
                logits.view(-1, model.config.vocab_size),
                y.view(-1),
                ignore_index=pad_id,
                reduction="sum",
            )
            non_pad = (y != pad_id).sum().item()
            total_loss += loss.item()
            total_tokens += non_pad

    mean_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, mean_loss))

    return {
        "loss": round(mean_loss, 4),
        "perplexity": round(ppl, 2),
        "evaluated_tokens": total_tokens,
    }
