"""Tokenizer Scaling and Model Vocabulary Expansion Utilities (M28 Phase 3).

Implements:
  - Fast, deterministic BPE tokenizer training up to target vocabulary sizes (e.g. 1024).
  - Quantitative comparison between 400-token BPE baseline and 1024-token BPE.
  - Metrics: characters per token (compression ratio), sequence length reduction, subwords per word (fragmentation).
  - Function-preserving model vocabulary expansion: adapting embedding matrix (wte) and language modeling head (lm_head)
    from 400-vocab to 1024-vocab with zero silent mismatches.
"""

from __future__ import annotations

import collections
import copy
from typing import TYPE_CHECKING, Any

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer

if TYPE_CHECKING:
    import torch
    from adaptive_mini_gpt.model.transformer import MiniGPT


def train_scaled_bpe_tokenizer(
    text: str,
    target_vocab_size: int = 1024,
) -> BPETokenizer:
    """Train deterministic BPE tokenizer on text corpus using word-frequency pairs."""
    tok = BPETokenizer()
    base_count = len(tok.special_tokens) + 256
    if target_vocab_size <= base_count:
        tok._rebuild_vocab()
        return tok

    max_merges = target_vocab_size - base_count

    # Split text into whitespace words with leading space marker for subword boundary awareness
    raw_words = text.split()
    word_freq: dict[tuple[bytes, ...], int] = collections.defaultdict(int)
    for w in raw_words:
        # Prepend space to word bytes to preserve word boundary merges
        w_bytes = b" " + w.encode("utf-8")
        token_tuple = tuple(bytes([b]) for b in w_bytes)
        word_freq[token_tuple] += 1

    merges: list[tuple[bytes, bytes]] = []

    for _ in range(max_merges):
        pair_counts: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
        for w_tuple, count in word_freq.items():
            for i in range(len(w_tuple) - 1):
                pair_counts[(w_tuple[i], w_tuple[i + 1])] += count

        if not pair_counts:
            break

        # Tie-breaking: highest count, then lexicographically smallest
        best_pair = max(sorted(pair_counts.keys()), key=lambda p: pair_counts[p])
        if pair_counts[best_pair] <= 1:
            break

        merges.append(best_pair)
        merged = best_pair[0] + best_pair[1]

        # Update word token tuples
        new_word_freq: dict[tuple[bytes, ...], int] = {}
        for w_tuple, count in word_freq.items():
            new_tuple: list[bytes] = []
            i = 0
            n = len(w_tuple)
            while i < n:
                if i < n - 1 and w_tuple[i] == best_pair[0] and w_tuple[i + 1] == best_pair[1]:
                    new_tuple.append(merged)
                    i += 2
                else:
                    new_tuple.append(w_tuple[i])
                    i += 1
            new_word_freq[tuple(new_tuple)] = count
        word_freq = new_word_freq

    tok.merges = merges
    tok._rebuild_vocab()
    return tok


def compare_tokenizers(
    tok_baseline: BPETokenizer,
    tok_scaled: BPETokenizer,
    sample_texts: list[str],
) -> dict[str, Any]:
    """Quantitatively compare two tokenizers on compression and fragmentation."""
    total_chars = sum(len(t) for t in sample_texts)
    total_words = sum(len(t.split()) for t in sample_texts)

    base_tokens = [tok_baseline.encode(t) for t in sample_texts]
    scaled_tokens = [tok_scaled.encode(t) for t in sample_texts]

    base_token_count = sum(len(toks) for toks in base_tokens)
    scaled_token_count = sum(len(toks) for toks in scaled_tokens)

    base_cpt = total_chars / max(1, base_token_count)
    scaled_cpt = total_chars / max(1, scaled_token_count)

    base_spw = base_token_count / max(1, total_words)
    scaled_spw = scaled_token_count / max(1, total_words)

    reduction = ((base_token_count - scaled_token_count) / max(1, base_token_count)) * 100.0

    return {
        "baseline_vocab_size": tok_baseline.vocab_size,
        "baseline_merges": len(tok_baseline.merges),
        "scaled_vocab_size": tok_scaled.vocab_size,
        "scaled_merges": len(tok_scaled.merges),
        "total_characters": total_chars,
        "total_words": total_words,
        "baseline_total_tokens": base_token_count,
        "scaled_total_tokens": scaled_token_count,
        "baseline_chars_per_token": round(base_cpt, 3),
        "scaled_chars_per_token": round(scaled_cpt, 3),
        "baseline_subwords_per_word": round(base_spw, 3),
        "scaled_subwords_per_word": round(scaled_spw, 3),
        "sequence_length_reduction_pct": round(reduction, 2),
    }


def expand_model_vocabulary(
    model: MiniGPT,
    old_tokenizer: BPETokenizer,
    new_tokenizer: BPETokenizer,
) -> MiniGPT:
    """Expand model embeddings and lm_head from old vocab size to new vocab size."""
    import torch
    import torch.nn as nn

    old_vocab = old_tokenizer.vocab_size
    new_vocab = new_tokenizer.vocab_size

    if new_vocab < old_vocab:
        raise ValueError(f"New vocab size ({new_vocab}) must be >= old vocab size ({old_vocab})")

    if new_vocab == old_vocab:
        return model

    d_model = model.config.d_model
    expanded_model = copy.deepcopy(model)

    # 1. Expand wte (word token embeddings)
    old_wte = expanded_model.wte.weight.data
    new_wte = nn.Embedding(new_vocab, d_model)
    # Initialize with normal distribution matching existing embedding statistics
    mean = old_wte.mean().item()
    std = old_wte.std().item()
    nn.init.normal_(new_wte.weight, mean=mean, std=std)
    # Copy existing weights to preserve representations for original vocabulary
    new_wte.weight.data[:old_vocab, :] = old_wte
    expanded_model.wte = new_wte

    # 2. Expand lm_head (linear projection to logits)
    old_lm = expanded_model.lm_head.weight.data
    new_lm = nn.Linear(d_model, new_vocab, bias=expanded_model.config.bias)
    nn.init.normal_(new_lm.weight, mean=old_lm.mean().item(), std=old_lm.std().item())
    new_lm.weight.data[:old_vocab, :] = old_lm

    if expanded_model.config.bias and expanded_model.lm_head.bias is not None:
        old_bias = expanded_model.lm_head.bias.data
        new_bias = torch.zeros(new_vocab, dtype=old_bias.dtype, device=old_bias.device)
        new_bias[:old_vocab] = old_bias
        new_lm.bias.data = new_bias

    expanded_model.lm_head = new_lm
    expanded_model.config.vocab_size = new_vocab

    return expanded_model
