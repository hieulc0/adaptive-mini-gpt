"""Data modules for Adaptive Mini-GPT."""

from __future__ import annotations

try:
    from adaptive_mini_gpt.data.dataset import (
        DEFAULT_LAWFUL_CORPUS,
        TokenDataset,
        compute_fingerprint,
        generate_arithmetic_corpus,
        is_holdout_arithmetic_pair,
        load_corpus,
        mask_targets_before_equals,
        split_data,
    )
except ImportError:
    DEFAULT_LAWFUL_CORPUS = None  # type: ignore
    TokenDataset = None  # type: ignore
    compute_fingerprint = None  # type: ignore
    generate_arithmetic_corpus = None  # type: ignore
    is_holdout_arithmetic_pair = None  # type: ignore
    load_corpus = None  # type: ignore
    mask_targets_before_equals = None  # type: ignore
    split_data = None  # type: ignore

from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer

__all__ = [
    "DEFAULT_LAWFUL_CORPUS",
    "CharacterTokenizer",
    "TokenDataset",
    "compute_fingerprint",
    "generate_arithmetic_corpus",
    "is_holdout_arithmetic_pair",
    "load_corpus",
    "mask_targets_before_equals",
    "split_data",
]
