"""Character-level tokenizer for Adaptive Mini-GPT."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


class CharacterTokenizer:
    """Deterministic character-level tokenizer with fingerprinting and persistence."""

    def __init__(
        self,
        chars: list[str] | None = None,
        unk_token: str = "<unk>",
    ) -> None:
        self.unk_token = unk_token
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        if chars is not None:
            self._build_vocab(chars)

    def _build_vocab(self, unique_chars: list[str]) -> None:
        # Deterministically sorted vocabulary
        sorted_chars = sorted(list(set(unique_chars)))
        # Put unk_token at index 0 if not present
        vocab_list = [self.unk_token] + [c for c in sorted_chars if c != self.unk_token]
        self.stoi = {ch: i for i, ch in enumerate(vocab_list)}
        self.itos = {i: ch for i, ch in enumerate(vocab_list)}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    def fit(self, text: str) -> CharacterTokenizer:
        """Fit tokenizer vocabulary on the given text corpus."""
        unique_chars = list(set(text))
        self._build_vocab(unique_chars)
        return self

    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of token integers."""
        if not self.stoi:
            raise RuntimeError("Tokenizer vocabulary has not been fitted or loaded.")
        unk_idx = self.unk_id
        return [self.stoi.get(ch, unk_idx) for ch in text]

    def decode(self, tokens: list[int] | Tensor) -> str:
        """Convert a list or tensor of token integers back to a string."""
        if isinstance(tokens, Tensor):
            token_list = tokens.tolist()
            if isinstance(token_list, list) and token_list and isinstance(token_list[0], list):
                # Flatten 2D tensor if needed
                token_list = [item for sublist in token_list for item in sublist]
        else:
            token_list = tokens

        result: list[str] = []
        for t in token_list:
            ch = self.itos.get(t, self.unk_token)
            result.append(ch)
        return "".join(result)

    def get_fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint of the vocabulary."""
        if not self.stoi:
            return ""
        # Create deterministic representation of the vocabulary
        canonical = json.dumps(
            {"vocab": [self.itos[i] for i in range(len(self.itos))], "unk": self.unk_token},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize tokenizer state to dictionary."""
        return {
            "vocab": [self.itos[i] for i in range(len(self.itos))],
            "unk_token": self.unk_token,
            "fingerprint": self.get_fingerprint(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CharacterTokenizer:
        """Reconstruct tokenizer from serialized dictionary."""
        unk_token = data.get("unk_token", "<unk>")
        vocab = data.get("vocab", [])
        tokenizer = cls(chars=vocab, unk_token=unk_token)
        return tokenizer

    def save(self, path: str | Path) -> None:
        """Save tokenizer to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> CharacterTokenizer:
        """Load tokenizer from a JSON file."""
        p = Path(path)
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
