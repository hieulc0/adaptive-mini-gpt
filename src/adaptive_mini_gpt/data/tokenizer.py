"""Deterministic character-level tokenizer with special tokens and multi-domain ASCII support."""

from __future__ import annotations

import hashlib
import json
import string
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<|task:text|>",
    "<|task:math|>",
    "<|task:logic|>",
    "<|ans|>",
]

# Full printable ASCII (digits, letters, punctuation, space, newline, tab)
BASE_ASCII_CHARS = sorted(set(string.printable))


class CharacterTokenizer:
    """Deterministic character-level tokenizer with special-token and base-ASCII support."""

    def __init__(
        self,
        chars: list[str] | None = None,
        unk_token: str = "<unk>",
        use_base_ascii: bool = False,
    ) -> None:
        self.unk_token = unk_token
        self.use_base_ascii = use_base_ascii
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        if chars is not None:
            self._build_vocab(chars, include_base=False)

    def _build_vocab(self, unique_chars: list[str], include_base: bool = True) -> None:
        base_set = set(BASE_ASCII_CHARS) if (include_base and self.use_base_ascii) else set()
        combined = sorted((base_set | set(unique_chars)) - set(SPECIAL_TOKENS))

        vocab_list: list[str] = []
        # Ensure unk_token is always index 0 for backward compatibility with existing tests
        vocab_list.append(self.unk_token)
        for tok in SPECIAL_TOKENS:
            if tok != self.unk_token and tok not in vocab_list:
                vocab_list.append(tok)
        vocab_list.extend(combined)

        self.stoi = {ch: i for i, ch in enumerate(vocab_list)}
        self.itos = {i: ch for i, ch in enumerate(vocab_list)}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    @property
    def pad_id(self) -> int:
        return self.stoi.get("<pad>", 0)

    @property
    def bos_id(self) -> int:
        return self.stoi.get("<bos>", 0)

    @property
    def eos_id(self) -> int:
        return self.stoi.get("<eos>", 0)

    def get_token_id(self, token: str | bytes) -> int | None:
        """Retrieve token ID for string representation."""
        if isinstance(token, bytes):
            s = token.decode("utf-8", errors="replace")
        else:
            s = token
        return self.stoi.get(s)

    def fit(self, text: str, strict_chars_only: bool = False) -> CharacterTokenizer:
        """Fit tokenizer vocabulary on text corpus (plus base ASCII unless strict_chars_only)."""
        unique_chars = list(set(text))
        self._build_vocab(unique_chars, include_base=not strict_chars_only)
        return self

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        task_tag: str | None = None,
    ) -> list[int]:
        """Convert a string into token IDs, greedily matching registered special tokens."""
        if not self.stoi:
            raise RuntimeError("Tokenizer vocabulary has not been fitted or loaded.")

        ids: list[int] = []
        if add_bos and "<bos>" in self.stoi:
            ids.append(self.stoi["<bos>"])
        if task_tag and task_tag in self.stoi:
            ids.append(self.stoi[task_tag])

        unk_idx = self.unk_id
        i = 0
        n = len(text)
        while i < n:
            matched_special = False
            if text[i] == "<":
                for sp in SPECIAL_TOKENS:
                    if text.startswith(sp, i) and sp in self.stoi:
                        ids.append(self.stoi[sp])
                        i += len(sp)
                        matched_special = True
                        break
            if not matched_special:
                ids.append(self.stoi.get(text[i], unk_idx))
                i += 1

        if add_eos and "<eos>" in self.stoi:
            ids.append(self.stoi["<eos>"])
        return ids

    def decode(self, tokens: list[int] | Tensor, skip_special_tokens: bool = False) -> str:
        """Convert a list or tensor of token integers back to a string."""
        if hasattr(tokens, "tolist"):
            token_list = tokens.tolist()
            if isinstance(token_list, list) and token_list and isinstance(token_list[0], list):
                token_list = [item for sublist in token_list for item in sublist]
        else:
            token_list = tokens

        special_set = set(SPECIAL_TOKENS) - {self.unk_token}
        result: list[str] = []
        for t in token_list:
            ch = self.itos.get(t, self.unk_token)
            if skip_special_tokens and ch in special_set:
                continue
            result.append(ch)
        return "".join(result)

    def get_fingerprint(self) -> str:
        if not self.stoi:
            return ""
        canonical = json.dumps(
            {"vocab": [self.itos[i] for i in range(len(self.itos))], "unk": self.unk_token},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab": [self.itos[i] for i in range(len(self.itos))],
            "unk_token": self.unk_token,
            "use_base_ascii": self.use_base_ascii,
            "fingerprint": self.get_fingerprint(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CharacterTokenizer:
        unk_token = data.get("unk_token", "<unk>")
        vocab = data.get("vocab", [])
        use_base_ascii = data.get("use_base_ascii", False)
        tok = cls(chars=None, unk_token=unk_token, use_base_ascii=use_base_ascii)
        # Preserve exact serialized index order
        tok.stoi = {ch: i for i, ch in enumerate(vocab)}
        tok.itos = {i: ch for i, ch in enumerate(vocab)}
        return tok

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> CharacterTokenizer:
        p = Path(path)
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
