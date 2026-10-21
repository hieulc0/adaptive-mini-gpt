"""Deterministic Byte-Pair Encoding (BPE) subword tokenizer with special tokens (M9)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
]


class BPETokenizer:
    """Byte-level Byte Pair Encoding (BPE) subword tokenizer."""

    tokenizer_type: str = "bpe"
    version: str = "bpe-v1"

    def __init__(
        self,
        merges: list[tuple[bytes, bytes]] | None = None,
        vocab: list[bytes] | None = None,
    ) -> None:
        self.merges: list[tuple[bytes, bytes]] = merges or []
        self.special_tokens: list[str] = list(SPECIAL_TOKENS)
        self.vocab: list[bytes] = []
        self.stoi: dict[bytes, int] = {}
        self.itos: dict[int, bytes] = {}

        if vocab is not None:
            self._set_vocab(vocab)
        elif merges is not None:
            self._rebuild_vocab()

    def _set_vocab(self, vocab: list[bytes]) -> None:
        self.vocab = list(vocab)
        self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
        self.itos = {i: tok for i, tok in enumerate(self.vocab)}

    def _rebuild_vocab(self) -> None:
        """Construct full vocabulary: special tokens + base 256 bytes + merged pairs."""
        v: list[bytes] = []
        # 1. Special tokens first
        for sp in self.special_tokens:
            v.append(sp.encode("utf-8"))
        # 2. Base 256 byte tokens
        for b in range(256):
            v.append(bytes([b]))
        # 3. Merges in order
        for p1, p2 in self.merges:
            merged = p1 + p2
            if merged not in v:
                v.append(merged)
        self._set_vocab(v)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.stoi[b"<pad>"]

    @property
    def unk_id(self) -> int:
        return self.stoi[b"<unk>"]

    @property
    def bos_id(self) -> int:
        return self.stoi[b"<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi[b"<eos>"]

    @property
    def user_id(self) -> int:
        return self.stoi[b"<|user|>"]

    @property
    def assistant_id(self) -> int:
        return self.stoi[b"<|assistant|>"]

    @property
    def system_id(self) -> int:
        return self.stoi[b"<|system|>"]

    def get_token_id(self, token: str | bytes) -> int | None:
        """Retrieve token ID for string or byte representation."""
        if isinstance(token, str):
            b = token.encode("utf-8")
        else:
            b = token
        return self.stoi.get(b)

    def train(self, text: str, target_vocab_size: int = 512) -> BPETokenizer:
        """Train BPE merges on input corpus up to target_vocab_size."""
        base_count = len(self.special_tokens) + 256
        if target_vocab_size <= base_count:
            self._rebuild_vocab()
            return self

        max_merges = target_vocab_size - base_count

        # Split into initial byte sequences (ignoring special tokens during train)
        raw_bytes = text.encode("utf-8")
        tokens: list[bytes] = [bytes([b]) for b in raw_bytes]

        self.merges = []
        for _ in range(max_merges):
            pair_counts: dict[tuple[bytes, bytes], int] = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break
            # Find most frequent pair (ties broken deterministically)
            best_pair = max(sorted(pair_counts.keys()), key=lambda p: pair_counts[p])
            if pair_counts[best_pair] <= 1:
                break

            self.merges.append(best_pair)
            merged = best_pair[0] + best_pair[1]

            # Replace pairs in token stream
            new_tokens: list[bytes] = []
            i = 0
            n = len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        self._rebuild_vocab()
        return self

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text into subword token IDs, preserving special tokens."""
        if not self.vocab:
            raise RuntimeError("BPETokenizer has not been trained or loaded.")

        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)

        # Segment text by special tokens
        i = 0
        n = len(text)
        while i < n:
            matched_special = False
            for sp in self.special_tokens:
                if text.startswith(sp, i):
                    ids.append(self.stoi[sp.encode("utf-8")])
                    i += len(sp)
                    matched_special = True
                    break
            if matched_special:
                continue

            # Accumulate normal text chunk until next special token
            next_sp_pos = n
            for sp in self.special_tokens:
                pos = text.find(sp, i)
                if pos != -1 and pos < next_sp_pos:
                    next_sp_pos = pos

            chunk = text[i:next_sp_pos]
            i = next_sp_pos

            # BPE encode the chunk
            chunk_tokens = [bytes([b]) for b in chunk.encode("utf-8")]
            for p1, p2 in self.merges:
                new_chunk: list[bytes] = []
                j = 0
                m = len(chunk_tokens)
                merged = p1 + p2
                while j < m:
                    if j < m - 1 and chunk_tokens[j] == p1 and chunk_tokens[j + 1] == p2:
                        new_chunk.append(merged)
                        j += 2
                    else:
                        new_chunk.append(chunk_tokens[j])
                        j += 1
                chunk_tokens = new_chunk

            for t in chunk_tokens:
                ids.append(self.stoi.get(t, self.unk_id))

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, token_ids: list[int] | Tensor, skip_special_tokens: bool = False) -> str:
        """Decode a list or tensor of token IDs back into string."""
        if hasattr(token_ids, "flatten"):
            flat = token_ids.flatten().tolist()
        elif hasattr(token_ids, "tolist"):
            flat = token_ids.tolist()
        else:
            flat = list(token_ids)

        byte_chunks: list[bytes] = []
        for tid in flat:
            tok = self.itos.get(tid, b"<unk>")
            if skip_special_tokens:
                if any(tok == sp.encode("utf-8") for sp in self.special_tokens):
                    continue
            byte_chunks.append(tok)

        raw = b"".join(byte_chunks)
        return raw.decode("utf-8", errors="replace")

    def get_fingerprint(self) -> str:
        data = {
            "version": self.version,
            "vocab": [tok.decode("latin1") for tok in self.vocab],
            "merges": [(p1.decode("latin1"), p2.decode("latin1")) for p1, p2 in self.merges],
        }
        canonical = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_type": self.tokenizer_type,
            "version": self.version,
            "vocab": [tok.decode("latin1") for tok in self.vocab],
            "merges": [(p1.decode("latin1"), p2.decode("latin1")) for p1, p2 in self.merges],
            "fingerprint": self.get_fingerprint(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BPETokenizer:
        vocab = [tok.encode("latin1") for tok in data.get("vocab", [])]
        merges = [(p1.encode("latin1"), p2.encode("latin1")) for p1, p2 in data.get("merges", [])]
        return cls(merges=merges, vocab=vocab)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
