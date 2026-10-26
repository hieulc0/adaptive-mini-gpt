"""Milestone M103-D Number-Aware & Scratchpad Text Tokenizer.

Extends ByteTextTokenizerM103 with:
1. Structured numerical tokens (integers 0-99 mapped to dedicated token IDs 900-999).
2. Intermediate scratchpad / chain-of-thought token delimiter (<scratchpad> = 264).
3. Backwards-compatible decoding: maps number tokens 900-999 directly back to digit strings.
4. Flexible formatting supporting:
   - Mode 1: raw_bytes (standard UTF-8 byte digits)
   - Mode 2: structured_tokens (atomic integer tokens)
   - Mode 3: scratchpad (intermediate calculation trace)
"""

from __future__ import annotations

import re
import torch

from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)


class NumberAwareTokenizerM103_D(ByteTextTokenizerM103):
    """Tokenizer supporting byte-level text, atomic number tokens, and scratchpads."""

    NUM_TOKEN_BASE = 900
    NUM_TOKEN_MAX = 999  # Covers integers 0..99
    SCRATCHPAD_ID = 264

    def __init__(self, vocab_size: int = 1024) -> None:
        super().__init__(vocab_size=vocab_size)
        self.SPECIAL_TOKENS["<scratchpad>"] = self.SCRATCHPAD_ID
        self.REVERSE_SPECIAL_TOKENS[self.SCRATCHPAD_ID] = "<scratchpad>"

        # Map 900-999 to <num_0>..<num_99>
        for n in range(100):
            tid = self.NUM_TOKEN_BASE + n
            tag = f"<num_{n}>"
            self.SPECIAL_TOKENS[tag] = tid
            self.REVERSE_SPECIAL_TOKENS[tid] = tag

    def is_number_token(self, token_id: int) -> bool:
        """Check if token ID corresponds to an atomic number."""
        return self.NUM_TOKEN_BASE <= token_id <= self.NUM_TOKEN_MAX

    def number_token_to_int(self, token_id: int) -> int:
        """Convert atomic number token ID back to integer."""
        if not self.is_number_token(token_id):
            raise ValueError(f"Token {token_id} is not an atomic number token")
        return token_id - self.NUM_TOKEN_BASE

    def int_to_number_token(self, n: int) -> int:
        """Convert integer 0..99 to token ID."""
        if 0 <= n <= 99:
            return self.NUM_TOKEN_BASE + n
        raise ValueError(f"Integer {n} out of structured range [0..99]")

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        use_number_tokens: bool = False,
    ) -> list[int]:
        """Encode text, optionally mapping integers 0-99 to single number tokens."""
        if not use_number_tokens:
            return super().encode(text, add_bos=add_bos, add_eos=add_eos)

        tokens: list[int] = []
        if add_bos:
            tokens.append(self.BOS_ID)

        # Tokenize with regex splitting on integer numbers 0..99
        # Pattern: match standalone numbers 0..99
        parts = re.split(r"(?<!\w)(\d+)(?!\w)", text)
        for part in parts:
            if not part:
                continue
            if part.isdigit():
                val = int(part)
                if 0 <= val <= 99:
                    tokens.append(self.int_to_number_token(val))
                else:
                    # Numbers outside 0..99 fallback to raw bytes
                    tokens.extend(part.encode("utf-8"))
            else:
                tokens.extend(part.encode("utf-8"))

        if add_eos:
            tokens.append(self.EOS_ID)
        return tokens

    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to text, converting number tokens to digit strings."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()

        result_parts: list[str] = []
        byte_buffer: list[int] = []

        def flush_bytes():
            if byte_buffer:
                result_parts.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for tid in token_ids:
            if self.is_number_token(tid):
                flush_bytes()
                result_parts.append(str(self.number_token_to_int(tid)))
                continue

            if tid in self.REVERSE_SPECIAL_TOKENS:
                flush_bytes()
                if not skip_special_tokens:
                    result_parts.append(self.REVERSE_SPECIAL_TOKENS[tid])
                continue

            if 0 <= tid < 256:
                byte_buffer.append(tid)

        flush_bytes()
        return "".join(result_parts)

    def format_qa_sequence(
        self,
        question: str,
        answer: str,
        explanation: str | None = None,
        scratchpad: str | None = None,
        use_number_tokens: bool = False,
    ) -> tuple[list[int], int]:
        """Format an autoregressive training sequence with prompt masking boundary.

        If scratchpad is provided, the answer format becomes:
            [Prompt] <scratchpad> {scratchpad} <answer> {answer} <eos>
        Otherwise:
            [Prompt] <answer> {answer} <eos>
        """
        prefix_tokens: list[int] = [self.BOS_ID]

        if explanation is not None and explanation.strip():
            prefix_tokens.append(self.RULE_ID)
            rule_tokens = self.encode(explanation.strip(), use_number_tokens=use_number_tokens)
            prefix_tokens.extend(rule_tokens)
            prefix_tokens.append(self.SEP_ID)

        prefix_tokens.append(self.QUESTION_ID)
        q_tokens = self.encode(question.strip(), use_number_tokens=use_number_tokens)
        prefix_tokens.extend(q_tokens)

        if scratchpad is not None and scratchpad.strip():
            prefix_tokens.append(self.SCRATCHPAD_ID)
            prompt_len = len(prefix_tokens)

            # Target includes scratchpad followed by answer
            scratch_tokens = self.encode(scratchpad.strip(), use_number_tokens=use_number_tokens)
            ans_tokens = self.encode(answer.strip(), use_number_tokens=use_number_tokens)
            full_tokens = (
                prefix_tokens
                + scratch_tokens
                + [self.ANSWER_ID]
                + ans_tokens
                + [self.EOS_ID]
            )
        else:
            prefix_tokens.append(self.ANSWER_ID)
            prompt_len = len(prefix_tokens)
            ans_tokens = self.encode(answer.strip(), use_number_tokens=use_number_tokens)
            full_tokens = prefix_tokens + ans_tokens + [self.EOS_ID]

        return full_tokens, prompt_len
