"""Milestone M103 Minimal Text Tokenizer.

Provides deterministic, zero-OOV byte-level text tokenization with dedicated
special tokens for rule explanations, questions, and autoregressive answers.
Compatible with MiniGPT vocabulary (vocab_size=1024).
"""

from __future__ import annotations

import torch


class ByteTextTokenizerM103:
    """Byte-level UTF-8 text tokenizer with structured prompt delimiters."""

    PAD_ID = 256
    BOS_ID = 257
    EOS_ID = 258
    UNK_ID = 259
    RULE_ID = 260
    QUESTION_ID = 261
    ANSWER_ID = 262
    SEP_ID = 263

    SPECIAL_TOKENS: dict[str, int] = {
        "<pad>": PAD_ID,
        "<bos>": BOS_ID,
        "<eos>": EOS_ID,
        "<unk>": UNK_ID,
        "<rule>": RULE_ID,
        "<question>": QUESTION_ID,
        "<answer>": ANSWER_ID,
        "<sep>": SEP_ID,
    }

    REVERSE_SPECIAL_TOKENS: dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}

    def __init__(self, vocab_size: int = 1024) -> None:
        self.vocab_size = vocab_size
        self.num_special = len(self.SPECIAL_TOKENS)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode UTF-8 text into a sequence of token IDs."""
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.BOS_ID)

        raw_bytes = text.encode("utf-8")
        tokens.extend(raw_bytes)

        if add_eos:
            tokens.append(self.EOS_ID)
        return tokens

    def encode_prompt(
        self,
        question: str,
        explanation: str | None = None,
        max_length: int = 256,
    ) -> list[int]:
        """Encode structured boundaries without silently discarding observable evidence.

        Raw ``encode`` deliberately remains byte-only. Callers must choose an
        explicit context policy if the complete prompt does not fit.
        """
        if not question.strip():
            raise ValueError("A complete observation requires a nonempty question")
        prompt = [self.BOS_ID]
        if explanation is not None and explanation.strip():
            prompt += [self.RULE_ID] + self.encode(explanation) + [self.SEP_ID]
        prompt += [self.QUESTION_ID] + self.encode(question) + [self.ANSWER_ID]
        if len(prompt) > max_length:
            raise ValueError(
                f"Complete prompt length {len(prompt)} exceeds context budget {max_length}; "
                "refusing to truncate entity or causal evidence"
            )
        return prompt

    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode a sequence of token IDs back into a UTF-8 text string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()

        byte_list: list[int] = []
        for tid in token_ids:
            if tid in self.REVERSE_SPECIAL_TOKENS:
                if not skip_special_tokens:
                    # Append special token string encoded in UTF-8
                    spec_str = self.REVERSE_SPECIAL_TOKENS[tid]
                    byte_list.extend(spec_str.encode("utf-8"))
                continue
            if 0 <= tid < 256:
                byte_list.append(tid)

        return bytes(byte_list).decode("utf-8", errors="replace")

    def format_qa_sequence(
        self,
        question: str,
        answer: str,
        explanation: str | None = None,
    ) -> tuple[list[int], int]:
        """Format an autoregressive training sequence: [Rule] [Question] [Answer] <eos>.

        Returns:
            (full_token_ids, prompt_len)
            The prompt_len indicates the boundary before answer tokens,
            enabling prompt masking in the next-token cross-entropy loss.
        """
        prefix_tokens: list[int] = [self.BOS_ID]

        if explanation is not None and explanation.strip():
            prefix_tokens.append(self.RULE_ID)
            prefix_tokens.extend(explanation.strip().encode("utf-8"))
            prefix_tokens.append(self.SEP_ID)

        prefix_tokens.append(self.QUESTION_ID)
        prefix_tokens.extend(question.strip().encode("utf-8"))
        prefix_tokens.append(self.ANSWER_ID)

        prompt_len = len(prefix_tokens)

        answer_tokens = list(answer.strip().encode("utf-8"))
        full_tokens = prefix_tokens + answer_tokens + [self.EOS_ID]
        return full_tokens, prompt_len

    def pad_batch(
        self,
        token_lists: list[list[int]],
        max_length: int | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad a batch of token lists into a rectangular tensor.

        Returns:
            (batch_tokens, attention_mask)
        """
        if not token_lists:
            t = torch.empty((0, 0), dtype=torch.long, device=device)
            return t, t

        max_len = max_length if max_length is not None else max(len(seq) for seq in token_lists)
        if max_len <= 0 or any(len(seq) > max_len for seq in token_lists):
            raise ValueError("Complete sequence exceeds padding budget; refusing silent truncation")
        batch_size = len(token_lists)

        padded = torch.full((batch_size, max_len), self.PAD_ID, dtype=torch.long, device=device)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

        for i, seq in enumerate(token_lists):
            trunc = seq
            padded[i, :len(trunc)] = torch.tensor(trunc, dtype=torch.long, device=device)
            mask[i, :len(trunc)] = True

        return padded, mask
