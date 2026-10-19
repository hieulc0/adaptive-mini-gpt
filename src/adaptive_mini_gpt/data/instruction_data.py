"""Curated high-quality natural language and instruction-following dataset (M9)."""

from __future__ import annotations

import hashlib
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset

# Curated, public-domain / MIT seed instruction and reasoning pairs
SEED_INSTRUCTION_EXAMPLES: list[dict[str, str]] = [
    {
        "instruction": "What is the capital of France?",
        "response": "The capital of France is Paris.",
    },
    {
        "instruction": "Explain what a prime number is.",
        "response": (
            "A prime number is a natural number greater than 1 that cannot be formed by "
            "multiplying two smaller natural numbers. It is divisible only by 1 and itself."
        ),
    },
    {
        "instruction": "Calculate 48 + 37 with step-by-step reasoning.",
        "response": (
            "To calculate 48 + 37:\n"
            "1. Add the tens: 40 + 30 = 70.\n"
            "2. Add the units: 8 + 7 = 15.\n"
            "3. Combine the sums: 70 + 15 = 85.\n"
            "Therefore, 48 + 37 = 85."
        ),
    },
    {
        "instruction": "Calculate 92 - 45 with step-by-step reasoning.",
        "response": (
            "To calculate 92 - 45:\n"
            "1. Subtract the tens: 92 - 40 = 52.\n"
            "2. Subtract the units: 52 - 5 = 47.\n"
            "Therefore, 92 - 45 = 47."
        ),
    },
    {
        "instruction": "What is Python list comprehension?",
        "response": (
            "List comprehension in Python provides a concise syntactic way to create lists "
            "based on existing iterables, for example: [x * 2 for x in numbers if x > 0]."
        ),
    },
    {
        "instruction": "Is 1024 divisible by 8?",
        "response": (
            "Yes. 1024 / 8 = 128 with no remainder. A number is divisible by 8 if its last three "
            "digits form a number divisible by 8, and 024 is divisible by 8."
        ),
    },
    {
        "instruction": "Summarize what an API is.",
        "response": (
            "An API (Application Programming Interface) is a set of defined rules and "
            "specifications that enables different software programs to communicate."
        ),
    },
    {
        "instruction": "Solve for x in the equation 2x + 6 = 14.",
        "response": (
            "To solve 2x + 6 = 14:\n"
            "1. Subtract 6 from both sides: 2x = 8.\n"
            "2. Divide both sides by 2: x = 4.\n"
            "Therefore, x = 4."
        ),
    },
    {
        "instruction": "What is the function of the transformer attention mechanism?",
        "response": (
            "The attention mechanism computes dynamic affinity weights between all pairs of tokens "
            "in a sequence, allowing the model to aggregate relevant context across distance."
        ),
    },
    {
        "instruction": "Reverse the word 'algorithm'.",
        "response": "The reverse of 'algorithm' is 'mhtirogla'.",
    },
]


def format_instruction_pair(instruction: str, response: str) -> str:
    """Format an instruction-response pair with reserved special tokens."""
    return f"<|user|>\n{instruction.strip()}\n<|assistant|>\n{response.strip()}<|eos|>\n"


def generate_curated_instruction_corpus(
    num_synthetic_reasoning: int = 200, seed: int = 42
) -> tuple[str, str]:
    """Generate a high-quality curated instruction corpus with provenance hash."""
    rng = random.Random(seed)
    pairs: list[str] = []

    # 1. Base curated knowledge examples
    for ex in SEED_INSTRUCTION_EXAMPLES:
        pairs.append(format_instruction_pair(ex["instruction"], ex["response"]))

    # 2. Procedural reasoning arithmetic with scratchpads
    for _ in range(num_synthetic_reasoning):
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        if rng.random() > 0.5:
            # Addition
            ans = a + b
            tens_a, units_a = (a // 10) * 10, a % 10
            tens_b, units_b = (b // 10) * 10, b % 10
            resp = (
                f"To add {a} + {b}:\n"
                f"1. {tens_a} + {tens_b} = {tens_a + tens_b}\n"
                f"2. {units_a} + {units_b} = {units_a + units_b}\n"
                f"Result: {ans}."
            )
            pairs.append(format_instruction_pair(f"What is {a} + {b}?", resp))
        else:
            # Subtraction
            if a < b:
                a, b = b, a
            ans = a - b
            resp = f"To subtract {a} - {b}:\nSubtract tens and units: {a} - {b} = {ans}."
            pairs.append(format_instruction_pair(f"What is {a} - {b}?", resp))

    text = "".join(pairs)
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


class InstructionDataset(Dataset):
    """Dataset with loss masking on prompt tokens (only assistant tokens are supervised)."""

    def __init__(
        self,
        token_ids: list[int] | Tensor,
        context_length: int,
        assistant_token_id: int,
        eos_token_id: int,
        user_token_id: int,
        ignore_index: int = -1,
    ) -> None:
        if isinstance(token_ids, Tensor):
            tokens = token_ids.view(-1)
        else:
            tokens = torch.tensor(token_ids, dtype=torch.long)

        if len(tokens) <= context_length:
            raise ValueError(
                f"Dataset size ({len(tokens)}) must be strictly greater than "
                f"context_length ({context_length})"
            )

        self.tokens = tokens
        self.context_length = context_length
        self.assistant_token_id = assistant_token_id
        self.eos_token_id = eos_token_id
        self.user_token_id = user_token_id
        self.ignore_index = ignore_index

        # Precompute 1D assistant response mask
        n = len(tokens)
        mask = torch.zeros(n, dtype=torch.bool)
        is_assistant = False
        tok_list = tokens.tolist()
        for i, t in enumerate(tok_list):
            if t == assistant_token_id:
                is_assistant = True
            elif t == eos_token_id or t == user_token_id:
                is_assistant = False
            mask[i] = is_assistant

        # Targets are tokens at next position where previous token is in assistant response
        self.targets = tokens.clone()
        valid_targets = torch.zeros_like(mask)
        valid_targets[1:] = mask[:-1]
        self.targets[~valid_targets] = ignore_index

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = self.tokens[idx : idx + self.context_length]
        y = self.targets[idx + 1 : idx + self.context_length + 1]
        return x, y

    def get_batch(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        max_start = len(self.tokens) - self.context_length
        ix = torch.randint(0, max_start, (batch_size,), generator=generator)
        x = torch.stack([self.tokens[i : i + self.context_length] for i in ix]).to(device)
        y = torch.stack([self.targets[i + 1 : i + self.context_length + 1] for i in ix]).to(device)
        return x, y

    def get_sequential_batch(
        self,
        start_idx: int,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[Tensor, Tensor, int]:
        max_start = len(self.tokens) - self.context_length
        ix = [(start_idx + i * self.context_length) % max_start for i in range(batch_size)]
        x = torch.stack([self.tokens[i : i + self.context_length] for i in ix]).to(device)
        y = torch.stack([self.targets[i + 1 : i + self.context_length + 1] for i in ix]).to(device)
        next_start = (start_idx + batch_size * self.context_length) % max_start
        return x, y, next_start
