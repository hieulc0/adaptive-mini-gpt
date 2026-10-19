"""Natural language and instruction-following evaluation suite with lineage checking (M9)."""

from __future__ import annotations

from typing import Any

import torch

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.transformer import MiniGPT


def validate_checkpoint_tokenizer_lineage(
    model: MiniGPT,
    tokenizer_data: dict[str, Any],
) -> None:
    """Explicitly verify model checkpoint and tokenizer share compatible vocabulary lineages."""
    tok_type = tokenizer_data.get("tokenizer_type", "character")
    tok_vocab_size = len(tokenizer_data.get("vocab", []))

    if tok_vocab_size != model.config.vocab_size:
        raise ValueError(
            f"Lineage Mismatch: Checkpoint vocabulary size ({model.config.vocab_size}) "
            f"does not match tokenizer vocabulary size ({tok_vocab_size}). "
            f"Old character embeddings must not be silently reinterpreted as subword embeddings."
        )

    if tok_type == "bpe" and getattr(model.config, "tokenizer_type", None) == "character":
        raise ValueError(
            "Lineage Mismatch: Cannot load BPE tokenizer onto a character-model "
            "checkpoint without migration."
        )


def evaluate_instruction_following(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    test_queries: list[dict[str, str]],
    device: torch.device,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    """Evaluate instruction-following compliance and reasoning accuracy on held-out queries."""
    model.eval()
    was_training = model.training

    results: list[dict[str, Any]] = []
    format_compliant = 0
    correct_answers = 0

    with torch.no_grad():
        for item in test_queries:
            prompt = f"<|user|>\n{item['instruction']}\n<|assistant|>\n"
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

            for _ in range(max_new_tokens):
                if input_tensor.size(1) >= model.config.context_length:
                    break
                logits, _ = model(input_tensor)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_tensor = torch.cat([input_tensor, next_tok], dim=1)

                if next_tok.item() == tokenizer.eos_id:
                    break

            generated_full = tokenizer.decode(input_tensor[0].tolist())
            completion = generated_full[len(prompt) :]

            # Check format compliance: stopped properly or produced non-empty output
            is_compliant = len(completion.strip()) > 0 and (
                "<|eos|>" in generated_full or input_tensor.size(1) < model.config.context_length
            )
            if is_compliant:
                format_compliant += 1

            # Check expected substring if provided
            expected = item.get("expected_contains")
            is_correct = False
            if expected:
                is_correct = expected.lower() in completion.lower()
                if is_correct:
                    correct_answers += 1

            results.append(
                {
                    "instruction": item["instruction"],
                    "completion": completion.replace("<|eos|>", "").strip(),
                    "compliant": is_compliant,
                    "correct": is_correct,
                }
            )

    if was_training:
        model.train()

    total = len(test_queries)
    return {
        "format_compliance_rate": round(format_compliant / total, 4) if total > 0 else 0.0,
        "exact_answer_accuracy": round(correct_answers / total, 4) if total > 0 else 0.0,
        "total_evaluated": total,
        "sample_outputs": results[:5],
    }
