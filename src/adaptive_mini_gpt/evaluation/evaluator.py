"""Evaluation, exact-match skill verification, and generation utilities."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from adaptive_mini_gpt.data.dataset import TokenDataset, is_holdout_arithmetic_pair
from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer
from adaptive_mini_gpt.model.transformer import MiniGPT


@torch.no_grad()
def evaluate_dataset(
    model: MiniGPT,
    dataset: TokenDataset,
    batch_size: int,
    device: torch.device | str = "cpu",
    max_batches: int = 50,
    task_id: int | None = None,
    bypass_moe: bool = False,
) -> dict[str, float]:
    """Evaluate model loss, perplexity, and bits-per-character without redundant wrapping."""
    model.eval()
    total_loss = 0.0
    batches_run = 0

    tokens_per_batch = batch_size * model.config.context_length
    natural_batches = max(1, math.ceil(len(dataset) / tokens_per_batch))
    effective_batches = min(max_batches, natural_batches)

    for i in range(effective_batches):
        start_idx = i * tokens_per_batch
        x, y, _ = dataset.get_sequential_batch(
            start_idx=start_idx,
            batch_size=batch_size,
            device=device,
        )
        _, loss = model(x, y, task_id=task_id, bypass_moe=bypass_moe)
        if loss is not None and model.last_aux_info.get("ce_loss", 0.0) > 0.0:
            total_loss += model.last_aux_info["ce_loss"]
            batches_run += 1
        elif loss is not None:
            total_loss += loss.item()
            batches_run += 1

    mean_loss = total_loss / batches_run if batches_run > 0 else 0.0
    perplexity = math.exp(min(mean_loss, 20.0))
    bpc = mean_loss / math.log(2)

    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "bits_per_character": bpc,
        "batches_evaluated": batches_run,
    }


@torch.no_grad()
def evaluate_arithmetic_accuracy(
    model: MiniGPT,
    tokenizer: CharacterTokenizer,
    num_problems: int = 100,
    seed: int = 999,
    device: torch.device | str = "cpu",
    task_id: int | None = None,
    ood_extrapolation: bool = False,
    bypass_moe: bool = False,
) -> dict[str, Any]:
    """Verify exact-match greedy accuracy on strictly disjoint held-out arithmetic problems."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    correct = 0
    tested = 0
    samples: list[dict[str, str]] = []

    low = 100 if ood_extrapolation else 0
    high = 200 if ood_extrapolation else 100

    while tested < num_problems:
        op_type = int(torch.randint(0, 3, (1,), generator=g).item())
        a = int(torch.randint(low, high, (1,), generator=g).item())
        b = int(torch.randint(low, high, (1,), generator=g).item())
        if op_type == 1:
            a, b = max(a, b), min(a, b)
        elif op_type == 2:
            b = 0

        if not ood_extrapolation and not is_holdout_arithmetic_pair(op_type, a, b):
            continue

        if op_type == 0:
            prompt = f"ADD {a}+{b}="
            expected = str(a + b)
        elif op_type == 1:
            prompt = f"SUB {a}-{b}="
            expected = str(a - b)
        else:
            prompt = f"PAR {a}="
            expected = "EVEN" if (a % 2 == 0) else "ODD"

        res = generate_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=6,
            temperature=0.0,  # Greedy decoding for verification
            stop_on_newline=True,
            device=device,
            task_id=task_id,
            bypass_moe=bypass_moe,
        )
        pred = res["completion"].strip()
        if pred == expected:
            correct += 1
        if len(samples) < 5:
            samples.append({"prompt": prompt, "expected": expected, "predicted": pred})
        tested += 1

    return {
        "exact_match_accuracy": round(correct / max(1, num_problems), 4),
        "correct": correct,
        "total_problems": num_problems,
        "ood_extrapolation": ood_extrapolation,
        "sample_predictions": samples,
    }


@torch.no_grad()
def generate_sample(
    model: MiniGPT,
    tokenizer: CharacterTokenizer,
    prompt: str = "The ",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    stop_on_newline: bool = False,
    device: torch.device | str = "cpu",
    task_id: int | None = None,
    bypass_moe: bool = False,
) -> dict[str, Any]:
    """Generate structured text completion from a prompt using the model and tokenizer."""
    model.eval()
    encoded = tokenizer.encode(prompt)
    if not encoded:
        encoded = [0]

    newline_ids = tokenizer.encode("\n")
    eos_id = newline_ids[0] if (stop_on_newline and newline_ids) else None

    input_tokens = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    output_tokens: Tensor = model.generate(
        idx=input_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_id=eos_id,
        task_id=task_id,
        bypass_moe=bypass_moe,
    )

    full_list = output_tokens[0].tolist()
    completion_ids = full_list[len(encoded) :]
    generated_text = tokenizer.decode(full_list)
    completion_text = tokenizer.decode(completion_ids)

    return {
        "prompt": prompt,
        "completion": completion_text,
        "generated_text": generated_text,
        "prompt_tokens": len(encoded),
        "new_tokens": len(completion_ids),
        "total_tokens": len(full_list),
        "routing_telemetry": model.last_aux_info if model.config.adaptive_enabled else None,
    }
