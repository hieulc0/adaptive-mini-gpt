"""Procedural task definitions, generators, and independent external verifiers (M8)."""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable
from dataclasses import dataclass

import torch

from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer
from adaptive_mini_gpt.model.transformer import MiniGPT


@dataclass
class TaskSpecification:
    """Specification of a machine-verifiable procedural task."""

    task_id: str
    name: str
    description: str
    domain_trigger: str  # Token that activates specialist domain (e.g. '=')
    generator_fn: Callable[[int, int], tuple[str, str]]
    verifier_fn: Callable[[str, str], bool]


# --- Independent Ground-Truth Verifiers ---


def verify_addition(prompt: str, completion: str) -> bool:
    """Independently verify whether completion correctly solves prompt 'a+b='."""
    match = re.search(r"(\d+)\s*\+\s*(\d+)\s*=", prompt)
    if not match:
        return False
    a, b = int(match.group(1)), int(match.group(2))
    expected = str(a + b)
    comp_clean = completion.strip().split("\n")[0].split(" ")[0].strip()
    return comp_clean == expected or comp_clean.startswith(expected)


def verify_subtraction(prompt: str, completion: str) -> bool:
    """Independently verify whether completion correctly solves prompt 'a-b='."""
    match = re.search(r"(\d+)\s*-\s*(\d+)\s*=", prompt)
    if not match:
        return False
    a, b = int(match.group(1)), int(match.group(2))
    expected = str(a - b)
    comp_clean = completion.strip().split("\n")[0].split(" ")[0].strip()
    return comp_clean == expected or comp_clean.startswith(expected)


def verify_multiplication(prompt: str, completion: str) -> bool:
    """Independently verify whether completion correctly solves prompt 'a*b='."""
    match = re.search(r"(\d+)\s*\*\s*(\d+)\s*=", prompt)
    if not match:
        return False
    a, b = int(match.group(1)), int(match.group(2))
    expected = str(a * b)
    comp_clean = completion.strip().split("\n")[0].split(" ")[0].strip()
    return comp_clean == expected or comp_clean.startswith(expected)


def verify_parity(prompt: str, completion: str) -> bool:
    """Independently verify whether completion correctly predicts parity."""
    match = re.search(r"parity\s+(\d+)\s*=", prompt)
    if not match:
        match = re.search(r"(\d+)\s+is\s*=", prompt)
        if not match:
            return False
    n = int(match.group(1))
    expected = "even" if n % 2 == 0 else "odd"
    comp_clean = completion.strip().lower().split("\n")[0].split(" ")[0].strip()
    return comp_clean == expected or comp_clean.startswith(expected)


def verify_reverse(prompt: str, completion: str) -> bool:
    """Independently verify whether completion reverses a short token string."""
    match = re.search(r"rev\(([a-zA-Z0-9]+)\)\s*=", prompt)
    if not match:
        return False
    s = match.group(1)
    expected = s[::-1]
    comp_clean = completion.strip().split("\n")[0].split(" ")[0].strip()
    return comp_clean == expected or comp_clean.startswith(expected)


# --- Procedural Task Generators ---


def generate_addition_data(num_samples: int = 1000, seed: int = 42) -> tuple[str, str]:
    """Generate procedurally verified two-digit addition examples."""
    rng = random.Random(seed)
    lines: list[str] = []
    for _ in range(num_samples):
        a = rng.randint(0, 99)
        b = rng.randint(0, 99)
        lines.append(f"{a}+{b}={a+b}")
    text = "\n".join(lines) + "\n"
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


def generate_subtraction_data(num_samples: int = 1000, seed: int = 42) -> tuple[str, str]:
    """Generate procedurally verified two-digit subtraction examples."""
    rng = random.Random(seed)
    lines: list[str] = []
    for _ in range(num_samples):
        a = rng.randint(0, 99)
        b = rng.randint(0, a)  # Non-negative result
        lines.append(f"{a}-{b}={a-b}")
    text = "\n".join(lines) + "\n"
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


def generate_multiplication_data(num_samples: int = 1000, seed: int = 42) -> tuple[str, str]:
    """Generate procedurally verified single-digit multiplication examples."""
    rng = random.Random(seed)
    lines: list[str] = []
    for _ in range(num_samples):
        a = rng.randint(0, 9)
        b = rng.randint(0, 9)
        lines.append(f"{a}*{b}={a*b}")
    text = "\n".join(lines) + "\n"
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


def generate_parity_data(num_samples: int = 1000, seed: int = 42) -> tuple[str, str]:
    """Generate procedurally verified parity examples."""
    rng = random.Random(seed)
    lines: list[str] = []
    for _ in range(num_samples):
        n = rng.randint(0, 999)
        ans = "even" if n % 2 == 0 else "odd"
        lines.append(f"parity {n}={ans}")
    text = "\n".join(lines) + "\n"
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


def generate_reverse_data(num_samples: int = 1000, seed: int = 42) -> tuple[str, str]:
    """Generate procedurally verified string reversal examples."""
    rng = random.Random(seed)
    chars = "abcdefghijklmnopqrstuvwxyz"
    lines: list[str] = []
    for _ in range(num_samples):
        length = rng.randint(3, 5)
        s = "".join(rng.choice(chars) for _ in range(length))
        lines.append(f"rev({s})={s[::-1]}")
    text = "\n".join(lines) + "\n"
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


TASK_REGISTRY: dict[str, TaskSpecification] = {
    "addition": TaskSpecification(
        task_id="addition",
        name="Addition",
        description="Two-digit addition (a+b=c)",
        domain_trigger="=",
        generator_fn=generate_addition_data,
        verifier_fn=verify_addition,
    ),
    "subtraction": TaskSpecification(
        task_id="subtraction",
        name="Subtraction",
        description="Two-digit non-negative subtraction (a-b=c)",
        domain_trigger="=",
        generator_fn=generate_subtraction_data,
        verifier_fn=verify_subtraction,
    ),
    "multiplication": TaskSpecification(
        task_id="multiplication",
        name="Multiplication",
        description="Single-digit multiplication (a*b=c)",
        domain_trigger="=",
        generator_fn=generate_multiplication_data,
        verifier_fn=verify_multiplication,
    ),
    "parity": TaskSpecification(
        task_id="parity",
        name="Parity",
        description="Integer parity classification (parity n=even/odd)",
        domain_trigger="=",
        generator_fn=generate_parity_data,
        verifier_fn=verify_parity,
    ),
    "reverse": TaskSpecification(
        task_id="reverse",
        name="String Reversal",
        description="Short sequence inversion (rev(s)=reversed)",
        domain_trigger="=",
        generator_fn=generate_reverse_data,
        verifier_fn=verify_reverse,
    ),
}


def evaluate_task_holdout(
    model: MiniGPT,
    tokenizer: CharacterTokenizer,
    task: TaskSpecification,
    device: torch.device,
    num_samples: int = 100,
    seed: int = 9999,
) -> dict[str, float]:
    """Evaluate model on a clean holdout test set using the task's independent verifier."""
    text, _ = task.generator_fn(num_samples, seed)
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

    was_training = model.training
    model.eval()

    correct = 0
    total = len(lines)

    with torch.no_grad():
        for line in lines:
            if "=" not in line:
                continue
            prompt = line.split("=")[0] + "="
            expected_ans = line.split("=")[1].strip()

            prompt_tokens = tokenizer.encode(prompt)
            input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

            # Autoregressively generate next tokens up to length of expected answer + 2
            gen_len = len(expected_ans) + 2
            for _ in range(gen_len):
                if input_tensor.size(1) >= model.config.context_length:
                    break
                logits, _ = model(input_tensor)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_tensor = torch.cat([input_tensor, next_token], dim=1)
                # Break on newline
                if next_token.item() == tokenizer.stoi.get("\n", -1):
                    break

            generated_full = tokenizer.decode(input_tensor[0].tolist())
            completion = generated_full[len(prompt) :]

            if task.verifier_fn(prompt, completion):
                correct += 1

    if was_training:
        model.train()

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "correct": float(correct),
        "total": float(total),
    }
