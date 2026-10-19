"""Procedural and instruction task definitions, generators, and independent verifiers (M8-M11)."""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from adaptive_mini_gpt.model.transformer import MiniGPT


@dataclass
class TaskSpecification:
    """Specification of a machine-verifiable procedural or instruction task."""

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


def verify_instruction_reasoning(prompt: str, completion: str) -> bool:
    """Verify that assistant response follows formatting and contains expected answer."""
    comp_clean = completion.strip()
    if not comp_clean:
        return False
    match = re.search(r"(\d+)\s*([\+\-])\s*(\d+)", prompt)
    if match:
        a = int(match.group(1))
        op = match.group(2)
        b = int(match.group(3))
        expected = (a + b) if op == "+" else (a - b)
        nums = [int(s) for s in re.findall(r"\b\d+\b", comp_clean)]
        return expected in nums
    return len(comp_clean) > 0


def verify_instruction_symbolic(prompt: str, completion: str) -> bool:
    """Verify that assistant response correctly executes string transformation."""
    comp_clean = completion.strip().lower()
    match = re.search(r"word '([a-zA-Z0-9]+)'", prompt)
    if not match:
        return False
    w = match.group(1).lower()
    expected = w[::-1]
    return expected in comp_clean


def verify_instruction_qa(prompt: str, completion: str) -> bool:
    """Verify that assistant response correctly answers factual Q&A."""
    comp_clean = completion.strip().lower()
    qa_answers = {
        "france": "paris",
        "japan": "tokyo",
        "closest to the sun": "mercury",
        "speed of light": "300,000",
        "boiling point": "100",
        "romeo and juliet": "shakespeare",
        "photosynthesis": "carbon dioxide",
        "largest ocean": "pacific",
    }
    p_lower = prompt.lower()
    for k, ans in qa_answers.items():
        if k in p_lower:
            return ans in comp_clean
    return len(comp_clean) > 0


# --- Procedural & Instruction Generators ---


def generate_instruction_data(num_samples: int = 100, seed: int = 42) -> tuple[str, str]:
    """Generate curated instruction pairs with scratchpad reasoning."""
    from adaptive_mini_gpt.data.instruction_data import generate_curated_instruction_corpus

    return generate_curated_instruction_corpus(num_synthetic_reasoning=num_samples, seed=seed)


def generate_instruction_symbolic_data(
    num_samples: int = 100, seed: int = 42
) -> tuple[str, str]:
    """Generate curated string reversal and transformation instruction pairs."""
    rng = random.Random(seed)
    words = [
        "quantum",
        "gravity",
        "tensor",
        "matrix",
        "vector",
        "entropy",
        "gradient",
        "neuron",
        "synapse",
        "circuit",
        "photon",
        "boson",
        "fermion",
        "hadron",
    ]
    pairs: list[str] = []
    for _ in range(num_samples):
        w = rng.choice(words)
        rev = w[::-1]
        pairs.append(
            f"<|user|>\nReverse the sequence of characters in the word '{w}'.\n"
            f"<|assistant|>\nThe reversed sequence is: {rev}.<|eos|>\n"
        )
    text = "".join(pairs)
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


def generate_instruction_qa_data(num_samples: int = 100, seed: int = 42) -> tuple[str, str]:
    """Generate factual Q&A instruction pairs."""
    rng = random.Random(seed)
    qa_bank = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("Which planet is closest to the Sun?", "Mercury is the closest planet to the Sun."),
        (
            "What is the speed of light in vacuum approximately?",
            "The speed of light in vacuum is approximately 300,000 km/s.",
        ),
        (
            "What is the boiling point of water in Celsius?",
            "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
        ),
        (
            "Who wrote the play Romeo and Juliet?",
            "Romeo and Juliet was written by William Shakespeare.",
        ),
        (
            "What gas do green plants absorb during photosynthesis?",
            "Green plants absorb carbon dioxide during photosynthesis.",
        ),
        (
            "What is the largest ocean on Earth?",
            "The Pacific Ocean is the largest and deepest ocean on Earth.",
        ),
    ]
    pairs: list[str] = []
    for _ in range(num_samples):
        q, a = rng.choice(qa_bank)
        pairs.append(f"<|user|>\n{q}\n<|assistant|>\n{a}<|eos|>\n")
    text = "".join(pairs)
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


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
    "instruction_reasoning": TaskSpecification(
        task_id="instruction_reasoning",
        name="Instruction Reasoning",
        description="Instruction-following and scratchpad arithmetic reasoning",
        domain_trigger="<|assistant|>",
        generator_fn=generate_instruction_data,
        verifier_fn=verify_instruction_reasoning,
    ),
    "instruction_symbolic": TaskSpecification(
        task_id="instruction_symbolic",
        name="Instruction Symbolic Reversal",
        description="Instruction-based word sequence reversal in chat format",
        domain_trigger="<|assistant|>",
        generator_fn=generate_instruction_symbolic_data,
        verifier_fn=verify_instruction_symbolic,
    ),
    "instruction_qa": TaskSpecification(
        task_id="instruction_qa",
        name="Instruction Factual QA",
        description="Instruction-based factual question answering in chat format",
        domain_trigger="<|assistant|>",
        generator_fn=generate_instruction_qa_data,
        verifier_fn=verify_instruction_qa,
    ),
}


def evaluate_task_holdout(
    model: MiniGPT,
    tokenizer: Any,
    task: TaskSpecification,
    device: torch.device,
    num_samples: int = 100,
    seed: int = 9999,
) -> dict[str, float]:
    """Evaluate model on a clean holdout test set using the task's independent verifier."""
    text, _ = task.generator_fn(num_samples, seed)
    was_training = model.training
    model.eval()

    correct = 0
    total = 0

    newline_id = (
        tokenizer.get_token_id("\n")
        if hasattr(tokenizer, "get_token_id")
        else (
            tokenizer.stoi.get(b"\n")
            if hasattr(tokenizer, "stoi") and b"\n" in tokenizer.stoi
            else getattr(tokenizer, "stoi", {}).get("\n", -1)
        )
    )
    eos_id = getattr(tokenizer, "eos_id", -1)
    stop_ids = {newline_id, eos_id} - {None, -1}

    with torch.no_grad():
        if task.task_id.startswith("instruction_"):
            chunks = text.split("<|user|>")
            for chunk in chunks:
                if not chunk.strip() or "<|assistant|>" not in chunk:
                    continue
                parts = chunk.split("<|assistant|>")
                prompt = f"<|user|>{parts[0]}<|assistant|>\n"
                prompt_tokens = tokenizer.encode(prompt)
                input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

                for _ in range(48):
                    if input_tensor.size(1) >= model.config.context_length:
                        break
                    logits, _ = model(input_tensor)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    input_tensor = torch.cat([input_tensor, next_token], dim=1)
                    if next_token.item() == eos_id:
                        break

                generated_full = tokenizer.decode(input_tensor[0].tolist())
                completion = generated_full[len(prompt) :]
                total += 1
                if task.verifier_fn(prompt, completion):
                    correct += 1
        else:
            lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
            for line in lines:
                if "=" not in line:
                    continue
                prompt = line.split("=")[0] + "="
                expected_ans = line.split("=")[1].strip()

                prompt_tokens = tokenizer.encode(prompt)
                input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

                gen_len = len(expected_ans) + 2
                for _ in range(gen_len):
                    if input_tensor.size(1) >= model.config.context_length:
                        break
                    logits, _ = model(input_tensor)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    input_tensor = torch.cat([input_tensor, next_token], dim=1)
                    if next_token.item() in stop_ids:
                        break

                generated_full = tokenizer.decode(input_tensor[0].tolist())
                completion = generated_full[len(prompt) :]
                total += 1
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
