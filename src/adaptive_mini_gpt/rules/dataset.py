"""Dataset generation for Verified Rules and Reasoning Loop (M14).

Generates partitioned, unfamiliar natural-language questions across 4 operations:
- 'reverse'
- 'add'
- 'compare'
- 'average_speed'

Maintains strictly disjoint train (70%), val (15%), and test (15%) splits via deterministic cryptographic partitioning.
"""

from __future__ import annotations

import hashlib
import random
import string
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.rules.loop import format_rule_call


@dataclass
class RuleProblem:
    """A single rule-reasoning problem instance with metadata."""

    question: str
    rule_name: str
    args: dict[str, Any]
    expected_answer: str
    split: str

    def to_tool_pair(self) -> str:
        """Instruction pair with ground truth tool call as assistant completion."""
        call_str = format_rule_call(self.rule_name, self.args)
        return f"<|user|>\n{self.question}\n<|assistant|>\n{call_str}<|eos|>\n"

    def to_direct_pair(self) -> str:
        """Instruction pair with direct answer as assistant completion (neural baseline)."""
        return f"<|user|>\n{self.question}\n<|assistant|>\n{self.expected_answer}<|eos|>\n"


def get_rule_split(key: str) -> str:
    """Deterministically partition key into train (70%), val (15%), or test (15%)."""
    h = int(hashlib.sha256(f"m14_rule_salt:{key}".encode()).hexdigest()[:8], 16)
    bucket = h % 100
    if bucket < 70:
        return "train"
    elif bucket < 85:
        return "val"
    else:
        return "test"


def generate_rule_problems(
    num_samples_per_rule: int = 50,
    seed: int = 42,
    split: str = "train",
) -> list[RuleProblem]:
    """Generate balanced, partitioned rule problems with varied natural language paraphrasing."""
    rng = random.Random(seed)
    problems: list[RuleProblem] = []

    # -------------------------------------------------------------
    # 1. Reverse Problems
    # -------------------------------------------------------------
    chars = string.ascii_lowercase + string.digits
    reverse_templates = [
        "Write '{w}' backward.",
        "Reverse the sequence of characters in the word '{w}'.",
        "Invert the character string '{w}'.",
        "Spell the word '{w}' in reverse order.",
        "What is the reverse of '{w}'?",
    ]
    seen_words: set[str] = set()
    rev_count = 0
    attempts = 0
    while rev_count < num_samples_per_rule and attempts < num_samples_per_rule * 200:
        attempts += 1
        length = rng.randint(4, 7)
        w = "".join(rng.choice(chars) for _ in range(length))
        if w in seen_words:
            continue
        if get_rule_split(f"rev_{w}") != split:
            continue
        seen_words.add(w)

        template = rng.choice(reverse_templates)
        q = template.format(w=w)
        expected = w[::-1]
        problems.append(
            RuleProblem(
                question=q,
                rule_name="reverse",
                args={"s": w},
                expected_answer=expected,
                split=split,
            )
        )
        rev_count += 1

    # -------------------------------------------------------------
    # 2. Add Problems
    # -------------------------------------------------------------
    add_templates = [
        "What is {a} plus {b}?",
        "Calculate the sum of {a} and {b}.",
        "Add {a} and {b} together.",
        "What is the total of {a} and {b}?",
        "Compute {a} + {b}.",
    ]
    seen_adds: set[tuple[int, int]] = set()
    add_count = 0
    attempts = 0
    while add_count < num_samples_per_rule and attempts < num_samples_per_rule * 200:
        attempts += 1
        a = rng.randint(100, 899)
        b = rng.randint(100, 899)
        pair = (min(a, b), max(a, b))
        if pair in seen_adds:
            continue
        if get_rule_split(f"add_{pair[0]}_{pair[1]}") != split:
            continue
        seen_adds.add(pair)

        template = rng.choice(add_templates)
        q = template.format(a=a, b=b)
        expected = str(a + b)
        problems.append(
            RuleProblem(
                question=q,
                rule_name="add",
                args={"a": a, "b": b},
                expected_answer=expected,
                split=split,
            )
        )
        add_count += 1

    # -------------------------------------------------------------
    # 3. Compare Problems
    # -------------------------------------------------------------
    name_pool = [
        "Anna", "Ben", "Chris", "David", "Emma", "Frank", "Grace", "Henry",
        "Ivy", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Paul",
        "Queen", "Ryan", "Sophia", "Tom", "Uma", "Victor", "Wendy", "Zack",
    ]
    attributes = [
        ("taller", "taller"),
        ("faster", "faster"),
        ("heavier", "heavier"),
        ("older", "older"),
        ("stronger", "stronger"),
    ]
    seen_compares: set[tuple[str, str, str]] = set()
    comp_count = 0
    attempts = 0
    while comp_count < num_samples_per_rule and attempts < num_samples_per_rule * 200:
        attempts += 1
        triple = tuple(rng.sample(name_pool, 3))
        if triple in seen_compares:
            continue
        if get_rule_split(f"cmp_{triple[0]}_{triple[1]}_{triple[2]}") != split:
            continue
        seen_compares.add(triple)

        e_a, e_mid, e_b = triple[0], triple[1], triple[2]
        adj_word, sup_word = rng.choice(attributes)

        templates = [
            f"If {e_a} is {adj_word} than {e_mid} and {e_mid} is {adj_word} than {e_b}, who is {sup_word}, {e_a} or {e_b}?",
            f"Suppose {e_a} is {adj_word} than {e_mid} and {e_mid} is {adj_word} than {e_b}. Between {e_a} and {e_b}, who is {sup_word}?",
            f"Given that {e_a} is {adj_word} than {e_mid} and {e_mid} is {adj_word} than {e_b}, which one is {sup_word}, {e_a} or {e_b}?",
        ]
        q = rng.choice(templates)
        relations = [f"{e_a} > {e_mid}", f"{e_mid} > {e_b}"]
        problems.append(
            RuleProblem(
                question=q,
                rule_name="compare",
                args={"entity_a": e_a, "entity_b": e_b, "relations": relations},
                expected_answer=e_a,
                split=split,
            )
        )
        comp_count += 1

    # -------------------------------------------------------------
    # 4. Average Speed Problems
    # -------------------------------------------------------------
    speed_templates = [
        "A train travels {d} km in {t} hours. Calculate average speed.",
        "A car covers a distance of {d} km in {t} hours. What is its average speed?",
        "An airplane flies {d} km in {t} hours. Find the average speed.",
        "Calculate the average speed of a vehicle traveling {d} km in {t} hours.",
        "If a runner covers {d} km in {t} hours, what is the average speed?",
    ]
    seen_speeds: set[tuple[int, int]] = set()
    speed_count = 0
    attempts = 0
    while speed_count < num_samples_per_rule and attempts < num_samples_per_rule * 200:
        attempts += 1
        t = rng.randint(2, 6)
        base_speed = rng.choice([40, 50, 60, 75, 80, 90, 100, 120])
        d = base_speed * t
        pair = (d, t)
        if pair in seen_speeds:
            continue
        if get_rule_split(f"speed_{d}_{t}") != split:
            continue
        seen_speeds.add(pair)

        template = rng.choice(speed_templates)
        q = template.format(d=d, t=t)
        expected = f"{base_speed} km/h"
        problems.append(
            RuleProblem(
                question=q,
                rule_name="average_speed",
                args={"distance_km": float(d), "time_hours": float(t)},
                expected_answer=expected,
                split=split,
            )
        )
        speed_count += 1

    rng.shuffle(problems)
    return problems


def generate_rule_instruction_corpus(
    num_samples_per_rule: int = 50,
    seed: int = 42,
    split: str = "train",
    format_type: str = "tool",
) -> tuple[str, str]:
    """Generate training or evaluation text corpus of rule problem pairs.

    - format_type='tool': Action: rule(...)
    - format_type='direct': direct answer for neural baseline
    """
    problems = generate_rule_problems(num_samples_per_rule, seed=seed, split=split)
    pairs = [
        p.to_tool_pair() if format_type == "tool" else p.to_direct_pair() for p in problems
    ]
    text = "".join(pairs)
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp
