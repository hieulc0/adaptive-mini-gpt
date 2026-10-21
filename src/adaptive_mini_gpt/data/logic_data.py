"""Symbolic Logic and Procedural Reasoning Benchmark Dataset (M25-C / M26).

Implements reproducible neural reasoning with strictly disjoint variable names,
entities, and rule compositions across train, validation, and sealed-test splits.

Features:
  - Boolean logic operations (AND, OR, NOT).
  - Transitive comparisons and ordering (A > B > C).
  - Categorical syllogisms and property deduction (Modus Ponens).
  - Variable binding across 2-step and 3-step deductive chains.
  - Contradiction detection and safe abstention on undetermined queries.
  - Strict disjoint variable names to guarantee out-of-distribution evaluation.
  - Explicit separation between unseen variable names and genuinely unseen logical compositions (M26).
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.evaluation.multi_domain import (
    BenchmarkSample,
    BenchmarkSplit,
    DomainType,
)

LOGIC_ABSTENTION_TARGET = "Undetermined."
CONTRADICTION_TARGET = "Contradiction detected."

LOGIC_ABSTAIN_TERMS = {
    "undetermined",
    "undetermined.",
    "cannot be determined",
    "insufficient information",
    "unknown",
    "contradiction",
    "contradiction detected",
    "contradiction detected.",
}


class LogicTaskCategory(str, Enum):
    BOOLEAN = "boolean_logic"
    COMPARISON = "transitive_comparison"
    DEDUCTION_MODUS_PONENS = "modus_ponens"
    CHAINED_COMPOSITION = "chained_composition"
    CONTRADICTION = "contradiction_abstention"
    # Unseen logical compositions (M26)
    MODUS_TOLLENS = "modus_tollens"
    DISJUNCTIVE_SYLLOGISM = "disjunctive_syllogism"
    BICONDITIONAL = "biconditional_equivalence"


class LogicGeneralizationType(str, Enum):
    UNSEEN_VARIABLES = "unseen_variables"
    UNSEEN_STRUCTURES = "unseen_structures"
    CONTRADICTION_ABSTENTION = "contradiction_abstention"


# =====================================================================
# Strictly Disjoint Entity & Variable Partitions
# =====================================================================

TRAIN_VARS = ["Alpha", "Beta", "Gamma", "Delta", "A", "B", "C", "D", "E"]
TRAIN_ENTITIES = ["orb", "cube", "cylinder", "prism", "token"]
TRAIN_PROPERTIES = ["red", "blue", "metallic", "heavy", "warm", "solid"]

VAL_VARS = ["Epsilon", "Zeta", "Eta", "F", "G", "H"]
VAL_ENTITIES = ["disc", "beacon", "crest", "pyramid"]
VAL_PROPERTIES = ["bright", "dense", "cold", "smooth"]

TEST_VARS = ["Theta", "Lambda", "Sigma", "Omega", "Psi", "Phi", "X", "Y", "Z"]
TEST_ENTITIES = ["monolith", "vessel", "matrix", "conduit", "relic"]
TEST_PROPERTIES = ["resonant", "inert", "fragile", "kinetic", "crystalline", "radiant"]


def format_logic_prompt(premises: list[str], question: str) -> str:
    """Format standardized symbolic logic prompt with reserved special tokens."""
    premise_str = "\n".join(f"- {p}" for p in premises)
    return (
        f"<|user|>\n"
        f"Premises:\n{premise_str}\n"
        f"Question: {question.strip()}\n"
        f"Conclude with: 'True', 'False', 'Yes', 'No', 'Undetermined', or 'Contradiction detected'.\n"
        f"<|assistant|>\n"
    )


def format_logic_completion(answer: str) -> str:
    """Format assistant target logic completion."""
    return f"{answer.strip()}<|eos|>\n"


def parse_logic_completion(completion: str) -> tuple[str, bool, bool]:
    """Parse logic completion into (cleaned_answer, is_compliant, is_abstained)."""
    cleaned = completion.replace("<|eos|>", "").strip()
    if not cleaned:
        return "", False, False

    first_line = cleaned.split("\n")[0].strip()
    norm = re.sub(r"[^\w\s]", "", first_line.lower()).strip()

    is_abstained = any(term in norm for term in ["undetermined", "cannot be determined", "insufficient", "unknown", "contradiction"])
    is_compliant = len(first_line) > 0

    return first_line, is_compliant, is_abstained


def verify_logic_answer(parsed_answer: str, sample: BenchmarkSample) -> bool:
    """Independently verify whether parsed logic conclusion matches expected ground truth."""
    if not parsed_answer:
        return False

    norm_parsed = re.sub(r"[^\w\s]", "", parsed_answer.lower()).strip()
    norm_expected = re.sub(r"[^\w\s]", "", sample.expected_answer.lower()).strip()

    # Case 1: Abstention / Contradiction expected
    if sample.is_unanswerable or "contradiction" in norm_expected or "undetermined" in norm_expected:
        if "contradiction" in norm_expected:
            return "contradiction" in norm_parsed
        return any(term in norm_parsed for term in ["undetermined", "cannot be determined", "insufficient", "unknown"])

    # If the model emitted abstention on a determinate problem, fail
    if any(term in norm_parsed for term in ["undetermined", "contradiction", "cannot be determined"]):
        return False

    # Case 2: Boolean True / False
    if norm_expected in {"true", "false"}:
        first_word = norm_parsed.split()[0] if norm_parsed.split() else ""
        if first_word in {"true", "false"}:
            return first_word == norm_expected
        return norm_expected in norm_parsed and ("not " + norm_expected) not in norm_parsed

    # Case 3: Yes / No
    if norm_expected in {"yes", "no"}:
        first_word = norm_parsed.split()[0] if norm_parsed.split() else ""
        if first_word in {"yes", "no"}:
            return first_word == norm_expected
        if norm_expected == "yes":
            return "yes" in norm_parsed and "no" not in norm_parsed
        if norm_expected == "no":
            return "no" in norm_parsed and "yes" not in norm_parsed

    return norm_expected in norm_parsed


def generate_logic_samples_for_split(
    split: BenchmarkSplit,
    count_per_category: int = 10,
    seed: int = 42,
    include_unseen_structures: bool = False,
) -> list[BenchmarkSample]:
    """Deterministically generate logic problems strictly using partitioned variables & entities."""
    if split == BenchmarkSplit.TRAIN:
        v_pool, e_pool, p_pool = TRAIN_VARS, TRAIN_ENTITIES, TRAIN_PROPERTIES
    elif split == BenchmarkSplit.VAL:
        v_pool, e_pool, p_pool = VAL_VARS, VAL_ENTITIES, VAL_PROPERTIES
    elif split == BenchmarkSplit.TEST:
        v_pool, e_pool, p_pool = TEST_VARS, TEST_ENTITIES, TEST_PROPERTIES
    else:
        raise ValueError(f"Unknown split: {split}")

    rng = random.Random(seed + (hash(split.value) % 10000))
    samples: list[BenchmarkSample] = []

    # 1. Boolean Logic
    for i in range(count_per_category):
        var_a = rng.choice(v_pool)
        var_b = rng.choice([v for v in v_pool if v != var_a])
        val_a = rng.choice([True, False])
        val_b = rng.choice([True, False])

        op = rng.choice(["AND", "OR", "NOT_A_AND_B", "A_OR_NOT_B"])
        if op == "AND":
            expected = "True" if (val_a and val_b) else "False"
            q = f"What is the truth value of ({var_a} AND {var_b})?"
        elif op == "OR":
            expected = "True" if (val_a or val_b) else "False"
            q = f"What is the truth value of ({var_a} OR {var_b})?"
        elif op == "NOT_A_AND_B":
            expected = "True" if ((not val_a) and val_b) else "False"
            q = f"What is the truth value of (NOT {var_a} AND {var_b})?"
        else:
            expected = "True" if (val_a or (not val_b)) else "False"
            q = f"What is the truth value of ({var_a} OR NOT {var_b})?"

        premises = [f"{var_a} is {val_a}.", f"{var_b} is {val_b}."]
        prompt = format_logic_prompt(premises, q)
        samples.append(
            BenchmarkSample(
                sample_id=f"{split.value}_bool_{i}",
                domain=DomainType.LOGIC,
                split=split,
                prompt=prompt,
                expected_answer=expected,
                allowed_answers=[expected],
                is_unanswerable=False,
                metadata={
                    "category": LogicTaskCategory.BOOLEAN.value,
                    "generalization_type": LogicGeneralizationType.UNSEEN_VARIABLES.value,
                    "operation": op,
                },
            )
        )

    # 2. Transitive Comparisons
    for i in range(count_per_category):
        sampled = rng.sample(v_pool, min(4, len(v_pool)))
        v1, v2, v3 = sampled[0], sampled[1], sampled[2]
        is_transitive_query = (i % 3 != 2)

        if is_transitive_query:
            premises = [f"{v1} is greater than {v2}.", f"{v2} is greater than {v3}."]
            if i % 2 == 0:
                q = f"Is {v1} greater than {v3}?"
                expected = "Yes"
                is_unans = False
            else:
                q = f"Is {v3} greater than {v1}?"
                expected = "No"
                is_unans = False
            gen_type = LogicGeneralizationType.UNSEEN_VARIABLES.value
        else:
            v4 = sampled[3]
            premises = [f"{v1} is greater than {v2}.", f"{v3} is greater than {v4}."]
            q = f"Is {v1} greater than {v3}?"
            expected = LOGIC_ABSTENTION_TARGET
            is_unans = True
            gen_type = LogicGeneralizationType.CONTRADICTION_ABSTENTION.value

        prompt = format_logic_prompt(premises, q)
        samples.append(
            BenchmarkSample(
                sample_id=f"{split.value}_comp_{i}",
                domain=DomainType.LOGIC,
                split=split,
                prompt=prompt,
                expected_answer=expected,
                allowed_answers=[expected],
                is_unanswerable=is_unans,
                metadata={
                    "category": LogicTaskCategory.COMPARISON.value,
                    "generalization_type": gen_type,
                },
            )
        )

    # 3. Modus Ponens / Categorical Property Deduction
    for i in range(count_per_category):
        ent = rng.choice(e_pool)
        prop = rng.choice(p_pool)
        var = rng.choice(v_pool)

        if i % 2 == 0:
            premises = [f"Every {ent} is {prop}.", f"Item {var} is a {ent}."]
            q = f"Is Item {var} {prop}?"
            expected = "Yes"
            is_unans = False
        else:
            other_prop = rng.choice([p for p in p_pool if p != prop])
            premises = [f"Every {ent} is {prop}.", f"Item {var} is not {prop}."]
            q = f"Is Item {var} a {ent}?"
            expected = "No"
            is_unans = False

        prompt = format_logic_prompt(premises, q)
        samples.append(
            BenchmarkSample(
                sample_id=f"{split.value}_mp_{i}",
                domain=DomainType.LOGIC,
                split=split,
                prompt=prompt,
                expected_answer=expected,
                allowed_answers=[expected],
                is_unanswerable=is_unans,
                metadata={
                    "category": LogicTaskCategory.DEDUCTION_MODUS_PONENS.value,
                    "generalization_type": LogicGeneralizationType.UNSEEN_VARIABLES.value,
                },
            )
        )

    # 4. Multi-Step Chained Composition (2-step and 3-step)
    for i in range(count_per_category):
        props = rng.sample(p_pool, 4)
        p1, p2, p3, p4 = props[0], props[1], props[2], props[3]
        var = rng.choice(v_pool)
        depth = 2 if (i % 2 == 0) else 3

        if depth == 2:
            premises = [
                f"If an object is {p1}, then it is {p2}.",
                f"If an object is {p2}, then it is {p3}.",
                f"Item {var} is {p1}.",
            ]
            q = f"Is Item {var} {p3}?"
            expected = "Yes"
        else:
            premises = [
                f"If an object is {p1}, then it is {p2}.",
                f"If an object is {p2}, then it is {p3}.",
                f"If an object is {p3}, then it is {p4}.",
                f"Item {var} is {p1}.",
            ]
            q = f"Is Item {var} {p4}?"
            expected = "Yes"

        prompt = format_logic_prompt(premises, q)
        samples.append(
            BenchmarkSample(
                sample_id=f"{split.value}_chain_{i}",
                domain=DomainType.LOGIC,
                split=split,
                prompt=prompt,
                expected_answer=expected,
                allowed_answers=[expected],
                is_unanswerable=False,
                metadata={
                    "category": LogicTaskCategory.CHAINED_COMPOSITION.value,
                    "generalization_type": LogicGeneralizationType.UNSEEN_VARIABLES.value,
                    "depth": depth,
                },
            )
        )

    # 5. Contradictions & Abstention
    for i in range(count_per_category):
        prop = rng.choice(p_pool)
        var = rng.choice(v_pool)
        other_prop = rng.choice([p for p in p_pool if p != prop])

        if i % 2 == 0:
            premises = [f"Item {var} is {prop}.", f"Item {var} is not {prop}."]
            q = f"Is Item {var} {other_prop}?"
            expected = CONTRADICTION_TARGET
            is_unans = True
        else:
            ent = rng.choice(e_pool)
            premises = [f"Every {ent} is {prop}.", f"Item {var} is {other_prop}."]
            q = f"Is Item {var} {prop}?"
            expected = LOGIC_ABSTENTION_TARGET
            is_unans = True

        prompt = format_logic_prompt(premises, q)
        samples.append(
            BenchmarkSample(
                sample_id=f"{split.value}_contra_{i}",
                domain=DomainType.LOGIC,
                split=split,
                prompt=prompt,
                expected_answer=expected,
                allowed_answers=list(LOGIC_ABSTAIN_TERMS),
                is_unanswerable=is_unans,
                metadata={
                    "category": LogicTaskCategory.CONTRADICTION.value,
                    "generalization_type": LogicGeneralizationType.CONTRADICTION_ABSTENTION.value,
                },
            )
        )

    # 6. Genuinely Unseen Logical Compositions (M26: Disjunctive Syllogism, Modus Tollens, Biconditional)
    # Generated for TEST (and optionally VAL) to measure structural generalization beyond variable substitution
    if include_unseen_structures and split in {BenchmarkSplit.VAL, BenchmarkSplit.TEST}:
        unseen_count = count_per_category // 2
        for i in range(unseen_count):
            # A. Modus Tollens (contrapositive reasoning)
            prop_a = rng.choice(p_pool)
            prop_b = rng.choice([p for p in p_pool if p != prop_a])
            var = rng.choice(v_pool)
            premises = [
                f"If Item {var} is {prop_a}, then Item {var} is {prop_b}.",
                f"Item {var} is not {prop_b}.",
            ]
            q = f"Is Item {var} {prop_a}?"
            expected = "No"
            prompt = format_logic_prompt(premises, q)
            samples.append(
                BenchmarkSample(
                    sample_id=f"{split.value}_unseen_tollens_{i}",
                    domain=DomainType.LOGIC,
                    split=split,
                    prompt=prompt,
                    expected_answer=expected,
                    allowed_answers=[expected],
                    is_unanswerable=False,
                    metadata={
                        "category": LogicTaskCategory.MODUS_TOLLENS.value,
                        "generalization_type": LogicGeneralizationType.UNSEEN_STRUCTURES.value,
                    },
                )
            )

            # B. Disjunctive Syllogism (P OR Q; NOT P -> Q)
            var_x = rng.choice(v_pool)
            var_y = rng.choice([v for v in v_pool if v != var_x])
            premises = [
                f"Either {var_x} is True or {var_y} is True.",
                f"{var_x} is False.",
            ]
            q = f"Is {var_y} True?"
            expected = "Yes"
            prompt = format_logic_prompt(premises, q)
            samples.append(
                BenchmarkSample(
                    sample_id=f"{split.value}_unseen_disjunctive_{i}",
                    domain=DomainType.LOGIC,
                    split=split,
                    prompt=prompt,
                    expected_answer=expected,
                    allowed_answers=[expected],
                    is_unanswerable=False,
                    metadata={
                        "category": LogicTaskCategory.DISJUNCTIVE_SYLLOGISM.value,
                        "generalization_type": LogicGeneralizationType.UNSEEN_STRUCTURES.value,
                    },
                )
            )

            # C. Biconditional Equivalence (P <-> Q)
            var_p = rng.choice(v_pool)
            var_q = rng.choice([v for v in v_pool if v != var_p])
            premises = [
                f"{var_p} is equivalent to {var_q}.",
                f"{var_p} is True.",
            ]
            q = f"Is {var_q} True?"
            expected = "Yes"
            prompt = format_logic_prompt(premises, q)
            samples.append(
                BenchmarkSample(
                    sample_id=f"{split.value}_unseen_bicond_{i}",
                    domain=DomainType.LOGIC,
                    split=split,
                    prompt=prompt,
                    expected_answer=expected,
                    allowed_answers=[expected],
                    is_unanswerable=False,
                    metadata={
                        "category": LogicTaskCategory.BICONDITIONAL.value,
                        "generalization_type": LogicGeneralizationType.UNSEEN_STRUCTURES.value,
                    },
                )
            )

    return samples


class LogicReasoningDataset(Dataset):
    """PyTorch Dataset for symbolic logic with prompt token loss masking."""

    def __init__(
        self,
        samples: list[BenchmarkSample],
        tokenizer: BPETokenizer,
        context_length: int = 256,
        ignore_index: int = -100,
    ) -> None:
        self.context_length = context_length
        self.ignore_index = ignore_index
        self.data: list[tuple[Tensor, Tensor]] = []

        for s in samples:
            prompt_ids = tokenizer.encode(s.prompt)
            answer_ids = tokenizer.encode(format_logic_completion(s.expected_answer))
            token_ids = prompt_ids + answer_ids

            if len(token_ids) > context_length:
                token_ids = token_ids[-context_length:]
                prompt_len = max(0, len(token_ids) - len(answer_ids))
            else:
                prompt_len = len(prompt_ids)

            input_ids = torch.tensor(token_ids, dtype=torch.long)
            target_ids = input_ids.clone()
            target_ids[:prompt_len] = ignore_index

            self.data.append((input_ids, target_ids))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.data[idx]
