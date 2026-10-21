"""Controlled Learning Episodes for M29 (Language, Logic, Arithmetic).

Implements three targeted learning episodes evaluating:
  1. Immediate retry without training.
  2. External retrieval/memory without weight updates.
  3. Targeted candidate training without replay.
  4. Targeted candidate training with balanced replay.

Crucial invariant: Weight learning benefits MUST be measured on fresh,
unseen test problems strictly disjoint from practice/training samples.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.data.language_corpus import (
    ABSTENTION_TARGET,
    format_qa_prompt,
)
from adaptive_mini_gpt.data.language_data import BenchmarkSample, BenchmarkSplit, DomainType


@dataclass
class EpisodeData:
    """Packaged data for a controlled learning episode."""

    episode_id: str
    domain: DomainType
    title: str
    trigger_question: str
    trigger_context: str | None
    trigger_ground_truth: str
    approved_reference_doc: str
    practice_samples: list[BenchmarkSample]
    fresh_test_samples: list[BenchmarkSample]


# =====================================================================
# Episode A: Language — Unfamiliar Scientific Concept (Ribosomes & Cell Biology)
# =====================================================================
EPISODE_A_REFERENCE_DOC = (
    "Ribosomes are complex cellular structures responsible for protein synthesis. "
    "They translate messenger RNA transcripts into polypeptide chains using transfer RNA. "
    "Ribosomes consist of a small subunit and a large subunit composed of ribosomal RNA. "
    "Free ribosomes produce proteins for the cytoplasm, whereas bound ribosomes on the "
    "rough endoplasmic reticulum assemble proteins destined for membranes or secretion."
)

EPISODE_A_PRACTICE_RAW = [
    ("What cellular structure is responsible for protein synthesis?", "Ribosomes"),
    ("Which structure translates messenger RNA into polypeptide chains?", "Ribosomes"),
    ("What are ribosomes composed of?", "ribosomal RNA"),
    ("What do free ribosomes produce proteins for?", "cytoplasm"),
    ("Where are bound ribosomes located?", "rough endoplasmic reticulum"),
    ("What do bound ribosomes produce proteins for?", "membranes or secretion"),
    ("What are the two components of ribosomes?", "small subunit and large subunit"),
    ("What molecule do ribosomes translate into proteins?", "messenger RNA"),
]

EPISODE_A_FRESH_TEST_RAW = [
    # Paraphrased and fresh held-out questions on the same underlying concept
    ("Which cellular organelle synthesizes peptide chains from RNA?", "Ribosomes"),
    ("Where does cytoplasmic protein assembly occur?", "Free ribosomes"),
    ("What organelle carries bound ribosomes on its surface?", "rough endoplasmic reticulum"),
    ("Which subunit pairs with the large subunit to form a ribosome?", "small subunit"),
    ("What type of RNA forms the structural core of ribosomes?", "ribosomal RNA"),
    ("What is the primary cellular function of ribosomes?", "protein synthesis"),
    ("Proteins destined for secretion are assembled by which ribosomes?", "bound ribosomes"),
    ("What nucleic acid template is read during protein translation?", "messenger RNA"),
]


def build_episode_a_language() -> EpisodeData:
    """Build Episode A: Unfamiliar factual concept learning."""
    practice: list[BenchmarkSample] = []
    for i, (q, a) in enumerate(EPISODE_A_PRACTICE_RAW):
        prompt = format_qa_prompt(passage=EPISODE_A_REFERENCE_DOC, question=q)
        practice.append(BenchmarkSample(
            sample_id=f"epA_train_{i:02d}",
            domain=DomainType.LANGUAGE,
            split=BenchmarkSplit.TRAIN,
            prompt=prompt,
            expected_answer=a,
            allowed_answers=[a],
            is_unanswerable=False,
            metadata={"question": q, "context": EPISODE_A_REFERENCE_DOC},
        ))

    test: list[BenchmarkSample] = []
    for i, (q, a) in enumerate(EPISODE_A_FRESH_TEST_RAW):
        prompt = format_qa_prompt(passage=EPISODE_A_REFERENCE_DOC, question=q)
        test.append(BenchmarkSample(
            sample_id=f"epA_test_{i:02d}",
            domain=DomainType.LANGUAGE,
            split=BenchmarkSplit.TEST,
            prompt=prompt,
            expected_answer=a,
            allowed_answers=[a],
            is_unanswerable=False,
            metadata={"question": q, "context": EPISODE_A_REFERENCE_DOC},
        ))

    return EpisodeData(
        episode_id="episode_a_language",
        domain=DomainType.LANGUAGE,
        title="Episode A: Unfamiliar Concept Acquisition (Cellular Ribosomes)",
        trigger_question="What cellular structure is responsible for protein synthesis?",
        trigger_context=None,
        trigger_ground_truth="Ribosomes",
        approved_reference_doc=EPISODE_A_REFERENCE_DOC,
        practice_samples=practice,
        fresh_test_samples=test,
    )


# =====================================================================
# Episode B: Logic — Unfamiliar 3-Step Chained Implication with Negation
# =====================================================================
def build_episode_b_logic(seed: int = 42) -> EpisodeData:
    """Build Episode B: Unfamiliar rule combination with disjoint variable sets."""
    rng = random.Random(seed)

    # Disjoint variable sets
    train_vars = [
        ("A", "B", "C"),
        ("D", "E", "F"),
        ("G", "H", "I"),
        ("J", "K", "L"),
    ]
    test_vars = [
        ("W", "X", "Y"),
        ("M", "N", "O"),
        ("P", "Q", "R"),
        ("S", "T", "U"),
    ]

    def make_chained_sample(
        v1: str, v2: str, v3: str, negate: bool, sid: str, split: BenchmarkSplit
    ) -> BenchmarkSample:
        if negate:
            # Rule: v1 -> v2, v2 -> not v3. v1 is True. What is v3? -> False
            prompt = f"Rule: If {v1} then {v2}. If {v2} then not {v3}. Fact: {v1} is True. Question: Is {v3} True?\n#### "
            target = "False"
        else:
            # Rule: v1 -> v2, v2 -> v3. v1 is True. What is v3? -> True
            prompt = f"Rule: If {v1} then {v2}. If {v2} then {v3}. Fact: {v1} is True. Question: Is {v3} True?\n#### "
            target = "True"

        return BenchmarkSample(
            sample_id=sid,
            domain=DomainType.LOGIC,
            split=split,
            prompt=prompt,
            expected_answer=target,
            allowed_answers=[target],
            is_unanswerable=False,
            metadata={"question": f"Deduce {v3} from {v1}"},
        )

    practice: list[BenchmarkSample] = []
    for i, (v1, v2, v3) in enumerate(train_vars):
        practice.append(make_chained_sample(v1, v2, v3, negate=False, sid=f"epB_train_{i*2:02d}", split=BenchmarkSplit.TRAIN))
        practice.append(make_chained_sample(v1, v2, v3, negate=True, sid=f"epB_train_{i*2+1:02d}", split=BenchmarkSplit.TRAIN))

    test: list[BenchmarkSample] = []
    for i, (v1, v2, v3) in enumerate(test_vars):
        test.append(make_chained_sample(v1, v2, v3, negate=False, sid=f"epB_test_{i*2:02d}", split=BenchmarkSplit.TEST))
        test.append(make_chained_sample(v1, v2, v3, negate=True, sid=f"epB_test_{i*2+1:02d}", split=BenchmarkSplit.TEST))

    return EpisodeData(
        episode_id="episode_b_logic",
        domain=DomainType.LOGIC,
        title="Episode B: Chained Implication with Negation Deduction",
        trigger_question="Rule: If A then B. If B then not C. Fact: A is True. Question: Is C True?",
        trigger_context=None,
        trigger_ground_truth="False",
        approved_reference_doc="Logical Deduction Engine Oracle",
        practice_samples=practice,
        fresh_test_samples=test,
    )


# =====================================================================
# Episode C: Arithmetic — Double Carry Transition Error (9+9+carry=1 & 9+0+carry=1)
# =====================================================================
def build_episode_c_arithmetic() -> EpisodeData:
    """Build Episode C: Repeated elementary transition error practice and test."""
    practice_specs = [
        (9, 9, 1, 9, 1),
        (9, 0, 1, 0, 1),
        (0, 9, 1, 0, 1),
        (8, 8, 1, 7, 1),
        (7, 8, 1, 6, 1),
        (8, 7, 1, 6, 1),
        (9, 8, 1, 8, 1),
        (8, 9, 1, 8, 1),
    ]

    fresh_test_specs = [
        # Disjoint operand pairs that require the exact same carry logic
        (9, 1, 1, 1, 1),
        (1, 9, 1, 1, 1),
        (9, 2, 1, 2, 1),
        (2, 9, 1, 2, 1),
        (9, 3, 1, 3, 1),
        (3, 9, 1, 3, 1),
        (9, 4, 1, 4, 1),
        (4, 9, 1, 4, 1),
    ]

    practice: list[BenchmarkSample] = []
    for i, (d1, d2, c_in, s_out, c_out) in enumerate(practice_specs):
        prompt = f"c:{c_in} d1:{d1} d2:{d2} ->"
        target = f"s:{s_out} c:{c_out}"
        practice.append(BenchmarkSample(
            sample_id=f"epC_train_{i:02d}",
            domain=DomainType.ARITHMETIC,
            split=BenchmarkSplit.TRAIN,
            prompt=prompt,
            expected_answer=target,
            allowed_answers=[target],
            is_unanswerable=False,
            metadata={"question": f"Add column {d1}+{d2} with carry {c_in}"},
        ))

    test: list[BenchmarkSample] = []
    for i, (d1, d2, c_in, s_out, c_out) in enumerate(fresh_test_specs):
        prompt = f"c:{c_in} d1:{d1} d2:{d2} ->"
        target = f"s:{s_out} c:{c_out}"
        test.append(BenchmarkSample(
            sample_id=f"epC_test_{i:02d}",
            domain=DomainType.ARITHMETIC,
            split=BenchmarkSplit.TEST,
            prompt=prompt,
            expected_answer=target,
            allowed_answers=[target],
            is_unanswerable=False,
            metadata={"question": f"Add column {d1}+{d2} with carry {c_in}"},
        ))

    return EpisodeData(
        episode_id="episode_c_arithmetic",
        domain=DomainType.ARITHMETIC,
        title="Episode C: Repeated High-Carry Transition Repair",
        trigger_question="c:1 d1:9 d2:9 ->",
        trigger_context=None,
        trigger_ground_truth="s:9 c:1",
        approved_reference_doc="Elementary Addition Transition Table Oracle",
        practice_samples=practice,
        fresh_test_samples=test,
    )
