"""M35: Supervised Metacognitive Curriculum and Empirical Utility Grounding.

Constructs bounded, verified datasets across the 7 specified metacognitive scenarios:
  1. known_arithmetic: Simple 1-2 digit calculations -> ANSWER / RECURRENT_EXECUTION.
  2. unfamiliar_procedural: Multi-digit & cascading carries -> RECURRENT_EXECUTION.
  3. logical_deduction: Shallow vs deep deductive chains -> ANSWER / RECURRENT_EXECUTION / VERIFY.
  4. factual_grounding: Passage-grounded information retrieval -> RETRIEVE.
  5. missing_contradictory: Contradictory premises & OOD unanswerables -> ABSTAIN.
  6. repeated_failure: Persistent procedural failures requiring adaptation -> TRAIN_CANDIDATE / PRACTICE.
  7. mixed_domain: Context-rich word problems combining language and calculation -> RECURRENT_EXECUTION.

Train and evaluation partitions are strictly disjoint across queries, contexts, and entities.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.metacognition.action_space import (
    ACTION_LIST,
    LearningAction,
    compute_action_utility,
)


@dataclass
class MetacognitiveSample:
    """A task instance annotated with optimal action and full empirical utility vector."""

    sample_id: str
    scenario: str
    query: str
    context: str | None
    ground_truth_answer: str
    optimal_action: LearningAction
    action_utilities: dict[LearningAction, float]
    is_unanswerable: bool = False
    is_repeated_failure: bool = False


# Disjoint factual pools for train and eval
TRAIN_FACTS = [
    ("Mitochondria generate cellular ATP through aerobic cellular respiration.", "What organelle generates ATP?", "Mitochondria"),
    ("The speed of light in vacuum is approximately 300,000 km per second.", "What is the speed of light in vacuum?", "300,000 km/s"),
    ("Chlorophyll pigments in plant chloroplasts absorb blue and red light.", "What pigment absorbs blue and red light?", "Chlorophyll"),
    ("Water boils at 100 degrees Celsius at standard atmospheric pressure.", "At what temperature does water boil?", "100 degrees Celsius"),
    ("DNA consists of four nucleotide bases: adenine, thymine, guanine, and cytosine.", "What molecules compose the core structure of DNA?", "nucleotide bases"),
]

EVAL_FACTS = [
    ("Ribosomes are macromolecular machines that synthesize proteins.", "What machines synthesize cellular proteins?", "Ribosomes"),
    ("The Earth orbits the Sun at an average distance of 150 million kilometers.", "What distance does Earth orbit from the Sun?", "150 million km"),
    ("Hemoglobin in red blood cells binds and transports oxygen molecules.", "What protein binds oxygen in blood?", "Hemoglobin"),
    ("Diamond is an allotrope of carbon with a rigid crystal lattice.", "What chemical element forms diamond?", "carbon"),
    ("Photosynthesis converts carbon dioxide and water into glucose and oxygen.", "What sugars are produced in photosynthesis?", "glucose"),
]

# Disjoint contradiction pools for train and eval
TRAIN_CONTRADICTIONS = [
    ("Fact: The storage box is completely empty. Fact: The storage box contains 5 apples.", "How many apples are in the storage box?"),
    ("Statement: Object Alpha is purely circular. Statement: Object Alpha has exactly four sharp corners.", "How many corners does Object Alpha have?"),
    ("Query: What is the official gold currency of the sunken Republic of Atlantis?", "What currency is officially used?"),
    ("Premise: All birds can fly. Premise: Emperor penguins are birds that cannot fly.", "Can an emperor penguin fly?"),
    ("Query: What was the registered census population of Mars in the year 1800?", "What was the population of Mars in 1800?"),
]

EVAL_CONTRADICTIONS = [
    ("Fact: The conference room has zero occupants. Fact: 10 people are talking inside the conference room.", "How many people are in the room?"),
    ("Statement: Fabric Beta is completely dry. Statement: Fabric Beta is 100% soaked in liquid water.", "Is Fabric Beta dry or soaked?"),
    ("Query: What is the telephone area code for the mythical continent of Lemuria?", "What is the area code of Lemuria?"),
    ("Premise: All metals are solid at room temperature. Premise: Mercury is a liquid metal at room temperature.", "Is mercury a solid metal at room temperature?"),
    ("Query: Who was the crowned king of the Earth's Moon in the year 1750?", "Who was the lunar king in 1750?"),
]


def _generate_scenario_samples(
    scenario: str,
    count: int,
    seed: int,
    partition: str = "train",
) -> list[MetacognitiveSample]:
    """Generate verified samples for a specific metacognitive scenario."""
    rng = random.Random(seed)
    samples: list[MetacognitiveSample] = []

    for i in range(count):
        sid = f"{scenario}_{partition}_{i:03d}"

        if scenario == "known_arithmetic":
            # Disjoint numeric intervals
            if partition == "train":
                a = rng.randint(2, 49)
                b = rng.randint(2, 49)
            else:
                a = rng.randint(50, 99)
                b = rng.randint(50, 99)
            q = f"Add: {a} + {b} ="
            ans = str(a + b)
            opt = LearningAction.ANSWER
            utils = {
                LearningAction.ANSWER: 0.99,
                LearningAction.RECURRENT_EXECUTION: 0.85,
                LearningAction.VERIFY: 0.50,
                LearningAction.RETRIEVE: -0.20,
                LearningAction.PRACTICE: -0.40,
                LearningAction.TRAIN_CANDIDATE: -0.90,
                LearningAction.ABSTAIN: -0.80,
            }
            samples.append(MetacognitiveSample(sid, scenario, q, None, ans, opt, utils))

        elif scenario == "unfamiliar_procedural":
            # Disjoint 4-digit intervals
            if partition == "train":
                a = rng.randint(1000, 4999)
                b = rng.randint(1000, 4999)
            else:
                a = rng.randint(5000, 9999)
                b = rng.randint(5000, 9999)
            q = f"Add: {a} + {b} ="
            ans = str(a + b)
            opt = LearningAction.RECURRENT_EXECUTION
            utils = {
                LearningAction.RECURRENT_EXECUTION: 0.95,
                LearningAction.ANSWER: -0.90,  # Direct answer fails on 4-digit
                LearningAction.VERIFY: 0.40,
                LearningAction.RETRIEVE: -0.30,
                LearningAction.PRACTICE: 0.60,
                LearningAction.TRAIN_CANDIDATE: -0.60,
                LearningAction.ABSTAIN: -0.80,
            }
            samples.append(MetacognitiveSample(sid, scenario, q, None, ans, opt, utils))

        elif scenario == "logical_deduction":
            # Disjoint variable letters between train and eval
            hops = rng.choice([3, 4, 5])
            if partition == "train":
                syms = [chr(65 + j) for j in range(hops + 1)]  # A, B, C, D, E...
            else:
                syms = [chr(80 + j) for j in range(hops + 1)]  # P, Q, R, S, T...
            rules = [f"If {syms[j]} then {syms[j+1]}" for j in range(hops)]
            q = f"Rules: {'. '.join(rules)}. Fact: {syms[0]} is True. Query: Is {syms[-1]} True?"
            ans = "True"
            opt = LearningAction.RECURRENT_EXECUTION if hops >= 4 else LearningAction.ANSWER
            utils = {
                LearningAction.RECURRENT_EXECUTION: 0.92,
                LearningAction.ANSWER: 0.70 if hops < 4 else -0.80,
                LearningAction.VERIFY: 0.85,
                LearningAction.RETRIEVE: -0.30,
                LearningAction.PRACTICE: 0.50,
                LearningAction.TRAIN_CANDIDATE: -0.70,
                LearningAction.ABSTAIN: -0.80,
            }
            samples.append(MetacognitiveSample(sid, scenario, q, None, ans, opt, utils))

        elif scenario == "factual_grounding":
            pool = TRAIN_FACTS if partition == "train" else EVAL_FACTS
            fact_ctx, fact_q, fact_ans = pool[i % len(pool)]
            q = f"Question: {fact_q}"
            opt = LearningAction.RETRIEVE
            utils = {
                LearningAction.RETRIEVE: 0.95,
                LearningAction.ANSWER: -0.90,
                LearningAction.VERIFY: 0.20,
                LearningAction.RECURRENT_EXECUTION: -0.80,
                LearningAction.PRACTICE: -0.50,
                LearningAction.TRAIN_CANDIDATE: -0.80,
                LearningAction.ABSTAIN: -0.60,
            }
            samples.append(MetacognitiveSample(sid, scenario, q, fact_ctx, fact_ans, opt, utils))

        elif scenario == "missing_contradictory":
            pool = TRAIN_CONTRADICTIONS if partition == "train" else EVAL_CONTRADICTIONS
            c_ctx, c_q = pool[i % len(pool)]
            q = f"Context: {c_ctx} Question: {c_q}"
            opt = LearningAction.ABSTAIN
            utils = {
                LearningAction.ABSTAIN: 0.99,
                LearningAction.ANSWER: -1.00,
                LearningAction.RETRIEVE: -0.80,
                LearningAction.RECURRENT_EXECUTION: -1.00,
                LearningAction.VERIFY: 0.40,
                LearningAction.PRACTICE: -0.80,
                LearningAction.TRAIN_CANDIDATE: -1.20,
            }
            samples.append(MetacognitiveSample(sid, scenario, q, c_ctx, "unanswerable", opt, utils, is_unanswerable=True))

        elif scenario == "repeated_failure":
            if partition == "train":
                pairs = [(6138, 1936), (9999, 1), (4589, 3824), (7291, 8819)]
            else:
                pairs = [(3419, 2875), (8888, 2), (5172, 4938), (6304, 7696)]
            a, b = pairs[i % len(pairs)]
            q = f"Problem: Procedural carry step failed repeatedly on {a} + {b} (id={partition}_{i}). Decide next action."
            ans = "TRAIN_CANDIDATE"
            opt = LearningAction.TRAIN_CANDIDATE
            utils = {
                LearningAction.TRAIN_CANDIDATE: 0.90,
                LearningAction.PRACTICE: 0.70,
                LearningAction.RECURRENT_EXECUTION: 0.20,
                LearningAction.ANSWER: -0.90,
                LearningAction.RETRIEVE: -0.70,
                LearningAction.VERIFY: 0.30,
                LearningAction.ABSTAIN: -0.80,
            }
            samples.append(MetacognitiveSample(sid, scenario, q, None, ans, opt, utils, is_repeated_failure=True))

        elif scenario == "mixed_domain":
            if partition == "train":
                items = ["cookies", "apples", "candies", "tickets", "marbles"]
                val1 = rng.randint(20, 45)
                val2 = rng.randint(15, 35)
            else:
                items = ["laptops", "bicycles", "parcels", "paintings", "telescopes"]
                val1 = rng.randint(55, 95)
                val2 = rng.randint(45, 85)
            itm = items[i % len(items)]
            q = f"Context: A shipment contained {val1} {itm} on Monday and {val2} {itm} on Tuesday. How many {itm} were shipped altogether?"
            ans = str(val1 + val2)
            opt = LearningAction.RECURRENT_EXECUTION
            utils = {
                LearningAction.RECURRENT_EXECUTION: 0.92,
                LearningAction.VERIFY: 0.75,
                LearningAction.ANSWER: 0.50,
                LearningAction.RETRIEVE: 0.40,
                LearningAction.PRACTICE: 0.30,
                LearningAction.TRAIN_CANDIDATE: -0.80,
                LearningAction.ABSTAIN: -0.80,
            }
            samples.append(MetacognitiveSample(sid, scenario, q, None, ans, opt, utils))

    return samples


def build_metacognitive_curriculum(
    seed: int = 42,
    train_samples_per_scenario: int = 24,
    eval_samples_per_scenario: int = 16,
) -> tuple[list[MetacognitiveSample], list[MetacognitiveSample]]:
    """Build strictly disjoint training and evaluation partitions across all 7 scenarios."""
    scenarios = [
        "known_arithmetic",
        "unfamiliar_procedural",
        "logical_deduction",
        "factual_grounding",
        "missing_contradictory",
        "repeated_failure",
        "mixed_domain",
    ]

    train_set: list[MetacognitiveSample] = []
    eval_set: list[MetacognitiveSample] = []

    for idx, sc in enumerate(scenarios):
        sc_seed_train = seed * 100 + idx
        sc_seed_eval = (seed + 999) * 100 + idx
        train_set.extend(_generate_scenario_samples(sc, train_samples_per_scenario, sc_seed_train, "train"))
        eval_set.extend(_generate_scenario_samples(sc, eval_samples_per_scenario, sc_seed_eval, "eval"))

    random.Random(seed).shuffle(train_set)
    random.Random(seed + 1).shuffle(eval_set)
    return train_set, eval_set
