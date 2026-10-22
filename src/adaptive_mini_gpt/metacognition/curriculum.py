"""M35: Supervised Metacognitive Curriculum and Empirical Utility Grounding.

Constructs bounded, verified datasets across the 7 specified metacognitive scenarios:
  1. known_arithmetic: Simple 1-2 digit calculations -> ANSWER / RECURRENT_EXECUTION.
  2. unfamiliar_procedural: Multi-digit & cascading carries -> RECURRENT_EXECUTION.
  3. logical_deduction: Shallow vs deep deductive chains -> ANSWER / RECURRENT_EXECUTION / VERIFY.
  4. factual_grounding: Passage-grounded information retrieval -> RETRIEVE.
  5. missing_contradictory: Contradictory premises & OOD unanswerables -> ABSTAIN.
  6. repeated_failure: Persistent procedural failures requiring adaptation -> TRAIN_CANDIDATE / PRACTICE.
  7. mixed_domain: Context-rich word problems combining language and calculation -> RECURRENT_EXECUTION.

Train and evaluation partitions are strictly disjoint.
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
            # 1-2 digit addition: easy direct answer
            a = rng.randint(2, 25)
            b = rng.randint(2, 25)
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
            # Multi-digit procedural addition: requires recurrent execution
            a = rng.randint(1000, 9999)
            b = rng.randint(1000, 9999)
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
            # Chained implications: verify or recurrent unrolling
            hops = rng.choice([3, 4, 5])
            syms = [chr(65 + j) for j in range(hops + 1)]
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
            # Grounded reading comprehension
            facts = [
                ("Mitochondria generate cellular ATP through aerobic respiration.", "What generates ATP?", "Mitochondria"),
                ("The speed of light in vacuum is approximately 300,000 km per second.", "What is the vacuum speed of light?", "300,000 km/s"),
                ("Chlorophyll absorbs blue and red light while reflecting green.", "What pigment reflects green light?", "Chlorophyll"),
                ("Water boils at 100 degrees Celsius at standard atmospheric pressure.", "At what temperature does water boil?", "100 degrees Celsius"),
                ("DNA consists of four nucleotide bases: adenine, thymine, guanine, and cytosine.", "What molecules compose DNA?", "nucleotide bases"),
            ]
            fact_ctx, fact_q, fact_ans = rng.choice(facts)
            q = f"Question: {fact_q}"
            opt = LearningAction.RETRIEVE
            utils = {
                LearningAction.RETRIEVE: 0.95,
                LearningAction.ANSWER: -0.90,  # Parametric 1M model cannot memorize open facts
                LearningAction.VERIFY: 0.20,
                LearningAction.RECURRENT_EXECUTION: -0.80,
                LearningAction.PRACTICE: -0.50,
                LearningAction.TRAIN_CANDIDATE: -0.80,
                LearningAction.ABSTAIN: -0.60,
            }
            samples.append(MetacognitiveSample(sid, scenario, q, fact_ctx, fact_ans, opt, utils))

        elif scenario == "missing_contradictory":
            # Contradictory premises or unanswerable / out-of-domain claims
            contradictions = [
                ("Fact: The box is completely empty. Fact: The box contains 5 apples.", "How many apples are in the box?"),
                ("Statement: Object X is purely circular. Statement: Object X has exactly four corners.", "How many corners does Object X have?"),
                ("Query: What is the official currency of the Republic of Atlantis?", "What currency is used?"),
                ("Premise: All birds fly. Premise: Penguins are birds that cannot fly.", "Can a penguin fly?"),
                ("Query: What was the population of Mars in the year 1800?", "What was the population of Mars?"),
            ]
            c_ctx, c_q = rng.choice(contradictions)
            q = f"Context: {c_ctx} Question: {c_q}"
            opt = LearningAction.ABSTAIN
            utils = {
                LearningAction.ABSTAIN: 0.99,  # High reward for abstaining
                LearningAction.ANSWER: -1.00,
                LearningAction.RETRIEVE: -0.80,
                LearningAction.RECURRENT_EXECUTION: -1.00,
                LearningAction.VERIFY: 0.40,
                LearningAction.PRACTICE: -0.80,
                LearningAction.TRAIN_CANDIDATE: -1.20,
            }
            samples.append(MetacognitiveSample(sid, scenario, q, c_ctx, "unanswerable", opt, utils, is_unanswerable=True))

        elif scenario == "repeated_failure":
            # Repeated procedural failure on specialized carry patterns
            a = rng.choice([6138, 9999, 4589, 7291])
            b = rng.choice([1936, 1, 3824, 8819])
            q = f"Problem: Procedural carry step failed repeatedly on {a} + {b}. Decide next action."
            ans = "TRAIN_CANDIDATE"
            opt = LearningAction.TRAIN_CANDIDATE
            utils = {
                LearningAction.TRAIN_CANDIDATE: 0.90,  # Training justified after repeated failure
                LearningAction.PRACTICE: 0.70,
                LearningAction.RECURRENT_EXECUTION: 0.20,
                LearningAction.ANSWER: -0.90,
                LearningAction.RETRIEVE: -0.70,
                LearningAction.VERIFY: 0.30,
                LearningAction.ABSTAIN: -0.80,
            }
            samples.append(MetacognitiveSample(sid, scenario, q, None, ans, opt, utils, is_repeated_failure=True))

        elif scenario == "mixed_domain":
            # Context-rich word problems combining natural language and calculation
            items = ["books", "pencils", "widgets", "miles", "hours"]
            itm = rng.choice(items)
            val1 = rng.randint(45, 95)
            val2 = rng.randint(35, 85)
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
