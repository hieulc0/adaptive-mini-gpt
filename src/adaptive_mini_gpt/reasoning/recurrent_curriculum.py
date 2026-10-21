"""M32 Phase 2: Recurrent Practice Curricula and Multi-Digit Extrapolation Suites.

Constructs:
  1. Complete multi-digit addition evaluation suites (2, 4, 8, 16, 32 digits).
  2. Specialized cascading carry suite (e.g. 9999 + 1) and final overflow suite.
  3. Deep logic deduction suites (3, 4, 5, 6 hops).
  4. Failure-targeted recurrent practice curriculum (focused on ripple carries, double carries, and cascading overflow).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.reasoning.scratchpad import (
    ReasoningDomain,
    ReasoningSample,
    build_arithmetic_reasoning_sample,
    build_logic_reasoning_sample,
)
from adaptive_mini_gpt.rules.reliable_transition import (
    ElementaryTransition,
    format_elementary_prompt_and_target,
    get_all_200_transitions,
)


@dataclass
class AdditionProblemSpec:
    """Specification of an addition test problem."""

    a: int
    b: int
    num_digits: int
    is_cascading_carry: bool
    is_final_overflow: bool


def generate_n_digit_number(digits: int, rng: random.Random) -> int:
    """Generate a random integer with exactly `digits` digits."""
    if digits == 1:
        return rng.randint(0, 9)
    low = 10 ** (digits - 1)
    high = (10 ** digits) - 1
    return rng.randint(low, high)


def build_multi_digit_addition_suite(seed: int = 101) -> dict[str, list[AdditionProblemSpec]]:
    """Build evaluation suites across 2, 4, 8, 16, 32 digits, cascading carries, and final overflow."""
    rng = random.Random(seed)
    suite: dict[str, list[AdditionProblemSpec]] = {}

    # 1. Standard Lengths
    counts = {
        "arith_2_digits": (2, 20),
        "arith_4_digits": (4, 16),
        "arith_8_digits": (8, 12),
        "arith_16_digits": (16, 8),
        "arith_32_digits": (32, 6),
    }

    for split_name, (n_digits, count) in counts.items():
        problems: list[AdditionProblemSpec] = []
        for _ in range(count):
            a = generate_n_digit_number(n_digits, rng)
            b = generate_n_digit_number(n_digits, rng)
            overflow = (a + b) >= (10 ** n_digits)
            problems.append(
                AdditionProblemSpec(
                    a=a,
                    b=b,
                    num_digits=n_digits,
                    is_cascading_carry=False,
                    is_final_overflow=overflow,
                )
            )
        suite[split_name] = problems

    # 2. Cascading Carries Suite (Ripple carry propagation across multiple 9s)
    cascading: list[AdditionProblemSpec] = []
    ripple_lengths = [2, 3, 4, 5, 8, 12, 16, 24, 32]
    for d in ripple_lengths:
        # e.g. 99 + 1, 999 + 1, 9999 + 1
        a = (10 ** d) - 1
        b = 1
        cascading.append(AdditionProblemSpec(a=a, b=b, num_digits=d, is_cascading_carry=True, is_final_overflow=True))
        # e.g. 9998 + 3 = 10001
        if d >= 3:
            a2 = (10 ** d) - 2
            b2 = rng.randint(3, 9)
            cascading.append(AdditionProblemSpec(a=a2, b=b2, num_digits=d, is_cascading_carry=True, is_final_overflow=True))

    suite["arith_cascading_carry"] = cascading

    # 3. Final Overflow Suite (carry emitted at the most significant digit)
    overflow_list: list[AdditionProblemSpec] = []
    for d in [2, 4, 8, 16]:
        for _ in range(3):
            # high leading digits e.g. 6xxxx + 7xxxx >= 13xxxx
            lead_a = rng.randint(5, 9)
            lead_b = rng.randint(5, 9)
            if d == 2:
                a = lead_a * 10 + rng.randint(0, 9)
                b = lead_b * 10 + rng.randint(0, 9)
            else:
                rem_a = rng.randint(0, (10 ** (d - 1)) - 1)
                rem_b = rng.randint(0, (10 ** (d - 1)) - 1)
                a = lead_a * (10 ** (d - 1)) + rem_a
                b = lead_b * (10 ** (d - 1)) + rem_b
            overflow_list.append(AdditionProblemSpec(a=a, b=b, num_digits=d, is_cascading_carry=False, is_final_overflow=True))

    suite["arith_final_overflow"] = overflow_list
    return suite


def build_deep_logic_suite(seed: int = 202) -> dict[str, list[ReasoningSample]]:
    """Build evaluation logic suites across 3, 4, 5, and 6 hops."""
    rng = random.Random(seed)
    suite: dict[str, list[ReasoningSample]] = {}
    hop_counts = {
        "logic_3_hop": (3, 16),
        "logic_4_hop": (4, 12),
        "logic_5_hop": (5, 8),
        "logic_6_hop": (6, 6),
    }

    all_vars = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]

    for split_name, (hops, count) in hop_counts.items():
        samples: list[ReasoningSample] = []
        for i in range(count):
            start_var_idx = rng.randint(0, len(all_vars) - hops - 1)
            var_chain = all_vars[start_var_idx : start_var_idx + hops + 1]
            negations = [rng.choice([False, False, True]) for _ in range(hops)]
            initial_val = rng.choice([True, False])
            sample = build_logic_reasoning_sample(
                var_chain=var_chain,
                negations=negations,
                query_var=var_chain[-1],
                initial_fact=(var_chain[0], initial_val),
            )
            sample.sample_id = f"{split_name}_{i}"
            samples.append(sample)
        suite[split_name] = samples

    return suite


def build_failure_targeted_recurrent_practice(
    target_sample_count: int = 72,
    seed: int = 303,
) -> list[ReasoningSample]:
    """Construct targeted practice samples prioritizing audited recurrent transition bottlenecks:

    Bottlenecks targeted:
      1. Propagate states: c_in = 1, sum in [10..18]
      2. High double carries: a in [5..9], b in [5..9], c_in in [0, 1]
      3. Cascading ripple carries: numbers like 99, 999, 9999 added to small digits
      4. Overflow additions requiring MSB carry out
    """
    rng = random.Random(seed)
    practice_samples: list[ReasoningSample] = []

    # 1. High carry and propagate pairs
    for _ in range(target_sample_count // 3):
        # 2-digit high carry (e.g. 89 + 78, 67 + 58)
        a = rng.randint(55, 99)
        b = rng.randint(55, 99)
        practice_samples.append(build_arithmetic_reasoning_sample(a, b))

    # 2. Cascading carry pairs (e.g. 99 + 5, 999 + 8, 199 + 1)
    for _ in range(target_sample_count // 3):
        n_9s = rng.randint(2, 4)
        base = (10 ** n_9s) - rng.randint(1, 5)
        add = rng.randint(2, 9)
        practice_samples.append(build_arithmetic_reasoning_sample(base, add))

    # 3. 3-digit and 4-digit general overflow additions
    while len(practice_samples) < target_sample_count:
        d = rng.choice([3, 4])
        a = generate_n_digit_number(d, rng)
        b = generate_n_digit_number(d, rng)
        practice_samples.append(build_arithmetic_reasoning_sample(a, b))

    return practice_samples
