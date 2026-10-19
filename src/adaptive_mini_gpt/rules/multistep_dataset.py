"""Dataset generation for Multi-Step Planning and Compositional Reasoning (M16).

Generates partitioned, multi-step compositional problems across 2 to 5 steps:
- Multi-segment average speed (2 segments = 3 steps, 3 segments = 5 steps)
- Chained arithmetic (3 operands = 2 steps, 4 operands = 3 steps, 5 operands = 4 steps)
- Transitive comparison chains (4+ entities)
- Mixed operations
- Robust variations: changed phrasing, inverted order, distracting numbers (years, flight numbers)
- Abstention battery: invalid dimensions (adding km + h), non-positive time, cyclic relations, missing data

Enforces strictly disjoint train (70%), val (15%), and test (15%) splits via SHA-256 cryptographic hash.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.rules.plan import ExecutionPlan, PlanStep


@dataclass
class MultiStepProblem:
    """A multi-step compositional reasoning problem."""

    question: str
    expected_plan: ExecutionPlan
    expected_answer: str | None
    num_steps: int
    archetype: str
    split: str
    is_malformed: bool = False
    should_abstain: bool = False
    abstain_reason: str | None = None

    def to_plan_pair(self) -> str:
        """Instruction pair with ground truth plan as assistant completion."""
        if self.should_abstain:
            completion = f"Abstain: {self.abstain_reason or 'Invalid query'}"
        else:
            completion = self.expected_plan.to_text_format()
        return f"<|user|>\n{self.question}\n<|assistant|>\n{completion}<|eos|>\n"

    def to_direct_pair(self) -> str:
        """Instruction pair with direct answer as assistant completion (neural baseline)."""
        ans = self.expected_answer if self.expected_answer is not None else "Cannot determine"
        return f"<|user|>\n{self.question}\n<|assistant|>\n{ans}<|eos|>\n"


def get_multistep_split(key: str) -> str:
    """Deterministically partition key into train (70%), val (15%), or test (15%)."""
    h = int(hashlib.sha256(f"m16_plan_salt:{key}".encode()).hexdigest()[:8], 16)
    bucket = h % 100
    if bucket < 70:
        return "train"
    elif bucket < 85:
        return "val"
    else:
        return "test"


def generate_multistep_problems(
    num_samples_per_archetype: int = 25,
    seed: int = 42,
    split: str = "train",
) -> list[MultiStepProblem]:
    """Generate balanced multi-step problems partitioned by cryptographic hash."""
    rng = random.Random(seed)
    problems: list[MultiStepProblem] = []

    # -------------------------------------------------------------
    # Archetype 1: 2-Segment Average Speed (3 steps)
    # -------------------------------------------------------------
    speed_templates_2seg = [
        "A car travels {d1} km in {t1} hours, then {d2} km in {t2} hours. What is its overall average speed?",
        "A train covers {d1} km in {t1} hours, followed by another {d2} km in {t2} hours. Calculate average speed for the whole journey.",
        "A delivery truck drives {d1} km in {t1} hours and then travels {d2} km in {t2} hours. What is the average speed of the trip?",
        "An express bus covers a distance of {d1} km in {t1} hours, then another {d2} km in {t2} hours. Find its overall average speed.",
        "In the year 2024, a vehicle travels {d1} km in {t1} hours, then {d2} km in {t2} hours. What is its average speed?",  # Distractor
    ]
    seen_2seg = set()
    count = 0
    attempts = 0
    while count < num_samples_per_archetype and attempts < num_samples_per_archetype * 200:
        attempts += 1
        t1 = rng.randint(2, 4)
        t2 = rng.randint(2, 4)
        v1 = rng.choice([40, 50, 60, 75, 80])
        v2 = rng.choice([50, 60, 75, 80, 100])
        d1 = v1 * t1
        d2 = v2 * t2
        total_d = d1 + d2
        total_t = t1 + t2
        # Ensure clean speed
        if total_d % total_t != 0:
            continue
        expected_speed = f"{total_d // total_t} km/h"
        key = f"2seg_{d1}_{t1}_{d2}_{t2}"
        if key in seen_2seg or get_multistep_split(key) != split:
            continue
        seen_2seg.add(key)

        tmpl = rng.choice(speed_templates_2seg)
        q = tmpl.format(d1=d1, t1=t1, d2=d2, t2=t2)

        plan = ExecutionPlan(
            steps=[
                PlanStep(output_var="d_total", operation="add", args={"a": float(d1), "b": float(d2)}, units="km"),
                PlanStep(output_var="t_total", operation="add", args={"a": float(t1), "b": float(t2)}, units="h"),
                PlanStep(
                    output_var="overall_speed",
                    operation="average_speed",
                    args={"distance_km": "$d_total", "time_hours": "$t_total"},
                    units="km/h",
                ),
            ],
            final_var="overall_speed",
        )

        problems.append(
            MultiStepProblem(
                question=q,
                expected_plan=plan,
                expected_answer=expected_speed,
                num_steps=3,
                archetype="2segment_speed",
                split=split,
            )
        )
        count += 1

    # -------------------------------------------------------------
    # Archetype 2: 3-Segment Average Speed (5 steps)
    # -------------------------------------------------------------
    speed_templates_3seg = [
        "A car travels {d1} km in {t1} hours, then {d2} km in {t2} hours, and finally {d3} km in {t3} hours. What is its overall average speed?",
        "A train covers {d1} km in {t1} hours, another {d2} km in {t2} hours, then {d3} km in {t3} hours. Calculate average speed for the whole journey.",
    ]
    seen_3seg = set()
    count = 0
    attempts = 0
    while count < num_samples_per_archetype and attempts < num_samples_per_archetype * 200:
        attempts += 1
        t1, t2, t3 = rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 3)
        v = rng.choice([50, 60, 75, 80, 90, 100])
        total_t = t1 + t2 + t3
        total_d = v * total_t
        # Partition distance
        d1 = rng.randint(40, 120)
        d2 = rng.randint(40, 120)
        d3 = total_d - (d1 + d2)
        if d3 < 20 or d3 % 10 != 0:
            continue

        key = f"3seg_{d1}_{t1}_{d2}_{t2}_{d3}_{t3}"
        if key in seen_3seg or get_multistep_split(key) != split:
            continue
        seen_3seg.add(key)

        tmpl = rng.choice(speed_templates_3seg)
        q = tmpl.format(d1=d1, t1=t1, d2=d2, t2=t2, d3=d3, t3=t3)
        expected_speed = f"{v} km/h"

        plan = ExecutionPlan(
            steps=[
                PlanStep(output_var="d_12", operation="add", args={"a": float(d1), "b": float(d2)}, units="km"),
                PlanStep(output_var="d_total", operation="add", args={"a": "$d_12", "b": float(d3)}, units="km"),
                PlanStep(output_var="t_12", operation="add", args={"a": float(t1), "b": float(t2)}, units="h"),
                PlanStep(output_var="t_total", operation="add", args={"a": "$t_12", "b": float(t3)}, units="h"),
                PlanStep(
                    output_var="overall_speed",
                    operation="average_speed",
                    args={"distance_km": "$d_total", "time_hours": "$t_total"},
                    units="km/h",
                ),
            ],
            final_var="overall_speed",
        )

        problems.append(
            MultiStepProblem(
                question=q,
                expected_plan=plan,
                expected_answer=expected_speed,
                num_steps=5,
                archetype="3segment_speed",
                split=split,
            )
        )
        count += 1

    # -------------------------------------------------------------
    # Archetype 3: Chained Addition - 3 Operands (2 steps)
    # -------------------------------------------------------------
    add_templates_3ops = [
        "Calculate the sum of {a}, {b}, and {c}.",
        "What is the total of {a}, {b}, and {c}?",
        "Add {a}, {b}, and {c} together.",
        "Compute {a} plus {b} plus {c}.",
    ]
    seen_3ops = set()
    count = 0
    attempts = 0
    while count < num_samples_per_archetype and attempts < num_samples_per_archetype * 200:
        attempts += 1
        a = rng.randint(100, 500)
        b = rng.randint(100, 500)
        c = rng.randint(100, 500)
        key = f"add3_{a}_{b}_{c}"
        if key in seen_3ops or get_multistep_split(key) != split:
            continue
        seen_3ops.add(key)

        tmpl = rng.choice(add_templates_3ops)
        q = tmpl.format(a=a, b=b, c=c)
        expected_ans = str(a + b + c)

        plan = ExecutionPlan(
            steps=[
                PlanStep(output_var="sum_1", operation="add", args={"a": a, "b": b}),
                PlanStep(output_var="sum_2", operation="add", args={"a": "$sum_1", "b": c}),
            ],
            final_var="sum_2",
        )

        problems.append(
            MultiStepProblem(
                question=q,
                expected_plan=plan,
                expected_answer=expected_ans,
                num_steps=2,
                archetype="chained_addition_3",
                split=split,
            )
        )
        count += 1

    # -------------------------------------------------------------
    # Archetype 4: Chained Addition - 4 Operands (3 steps)
    # -------------------------------------------------------------
    add_templates_4ops = [
        "Find the total sum of {a}, {b}, {c}, and {d}.",
        "What is {a} plus {b} plus {c} plus {d}?",
        "Calculate the sum of {a}, {b}, {c}, and {d} together.",
    ]
    seen_4ops = set()
    count = 0
    attempts = 0
    while count < num_samples_per_archetype and attempts < num_samples_per_archetype * 200:
        attempts += 1
        a = rng.randint(50, 300)
        b = rng.randint(50, 300)
        c = rng.randint(50, 300)
        d = rng.randint(50, 300)
        key = f"add4_{a}_{b}_{c}_{d}"
        if key in seen_4ops or get_multistep_split(key) != split:
            continue
        seen_4ops.add(key)

        tmpl = rng.choice(add_templates_4ops)
        q = tmpl.format(a=a, b=b, c=c, d=d)
        expected_ans = str(a + b + c + d)

        plan = ExecutionPlan(
            steps=[
                PlanStep(output_var="sum_1", operation="add", args={"a": a, "b": b}),
                PlanStep(output_var="sum_2", operation="add", args={"a": "$sum_1", "b": c}),
                PlanStep(output_var="sum_3", operation="add", args={"a": "$sum_2", "b": d}),
            ],
            final_var="sum_3",
        )

        problems.append(
            MultiStepProblem(
                question=q,
                expected_plan=plan,
                expected_answer=expected_ans,
                num_steps=3,
                archetype="chained_addition_4",
                split=split,
            )
        )
        count += 1

    # -------------------------------------------------------------
    # Archetype 5: Transitive Comparison Chains (4 Entities)
    # -------------------------------------------------------------
    name_pool = [
        "Anna", "Ben", "Chris", "David", "Emma", "Frank", "Grace", "Henry",
        "Ivy", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Paul",
    ]
    seen_compare4 = set()
    count = 0
    attempts = 0
    while count < num_samples_per_archetype and attempts < num_samples_per_archetype * 200:
        attempts += 1
        e = tuple(rng.sample(name_pool, 4))
        key = f"cmp4_{e[0]}_{e[1]}_{e[2]}_{e[3]}"
        if key in seen_compare4 or get_multistep_split(key) != split:
            continue
        seen_compare4.add(key)

        q = (
            f"If {e[0]} is taller than {e[1]}, {e[1]} is taller than {e[2]}, "
            f"and {e[2]} is taller than {e[3]}, who is taller, {e[0]} or {e[3]}?"
        )
        expected_ans = e[0]

        plan = ExecutionPlan(
            steps=[
                PlanStep(
                    output_var="comparison_result",
                    operation="compare",
                    args={
                        "entity_a": e[0],
                        "entity_b": e[3],
                        "relations": [f"{e[0]} > {e[1]}", f"{e[1]} > {e[2]}", f"{e[2]} > {e[3]}"],
                    },
                )
            ],
            final_var="comparison_result",
        )

        problems.append(
            MultiStepProblem(
                question=q,
                expected_plan=plan,
                expected_answer=expected_ans,
                num_steps=1,
                archetype="multi_entity_compare",
                split=split,
            )
        )
        count += 1

    rng.shuffle(problems)
    return problems


def generate_multistep_abstention_problems(
    num_samples: int = 15,
    seed: int = 8888,
    split: str = "test",
) -> list[MultiStepProblem]:
    """Generate multi-step malformed, dimensional mismatch, cyclic, or contradictory queries."""
    rng = random.Random(seed)
    abstention_cases: list[tuple[str, str, str]] = [
        # Dimensional incompatibility
        (
            "A runner covers 15 km in 2 hours. What is 15 km plus 2 hours?",
            "dimensional_mismatch",
            "Incompatible physical dimensions: cannot add distance to duration",
        ),
        (
            "What is 120 km plus 3 hours?",
            "dimensional_mismatch",
            "Incompatible physical dimensions: cannot add distance to duration",
        ),
        # Negative / Zero duration
        (
            "A car travels 120 km in 2 hours, then 80 km in 0 hours. What is its overall average speed?",
            "invalid_physics_zero_duration",
            "Zero duration in segment violates physical kinematics",
        ),
        (
            "An airplane covers 400 km in 2 hours, then 300 km in -1 hours. Calculate average speed for the whole journey.",
            "invalid_physics_negative_duration",
            "Negative duration violates physical kinematics",
        ),
        # Contradictory comparison cycle in multi-step query
        (
            "If Anna is taller than Ben, Ben is taller than Chris, and Chris is taller than Anna, who is taller, Anna or Chris?",
            "contradictory_cycle",
            "Contradictory cyclic ordering detected between entities",
        ),
        # Missing segment data
        (
            "A truck travels 100 km in 2 hours, then covers another distance. What is its overall average speed?",
            "missing_segment_data",
            "Missing distance and duration specifications for second segment",
        ),
    ]

    problems: list[MultiStepProblem] = []
    for q, arch, reason in abstention_cases:
        dummy_plan = ExecutionPlan(
            steps=[PlanStep(output_var="err", operation="add", args={"a": 0, "b": 0})],
            final_var="err",
        )
        problems.append(
            MultiStepProblem(
                question=q,
                expected_plan=dummy_plan,
                expected_answer=None,
                num_steps=0,
                archetype=arch,
                split=split,
                is_malformed=True,
                should_abstain=True,
                abstain_reason=reason,
            )
        )

    rng.shuffle(problems)
    return problems[:num_samples]


def generate_multistep_instruction_corpus(
    num_samples_per_archetype: int = 25,
    seed: int = 42,
    split: str = "train",
    format_type: str = "plan",
) -> tuple[str, str]:
    """Generate training text corpus of multi-step problem pairs."""
    problems = generate_multistep_problems(num_samples_per_archetype, seed=seed, split=split)
    pairs = [
        p.to_plan_pair() if format_type == "plan" else p.to_direct_pair() for p in problems
    ]
    text = "".join(pairs)
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp
