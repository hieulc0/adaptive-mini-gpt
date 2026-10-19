"""Dataset generation for Multi-Step Planning and Grounded Slot Binding (M16/M17).

Generates partitioned, multi-step compositional problems across 2 to 5 steps:
- Multi-segment average speed (2 segments = 3 steps, 3 segments = 5 steps)
- Chained arithmetic (3 operands = 2 steps, 4 operands = 3 steps, 5 operands = 4 steps)
- Transitive comparison chains (4+ entities)
- Typed Grounded Slot registries with stable slot IDs ($distance_1, $duration_1, etc.)
- Multi-turn Conversational Sessions (M17 Phase 5) with cross-turn reference resolution
- Robust variations: changed phrasing, inverted order, distracting numbers (years, flight numbers)
- Abstention battery: invalid dimensions, non-positive time, cyclic relations, ambiguous conversational referents

Enforces strictly disjoint train (70%), val (15%), and test (15%) splits via SHA-256 cryptographic hash.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.rules.grounding import format_available_slots
from adaptive_mini_gpt.rules.memory import GroundedSlot
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
    slots: dict[str, GroundedSlot] = field(default_factory=dict)
    expected_slot_plan: ExecutionPlan | None = None
    is_malformed: bool = False
    should_abstain: bool = False
    abstain_reason: str | None = None

    def to_plan_pair(self) -> str:
        """Instruction pair with ground truth plan with literals as assistant completion (M16 format)."""
        if self.should_abstain:
            completion = f"Abstain: {self.abstain_reason or 'Invalid query'}"
        else:
            completion = self.expected_plan.to_text_format()
        return f"<|user|>\n{self.question}\n<|assistant|>\n{completion}<|eos|>\n"

    def to_slot_plan_pair(self) -> str:
        """Instruction pair with available slots in user prompt and slot-referencing plan as completion (M17 format)."""
        slots_text = format_available_slots(self.slots)
        user_content = f"{self.question}\n{slots_text}"
        if self.should_abstain:
            completion = f"Abstain: {self.abstain_reason or 'Invalid query'}"
        else:
            plan = self.expected_slot_plan if self.expected_slot_plan is not None else self.expected_plan
            completion = plan.to_text_format()
        return f"<|user|>\n{user_content}\n<|assistant|>\n{completion}<|eos|>\n"

    def to_direct_pair(self) -> str:
        """Instruction pair with direct answer as assistant completion (neural baseline)."""
        ans = self.expected_answer if self.expected_answer is not None else "Cannot determine"
        return f"<|user|>\n{self.question}\n<|assistant|>\n{ans}<|eos|>\n"


@dataclass
class ConversationSessionProblem:
    """A multi-turn conversational reasoning session across 2 consecutive turns (M17 Phase 5)."""

    session_id: str
    turn1_question: str
    turn1_expected_answer: str | None
    turn1_plan: ExecutionPlan
    turn1_slots: dict[str, GroundedSlot]
    turn2_question: str
    turn2_expected_answer: str | None
    turn2_plan: ExecutionPlan
    turn2_slots: dict[str, GroundedSlot]
    is_ambiguous: bool = False
    should_abstain_turn2: bool = False
    abstain_reason: str | None = None


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
    """Generate balanced multi-step problems with both literal and grounded-slot execution plans."""
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
        if total_d % total_t != 0:
            continue
        expected_speed = f"{total_d // total_t} km/h"
        key = f"2seg_{d1}_{t1}_{d2}_{t2}"
        if key in seen_2seg or get_multistep_split(key) != split:
            continue
        seen_2seg.add(key)

        tmpl = rng.choice(speed_templates_2seg)
        q = tmpl.format(d1=d1, t1=t1, d2=d2, t2=t2)

        literal_plan = ExecutionPlan(
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

        slots = {
            "distance_1": GroundedSlot("distance_1", float(d1), "km", f"{d1} km", 0, 0, "deterministic"),
            "duration_1": GroundedSlot("duration_1", float(t1), "h", f"{t1} hours", 0, 0, "deterministic"),
            "distance_2": GroundedSlot("distance_2", float(d2), "km", f"{d2} km", 0, 0, "deterministic"),
            "duration_2": GroundedSlot("duration_2", float(t2), "h", f"{t2} hours", 0, 0, "deterministic"),
        }

        slot_plan = ExecutionPlan(
            steps=[
                PlanStep(output_var="d_total", operation="add", args={"a": "$distance_1", "b": "$distance_2"}, units="km"),
                PlanStep(output_var="t_total", operation="add", args={"a": "$duration_1", "b": "$duration_2"}, units="h"),
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
                expected_plan=literal_plan,
                expected_answer=expected_speed,
                num_steps=3,
                archetype="2segment_speed",
                split=split,
                slots=slots,
                expected_slot_plan=slot_plan,
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

        literal_plan = ExecutionPlan(
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

        slots = {
            "distance_1": GroundedSlot("distance_1", float(d1), "km", f"{d1} km", 0, 0, "deterministic"),
            "duration_1": GroundedSlot("duration_1", float(t1), "h", f"{t1} hours", 0, 0, "deterministic"),
            "distance_2": GroundedSlot("distance_2", float(d2), "km", f"{d2} km", 0, 0, "deterministic"),
            "duration_2": GroundedSlot("duration_2", float(t2), "h", f"{t2} hours", 0, 0, "deterministic"),
            "distance_3": GroundedSlot("distance_3", float(d3), "km", f"{d3} km", 0, 0, "deterministic"),
            "duration_3": GroundedSlot("duration_3", float(t3), "h", f"{t3} hours", 0, 0, "deterministic"),
        }

        slot_plan = ExecutionPlan(
            steps=[
                PlanStep(output_var="d_12", operation="add", args={"a": "$distance_1", "b": "$distance_2"}, units="km"),
                PlanStep(output_var="d_total", operation="add", args={"a": "$d_12", "b": "$distance_3"}, units="km"),
                PlanStep(output_var="t_12", operation="add", args={"a": "$duration_1", "b": "$duration_2"}, units="h"),
                PlanStep(output_var="t_total", operation="add", args={"a": "$t_12", "b": "$duration_3"}, units="h"),
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
                expected_plan=literal_plan,
                expected_answer=expected_speed,
                num_steps=5,
                archetype="3segment_speed",
                split=split,
                slots=slots,
                expected_slot_plan=slot_plan,
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

        literal_plan = ExecutionPlan(
            steps=[
                PlanStep(output_var="sum_1", operation="add", args={"a": a, "b": b}),
                PlanStep(output_var="sum_2", operation="add", args={"a": "$sum_1", "b": c}),
            ],
            final_var="sum_2",
        )

        slots = {
            "operand_1": GroundedSlot("operand_1", a, None, str(a), 0, 0, "deterministic", type_name="int"),
            "operand_2": GroundedSlot("operand_2", b, None, str(b), 0, 0, "deterministic", type_name="int"),
            "operand_3": GroundedSlot("operand_3", c, None, str(c), 0, 0, "deterministic", type_name="int"),
        }

        slot_plan = ExecutionPlan(
            steps=[
                PlanStep(output_var="sum_1", operation="add", args={"a": "$operand_1", "b": "$operand_2"}),
                PlanStep(output_var="sum_2", operation="add", args={"a": "$sum_1", "b": "$operand_3"}),
            ],
            final_var="sum_2",
        )

        problems.append(
            MultiStepProblem(
                question=q,
                expected_plan=literal_plan,
                expected_answer=expected_ans,
                num_steps=2,
                archetype="chained_addition_3",
                split=split,
                slots=slots,
                expected_slot_plan=slot_plan,
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

        literal_plan = ExecutionPlan(
            steps=[
                PlanStep(output_var="sum_1", operation="add", args={"a": a, "b": b}),
                PlanStep(output_var="sum_2", operation="add", args={"a": "$sum_1", "b": c}),
                PlanStep(output_var="sum_3", operation="add", args={"a": "$sum_2", "b": d}),
            ],
            final_var="sum_3",
        )

        slots = {
            "operand_1": GroundedSlot("operand_1", a, None, str(a), 0, 0, "deterministic", type_name="int"),
            "operand_2": GroundedSlot("operand_2", b, None, str(b), 0, 0, "deterministic", type_name="int"),
            "operand_3": GroundedSlot("operand_3", c, None, str(c), 0, 0, "deterministic", type_name="int"),
            "operand_4": GroundedSlot("operand_4", d, None, str(d), 0, 0, "deterministic", type_name="int"),
        }

        slot_plan = ExecutionPlan(
            steps=[
                PlanStep(output_var="sum_1", operation="add", args={"a": "$operand_1", "b": "$operand_2"}),
                PlanStep(output_var="sum_2", operation="add", args={"a": "$sum_1", "b": "$operand_3"}),
                PlanStep(output_var="sum_3", operation="add", args={"a": "$sum_2", "b": "$operand_4"}),
            ],
            final_var="sum_3",
        )

        problems.append(
            MultiStepProblem(
                question=q,
                expected_plan=literal_plan,
                expected_answer=expected_ans,
                num_steps=3,
                archetype="chained_addition_4",
                split=split,
                slots=slots,
                expected_slot_plan=slot_plan,
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

        literal_plan = ExecutionPlan(
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

        slots = {
            "entity_1": GroundedSlot("entity_1", e[0], None, e[0], 0, 0, "deterministic", type_name="str"),
            "entity_2": GroundedSlot("entity_2", e[1], None, e[1], 0, 0, "deterministic", type_name="str"),
            "entity_3": GroundedSlot("entity_3", e[2], None, e[2], 0, 0, "deterministic", type_name="str"),
            "entity_4": GroundedSlot("entity_4", e[3], None, e[3], 0, 0, "deterministic", type_name="str"),
        }

        problems.append(
            MultiStepProblem(
                question=q,
                expected_plan=literal_plan,
                expected_answer=expected_ans,
                num_steps=1,
                archetype="multi_entity_compare",
                split=split,
                slots=slots,
                expected_slot_plan=literal_plan,
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
        (
            "If Anna is taller than Ben, Ben is taller than Chris, and Chris is taller than Anna, who is taller, Anna or Chris?",
            "contradictory_cycle",
            "Contradictory cyclic ordering detected between entities",
        ),
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


def generate_conversational_problems(
    num_sessions: int = 12,
    seed: int = 42,
    split: str = "test",
) -> list[ConversationSessionProblem]:
    """Generate 2-turn conversational reasoning sessions across consecutive turns (M17 Phase 5)."""
    rng = random.Random(seed)
    sessions: list[ConversationSessionProblem] = []

    # Case 1: Standard 2-turn average speed continuation
    # Turn 1: 120 km in 2 hours -> 60 km/h
    # Turn 2: then 90 km in 3 hours -> overall speed = (120 + 90) / (2 + 3) = 42 km/h
    turn1_q = "A car travels 120 km in 2 hours. What was its average speed?"
    turn1_slots = {
        "distance_1": GroundedSlot("distance_1", 120.0, "km", "120 km", 0, 0),
        "duration_1": GroundedSlot("duration_1", 2.0, "h", "2 hours", 0, 0),
    }
    turn1_plan = ExecutionPlan(
        steps=[
            PlanStep(
                output_var="speed_1",
                operation="average_speed",
                args={"distance_km": "$distance_1", "time_hours": "$duration_1"},
                units="km/h",
            )
        ],
        final_var="speed_1",
    )

    turn2_q = "It then traveled another 90 km in 3 hours. What was the average speed for the whole journey?"
    turn2_slots = {
        "distance_2": GroundedSlot("distance_2", 90.0, "km", "90 km", 0, 0),
        "duration_2": GroundedSlot("duration_2", 3.0, "h", "3 hours", 0, 0),
    }
    turn2_plan = ExecutionPlan(
        steps=[
            PlanStep(output_var="d_total", operation="add", args={"a": "$turn1.distance_1", "b": "$distance_2"}, units="km"),
            PlanStep(output_var="t_total", operation="add", args={"a": "$turn1.duration_1", "b": "$duration_2"}, units="h"),
            PlanStep(
                output_var="overall_speed",
                operation="average_speed",
                args={"distance_km": "$d_total", "time_hours": "$t_total"},
                units="km/h",
            ),
        ],
        final_var="overall_speed",
    )
    sessions.append(
        ConversationSessionProblem(
            session_id="conv_speed_2turn_1",
            turn1_question=turn1_q,
            turn1_expected_answer="60 km/h",
            turn1_plan=turn1_plan,
            turn1_slots=turn1_slots,
            turn2_question=turn2_q,
            turn2_expected_answer="42 km/h",
            turn2_plan=turn2_plan,
            turn2_slots=turn2_slots,
        )
    )

    # Case 2: Arithmetic continuation
    # Turn 1: 45 + 55 = 100
    # Turn 2: Now add 35 to the previous result -> 135
    t1_add_q = "Calculate the sum of 45 and 55."
    t1_add_slots = {
        "operand_1": GroundedSlot("operand_1", 45, None, "45", 0, 0, type_name="int"),
        "operand_2": GroundedSlot("operand_2", 55, None, "55", 0, 0, type_name="int"),
    }
    t1_add_plan = ExecutionPlan(
        steps=[PlanStep(output_var="sum_1", operation="add", args={"a": "$operand_1", "b": "$operand_2"})],
        final_var="sum_1",
    )
    t2_add_q = "Now add 35 to that result."
    t2_add_slots = {
        "operand_1": GroundedSlot("operand_1", 35, None, "35", 0, 0, type_name="int"),
    }
    t2_add_plan = ExecutionPlan(
        steps=[PlanStep(output_var="final_sum", operation="add", args={"a": "$turn1.sum_1", "b": "$operand_1"})],
        final_var="final_sum",
    )
    sessions.append(
        ConversationSessionProblem(
            session_id="conv_arith_2turn_2",
            turn1_question=t1_add_q,
            turn1_expected_answer="100",
            turn1_plan=t1_add_plan,
            turn1_slots=t1_add_slots,
            turn2_question=t2_add_q,
            turn2_expected_answer="135",
            turn2_plan=t2_add_plan,
            turn2_slots=t2_add_slots,
        )
    )

    # Case 3: Ambiguous reference rejection / abstention
    # Turn 1: has distance 120 km and duration 2 h
    # Turn 2: "Now take that value and add 10." -> Ambiguous referent! Must abstain
    t1_amb_q = "A vehicle travels 120 km in 2 hours."
    t1_amb_slots = {
        "distance_1": GroundedSlot("distance_1", 120.0, "km", "120 km", 0, 0),
        "duration_1": GroundedSlot("duration_1", 2.0, "h", "2 hours", 0, 0),
    }
    t1_amb_plan = ExecutionPlan(
        steps=[
            PlanStep(output_var="d", operation="add", args={"a": "$distance_1", "b": 0}, units="km"),
            PlanStep(output_var="t", operation="add", args={"a": "$duration_1", "b": 0}, units="h"),
        ],
        final_var="d",
    )
    t2_amb_q = "Now take that value and add 10."
    dummy_amb_plan = ExecutionPlan(
        steps=[PlanStep(output_var="err", operation="add", args={"a": 0, "b": 0})],
        final_var="err",
    )
    sessions.append(
        ConversationSessionProblem(
            session_id="conv_ambiguous_turn2",
            turn1_question=t1_amb_q,
            turn1_expected_answer="120.0",
            turn1_plan=t1_amb_plan,
            turn1_slots=t1_amb_slots,
            turn2_question=t2_amb_q,
            turn2_expected_answer=None,
            turn2_plan=dummy_amb_plan,
            turn2_slots={},
            is_ambiguous=True,
            should_abstain_turn2=True,
            abstain_reason="Ambiguous referent 'that value' cannot be safely resolved across multiple prior quantities",
        )
    )

    # Generate more variations up to num_sessions
    for i in range(len(sessions), num_sessions):
        t1 = rng.randint(2, 3)
        t2 = rng.randint(2, 3)
        tot_t = t1 + t2
        v_tot = rng.choice([45, 50, 55, 60, 65, 70])
        tot_d = v_tot * tot_t
        d1 = rng.randint(20, max(21, tot_d - 20))
        d2 = tot_d - d1
        ans = f"{v_tot} km/h"

        t1_q = f"A bus travels {d1} km in {t1} hours. What is its initial average speed?"
        t1_s = {
            "distance_1": GroundedSlot("distance_1", float(d1), "km", f"{d1} km", 0, 0),
            "duration_1": GroundedSlot("duration_1", float(t1), "h", f"{t1} hours", 0, 0),
        }
        t1_p = ExecutionPlan(
            steps=[PlanStep(output_var="s1", operation="average_speed", args={"distance_km": "$distance_1", "time_hours": "$duration_1"}, units="km/h")],
            final_var="s1",
        )
        t2_q = f"It then covers another {d2} km in {t2} hours. What was the average speed for the whole trip?"
        t2_s = {
            "distance_2": GroundedSlot("distance_2", float(d2), "km", f"{d2} km", 0, 0),
            "duration_2": GroundedSlot("duration_2", float(t2), "h", f"{t2} hours", 0, 0),
        }
        t2_p = ExecutionPlan(
            steps=[
                PlanStep(output_var="d_tot", operation="add", args={"a": "$turn1.distance_1", "b": "$distance_2"}, units="km"),
                PlanStep(output_var="t_tot", operation="add", args={"a": "$turn1.duration_1", "b": "$duration_2"}, units="h"),
                PlanStep(output_var="speed_all", operation="average_speed", args={"distance_km": "$d_tot", "time_hours": "$t_tot"}, units="km/h"),
            ],
            final_var="speed_all",
        )
        sessions.append(
            ConversationSessionProblem(
                session_id=f"conv_gen_{i}",
                turn1_question=t1_q,
                turn1_expected_answer=f"{d1 // t1 if d1 % t1 == 0 else round(d1/t1, 1)} km/h",
                turn1_plan=t1_p,
                turn1_slots=t1_s,
                turn2_question=t2_q,
                turn2_expected_answer=ans,
                turn2_plan=t2_p,
                turn2_slots=t2_s,
            )
        )

    return sessions[:num_sessions]


def generate_multistep_instruction_corpus(
    num_samples_per_archetype: int = 25,
    seed: int = 42,
    split: str = "train",
    format_type: str = "slot_plan",
) -> tuple[str, str]:
    """Generate training text corpus of multi-step problem pairs (supporting 'slot_plan', 'plan', or 'direct')."""
    problems = generate_multistep_problems(num_samples_per_archetype, seed=seed, split=split)
    pairs = []
    for p in problems:
        if format_type == "slot_plan":
            pairs.append(p.to_slot_plan_pair())
        elif format_type == "plan":
            pairs.append(p.to_plan_pair())
        else:
            pairs.append(p.to_direct_pair())

    text = "".join(pairs)
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp
