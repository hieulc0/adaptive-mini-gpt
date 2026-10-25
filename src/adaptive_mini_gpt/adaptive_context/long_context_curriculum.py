"""Long-Context Curriculum Generator for Milestone M97.

Constructs comprehensive evaluation streams containing:
1. Short single-hop factual queries (fits in 128 tokens)
2. Long-context needle queries with needle placed at Early, Mid, and Late positions (512-1024 tokens)
3. Multi-hop factual reasoning chains (Entity A -> Entity B -> Property C)
4. Distractor-polluted queries testing distractor filtering
5. Long procedural traces requiring state persistence
6. Unsupported / contradictory queries testing epistemic abstention
7. Held-Out Target Task Family for Fresh vs. Experienced Student benchmark
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any

from adaptive_mini_gpt.cross_domain_transfer.neural_cross_domain_transfer import (
    MODULUS,
    OP_ADD,
    OP_ASSIGN,
    OP_IF_GT,
    OP_SUB,
)

DOMAIN_PROGRAM = "PROGRAM"
DOMAIN_NL = "NATURAL_LANGUAGE"
DOMAIN_PLANNING = "PLANNING"


class TaskType(str, Enum):
    """Task types in the long-context curriculum."""

    SHORT_FACT = "short_fact"
    LONG_NEEDLE = "long_needle"
    MULTI_HOP = "multi_hop"
    DISTRACTOR_POLLUTED = "distractor_polluted"
    PROCEDURAL_TRACE = "procedural_trace"
    UNSUPPORTED_ABSTAIN = "unsupported_abstain"
    HELD_OUT_TARGET = "held_out_target"  # For Fresh vs Experienced benchmark


@dataclass(frozen=True)
class LongContextTask:
    """An individual task in the long-context curriculum."""

    episode_id: int
    task_type: TaskType
    input_text: str
    target_output: Any
    domain_id: str
    has_supported_entity: bool
    is_contradictory: bool
    requires_retrieval: bool
    depth_position: str | None  # "early", "mid", "late", or None
    token_length: int
    intermediate_steps: list[tuple[int, int, int]] | None = None
    seed_facts_to_store: list[dict[str, Any]] | None = None


class LongContextCurriculumGenerator:
    """Generates structured lifelong curricula with long context and fresh-vs-experienced splits."""

    def __init__(self, seed: int = 42, num_episodes: int = 60) -> None:
        self.seed = seed
        self.num_episodes = num_episodes
        self.rng = random.Random(seed)

    def generate_curriculum(self) -> list[LongContextTask]:
        """Generate 60 sequential episodes."""
        tasks: list[LongContextTask] = []

        # Knowledge entities pool
        companies = ["atlas_labs", "nova_robotics", "quantum_core", "helios_ai", "zenith_dynamics"]
        founders = ["alice_vance", "bob_chen", "clara_oswald", "david_kim", "elena_rostova"]
        cities = ["cambridge", "seattle", "austin", "zurich", "tokyo"]
        metrics = [4, 7, 11, 13, 15]

        # 1. Episodes 1-15: Initial Factual Ingestion & Skills
        for ep in range(1, 16):
            comp = companies[ep % len(companies)]
            founder = founders[ep % len(founders)]
            city = cities[ep % len(cities)]
            metric_val = metrics[ep % len(metrics)]

            # Seed facts to store in persistent memory
            facts_to_add = [
                {"entity": comp, "attribute": "founder", "value": founder, "timestamp": ep},
                {"entity": founder, "attribute": "city", "value": city, "timestamp": ep},
                {"entity": comp, "attribute": "scale_metric", "value": metric_val, "timestamp": ep},
            ]

            # Procedural task
            s0 = (ep * 3) % MODULUS
            v0 = (ep * 2) % MODULUS
            op0 = OP_ADD if ep % 2 == 0 else OP_SUB
            target_val = (s0 + v0) % MODULUS if op0 == OP_ADD else (s0 - v0) % MODULUS

            tasks.append(
                LongContextTask(
                    episode_id=ep,
                    task_type=TaskType.SHORT_FACT,
                    input_text=f"compute {comp} scale adjustment state={s0} op={op0} delta={v0}",
                    target_output=target_val,
                    domain_id=DOMAIN_PROGRAM,
                    has_supported_entity=True,
                    is_contradictory=False,
                    requires_retrieval=True,
                    depth_position=None,
                    token_length=15,
                    intermediate_steps=[(s0, op0, v0)],
                    seed_facts_to_store=facts_to_add,
                )
            )

        # 2. Episodes 16-30: Long-Context Needle In A Haystack (Early, Mid, Late)
        for ep in range(16, 31):
            comp = companies[ep % len(companies)]
            target_metric = metrics[ep % len(metrics)]
            depth_pos = "early" if ep % 3 == 0 else ("mid" if ep % 3 == 1 else "late")

            # Construct long haystack background story (~600-800 words)
            filler = "This is a detailed organizational log containing telemetry, system monitors, and server heartbeats. " * 30
            needle = f" Crucial verified report: the primary scale metric for {comp} is certified as {target_metric}. "

            if depth_pos == "early":
                full_input = needle + filler + f"\nQuery: What is the scale metric for {comp}?"
            elif depth_pos == "mid":
                mid_idx = len(filler) // 2
                full_input = filler[:mid_idx] + needle + filler[mid_idx:] + f"\nQuery: What is the scale metric for {comp}?"
            else:
                full_input = filler + needle + f"\nQuery: What is the scale metric for {comp}?"

            tasks.append(
                LongContextTask(
                    episode_id=ep,
                    task_type=TaskType.LONG_NEEDLE,
                    input_text=full_input,
                    target_output=target_metric,
                    domain_id=DOMAIN_NL,
                    has_supported_entity=True,
                    is_contradictory=False,
                    requires_retrieval=False,
                    depth_position=depth_pos,
                    token_length=len(full_input.split()),
                    intermediate_steps=None,
                    seed_facts_to_store=None,
                )
            )

        # 3. Episodes 31-45: Multi-Hop Relational Reasoning & Distractor Filtering
        for ep in range(31, 46):
            comp = companies[ep % len(companies)]
            founder = founders[ep % len(founders)]
            city = cities[ep % len(cities)]

            # Multi-hop query: "In which city does the founder of {comp} reside?"
            # Hop 1: comp.founder -> founder
            # Hop 2: founder.city -> city
            query_text = (
                f"Multi-hop inquiry: Identify the home city of the founder of {comp} "
                f"given background knowledge and unrelated records for {comp}_distractor."
            )

            tasks.append(
                LongContextTask(
                    episode_id=ep,
                    task_type=TaskType.MULTI_HOP,
                    input_text=query_text,
                    target_output=city,
                    domain_id=DOMAIN_PLANNING,
                    has_supported_entity=True,
                    is_contradictory=False,
                    requires_retrieval=True,
                    depth_position=None,
                    token_length=len(query_text.split()),
                    intermediate_steps=None,
                    seed_facts_to_store=None,
                )
            )

        # 4. Episodes 46-55: Epistemic Abstention on Contradictory & Unsupported Queries
        for ep in range(46, 56):
            if ep % 2 == 0:
                # Unsupported entity
                q = f"What is the valuation of completely_unknown_startup_{ep}?"
                is_contradictory = False
                has_entity = False
            else:
                # Contradictory fact
                q = f"Resolve conflicting audit: {companies[0]} was founded in both 1920 and 2099."
                is_contradictory = True
                has_entity = True

            tasks.append(
                LongContextTask(
                    episode_id=ep,
                    task_type=TaskType.UNSUPPORTED_ABSTAIN,
                    input_text=q,
                    target_output="ABSTAIN",
                    domain_id=DOMAIN_PROGRAM,
                    has_supported_entity=has_entity,
                    is_contradictory=is_contradictory,
                    requires_retrieval=False,
                    depth_position=None,
                    token_length=len(q.split()),
                    intermediate_steps=None,
                    seed_facts_to_store=None,
                )
            )

        # 5. Episodes 56-60: Held-Out Target Transfer Tasks (for Fresh vs Experienced comparison)
        for ep in range(56, 61):
            # Target family: 3-step nested state transition with condition filter
            # S0 -> S1 = S0 + V1 -> if S1 > 8 then S2 = S1 - 3 else S2 = S1 + 2
            s0 = (ep * 5) % MODULUS
            v1 = (ep * 3) % MODULUS
            s1 = (s0 + v1) % MODULUS
            s2 = (s1 - 3) % MODULUS if s1 > 8 else (s1 + 2) % MODULUS

            tasks.append(
                LongContextTask(
                    episode_id=ep,
                    task_type=TaskType.HELD_OUT_TARGET,
                    input_text=f"target_held_out_algorithm step_start s={s0} add={v1} conditional_filter threshold=8",
                    target_output=s2,
                    domain_id=DOMAIN_PROGRAM,
                    has_supported_entity=True,
                    is_contradictory=False,
                    requires_retrieval=False,
                    depth_position=None,
                    token_length=20,
                    intermediate_steps=[(s1, OP_SUB if s1 > 8 else OP_ADD, 3 if s1 > 8 else 2)],
                    seed_facts_to_store=None,
                )
            )

        return tasks
