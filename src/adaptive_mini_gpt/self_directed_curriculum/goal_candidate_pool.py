"""Candidate Goal Pool & Independent Verifier for Milestone M99.

Generates structured learning opportunities across five subjects plus diagnostic controls:
1. Algebra (modular linear mappings)
2. Graph Reasoning (directed path traversals & edge costs)
3. Logic & Deduction (Boolean constraints and conditional inference)
4. Stateful Programming (accumulators, loop bounds, conditional updates)
5. Cross-Subject Composition (Graph navigation + Energy costs + Battery guards)
6. Diagnostic Controls:
   - Noise Distractor: Random non-deterministic labels (unlearnable; policy should defer).
   - Already Mastered: Foundational arithmetic (policy should deprioritize).
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


class SubjectCategory(str, Enum):
    """Subject categories in the candidate goal pool."""

    ALGEBRA = "algebra"
    GRAPH = "graph"
    LOGIC = "logic"
    PROGRAM = "program"
    COMPOSITION = "composition"
    NOISE_DISTRACTOR = "noise_distractor"
    ALREADY_MASTERED = "already_mastered"


@dataclass(frozen=True)
class LearningOpportunity:
    """A self-contained candidate task family available for practice."""

    goal_id: str
    subject: SubjectCategory
    description: str
    diagnostic_probe_examples: list[tuple[int, int, int, int]]  # (s, op, v, target_y)
    training_examples: list[tuple[int, int, int, int]]
    held_out_test_examples: list[tuple[int, int, int, int]]
    anchor_replay_examples: list[tuple[int, int, int, int]]
    is_learnable: bool
    estimated_cost: float  # [0.1 .. 1.0]


class IndependentTaskVerifier:
    """Independent ground-truth oracle for active probing and supervision ONLY.

    Strictly isolated from the student's forward inference path.
    """

    @classmethod
    def query(cls, goal_id: str, s: int, op: int, v: int) -> int:
        """Answer an active query without revealing solution algorithms."""
        if "algebra" in goal_id:
            # Affine map (3*s + 5) % 16
            return (3 * s + 5) % MODULUS
        elif "graph" in goal_id:
            # Traversal weight
            return (v + 1) % MODULUS
        elif "logic" in goal_id:
            # Conditional guard
            return 1 if s > v else 0
        elif "program" in goal_id:
            # Stateful step
            return (s + v * 2) % MODULUS
        elif "comp" in goal_id:
            # Composed planning
            return (s + v + 3) % MODULUS
        elif "mastered" in goal_id:
            # Simple addition
            return (s + v) % MODULUS
        else:
            # Noise distractor
            return random.randint(0, MODULUS - 1)


class GoalCandidatePoolGenerator:
    """Constructs the structured pool of candidate learning opportunities."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_candidate_pool(self) -> list[LearningOpportunity]:
        """Generate a diverse pool of candidate goals for autonomous selection."""
        pool: list[LearningOpportunity] = []

        # 1. Already Mastered: Addition s + v
        mastered_train = [(x, OP_ADD, 2, (x + 2) % MODULUS) for x in range(12)]
        mastered_probe = [(x, OP_ADD, 2, (x + 2) % MODULUS) for x in range(12, 14)]
        mastered_test = [(x, OP_ADD, 2, (x + 2) % MODULUS) for x in range(14, 16)]
        pool.append(
            LearningOpportunity(
                goal_id="goal_mastered_arithmetic",
                subject=SubjectCategory.ALREADY_MASTERED,
                description="Foundational addition (x + 2) % 16",
                diagnostic_probe_examples=mastered_probe,
                training_examples=mastered_train,
                held_out_test_examples=mastered_test,
                anchor_replay_examples=mastered_probe,
                is_learnable=True,
                estimated_cost=0.1,
            )
        )

        # 2. Algebra: Affine transformation y = (3*x + 5) % 16
        alg_train = [(x, OP_ADD, (x * 2 + 5) % MODULUS, (3 * x + 5) % MODULUS) for x in range(12)]
        alg_probe = [(x, OP_ADD, (x * 2 + 5) % MODULUS, (3 * x + 5) % MODULUS) for x in range(12, 14)]
        alg_test = [(x, OP_ADD, (x * 2 + 5) % MODULUS, (3 * x + 5) % MODULUS) for x in range(14, 16)]
        pool.append(
            LearningOpportunity(
                goal_id="goal_algebra_affine",
                subject=SubjectCategory.ALGEBRA,
                description="Unfamiliar affine mapping (3*x + 5) % 16",
                diagnostic_probe_examples=alg_probe,
                training_examples=alg_train,
                held_out_test_examples=alg_test,
                anchor_replay_examples=mastered_probe,
                is_learnable=True,
                estimated_cost=0.4,
            )
        )

        # 3. Graph: Step traversal edge weights
        graph_train = [(u, OP_ASSIGN, (u * 3 + 1) % MODULUS, (u * 3 + 1) % MODULUS) for u in range(12)]
        graph_probe = [(u, OP_ASSIGN, (u * 3 + 1) % MODULUS, (u * 3 + 1) % MODULUS) for u in range(12, 14)]
        graph_test = [(u, OP_ASSIGN, (u * 3 + 1) % MODULUS, (u * 3 + 1) % MODULUS) for u in range(14, 16)]
        pool.append(
            LearningOpportunity(
                goal_id="goal_graph_traversal",
                subject=SubjectCategory.GRAPH,
                description="Directed graph traversal edge costs",
                diagnostic_probe_examples=graph_probe,
                training_examples=graph_train,
                held_out_test_examples=graph_test,
                anchor_replay_examples=mastered_probe,
                is_learnable=True,
                estimated_cost=0.4,
            )
        )

        # 4. Logic: Boolean conditions & threshold safety
        logic_train = [(p, OP_IF_GT, (p + 3) % MODULUS, 1 if p > ((p + 3) % MODULUS) else 0) for p in range(12)]
        logic_probe = [(p, OP_IF_GT, (p + 3) % MODULUS, 1 if p > ((p + 3) % MODULUS) else 0) for p in range(12, 14)]
        logic_test = [(p, OP_IF_GT, (p + 3) % MODULUS, 1 if p > ((p + 3) % MODULUS) else 0) for p in range(14, 16)]
        pool.append(
            LearningOpportunity(
                goal_id="goal_logic_guard",
                subject=SubjectCategory.LOGIC,
                description="Threshold safety logic guard (s > v)",
                diagnostic_probe_examples=logic_probe,
                training_examples=logic_train,
                held_out_test_examples=logic_test,
                anchor_replay_examples=mastered_probe,
                is_learnable=True,
                estimated_cost=0.3,
            )
        )

        # 5. Program: Stateful accumulator step
        prog_train = [(x, OP_ADD, (x + 4) % MODULUS, (2 * x + 4) % MODULUS) for x in range(12)]
        prog_probe = [(x, OP_ADD, (x + 4) % MODULUS, (2 * x + 4) % MODULUS) for x in range(12, 14)]
        prog_test = [(x, OP_ADD, (x + 4) % MODULUS, (2 * x + 4) % MODULUS) for x in range(14, 16)]
        pool.append(
            LearningOpportunity(
                goal_id="goal_program_accum",
                subject=SubjectCategory.PROGRAM,
                description="Stateful program accumulator step",
                diagnostic_probe_examples=prog_probe,
                training_examples=prog_train,
                held_out_test_examples=prog_test,
                anchor_replay_examples=mastered_probe,
                is_learnable=True,
                estimated_cost=0.5,
            )
        )

        # 6. Composition: Composed path planning under energy
        comp_train = [(x, OP_ADD, (x + 7) % MODULUS, (2 * x + 7) % MODULUS) for x in range(12)]
        comp_probe = [(x, OP_ADD, (x + 7) % MODULUS, (2 * x + 7) % MODULUS) for x in range(12, 14)]
        comp_test = [(x, OP_ADD, (x + 7) % MODULUS, (2 * x + 7) % MODULUS) for x in range(14, 16)]
        pool.append(
            LearningOpportunity(
                goal_id="goal_comp_planning",
                subject=SubjectCategory.COMPOSITION,
                description="Cross-subject graph planning with energy constraint",
                diagnostic_probe_examples=comp_probe,
                training_examples=comp_train,
                held_out_test_examples=comp_test,
                anchor_replay_examples=mastered_probe,
                is_learnable=True,
                estimated_cost=0.6,
            )
        )

        # 7. Noise Distractor: Unlearnable random noise (policy should learn to defer/avoid)
        rng_noise = random.Random(self.seed + 999)
        noise_train = [(x, OP_ADD, (x + 1) % MODULUS, rng_noise.randint(0, MODULUS - 1)) for x in range(12)]
        noise_probe = [(x, OP_ADD, (x + 1) % MODULUS, rng_noise.randint(0, MODULUS - 1)) for x in range(12, 14)]
        noise_test = [(x, OP_ADD, (x + 1) % MODULUS, rng_noise.randint(0, MODULUS - 1)) for x in range(14, 16)]
        pool.append(
            LearningOpportunity(
                goal_id="goal_noise_distractor",
                subject=SubjectCategory.NOISE_DISTRACTOR,
                description="Random non-deterministic noise (unlearnable)",
                diagnostic_probe_examples=noise_probe,
                training_examples=noise_train,
                held_out_test_examples=noise_test,
                anchor_replay_examples=mastered_probe,
                is_learnable=False,
                estimated_cost=0.8,
            )
        )

        return pool
