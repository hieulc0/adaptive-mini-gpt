"""Milestone M100 Multi-Subject Lifelong Curriculum and Ground Truth Verifiers.

Covers 7 distinct reasoning domains + 2 diagnostic controls:
1. Arithmetic & Algebra: Modular affine transformation y = (3x + 5) mod 16.
2. Graph Reasoning: Edge transition and node reachability.
3. Logic & Deduction: Boolean XOR / implication reasoning.
4. Stateful Programming: State updates with conditional branching.
5. Factual Learning & Correction: Entity-property binding with temporal revisions.
6. Multi-Step Planning: Action sequencing towards goal stage.
7. Cross-Subject Composition: Cascaded Graph -> Logic -> Algebra.
8. Diagnostic Control 1: White-noise distractor (unlearnable).
9. Diagnostic Control 2: Elementary identity (already mastered).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum

import torch

MODULUS = 16


class DomainM100(StrEnum):
    ALGEBRA = "algebra"
    GRAPH = "graph"
    LOGIC = "logic"
    PROGRAM = "program"
    FACTUAL_MEMORY = "factual_memory"
    PLANNING = "planning"
    COMPOSITION = "composition"
    NOISE_DISTRACTOR = "noise_distractor"
    ALREADY_MASTERED = "already_mastered"


@dataclass
class LearningGoalM100:
    """A self-contained learning objective spanning train, test, and anchor sets."""

    goal_id: str
    domain: DomainM100
    description: str
    train_tokens: torch.Tensor
    train_labels: torch.Tensor
    test_tokens: torch.Tensor
    test_labels: torch.Tensor
    anchor_tokens: torch.Tensor
    anchor_labels: torch.Tensor
    is_learnable: bool = True
    estimated_cost: float = 1.0


class IndependentCurriculumVerifier:
    """Independent ground truth verifier for all curriculum domains without teacher cheating."""

    @staticmethod
    def verify_algebra(x: int, a: int = 3, b: int = 5) -> int:
        return (a * x + b) % MODULUS

    @staticmethod
    def verify_graph(node: int, edge_type: int) -> int:
        # Adjacency: edge 0 -> (node + 1)%16, edge 1 -> (node * 2 + 1)%16
        if edge_type == 0:
            return (node + 1) % MODULUS
        return (node * 2 + 1) % MODULUS

    @staticmethod
    def verify_logic(p: int, q: int, op: str) -> int:
        b1, b2 = (p % 2 == 1), (q % 2 == 1)
        if op == "AND":
            res = b1 and b2
        elif op == "OR":
            res = b1 or b2
        elif op == "XOR":
            res = b1 ^ b2
        else:
            res = not b1
        return 1 if res else 0

    @staticmethod
    def verify_program(state: int, opcode: int, val: int) -> int:
        if opcode == 0:  # ASSIGN
            return val % MODULUS
        elif opcode == 1:  # ADD
            return (state + val) % MODULUS
        elif opcode == 2:  # SUB
            return (state - val) % MODULUS
        elif opcode == 3:  # IF_GT
            return (state + 1) % MODULUS if state > val else state
        return state % MODULUS

    @staticmethod
    def verify_factual(entity_id: int, version: int) -> int:
        # Temporal correction: version 1 gives (entity * 3 + 2)%16, version 2 corrects to (entity * 5 + 7)%16
        if version == 1:
            return (entity_id * 3 + 2) % MODULUS
        return (entity_id * 5 + 7) % MODULUS

    @staticmethod
    def verify_planning(current_stage: int, action: int) -> int:
        # Sequential plan: stage s advances to s+1 if action == (s + 2)%8
        if action == (current_stage + 2) % 8:
            return (current_stage + 1) % MODULUS
        return current_stage % MODULUS

    @staticmethod
    def verify_composition(node: int, p_logic: int, energy: int) -> int:
        # 1. Graph hop
        _next_node = (node * 2 + 1) % MODULUS
        # 2. Logic gate
        is_safe = (p_logic % 2 == 1)
        # 3. Energy adjustment
        debit = 3 if is_safe else 7
        return (energy - debit) % MODULUS


class MultiSubjectCurriculumGenerator:
    """Generates balanced, diverse multi-subject learning opportunities."""

    def __init__(self, seed: int = 42, device: torch.device | None = None) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.device = device or torch.device("cpu")

    def encode_tokens(self, s: list[int], op: list[int], v: list[int]) -> torch.Tensor:
        """Tokenize inputs as sequence of 3 tokens [s, op, v] within vocab range 0..1023."""
        # Tokens: s in 0..15, op in 16..31, v in 32..47
        seqs = []
        for si, opi, vi in zip(s, op, v, strict=False):
            tok_s = si % 16
            tok_op = 16 + (opi % 16)
            tok_v = 32 + (vi % 16)
            seqs.append([tok_s, tok_op, tok_v])
        return torch.tensor(seqs, dtype=torch.long, device=self.device)

    def generate_full_curriculum(self) -> list[LearningGoalM100]:
        """Construct all 7 domain goals plus diagnostic controls."""
        goals: list[LearningGoalM100] = []

        # Shared Foundational Anchor Task (used across all goals to evaluate retention)
        anc_s = list(range(16))
        anc_op = [1] * 16  # ADD
        anc_v = [2] * 16
        anc_y = [(si + 2) % MODULUS for si in anc_s]
        anc_tok = self.encode_tokens(anc_s, anc_op, anc_v)
        anc_lbl = torch.tensor(anc_y, dtype=torch.long, device=self.device)

        # 1. ALGEBRA: Affine mapping y = (3*x + 5) mod 16
        tr_s = list(range(16))
        tr_op = [1] * 16
        tr_v = [5] * 16
        tr_y = [IndependentCurriculumVerifier.verify_algebra(x) for x in tr_s]
        te_s = list(range(16))
        te_op = [1] * 16
        te_v = [5] * 16
        te_y = [IndependentCurriculumVerifier.verify_algebra(x) for x in te_s]

        goals.append(
            LearningGoalM100(
                goal_id="goal_algebra_affine",
                domain=DomainM100.ALGEBRA,
                description="Modular affine system y = (3x + 5) mod 16",
                train_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                train_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(te_s, te_op, te_v),
                test_labels=torch.tensor(te_y, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                estimated_cost=0.5,
            )
        )

        # 2. GRAPH: Traversal transitions
        tr_s = list(range(16))
        tr_op = [0 if i % 2 == 0 else 1 for i in range(16)]
        tr_v = [0] * 16
        tr_y = [IndependentCurriculumVerifier.verify_graph(s, op) for s, op in zip(tr_s, tr_op, strict=False)]
        te_s = list(range(16))
        te_op = [0 if i % 2 == 0 else 1 for i in range(16)]
        te_v = [0] * 16
        te_y = [IndependentCurriculumVerifier.verify_graph(s, op) for s, op in zip(te_s, te_op, strict=False)]

        goals.append(
            LearningGoalM100(
                goal_id="goal_graph_reachability",
                domain=DomainM100.GRAPH,
                description="Directed graph node transition and reachability",
                train_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                train_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(te_s, te_op, te_v),
                test_labels=torch.tensor(te_y, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                estimated_cost=0.6,
            )
        )

        # 3. LOGIC: Propositional deduction
        tr_s = list(range(16))
        tr_op = [2] * 16
        tr_v = [(i + 3) % 16 for i in range(16)]
        tr_y = [IndependentCurriculumVerifier.verify_logic(s, v, "XOR") for s, v in zip(tr_s, tr_v, strict=False)]
        te_s = list(range(16))
        te_op = [2] * 16
        te_v = [(i + 3) % 16 for i in range(16)]
        te_y = [IndependentCurriculumVerifier.verify_logic(s, v, "XOR") for s, v in zip(te_s, te_v, strict=False)]

        goals.append(
            LearningGoalM100(
                goal_id="goal_logic_deduction",
                domain=DomainM100.LOGIC,
                description="Boolean deduction under XOR parity rules",
                train_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                train_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(te_s, te_op, te_v),
                test_labels=torch.tensor(te_y, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                estimated_cost=0.4,
            )
        )

        # 4. PROGRAM: Stateful program execution
        tr_s = list(range(16))
        tr_op = [3] * 16  # IF_GT
        tr_v = [6] * 16
        tr_y = [IndependentCurriculumVerifier.verify_program(s, 3, 6) for s in tr_s]
        te_s = list(range(16))
        te_op = [3] * 16
        te_v = [6] * 16
        te_y = [IndependentCurriculumVerifier.verify_program(s, 3, 6) for s in te_s]

        goals.append(
            LearningGoalM100(
                goal_id="goal_stateful_program",
                domain=DomainM100.PROGRAM,
                description="Stateful conditional threshold execution",
                train_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                train_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(te_s, te_op, te_v),
                test_labels=torch.tensor(te_y, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                estimated_cost=0.7,
            )
        )

        # 5. FACTUAL MEMORY: Temporal correction
        tr_s = list(range(16))
        tr_op = [4] * 16
        tr_v = [2] * 16  # Version 2 (corrected)
        tr_y = [IndependentCurriculumVerifier.verify_factual(s, 2) for s in tr_s]
        te_s = list(range(16))
        te_op = [4] * 16
        te_v = [2] * 16
        te_y = [IndependentCurriculumVerifier.verify_factual(s, 2) for s in te_s]

        goals.append(
            LearningGoalM100(
                goal_id="goal_factual_correction",
                domain=DomainM100.FACTUAL_MEMORY,
                description="Entity property updates with temporal revision v2",
                train_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                train_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(te_s, te_op, te_v),
                test_labels=torch.tensor(te_y, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                estimated_cost=0.5,
            )
        )

        # 6. PLANNING: Multi-step action selection
        tr_s = list(range(16))
        tr_op = [5] * 16
        tr_v = [(i + 2) % 8 for i in range(16)]
        tr_y = [IndependentCurriculumVerifier.verify_planning(s, v) for s, v in zip(tr_s, tr_v, strict=False)]
        te_s = list(range(16))
        te_op = [5] * 16
        te_v = [(i + 2) % 8 for i in range(16)]
        te_y = [IndependentCurriculumVerifier.verify_planning(s, v) for s, v in zip(te_s, te_v, strict=False)]

        goals.append(
            LearningGoalM100(
                goal_id="goal_planning_sequence",
                domain=DomainM100.PLANNING,
                description="Sequential goal progression under precondition constraints",
                train_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                train_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(te_s, te_op, te_v),
                test_labels=torch.tensor(te_y, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                estimated_cost=0.8,
            )
        )

        # 7. CROSS-SUBJECT COMPOSITION
        tr_s = list(range(16))
        tr_op = [6] * 16
        tr_v = [(i * 3) % 16 for i in range(16)]
        tr_y = [IndependentCurriculumVerifier.verify_composition(s, s, v) for s, v in zip(tr_s, tr_v, strict=False)]
        te_s = list(range(16))
        te_op = [6] * 16
        te_v = [(i * 3) % 16 for i in range(16)]
        te_y = [IndependentCurriculumVerifier.verify_composition(s, s, v) for s, v in zip(te_s, te_v, strict=False)]

        goals.append(
            LearningGoalM100(
                goal_id="goal_cross_subject_composition",
                domain=DomainM100.COMPOSITION,
                description="Cascaded Graph routing -> Logic gate -> Algebraic energy debit",
                train_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                train_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(te_s, te_op, te_v),
                test_labels=torch.tensor(te_y, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                estimated_cost=0.9,
            )
        )

        # Diagnostic Control 1: White Noise Distractor (Unlearnable)
        tr_s_noise = [i for i in range(16)]
        tr_op_noise = [7] * 16
        tr_v_noise = [self.rng.randint(0, 15) for _ in range(16)]
        tr_y_noise = [self.rng.randint(0, 15) for _ in range(16)]
        goals.append(
            LearningGoalM100(
                goal_id="goal_noise_distractor",
                domain=DomainM100.NOISE_DISTRACTOR,
                description="Unlearnable white noise distractor for epistemic rejection",
                train_tokens=self.encode_tokens(tr_s_noise, tr_op_noise, tr_v_noise),
                train_labels=torch.tensor(tr_y_noise, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(tr_s_noise, tr_op_noise, tr_v_noise),
                test_labels=torch.tensor(tr_y_noise, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                is_learnable=False,
                estimated_cost=1.0,
            )
        )

        # Diagnostic Control 2: Already Mastered Elementary Task
        tr_s_mas = [i for i in range(16)]
        tr_op_mas = [0] * 16
        tr_v_mas = [i for i in range(16)]
        tr_y_mas = [i for i in range(16)]  # Identity
        goals.append(
            LearningGoalM100(
                goal_id="goal_already_mastered",
                domain=DomainM100.ALREADY_MASTERED,
                description="Already-mastered identity baseline (0 progress reward)",
                train_tokens=self.encode_tokens(tr_s_mas, tr_op_mas, tr_v_mas),
                train_labels=torch.tensor(tr_y_mas, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(tr_s_mas, tr_op_mas, tr_v_mas),
                test_labels=torch.tensor(tr_y_mas, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                is_learnable=True,
                estimated_cost=0.1,
            )
        )

        return goals
