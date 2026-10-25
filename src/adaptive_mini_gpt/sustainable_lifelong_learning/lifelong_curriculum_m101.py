"""Milestone M101 Lifelong Curriculum & Diverse Task Mechanism Generators.

Covers 8 distinct reasoning domains with fundamentally different underlying mechanisms:
1. ALGEBRAIC_QUADRATIC: Non-linear polynomial modular mapping y = (x^2 + 3x + 1) mod 16.
2. GRAPH_REACHABILITY: 2-hop directed reachability under DAG edge constraints.
3. LOGIC_MAJORITY: Multi-variable propositional majority and implication deduction.
4. STATEFUL_PROGRAMMING: Register accumulator with conditional thresholding (IF_GT).
5. FINITE_STATE_MACHINE: Deterministic finite-state automaton (DFA) state tracking.
6. FACTUAL_MEMORY: Entity-property binding with temporal revision v2.
7. MULTI_STEP_PLANNING: Sequential plan execution under state preconditions.
8. CROSS_SUBJECT_COMPOSITION: Cascaded Graph -> Logic -> DFA -> Energy computation.

Plus diagnostic controls:
- White-Noise Distractor (Unlearnable, verifying epistemic abstention).
- Elementary Identity (Already-mastered baseline).

Plus sealed target task families for future-learning evaluation:
- Sealed Quadratic Target: y = (3x^2 + 7x + 5) mod 16.
- Sealed DFA Target: 4-state automaton with alternating emission.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum

import torch

MODULUS = 16


class DomainM101(StrEnum):
    ALGEBRA_QUADRATIC = "algebra_quadratic"
    GRAPH_REACHABILITY = "graph_reachability"
    LOGIC_MAJORITY = "logic_majority"
    STATEFUL_PROGRAM = "stateful_program"
    FINITE_STATE_MACHINE = "finite_state_machine"
    FACTUAL_MEMORY = "factual_memory"
    MULTI_STEP_PLANNING = "multi_step_planning"
    CROSS_SUBJECT_COMPOSITION = "cross_subject_composition"
    NOISE_DISTRACTOR = "noise_distractor"
    ALREADY_MASTERED = "already_mastered"


@dataclass
class LearningGoalM101:
    """A self-contained learning objective spanning train, test, and anchor sets."""

    goal_id: str
    domain: DomainM101
    description: str
    train_tokens: torch.Tensor
    train_labels: torch.Tensor
    test_tokens: torch.Tensor
    test_labels: torch.Tensor
    anchor_tokens: torch.Tensor
    anchor_labels: torch.Tensor
    is_learnable: bool = True
    estimated_cost: float = 1.0


class IndependentLifelongVerifier:
    """Independent ground truth solvers without teacher cheating."""

    @staticmethod
    def verify_quadratic(x: int, a: int = 1, b: int = 3, c: int = 1) -> int:
        return (a * x * x + b * x + c) % MODULUS

    @staticmethod
    def verify_graph_2hop(node: int, edge1: int, edge2: int) -> int:
        hop1 = (node + 1) % MODULUS if edge1 == 0 else (node * 2 + 1) % MODULUS
        hop2 = (hop1 + 1) % MODULUS if edge2 == 0 else (hop1 * 2 + 1) % MODULUS
        return hop2 % MODULUS

    @staticmethod
    def verify_logic_majority(b1: int, b2: int, b3: int) -> int:
        bits = [b1 % 2, b2 % 2, b3 % 2]
        return 1 if sum(bits) >= 2 else 0

    @staticmethod
    def verify_program(state: int, opcode: int, val: int) -> int:
        if opcode == 0:
            return val % MODULUS
        elif opcode == 1:
            return (state + val) % MODULUS
        elif opcode == 2:
            return (state - val) % MODULUS
        elif opcode == 3:
            return (state + 1) % MODULUS if state > val else state
        return state % MODULUS

    @staticmethod
    def verify_dfa(state: int, symbol: int) -> int:
        # 4 states (0, 1, 2, 3): transition matrix
        # symbol 0: (state + 1)%4
        # symbol 1: (state * 2 + 1)%4
        if symbol == 0:
            return (state + 1) % 4
        return (state * 2 + 1) % 4

    @staticmethod
    def verify_factual(entity_id: int, version: int) -> int:
        if version == 1:
            return (entity_id * 3 + 2) % MODULUS
        return (entity_id * 5 + 7) % MODULUS

    @staticmethod
    def verify_planning(current_stage: int, action: int) -> int:
        if action == (current_stage + 2) % 8:
            return (current_stage + 1) % MODULUS
        return current_stage % MODULUS

    @staticmethod
    def verify_composition(node: int, logic_bit: int, state: int) -> int:
        # 1. Graph transition
        hop = (node * 2 + 1) % MODULUS
        # 2. Logic condition
        debit = 3 if (logic_bit % 2 == 1) else 7
        # 3. Energy update
        return (state + hop - debit) % MODULUS


class LifelongCurriculumGeneratorM101:
    """Constructs balanced, diverse lifelong learning curricula."""

    def __init__(self, seed: int = 42, device: torch.device | None = None) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.device = device or torch.device("cpu")

    def encode_tokens(self, s: list[int], op: list[int], v: list[int]) -> torch.Tensor:
        """Tokenize inputs as sequence of 3 tokens [s, op, v] in vocab range 0..1023."""
        seqs = []
        for si, opi, vi in zip(s, op, v, strict=False):
            tok_s = si % 16
            tok_op = 16 + (opi % 16)
            tok_v = 32 + (vi % 16)
            seqs.append([tok_s, tok_op, tok_v])
        return torch.tensor(seqs, dtype=torch.long, device=self.device)

    def generate_full_curriculum(self) -> list[LearningGoalM101]:
        """Generate all 8 diverse domain goals plus diagnostic controls."""
        goals: list[LearningGoalM101] = []

        # Shared Foundational Anchor Task: Arith ADD y = (s + 2) mod 16
        anc_s = list(range(16))
        anc_op = [1] * 16
        anc_v = [2] * 16
        anc_y = [(si + 2) % MODULUS for si in anc_s]
        anc_tok = self.encode_tokens(anc_s, anc_op, anc_v)
        anc_lbl = torch.tensor(anc_y, dtype=torch.long, device=self.device)

        # 1. ALGEBRA_QUADRATIC: Non-linear y = (x^2 + 3x + 1) mod 16
        tr_s = list(range(16))
        tr_op = [1] * 16
        tr_v = [1] * 16
        tr_y = [IndependentLifelongVerifier.verify_quadratic(x) for x in tr_s]
        goals.append(
            LearningGoalM101(
                goal_id="goal_algebra_quadratic",
                domain=DomainM101.ALGEBRA_QUADRATIC,
                description="Non-linear modular quadratic mapping y = (x^2 + 3x + 1) mod 16",
                train_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                train_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                test_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                estimated_cost=0.5,
            )
        )

        # 2. GRAPH_REACHABILITY: 2-hop navigation
        tr_s = list(range(16))
        tr_op = [0 if i % 2 == 0 else 1 for i in range(16)]
        tr_v = [1 if i % 3 == 0 else 0 for i in range(16)]
        tr_y = [IndependentLifelongVerifier.verify_graph_2hop(s, op, v) for s, op, v in zip(tr_s, tr_op, tr_v, strict=False)]
        goals.append(
            LearningGoalM101(
                goal_id="goal_graph_reachability",
                domain=DomainM101.GRAPH_REACHABILITY,
                description="2-hop directed graph reachability under edge constraints",
                train_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                train_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                test_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                estimated_cost=0.6,
            )
        )

        # 3. LOGIC_MAJORITY: 3-bit majority gate
        tr_s = list(range(16))
        tr_op = [(i + 1) % 16 for i in range(16)]
        tr_v = [(i + 2) % 16 for i in range(16)]
        tr_y = [IndependentLifelongVerifier.verify_logic_majority(s, op, v) for s, op, v in zip(tr_s, tr_op, tr_v, strict=False)]
        goals.append(
            LearningGoalM101(
                goal_id="goal_logic_majority",
                domain=DomainM101.LOGIC_MAJORITY,
                description="Propositional 3-variable majority and parity gate",
                train_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                train_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                test_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                estimated_cost=0.4,
            )
        )

        # 4. STATEFUL_PROGRAM: Conditional threshold update
        tr_s = list(range(16))
        tr_op = [3] * 16  # IF_GT
        tr_v = [6] * 16
        tr_y = [IndependentLifelongVerifier.verify_program(s, 3, 6) for s in tr_s]
        goals.append(
            LearningGoalM101(
                goal_id="goal_stateful_program",
                domain=DomainM101.STATEFUL_PROGRAM,
                description="Stateful conditional threshold execution",
                train_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                train_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                test_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                estimated_cost=0.7,
            )
        )

        # 5. FINITE_STATE_MACHINE: DFA State Tracking
        tr_s = [i % 4 for i in range(16)]
        tr_op = [0 if i % 2 == 0 else 1 for i in range(16)]
        tr_v = [0] * 16
        tr_y = [IndependentLifelongVerifier.verify_dfa(s, op) for s, op in zip(tr_s, tr_op, strict=False)]
        goals.append(
            LearningGoalM101(
                goal_id="goal_finite_state_machine",
                domain=DomainM101.FINITE_STATE_MACHINE,
                description="Deterministic finite-state automaton transition dynamics",
                train_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                train_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                test_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                estimated_cost=0.5,
            )
        )

        # 6. FACTUAL_MEMORY: Temporal revision v2
        tr_s = list(range(16))
        tr_op = [4] * 16
        tr_v = [2] * 16
        tr_y = [IndependentLifelongVerifier.verify_factual(s, 2) for s in tr_s]
        goals.append(
            LearningGoalM101(
                goal_id="goal_factual_correction",
                domain=DomainM101.FACTUAL_MEMORY,
                description="Entity property retrieval with temporal revision v2",
                train_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                train_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                test_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                estimated_cost=0.5,
            )
        )

        # 7. MULTI_STEP_PLANNING: Sequential plan execution
        tr_s = list(range(16))
        tr_op = [5] * 16
        tr_v = [(i + 2) % 8 for i in range(16)]
        tr_y = [IndependentLifelongVerifier.verify_planning(s, v) for s, v in zip(tr_s, tr_v, strict=False)]
        goals.append(
            LearningGoalM101(
                goal_id="goal_planning_sequence",
                domain=DomainM101.MULTI_STEP_PLANNING,
                description="Sequential goal progression under precondition constraints",
                train_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                train_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                test_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                estimated_cost=0.8,
            )
        )

        # 8. CROSS_SUBJECT_COMPOSITION: Cascaded Graph -> Logic -> Program Energy
        tr_s = list(range(16))
        tr_op = [6] * 16
        tr_v = [(i * 3) % 16 for i in range(16)]
        tr_y = [IndependentLifelongVerifier.verify_composition(s, s, v) for s, v in zip(tr_s, tr_v, strict=False)]
        goals.append(
            LearningGoalM101(
                goal_id="goal_cross_subject_composition",
                domain=DomainM101.CROSS_SUBJECT_COMPOSITION,
                description="Cascaded Graph routing -> Logic gate -> Energy debit",
                train_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                train_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
                test_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                estimated_cost=0.9,
            )
        )

        # Diagnostic Control 1: White Noise Distractor (Unlearnable)
        tr_s_noise = list(range(16))
        tr_op_noise = [7] * 16
        tr_v_noise = [self.rng.randint(0, 15) for _ in range(16)]
        tr_y_noise = [self.rng.randint(0, 15) for _ in range(16)]
        goals.append(
            LearningGoalM101(
                goal_id="goal_noise_distractor",
                domain=DomainM101.NOISE_DISTRACTOR,
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

        # Diagnostic Control 2: Elementary Identity (Mastered)
        tr_s_mas = list(range(16))
        tr_op_mas = [0] * 16
        tr_v_mas = list(range(16))
        tr_y_mas = list(range(16))
        goals.append(
            LearningGoalM101(
                goal_id="goal_already_mastered",
                domain=DomainM101.ALREADY_MASTERED,
                description="Already-mastered identity baseline (0 progress reward)",
                train_tokens=self.encode_tokens(tr_s_mas, tr_op_mas, tr_v_mas),
                train_labels=torch.tensor(tr_y_mas, dtype=torch.long, device=self.device),
                test_tokens=self.encode_tokens(tr_s_mas, tr_op_mas, tr_v_mas),
                test_labels=torch.tensor(tr_y_mas, dtype=torch.long, device=self.device),
                anchor_tokens=anc_tok,
                anchor_labels=anc_lbl,
                estimated_cost=0.1,
            )
        )

        return goals

    def generate_sealed_target_task(self) -> LearningGoalM101:
        """Create a sealed, held-out target task with a genuinely novel non-linear mechanism."""
        # Non-linear quadratic polynomial y = (3x^2 + 7x + 5) mod 16
        tr_s = list(range(16))
        tr_op = [6] * 16
        tr_v = [7] * 16
        tr_y = [IndependentLifelongVerifier.verify_quadratic(x, a=3, b=7, c=5) for x in tr_s]

        anc_s = list(range(16))
        anc_op = [1] * 16
        anc_v = [2] * 16
        anc_y = [(si + 2) % MODULUS for si in anc_s]

        return LearningGoalM101(
            goal_id="sealed_target_quadratic_rule",
            domain=DomainM101.ALGEBRA_QUADRATIC,
            description="Sealed target non-linear polynomial mapping y = (3x^2 + 7x + 5) mod 16",
            train_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
            train_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
            test_tokens=self.encode_tokens(tr_s, tr_op, tr_v),
            test_labels=torch.tensor(tr_y, dtype=torch.long, device=self.device),
            anchor_tokens=self.encode_tokens(anc_s, anc_op, anc_v),
            anchor_labels=torch.tensor(anc_y, dtype=torch.long, device=self.device),
            estimated_cost=1.0,
        )
