"""Unfamiliar Cross-Subject Curriculum Generator for Milestone M98.

Generates structured tasks across three distinct subject areas and their combinations:
1. Algebra & Numerical Relationships (affine transformations, modular equations)
2. Graphs & Navigation (directed graph traversal, path validity, reachability)
3. Logic & Deduction (Boolean constraints, conditional inference, rule satisfaction)
4. Cross-Subject Compositions (e.g. Graph navigation with Algebraic energy costs and Logic safety guards)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any

MODULUS = 16


class Subject(str, Enum):
    """Subject areas evaluated in Milestone M98."""

    ALGEBRA = "algebra"
    GRAPH = "graph"
    LOGIC = "logic"
    CROSS_SUBJECT_COMPOSITION = "cross_subject_composition"
    UNSUPPORTED_ABSTAIN = "unsupported_abstain"


@dataclass(frozen=True)
class CrossSubjectTask:
    """An individual problem instance in the cross-subject curriculum."""

    episode_id: int
    subject: Subject
    input_text: str
    target_output: Any
    has_supported_concept: bool
    is_contradictory: bool
    intermediate_steps: list[tuple[int, int, int]] | None
    graph_adj: dict[int, list[tuple[int, int]]] | None  # u -> list of (v, weight)
    logic_rules: list[str] | None
    algebra_params: tuple[int, int] | None  # (a, b) in a*x + b
    token_length: int


class IndependentGroundTruthSolver:
    """Deterministic reference solver for supervision and grading ONLY (strictly excluded from student inference)."""

    @classmethod
    def solve_algebra(cls, x: int, a: int, b: int) -> int:
        """y = (a * x + b) % MODULUS."""
        return (a * x + b) % MODULUS

    @classmethod
    def solve_graph_step(cls, current_node: int, next_node: int, adj: dict[int, list[tuple[int, int]]]) -> int:
        """Find weight of edge (current_node, next_node) or MODULUS - 1 if invalid."""
        for v, w in adj.get(current_node, []):
            if v == next_node:
                return w % MODULUS
        return (MODULUS - 1)

    @classmethod
    def solve_logic(cls, p: bool, q: bool, rule_type: str) -> int:
        """Evaluate Boolean logic condition."""
        if rule_type == "AND":
            return 1 if (p and q) else 0
        elif rule_type == "OR":
            return 1 if (p or q) else 0
        elif rule_type == "IMPLIES":
            return 1 if (not p or q) else 0
        elif rule_type == "XOR":
            return 1 if (p != q) else 0
        return 0

    @classmethod
    def solve_composition(
        cls,
        path: list[int],
        adj: dict[int, list[tuple[int, int]]],
        cost_scale: int,
        cost_bias: int,
        battery_threshold: int,
        initial_battery: int,
    ) -> int:
        """Graph traversal with algebraic energy consumption and logic battery threshold.
        
        Returns total energy consumed if path is valid and battery >= threshold at each step;
        else returns -1 (blocked/invalid).
        """
        battery = initial_battery
        total_energy = 0

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            # 1. Graph edge weight
            edge_weight = None
            for dest, w in adj.get(u, []):
                if dest == v:
                    edge_weight = w
                    break
            if edge_weight is None:
                return -1  # Invalid edge

            # 2. Algebraic energy formula: cost = (cost_scale * edge_weight + cost_bias) % MODULUS
            step_energy = (cost_scale * edge_weight + cost_bias) % MODULUS
            total_energy = (total_energy + step_energy) % MODULUS
            battery = (battery - step_energy) % MODULUS

            # 3. Logic safety constraint: battery >= battery_threshold
            if battery < battery_threshold:
                return -1  # Safety violation

        return total_energy


class CrossSubjectCurriculumGenerator:
    """Generates rigorous sequential multi-subject curricula for M98."""

    def __init__(self, seed: int = 42, num_episodes: int = 60) -> None:
        self.seed = seed
        self.num_episodes = num_episodes
        self.rng = random.Random(seed)

    def generate_curriculum(self) -> list[CrossSubjectTask]:
        """Construct balanced curriculum across Algebra, Graph, Logic, and Compositions."""
        tasks: list[CrossSubjectTask] = []

        # Graph templates
        fixed_graph: dict[int, list[tuple[int, int]]] = {
            0: [(1, 2), (2, 5)],
            1: [(3, 1), (4, 4)],
            2: [(4, 2), (5, 6)],
            3: [(6, 3)],
            4: [(6, 1), (7, 5)],
            5: [(7, 2)],
            6: [(8, 4)],
            7: [(8, 2)],
            8: [],
        }

        for ep in range(1, self.num_episodes + 1):
            # Partition episodes:
            # 1-15: Algebra
            # 16-30: Graph Navigation
            # 31-45: Logic Deduction
            # 46-55: Unsupported Abstention & Contradictions
            # 56-60: Cross-Subject Compositions (Target evaluation)

            if ep <= 15:
                # 1. Algebra: y = (a * x + b) % MODULUS
                a = (ep % 4) * 2 + 1  # Coprime to 16: 1, 3, 5, 7
                b = (ep * 3) % MODULUS
                x = (ep * 5) % MODULUS
                y = IndependentGroundTruthSolver.solve_algebra(x, a, b)
                input_text = f"algebra_eval linear_map x={x} scale={a} bias={b} modulus=16"

                tasks.append(
                    CrossSubjectTask(
                        episode_id=ep,
                        subject=Subject.ALGEBRA,
                        input_text=input_text,
                        target_output=y,
                        has_supported_concept=True,
                        is_contradictory=False,
                        intermediate_steps=[(x, a, b)],
                        graph_adj=None,
                        logic_rules=None,
                        algebra_params=(a, b),
                        token_length=len(input_text.split()),
                    )
                )

            elif ep <= 30:
                # 2. Graph: edge transition u -> v weight
                u = (ep - 16) % 8
                candidates = fixed_graph.get(u, [(0, 1)])
                chosen_edge = candidates[(ep) % len(candidates)]
                v = chosen_edge[0]
                expected_weight = chosen_edge[1]
                input_text = f"graph_traverse source_node={u} target_node={v} query=edge_weight"

                tasks.append(
                    CrossSubjectTask(
                        episode_id=ep,
                        subject=Subject.GRAPH,
                        input_text=input_text,
                        target_output=expected_weight,
                        has_supported_concept=True,
                        is_contradictory=False,
                        intermediate_steps=[(u, v, expected_weight)],
                        graph_adj=fixed_graph,
                        logic_rules=None,
                        algebra_params=None,
                        token_length=len(input_text.split()),
                    )
                )

            elif ep <= 45:
                # 3. Logic: Boolean conditions and threshold rules
                p_val = bool(ep % 2 == 0)
                q_val = bool((ep // 2) % 2 == 1)
                rule_types = ["AND", "OR", "IMPLIES", "XOR"]
                rule_type = rule_types[ep % len(rule_types)]
                res = IndependentGroundTruthSolver.solve_logic(p_val, q_val, rule_type)
                input_text = f"logic_deduce predicate_p={int(p_val)} predicate_q={int(q_val)} rule={rule_type}"

                tasks.append(
                    CrossSubjectTask(
                        episode_id=ep,
                        subject=Subject.LOGIC,
                        input_text=input_text,
                        target_output=res,
                        has_supported_concept=True,
                        is_contradictory=False,
                        intermediate_steps=[(int(p_val), int(q_val), res)],
                        graph_adj=None,
                        logic_rules=[rule_type],
                        algebra_params=None,
                        token_length=len(input_text.split()),
                    )
                )

            elif ep <= 55:
                # 4. Epistemic Abstention on Unsupported Concepts & Contradictions
                if ep % 2 == 0:
                    input_text = f"unknown_subject_task quantum_hamiltonian_spin state={ep} dimension=128"
                    is_contra = False
                    has_concept = False
                else:
                    input_text = f"algebra_contradiction assert 2+2=5 and 2+2=4 under modulus=16"
                    is_contra = True
                    has_concept = True

                tasks.append(
                    CrossSubjectTask(
                        episode_id=ep,
                        subject=Subject.UNSUPPORTED_ABSTAIN,
                        input_text=input_text,
                        target_output="ABSTAIN",
                        has_supported_concept=has_concept,
                        is_contradictory=is_contra,
                        intermediate_steps=None,
                        graph_adj=None,
                        logic_rules=None,
                        algebra_params=None,
                        token_length=len(input_text.split()),
                    )
                )

            else:
                # 5. Episodes 56-60: Cross-Subject Compositions (Graph + Algebra + Logic)
                # Robot navigation along path 0 -> 1 -> 4 -> 6 with algebraic energy scaling and logic battery threshold
                path = [0, 1, 4, 6]
                scale = 2
                bias = 1
                threshold = 3
                initial_battery = 15

                total_energy = IndependentGroundTruthSolver.solve_composition(
                    path=path,
                    adj=fixed_graph,
                    cost_scale=scale,
                    cost_bias=bias,
                    battery_threshold=threshold,
                    initial_battery=initial_battery,
                )
                input_text = (
                    f"robot_path_planning path=0_1_4_6 energy_scale={scale} energy_bias={bias} "
                    f"battery_threshold={threshold} initial_battery={initial_battery}"
                )

                tasks.append(
                    CrossSubjectTask(
                        episode_id=ep,
                        subject=Subject.CROSS_SUBJECT_COMPOSITION,
                        input_text=input_text,
                        target_output=total_energy,
                        has_supported_concept=True,
                        is_contradictory=False,
                        intermediate_steps=[(0, 1, 4), (1, 4, 6)],
                        graph_adj=fixed_graph,
                        logic_rules=["battery_guard"],
                        algebra_params=(scale, bias),
                        token_length=len(input_text.split()),
                    )
                )

        return tasks
