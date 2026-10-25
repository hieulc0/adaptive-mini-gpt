"""Milestone M92 Phase 1: Unknown-Rule Learning Environment.

Generates procedurally unfamiliar task families without providing symbolic equations,
opcode names, or solution code to the student model:
1. Task Families:
   - Quadratic / Polynomial: y = (a * x^2 + b * x + c) % 16
   - Piecewise / Conditional: y = (3x + 1) % 16 if x < 8 else (15 - x) % 16
   - Bitwise Permutation: Circular bit-shift and nibble reflection
   - Disambiguation Minimal Pairs: Rules identical on initial 2-3 observations that diverge on query space
   - Unsupported / Contradictory Tasks: Inconsistent mappings requiring epistemic abstention
2. Strict Information Isolation:
   - Student receives only input-output observation pairs (x_i, y_i).
   - Student never receives equation strings, rule IDs, or oracle source code.
3. Interactive Query Interface:
   - Environment provides an interactive oracle under a bounded query budget (Q <= 3).
   - Student proposes candidate query point x* in {0..15} and receives verified feedback y*.
4. Disjoint Partitions:
   - Initial observations (N0 = 2-3)
   - Dev / validation observations (N_dev = 3)
   - Sealed held-out test observations (N_test = 10-11)
   - Separate held-out rule families for learning-to-learn evaluation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class RuleObservation:
    """A single input-output observation pair."""

    x: int
    y: int


class ProceduralRule:
    """Deterministic mathematical rule mapping inputs in {0..15} to {0..15}."""

    def __init__(
        self,
        rule_id: str,
        family: str,
        eval_fn: Callable[[int], int],
        is_learnable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.rule_id = rule_id
        self.family = family
        self._eval_fn = eval_fn
        self.is_learnable = is_learnable
        self.metadata = metadata or {}

    def apply(self, x: int) -> int:
        """Evaluate rule on integer input x in {0..15}."""
        return self._eval_fn(x % 16) % 16

    def generate_table(self) -> dict[int, int]:
        """Generate full ground-truth lookup table for evaluation."""
        return {x: self.apply(x) for x in range(16)}


class UnknownRuleTask:
    """An unfamiliar task instance with bounded query oracle and isolated splits."""

    def __init__(
        self,
        task_id: str,
        rule: ProceduralRule,
        initial_observations: tuple[RuleObservation, ...],
        dev_observations: tuple[RuleObservation, ...],
        test_observations: tuple[RuleObservation, ...],
        query_budget: int = 3,
    ) -> None:
        self.task_id = task_id
        self.family = rule.family
        self.is_learnable = rule.is_learnable
        self._rule = rule
        self.initial_observations = list(initial_observations)
        self.dev_observations = list(dev_observations)
        self.test_observations = list(test_observations)
        self.query_budget = query_budget
        self._queries_used = 0
        self._queried_history: list[RuleObservation] = []

    @property
    def queries_used(self) -> int:
        return self._queries_used

    @property
    def remaining_budget(self) -> int:
        return max(0, self.query_budget - self._queries_used)

    @property
    def all_observed(self) -> list[RuleObservation]:
        """All observations collected so far (initial + queried)."""
        return self.initial_observations + self._queried_history

    def oracle_query(self, x: int) -> RuleObservation:
        """Query the environment for point x under budget constraint."""
        if self._queries_used >= self.query_budget:
            raise RuntimeError(f"Task {self.task_id}: query budget {self.query_budget} exhausted")
        y = self._rule.apply(x % 16)
        obs = RuleObservation(x=x % 16, y=y)
        self._queries_used += 1
        self._queried_history.append(obs)
        return obs

    def evaluate_predictions(self, preds: dict[int, int]) -> float:
        """Evaluate exact match on held-out test observations."""
        if not self.test_observations:
            return 0.0
        correct = sum(1 for obs in self.test_observations if preds.get(obs.x) == obs.y)
        return (correct / len(self.test_observations)) * 100.0


class UnknownRuleEnvironment:
    """Factory and curriculum generator for unfamiliar procedural tasks."""

    @classmethod
    def create_quadratic_rule(cls, rule_id: str, a: int, b: int, c: int) -> ProceduralRule:
        """Create quadratic modulo rule: y = (a*x^2 + b*x + c) % 16."""
        return ProceduralRule(
            rule_id=rule_id,
            family="quadratic",
            eval_fn=lambda x: (a * (x ** 2) + b * x + c) % 16,
            is_learnable=True,
            metadata={"a": a, "b": b, "c": c},
        )

    @classmethod
    def create_piecewise_rule(cls, rule_id: str, thresh: int, m1: int, c1: int, m2: int, c2: int) -> ProceduralRule:
        """Create threshold piecewise rule."""
        def fn(x: int) -> int:
            if x < thresh:
                return (m1 * x + c1) % 16
            else:
                return (m2 * x + c2) % 16

        return ProceduralRule(
            rule_id=rule_id,
            family="piecewise",
            eval_fn=fn,
            is_learnable=True,
            metadata={"thresh": thresh, "m1": m1, "c1": c1, "m2": m2, "c2": c2},
        )

    @classmethod
    def create_bit_perm_rule(cls, rule_id: str, shift: int, mask: int) -> ProceduralRule:
        """Create bitwise circular rotation and mask rule."""
        def fn(x: int) -> int:
            # 4-bit circular left shift by 'shift' then XOR mask
            val = ((x << shift) | (x >> (4 - shift))) & 0xF
            return val ^ mask

        return ProceduralRule(
            rule_id=rule_id,
            family="bit_perm",
            eval_fn=fn,
            is_learnable=True,
            metadata={"shift": shift, "mask": mask},
        )

    @classmethod
    def create_disambiguation_pair(cls, base_id: str, seed: int = 42) -> tuple[ProceduralRule, ProceduralRule, list[int]]:
        """Create pair of rules that agree on points {0, 1} but diverge elsewhere."""
        def fn_a(x: int) -> int:
            return (3 * x + 2) % 16

        def fn_b(x: int) -> int:
            return (x ** 2 + 2 * x + 2) % 16

        rule_a = ProceduralRule(f"{base_id}_A", "linear", fn_a, True, {"agrees_on": [0, 1]})
        rule_b = ProceduralRule(f"{base_id}_B", "quadratic", fn_b, True, {"agrees_on": [0, 1]})
        return rule_a, rule_b, [0, 1]

    @classmethod
    def create_unsupported_task(cls, task_id: str, seed: int = 999) -> UnknownRuleTask:
        """Create an unsupported / underdetermined task where no stable rule exists."""
        rng = random.Random(seed)
        # Contradictory mappings: input 2 maps to both 5 and 11
        inits = (RuleObservation(x=2, y=5), RuleObservation(x=4, y=8), RuleObservation(x=2, y=11))
        devs = (RuleObservation(x=7, y=3), RuleObservation(x=9, y=12))
        tests = tuple(RuleObservation(x=i, y=rng.randint(0, 15)) for i in range(16))

        dummy_rule = ProceduralRule(task_id, "unsupported", lambda x: rng.randint(0, 15), is_learnable=False)
        return UnknownRuleTask(
            task_id=task_id,
            rule=dummy_rule,
            initial_observations=inits,
            dev_observations=devs,
            test_observations=tests,
            query_budget=3,
        )

    @classmethod
    def generate_task_from_rule(
        cls,
        task_id: str,
        rule: ProceduralRule,
        fixed_inits: list[int] | None = None,
        seed: int = 42,
    ) -> UnknownRuleTask:
        """Construct an UnknownRuleTask with isolated partitions."""
        rng = random.Random(seed)
        all_xs = list(range(16))
        rng.shuffle(all_xs)

        if fixed_inits is not None:
            init_xs = fixed_inits
            rem_xs = [x for x in all_xs if x not in init_xs]
        else:
            init_xs = all_xs[:3]
            rem_xs = all_xs[3:]

        dev_xs = rem_xs[:3]
        test_xs = rem_xs[3:]  # held-out test points

        inits = tuple(RuleObservation(x=x, y=rule.apply(x)) for x in init_xs)
        devs = tuple(RuleObservation(x=x, y=rule.apply(x)) for x in dev_xs)
        tests = tuple(RuleObservation(x=x, y=rule.apply(x)) for x in test_xs)

        return UnknownRuleTask(
            task_id=task_id,
            rule=rule,
            initial_observations=inits,
            dev_observations=devs,
            test_observations=tests,
            query_budget=3,
        )

    @classmethod
    def generate_benchmark_suite(cls, seed: int = 42) -> list[UnknownRuleTask]:
        """Generate standard 9-task benchmark suite across rule families (8 learnable + 1 unsupported)."""
        tasks: list[UnknownRuleTask] = []

        # 1. Quadratic Tasks
        r_q1 = cls.create_quadratic_rule("quad_1", a=1, b=0, c=3)  # y = (x^2 + 3) % 16
        r_q2 = cls.create_quadratic_rule("quad_2", a=1, b=2, c=1)  # y = (x^2 + 2x + 1) % 16
        tasks.append(cls.generate_task_from_rule("task_quad_1", r_q1, seed=seed + 1))
        tasks.append(cls.generate_task_from_rule("task_quad_2", r_q2, seed=seed + 2))

        # 2. Piecewise Tasks
        r_p1 = cls.create_piecewise_rule("piece_1", thresh=8, m1=2, c1=1, m2=15, c2=0)
        r_p2 = cls.create_piecewise_rule("piece_2", thresh=6, m1=3, c1=2, m2=1, c2=7)
        tasks.append(cls.generate_task_from_rule("task_piece_1", r_p1, seed=seed + 11))
        tasks.append(cls.generate_task_from_rule("task_piece_2", r_p2, seed=seed + 12))

        # 3. Bitwise Permutation Tasks
        r_b1 = cls.create_bit_perm_rule("bit_1", shift=1, mask=0x5)
        r_b2 = cls.create_bit_perm_rule("bit_2", shift=2, mask=0xA)
        tasks.append(cls.generate_task_from_rule("task_bit_1", r_b1, seed=seed + 21))
        tasks.append(cls.generate_task_from_rule("task_bit_2", r_b2, seed=seed + 22))

        # 4. Disambiguation Minimal Pair Tasks
        r_da, r_db, agreed_inits = cls.create_disambiguation_pair("disambig_pair", seed=seed)
        tasks.append(cls.generate_task_from_rule("task_disambig_A", r_da, fixed_inits=agreed_inits, seed=seed + 31))
        tasks.append(cls.generate_task_from_rule("task_disambig_B", r_db, fixed_inits=agreed_inits, seed=seed + 32))

        # 5. Unsupported / Ambiguous Control Task
        tasks.append(cls.create_unsupported_task("task_unsupported_ctrl", seed=seed + 99))

        return tasks

    @classmethod
    def generate_held_out_meta_test_suite(cls, seed: int = 142) -> list[UnknownRuleTask]:
        """Generate entirely held-out rule families for Learning-to-Learn test."""
        # Novel family: Cubic Polynomials y = (lead_a * x^3 + a*x + b) % 16
        tasks: list[UnknownRuleTask] = []
        r_c1 = ProceduralRule("cubic_1", "cubic", lambda x: (x ** 3 + 3 * x + 2) % 16, True, {"family": "cubic"})
        r_c2 = ProceduralRule("cubic_2", "cubic", lambda x: (x ** 3 + x + 5) % 16, True, {"family": "cubic"})
        r_c3 = ProceduralRule("cubic_3", "cubic", lambda x: (2 * (x ** 3) + 7) % 16, True, {"family": "cubic"})

        tasks.append(cls.generate_task_from_rule("heldout_cubic_1", r_c1, seed=seed + 101))
        tasks.append(cls.generate_task_from_rule("heldout_cubic_2", r_c2, seed=seed + 102))
        tasks.append(cls.generate_task_from_rule("heldout_cubic_3", r_c3, seed=seed + 103))
        return tasks
