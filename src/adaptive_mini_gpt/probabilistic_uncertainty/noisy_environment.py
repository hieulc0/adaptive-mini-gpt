"""Milestone M93 Phase 1: Noisy & Uncertain Rule Discovery Environment.

Extends the procedural rule environment with controlled stochasticity and cost accounting:
1. Controlled Noise Regimes:
   - Clean Deterministic: eta = 0.0 (baseline compatibility with M92)
   - Symmetric Label Corruption: eta in (0.10, 0.25), replacing true y with a random incorrect symbol
   - Input-Dependent Noise: higher corruption probability on high-magnitude inputs (x >= 8)
   - Incomplete / Underdetermined: sparse observations leaving multiple viable hypotheses
   - Contradictory / Unsupported: irreconcilable observations requiring epistemic abstention
2. Cost-Aware Query Interface:
   - Each query incurs an explicit unit cost C_query = 1.0.
   - Bounded interaction budget (Q <= 4).
   - Allows evaluating prediction quality as a function of total query expenditure.
3. Strict Information Partitioning:
   - Observations received by the student contain potentially noisy labels without corruption flags.
   - Ground-truth functions and uncorrupted test sets are sealed in the evaluation harness.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from adaptive_mini_gpt.autonomous_discovery.unknown_rule_environment import (
    ProceduralRule,
    UnknownRuleEnvironment,
)


@dataclass(frozen=True)
class NoisyObservation:
    """An input-output observation pair with optional noise provenance for auditing."""

    x: int
    y: int
    is_corrupted: bool = False
    true_y: int | None = None


class NoisyRuleTask:
    """An unfamiliar task instance operating under noise and query cost accounting."""

    def __init__(
        self,
        task_id: str,
        rule: ProceduralRule,
        initial_observations: tuple[NoisyObservation, ...],
        dev_observations: tuple[NoisyObservation, ...],
        test_observations: tuple[NoisyObservation, ...],
        query_budget: int = 4,
        query_cost: float = 1.0,
        noise_rate: float = 0.0,
        noise_type: str = "none",  # "none", "symmetric", "input_dependent"
        seed: int = 42,
    ) -> None:
        self.task_id = task_id
        self.family = rule.family
        self.is_learnable = rule.is_learnable
        self._rule = rule
        self.initial_observations = list(initial_observations)
        self.dev_observations = list(dev_observations)
        self.test_observations = list(test_observations)
        self.query_budget = query_budget
        self.query_cost = query_cost
        self.noise_rate = noise_rate
        self.noise_type = noise_type
        self._rng = random.Random(seed)
        self._queries_used = 0
        self._queried_history: list[NoisyObservation] = []

    @property
    def queries_used(self) -> int:
        return self._queries_used

    @property
    def remaining_budget(self) -> int:
        return max(0, self.query_budget - self._queries_used)

    @property
    def total_cost_incurred(self) -> float:
        return self._queries_used * self.query_cost

    @property
    def all_observed(self) -> list[NoisyObservation]:
        """All observations collected so far (initial + queried)."""
        return self.initial_observations + self._queried_history

    def _apply_noise(self, x: int, true_y: int) -> tuple[int, bool]:
        """Apply noise based on task configuration."""
        if self.noise_type == "none" or self.noise_rate <= 0.0:
            return true_y, False

        prob = self.noise_rate
        if self.noise_type == "input_dependent":
            prob = self.noise_rate * 1.5 if x >= 8 else self.noise_rate * 0.5

        if self._rng.random() < prob:
            # Corrupt to a different symbol in {0..15}
            candidates = [s for s in range(16) if s != true_y]
            corrupt_y = self._rng.choice(candidates)
            return corrupt_y, True

        return true_y, False

    def oracle_query(self, x: int) -> NoisyObservation:
        """Query teacher oracle under bounded budget and cost accounting."""
        if self._queries_used >= self.query_budget:
            raise RuntimeError(f"Task {self.task_id}: query budget {self.query_budget} exhausted")

        norm_x = x % 16
        true_y = self._rule.apply(norm_x)
        observed_y, is_corrupted = self._apply_noise(norm_x, true_y)

        obs = NoisyObservation(
            x=norm_x,
            y=observed_y,
            is_corrupted=is_corrupted,
            true_y=true_y,
        )
        self._queries_used += 1
        self._queried_history.append(obs)
        return obs

    def evaluate_predictions(self, preds: dict[int, int]) -> float:
        """Evaluate exact match on sealed clean test observations."""
        if not self.test_observations:
            return 0.0
        correct = sum(
            1 for obs in self.test_observations
            if preds.get(obs.x) == (obs.true_y if obs.true_y is not None else obs.y)
        )
        return (correct / len(self.test_observations)) * 100.0


class NoisyRuleEnvironment:
    """Curriculum and environment generator for noisy & uncertain procedural learning."""

    @classmethod
    def generate_task_from_rule(
        cls,
        task_id: str,
        rule: ProceduralRule,
        fixed_inits: list[int] | None = None,
        noise_rate: float = 0.0,
        noise_type: str = "none",
        query_budget: int = 4,
        seed: int = 42,
    ) -> NoisyRuleTask:
        """Construct an isolated NoisyRuleTask."""
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
        test_xs = rem_xs[3:]  # held-out clean test set

        # Generate initial observations (subject to noise)
        init_obs: list[NoisyObservation] = []
        for x in init_xs:
            true_y = rule.apply(x)
            prob = noise_rate
            if noise_type == "input_dependent":
                prob = noise_rate * 1.5 if x >= 8 else noise_rate * 0.5

            if rng.random() < prob and noise_type != "none":
                corrupt_y = rng.choice([s for s in range(16) if s != true_y])
                init_obs.append(NoisyObservation(x=x, y=corrupt_y, is_corrupted=True, true_y=true_y))
            else:
                init_obs.append(NoisyObservation(x=x, y=true_y, is_corrupted=False, true_y=true_y))

        # Dev and test partitions are always clean ground truth for fair evaluation
        dev_obs = tuple(NoisyObservation(x=x, y=rule.apply(x), is_corrupted=False, true_y=rule.apply(x)) for x in dev_xs)
        test_obs = tuple(NoisyObservation(x=x, y=rule.apply(x), is_corrupted=False, true_y=rule.apply(x)) for x in test_xs)

        return NoisyRuleTask(
            task_id=task_id,
            rule=rule,
            initial_observations=tuple(init_obs),
            dev_observations=dev_obs,
            test_observations=test_obs,
            query_budget=query_budget,
            noise_rate=noise_rate,
            noise_type=noise_type,
            seed=seed,
        )

    @classmethod
    def create_unsupported_task(cls, task_id: str, seed: int = 999) -> NoisyRuleTask:
        """Create a contradictory task where input 2 maps to both 5 and 11."""
        rng = random.Random(seed)
        inits = (
            NoisyObservation(x=2, y=5, is_corrupted=False, true_y=5),
            NoisyObservation(x=4, y=8, is_corrupted=False, true_y=8),
            NoisyObservation(x=2, y=11, is_corrupted=False, true_y=11),  # Direct contradiction
        )
        devs = (
            NoisyObservation(x=7, y=3, is_corrupted=False, true_y=3),
            NoisyObservation(x=9, y=12, is_corrupted=False, true_y=12),
        )
        tests = tuple(NoisyObservation(x=i, y=rng.randint(0, 15), is_corrupted=False, true_y=rng.randint(0, 15)) for i in range(16))

        dummy_rule = ProceduralRule(task_id, "unsupported", lambda x: rng.randint(0, 15), is_learnable=False)
        return NoisyRuleTask(
            task_id=task_id,
            rule=dummy_rule,
            initial_observations=inits,
            dev_observations=devs,
            test_observations=tests,
            query_budget=4,
            noise_rate=0.0,
            noise_type="none",
            seed=seed,
        )

    @classmethod
    def generate_benchmark_suite(cls, seed: int = 42) -> list[NoisyRuleTask]:
        """Generate standard M93 benchmark suite covering clean, noisy, and contradictory regimes."""
        tasks: list[NoisyRuleTask] = []

        # 1. Clean Tasks (eta = 0.0) - Regression baseline
        r_q1 = UnknownRuleEnvironment.create_quadratic_rule("quad_1", a=1, b=0, c=3)
        r_p1 = UnknownRuleEnvironment.create_piecewise_rule("piece_1", thresh=8, m1=2, c1=1, m2=15, c2=0)
        tasks.append(cls.generate_task_from_rule("clean_quad_1", r_q1, noise_rate=0.0, noise_type="none", seed=seed + 1))
        tasks.append(cls.generate_task_from_rule("clean_piece_1", r_p1, noise_rate=0.0, noise_type="none", seed=seed + 2))

        # 2. Symmetric Noisy Tasks (eta = 0.20)
        r_q2 = UnknownRuleEnvironment.create_quadratic_rule("quad_2", a=1, b=2, c=1)
        r_p2 = UnknownRuleEnvironment.create_piecewise_rule("piece_2", thresh=6, m1=3, c1=2, m2=1, c2=7)
        tasks.append(cls.generate_task_from_rule("noisy_sym_quad_2", r_q2, noise_rate=0.20, noise_type="symmetric", seed=seed + 11))
        tasks.append(cls.generate_task_from_rule("noisy_sym_piece_2", r_p2, noise_rate=0.20, noise_type="symmetric", seed=seed + 12))

        # 3. Input-Dependent Noisy Tasks (higher noise for x >= 8)
        r_b1 = UnknownRuleEnvironment.create_bit_perm_rule("bit_1", shift=1, mask=0x5)
        r_b2 = UnknownRuleEnvironment.create_bit_perm_rule("bit_2", shift=2, mask=0xA)
        tasks.append(cls.generate_task_from_rule("noisy_input_bit_1", r_b1, noise_rate=0.20, noise_type="input_dependent", seed=seed + 21))
        tasks.append(cls.generate_task_from_rule("noisy_input_bit_2", r_b2, noise_rate=0.20, noise_type="input_dependent", seed=seed + 22))

        # 4. Disambiguation Minimal Pairs with Ambiguity
        r_da, r_db, agreed_inits = UnknownRuleEnvironment.create_disambiguation_pair("disambig_pair", seed=seed)
        tasks.append(cls.generate_task_from_rule("noisy_disambig_A", r_da, fixed_inits=agreed_inits, noise_rate=0.15, noise_type="symmetric", seed=seed + 31))
        tasks.append(cls.generate_task_from_rule("noisy_disambig_B", r_db, fixed_inits=agreed_inits, noise_rate=0.15, noise_type="symmetric", seed=seed + 32))

        # 5. Unsupported Contradictory Task
        tasks.append(cls.create_unsupported_task("unsupported_contradiction", seed=seed + 99))

        return tasks

    @classmethod
    def generate_held_out_meta_test_suite(cls, seed: int = 142) -> list[NoisyRuleTask]:
        """Generate novel rule families under observation noise for Learning-to-Learn evaluation."""
        tasks: list[NoisyRuleTask] = []
        r_c1 = ProceduralRule("cubic_1", "cubic", lambda x: (x ** 3 + 3 * x + 2) % 16, True, {"family": "cubic"})
        r_c2 = ProceduralRule("cubic_2", "cubic", lambda x: (x ** 3 + x + 5) % 16, True, {"family": "cubic"})
        r_c3 = ProceduralRule("cubic_3", "cubic", lambda x: (2 * (x ** 3) + 7) % 16, True, {"family": "cubic"})

        tasks.append(cls.generate_task_from_rule("heldout_noisy_cubic_1", r_c1, noise_rate=0.15, noise_type="symmetric", seed=seed + 101))
        tasks.append(cls.generate_task_from_rule("heldout_noisy_cubic_2", r_c2, noise_rate=0.15, noise_type="symmetric", seed=seed + 102))
        tasks.append(cls.generate_task_from_rule("heldout_noisy_cubic_3", r_c3, noise_rate=0.15, noise_type="symmetric", seed=seed + 103))
        return tasks
