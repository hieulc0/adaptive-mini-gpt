"""Milestone M91: Multi-Step Reasoning Curriculum Generator.

Generates controlled reasoning problems across depths and task families:
1. Chain Depths:
   - 1-Step: Foundational atomic operation
   - 2-Step: Two-stage operation chaining
   - 3-Step: Three-stage intermediate carrying
   - 4-Step & 5-Step: Long-horizon reasoning chains
   - Conditional: Branching logic based on intermediate results
2. Generalization Splits:
   - Seen operands: k in {2, 3, 5}
   - Held-out operands: k in {4, 7}
   - Held-out paraphrases: INCR, DECR, MULT, FLIP, THRESH
3. Strictly disjoint partitions for train (48), dev (16), qual (16), and test (48).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from adaptive_mini_gpt.concept_instruction.instruction_grammar import (
    Opcode,
    ParaphraseOpcode,
)
from adaptive_mini_gpt.neural_reasoning.reasoning_grammar import (
    MultiStepEvaluator,
    MultiStepProblem,
)


@dataclass(frozen=True)
class ReasoningPartition:
    """Disjoint problem sets for one reasoning task family."""

    family_name: str
    train_problems: list[MultiStepProblem]
    dev_problems: list[MultiStepProblem]
    qual_problems: list[MultiStepProblem]
    test_problems: list[MultiStepProblem]


@dataclass(frozen=True)
class M91ReasoningSuite:
    """Full empirical suite for Milestone M91 multi-step evaluation."""

    one_step_suite: ReasoningPartition
    two_step_suite: ReasoningPartition
    three_step_suite: ReasoningPartition
    four_step_suite: ReasoningPartition
    five_step_suite: ReasoningPartition
    conditional_suite: ReasoningPartition
    held_out_operand_suite: ReasoningPartition
    held_out_paraphrase_suite: ReasoningPartition


class M91ReasoningCurriculumGenerator:
    """Generator for multi-step reasoning curricula."""

    def __init__(self, vocab_size: int = 16) -> None:
        self.vocab_size = vocab_size

    def _generate_partition(
        self,
        family_name: str,
        chain_length: int,
        op_pool: Sequence[Opcode | ParaphraseOpcode | int],
        operand_pool: Sequence[int],
        is_conditional: bool = False,
        is_paraphrase: bool = False,
        seed: int = 42,
    ) -> ReasoningPartition:
        """Construct four strictly disjoint partitions of multi-step problems."""
        rng = random.Random(seed)
        target_count = 128  # 48 train + 16 dev + 16 qual + 48 test

        problems: list[MultiStepProblem] = []
        seen_specs: set[tuple] = set()

        attempts = 0
        while len(problems) < target_count and attempts < 20_000:
            attempts += 1
            v0 = rng.randint(0, self.vocab_size - 1)
            ops: list[tuple[Opcode | ParaphraseOpcode | int, int]] = []
            for _ in range(chain_length):
                op = rng.choice(op_pool)
                k = rng.choice(operand_pool)
                ops.append((op, k))

            spec_key = (v0, tuple(ops))
            if spec_key in seen_specs:
                continue
            seen_specs.add(spec_key)

            steps, ans = MultiStepEvaluator.evaluate_chain(
                initial_value=v0,
                operations=ops,
                vocab_size=self.vocab_size,
            )

            p_id = f"{family_name}_{len(problems)}"
            desc = f"Start with {v0}. " + ", ".join(f"Op {int(op)} by {k}" for op, k in ops)

            problems.append(
                MultiStepProblem(
                    problem_id=p_id,
                    chain_length=chain_length,
                    initial_value=v0,
                    operations=tuple(ops),
                    intermediate_steps=steps,
                    final_answer=ans,
                    is_conditional=is_conditional,
                    is_paraphrase=is_paraphrase,
                    description=desc,
                )
            )

        rng.shuffle(problems)
        train_p = problems[:48]
        dev_p = problems[48:64]
        qual_p = problems[64:80]
        test_p = problems[80:128]

        return ReasoningPartition(
            family_name=family_name,
            train_problems=train_p,
            dev_problems=dev_p,
            qual_problems=qual_p,
            test_problems=test_p,
        )

    def generate_suite(self, seed: int = 42) -> M91ReasoningSuite:
        """Generate comprehensive M91 evaluation suite across all chain lengths."""
        standard_ops = [Opcode.ADD, Opcode.SUB, Opcode.SCALE, Opcode.XOR]
        seen_operands = [2, 3, 5]
        held_out_operands = [4, 7]

        paraphrase_ops = [
            ParaphraseOpcode.INCR,
            ParaphraseOpcode.DECR,
            ParaphraseOpcode.MULT,
            ParaphraseOpcode.FLIP,
        ]

        p1 = self._generate_partition("one_step", 1, standard_ops, seen_operands, seed=seed + 1)
        p2 = self._generate_partition("two_step", 2, standard_ops, seen_operands, seed=seed + 2)
        p3 = self._generate_partition("three_step", 3, standard_ops, seen_operands, seed=seed + 3)
        p4 = self._generate_partition("four_step", 4, standard_ops, seen_operands, seed=seed + 4)
        p5 = self._generate_partition("five_step", 5, standard_ops, seen_operands, seed=seed + 5)
        p_cond = self._generate_partition("conditional", 3, [Opcode.COND, Opcode.ADD, Opcode.XOR], seen_operands, is_conditional=True, seed=seed + 6)
        p_ho_op = self._generate_partition("heldout_operand", 3, standard_ops, held_out_operands, seed=seed + 7)
        p_ho_para = self._generate_partition("heldout_paraphrase", 3, paraphrase_ops, seen_operands, is_paraphrase=True, seed=seed + 8)

        return M91ReasoningSuite(
            one_step_suite=p1,
            two_step_suite=p2,
            three_step_suite=p3,
            four_step_suite=p4,
            five_step_suite=p5,
            conditional_suite=p_cond,
            held_out_operand_suite=p_ho_op,
            held_out_paraphrase_suite=p_ho_para,
        )
