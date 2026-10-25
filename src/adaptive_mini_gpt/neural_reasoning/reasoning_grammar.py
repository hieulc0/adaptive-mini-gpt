"""Milestone M91: Multi-Step Reasoning Grammar and Trace Representation.

Defines the structure for multi-step reasoning problems across depths 1 through 5:
1. Multi-Step Problem Specification:
   - initial_value: integer v0 in [0, 15]
   - operations: sequence of (opcode, operand) pairs
   - intermediate_values: ground-truth intermediate evaluation at each step
   - final_answer: final scalar outcome
2. Evaluator Functions:
   - Ground truth step-by-step evaluator (for verification and teacher forcing)
   - Conditional rule evaluation (rules that branch based on intermediate values)
3. Step Trace Data Structure:
   - StepRecord: holds (step_index, opcode, operand, input_val, output_val)
   - MultiStepProblem: complete specification with intermediate targets and prompt metadata
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from adaptive_mini_gpt.concept_instruction.instruction_grammar import (
    InstructionGrammar,
    InstructionPrompt,
    Opcode,
    ParaphraseOpcode,
    SYNONYM_MAP,
)


@dataclass(frozen=True)
class StepRecord:
    """Record of a single intermediate reasoning step."""

    step_index: int
    opcode: Opcode | ParaphraseOpcode | int
    operand: int
    input_value: int
    output_value: int


@dataclass(frozen=True)
class MultiStepProblem:
    """Complete multi-step reasoning problem with ground-truth intermediate traces."""

    problem_id: str
    chain_length: int
    initial_value: int
    operations: tuple[tuple[Opcode | ParaphraseOpcode | int, int], ...]
    intermediate_steps: tuple[StepRecord, ...]
    final_answer: int
    is_conditional: bool
    is_paraphrase: bool
    description: str


class MultiStepEvaluator:
    """Deterministic step-by-step mathematical evaluator for ground-truth and verification."""

    @staticmethod
    def evaluate_atomic_step(
        opcode: Opcode | ParaphraseOpcode | int,
        operand: int,
        val: int,
        vocab_size: int = 16,
    ) -> int:
        """Compute the result of one operation on a single value."""
        canonical_op = opcode
        if isinstance(opcode, ParaphraseOpcode) or int(opcode) in SYNONYM_MAP:
            canonical_op = SYNONYM_MAP.get(ParaphraseOpcode(int(opcode)), Opcode.ADD)
        canonical_op = int(canonical_op)

        v = val % vocab_size
        k = operand % vocab_size

        if canonical_op == int(Opcode.ADD):
            return (v + k) % vocab_size
        elif canonical_op == int(Opcode.SUB):
            return (v - k) % vocab_size
        elif canonical_op == int(Opcode.SCALE):
            return (v * k) % vocab_size
        elif canonical_op == int(Opcode.XOR):
            return (v ^ k) % vocab_size
        elif canonical_op == int(Opcode.COND):
            # Conditional: if intermediate value >= 8, XOR k, else ADD k
            return (v ^ k) % vocab_size if v >= 8 else (v + k) % vocab_size
        else:
            return (v + k) % vocab_size

    evaluate_step = evaluate_atomic_step

    @classmethod
    def evaluate_chain(
        cls,
        initial_value: int,
        operations: Sequence[tuple[Opcode | ParaphraseOpcode | int, int]],
        vocab_size: int = 16,
    ) -> tuple[tuple[StepRecord, ...], int]:
        """Compute full sequence of intermediate step records and final answer."""
        steps: list[StepRecord] = []
        curr = initial_value % vocab_size

        for idx, (op, k) in enumerate(operations):
            out = cls.evaluate_atomic_step(op, k, curr, vocab_size=vocab_size)
            steps.append(
                StepRecord(
                    step_index=idx + 1,
                    opcode=op,
                    operand=k,
                    input_value=curr,
                    output_value=out,
                )
            )
            curr = out

        return tuple(steps), curr
