"""Milestone M90: Controlled Instruction Grammar & Vocabulary.

Defines a tokenized controlled instruction language for foundational concept learning:
1. Core Operations:
   - OP_ADD: Modular addition (s + k mod 16)
   - OP_SUB: Modular subtraction (s - k mod 16)
   - OP_SCALE: Modular scaling (s * k mod 16)
   - OP_XOR: Bitwise XOR (s ^ k mod 16)
   - OP_COND: Conditional logic (s ^ k if s >= 8 else (s + k) mod 16)
   - OP_COMP: Composition marker for two-stage chained operations
2. Controlled Paraphrases / Synonyms:
   - OP_INCR -> synonym for OP_ADD
   - OP_DECR -> synonym for OP_SUB
   - OP_MULT -> synonym for OP_SCALE
   - OP_FLIP -> synonym for OP_XOR
   - OP_THRESH -> synonym for OP_COND
3. Control and Delimiter Tokens:
   - ARG_MARKER: Delimits argument / operand values
   - SEP_MARKER: Separates instruction context from sequence payload
   - THEN_MARKER: Sequences chained operations in composition
4. Sequence Formatters:
   - Formats atomic instructions into causal token sequences
   - Formats compositional instructions into multi-clause token sequences
   - Strictly respects causal self-attention visibility
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence


class Opcode(IntEnum):
    """Canonical operation codes."""

    ADD = 200
    SUB = 201
    SCALE = 202
    XOR = 203
    COND = 204
    COMP = 205


class ParaphraseOpcode(IntEnum):
    """Equivalent instruction paraphrases."""

    INCR = 210  # Paraphrase for ADD
    DECR = 211  # Paraphrase for SUB
    MULT = 212  # Paraphrase for SCALE
    FLIP = 213  # Paraphrase for XOR
    THRESH = 214  # Paraphrase for COND


class ControlToken(IntEnum):
    """Control and separator tokens."""

    ARG = 220
    SEP = 221
    THEN = 222


# Base offsets for operands and sequence symbols
OPR_BASE: int = 230  # Operands 0..15 -> 230..245
SYM_BASE: int = 260  # Input symbols 0..15 -> 260..275


SYNONYM_MAP: dict[ParaphraseOpcode, Opcode] = {
    ParaphraseOpcode.INCR: Opcode.ADD,
    ParaphraseOpcode.DECR: Opcode.SUB,
    ParaphraseOpcode.MULT: Opcode.SCALE,
    ParaphraseOpcode.FLIP: Opcode.XOR,
    ParaphraseOpcode.THRESH: Opcode.COND,
}


@dataclass(frozen=True)
class InstructionPrompt:
    """Formatted instruction prompt with payload metadata."""

    token_ids: tuple[int, ...]
    symbol_start_idx: int
    symbol_end_idx: int
    target_tokens: tuple[int, ...]
    is_composition: bool
    is_paraphrase: bool
    op_family: str
    operand: int | tuple[int, int]


class InstructionGrammar:
    """Formatter and parser for tokenized instruction sequences."""

    @staticmethod
    def format_atomic(
        opcode: Opcode | ParaphraseOpcode | int,
        operand: int,
        symbols: Sequence[int],
        target_symbols: Sequence[int],
        op_family: str = "atomic",
    ) -> InstructionPrompt:
        """Format an atomic instruction sequence: [OP, ARG, OPR_k, SEP, s0, s1, s2, s3]."""
        is_paraphrase = isinstance(opcode, ParaphraseOpcode) or int(opcode) in SYNONYM_MAP
        op_token = int(opcode)
        opr_token = OPR_BASE + (operand % 16)
        payload_tokens = [SYM_BASE + (s % 16) for s in symbols]

        tokens = [
            op_token,
            int(ControlToken.ARG),
            opr_token,
            int(ControlToken.SEP),
        ] + payload_tokens

        start_idx = 4
        end_idx = start_idx + len(symbols)

        return InstructionPrompt(
            token_ids=tuple(tokens),
            symbol_start_idx=start_idx,
            symbol_end_idx=end_idx,
            target_tokens=tuple(target_symbols),
            is_composition=False,
            is_paraphrase=is_paraphrase,
            op_family=op_family,
            operand=operand,
        )

    @staticmethod
    def format_composition(
        op1: Opcode | int,
        opr1: int,
        op2: Opcode | int,
        opr2: int,
        symbols: Sequence[int],
        target_symbols: Sequence[int],
        op_family: str = "composition",
    ) -> InstructionPrompt:
        """Format a two-stage composition sequence: [COMP, OP1, ARG, OPR1, THEN, OP2, ARG, OPR2, SEP, s0..s3]."""
        tokens = [
            int(Opcode.COMP),
            int(op1),
            int(ControlToken.ARG),
            OPR_BASE + (opr1 % 16),
            int(ControlToken.THEN),
            int(op2),
            int(ControlToken.ARG),
            OPR_BASE + (opr2 % 16),
            int(ControlToken.SEP),
        ] + [SYM_BASE + (s % 16) for s in symbols]

        start_idx = 9
        end_idx = start_idx + len(symbols)

        return InstructionPrompt(
            token_ids=tuple(tokens),
            symbol_start_idx=start_idx,
            symbol_end_idx=end_idx,
            target_tokens=tuple(target_symbols),
            is_composition=True,
            is_paraphrase=False,
            op_family=op_family,
            operand=(opr1, opr2),
        )
