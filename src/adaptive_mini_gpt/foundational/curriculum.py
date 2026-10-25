"""Milestone M88 Phase 2: Foundational Multi-Domain Curriculum Generator.

Constructs controlled learning environments with shared underlying mathematical, logical,
and procedural structures across six foundational domains:
1. Numbers & Place-Value: Multi-digit representation, units/tens extraction, magnitude comparison.
2. Arithmetic: Multi-digit addition, subtraction, carry handling, borrowing.
3. Boolean & Conditional Logic: Bitwise AND, OR, NOT, XOR, implication, conditional branching.
4. Sequence Manipulation: Reversal, cyclic permutation, pairwise sorting, mirror symmetry.
5. Compositional Procedures: Cascaded combinations of arithmetic, logic, and sequence operations.
6. Instruction Semantics: Token-conditioned operation dispatch ('ADD', 'SUB', 'AND', 'REV', etc.).

Generates strictly isolated splits:
- train (48): adaptation examples
- dev (16): validation & probe observations
- qualification (16): qualification gate (EM >= 0.85)
- sealed_test (48): final generalization evaluation on unseen operands
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from adaptive_mini_gpt.meta_acquisition.environment import (
    MetaTaskDataset,
    SplitType,
    TaskExample,
    TaskFamily,
)


class FoundationalDomain(StrEnum):
    """Categorical domain classification for foundational skills."""

    PLACE_VALUE_REASONING = "PLACE_VALUE_REASONING"
    ARITHMETIC_OPERATIONS = "ARITHMETIC_OPERATIONS"
    BOOLEAN_CONDITIONAL_LOGIC = "BOOLEAN_CONDITIONAL_LOGIC"
    SEQUENCE_MANIPULATION = "SEQUENCE_MANIPULATION"
    COMPOSITIONAL_PROCEDURE = "COMPOSITIONAL_PROCEDURE"
    INSTRUCTION_SEMANTICS = "INSTRUCTION_SEMANTICS"


def apply_foundational_rule(
    domain: FoundationalDomain,
    rule_name: str,
    params: dict[str, Any],
    seq: tuple[int, ...],
    vocab_size: int = 16,
) -> tuple[int, ...]:
    """Execute deterministic foundational rule on integer token sequence."""
    L = len(seq)

    # 1. PLACE-VALUE & NUMBERS
    if domain == FoundationalDomain.PLACE_VALUE_REASONING:
        op = params.get("operation", "magnitude_compare")
        if op == "magnitude_compare":
            # Compare first half vs second half as base-vocab_size digits
            mid = max(1, L // 2)
            val1 = sum(s * (vocab_size ** (mid - 1 - i)) for i, s in enumerate(seq[:mid]))
            val2 = sum(s * (vocab_size ** (L - mid - 1 - i)) for i, s in enumerate(seq[mid:]))
            cmp_val = 1 if val1 > val2 else (2 if val1 < val2 else 0)
            return tuple((s + cmp_val) % vocab_size for s in seq)
        elif op == "place_extraction":
            # Extract units place vs tens place offset
            offset = params.get("offset", 3)
            return tuple(((s // 2) + offset) % vocab_size for s in seq)
        else:
            mod_val = params.get("mod", 5)
            return tuple((s % mod_val) for s in seq)

    # 2. ARITHMETIC OPERATIONS
    elif domain == FoundationalDomain.ARITHMETIC_OPERATIONS:
        op = params.get("operation", "add_with_carry")
        if op == "add_with_carry":
            addend = params.get("addend", 3)
            carry_in = params.get("carry", 1)
            res = []
            c = carry_in
            for s in reversed(seq):
                total = s + addend + c
                res.append(total % vocab_size)
                c = total // vocab_size
            return tuple(reversed(res))
        elif op == "sub_with_borrow":
            subtrahend = params.get("subtrahend", 2)
            borrow_in = params.get("borrow", 0)
            res = []
            b = borrow_in
            for s in reversed(seq):
                diff = s - subtrahend - b
                if diff < 0:
                    diff += vocab_size
                    b = 1
                else:
                    b = 0
                res.append(diff % vocab_size)
            return tuple(reversed(res))
        else:
            mult = params.get("mult", 3)
            bias = params.get("bias", 1)
            return tuple((s * mult + bias) % vocab_size for s in seq)

    # 3. BOOLEAN & CONDITIONAL LOGIC
    elif domain == FoundationalDomain.BOOLEAN_CONDITIONAL_LOGIC:
        op = params.get("operation", "bitwise_xor_cond")
        mask = params.get("mask", 7)
        if op == "bitwise_xor_cond":
            cond_threshold = params.get("threshold", 8)
            return tuple(
                (s ^ mask) % vocab_size if s >= cond_threshold else (s ^ (mask >> 1)) % vocab_size
                for s in seq
            )
        elif op == "boolean_implication":
            # Implication logic: (A -> B) encoded as ((~A) | B)
            target_mask = params.get("target_mask", 5)
            return tuple((((~s) & 15) | target_mask) % vocab_size for s in seq)
        else:
            # Conditional parity branch
            offset_odd = params.get("offset_odd", 3)
            offset_even = params.get("offset_even", 5)
            return tuple(
                (s + offset_odd) % vocab_size if (s % 2 == 1) else (s + offset_even) % vocab_size
                for s in seq
            )

    # 4. SEQUENCE MANIPULATION
    elif domain == FoundationalDomain.SEQUENCE_MANIPULATION:
        op = params.get("operation", "cyclic_shift")
        shift = params.get("shift", 1)
        if op == "cyclic_shift":
            shifted = [seq[(i - shift) % L] for i in range(L)]
            return tuple(shifted)
        elif op == "sequence_reversal":
            offset = params.get("offset", 1)
            return tuple((s + offset) % vocab_size for s in reversed(seq))
        else:
            # Pairwise sorting / swap
            seq_list = list(seq)
            for i in range(0, L - 1, 2):
                if seq_list[i] > seq_list[i + 1]:
                    seq_list[i], seq_list[i + 1] = seq_list[i + 1], seq_list[i]
            return tuple(seq_list)

    # 5. COMPOSITIONAL PROCEDURES
    elif domain == FoundationalDomain.COMPOSITIONAL_PROCEDURE:
        # Two-stage composition: arithmetic step then logic/permutation step
        add_k = params.get("stage1_add", 4)
        xor_k = params.get("stage2_xor", 6)
        inter = [(s + add_k) % vocab_size for s in seq]
        return tuple((s ^ xor_k) % vocab_size for s in inter)

    # 6. INSTRUCTION SEMANTICS
    elif domain == FoundationalDomain.INSTRUCTION_SEMANTICS:
        # First token is the opcode instruction: 1=ADD, 2=SUB, 3=REV, 4=XOR
        opcode = params.get("opcode", 1)
        operand = params.get("operand", 3)
        res = [opcode]
        for s in seq[1:]:
            if opcode == 1:
                res.append((s + operand) % vocab_size)
            elif opcode == 2:
                res.append((s - operand + vocab_size) % vocab_size)
            elif opcode == 3:
                res.append((s ^ operand) % vocab_size)
            else:
                res.append((s * 2 + 1) % vocab_size)
        return tuple(res)

    # Default fallback: affine map
    return tuple((s * 3 + 1) % vocab_size for s in seq)


class FoundationalCurriculumGenerator:
    """Generates strictly partitioned foundational task datasets with disjoint inputs."""

    def __init__(self, vocab_size: int = 16, seq_len: int = 4) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self._universe: list[tuple[int, ...]] = list(
            itertools.product(range(vocab_size), repeat=seq_len)
        )

    def generate_task(
        self,
        task_id: str,
        domain: FoundationalDomain,
        rule_name: str,
        rule_params: dict[str, Any],
        split: SplitType = SplitType.META_TRAIN,
        seed: int = 42,
        train_size: int = 48,
        dev_size: int = 16,
        qualification_size: int = 16,
        sealed_test_size: int = 48,
    ) -> MetaTaskDataset:
        """Deterministically generate dataset with strictly disjoint partitions."""
        rng = random.Random(seed)
        total_needed = train_size + dev_size + qualification_size + sealed_test_size

        # Sample disjoint tuples without replacement
        sampled_tuples = rng.sample(self._universe, total_needed)

        train_tuples = sampled_tuples[:train_size]
        dev_tuples = sampled_tuples[train_size : train_size + dev_size]
        qual_tuples = sampled_tuples[
            train_size + dev_size : train_size + dev_size + qualification_size
        ]
        test_tuples = sampled_tuples[train_size + dev_size + qualification_size :]

        # Verify strict disjointness
        assert len(set(train_tuples).intersection(set(dev_tuples))) == 0
        assert len(set(train_tuples).intersection(set(qual_tuples))) == 0
        assert len(set(train_tuples).intersection(set(test_tuples))) == 0
        assert len(set(qual_tuples).intersection(set(test_tuples))) == 0

        # Helper to construct examples
        def _make_examples(tuples: list[tuple[int, ...]], prefix: str) -> list[TaskExample]:
            ex_list = []
            for i, in_tup in enumerate(tuples):
                out_tup = apply_foundational_rule(
                    domain=domain,
                    rule_name=rule_name,
                    params=rule_params,
                    seq=in_tup,
                    vocab_size=self.vocab_size,
                )
                ex_list.append(
                    TaskExample(
                        example_id=f"{task_id}_{prefix}_{i}",
                        family=TaskFamily.ARITHMETIC_AFFINE,  # compatibility type
                        rule_instance_id=rule_name,
                        input_tokens=in_tup,
                        target_tokens=out_tup,
                        input_text=" ".join(str(x) for x in in_tup),
                        target_text=" ".join(str(x) for x in out_tup),
                    )
                )
            return ex_list

        return MetaTaskDataset(
            task_id=task_id,
            family=TaskFamily.ARITHMETIC_AFFINE,
            split=split,
            rule_instance_id=rule_name,
            rule_params=rule_params,
            seed=seed,
            train_examples=_make_examples(train_tuples, "tr"),
            dev_examples=_make_examples(dev_tuples, "dev"),
            qualification_examples=_make_examples(qual_tuples, "ql"),
            sealed_test_examples=_make_examples(test_tuples, "test"),
        )
