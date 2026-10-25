"""Milestone M90: Instruction-Conditioned Concept Curriculum Generator.

Generates controlled instruction-following tasks across 6 foundational concept families:
1. Modular Addition (OP_ADD): (s + k) % 16
2. Modular Subtraction (OP_SUB): (s - k) % 16
3. Modular Scaling (OP_SCALE): (s * k) % 16
4. Bitwise XOR (OP_XOR): (s ^ k) % 16
5. Conditional Logic (OP_COND): (s ^ k) % 16 if s >= 8 else (s + k) % 16
6. Two-Stage Composition (OP_COMP): OP2(OP1(s, k1), k2)

Provides rigorous splits:
- Training: Seen operations, seen operands k in {2, 3, 5}, canonical wording (48 ex/task)
- Dev / Validation: Seen operations & operands, disjoint sequences (16 ex/task)
- Qualification: Seen operations & operands, disjoint sequences (16 ex/task)
- Sealed Test: Seen operations & operands, disjoint sequences (48 ex/task)
- Held-out Operands: Seen operations, held-out operands k in {4, 7}
- Held-out Paraphrases: Equivalent instruction wordings (INCR, DECR, MULT, FLIP, THRESH)
- Held-out Compositions: Novel combinations of learned operations
- Negative Controls: Shuffled instruction prompts and randomized labels
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from adaptive_mini_gpt.concept_instruction.instruction_grammar import (
    InstructionGrammar,
    InstructionPrompt,
    Opcode,
    ParaphraseOpcode,
    SYNONYM_MAP,
)
from adaptive_mini_gpt.model import MiniGPT


@dataclass(frozen=True)
class InstructionTask:
    """Task dataset with strictly disjoint partitions."""

    task_id: str
    family: str
    opcode: Opcode | ParaphraseOpcode | int
    operand: int | tuple[int, int]
    train_prompts: list[InstructionPrompt]
    dev_prompts: list[InstructionPrompt]
    qual_prompts: list[InstructionPrompt]
    test_prompts: list[InstructionPrompt]


@dataclass(frozen=True)
class StageBSuite:
    """Comprehensive evaluation suite for Stage B instruction transfer."""

    train_tasks: list[InstructionTask]
    held_out_operand_tasks: list[InstructionTask]
    held_out_paraphrase_tasks: list[InstructionTask]
    held_out_composition_tasks: list[InstructionTask]
    negative_shuffled_prompts: list[InstructionPrompt]
    negative_random_prompts: list[InstructionPrompt]


class M90CurriculumGenerator:
    """Curriculum and evaluation suite generator for M90."""

    def __init__(self, vocab_size: int = 16, seq_len: int = 4) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def _compute_atomic_target(
        self,
        opcode: Opcode | ParaphraseOpcode | int,
        operand: int,
        seq: Sequence[int],
    ) -> tuple[int, ...]:
        """Compute ground-truth target sequence for atomic instruction."""
        canonical_op = opcode
        if isinstance(opcode, ParaphraseOpcode) or int(opcode) in SYNONYM_MAP:
            canonical_op = SYNONYM_MAP.get(ParaphraseOpcode(int(opcode)), Opcode.ADD)

        canonical_op = int(canonical_op)
        v = self.vocab_size
        k = operand % v

        if canonical_op == int(Opcode.ADD):
            return tuple((s + k) % v for s in seq)
        elif canonical_op == int(Opcode.SUB):
            return tuple((s - k) % v for s in seq)
        elif canonical_op == int(Opcode.SCALE):
            return tuple((s * k) % v for s in seq)
        elif canonical_op == int(Opcode.XOR):
            return tuple((s ^ k) % v for s in seq)
        elif canonical_op == int(Opcode.COND):
            return tuple((s ^ k) % v if s >= 8 else (s + k) % v for s in seq)
        else:
            return tuple((s + k) % v for s in seq)

    def _compute_composition_target(
        self,
        op1: Opcode | int,
        opr1: int,
        op2: Opcode | int,
        opr2: int,
        seq: Sequence[int],
    ) -> tuple[int, ...]:
        """Compute ground-truth target sequence for two-stage composition."""
        stage1 = self._compute_atomic_target(op1, opr1, seq)
        stage2 = self._compute_atomic_target(op2, opr2, stage1)
        return stage2

    def generate_task(
        self,
        task_id: str,
        opcode: Opcode | ParaphraseOpcode | int,
        operand: int,
        seed: int = 42,
    ) -> InstructionTask:
        """Generate a single atomic instruction task with disjoint partitions."""
        rng = random.Random(seed)
        L = self.seq_len
        target_count = 128  # 48 train + 16 dev + 16 qual + 48 test

        all_seqs: set[tuple[int, ...]] = set()
        attempts = 0
        while len(all_seqs) < target_count and attempts < 10_000:
            attempts += 1
            seq = tuple(rng.randint(0, self.vocab_size - 1) for _ in range(L))
            all_seqs.add(seq)

        seq_list = sorted(all_seqs)
        rng.shuffle(seq_list)

        train_s = seq_list[:48]
        dev_s = seq_list[48:64]
        qual_s = seq_list[64:80]
        test_s = seq_list[80:128]

        family = (
            opcode.name.lower()
            if hasattr(opcode, "name")
            else f"op_{int(opcode)}"
        )

        def make_prompts(seqs: list[tuple[int, ...]]) -> list[InstructionPrompt]:
            return [
                InstructionGrammar.format_atomic(
                    opcode=opcode,
                    operand=operand,
                    symbols=s,
                    target_symbols=self._compute_atomic_target(opcode, operand, s),
                    op_family=family,
                )
                for s in seqs
            ]

        return InstructionTask(
            task_id=task_id,
            family=family,
            opcode=opcode,
            operand=operand,
            train_prompts=make_prompts(train_s),
            dev_prompts=make_prompts(dev_s),
            qual_prompts=make_prompts(qual_s),
            test_prompts=make_prompts(test_s),
        )

    def generate_composition_task(
        self,
        task_id: str,
        op1: Opcode | int,
        opr1: int,
        op2: Opcode | int,
        opr2: int,
        seed: int = 42,
    ) -> InstructionTask:
        """Generate a two-stage composition task with disjoint partitions."""
        rng = random.Random(seed)
        L = self.seq_len
        target_count = 128

        all_seqs: set[tuple[int, ...]] = set()
        attempts = 0
        while len(all_seqs) < target_count and attempts < 10_000:
            attempts += 1
            seq = tuple(rng.randint(0, self.vocab_size - 1) for _ in range(L))
            all_seqs.add(seq)

        seq_list = sorted(all_seqs)
        rng.shuffle(seq_list)

        train_s = seq_list[:48]
        dev_s = seq_list[48:64]
        qual_s = seq_list[64:80]
        test_s = seq_list[80:128]

        def make_prompts(seqs: list[tuple[int, ...]]) -> list[InstructionPrompt]:
            return [
                InstructionGrammar.format_composition(
                    op1=op1,
                    opr1=opr1,
                    op2=op2,
                    opr2=opr2,
                    symbols=s,
                    target_symbols=self._compute_composition_target(op1, opr1, op2, opr2, s),
                    op_family="composition",
                )
                for s in seqs
            ]

        return InstructionTask(
            task_id=task_id,
            family="composition",
            opcode=Opcode.COMP,
            operand=(opr1, opr2),
            train_prompts=make_prompts(train_s),
            dev_prompts=make_prompts(dev_s),
            qual_prompts=make_prompts(qual_s),
            test_prompts=make_prompts(test_s),
        )

    def generate_stage_b_suite(self, seed: int = 42) -> StageBSuite:
        """Construct complete Stage B evaluation suite."""
        train_ops = [
            (Opcode.ADD, "add"),
            (Opcode.SUB, "sub"),
            (Opcode.SCALE, "scale"),
            (Opcode.XOR, "xor"),
            (Opcode.COND, "cond"),
        ]
        seen_operands = [2, 3, 5]
        held_out_operands = [4, 7]

        # 1. Train tasks: Seen operations with seen operands
        train_tasks: list[InstructionTask] = []
        for op, name in train_ops:
            for k in seen_operands:
                t = self.generate_task(
                    task_id=f"train_{name}_k{k}",
                    opcode=op,
                    operand=k,
                    seed=seed + int(op) * 10 + k,
                )
                train_tasks.append(t)

        # 2. Held-out Operands: Seen operations with unseen operands k in {4, 7}
        held_out_operand_tasks: list[InstructionTask] = []
        for op, name in train_ops:
            for k in held_out_operands:
                t = self.generate_task(
                    task_id=f"heldout_operand_{name}_k{k}",
                    opcode=op,
                    operand=k,
                    seed=seed + 1000 + int(op) * 10 + k,
                )
                held_out_operand_tasks.append(t)

        # 3. Held-out Paraphrases: Equivalent wording on seen operands
        paraphrase_map = [
            (ParaphraseOpcode.INCR, "incr"),
            (ParaphraseOpcode.DECR, "decr"),
            (ParaphraseOpcode.MULT, "mult"),
            (ParaphraseOpcode.FLIP, "flip"),
            (ParaphraseOpcode.THRESH, "thresh"),
        ]
        held_out_paraphrase_tasks: list[InstructionTask] = []
        for pop, name in paraphrase_map:
            for k in seen_operands[:2]:  # k in {2, 3}
                t = self.generate_task(
                    task_id=f"paraphrase_{name}_k{k}",
                    opcode=pop,
                    operand=k,
                    seed=seed + 2000 + int(pop) * 10 + k,
                )
                held_out_paraphrase_tasks.append(t)

        # 4. Held-out Compositions: Novel two-stage chained combinations
        comp_pairs = [
            (Opcode.ADD, 3, Opcode.XOR, 5),
            (Opcode.SCALE, 3, Opcode.SUB, 2),
            (Opcode.XOR, 6, Opcode.ADD, 5),
        ]
        held_out_comp_tasks: list[InstructionTask] = []
        for i, (op1, k1, op2, k2) in enumerate(comp_pairs):
            t = self.generate_composition_task(
                task_id=f"comp_{i}",
                op1=op1,
                opr1=k1,
                op2=op2,
                opr2=k2,
                seed=seed + 3000 + i * 100,
            )
            held_out_comp_tasks.append(t)

        # 5. Negative controls: Shuffled instructions & random labels
        # Pick 32 test prompts from train tasks and corrupt them
        all_test_prompts = [p for t in train_tasks for p in t.test_prompts[:3]]
        rng = random.Random(seed + 999)

        shuffled_prompts: list[InstructionPrompt] = []
        for p in all_test_prompts[:32]:
            # Mismatched opcode: change ADD to XOR, SCALE to SUB, etc.
            current_op = p.token_ids[0]
            mismatched_op = int(Opcode.XOR) if current_op == int(Opcode.ADD) else int(Opcode.ADD)
            new_tokens = list(p.token_ids)
            new_tokens[0] = mismatched_op
            shuffled_prompts.append(
                InstructionPrompt(
                    token_ids=tuple(new_tokens),
                    symbol_start_idx=p.symbol_start_idx,
                    symbol_end_idx=p.symbol_end_idx,
                    target_tokens=p.target_tokens,  # ground truth unchanged -> mismatch!
                    is_composition=p.is_composition,
                    is_paraphrase=False,
                    op_family="negative_shuffled",
                    operand=p.operand,
                )
            )

        random_prompts: list[InstructionPrompt] = []
        for p in all_test_prompts[:32]:
            # Randomize target tokens
            rand_targets = tuple(rng.randint(0, self.vocab_size - 1) for _ in range(self.seq_len))
            random_prompts.append(
                InstructionPrompt(
                    token_ids=p.token_ids,
                    symbol_start_idx=p.symbol_start_idx,
                    symbol_end_idx=p.symbol_end_idx,
                    target_tokens=rand_targets,
                    is_composition=p.is_composition,
                    is_paraphrase=False,
                    op_family="negative_random",
                    operand=p.operand,
                )
            )

        return StageBSuite(
            train_tasks=train_tasks,
            held_out_operand_tasks=held_out_operand_tasks,
            held_out_paraphrase_tasks=held_out_paraphrase_tasks,
            held_out_composition_tasks=held_out_comp_tasks,
            negative_shuffled_prompts=shuffled_prompts,
            negative_random_prompts=random_prompts,
        )


def extract_prompt_features(
    model: MiniGPT,
    prompts: Sequence[InstructionPrompt],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pass instruction prompts through frozen MiniGPT backbone and extract symbol hidden states.

    Returns:
        h_symbols: Tensor of shape (len(prompts), seq_len, d_model)
        targets: Tensor of shape (len(prompts), seq_len)
    """
    if not prompts:
        return torch.empty(0, 4, 128, device=device), torch.empty(0, 4, dtype=torch.long, device=device)

    # Group by prompt length for batched processing
    len_groups: dict[int, list[tuple[int, InstructionPrompt]]] = {}
    for idx, p in enumerate(prompts):
        L = len(p.token_ids)
        len_groups.setdefault(L, []).append((idx, p))

    all_h = [None] * len(prompts)
    all_y = [None] * len(prompts)

    model.eval()
    with torch.no_grad():
        for L, group in len_groups.items():
            indices = [g[0] for g in group]
            group_prompts = [g[1] for g in group]

            input_tensor = torch.tensor(
                [p.token_ids for p in group_prompts],
                dtype=torch.long,
                device=device,
            )

            tok_emb = model.wte(input_tensor)
            if model.wpe is not None:
                pos = torch.arange(0, input_tensor.size(1), dtype=torch.long, device=device)
                h = model.drop(tok_emb + model.wpe(pos))
            else:
                h = model.drop(tok_emb)

            for block in model.blocks:
                h, _ = block(h)
            h = model.ln_f(h)

            for i, p in enumerate(group_prompts):
                global_idx = indices[i]
                h_sym = h[i, p.symbol_start_idx : p.symbol_end_idx, :]
                all_h[global_idx] = h_sym
                all_y[global_idx] = torch.tensor(p.target_tokens, dtype=torch.long, device=device)

    return torch.stack(all_h), torch.stack(all_y)
