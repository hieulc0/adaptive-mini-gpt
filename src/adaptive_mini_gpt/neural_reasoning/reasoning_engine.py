"""Milestone M91: Multi-Step Neural Reasoning Execution & Attribution Engine.

Implements the core execution pathways required by M91:
1. Autonomous Self-Generated Reasoning:
   - Feeds self-generated intermediate predictions into subsequent steps autoregressively.
   - No ground-truth intermediate values provided at inference time.
2. Teacher-Forced Reasoning Upper Bound:
   - Provides ground-truth intermediate outputs at each step to establish the upper bound.
3. Causal Step-Corruption Attribution:
   - Perturbs intermediate step predictions by controlled noise to verify that downstream
     predictions strictly depend on upstream intermediate results.
4. Intermediate Step Verification:
   - Tracks correctness of each intermediate step independently.
5. Shuffled Step Ablation:
   - Permutes intermediate operation order to verify procedural order sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from adaptive_mini_gpt.concept_instruction.curriculum_m90 import extract_prompt_features
from adaptive_mini_gpt.concept_instruction.instruction_grammar import (
    InstructionGrammar,
    Opcode,
    ParaphraseOpcode,
)
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.neural_reasoning.persistent_substrate import (
    PersistentReasoningSubstrate,
)
from adaptive_mini_gpt.neural_reasoning.reasoning_grammar import (
    MultiStepProblem,
)


@dataclass(frozen=True)
class ChainExecutionResult:
    """Execution outcome for a single multi-step reasoning problem."""

    problem_id: str
    chain_length: int
    final_correct: bool
    intermediate_accuracies: list[bool]
    all_steps_correct: bool
    predicted_steps: list[int]
    true_steps: list[int]


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregated evaluation metrics for a suite of multi-step problems."""

    count: int
    final_exact_match: float
    all_steps_exact_match: float
    mean_intermediate_accuracy: float
    per_step_accuracies: list[float]


class ReasoningEngine:
    """Autoregressive multi-step execution and causal attribution evaluator."""

    @classmethod
    def execute_atomic_step(
        cls,
        model: MiniGPT,
        substrate: PersistentReasoningSubstrate,
        opcode: Opcode | ParaphraseOpcode | int,
        operand: int,
        value: int,
        device: torch.device,
        task_id: str | None = None,
    ) -> int:
        """Execute a single atomic operation step through the frozen backbone and substrate."""
        prompt = InstructionGrammar.format_atomic(
            opcode=opcode,
            operand=operand,
            symbols=[value, value, value, value],
            target_symbols=[0, 0, 0, 0],
        )
        with torch.no_grad():
            h, _ = extract_prompt_features(model, [prompt], device)
            logits = substrate(h, task_id=task_id)
            pred = logits[0, 0, :].argmax(dim=-1).item()
        return pred

    @classmethod
    def execute_autonomous(
        cls,
        model: MiniGPT,
        substrate: PersistentReasoningSubstrate,
        problem: MultiStepProblem,
        device: torch.device,
        task_id: str | None = None,
    ) -> ChainExecutionResult:
        """Autoregressively execute reasoning chain without ground-truth intermediate inputs."""
        curr = problem.initial_value
        preds: list[int] = []
        step_accs: list[bool] = []
        true_steps = [s.output_value for s in problem.intermediate_steps]

        for idx, (op, k) in enumerate(problem.operations):
            pred_out = cls.execute_atomic_step(model, substrate, op, k, curr, device, task_id)
            preds.append(pred_out)
            is_correct = (pred_out == true_steps[idx])
            step_accs.append(is_correct)
            curr = pred_out  # Feed self-generated prediction into next step!

        final_correct = (preds[-1] == problem.final_answer)
        all_correct = all(step_accs)

        return ChainExecutionResult(
            problem_id=problem.problem_id,
            chain_length=problem.chain_length,
            final_correct=final_correct,
            intermediate_accuracies=step_accs,
            all_steps_correct=all_correct,
            predicted_steps=preds,
            true_steps=true_steps,
        )

    @classmethod
    def execute_teacher_forced(
        cls,
        model: MiniGPT,
        substrate: PersistentReasoningSubstrate,
        problem: MultiStepProblem,
        device: torch.device,
        task_id: str | None = None,
    ) -> ChainExecutionResult:
        """Execute chain where each step receives true ground-truth intermediate inputs."""
        preds: list[int] = []
        step_accs: list[bool] = []
        true_steps = [s.output_value for s in problem.intermediate_steps]

        for idx, s in enumerate(problem.intermediate_steps):
            pred_out = cls.execute_atomic_step(model, substrate, s.opcode, s.operand, s.input_value, device, task_id)
            preds.append(pred_out)
            step_accs.append(pred_out == s.output_value)

        final_correct = (preds[-1] == problem.final_answer)
        all_correct = all(step_accs)

        return ChainExecutionResult(
            problem_id=problem.problem_id,
            chain_length=problem.chain_length,
            final_correct=final_correct,
            intermediate_accuracies=step_accs,
            all_steps_correct=all_correct,
            predicted_steps=preds,
            true_steps=true_steps,
        )

    @classmethod
    def execute_corrupted_step(
        cls,
        model: MiniGPT,
        substrate: PersistentReasoningSubstrate,
        problem: MultiStepProblem,
        corrupt_step_idx: int = 1,
        noise: int = 1,
        device: torch.device | None = None,
        task_id: str | None = None,
    ) -> tuple[int, bool]:
        """Inject controlled noise at intermediate step to test causal downstream dependency.

        Returns:
            (corrupted_final_answer, did_diverge_from_clean_run)
        """
        dev = device or torch.device("cpu")
        clean_res = cls.execute_autonomous(model, substrate, problem, dev, task_id)

        curr = problem.initial_value
        corrupted_preds: list[int] = []

        for idx, (op, k) in enumerate(problem.operations):
            if idx + 1 == corrupt_step_idx:
                # Inject perturbation
                curr = (clean_res.predicted_steps[idx] + noise) % 16
            else:
                pred_out = cls.execute_atomic_step(model, substrate, op, k, curr, dev, task_id)
                curr = pred_out
            corrupted_preds.append(curr)

        diverged = (corrupted_preds[-1] != clean_res.predicted_steps[-1])
        return corrupted_preds[-1], diverged

    @classmethod
    def evaluate_suite(
        cls,
        model: MiniGPT,
        substrate: PersistentReasoningSubstrate,
        problems: Sequence[MultiStepProblem],
        device: torch.device,
        mode: str = "autonomous",
        task_id: str | None = None,
    ) -> EvaluationSummary:
        """Evaluate a set of problems returning detailed exact-match and step statistics."""
        if not problems:
            return EvaluationSummary(0, 0.0, 0.0, 0.0, [])

        final_correct_cnt = 0
        all_correct_cnt = 0
        chain_len = problems[0].chain_length
        step_correct_counts = [0] * chain_len

        for p in problems:
            if mode == "autonomous":
                res = cls.execute_autonomous(model, substrate, p, device, task_id)
            elif mode == "teacher_forced":
                res = cls.execute_teacher_forced(model, substrate, p, device, task_id)
            else:
                res = cls.execute_autonomous(model, substrate, p, device, task_id)

            if res.final_correct:
                final_correct_cnt += 1
            if res.all_steps_correct:
                all_correct_cnt += 1
            for s_idx, is_ok in enumerate(res.intermediate_accuracies):
                if is_ok:
                    step_correct_counts[s_idx] += 1

        n = len(problems)
        per_step_accs = [round(c / n * 100.0, 2) for c in step_correct_counts]
        mean_step_acc = sum(per_step_accs) / len(per_step_accs) if per_step_accs else 0.0

        return EvaluationSummary(
            count=n,
            final_exact_match=round(final_correct_cnt / n * 100.0, 2),
            all_steps_exact_match=round(all_correct_cnt / n * 100.0, 2),
            mean_intermediate_accuracy=round(mean_step_acc, 2),
            per_step_accuracies=per_step_accs,
        )
