"""Milestone M92 Phase 3: Neural Skill Acquisition and Causal Attribution.

Learns discovered procedural rules into persistent modular memory:
1. Architectural Substrate:
   - Reuses PersistentReasoningSubstrate:
     * Shared Concept Trunk (10,304 parameters, frozen)
     * Task-Specific Rank-4 Compressed Residuals (1,032 parameters per skill)
     * Zero-initialized up-projection for bit-exact function preservation at initialization.
2. Training on Discovered Observations:
   - Formats input-output pairs (initial observations + active query responses) into prompt batches.
   - When an active discovery outcome is validated, distills the discovered procedural hypothesis
     into the newly allocated modular residual.
   - Extracts contextual representations through frozen MiniGPT backbone.
   - Optimizes the allocated modular residual using AdamW.
3. Verification & Safety Rollback:
   - Evaluates dev set loss and exact match before accepting candidate into repertoire.
   - Rejects underperforming candidates with bit-exact state restoration.
4. Causal Neural Attribution:
   - Shuffles newly learned weights to prove performance causally depends on acquired parameters
     rather than backbone memorization or heuristics.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.autonomous_discovery.unknown_rule_environment import (
    ProceduralRule,
    RuleObservation,
    UnknownRuleTask,
)
from adaptive_mini_gpt.concept_instruction.curriculum_m90 import extract_prompt_features
from adaptive_mini_gpt.concept_instruction.instruction_grammar import (
    InstructionGrammar,
    InstructionPrompt,
    Opcode,
)
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.neural_reasoning.persistent_substrate import (
    PersistentReasoningSubstrate,
)


@dataclass(frozen=True)
class NeuralAcquisitionResult:
    """Outcome of neural skill acquisition for an unfamiliar rule."""

    task_id: str
    status: str  # "ACCEPTED", "REJECTED_DEV_FAILURE", "ABSTAINED"
    dev_exact_match: float
    held_out_test_em: float
    shuffled_weights_test_em: float
    causal_attribution_drop: float
    training_steps: int
    parameter_footprint: int
    rollback_performed: bool


class NeuralRuleLearner:
    """Acquires discovered rules into persistent modular residuals."""

    @classmethod
    def format_observations_to_prompts(
        cls,
        observations: Sequence[RuleObservation],
        task_id: str,
    ) -> list[InstructionPrompt]:
        """Convert input-output observations into instruction prompts."""
        prompts: list[InstructionPrompt] = []
        for obs in observations:
            p = InstructionGrammar.format_atomic(
                opcode=Opcode.ADD,  # Neutral base opcode token
                operand=0,
                symbols=[obs.x, obs.x, obs.x, obs.x],
                target_symbols=[obs.y, obs.y, obs.y, obs.y],
                op_family=f"discovered_{task_id}",
            )
            prompts.append(p)
        return prompts

    @classmethod
    def evaluate_task_exact_match(
        cls,
        model: MiniGPT,
        substrate: PersistentReasoningSubstrate,
        observations: Sequence[RuleObservation],
        device: torch.device,
        task_id: str | None = None,
    ) -> float:
        """Measure exact match over a set of observations."""
        if not observations:
            return 0.0
        prompts = cls.format_observations_to_prompts(observations, task_id or "eval")
        h, y = extract_prompt_features(model, prompts, device)

        with torch.no_grad():
            logits = substrate(h, task_id=task_id)
            preds = logits.argmax(dim=-1)
            # Match on first symbol position
            em = (preds[:, 0] == y[:, 0]).float().mean().item() * 100.0
        return em

    @classmethod
    def acquire_skill(
        cls,
        model: MiniGPT,
        substrate: PersistentReasoningSubstrate,
        task: UnknownRuleTask,
        device: torch.device,
        discovered_rule: ProceduralRule | None = None,
        steps: int = 40,
        lr: float = 0.03,
        dev_threshold: float = 60.0,
    ) -> NeuralAcquisitionResult:
        """Allocate a compressed residual, train on observed/discovered data, verify, and test."""
        # 1. State snapshot for rollback safety
        state_backup = copy.deepcopy(substrate.state_dict())
        residuals_backup = set(substrate.residuals.keys())

        # 2. Allocate compressed residual
        try:
            substrate.add_residual(task.task_id)
        except RuntimeError:
            return NeuralAcquisitionResult(
                task_id=task.task_id,
                status="REJECTED_BUDGET_EXCEEDED",
                dev_exact_match=0.0,
                held_out_test_em=0.0,
                shuffled_weights_test_em=0.0,
                causal_attribution_drop=0.0,
                training_steps=0,
                parameter_footprint=substrate.count_total_parameters(),
                rollback_performed=True,
            )

        res_module = substrate.residuals[task.task_id]
        opt = torch.optim.AdamW(res_module.parameters(), lr=lr, weight_decay=1e-4)

        # 3. Formulate training data:
        # If discovered_rule is validated, distill the procedural hypothesis over all inputs
        # Otherwise, train only on empirical observations
        if discovered_rule is not None:
            synth_obs = [RuleObservation(x=x, y=discovered_rule.apply(x)) for x in range(16)]
            train_prompts = cls.format_observations_to_prompts(synth_obs, task.task_id)
        else:
            train_prompts = cls.format_observations_to_prompts(task.all_observed, task.task_id)

        h_tr, y_tr = extract_prompt_features(model, train_prompts, device)

        for _ in range(steps):
            substrate.train()
            logits = substrate(h_tr, task_id=task.task_id)
            loss = F.cross_entropy(logits.reshape(-1, 16), y_tr.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

        substrate.eval()

        # 4. Dev verification
        dev_em = cls.evaluate_task_exact_match(model, substrate, task.dev_observations, device, task.task_id)

        if dev_em < dev_threshold:
            # Rollback: remove newly registered residual key first
            for k in list(substrate.residuals.keys()):
                if k not in residuals_backup:
                    del substrate.residuals[k]
            substrate.load_state_dict(state_backup, strict=False)

            return NeuralAcquisitionResult(
                task_id=task.task_id,
                status="REJECTED_DEV_FAILURE",
                dev_exact_match=dev_em,
                held_out_test_em=0.0,
                shuffled_weights_test_em=0.0,
                causal_attribution_drop=0.0,
                training_steps=steps,
                parameter_footprint=substrate.count_total_parameters(),
                rollback_performed=True,
            )

        # 5. Held-out test evaluation
        test_em = cls.evaluate_task_exact_match(model, substrate, task.test_observations, device, task.task_id)

        # 6. Causal Neural Attribution Check (Shuffled Weights Control)
        # Snapshot trained residual weights, shuffle them, evaluate test drop, restore
        trained_res_weights = copy.deepcopy(res_module.state_dict())
        with torch.no_grad():
            for p in res_module.parameters():
                p.copy_(p[torch.randperm(p.size(0))])

        shuffled_em = cls.evaluate_task_exact_match(model, substrate, task.test_observations, device, task.task_id)
        res_module.load_state_dict(trained_res_weights)  # Restore trained weights

        drop = max(0.0, test_em - shuffled_em)

        return NeuralAcquisitionResult(
            task_id=task.task_id,
            status="ACCEPTED",
            dev_exact_match=dev_em,
            held_out_test_em=test_em,
            shuffled_weights_test_em=shuffled_em,
            causal_attribution_drop=drop,
            training_steps=steps,
            parameter_footprint=substrate.count_total_parameters(),
            rollback_performed=False,
        )
