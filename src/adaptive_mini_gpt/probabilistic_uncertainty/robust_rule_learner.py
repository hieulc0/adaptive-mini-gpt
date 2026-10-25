"""Milestone M93 Phase 4 & 5: Robust Neural Acquisition, Calibration & Lifelong Memory.

Integrates robust neural skill acquisition, temperature calibration, and lifelong retention:
1. Substrate Capacity & Rollback Safeguards:
   - Shared Concept Trunk (10,304 parameters, frozen)
   - Task-Specific Rank-4 Modular Residuals (1,032 parameters per skill)
   - TemperatureScaler (1 scalar parameter for probability calibration)
   - Total Budget: 10,304 + 8 * 1,032 + 1 = 18,561 parameters (1,439 headroom below 20,000 cap).
   - Bit-exact state restoration if dev verification fails.
2. Noise-Robust Residual Optimization:
   - Distills validated procedural hypotheses into allocated modular residuals using AdamW.
   - Fits TemperatureScaler on dev logits to achieve calibrated predictive probabilities.
3. Causal Neural Attribution:
   - Shuffles newly acquired weights to confirm performance causally derives from neural weights.
4. Lifelong Memory Retention & Compositional Reasoning:
   - Evaluates all earlier admitted skills on clean held-out tests after sequential learning.
   - Chains discovered neural residuals into multi-step computations (Discovered -> ADD_3).
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from adaptive_mini_gpt.autonomous_discovery.unknown_rule_environment import (
    ProceduralRule,
    RuleObservation,
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
from adaptive_mini_gpt.probabilistic_uncertainty.neural_uncertainty import (
    CalibrationMetrics,
    NeuralUncertaintyEvaluator,
    SelectivePredictionProfile,
    TemperatureScaler,
)
from adaptive_mini_gpt.probabilistic_uncertainty.noisy_environment import (
    NoisyObservation,
    NoisyRuleTask,
)


@dataclass(frozen=True)
class RobustAcquisitionResult:
    """Outcome of robust neural skill acquisition under uncertainty."""

    task_id: str
    status: str  # "ACCEPTED", "REJECTED_DEV_FAILURE", "REJECTED_BUDGET"
    dev_exact_match: float
    held_out_test_em: float
    shuffled_weights_test_em: float
    causal_attribution_drop: float
    calibration_metrics: CalibrationMetrics
    selective_prediction: SelectivePredictionProfile
    parameter_footprint: int
    rollback_performed: bool


class RobustRuleLearner:
    """Manages robust neural training, temperature calibration, and attribution."""

    @classmethod
    def format_observations_to_prompts(
        cls,
        observations: Sequence[NoisyObservation | RuleObservation],
        task_id: str,
    ) -> list[InstructionPrompt]:
        """Convert observations into contextual instruction prompts."""
        prompts: list[InstructionPrompt] = []
        for obs in observations:
            p = InstructionGrammar.format_atomic(
                opcode=Opcode.ADD,
                operand=0,
                symbols=[obs.x, obs.x, obs.x, obs.x],
                target_symbols=[obs.y, obs.y, obs.y, obs.y],
                op_family=f"discovered_{task_id}",
            )
            prompts.append(p)
        return prompts

    @classmethod
    def evaluate_task_logits_and_labels(
        cls,
        model: MiniGPT,
        substrate: PersistentReasoningSubstrate,
        observations: Sequence[NoisyObservation],
        task_id: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract hidden features and compute raw logits and labels."""
        prompts = cls.format_observations_to_prompts(observations, task_id)
        h, y = extract_prompt_features(model, prompts, device)
        with torch.no_grad():
            logits = substrate(h, task_id=task_id)[:, 0]  # First symbol position (batch, 16)
            labels = y[:, 0]  # (batch,)
        return logits, labels

    @classmethod
    def evaluate_task_exact_match(
        cls,
        model: MiniGPT,
        substrate: PersistentReasoningSubstrate,
        observations: Sequence[NoisyObservation],
        task_id: str,
        device: torch.device,
    ) -> float:
        """Evaluate exact match accuracy on given observations."""
        if not observations:
            return 0.0
        logits, labels = cls.evaluate_task_logits_and_labels(model, substrate, observations, task_id, device)
        preds = logits.argmax(dim=-1)
        em = (preds == labels).float().mean().item() * 100.0
        return round(em, 2)

    @classmethod
    def acquire_skill(
        cls,
        model: MiniGPT,
        substrate: PersistentReasoningSubstrate,
        task: NoisyRuleTask,
        device: torch.device,
        discovered_rule: ProceduralRule | None = None,
        steps: int = 50,
        lr: float = 0.03,
        dev_threshold: float = 50.0,
    ) -> RobustAcquisitionResult:
        """Allocate modular residual, train on verified observations, calibrate, and verify."""
        state_backup = copy.deepcopy(substrate.state_dict())

        try:
            substrate.add_residual(task.task_id)
        except RuntimeError:
            return RobustAcquisitionResult(
                task_id=task.task_id,
                status="REJECTED_BUDGET",
                dev_exact_match=0.0,
                held_out_test_em=0.0,
                shuffled_weights_test_em=0.0,
                causal_attribution_drop=0.0,
                calibration_metrics=CalibrationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                selective_prediction=SelectivePredictionProfile([1.0], [0.0], True, 0.0),
                parameter_footprint=substrate.count_total_parameters(),
                rollback_performed=True,
            )

        res_module = substrate.residuals[task.task_id]
        opt = torch.optim.AdamW(res_module.parameters(), lr=lr, weight_decay=1e-4)

        # Distill discovered procedural rule or train on observed pairs
        if discovered_rule is not None:
            synth_obs = [NoisyObservation(x=x, y=discovered_rule.apply(x)) for x in range(16)]
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

        # Dev set verification
        dev_em = cls.evaluate_task_exact_match(model, substrate, task.dev_observations, task.task_id, device)

        if dev_em < dev_threshold:
            # Rollback on dev failure
            del substrate.residuals[task.task_id]
            substrate.load_state_dict(state_backup, strict=False)
            return RobustAcquisitionResult(
                task_id=task.task_id,
                status="REJECTED_DEV_FAILURE",
                dev_exact_match=dev_em,
                held_out_test_em=0.0,
                shuffled_weights_test_em=0.0,
                causal_attribution_drop=0.0,
                calibration_metrics=CalibrationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                selective_prediction=SelectivePredictionProfile([1.0], [0.0], True, 0.0),
                parameter_footprint=substrate.count_total_parameters(),
                rollback_performed=True,
            )

        # Fit temperature scaler on dev logits
        dev_logits, dev_labels = cls.evaluate_task_logits_and_labels(
            model, substrate, task.dev_observations, task.task_id, device
        )
        scaler = TemperatureScaler(initial_temp=1.0)
        scaler.fit(dev_logits, dev_labels)

        # Evaluate on clean sealed test set
        test_logits, test_labels = cls.evaluate_task_logits_and_labels(
            model, substrate, task.test_observations, task.task_id, device
        )
        test_em = (test_logits.argmax(dim=-1) == test_labels).float().mean().item() * 100.0

        calibrated_probs = F.softmax(scaler(test_logits), dim=-1)
        calib_metrics = NeuralUncertaintyEvaluator.compute_calibration_metrics(
            calibrated_probs, test_labels, temperature=scaler.temperature
        )
        sel_pred = NeuralUncertaintyEvaluator.compute_selective_prediction(calibrated_probs, test_labels)

        # Causal neural attribution: shuffle newly learned residual weights
        res_backup = copy.deepcopy(res_module.state_dict())
        with torch.no_grad():
            for p in res_module.parameters():
                flat = p.data.view(-1)
                perm = torch.randperm(flat.size(0))
                p.data.copy_(flat[perm].view_as(p.data))

        shuffled_em = cls.evaluate_task_exact_match(model, substrate, task.test_observations, task.task_id, device)
        res_module.load_state_dict(res_backup)
        drop = max(0.0, test_em - shuffled_em)

        return RobustAcquisitionResult(
            task_id=task.task_id,
            status="ACCEPTED",
            dev_exact_match=dev_em,
            held_out_test_em=round(test_em, 2),
            shuffled_weights_test_em=round(shuffled_em, 2),
            causal_attribution_drop=round(drop, 2),
            calibration_metrics=calib_metrics,
            selective_prediction=sel_pred,
            parameter_footprint=substrate.count_total_parameters(),
            rollback_performed=False,
        )
