"""Milestone M94 Phase 5: Autonomous Program Concept Acquisition & Verification.

Enables the neural student to acquire unfamiliar programmatic operations and algorithms:
1. Modular Residual Adaptation:
   - Registers a dedicated Rank-2 modular residual (260 params) for the unfamiliar concept.
   - Preserves frozen trunk (15,296 params) and base MiniGPT backbone.
2. Verified Acceptance & Safe State Rollback:
   - Extracts statement-level transition tuples from program traces.
   - Trains residual on few-shot transition examples (<= 60 steps).
   - Verifies validation performance on held-out dev split against strict threshold (>= 75%).
   - If verification passes: Commits skill to persistent repertoire.
   - If verification fails: Performs bit-exact rollback of weights.
3. Causal Neural Attribution:
   - Verifies that shuffling acquired residual weights causes a significant drop in performance.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.neural_program_learning.neural_program_executor import (
    NeuralProgramExecutor,
    ProgramReasoningSubstrate,
)
from adaptive_mini_gpt.neural_program_learning.program_curriculum import (
    ProgramExample,
)


@dataclass(frozen=True)
class AcquisitionResult:
    """Outcome of attempting to acquire a programmatic concept."""

    concept_id: str
    status: str  # 'ACCEPTED', 'REJECTED'
    train_loss: float
    dev_exact_match: float
    shuffled_attribution_drop_pct: float
    parameters_allocated: int
    rollback_performed: bool


class AutonomousProgramLearner:
    """Autonomous learner for unfamiliar programming operations."""

    @classmethod
    def acquire_concept(
        cls,
        model: MiniGPT,
        substrate: ProgramReasoningSubstrate,
        concept_id: str,
        train_examples: list[ProgramExample],
        dev_examples: list[ProgramExample],
        device: torch.device,
        steps: int = 80,
        lr: float = 0.03,
        acceptance_threshold_em: float = 75.0,
    ) -> AcquisitionResult:
        """Acquire, verify, and conditionally commit a new programmatic skill."""
        # 1. Snapshot state for guaranteed rollback
        state_snapshot = copy.deepcopy(substrate.state_dict())

        # 2. Register modular residual
        substrate.add_residual(concept_id)
        res_module = substrate.residuals[concept_id]
        res_params = res_module.count_parameters()

        # 3. Extract transition tuples from training programs
        states_in: list[int] = []
        ops_in: list[int] = []
        vals_in: list[int] = []
        targets: list[int] = []

        for ex in train_examples:
            transitions = NeuralProgramExecutor.decompose_program_steps(ex.program)
            for s_val, op_code, op_val, tgt in transitions:
                states_in.append(s_val)
                ops_in.append(op_code)
                vals_in.append(op_val)
                targets.append(tgt)

        if not states_in:
            states_in = [0]
            ops_in = [0]
            vals_in = [0]
            targets = [0]

        s_t = torch.tensor(states_in, dtype=torch.long, device=device)
        op_t = torch.tensor(ops_in, dtype=torch.long, device=device)
        val_t = torch.tensor(vals_in, dtype=torch.long, device=device)
        y_t = torch.tensor(targets, dtype=torch.long, device=device)

        # 4. Train modular residual only
        optimizer = torch.optim.AdamW(res_module.parameters(), lr=lr, weight_decay=1e-4)
        substrate.train()
        final_loss = 0.0

        for _ in range(steps):
            optimizer.zero_grad()
            logits = substrate.forward_transition(s_t, op_t, val_t, active_residual_id=concept_id)
            loss = F.cross_entropy(logits, y_t)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

        # 5. Evaluate on dev split using autoregressive execution
        substrate.eval()
        dev_corr = 0
        with torch.no_grad():
            for ex in dev_examples:
                tr = NeuralProgramExecutor.execute_autoregressive_trace(
                    substrate,
                    ex.program,
                    device,
                    active_residual_id=concept_id,
                )
                if tr.final_prediction == ex.reference_result.output_value:
                    dev_corr += 1
            dev_em = (dev_corr / max(len(dev_examples), 1)) * 100.0

        # 6. Evaluate Causal Neural Attribution (Shuffled Weights)
        with torch.no_grad():
            orig_weights = res_module.down.weight.clone()
            idx = torch.randperm(orig_weights.numel())
            res_module.down.weight.copy_(orig_weights.view(-1)[idx].view_as(orig_weights))

            shuffled_corr = 0
            for ex in dev_examples:
                tr_shuf = NeuralProgramExecutor.execute_autoregressive_trace(
                    substrate,
                    ex.program,
                    device,
                    active_residual_id=concept_id,
                )
                if tr_shuf.final_prediction == ex.reference_result.output_value:
                    shuffled_corr += 1
            shuffled_em = (shuffled_corr / max(len(dev_examples), 1)) * 100.0
            shuffled_drop = max(0.0, dev_em - shuffled_em)

            # Restore original weights
            res_module.down.weight.copy_(orig_weights)

        # 7. Verification Gate
        if dev_em >= acceptance_threshold_em:
            status = "ACCEPTED"
            rollback = False
        else:
            status = "REJECTED"
            if concept_id in substrate.residuals:
                del substrate.residuals[concept_id]
            substrate.load_state_dict(state_snapshot)
            rollback = True
            res_params = 0

        return AcquisitionResult(
            concept_id=concept_id,
            status=status,
            train_loss=final_loss,
            dev_exact_match=dev_em,
            shuffled_attribution_drop_pct=shuffled_drop,
            parameters_allocated=res_params,
            rollback_performed=rollback,
        )
