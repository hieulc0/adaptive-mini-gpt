"""Phase 3 & 5: Two-Stage Neural Composition Pipeline & Error Diagnostic Suite.

Implements:
- Closed-loop, two-stage neural composition where Adapter A's discrete output sequence
  is fed directly through the frozen MiniGPT backbone into Adapter B:
    Input x -> Backbone -> Adapter A -> Intermediate y1 -> Backbone -> Adapter B -> Final y2
- Teacher-Forced Stage 2 Diagnostic:
    Ground-truth intermediate y1* -> Backbone -> Adapter B -> Final y2_forced
- Fine-grained Phase 5 error decomposition:
    1. STAGE_1_DEFECT: Adapter A fails on input.
    2. STAGE_2_DEFECT: Adapter B fails even when provided with correct intermediate tokens.
    3. ERROR_PROPAGATION: Adapter B fails solely due to corrupted inputs from Adapter A.
    4. DUAL_DEFECT: Both stages fail independently.
    5. SUCCESS: Both stages predict exact sequences correctly.
- Exact-match, token accuracy, cross-entropy sequence loss, and active parameter tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.neural_skill_composition.compound_tasks import (
    CompoundProceduralExample,
)
from adaptive_mini_gpt.neural_weight_learning.sequence_adapter import Rank16SequenceAdapter

logger = logging.getLogger(__name__)


@dataclass
class CompositionStepDiagnostic:
    """Detailed error diagnosis for a single compound example."""

    example_id: str
    input_tokens: tuple[int, ...]
    expected_intermediate: tuple[int, ...]
    predicted_intermediate: tuple[int, ...]
    expected_final: tuple[int, ...]
    predicted_final: tuple[int, ...]
    teacher_forced_final: tuple[int, ...]
    stage1_exact_match: bool
    final_exact_match: bool
    teacher_forced_match: bool
    error_category: str  # SUCCESS, STAGE_1_DEFECT, ERROR_PROPAGATION, STAGE_2_DEFECT, DUAL_DEFECT


@dataclass
class CompositionExecutionReport:
    """Comprehensive performance and diagnostic report for a neural composition run."""

    pipeline_name: str
    num_examples: int
    cascaded_exact_match: float
    token_accuracy: float
    stage1_exact_match: float
    teacher_forced_exact_match: float
    error_proportions: dict[str, float]
    mean_stage1_loss: float
    mean_stage2_loss: float
    active_parameters_per_stage: int
    total_stored_adapter_parameters: int
    detailed_diagnostics: list[CompositionStepDiagnostic]


class NeuralCompositionPipeline:
    """Execution engine for chained two-stage neural adapter composition."""

    BASE_OFFSET = 64

    @classmethod
    def _tokens_to_hidden_states(
        cls,
        model: MiniGPT,
        tokens_list: list[tuple[int, ...] | list[int]],
        device: torch.device,
    ) -> torch.Tensor:
        """Map discrete symbol sequences to frozen MiniGPT backbone hidden representations."""
        input_ids = [
            [cls.BASE_OFFSET + s for s in seq] for seq in tokens_list
        ]
        x = torch.tensor(input_ids, dtype=torch.long, device=device)

        model.eval()
        with torch.no_grad():
            tok_emb = model.wte(x)
            if model.wpe is not None:
                pos = torch.arange(0, x.size(1), dtype=torch.long, device=device)
                h = model.drop(tok_emb + model.wpe(pos))
            else:
                h = model.drop(tok_emb)
            for block in model.blocks:
                h, _ = block(h)
            h = model.ln_f(h)
        return h

    @classmethod
    def execute_composition(
        cls,
        model: MiniGPT,
        adapter_stage1: Rank16SequenceAdapter,
        adapter_stage2: Rank16SequenceAdapter,
        examples: list[CompoundProceduralExample],
        device: torch.device,
        pipeline_name: str = "NEURAL_COMPOSITION",
    ) -> CompositionExecutionReport:
        """Execute closed-loop cascaded neural composition and teacher-forced diagnostics."""
        adapter_stage1.eval()
        adapter_stage2.eval()

        input_seqs = [ex.input_tokens for ex in examples]
        true_intermediates = [ex.intermediate_tokens for ex in examples]
        true_finals = [ex.target_tokens for ex in examples]

        # 1. Stage 1 Forward Pass: Input -> Backbone -> Adapter 1
        h1 = cls._tokens_to_hidden_states(model, input_seqs, device)
        with torch.no_grad():
            logits1 = adapter_stage1(h1)
            pred_y1_tensor = torch.argmax(logits1, dim=-1)
            pred_y1_list = [tuple(int(s) for s in row) for row in pred_y1_tensor.cpu().numpy()]

        # 2. Stage 2 Forward Pass: Predicted y1 -> Backbone -> Adapter 2 (Cascaded Neural Composition!)
        h2_cascaded = cls._tokens_to_hidden_states(model, pred_y1_list, device)
        with torch.no_grad():
            logits2_cascaded = adapter_stage2(h2_cascaded)
            pred_y2_tensor = torch.argmax(logits2_cascaded, dim=-1)
            pred_y2_list = [tuple(int(s) for s in row) for row in pred_y2_tensor.cpu().numpy()]

        # 3. Teacher-Forced Stage 2 Diagnostic: True y1* -> Backbone -> Adapter 2
        h2_forced = cls._tokens_to_hidden_states(model, true_intermediates, device)
        with torch.no_grad():
            logits2_forced = adapter_stage2(h2_forced)
            pred_y2_forced_tensor = torch.argmax(logits2_forced, dim=-1)
            pred_y2_forced_list = [tuple(int(s) for s in row) for row in pred_y2_forced_tensor.cpu().numpy()]

        # 4. Evaluate Stage Losses
        y1_true_t = torch.tensor(true_intermediates, dtype=torch.long, device=device)
        y2_true_t = torch.tensor(true_finals, dtype=torch.long, device=device)
        loss1 = float(F.cross_entropy(logits1.view(-1, 16), y1_true_t.view(-1)).item())
        loss2 = float(F.cross_entropy(logits2_cascaded.view(-1, 16), y2_true_t.view(-1)).item())

        # 5. Diagnostic Error Decomposition per Example
        diagnostics: list[CompositionStepDiagnostic] = []
        error_counts = {
            "SUCCESS": 0,
            "STAGE_1_DEFECT": 0,
            "ERROR_PROPAGATION": 0,
            "STAGE_2_DEFECT": 0,
            "DUAL_DEFECT": 0,
        }

        total_tokens = 0
        correct_tokens = 0

        for i, ex in enumerate(examples):
            y1_hat = pred_y1_list[i]
            y1_star = true_intermediates[i]
            y2_hat = pred_y2_list[i]
            y2_star = true_finals[i]
            y2_forced = pred_y2_forced_list[i]

            s1_match = (y1_hat == y1_star)
            final_match = (y2_hat == y2_star)
            forced_match = (y2_forced == y2_star)

            # Token level accuracy on final prediction
            for t_hat, t_star in zip(y2_hat, y2_star, strict=False):
                total_tokens += 1
                if t_hat == t_star:
                    correct_tokens += 1

            # Determine error category
            if s1_match and final_match:
                cat = "SUCCESS"
            elif not s1_match and forced_match:
                # Stage 1 was wrong, but Stage 2 would have succeeded with clean intermediate!
                cat = "ERROR_PROPAGATION"
            elif s1_match and not final_match:
                # Stage 1 was correct, but Stage 2 failed on clean intermediate!
                cat = "STAGE_2_DEFECT"
            elif not s1_match and not forced_match and not final_match:
                cat = "DUAL_DEFECT"
            else:
                cat = "STAGE_1_DEFECT"

            error_counts[cat] += 1
            diagnostics.append(
                CompositionStepDiagnostic(
                    example_id=ex.example_id,
                    input_tokens=ex.input_tokens,
                    expected_intermediate=y1_star,
                    predicted_intermediate=y1_hat,
                    expected_final=y2_star,
                    predicted_final=y2_hat,
                    teacher_forced_final=y2_forced,
                    stage1_exact_match=s1_match,
                    final_exact_match=final_match,
                    teacher_forced_match=forced_match,
                    error_category=cat,
                )
            )

        n = len(examples)
        cascaded_em = sum(1 for d in diagnostics if d.final_exact_match) / n
        stage1_em = sum(1 for d in diagnostics if d.stage1_exact_match) / n
        forced_em = sum(1 for d in diagnostics if d.teacher_forced_match) / n
        tok_acc = correct_tokens / max(1, total_tokens)
        error_props = {k: round(v / n, 4) for k, v in error_counts.items()}

        p1_params = adapter_stage1.count_parameters()
        p2_params = adapter_stage2.count_parameters()

        return CompositionExecutionReport(
            pipeline_name=pipeline_name,
            num_examples=n,
            cascaded_exact_match=round(cascaded_em, 4),
            token_accuracy=round(tok_acc, 4),
            stage1_exact_match=round(stage1_em, 4),
            teacher_forced_exact_match=round(forced_em, 4),
            error_proportions=error_props,
            mean_stage1_loss=round(loss1, 4),
            mean_stage2_loss=round(loss2, 4),
            active_parameters_per_stage=p1_params,
            total_stored_adapter_parameters=p1_params + p2_params,
            detailed_diagnostics=diagnostics,
        )
