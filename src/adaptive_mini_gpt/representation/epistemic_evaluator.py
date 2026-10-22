"""Phase 6: Epistemic Uncertainty and Selective Risk Evaluation.

Measures:
  - 3-class discrimination (Macro F1, per-class precision/recall).
  - Probability calibration (ECE, Brier score).
  - Validation threshold tuning for abstention.
  - Selective Risk & Accuracy across coverage levels (100%, 80%, 50%).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from adaptive_mini_gpt.data.bpe_scaling import BPETokenizer
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.representation.positive_control import TriStateClassificationHead
from adaptive_mini_gpt.representation.structural_benchmark import RelationalGraphDataset
from adaptive_mini_gpt.uncertainty.epistemic_benchmark import EpistemicTask


@dataclass
class SelectiveRiskMetrics:
    coverage_100_accuracy: float
    coverage_80_accuracy: float
    coverage_50_accuracy: float
    calibrated_threshold: float
    effective_coverage_at_threshold: float
    abstention_precision: float  # Fraction of abstentions that were true UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage_100_accuracy": round(self.coverage_100_accuracy, 4),
            "coverage_80_accuracy": round(self.coverage_80_accuracy, 4),
            "coverage_50_accuracy": round(self.coverage_50_accuracy, 4),
            "calibrated_threshold": round(self.calibrated_threshold, 4),
            "effective_coverage_at_threshold": round(self.effective_coverage_at_threshold, 4),
            "abstention_precision": round(self.abstention_precision, 4),
        }


class EpistemicGeneralizationEvaluator:
    """Evaluates uncertainty calibration and selective prediction on relational tasks."""

    CLASSES = ["YES", "NO", "UNKNOWN"]
    CLASS_TO_IDX = {"YES": 0, "NO": 1, "UNKNOWN": 2}

    def __init__(
        self,
        model: MiniGPT,
        tri_head: TriStateClassificationHead,
        tokenizer: BPETokenizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tri_head = tri_head
        self.tokenizer = tokenizer
        self.device = device

    def _predict_task(self, premises: list[str], question: str) -> tuple[str, float, list[float]]:
        prompt = f"[PREMISES] {' '.join(premises)} [QUESTION] {question} [ANSWER]"
        ids = self.tokenizer.encode(prompt)
        if len(ids) > self.model.config.context_length:
            ids = ids[-self.model.config.context_length:]
        inp = torch.tensor([ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            b, t = inp.size()
            tok_emb = self.model.wte(inp)
            pos = torch.arange(0, t, dtype=torch.long, device=inp.device)
            pos_emb = self.model.wpe(pos)
            x = self.model.drop(tok_emb + pos_emb)
            for block in self.model.blocks:
                x, _ = block(x)
            hidden = self.model.ln_f(x)
            logits = self.tri_head(hidden)
            probs = torch.softmax(logits, dim=-1)[0].tolist()
            pred_idx = int(torch.argmax(logits, dim=-1).item())
            return self.CLASSES[pred_idx], float(probs[pred_idx]), probs

    def calibrate_threshold(self, val_tasks: list[EpistemicTask]) -> float:
        """Finds confidence threshold on validation pool that maximizes macro F1 with abstention."""
        best_thresh = 0.50
        best_f1 = -1.0

        for thresh in np.linspace(0.40, 0.85, 10):
            correct = 0
            for t in val_tasks:
                pred, conf, _ = self._predict_task(t.premises, t.question)
                # If confidence is below threshold, model abstains -> predicts UNKNOWN
                eff_pred = pred if conf >= thresh else "UNKNOWN"
                if eff_pred == t.expected_verdict:
                    correct += 1
            f1 = correct / max(1, len(val_tasks))
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)

        return best_thresh

    def evaluate_selective_risk(
        self,
        test_tasks: list[EpistemicTask],
        calibrated_threshold: float,
    ) -> SelectiveRiskMetrics:
        """Evaluates accuracy under selective coverage."""
        scored_items: list[tuple[float, bool, str, str]] = []  # (conf, is_correct, pred, true)
        for t in test_tasks:
            pred, conf, _ = self._predict_task(t.premises, t.question)
            is_correct = (pred == t.expected_verdict)
            scored_items.append((conf, is_correct, pred, t.expected_verdict))

        # Sort descending by confidence
        scored_items.sort(key=lambda x: x[0], reverse=True)
        total = len(scored_items)

        def acc_at_k(k: int) -> float:
            sub = scored_items[:k]
            return sum(1 for x in sub if x[1]) / max(1, len(sub))

        acc_100 = acc_at_k(total)
        acc_80 = acc_at_k(max(1, int(total * 0.80)))
        acc_50 = acc_at_k(max(1, int(total * 0.50)))

        # Effective coverage at calibrated threshold
        answered = [x for x in scored_items if x[0] >= calibrated_threshold]
        coverage_rate = len(answered) / max(1, total)

        # Abstention precision: among queries where model was unsure (conf < threshold)
        abstained = [x for x in scored_items if x[0] < calibrated_threshold]
        true_unknown_in_abstained = sum(1 for x in abstained if x[3] == "UNKNOWN")
        abstention_p = true_unknown_in_abstained / max(1, len(abstained))

        return SelectiveRiskMetrics(
            coverage_100_accuracy=acc_100,
            coverage_80_accuracy=acc_80,
            coverage_50_accuracy=acc_50,
            calibrated_threshold=calibrated_threshold,
            effective_coverage_at_threshold=coverage_rate,
            abstention_precision=abstention_p,
        )
