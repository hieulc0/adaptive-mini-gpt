"""Phase 7: Causal Neural Attribution Auditor.

Verifies that retained learned neural parameters are causally responsible for
task performance improvements by conducting 4 ablation controls:
1. Retained Learned Adapter (Trained Candidate)
2. Fresh Randomly Initialized Adapter (Untrained Baseline)
3. Shuffled / Permuted Adapter Weights (Weight Distribution Control)
4. Unrelated Learned Adapter (Out-of-Family Control)
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.three_valued_reasoning.architectures import Rank16ResidualAdapter
from adaptive_mini_gpt.modular_skill_learning.sequential_transfer import extract_model_hidden
from adaptive_mini_gpt.three_valued_reasoning.semantics import GroundedThreeValuedEnvironment

logger = logging.getLogger(__name__)


@dataclass
class CausalNeuralAttributionReport:
    """Quantitative results verifying genuine neural parameter causality."""
    learned_adapter_f1: float
    fresh_random_f1: float
    shuffled_weights_f1: float
    unrelated_adapter_f1: float
    causal_neural_gain_over_fresh: float
    causal_neural_gain_over_shuffled: float
    causal_attribution_verified: bool
    diagnostics: str


class CausalNeuralAttributionAuditor:
    """Executes weight randomization and permutation ablations to confirm parameter causality."""

    @classmethod
    def evaluate_adapter_on_queries(
        cls,
        adapter: Rank16ResidualAdapter,
        model: MiniGPT,
        env: GroundedThreeValuedEnvironment,
        device: torch.device,
    ) -> float:
        """Evaluate Macro F1 of a specific adapter instance."""
        adapter.eval()
        label_to_idx = {"NO": 0, "YES": 1, "UNKNOWN": 2}
        conf_mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        with torch.no_grad():
            for q in env.held_out_test_queries:
                toks = model.tokenizer.encode(q.question_text)
                inp = torch.tensor([toks], dtype=torch.long, device=device)
                h = extract_model_hidden(model, inp)
                pred = int(torch.argmax(adapter(h), dim=-1).item())
                actual = label_to_idx[q.expected_answer]
                conf_mat[actual][pred] += 1

        f1s = []
        for c in range(3):
            tp = conf_mat[c][c]
            fp = sum(conf_mat[r][c] for r in range(3) if r != c)
            fn = sum(conf_mat[c][p] for p in range(3) if p != c)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
        return float(np.mean(f1s))

    @classmethod
    def run_causal_attribution_audit(
        cls,
        model: MiniGPT,
        learned_adapter: Rank16ResidualAdapter,
        unrelated_adapter: Rank16ResidualAdapter,
        target_env: GroundedThreeValuedEnvironment,
        device: torch.device,
        seed: int = 42,
    ) -> CausalNeuralAttributionReport:
        """Execute the 4-arm causal neural attribution ablation."""
        # 1. Learned Adapter
        learned_f1 = cls.evaluate_adapter_on_queries(learned_adapter, model, target_env, device)

        # 2. Fresh Random Adapter
        torch.manual_seed(seed + 999)
        fresh_adapter = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3).to(device)
        fresh_f1 = cls.evaluate_adapter_on_queries(fresh_adapter, model, target_env, device)

        # 3. Shuffled Weights Adapter (Permute parameter tensors in-place)
        shuffled_adapter = copy.deepcopy(learned_adapter)
        with torch.no_grad():
            for p in shuffled_adapter.parameters():
                flat = p.view(-1).cpu().numpy()
                np.random.shuffle(flat)
                p.copy_(torch.tensor(flat, device=device).view(p.shape))
        shuffled_f1 = cls.evaluate_adapter_on_queries(shuffled_adapter, model, target_env, device)

        # 4. Unrelated Learned Adapter
        unrelated_f1 = cls.evaluate_adapter_on_queries(unrelated_adapter, model, target_env, device)

        gain_over_fresh = learned_f1 - fresh_f1
        gain_over_shuffled = learned_f1 - shuffled_f1
        causal_verified = bool(gain_over_fresh > 0.05 and gain_over_shuffled > 0.05)

        diag = (
            f"LearnedF1={learned_f1:.4f}, FreshF1={fresh_f1:.4f}, ShuffledF1={shuffled_f1:.4f}, "
            f"UnrelatedF1={unrelated_f1:.4f}, GainOverFresh={gain_over_fresh:+.4f}, "
            f"GainOverShuffled={gain_over_shuffled:+.4f}"
        )

        return CausalNeuralAttributionReport(
            learned_adapter_f1=learned_f1,
            fresh_random_f1=fresh_f1,
            shuffled_weights_f1=shuffled_f1,
            unrelated_adapter_f1=unrelated_f1,
            causal_neural_gain_over_fresh=gain_over_fresh,
            causal_neural_gain_over_shuffled=gain_over_shuffled,
            causal_attribution_verified=causal_verified,
            diagnostics=diag,
        )
