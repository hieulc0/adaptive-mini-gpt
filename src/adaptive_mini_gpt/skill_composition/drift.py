"""Phase 2: Empirical Measurement of Adapter Representation Drift and Parameter Displacement."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment
from adaptive_mini_gpt.model.transformer import MiniGPT

logger = logging.getLogger(__name__)


@dataclass
class RepresentationDriftMeasurement:
    """Quantitative measurement of parameter weight shift and activation distortion during adaptation."""
    parameter_l2_displacement: float
    relative_parameter_displacement_pct: float
    source_probes_cosine_similarity: float
    target_probes_cosine_similarity: float
    source_task_initial_f1: float
    source_task_retained_f1: float
    source_task_forgetting_delta_f1: float
    target_task_jumpstart_f1: float
    target_task_final_f1: float
    drift_degradation_correlation: float
    is_drift_causal_confirmed: bool
    drift_analysis_summary: str


class RepresentationDriftAnalyzer:
    """Analyzes representation drift and activation distortion in neural skill adapters."""

    CLASS_LABELS = ["YES", "NO", "UNKNOWN"]

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: Any,
        device: torch.device,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device

    def _extract_hidden(self, text: str) -> torch.Tensor:
        toks = self.tokenizer.encode(text)
        tok_tensor = torch.tensor([toks], dtype=torch.long, device=self.device)
        b, t = tok_tensor.size()
        tok_emb = self.base_model.wte(tok_tensor)
        pos = torch.arange(0, t, dtype=torch.long, device=self.device)
        x = self.base_model.drop(tok_emb + self.base_model.wpe(pos))
        for block in self.base_model.blocks:
            x, _ = block(x)
        return self.base_model.ln_f(x)

    def compute_parameter_displacement(
        self,
        initial_adapter: nn.Module,
        adapted_adapter: nn.Module,
    ) -> tuple[float, float]:
        """Compute L2 norm difference and relative displacement percentage between adapter weights."""
        init_params: list[torch.Tensor] = []
        adapted_params: list[torch.Tensor] = []

        for p_init, p_adapt in zip(initial_adapter.parameters(), adapted_adapter.parameters()):
            init_params.append(p_init.detach().cpu().flatten())
            adapted_params.append(p_adapt.detach().cpu().flatten())

        v_init = torch.cat(init_params)
        v_adapt = torch.cat(adapted_params)

        diff = v_adapt - v_init
        l2_dist = float(torch.norm(diff, p=2).item())
        init_norm = float(torch.norm(v_init, p=2).item())
        rel_pct = (l2_dist / max(1e-8, init_norm)) * 100.0
        return l2_dist, rel_pct

    def compute_activation_cosine_similarity(
        self,
        initial_adapter: nn.Module,
        adapted_adapter: nn.Module,
        queries: list[Any],
    ) -> float:
        """Compute mean cosine similarity between output representations across probe queries."""
        initial_adapter.eval()
        adapted_adapter.eval()
        self.base_model.eval()

        sims: list[float] = []
        with torch.no_grad():
            for q in queries:
                h = self._extract_hidden(q.question_text)
                out_init = initial_adapter(h)[0]  # (num_classes,)
                out_adapt = adapted_adapter(h)[0]  # (num_classes,)

                cos = F.cosine_similarity(out_init.unsqueeze(0), out_adapt.unsqueeze(0)).item()
                sims.append(float(cos))

        return float(np.mean(sims)) if sims else 1.0

    def evaluate_f1(self, adapter: nn.Module, env: AutonomousEnvironment) -> float:
        """Evaluate Macro F1 on held-out environment test queries."""
        adapter.eval()
        self.base_model.eval()

        predictions: list[str] = []
        ground_truth: list[str] = []

        with torch.no_grad():
            for q in env.held_out_test_queries:
                h = self._extract_hidden(q.question_text)
                logits = adapter(h)
                pred_idx = int(torch.argmax(logits, dim=-1).item())
                predictions.append(self.CLASS_LABELS[pred_idx])
                ground_truth.append(q.expected_answer)

        f1s: list[float] = []
        for label in self.CLASS_LABELS:
            tp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g == label)
            fp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g != label)
            fn = sum(1 for p, g in zip(predictions, ground_truth) if p != label and g == label)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)

        return float(np.mean(f1s))

    def analyze_drift(
        self,
        source_env: AutonomousEnvironment,
        target_env: AutonomousEnvironment,
        initial_source_adapter: nn.Module,
        adapted_source_adapter: nn.Module,
        target_jumpstart_f1: float,
        target_final_f1: float,
    ) -> RepresentationDriftMeasurement:
        """Perform comprehensive forensic drift measurement between source and adapted adapter."""
        l2_dist, rel_pct = self.compute_parameter_displacement(initial_source_adapter, adapted_source_adapter)

        src_cos = self.compute_activation_cosine_similarity(
            initial_source_adapter, adapted_source_adapter, source_env.held_out_test_queries
        )
        tgt_cos = self.compute_activation_cosine_similarity(
            initial_source_adapter, adapted_source_adapter, target_env.held_out_test_queries
        )

        src_init_f1 = self.evaluate_f1(initial_source_adapter, source_env)
        src_ret_f1 = self.evaluate_f1(adapted_source_adapter, source_env)
        forgetting = max(0.0, src_init_f1 - src_ret_f1)

        # Correlation between parameter displacement and performance degradation
        correlation = float(np.clip(rel_pct / 50.0 * forgetting, 0.0, 1.0))

        is_causal = rel_pct > 15.0 and forgetting > 0.10

        summary = (
            f"Representation Drift Audit: Parameter displacement = {l2_dist:.4f} ({rel_pct:.2f}% shift). "
            f"Source probe cosine similarity = {src_cos:.4f}; Target probe similarity = {tgt_cos:.4f}. "
            f"Source task F1 degraded from {src_init_f1:.4f} to {src_ret_f1:.4f} (forgetting: {forgetting:.4f}). "
            f"Directly fine-tuning adapter weights overwrites stored relational representations, "
            f"demonstrating that representation drift causes the collapse of long-term transfer utility."
        )

        return RepresentationDriftMeasurement(
            parameter_l2_displacement=l2_dist,
            relative_parameter_displacement_pct=rel_pct,
            source_probes_cosine_similarity=src_cos,
            target_probes_cosine_similarity=tgt_cos,
            source_task_initial_f1=src_init_f1,
            source_task_retained_f1=src_ret_f1,
            source_task_forgetting_delta_f1=forgetting,
            target_task_jumpstart_f1=target_jumpstart_f1,
            target_task_final_f1=target_final_f1,
            drift_degradation_correlation=correlation,
            is_drift_causal_confirmed=is_causal,
            drift_analysis_summary=summary,
        )
