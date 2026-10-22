"""Phase 2: Representation and Optimization Positive Controls.

Verifies whether the existing 128-dimensional MiniGPT backbone + adapter
can learn a balanced 3-class relational dataset (YES, NO, UNKNOWN) and
generalize to held-out test tasks under the same graph rules.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from adaptive_mini_gpt.data.bpe_scaling import BPETokenizer
from adaptive_mini_gpt.model.modular_adapter import (
    add_domain_adapter,
    freeze_for_adapter_training,
    inject_multi_domain_adapters,
    set_active_domain_adapter,
)
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.uncertainty.epistemic_benchmark import EpistemicTask, EpistemicWorldDataset


class TriStateClassificationHead(nn.Module):
    """3-class classification head attached to Transformer token hidden states."""

    def __init__(self, d_model: int = 128, num_classes: int = 3) -> None:
        super().__init__()
        self.head = nn.Linear(d_model, num_classes, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Select last token representation: [batch, seq_len, d_model] -> [batch, d_model]
        last_hidden = hidden_states[:, -1, :]
        return self.head(last_hidden)


@dataclass
class PositiveControlResult:
    configuration: str  # e.g., "ADAPTER_PLUS_HEAD" or "FROZEN_BASE_HEAD_ONLY"
    loss_curve: list[float]
    final_loss: float
    gradient_norm_initial: float
    gradient_norm_final: float
    max_adapter_weight_delta: float
    train_accuracy: float
    held_out_accuracy: float
    train_confusion_matrix: dict[str, dict[str, int]]
    held_out_confusion_matrix: dict[str, dict[str, int]]
    per_class_sample_probs: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration,
            "loss_curve": [round(l, 4) for l in self.loss_curve],
            "final_loss": round(self.final_loss, 4),
            "gradient_norm_initial": round(self.gradient_norm_initial, 4),
            "gradient_norm_final": round(self.gradient_norm_final, 4),
            "max_adapter_weight_delta": round(self.max_adapter_weight_delta, 6),
            "train_accuracy": round(self.train_accuracy, 4),
            "held_out_accuracy": round(self.held_out_accuracy, 4),
            "train_confusion_matrix": self.train_confusion_matrix,
            "held_out_confusion_matrix": self.held_out_confusion_matrix,
            "per_class_sample_probs": {k: [round(p, 4) for p in v] for k, v in self.per_class_sample_probs.items()},
        }


class RepresentationPositiveControlRunner:
    """Runs positive control optimization diagnostics on balanced 3-way relational probes."""

    CLASSES = ["YES", "NO", "UNKNOWN"]
    CLASS_TO_IDX = {"YES": 0, "NO": 1, "UNKNOWN": 2}

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: BPETokenizer,
        device: torch.device,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device

    def _extract_hidden(self, model: MiniGPT, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.size()
        tok_emb = model.wte(input_ids)
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device)
        pos_emb = model.wpe(pos)
        x = model.drop(tok_emb + pos_emb)
        for block in model.blocks:
            x, _ = block(x)
        return model.ln_f(x)

    def _build_balanced_probe(
        self, world: EpistemicWorldDataset, n_per_class: int = 2
    ) -> tuple[list[EpistemicTask], list[EpistemicTask]]:
        """Extracts n_per_class items for each of YES, NO, UNKNOWN for train & test."""
        by_class: dict[str, list[EpistemicTask]] = {"YES": [], "NO": [], "UNKNOWN": []}
        for t in world.train_practice_pool:
            by_class[t.expected_verdict].append(t)

        train_probe: list[EpistemicTask] = []
        test_probe: list[EpistemicTask] = []

        for c in self.CLASSES:
            items = by_class[c]
            train_probe.extend(items[:n_per_class])
            test_probe.extend(items[n_per_class : n_per_class * 2])

        return train_probe, test_probe

    def run_control(
        self,
        world: EpistemicWorldDataset,
        use_adapter: bool = True,
        epochs: int = 60,
        lr: float = 2e-3,
    ) -> PositiveControlResult:
        """Executes positive control experiment comparing adapter vs frozen backbone."""
        train_probe, test_probe = self._build_balanced_probe(world, n_per_class=2)
        model = copy.deepcopy(self.base_model)

        init_weights: dict[str, torch.Tensor] = {}
        if use_adapter:
            inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
            add_domain_adapter(model, domain_name="pos_control", r=8)
            set_active_domain_adapter(model, "pos_control")
            freeze_for_adapter_training(model, "pos_control")
            for name, param in model.named_parameters():
                if "pos_control" in name:
                    init_weights[name] = param.detach().clone()
        else:
            for p in model.parameters():
                p.requires_grad = False

        tri_head = TriStateClassificationHead(d_model=model.config.d_model, num_classes=3).to(self.device)
        trainable_params = list(tri_head.parameters()) + [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=lr)
        criterion = nn.CrossEntropyLoss()

        loss_curve: list[float] = []
        grad_norm_init = 0.0
        grad_norm_final = 0.0

        model.train()
        tri_head.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for step, t in enumerate(train_probe):
                prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                ids = self.tokenizer.encode(prompt)
                if len(ids) > model.config.context_length:
                    ids = ids[-model.config.context_length:]
                inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                target = torch.tensor([self.CLASS_TO_IDX[t.expected_verdict]], dtype=torch.long, device=self.device)

                hidden = self._extract_hidden(model, inp)
                logits = tri_head(hidden)
                loss = criterion(logits, target)

                optimizer.zero_grad()
                loss.backward()

                # Record grad norm
                total_norm = 0.0
                for p in trainable_params:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                if epoch == 0 and step == 0:
                    grad_norm_init = total_norm
                if epoch == epochs - 1 and step == len(train_probe) - 1:
                    grad_norm_final = total_norm

                optimizer.step()
                epoch_loss += loss.item()

            mean_epoch_loss = epoch_loss / len(train_probe)
            loss_curve.append(mean_epoch_loss)

        # Max weight delta on adapters
        max_delta = 0.0
        if use_adapter:
            for name, param in model.named_parameters():
                if "pos_control" in name and name in init_weights:
                    delta = (param.detach() - init_weights[name]).abs().max().item()
                    if delta > max_delta:
                        max_delta = delta

        # Evaluate on train probe
        model.eval()
        tri_head.eval()

        def evaluate_probe(probe: list[EpistemicTask]) -> tuple[float, dict[str, dict[str, int]], dict[str, list[float]]]:
            cm = {c: {k: 0 for k in self.CLASSES} for c in self.CLASSES}
            correct = 0
            sample_probs: dict[str, list[float]] = {}
            with torch.no_grad():
                for t in probe:
                    prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                    ids = self.tokenizer.encode(prompt)
                    if len(ids) > model.config.context_length:
                        ids = ids[-model.config.context_length:]
                    inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                    hidden = self._extract_hidden(model, inp)
                    logits = tri_head(hidden)
                    probs = torch.softmax(logits, dim=-1)[0].tolist()
                    pred_idx = int(torch.argmax(logits, dim=-1).item())
                    pred = self.CLASSES[pred_idx]
                    cm[t.expected_verdict][pred] += 1
                    if pred == t.expected_verdict:
                        correct += 1
                    if t.expected_verdict not in sample_probs:
                        sample_probs[t.expected_verdict] = probs
            acc = correct / max(1, len(probe))
            return acc, cm, sample_probs

        train_acc, train_cm, train_probs = evaluate_probe(train_probe)
        held_out_acc, held_out_cm, _ = evaluate_probe(test_probe)

        return PositiveControlResult(
            configuration="ADAPTER_PLUS_HEAD" if use_adapter else "FROZEN_BASE_HEAD_ONLY",
            loss_curve=loss_curve,
            final_loss=loss_curve[-1] if loss_curve else 0.0,
            gradient_norm_initial=grad_norm_init,
            gradient_norm_final=grad_norm_final,
            max_adapter_weight_delta=max_delta,
            train_accuracy=train_acc,
            held_out_accuracy=held_out_acc,
            train_confusion_matrix=train_cm,
            held_out_confusion_matrix=held_out_cm,
            per_class_sample_probs=train_probs,
        )
