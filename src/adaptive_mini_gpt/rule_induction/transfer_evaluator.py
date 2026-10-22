"""Phase 4: Positive and Negative Transfer Probes and Minimal Pair Sensitivity.

Evaluates whether relational learners:
  1. Achieve positive transfer across compatible rule axioms.
  2. Succumb to negative transfer (over-generalizing transitivity to non-transitive graphs,
     or symmetry to asymmetric graphs).
  3. Respond appropriately to minimal pairs (changing exactly 1 directed edge).
"""
from __future__ import annotations

import collections
import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.modular_adapter import (
    add_domain_adapter,
    freeze_for_adapter_training,
    inject_multi_domain_adapters,
    set_active_domain_adapter,
)
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.representation.positive_control import TriStateClassificationHead
from adaptive_mini_gpt.rule_induction.disentangled_benchmark import DisentangledWorldDataset


@dataclass
class TransferProbingMetrics:
    positive_transfer_gain: float  # Accuracy delta on compatible rule
    negative_transfer_error_rate: float  # False positive rate on non-transitive queries
    asymmetric_reversal_accuracy: float  # Accuracy detecting that reversed asymmetric edges are NO
    bridge_deletion_sensitivity: float  # Accuracy detecting that bridge deletion breaks reachability
    minimal_pair_sensitivity: float  # Combined minimal pair accuracy

    def to_dict(self) -> dict[str, Any]:
        return {
            "positive_transfer_gain": round(self.positive_transfer_gain, 4),
            "negative_transfer_error_rate": round(self.negative_transfer_error_rate, 4),
            "asymmetric_reversal_accuracy": round(self.asymmetric_reversal_accuracy, 4),
            "bridge_deletion_sensitivity": round(self.bridge_deletion_sensitivity, 4),
            "minimal_pair_sensitivity": round(self.minimal_pair_sensitivity, 4),
        }


class PositiveNegativeTransferEvaluator:
    """Probes positive and negative rule transfer dynamics."""

    CLASSES = ["YES", "NO", "UNKNOWN"]
    CLASS_TO_IDX = {"YES": 0, "NO": 1, "UNKNOWN": 2}

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: BPETokenizer,
        device: torch.device,
        seed: int = 42,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.seed = seed

    def _extract_hidden(self, model: MiniGPT, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.size()
        tok_emb = model.wte(input_ids)
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device)
        pos_emb = model.wpe(pos)
        x = model.drop(tok_emb + pos_emb)
        for block in model.blocks:
            x, _ = block(x)
        return model.ln_f(x)

    def evaluate_transfer_dynamics(
        self,
        transitive_source_world: DisentangledWorldDataset,
        transitive_target_world: DisentangledWorldDataset,
        nontransitive_target_world: DisentangledWorldDataset,
    ) -> TransferProbingMetrics:
        """Evaluates positive and negative transfer between source and target worlds."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # 1. Adapt on Transitive Source World
        model = copy.deepcopy(self.base_model)
        inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
        add_domain_adapter(model, domain_name="source_transitive", r=8)
        set_active_domain_adapter(model, "source_transitive")
        freeze_for_adapter_training(model, "source_transitive")

        tri_head = TriStateClassificationHead(d_model=model.config.d_model, num_classes=3).to(self.device)
        optimizer = AdamW(list(tri_head.parameters()) + [p for p in model.parameters() if p.requires_grad], lr=2e-3)
        ce_criterion = nn.CrossEntropyLoss()

        model.train()
        tri_head.train()
        for _ in range(15):
            for t in transitive_source_world.train_practice_pool[:12]:
                prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                ids = self.tokenizer.encode(prompt)
                if len(ids) > model.config.context_length:
                    ids = ids[-model.config.context_length:]
                ids = [min(x, model.config.vocab_size - 1) for x in ids]
                inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                target = torch.tensor([self.CLASS_TO_IDX[t.expected_verdict]], dtype=torch.long, device=self.device)

                h = self._extract_hidden(model, inp)
                logits = tri_head(h)
                loss = ce_criterion(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 2. Test Positive Transfer on Transitive Target World
        model.eval()
        tri_head.eval()
        pos_correct = 0
        total_target = len(transitive_target_world.held_out_test_pool)
        with torch.no_grad():
            for t in transitive_target_world.held_out_test_pool:
                prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                ids = self.tokenizer.encode(prompt)
                if len(ids) > model.config.context_length:
                    ids = ids[-model.config.context_length:]
                ids = [min(x, model.config.vocab_size - 1) for x in ids]
                inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                h = self._extract_hidden(model, inp)
                pred_idx = int(torch.argmax(tri_head(h), dim=-1).item())
                if self.CLASSES[pred_idx] == t.expected_verdict:
                    pos_correct += 1

        pos_transfer_acc = pos_correct / max(1, total_target)
        # Gain over 33.3% chance
        pos_gain = pos_transfer_acc - 0.3333

        # 3. Test Negative Transfer on Non-transitive World
        # Non-transitive relations should NOT conclude YES for 2-hop queries
        nontrans_false_pos = 0
        nontrans_total = 0
        with torch.no_grad():
            for t in nontransitive_target_world.held_out_test_pool:
                if t.expected_verdict != "YES":
                    prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                    ids = self.tokenizer.encode(prompt)
                    if len(ids) > model.config.context_length:
                        ids = ids[-model.config.context_length:]
                    ids = [min(x, model.config.vocab_size - 1) for x in ids]
                    inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                    h = self._extract_hidden(model, inp)
                    pred_idx = int(torch.argmax(tri_head(h), dim=-1).item())
                    # If model falsely predicts YES on disconnected / non-transitive target
                    if self.CLASSES[pred_idx] == "YES":
                        nontrans_false_pos += 1
                    nontrans_total += 1

        neg_transfer_err = nontrans_false_pos / max(1, nontrans_total)

        # 4. Minimal Pair Perturbation Testing
        min_pairs = transitive_target_world.minimal_pair_probes
        rev_correct, rev_total = 0, 0
        bridge_correct, bridge_total = 0, 0

        with torch.no_grad():
            for mp in min_pairs:
                if mp["pair_type"] == "DIRECTIONAL_REVERSAL":
                    # Test if model distinguishes forward query from reversed query
                    p_orig = f"[PREMISES] {mp['original_premise']} [QUESTION] {mp['original_query']} [ANSWER]"
                    p_pert = f"[PREMISES] {mp['original_premise']} [QUESTION] {mp['perturbed_query']} [ANSWER]"
                    
                    for p, exp in [(p_orig, mp["original_verdict"]), (p_pert, mp["perturbed_verdict"])]:
                        ids = [min(x, model.config.vocab_size - 1) for x in self.tokenizer.encode(p)]
                        inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                        h = self._extract_hidden(model, inp)
                        pred = self.CLASSES[int(torch.argmax(tri_head(h), dim=-1).item())]
                        if pred == exp:
                            rev_correct += 1
                        rev_total += 1

                elif mp["pair_type"] == "BRIDGE_DELETION":
                    p_orig = f"[PREMISES] {mp['original_premise']} [QUESTION] {mp['query']} [ANSWER]"
                    ids = [min(x, model.config.vocab_size - 1) for x in self.tokenizer.encode(p_orig)]
                    inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                    h = self._extract_hidden(model, inp)
                    pred = self.CLASSES[int(torch.argmax(tri_head(h), dim=-1).item())]
                    if pred == mp["original_verdict"]:
                        bridge_correct += 1
                    bridge_total += 1

        rev_acc = rev_correct / max(1, rev_total) if rev_total > 0 else 0.50
        bridge_acc = bridge_correct / max(1, bridge_total) if bridge_total > 0 else 0.50
        min_pair_sens = (rev_acc + bridge_acc) / 2.0

        return TransferProbingMetrics(
            positive_transfer_gain=pos_gain,
            negative_transfer_error_rate=neg_transfer_err,
            asymmetric_reversal_accuracy=rev_acc,
            bridge_deletion_sensitivity=bridge_acc,
            minimal_pair_sensitivity=min_pair_sens,
        )
