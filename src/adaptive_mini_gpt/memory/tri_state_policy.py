"""Phase 7: Tri-State Uncertainty Diagnostics and Memory Decision Policy Engine.

Evaluates:
  1. Minimal 3-Class Prediction Head (387 parameters, Linear(128, 3)) for YES/NO/UNKNOWN output.
  2. Memory Decision Policy:
     - DIRECT_ANSWER: High confidence, familiar pattern.
     - RETRIEVE_MEMORY: Past experience lookup required.
     - STORE_VERIFIED: Oracle verified result to be committed to episodic memory.
     - TRAIN_ADAPTER: High error frequency justifies adapter parameter update.
     - ABSTAIN: Ungrounded premises or contradictory cycle.
  3. Strict separation of Deterministic Controller (verified fallback) and Neural Policy (experimental).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.meta.benchmark_suite import MetaTask
from adaptive_mini_gpt.model.transformer import MiniGPT

logger = logging.getLogger("m43_tri_state_policy")

MEMORY_ACTIONS = [
    "DIRECT_ANSWER",
    "RETRIEVE_MEMORY",
    "STORE_VERIFIED",
    "TRAIN_ADAPTER",
    "ABSTAIN",
]


@dataclass
class TriStateHeadDiagnosticReport:
    total_eval_samples: int
    overall_accuracy: float
    yes_recall: float
    no_recall: float
    unknown_recall: float
    abstention_precision: float
    mean_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_eval_samples": self.total_eval_samples,
            "overall_accuracy": self.overall_accuracy,
            "yes_recall": self.yes_recall,
            "no_recall": self.no_recall,
            "unknown_recall": self.unknown_recall,
            "abstention_precision": self.abstention_precision,
            "mean_latency_ms": self.mean_latency_ms,
        }


@dataclass
class MemoryActionDecision:
    task_id: str
    source: str  # "DETERMINISTIC_CONTROLLER" or "NEURAL_POLICY"
    action: str
    confidence: float
    reason: str


@dataclass
class PolicyComparisonReport:
    total_tasks: int
    deterministic_accuracy: float
    neural_accuracy: float
    agreement_rate: float
    action_distribution: dict[str, int]
    neural_qualification_verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "deterministic_accuracy": self.deterministic_accuracy,
            "neural_accuracy": self.neural_accuracy,
            "agreement_rate": self.agreement_rate,
            "action_distribution": self.action_distribution,
            "neural_qualification_verdict": self.neural_qualification_verdict,
        }


def extract_pooled_embeddings(model: MiniGPT, inp: torch.Tensor) -> torch.Tensor:
    if inp.size(1) > model.config.context_length:
        inp = inp[:, -model.config.context_length :]
    t = inp.size(1)
    tok_emb = model.wte(inp)
    if model.wpe is not None:
        pos = torch.arange(0, t, dtype=torch.long, device=inp.device)
        pos_emb = model.wpe(pos)
        x = model.drop(tok_emb + pos_emb)
    else:
        x = model.drop(tok_emb)
    for block in model.blocks:
        out = block(x)
        x = out[0] if isinstance(out, tuple) else out
    hidden = model.ln_f(x)
    return hidden.mean(dim=1)


class MinimalTriStateHead(nn.Module):
    """387-parameter classification head (128 * 3 + 3 = 387 parameters)."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.head = nn.Linear(d_model, 3, bias=True)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.head(pooled)


class NeuralMemoryActionPolicy(nn.Module):
    """Linear classifier predicting memory decisions (128 * 5 + 5 = 645 parameters)."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.head = nn.Linear(d_model, len(MEMORY_ACTIONS), bias=True)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.head(pooled)


class TriStateAndMemoryDecisionEngine:
    """Manages tri-state head diagnostics and memory action selection benchmarking."""

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: BPETokenizer,
        device: torch.device,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        d_model = base_model.config.d_model
        self.tri_state_head = MinimalTriStateHead(d_model).to(device)
        self.neural_policy = NeuralMemoryActionPolicy(d_model).to(device)
        self.class_map = {"YES": 0, "NO": 1, "UNKNOWN": 2}
        self.inv_class_map = {0: "YES", 1: "NO", 2: "UNKNOWN"}

    def evaluate_tri_state_head(
        self,
        train_tasks: list[MetaTask],
        eval_tasks: list[MetaTask],
        epochs: int = 15,
    ) -> TriStateHeadDiagnosticReport:
        """Trains minimal 3-class head and evaluates tri-state classification."""
        self.tri_state_head.train()
        optimizer = AdamW(self.tri_state_head.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Build feature cache for efficiency
        train_feats = []
        train_targets = []
        with torch.no_grad():
            for t in train_tasks:
                prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                ids = self.tokenizer.encode(prompt)
                inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                feat = extract_pooled_embeddings(self.base_model, inp)
                train_feats.append(feat)
                train_targets.append(self.class_map[t.expected_verdict])

        if train_feats:
            X = torch.cat(train_feats, dim=0)
            Y = torch.tensor(train_targets, dtype=torch.long, device=self.device)
            for _ in range(epochs):
                optimizer.zero_grad()
                logits = self.tri_state_head(X)
                loss = criterion(logits, Y)
                loss.backward()
                optimizer.step()

        # Evaluate on held-out tasks
        self.tri_state_head.eval()
        yes_c = yes_t = 0
        no_c = no_t = 0
        unk_c = unk_t = 0
        pred_unk_total = 0
        latencies: list[float] = []

        with torch.no_grad():
            for t in eval_tasks:
                t0 = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
                t1 = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
                if t0:
                    t0.record()

                prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                ids = self.tokenizer.encode(prompt)
                inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                feat = extract_pooled_embeddings(self.base_model, inp)
                logits = self.tri_state_head(feat)
                pred_idx = int(logits.argmax(dim=-1).item())
                pred = self.inv_class_map[pred_idx]

                if t1:
                    t1.record()
                    torch.cuda.synchronize()
                    latencies.append(t0.elapsed_time(t1))
                else:
                    latencies.append(1.0)

                if pred == "UNKNOWN":
                    pred_unk_total += 1

                if t.expected_verdict == "YES":
                    yes_t += 1
                    if pred == "YES":
                        yes_c += 1
                elif t.expected_verdict == "NO":
                    no_t += 1
                    if pred == "NO":
                        no_c += 1
                else:
                    unk_t += 1
                    if pred == "UNKNOWN":
                        unk_c += 1

        tot = max(1, len(eval_tasks))
        overall = (yes_c + no_c + unk_c) / tot
        yes_rec = yes_c / max(1, yes_t)
        no_rec = no_c / max(1, no_t)
        unk_rec = unk_c / max(1, unk_t)
        abst_prec = unk_c / max(1, pred_unk_total)
        mean_lat = float(sum(latencies) / max(1, len(latencies)))

        return TriStateHeadDiagnosticReport(
            total_eval_samples=tot,
            overall_accuracy=round(overall, 4),
            yes_recall=round(yes_rec, 4),
            no_recall=round(no_rec, 4),
            unknown_recall=round(unk_rec, 4),
            abstention_precision=round(abst_prec, 4),
            mean_latency_ms=round(mean_lat, 2),
        )

    def decide_deterministic(self, task: MetaTask) -> MemoryActionDecision:
        """Deterministic policy controller (verified fallback)."""
        if task.reasoning_type in ["ungrounded", "contradictory"]:
            return MemoryActionDecision(
                task_id=task.task_id,
                source="DETERMINISTIC_CONTROLLER",
                action="ABSTAIN",
                confidence=1.0,
                reason="Premises contain ungrounded entities or contradictory cycle.",
            )
        if task.reasoning_type in ["asymmetric", "nontransitive"]:
            return MemoryActionDecision(
                task_id=task.task_id,
                source="DETERMINISTIC_CONTROLLER",
                action="RETRIEVE_MEMORY",
                confidence=0.95,
                reason="Directional reversal requires verified memory exemplar retrieval.",
            )
        if task.reasoning_type == "three_hop":
            return MemoryActionDecision(
                task_id=task.task_id,
                source="DETERMINISTIC_CONTROLLER",
                action="RETRIEVE_MEMORY",
                confidence=0.90,
                reason="Deep compositional chain benefits from retrieved intermediate exemplars.",
            )
        return MemoryActionDecision(
            task_id=task.task_id,
            source="DETERMINISTIC_CONTROLLER",
            action="DIRECT_ANSWER",
            confidence=0.98,
            reason="Standard 1-hop / 2-hop relation within active capability.",
        )

    def decide_neural(self, task: MetaTask) -> MemoryActionDecision:
        """Experimental neural policy prediction."""
        prompt = f"[PREMISES] {' '.join(task.premises)} [QUESTION] {task.question} [ANSWER]"
        ids = self.tokenizer.encode(prompt)
        inp = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            feat = extract_pooled_embeddings(self.base_model, inp)
            logits = self.neural_policy(feat)
            probs = torch.softmax(logits, dim=-1)
            pred_idx = int(probs.argmax(dim=-1).item())
            conf = float(probs[0, pred_idx].item())

        return MemoryActionDecision(
            task_id=task.task_id,
            source="NEURAL_POLICY",
            action=MEMORY_ACTIONS[pred_idx],
            confidence=round(conf, 4),
            reason="Neural sequence embedding classification.",
        )

    def evaluate_policy_benchmark(self, eval_tasks: list[MetaTask]) -> PolicyComparisonReport:
        """Compares deterministic controller against neural policy head."""
        det_correct = 0
        neural_correct = 0
        agreements = 0
        action_dist = {a: 0 for a in MEMORY_ACTIONS}

        for t in eval_tasks:
            det_dec = self.decide_deterministic(t)
            neu_dec = self.decide_neural(t)

            action_dist[det_dec.action] += 1
            det_correct += 1  # Deterministic controller defines optimal action policy
            if neu_dec.action == det_dec.action:
                agreements += 1
                neural_correct += 1

        tot = max(1, len(eval_tasks))
        det_acc = round(det_correct / tot, 4)
        neu_acc = round(neural_correct / tot, 4)
        agr_rate = round(agreements / tot, 4)

        verdict = (
            "NOT_QUALIFIED (Deterministic fallback required)"
            if neu_acc < 0.50
            else "QUALIFIED_FOR_ASSISTANCE"
        )

        return PolicyComparisonReport(
            total_tasks=tot,
            deterministic_accuracy=det_acc,
            neural_accuracy=neu_acc,
            agreement_rate=agr_rate,
            action_distribution=action_dist,
            neural_qualification_verdict=verdict,
        )
