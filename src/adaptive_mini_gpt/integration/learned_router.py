"""M39 Phase 4: Learned Adapter Routing and Task Integration.

Implements a neural metacognitive routing policy that dynamically classifies incoming queries
and routes execution to the appropriate module:
  - "arithmetic": routes to arithmetic adapter with canonical transition format.
  - "logic": routes to logic adapter.
  - "factual_{subject}": routes to domain-specific or multi-view factual adapter.
  - "retrieval": triggers verified retrieval when neural confidence is low.
  - "abstain": safely abstains on unsupported or contradictory premises.

Compares Neural Routing against Deterministic Heuristic Routing on identical test tasks.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer

logger = logging.getLogger("integration.learned_router")

ROUTING_ACTIONS = [
    "arithmetic",
    "logic",
    "factual_physics",
    "factual_biology",
    "factual_computing",
    "factual_history",
    "retrieval",
    "abstain",
]
ACTION_TO_IDX = {act: i for i, act in enumerate(ROUTING_ACTIONS)}
IDX_TO_ACTION = {i: act for i, act in enumerate(ROUTING_ACTIONS)}


@dataclass
class RoutingEvaluationResult:
    """Evaluation summary comparing neural vs deterministic routing."""

    policy_name: str
    overall_routing_accuracy: float
    domain_accuracies: dict[str, float]
    mean_decision_latency_ms: float
    task_execution_success_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NeuralTaskRouter(nn.Module):
    """Compact neural router classifying incoming task embeddings into module routing decisions."""

    def __init__(self, vocab_size: int = 1024, d_model: int = 64, n_actions: int = len(ROUTING_ACTIONS)) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, token_ids: Tensor) -> Tensor:
        """token_ids: (batch_size, seq_len) -> logits: (batch_size, n_actions)."""
        emb = self.embedding(token_ids)  # (b, s, d)
        pooled = emb.mean(dim=1)         # mean pooling across sequence length
        h = self.relu(self.fc1(pooled))
        logits = self.fc2(h)
        return logits


class MetacognitiveRoutingSystem:
    """Manages training and evaluation of deterministic and neural routing policies."""

    def __init__(self, tokenizer: BPETokenizer, device: torch.device) -> None:
        self.tokenizer = tokenizer
        self.device = device
        self.neural_router = NeuralTaskRouter(
            vocab_size=max(1024, tokenizer.vocab_size),
            d_model=64,
            n_actions=len(ROUTING_ACTIONS),
        ).to(device)

    @staticmethod
    def deterministic_route(prompt: str) -> str:
        """Rule-based routing heuristic based on surface cues and keywords."""
        p_lower = prompt.lower()
        if re.search(r"(\d+\s*[\+\-\*\/]\s*\d+)|(\bc_in\b)|(\bcarry\b)|(\bdigit\b)", p_lower):
            return "arithmetic"
        if re.search(r"(\bsocrates\b)|(\bmodus ponens\b)|(\bif condition\b)|(\bpremise\b)", p_lower):
            return "logic"
        if re.search(r"(\bforce\b)|(\bacceleration\b)|(\bnewton\b)|(\bkepler\b)|(\bmechanics\b)", p_lower):
            return "factual_physics"
        if re.search(r"(\batp\b)|(\bmitochondria\b)|(\borganelle\b)|(\bcell\b)|(\bchloroplast\b)", p_lower):
            return "factual_biology"
        if re.search(r"(\bturing\b)|(\bcomputing\b)|(\bshannon\b)|(\balgorithm\b)|(\bmachine\b)", p_lower):
            return "factual_computing"
        if re.search(r"(\bgutenberg\b)|(\bprinting\b)|(\bapollo\b)|(\bcentury\b)|(\byear\b)", p_lower):
            return "factual_history"
        if re.search(r"(\btransmute\b)|(\btelepathy\b)|(\batlantis\b)|(\bvictorian london\b)", p_lower):
            return "abstain"
        return "retrieval"

    def predict_neural_route(self, prompt: str) -> str:
        """Route prompt using the trained neural router."""
        self.neural_router.eval()
        tokens = self.tokenizer.encode(prompt)
        if not tokens:
            tokens = [0]
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.neural_router(ids)
            act_idx = int(torch.argmax(logits, dim=-1).item())
        return IDX_TO_ACTION.get(act_idx, "retrieval")

    def train_neural_router(
        self,
        routing_samples: list[tuple[str, str]],
        epochs: int = 20,
        lr: float = 1e-3,
    ) -> float:
        """Train neural router with cross-entropy loss on balanced routing curriculum."""
        self.neural_router.train()
        optimizer = torch.optim.AdamW(self.neural_router.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        data = []
        for prompt, target_act in routing_samples:
            toks = self.tokenizer.encode(prompt) or [0]
            act_idx = ACTION_TO_IDX.get(target_act, 0)
            data.append((
                torch.tensor([toks], dtype=torch.long, device=self.device),
                torch.tensor([act_idx], dtype=torch.long, device=self.device),
            ))

        final_loss = 0.0
        for ep in range(epochs):
            total_loss = 0.0
            for ids, lbl in data:
                optimizer.zero_grad()
                logits = self.neural_router(ids)
                loss = criterion(logits, lbl)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            final_loss = total_loss / max(1, len(data))

        self.neural_router.eval()
        return round(final_loss, 4)

    def evaluate_routing_policies(
        self,
        test_samples: list[tuple[str, str]],
    ) -> tuple[RoutingEvaluationResult, RoutingEvaluationResult]:
        """Evaluate both deterministic and neural routing policies on held-out tasks."""
        # 1. Deterministic
        t0 = time.perf_counter()
        det_correct = 0
        det_domain_counts: dict[str, int] = {}
        det_domain_correct: dict[str, int] = {}

        for prompt, true_act in test_samples:
            det_domain_counts[true_act] = det_domain_counts.get(true_act, 0) + 1
            pred = self.deterministic_route(prompt)
            if pred == true_act:
                det_correct += 1
                det_domain_correct[true_act] = det_domain_correct.get(true_act, 0) + 1

        t1 = time.perf_counter()
        det_lat = ((t1 - t0) / max(1, len(test_samples))) * 1000.0

        det_dom_acc = {
            dom: round(det_domain_correct.get(dom, 0) / det_domain_counts[dom], 4)
            for dom in det_domain_counts
        }
        det_acc = round(det_correct / max(1, len(test_samples)), 4)

        det_result = RoutingEvaluationResult(
            policy_name="DETERMINISTIC_HEURISTIC",
            overall_routing_accuracy=det_acc,
            domain_accuracies=det_dom_acc,
            mean_decision_latency_ms=round(det_lat, 4),
            task_execution_success_rate=det_acc,
        )

        # 2. Neural
        t0 = time.perf_counter()
        neu_correct = 0
        neu_domain_counts: dict[str, int] = {}
        neu_domain_correct: dict[str, int] = {}

        for prompt, true_act in test_samples:
            neu_domain_counts[true_act] = neu_domain_counts.get(true_act, 0) + 1
            pred = self.predict_neural_route(prompt)
            if pred == true_act:
                neu_correct += 1
                neu_domain_correct[true_act] = neu_domain_correct.get(true_act, 0) + 1

        t1 = time.perf_counter()
        neu_lat = ((t1 - t0) / max(1, len(test_samples))) * 1000.0

        neu_dom_acc = {
            dom: round(neu_domain_correct.get(dom, 0) / neu_domain_counts[dom], 4)
            for dom in neu_domain_counts
        }
        neu_acc = round(neu_correct / max(1, len(test_samples)), 4)

        neu_result = RoutingEvaluationResult(
            policy_name="NEURAL_METACOGNITIVE_ROUTER",
            overall_routing_accuracy=neu_acc,
            domain_accuracies=neu_dom_acc,
            mean_decision_latency_ms=round(neu_lat, 4),
            task_execution_success_rate=neu_acc,
        )

        return det_result, neu_result
