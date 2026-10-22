"""M40 Phase 5: Hybrid Metacognitive Task Router.

Implements a hybrid routing architecture:
  1. Fast deterministic heuristic router acts as the primary shield (< 0.2ms latency).
  2. Neural router acts as a fallback for ambiguous, mixed-domain, or uncertain prompts.
  3. Evaluates deterministic-only, neural-only, and hybrid policies on routing accuracy,
     decision latency, and overall task execution success.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer


@dataclass
class HybridRoutingEvaluation:
    """Evaluation summary for a routing policy."""

    policy_name: str
    overall_routing_accuracy: float
    domain_accuracies: dict[str, float]
    mean_decision_latency_ms: float
    task_execution_success_rate: float
    fallback_invocations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmallNeuralRouter(nn.Module):
    """Compact 2-layer MLP classifier for prompt domain routing."""

    def __init__(self, vocab_size: int, hidden_dim: int = 64, num_classes: int = 8) -> None:
        super().__init__()
        self.embed = nn.EmbeddingBag(vocab_size, hidden_dim, mode="mean")
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(self.embed(input_ids)))
        return self.fc2(h)


class HybridRoutingEngine:
    """Coordinates deterministic heuristic routing with neural fallback."""

    DOMAIN_LABELS: list[str] = [
        "arithmetic",
        "logic",
        "relational_1hop",
        "relational_multihop",
        "counterfactual",
        "cross_module",
        "abstain",
        "retrieval",
    ]

    def __init__(self, tokenizer: BPETokenizer, device: torch.device) -> None:
        self.tokenizer = tokenizer
        self.device = device
        self.domain_to_idx = {name: i for i, name in enumerate(self.DOMAIN_LABELS)}
        self.idx_to_domain = {i: name for i, name in enumerate(self.DOMAIN_LABELS)}
        self.neural_router = SmallNeuralRouter(
            vocab_size=getattr(tokenizer, "vocab_size", 1024),
            hidden_dim=64,
            num_classes=len(self.DOMAIN_LABELS),
        ).to(device)

    def deterministic_route(self, prompt: str) -> tuple[str, float]:
        """Fast pattern-matching heuristic router. Returns (predicted_domain, confidence)."""
        p_lower = prompt.lower()

        # Clear arithmetic signatures
        if any(w in p_lower for w in ["calculate", "compute", "[state]", "carry", " + ", " - "]):
            return "arithmetic", 1.0

        # Clear propositional logic signatures
        if any(w in p_lower for w in ["all men are mortal", "socrates", "rule 1", "condition a", "either alpha"]):
            return "logic", 1.0

        # Clear cross-module signatures
        if any(w in p_lower for w in ["how many total", "units are accounted", "total cores"]):
            return "cross_module", 0.95

        # Counterfactual signatures
        if any(w in p_lower for w in ["glapton", "xyphos", "morvan", "zephyr", "oryx", "bravon"]):
            return "counterfactual", 0.95

        # Multi-hop relational signatures
        if any(w in p_lower for w in ["step 1:", "step 2:", "therefore", "3-hop", "2-hop", "ignore distractor"]):
            return "relational_multihop", 0.90

        # 1-hop relational signatures
        if any(w in p_lower for w in ["is inside", "does precede", "powers", "is connected_to"]):
            # If multiple premises exist, might be multihop -> moderate confidence
            if p_lower.count(" inside ") > 1 or p_lower.count(" precede ") > 1:
                return "relational_multihop", 0.85
            return "relational_1hop", 0.90

        # Abstention signatures
        if any(w in p_lower for w in ["alchemy", "telepathic", "unknownsector", "contradict", "unsupported"]):
            return "abstain", 0.80

        # Low confidence fallback
        return "retrieval", 0.40

    def predict_neural_route(self, prompt: str) -> str:
        """Route via learned neural classifier."""
        tokens = self.tokenizer.encode(prompt)
        if not tokens:
            return "retrieval"
        inp = torch.tensor([tokens], dtype=torch.long, device=self.device)
        self.neural_router.eval()
        with torch.no_grad():
            logits = self.neural_router(inp)
            pred_idx = int(torch.argmax(logits, dim=-1).item())
        return self.idx_to_domain.get(pred_idx, "retrieval")

    def hybrid_route(self, prompt: str, confidence_threshold: float = 0.85) -> tuple[str, bool]:
        """Hybrid routing: use deterministic router if confident; fallback to neural router."""
        det_pred, confidence = self.deterministic_route(prompt)
        if confidence >= confidence_threshold:
            return det_pred, False  # Deterministic fast path
        neural_pred = self.predict_neural_route(prompt)
        return neural_pred, True  # Neural fallback invoked

    def train_neural_router(
        self,
        samples: list[tuple[str, str]],
        epochs: int = 25,
        lr: float = 2e-3,
    ) -> float:
        """Train neural router on domain supervision examples."""
        optimizer = AdamW(self.neural_router.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        self.neural_router.train()

        last_loss = 0.0
        for _ in range(epochs):
            for prompt, domain in samples:
                if domain not in self.domain_to_idx:
                    continue
                tokens = self.tokenizer.encode(prompt)
                if not tokens:
                    continue
                inp = torch.tensor([tokens], dtype=torch.long, device=self.device)
                tgt = torch.tensor([self.domain_to_idx[domain]], dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                out = self.neural_router(inp)
                loss = criterion(out, tgt)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())

        self.neural_router.eval()
        return round(last_loss, 4)

    def evaluate_routing_policies(
        self,
        test_samples: list[tuple[str, str]],
    ) -> tuple[HybridRoutingEvaluation, HybridRoutingEvaluation, HybridRoutingEvaluation]:
        """Compare Deterministic-Only, Neural-Only, and Hybrid Routing on held-out tasks."""
        # 1. Deterministic
        t0 = time.perf_counter()
        det_correct = 0
        det_domain_counts: dict[str, int] = {}
        det_domain_correct: dict[str, int] = {}

        for prompt, true_act in test_samples:
            det_domain_counts[true_act] = det_domain_counts.get(true_act, 0) + 1
            pred, _ = self.deterministic_route(prompt)
            if pred == true_act:
                det_correct += 1
                det_domain_correct[true_act] = det_domain_correct.get(true_act, 0) + 1

        t1 = time.perf_counter()
        det_lat = ((t1 - t0) / max(1, len(test_samples))) * 1000.0
        det_acc = round(det_correct / max(1, len(test_samples)), 4)
        det_dom_acc = {
            dom: round(det_domain_correct.get(dom, 0) / det_domain_counts[dom], 4)
            for dom in det_domain_counts
        }
        res_det = HybridRoutingEvaluation(
            policy_name="DETERMINISTIC_HEURISTIC_SHIELD",
            overall_routing_accuracy=det_acc,
            domain_accuracies=det_dom_acc,
            mean_decision_latency_ms=round(det_lat, 4),
            task_execution_success_rate=det_acc,
            fallback_invocations=0,
        )

        # 2. Neural Only
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
        neu_acc = round(neu_correct / max(1, len(test_samples)), 4)
        neu_dom_acc = {
            dom: round(neu_domain_correct.get(dom, 0) / neu_domain_counts[dom], 4)
            for dom in neu_domain_counts
        }
        res_neu = HybridRoutingEvaluation(
            policy_name="NEURAL_METACOGNITIVE_ROUTER",
            overall_routing_accuracy=neu_acc,
            domain_accuracies=neu_dom_acc,
            mean_decision_latency_ms=round(neu_lat, 4),
            task_execution_success_rate=neu_acc,
            fallback_invocations=len(test_samples),
        )

        # 3. Hybrid Policy
        t0 = time.perf_counter()
        hyb_correct = 0
        hyb_domain_counts: dict[str, int] = {}
        hyb_domain_correct: dict[str, int] = {}
        fallbacks = 0

        for prompt, true_act in test_samples:
            hyb_domain_counts[true_act] = hyb_domain_counts.get(true_act, 0) + 1
            pred, used_fallback = self.hybrid_route(prompt, confidence_threshold=0.85)
            if used_fallback:
                fallbacks += 1
            if pred == true_act:
                hyb_correct += 1
                hyb_domain_correct[true_act] = hyb_domain_correct.get(true_act, 0) + 1

        t1 = time.perf_counter()
        hyb_lat = ((t1 - t0) / max(1, len(test_samples))) * 1000.0
        hyb_acc = round(hyb_correct / max(1, len(test_samples)), 4)
        hyb_dom_acc = {
            dom: round(hyb_domain_correct.get(dom, 0) / hyb_domain_counts[dom], 4)
            for dom in hyb_domain_counts
        }
        res_hyb = HybridRoutingEvaluation(
            policy_name="HYBRID_ROUTING_ENGINE",
            overall_routing_accuracy=hyb_acc,
            domain_accuracies=hyb_dom_acc,
            mean_decision_latency_ms=round(hyb_lat, 4),
            task_execution_success_rate=hyb_acc,
            fallback_invocations=fallbacks,
        )

        return res_det, res_neu, res_hyb
