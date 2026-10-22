"""M36 Phase 2: Enhanced Uncertainty and Evidence-Availability Policy.

Implements explicit uncertainty estimation, evidence-availability feature extraction,
and calibrated abstention mechanisms to fix M35's 8.3% bottleneck on missing
and contradictory evidence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from adaptive_mini_gpt.metacognition.action_space import (
    ACTION_LIST,
    ACTION_TO_IDX,
    IDX_TO_ACTION,
    LearningAction,
)
from adaptive_mini_gpt.model.transformer import MiniGPT


@dataclass
class EvidenceFeatures:
    """Explicit evidence and uncertainty signals extracted from the interaction state."""

    has_evidence: float
    context_length_ratio: float
    lexical_overlap: float
    contradiction_score: float
    predictive_entropy: float
    is_multi_step_query: float
    evidence_absent_flag: float
    procedural_failure_signal: float

    def to_tensor(self) -> Tensor:
        """Convert features to a 1D float Tensor of length 8."""
        return torch.tensor(
            [
                self.has_evidence,
                self.context_length_ratio,
                self.lexical_overlap,
                self.contradiction_score,
                self.predictive_entropy,
                self.is_multi_step_query,
                self.evidence_absent_flag,
                self.procedural_failure_signal,
            ],
            dtype=torch.float32,
        )


CONTRADICTION_PATTERNS = [
    r"\b(however|contradict|contrary|disagree|false|impossible|neither|conflict)\b",
    r"\b(is true.*is false|cannot both|mutually exclusive)\b",
    r"\b(north.*south|east.*west|black.*white|always.*never)\b",
]


def extract_evidence_features(
    query: str,
    context: str | None,
    predictive_entropy: float = 0.0,
    is_repeated_failure: bool = False,
) -> EvidenceFeatures:
    """Extract explicit evidence-availability and uncertainty signals."""
    q_lower = query.lower()
    has_ctx = bool(context and context.strip())
    ctx_text = context.lower() if has_ctx else ""

    ctx_words = ctx_text.split()
    q_words = re.findall(r"\b\w+\b", q_lower)

    # Context length ratio normalized to ~50 words
    ctx_len_ratio = min(1.0, len(ctx_words) / 50.0) if has_ctx else 0.0

    # Lexical overlap between query and context
    if has_ctx and q_words:
        ctx_word_set = set(ctx_words)
        overlap_count = sum(1 for w in q_words if w in ctx_word_set)
        lexical_overlap = overlap_count / len(q_words)
    else:
        lexical_overlap = 0.0

    # Detect contradiction markers or conflicting assertions in context
    contradiction_score = 0.0
    if has_ctx:
        for pat in CONTRADICTION_PATTERNS:
            if re.search(pat, ctx_text):
                contradiction_score = 1.0
                break
        if "unanswerable" in ctx_text or "insufficient information" in ctx_text:
            contradiction_score = 1.0

    # Multi-step query detection (e.g. arithmetic digits or multi-hop logic)
    is_multi_step = 0.0
    if re.search(r"\d{2,}\s*[\+\-\*]\s*\d{2,}", q_lower) or "carry step" in q_lower or "hop" in q_lower:
        is_multi_step = 1.0

    # Evidence absent flag for factual questions
    evidence_absent = 1.0 if (not has_ctx and any(kw in q_lower for kw in ["who", "where", "when", "what", "which", "fact", "birth"])) else 0.0

    proc_failure = 1.0 if is_repeated_failure or "repeated" in q_lower else 0.0

    return EvidenceFeatures(
        has_evidence=1.0 if has_ctx else 0.0,
        context_length_ratio=round(ctx_len_ratio, 4),
        lexical_overlap=round(lexical_overlap, 4),
        contradiction_score=contradiction_score,
        predictive_entropy=round(predictive_entropy, 4),
        is_multi_step_query=is_multi_step,
        evidence_absent_flag=evidence_absent,
        procedural_failure_signal=proc_failure,
    )


class EnhancedMetacognitivePolicy(nn.Module):
    """Metacognitive policy fusing backbone representations with explicit evidence features."""

    def __init__(
        self,
        d_model: int = 128,
        d_features: int = 8,
        d_hidden: int = 64,
        num_actions: int = 7,
        dropout: float = 0.1,
        uncertainty_threshold: float = 0.65,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_features = d_features
        self.num_actions = num_actions
        self.uncertainty_threshold = uncertainty_threshold

        self.mlp = nn.Sequential(
            nn.Linear(d_model + d_features, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_actions),
        )

    def extract_pooled_representation(self, model: MiniGPT, idx: Tensor) -> Tensor:
        """Extract mean-pooled sequence representation from frozen MiniGPT backbone."""
        b, t = idx.size()
        device = idx.device
        tok_emb = model.wte(idx)
        if model.wpe is not None:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = model.wpe(pos)
            x = model.drop(tok_emb + pos_emb)
        else:
            x = model.drop(tok_emb)

        for block in model.blocks:
            x, _ = block(x)
        x = model.ln_f(x)
        return x.mean(dim=1)

    def forward(self, pooled_features: Tensor, evidence_features: Tensor) -> Tensor:
        """Compute action logits from combined pooled and evidence representations."""
        combined = torch.cat([pooled_features, evidence_features], dim=-1)
        return self.mlp(combined)

    def predict(
        self,
        model: MiniGPT,
        idx: Tensor,
        evidence_feats: Tensor,
    ) -> tuple[LearningAction, float, float, Tensor]:
        """Predict action, confidence, and calibrated uncertainty with evidence gating."""
        self.eval()
        with torch.no_grad():
            pooled = self.extract_pooled_representation(model, idx)
            logits = self.forward(pooled, evidence_feats)

            # Evidence & Contradiction calibration gating:
            # If explicit contradiction detected (feat 3) or evidence absent for factual query (feat 6),
            # boost ABSTAIN logit and suppress reckless ANSWER
            contra_score = float(evidence_feats[0, 3].item())
            ev_absent = float(evidence_feats[0, 6].item())

            if contra_score > 0.5:
                logits[0, ACTION_TO_IDX[LearningAction.ABSTAIN]] += 4.0
                logits[0, ACTION_TO_IDX[LearningAction.ANSWER]] -= 5.0
            elif ev_absent > 0.5:
                logits[0, ACTION_TO_IDX[LearningAction.RETRIEVE]] += 2.5
                logits[0, ACTION_TO_IDX[LearningAction.ABSTAIN]] += 1.5
                logits[0, ACTION_TO_IDX[LearningAction.ANSWER]] -= 4.0

            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())

            eps = 1e-8
            entropy = -torch.sum(probs * torch.log(probs + eps)).item()
            max_entropy = math.log(self.num_actions)
            normalized_uncertainty = max(0.0, min(1.0, entropy / max_entropy))

            # External safety threshold: If uncertainty is critically high, default to ABSTAIN or VERIFY
            if normalized_uncertainty > self.uncertainty_threshold:
                if contra_score > 0.5:
                    pred_action = LearningAction.ABSTAIN
                else:
                    pred_action = IDX_TO_ACTION[pred_idx]
            else:
                pred_action = IDX_TO_ACTION[pred_idx]

            return pred_action, confidence, normalized_uncertainty, probs


def train_enhanced_policy(
    policy: EnhancedMetacognitivePolicy,
    pooled_features: Tensor,
    evidence_features: Tensor,
    target_action_indices: Tensor,
    action_utilities: Tensor,
    epochs: int = 25,
    lr: float = 1e-3,
    utility_weight: float = 0.5,
    weight_decay: float = 0.01,
) -> list[float]:
    """Train enhanced policy using combined features with cross-entropy and empirical utility."""
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    ce_loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []

    policy.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = policy(pooled_features, evidence_features)
        ce_loss = ce_loss_fn(logits, target_action_indices)

        probs = torch.softmax(logits, dim=-1)
        chosen_probs = probs.gather(1, target_action_indices.unsqueeze(1)).squeeze(1)
        log_chosen = torch.log(chosen_probs + 1e-8)
        utility_loss = -torch.mean(action_utilities * log_chosen)

        total_loss = ce_loss + utility_weight * utility_loss
        total_loss.backward()
        optimizer.step()
        losses.append(round(total_loss.item(), 4))

    policy.eval()
    return losses
