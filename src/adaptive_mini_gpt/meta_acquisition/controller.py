"""Milestone M82 Phase 1: Minimal Learned Acquisition Controller.

Implements a compact neural policy that inspects real-time training signals:
- Diagnostic State Features:
  1. probe_loss: Validation loss on dev batch.
  2. loss_slope: Progress slope (rate of loss decrease).
  3. prediction_entropy: Model prediction uncertainty.
  4. grad_norm: Gradient magnitude on probe batch.
  5. replay_loss: Retention probe loss on anchor replay tasks.
  6. progress_ratio: Current step / step budget.
  7. candidate_capacity_ratio: Current rank / 16.
  8. token_acc: Current dev token accuracy.

- STRICT INVARIANT: Absolutely NO task IDs, NO hidden formulas, NO sealed test leakage!

- Bounded Action Space:
  1. Initialization: FRESH_RANDOM (0) vs EXPERIENCE_PRIOR (1).
  2. Learning Rate: [1e-3, 3e-3, 6e-3, 1e-2].
  3. Capacity Rank: [Rank-4 (852p), Rank-8 (1432p), Rank-16 (2592p)].
  4. Stopping Decision: CONTINUE (0) vs STOP_AND_QUALIFY (1).

Total Parameters: ~1,835 (< 2,000 parameter strict budget).
"""

from __future__ import annotations

import copy
import hashlib
import logging
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InitChoice(IntEnum):
    """Categorical initialization choices."""

    FRESH_RANDOM = 0
    EXPERIENCE_PRIOR = 1


class LRChoice(IntEnum):
    """Categorical learning rate choices."""

    LR_1E_3 = 0  # 0.001
    LR_3E_3 = 1  # 0.003
    LR_6E_3 = 2  # 0.006
    LR_1E_2 = 3  # 0.010

    def to_float(self) -> float:
        mapping = {0: 0.001, 1: 0.003, 2: 0.006, 3: 0.010}
        return mapping[self.value]


class RankChoice(IntEnum):
    """Categorical rank capacity choices."""

    RANK_4 = 0   # 4 bottleneck dims (852 params)
    RANK_8 = 1   # 8 bottleneck dims (1,432 params)
    RANK_16 = 2  # 16 bottleneck dims (2,592 params)

    def to_rank(self) -> int:
        mapping = {0: 4, 1: 8, 2: 16}
        return mapping[self.value]

    def parameter_count(self) -> int:
        mapping = {0: 852, 1: 1432, 2: 2592}
        return mapping[self.value]


class StoppingChoice(IntEnum):
    """Categorical stopping decisions."""

    CONTINUE = 0
    STOP_AND_QUALIFY = 1


@dataclass(frozen=True)
class AcquisitionState:
    """State feature vector representing current learning dynamics."""

    probe_loss: float
    loss_slope: float
    prediction_entropy: float
    grad_norm: float
    replay_loss: float
    progress_ratio: float
    candidate_capacity_ratio: float
    token_acc: float

    def to_tensor(self) -> torch.Tensor:
        """Convert state features into a normalized 1D tensor [8]."""
        # Feature normalization and bounded clipping
        f0 = min(max(self.probe_loss / 3.0, 0.0), 2.0)
        f1 = min(max(self.loss_slope * 5.0, -1.0), 1.0)
        f2 = min(max(self.prediction_entropy / math.log(16.0), 0.0), 1.0)
        f3 = min(max(self.grad_norm / 5.0, 0.0), 2.0)
        f4 = min(max(self.replay_loss / 2.0, 0.0), 2.0)
        f5 = min(max(self.progress_ratio, 0.0), 1.0)
        f6 = min(max(self.candidate_capacity_ratio, 0.0), 1.0)
        f7 = min(max(self.token_acc, 0.0), 1.0)

        return torch.tensor([f0, f1, f2, f3, f4, f5, f6, f7], dtype=torch.float32)


@dataclass
class AcquisitionAction:
    """Action selected by the acquisition controller."""

    init_choice: InitChoice
    lr_choice: LRChoice
    rank_choice: RankChoice
    stopping_choice: StoppingChoice

    # Metadata for policy gradient updates
    log_prob: float = 0.0
    entropy: float = 0.0


class LearnedAcquisitionController(nn.Module):
    """Compact neural acquisition policy (< 2,000 parameters)."""

    def __init__(self, in_features: int = 8, hidden_dim: int = 32) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # Shared representation trunk
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Policy decision heads
        self.init_head = nn.Linear(hidden_dim, 2)
        self.lr_head = nn.Linear(hidden_dim, 4)
        self.rank_head = nn.Linear(hidden_dim, 3)
        self.stop_head = nn.Linear(hidden_dim, 2)

    def forward(self, state_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute logits for the 4 action dimensions."""
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        h = self.act(self.ln1(self.fc1(state_tensor)))
        h = self.act(self.ln2(self.fc2(h)))

        init_logits = self.init_head(h)
        lr_logits = self.lr_head(h)
        rank_logits = self.rank_head(h)
        stop_logits = self.stop_head(h)

        return init_logits, lr_logits, rank_logits, stop_logits

    def select_action(
        self,
        state: AcquisitionState,
        deterministic: bool = False,
        qual_em_so_far: float = 0.0,
        admission_threshold: float = 0.85,
    ) -> AcquisitionAction:
        """Sample or greedily select an acquisition action from policy."""
        self.eval()
        state_tensor = state.to_tensor().unsqueeze(0)
        with torch.no_grad():
            init_logits, lr_logits, rank_logits, stop_logits = self(state_tensor)

            init_dist = torch.distributions.Categorical(logits=init_logits)
            lr_dist = torch.distributions.Categorical(logits=lr_logits)
            rank_dist = torch.distributions.Categorical(logits=rank_logits)
            stop_dist = torch.distributions.Categorical(logits=stop_logits)

            if deterministic:
                init_idx = int(torch.argmax(init_logits, dim=-1).item())
                lr_idx = int(torch.argmax(lr_logits, dim=-1).item())
                rank_idx = int(torch.argmax(rank_logits, dim=-1).item())
                stop_idx = int(torch.argmax(stop_logits, dim=-1).item())
            else:
                init_idx = int(init_dist.sample().item())
                lr_idx = int(lr_dist.sample().item())
                rank_idx = int(rank_dist.sample().item())
                stop_idx = int(stop_dist.sample().item())

            # Policy safety rule: can only stop if qualification threshold met or exceeded
            if qual_em_so_far < admission_threshold:
                stop_idx = StoppingChoice.CONTINUE.value

            log_prob = (
                init_dist.log_prob(torch.tensor(init_idx)).item()
                + lr_dist.log_prob(torch.tensor(lr_idx)).item()
                + rank_dist.log_prob(torch.tensor(rank_idx)).item()
                + stop_dist.log_prob(torch.tensor(stop_idx)).item()
            )

            entropy = (
                init_dist.entropy().item()
                + lr_dist.entropy().item()
                + rank_dist.entropy().item()
                + stop_dist.entropy().item()
            )

        return AcquisitionAction(
            init_choice=InitChoice(init_idx),
            lr_choice=LRChoice(lr_idx),
            rank_choice=RankChoice(rank_idx),
            stopping_choice=StoppingChoice(stop_idx),
            log_prob=log_prob,
            entropy=entropy,
        )

    def evaluate_episode(
        self,
        states: list[AcquisitionState],
        actions: list[AcquisitionAction],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute batch log probabilities and entropy for policy gradient training."""
        state_tensors = torch.stack([s.to_tensor() for s in states])
        init_logits, lr_logits, rank_logits, stop_logits = self(state_tensors)

        init_dist = torch.distributions.Categorical(logits=init_logits)
        lr_dist = torch.distributions.Categorical(logits=lr_logits)
        rank_dist = torch.distributions.Categorical(logits=rank_logits)
        stop_dist = torch.distributions.Categorical(logits=stop_logits)

        init_targets = torch.tensor([a.init_choice.value for a in actions], device=state_tensors.device)
        lr_targets = torch.tensor([a.lr_choice.value for a in actions], device=state_tensors.device)
        rank_targets = torch.tensor([a.rank_choice.value for a in actions], device=state_tensors.device)
        stop_targets = torch.tensor([a.stopping_choice.value for a in actions], device=state_tensors.device)

        batch_log_probs = (
            init_dist.log_prob(init_targets)
            + lr_dist.log_prob(lr_targets)
            + rank_dist.log_prob(rank_targets)
            + stop_dist.log_prob(stop_targets)
        )

        batch_entropy = (
            init_dist.entropy()
            + lr_dist.entropy()
            + rank_dist.entropy()
            + stop_dist.entropy()
        )

        return batch_log_probs, batch_entropy

    def count_parameters(self) -> int:
        """Count total trainable parameters of the acquisition controller."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_parameter_hash(self) -> str:
        """Compute SHA-256 fingerprint of controller parameters."""
        hasher = hashlib.sha256()
        for p in self.parameters():
            hasher.update(p.detach().cpu().contiguous().numpy().tobytes())
        return hasher.hexdigest()[:16]
