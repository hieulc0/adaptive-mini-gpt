"""Two-Speed Neural Plasticity Manager for Milestone M100.

Implements:
1. Fast Plasticity:
   - Local adapter / modular skill updates (lr ~ 0.02 - 0.03).
   - Dense 1,071,872 backbone frozen.
   - Bounded parameter growth within capacity tiers.
2. Slow Plasticity (Selective Backbone Consolidation):
   - Gated unfreezing of selected backbone parameters (e.g. top transformer block or LayerNorm)
     when shared representation limitations or cross-subject interference are diagnosed.
   - Low learning rate (lr ~ 0.001 - 0.002).
   - Rehearsal anchor replay to prevent catastrophic forgetting.
   - Safe validation and rollback: Candidate update is rejected and reverted if anchor retention < 90%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F


class PlasticityMode(Enum):
    FAST_ADAPTER_ONLY = "fast_adapter_only"
    SLOW_SELECTIVE_BACKBONE = "slow_selective_backbone"
    EXPANDED_ADAPTER_FROZEN = "expanded_adapter_frozen"
    BROAD_BACKBONE_CONTROL = "broad_backbone_control"


@dataclass
class ConsolidationReport:
    """Outcome of a slow backbone consolidation episode."""

    plasticity_mode: PlasticityMode
    layers_unfrozen: list[str]
    trainable_backbone_params: int
    pre_consolidation_anchor_acc: float
    post_consolidation_anchor_acc: float
    target_task_acc: float
    promoted: bool
    rollback_triggered: bool
    loss_history: list[float] = field(default_factory=list)


class TwoSpeedPlasticityManager:
    """Orchestrates fast local skill adaptation and slow selective backbone consolidation."""

    def __init__(
        self,
        retention_threshold_pct: float = 90.0,
        slow_lr: float = 0.001,
        fast_lr: float = 0.025,
    ) -> None:
        self.retention_threshold_pct = retention_threshold_pct
        self.slow_lr = slow_lr
        self.fast_lr = fast_lr
        self.checkpoint_backup: dict[str, torch.Tensor] | None = None

    def configure_plasticity(
        self,
        student: Any,
        mode: PlasticityMode,
        target_layers: list[str] | None = None,
    ) -> list[str]:
        """Configure gradient requirements across backbone and adapters based on plasticity mode."""
        unfrozen_names: list[str] = []

        if mode == PlasticityMode.FAST_ADAPTER_ONLY:
            # Freeze all backbone parameters
            for _pn, p in student.backbone.named_parameters():
                p.requires_grad = False
            # Enable adapter gradients
            for _pn, p in student.adapter_modules.named_parameters():
                p.requires_grad = True
            for _pn, p in student.task_head.named_parameters():
                p.requires_grad = True

        elif mode == PlasticityMode.SLOW_SELECTIVE_BACKBONE:
            # Freeze entire backbone first
            for _pn, p in student.backbone.named_parameters():
                p.requires_grad = False

            # Selectively unfreeze designated layers (default: top transformer block and final norm)
            if target_layers is None:
                target_layers = ["blocks.3", "ln_f"]

            for pn, p in student.backbone.named_parameters():
                if any(tl in pn for tl in target_layers):
                    p.requires_grad = True
                    unfrozen_names.append(pn)

            # Adapters also participate in joint consolidation
            for _pn, p in student.adapter_modules.named_parameters():
                p.requires_grad = True
            for _pn, p in student.task_head.named_parameters():
                p.requires_grad = True

        elif mode == PlasticityMode.EXPANDED_ADAPTER_FROZEN:
            # Backbone strictly frozen
            for _pn, p in student.backbone.named_parameters():
                p.requires_grad = False
            for _pn, p in student.adapter_modules.named_parameters():
                p.requires_grad = True
            for _pn, p in student.task_head.named_parameters():
                p.requires_grad = True

        elif mode == PlasticityMode.BROAD_BACKBONE_CONTROL:
            # Diagnostic research control: unfreeze entire backbone
            for pn, p in student.backbone.named_parameters():
                p.requires_grad = True
                unfrozen_names.append(pn)
            for _pn, p in student.adapter_modules.named_parameters():
                p.requires_grad = True
            for _pn, p in student.task_head.named_parameters():
                p.requires_grad = True

        return unfrozen_names

    def backup_candidate_state(self, student: Any) -> None:
        """Create an in-memory bit-exact snapshot of the student state prior to consolidation."""
        self.checkpoint_backup = {
            k: v.detach().clone() for k, v in student.state_dict().items()
        }

    def rollback(self, student: Any) -> None:
        """Roll back student parameters to the pre-consolidation snapshot."""
        if self.checkpoint_backup is not None:
            student.load_state_dict(self.checkpoint_backup)

    def execute_slow_consolidation(
        self,
        student: Any,
        target_train_tokens: torch.Tensor,
        target_train_labels: torch.Tensor,
        anchor_tokens: torch.Tensor,
        anchor_labels: torch.Tensor,
        target_val_tokens: torch.Tensor,
        target_val_labels: torch.Tensor,
        steps: int = 15,
        target_layers: list[str] | None = None,
        anchor_replay_weight: float = 0.5,
    ) -> ConsolidationReport:
        """Execute rehearsal-protected selective backbone consolidation with validation and rollback."""
        # 1. Measure baseline anchor accuracy
        student.eval()
        with torch.no_grad():
            pre_anchor_logits = student.forward(anchor_tokens)
            pre_anchor_acc = (
                (pre_anchor_logits.argmax(dim=-1) == anchor_labels).float().mean().item() * 100.0
            )

        # 2. Backup qualified state before modifications
        self.backup_candidate_state(student)

        # 3. Configure selective backbone unfreezing
        unfrozen = self.configure_plasticity(
            student,
            PlasticityMode.SLOW_SELECTIVE_BACKBONE,
            target_layers=target_layers,
        )
        trainable_backbone_params = sum(
            p.numel() for p in student.backbone.parameters() if p.requires_grad
        )

        # 4. Joint Optimization with Low Learning Rate
        opt = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=self.slow_lr,
            weight_decay=1e-4,
        )

        loss_history = []
        student.train()
        for _ in range(steps):
            # Target loss
            tgt_logits = student.forward(target_train_tokens)
            loss_tgt = F.cross_entropy(tgt_logits, target_train_labels)

            # Rehearsal anchor loss
            anc_logits = student.forward(anchor_tokens)
            loss_anc = F.cross_entropy(anc_logits, anchor_labels)

            total_loss = (1.0 - anchor_replay_weight) * loss_tgt + anchor_replay_weight * loss_anc

            opt.zero_grad()
            total_loss.backward()
            opt.step()
            loss_history.append(float(total_loss.item()))

        # 5. Evaluate Post-Consolidation Capabilities
        student.eval()
        with torch.no_grad():
            post_anchor_logits = student.forward(anchor_tokens)
            post_anchor_acc = (
                (post_anchor_logits.argmax(dim=-1) == anchor_labels).float().mean().item() * 100.0
            )

            val_logits = student.forward(target_val_tokens)
            val_acc = (
                (val_logits.argmax(dim=-1) == target_val_labels).float().mean().item() * 100.0
            )

        # 6. Gated Promotion Decision
        promoted = post_anchor_acc >= self.retention_threshold_pct and val_acc >= 75.0
        rollback_triggered = False

        if not promoted:
            # Anchor retention dropped or validation failed: trigger immediate rollback!
            self.rollback(student)
            rollback_triggered = True

        # Re-freeze backbone post-consolidation
        self.configure_plasticity(student, PlasticityMode.FAST_ADAPTER_ONLY)

        return ConsolidationReport(
            plasticity_mode=PlasticityMode.SLOW_SELECTIVE_BACKBONE,
            layers_unfrozen=unfrozen,
            trainable_backbone_params=trainable_backbone_params,
            pre_consolidation_anchor_acc=pre_anchor_acc,
            post_consolidation_anchor_acc=post_anchor_acc if promoted else pre_anchor_acc,
            target_task_acc=val_acc if promoted else 0.0,
            promoted=promoted,
            rollback_triggered=rollback_triggered,
            loss_history=loss_history,
        )
