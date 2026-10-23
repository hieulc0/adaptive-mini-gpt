"""Phase 2: Safe Learning-Progress Controller for M61.

Implements a bounded, evidence-driven optimization controller with three discrete actions:
- STOP: Preserve the current qualified candidate when there is verified evidence of convergence or plateau.
- CONTINUE: Allocate one additional bounded block of optimization (10 optimizer steps).
- RESET: Reinitialize to a fresh adapter when the transferred source causes verified negative transfer.

Design Principles:
1. Block-wise Optimization: Operates in discrete 10-step blocks (up to max 40 steps).
2. Conservative Minimum: Requires a minimum training budget (e.g. 20 steps) before STOP can be triggered.
3. History as Advice, Not Authority: Historical kNN advice can suggest an initial target budget,
   but real-time validation progress checks can veto a premature STOP decision.
4. Convergence Verification: Requires consecutive validation checks with marginal gain < epsilon
   before declaring convergence.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ControllerAction(str, enum.Enum):
    STOP = "STOP"
    CONTINUE = "CONTINUE"
    RESET = "RESET"


@dataclass
class SafeControlDecision:
    """Actionable decision emitted by the safe learning controller at a block checkpoint."""
    action: ControllerAction
    step: int
    val_loss: float
    val_f1: float
    delta_loss: float
    delta_f1: float
    consecutive_plateaus: int
    negative_transfer_detected: bool
    history_advised_action: str
    history_vetoed_by_progress: bool
    rationale: str


class SafeLearningProgressController:
    """Bounded, progress-guarded controller governing adapter optimization."""

    def __init__(
        self,
        min_steps: int = 20,
        max_steps: int = 40,
        block_size: int = 10,
        loss_plateau_threshold: float = 0.02,
        f1_plateau_threshold: float = 0.02,
        required_plateau_blocks: int = 2,
        negative_transfer_loss_threshold: float = 1.10,
    ) -> None:
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.block_size = block_size
        self.loss_plateau_threshold = loss_plateau_threshold
        self.f1_plateau_threshold = f1_plateau_threshold
        self.required_plateau_blocks = required_plateau_blocks
        self.negative_transfer_loss_threshold = negative_transfer_loss_threshold

        self.loss_history: list[float] = []
        self.f1_history: list[float] = []
        self.consecutive_plateaus = 0
        self.has_reset = False

    def reset_state(self) -> None:
        """Reset tracking state for a new adaptation episode."""
        self.loss_history.clear()
        self.f1_history.clear()
        self.consecutive_plateaus = 0
        self.has_reset = False

    def record_initial_state(self, initial_loss: float, initial_f1: float) -> None:
        """Record state prior to any optimization steps."""
        self.loss_history.append(initial_loss)
        self.f1_history.append(initial_f1)

    def evaluate_checkpoint(
        self,
        current_step: int,
        val_loss: float,
        val_f1: float,
        history_recommended_action: str = "TRAIN",
        history_recommended_steps: int = 40,
    ) -> SafeControlDecision:
        """Evaluate learning trajectory at a 10-step boundary and decide next action."""
        prev_loss = self.loss_history[-1] if self.loss_history else val_loss
        prev_f1 = self.f1_history[-1] if self.f1_history else val_f1

        delta_loss = prev_loss - val_loss  # positive means loss decreased (improved)
        delta_f1 = val_f1 - prev_f1        # positive means f1 increased (improved)

        self.loss_history.append(val_loss)
        self.f1_history.append(val_f1)

        # 1. Hard Maximum Check
        if current_step >= self.max_steps:
            return SafeControlDecision(
                action=ControllerAction.STOP,
                step=current_step,
                val_loss=val_loss,
                val_f1=val_f1,
                delta_loss=delta_loss,
                delta_f1=delta_f1,
                consecutive_plateaus=self.consecutive_plateaus,
                negative_transfer_detected=False,
                history_advised_action=history_recommended_action,
                history_vetoed_by_progress=False,
                rationale=f"Reached maximum research budget ({self.max_steps} steps). Preserving candidate.",
            )

        # 2. Check Negative Transfer on Transferred Adapter
        # If initial steps show severe negative transfer (loss explosion and f1 collapse), trigger RESET once
        if (
            not self.has_reset
            and current_step <= 20
            and val_loss >= self.negative_transfer_loss_threshold
            and delta_loss < -0.15
        ):
            self.has_reset = True
            return SafeControlDecision(
                action=ControllerAction.RESET,
                step=current_step,
                val_loss=val_loss,
                val_f1=val_f1,
                delta_loss=delta_loss,
                delta_f1=delta_f1,
                consecutive_plateaus=0,
                negative_transfer_detected=True,
                history_advised_action=history_recommended_action,
                history_vetoed_by_progress=False,
                rationale=(
                    f"Severe negative transfer detected (val_loss={val_loss:.3f}, delta={delta_loss:.3f}). "
                    f"Reinitializing to clean fresh adapter."
                ),
            )

        # 3. Assess Convergence / Plateau
        is_plateau = (
            abs(delta_loss) < self.loss_plateau_threshold
            and abs(delta_f1) < self.f1_plateau_threshold
        )
        if is_plateau:
            self.consecutive_plateaus += 1
        else:
            self.consecutive_plateaus = 0

        # 4. Check for Safe Early Stopping
        # Conservative Minimum: never stop before min_steps unless performance is already near ceiling
        can_stop = False
        vetoed = False
        rationale = ""

        if val_f1 >= 0.95 and delta_loss >= 0.0:
            can_stop = True
            rationale = f"Task solved with near-perfect validation proficiency (f1={val_f1:.3f}). Stopping."
        elif current_step >= self.min_steps and self.consecutive_plateaus >= self.required_plateau_blocks:
            can_stop = True
            rationale = (
                f"Verified convergence over {self.consecutive_plateaus} consecutive blocks "
                f"(step {current_step} >= min {self.min_steps}). Stopping."
            )
        elif current_step < self.min_steps:
            # If history advised STOP (or small budget like 10 steps), but step < min_steps
            if history_recommended_action in ("STOP", "KEEP") or history_recommended_steps <= current_step:
                vetoed = True
                rationale = (
                    f"Historical advice recommended stopping at {history_recommended_steps} steps, "
                    f"but current progress safety check enforced minimum budget ({self.min_steps} steps). Continuing."
                )

        if can_stop:
            return SafeControlDecision(
                action=ControllerAction.STOP,
                step=current_step,
                val_loss=val_loss,
                val_f1=val_f1,
                delta_loss=delta_loss,
                delta_f1=delta_f1,
                consecutive_plateaus=self.consecutive_plateaus,
                negative_transfer_detected=False,
                history_advised_action=history_recommended_action,
                history_vetoed_by_progress=False,
                rationale=rationale,
            )

        # Otherwise, continue with next 10-step block
        if not rationale:
            rationale = (
                f"Continuous progress observed (delta_loss={delta_loss:+.3f}, delta_f1={delta_f1:+.3f}). "
                f"Allocating additional 10-step block."
            )

        return SafeControlDecision(
            action=ControllerAction.CONTINUE,
            step=current_step,
            val_loss=val_loss,
            val_f1=val_f1,
            delta_loss=delta_loss,
            delta_f1=delta_f1,
            consecutive_plateaus=self.consecutive_plateaus,
            negative_transfer_detected=False,
            history_advised_action=history_recommended_action,
            history_vetoed_by_progress=vetoed,
            rationale=rationale,
        )
