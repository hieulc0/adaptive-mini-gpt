"""Learning-action selection and decision policy controller (M29 Phase 3 & 8).

Implements explicit, auditable action selection across:
  ANSWER, VERIFY, RETRIEVE, REMEMBER, PRACTICE, TRAIN, ABSTAIN.

Compares gap-driven controller against baseline heuristics:
  - Always Answer / Retry
  - Always Retrieve
  - Fixed Scheduled Training
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from adaptive_mini_gpt.evaluation.multi_domain import DomainType
from adaptive_mini_gpt.interaction.knowledge_gap import (
    ErrorCategory,
    KnowledgeGapRecord,
    KnowledgeGapTracker,
)
from adaptive_mini_gpt.interaction.verified_memory import VerifiedMemoryBank


class LearningAction(str, Enum):
    """Actions the controller can select upon an interaction."""

    ANSWER = "answer"
    VERIFY = "verify"
    RETRIEVE = "retrieve"
    REMEMBER = "remember"
    PRACTICE = "practice"
    TRAIN = "train"
    ABSTAIN = "abstain"


@dataclass
class ActionDecision:
    """An auditable decision returned by an action selection policy."""

    action: LearningAction
    rationale: str
    target_domain: DomainType
    target_objective: str
    confidence: float
    requires_human_approval: bool = False
    policy_name: str = "gap_driven"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.value
        data["target_domain"] = self.target_domain.value
        return data


class BaseActionPolicy(ABC):
    """Abstract base class for action selection policies."""

    @abstractmethod
    def select_action(
        self,
        domain: DomainType,
        question: str,
        predicted_answer: str,
        confidence_score: float,
        is_verified: bool,
        verification_status: str,
        gap_tracker: KnowledgeGapTracker,
        memory_bank: VerifiedMemoryBank,
        has_approved_reference: bool = False,
        interaction_step: int = 1,
    ) -> ActionDecision:
        """Select an action given current interaction state."""
        pass


class AlwaysAnswerOrRetryPolicy(BaseActionPolicy):
    """Baseline policy: Always attempts direct answer or immediate blind retry."""

    def select_action(
        self,
        domain: DomainType,
        question: str,
        predicted_answer: str,
        confidence_score: float,
        is_verified: bool,
        verification_status: str,
        gap_tracker: KnowledgeGapTracker,
        memory_bank: VerifiedMemoryBank,
        has_approved_reference: bool = False,
        interaction_step: int = 1,
    ) -> ActionDecision:
        return ActionDecision(
            action=LearningAction.ANSWER,
            rationale="Baseline [Always-Answer]: Return raw prediction or retry without external learning.",
            target_domain=domain,
            target_objective=f"Answer {domain.value}",
            confidence=confidence_score,
            policy_name="always_answer",
        )


class AlwaysRetrievePolicy(BaseActionPolicy):
    """Baseline policy: Always attempts external retrieval regardless of confidence."""

    def select_action(
        self,
        domain: DomainType,
        question: str,
        predicted_answer: str,
        confidence_score: float,
        is_verified: bool,
        verification_status: str,
        gap_tracker: KnowledgeGapTracker,
        memory_bank: VerifiedMemoryBank,
        has_approved_reference: bool = False,
        interaction_step: int = 1,
    ) -> ActionDecision:
        action = LearningAction.RETRIEVE if has_approved_reference else LearningAction.ABSTAIN
        return ActionDecision(
            action=action,
            rationale="Baseline [Always-Retrieve]: Bypass neural weights and query external store.",
            target_domain=domain,
            target_objective=f"Retrieve for {domain.value}",
            confidence=confidence_score,
            policy_name="always_retrieve",
        )


class FixedScheduledTrainingPolicy(BaseActionPolicy):
    """Baseline policy: Triggers candidate GPU training on a fixed modulo interval."""

    def __init__(self, train_interval_steps: int = 5) -> None:
        self.train_interval_steps = train_interval_steps

    def select_action(
        self,
        domain: DomainType,
        question: str,
        predicted_answer: str,
        confidence_score: float,
        is_verified: bool,
        verification_status: str,
        gap_tracker: KnowledgeGapTracker,
        memory_bank: VerifiedMemoryBank,
        has_approved_reference: bool = False,
        interaction_step: int = 1,
    ) -> ActionDecision:
        if interaction_step % self.train_interval_steps == 0:
            return ActionDecision(
                action=LearningAction.TRAIN,
                rationale=f"Baseline [Fixed-Schedule]: Step {interaction_step} modulo {self.train_interval_steps} reached.",
                target_domain=domain,
                target_objective=f"Fixed training batch on {domain.value}",
                confidence=confidence_score,
                policy_name="fixed_scheduled_training",
            )
        return ActionDecision(
            action=LearningAction.ANSWER,
            rationale="Baseline [Fixed-Schedule]: Non-training step; answer directly.",
            target_domain=domain,
            target_objective=f"Answer {domain.value}",
            confidence=confidence_score,
            policy_name="fixed_scheduled_training",
        )


class InteractionActionSelector(BaseActionPolicy):
    """Adaptive gap-driven controller enforcing evidence boundaries and resource prudence."""

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        repeat_gap_training_threshold: int = 2,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.repeat_gap_training_threshold = repeat_gap_training_threshold

    def select_action(
        self,
        domain: DomainType,
        question: str,
        predicted_answer: str,
        confidence_score: float,
        is_verified: bool,
        verification_status: str,
        gap_tracker: KnowledgeGapTracker,
        memory_bank: VerifiedMemoryBank,
        has_approved_reference: bool = False,
        interaction_step: int = 1,
    ) -> ActionDecision:
        """Select evidence-grounded action based on verification status and gap history."""
        # 1. High confidence and verified: return answer directly
        if is_verified and verification_status == "PASS" and confidence_score >= self.confidence_threshold:
            return ActionDecision(
                action=LearningAction.ANSWER,
                rationale="Existing verified neural knowledge is sufficient with high confidence.",
                target_domain=domain,
                target_objective="Answer verified query",
                confidence=confidence_score,
                policy_name="gap_driven",
            )

        # 2. Check if answer is unverified but passes format: verify first
        if not is_verified and verification_status == "UNKNOWN":
            return ActionDecision(
                action=LearningAction.VERIFY,
                rationale="Unverified prediction requires validation before serving to user.",
                target_domain=domain,
                target_objective="Verify candidate prediction",
                confidence=confidence_score,
                policy_name="gap_driven",
            )

        # 3. Model abstains or expresses explicit uncertainty
        is_abstention = (
            predicted_answer.strip().lower() == "insufficient information."
            or verification_status == "ABSTAIN"
        )
        if is_abstention:
            if has_approved_reference or memory_bank.retrieve(domain, question) is not None:
                return ActionDecision(
                    action=LearningAction.RETRIEVE,
                    rationale="Model appropriately abstained on unfamiliar query; approved evidence is available to retrieve.",
                    target_domain=domain,
                    target_objective=f"Retrieve ground truth for {question[:30]}",
                    confidence=confidence_score,
                    policy_name="gap_driven",
                )
            return ActionDecision(
                action=LearningAction.ABSTAIN,
                rationale="No verified evidence exists in local repository; safe abstention enforced.",
                target_domain=domain,
                target_objective="Safe abstention",
                confidence=confidence_score,
                policy_name="gap_driven",
            )

        # 4. Model answered incorrectly or unsupported
        # Check gap frequency for this query/objective
        repeated_gaps = gap_tracker.get_repeated_gaps(min_occurrences=self.repeat_gap_training_threshold)
        domain_repeated = [g for g in repeated_gaps if g.domain == domain]

        if len(domain_repeated) > 0:
            # Repeated failure across this capability domain justifies training
            return ActionDecision(
                action=LearningAction.TRAIN,
                rationale=(
                    f"Repeated procedural failure detected ({len(domain_repeated)} persistent gaps "
                    f"in {domain.value}). Formulating bounded candidate GPU training."
                ),
                target_domain=domain,
                target_objective=f"Train isolated candidate on {domain.value} deficit",
                confidence=confidence_score,
                policy_name="gap_driven",
            )

        # 5. Single failure: If exercise generator or verified reference exists, practice or remember
        if has_approved_reference:
            return ActionDecision(
                action=LearningAction.RETRIEVE,
                rationale="First-time factual gap; retrieving approved local document to ground correction.",
                target_domain=domain,
                target_objective=f"Retrieve fact for {question[:30]}",
                confidence=confidence_score,
                policy_name="gap_driven",
            )

        if domain in (DomainType.ARITHMETIC, DomainType.LOGIC):
            return ActionDecision(
                action=LearningAction.PRACTICE,
                rationale=f"Procedural rule deficit in {domain.value}; generating targeted verified exercises.",
                target_domain=domain,
                target_objective=f"Practice {domain.value} rule",
                confidence=confidence_score,
                policy_name="gap_driven",
            )

        # Default fallback
        return ActionDecision(
            action=LearningAction.ABSTAIN,
            rationale="Unverified prediction failed audit and no approved evidence is reachable.",
            target_domain=domain,
            target_objective="Audit failure fallback",
            confidence=confidence_score,
            policy_name="gap_driven",
        )
