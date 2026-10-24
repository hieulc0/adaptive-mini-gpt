"""Phase 3: Minimal Experiment Proposal Mechanism.

Formulates structured, falsifiable research proposals:
- REUSE_EXISTING
- FRESH_ADAPT
- TRAIN_NEW_SKILL
- EVIDENCE_REPAIR
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class ResearchExperimentAction(str, enum.Enum):
    """The 4 bounded candidate learning experiments in M68."""
    REUSE_EXISTING = "REUSE_EXISTING"
    FRESH_ADAPT = "FRESH_ADAPT"
    TRAIN_NEW_SKILL = "TRAIN_NEW_SKILL"
    EVIDENCE_REPAIR = "EVIDENCE_REPAIR"


@dataclass
class ExperimentProposal:
    """A structured, self-contained candidate experiment proposal."""
    proposal_id: str
    action: ResearchExperimentAction
    target_env_id: str
    motivation: str
    hypothesis: str
    max_oracle_queries: int
    max_optimizer_steps: int
    matched_control_action: str
    min_validation_f1_gain: float
    max_active_parameters: int
    rollback_on_failure: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentProposalEngine:
    """Synthesizes bounded experiment proposals from task diagnostics."""

    def __init__(
        self,
        reuse_f1_threshold: float = 0.45,
        sprt_novelty_threshold: float = 2.0,
        evidence_repair_uncertainty_threshold: float = 0.50,
    ) -> None:
        self.reuse_f1_threshold = reuse_f1_threshold
        self.sprt_novelty_threshold = sprt_novelty_threshold
        self.evidence_repair_uncertainty_threshold = evidence_repair_uncertainty_threshold

    def formulate_proposal(
        self,
        target_env_id: str,
        best_reuse_f1: float,
        best_skill_id: str | None,
        sprt_llr: float,
        evidence_gap_ratio: float,
        is_known_family: bool = False,
    ) -> ExperimentProposal:
        """Formulate exactly one bounded candidate experiment proposal based on evidence."""
        # 1. Check if evidence is incomplete/corrupted (EVIDENCE_REPAIR)
        if evidence_gap_ratio >= self.evidence_repair_uncertainty_threshold:
            return ExperimentProposal(
                proposal_id=f"prop_ev_repair_{target_env_id}",
                action=ResearchExperimentAction.EVIDENCE_REPAIR,
                target_env_id=target_env_id,
                motivation=f"High observable evidence deficit (gap={evidence_gap_ratio:.2f})",
                hypothesis="Acquiring permitted factual observations will disambiguate relations without weight updates",
                max_oracle_queries=8,
                max_optimizer_steps=0,
                matched_control_action="REUSE_EXISTING",
                min_validation_f1_gain=0.05,
                max_active_parameters=0,
                metadata={"gap": evidence_gap_ratio},
            )

        # 2. Check if existing qualified skill is already proficient (REUSE_EXISTING)
        if is_known_family or best_reuse_f1 >= self.reuse_f1_threshold:
            return ExperimentProposal(
                proposal_id=f"prop_reuse_{target_env_id}",
                action=ResearchExperimentAction.REUSE_EXISTING,
                target_env_id=target_env_id,
                motivation=f"Existing skill '{best_skill_id}' demonstrates compatible proficiency (F1={best_reuse_f1:.4f})",
                hypothesis=f"Reusing qualified skill '{best_skill_id}' achieves high proficiency at zero parameter and training cost",
                max_oracle_queries=4,
                max_optimizer_steps=0,
                matched_control_action="ALWAYS_FRESH",
                min_validation_f1_gain=0.0,
                max_active_parameters=0,
                metadata={"skill_id": best_skill_id, "best_f1": best_reuse_f1},
            )

        # 3. Check for genuine structural rule novelty (TRAIN_NEW_SKILL)
        if sprt_llr >= self.sprt_novelty_threshold:
            return ExperimentProposal(
                proposal_id=f"prop_new_skill_{target_env_id}",
                action=ResearchExperimentAction.TRAIN_NEW_SKILL,
                target_env_id=target_env_id,
                motivation=f"Sequential Probability Ratio Test confirmed structural rule novelty (SPRT LLR={sprt_llr:.2f})",
                hypothesis="Acquiring an isolated rank-16 candidate skill will yield high transfer utility across future encounters",
                max_oracle_queries=32,
                max_optimizer_steps=40,
                matched_control_action="REUSE_EXISTING",
                min_validation_f1_gain=0.15,
                max_active_parameters=2371,
                metadata={"sprt_llr": sprt_llr},
            )

        # 4. Fallback: Moderate uncertainty on unfamiliar domain without confirmed rule novelty (FRESH_ADAPT)
        return ExperimentProposal(
            proposal_id=f"prop_fresh_adapt_{target_env_id}",
            action=ResearchExperimentAction.FRESH_ADAPT,
            target_env_id=target_env_id,
            motivation="Moderate uncertainty; structural novelty inconclusive. Testing temporary adaptation without storage commitment",
            hypothesis="Adapting fresh parameters improves current-task accuracy without incurring permanent storage overhead",
            max_oracle_queries=16,
            max_optimizer_steps=40,
            matched_control_action="REUSE_EXISTING",
            min_validation_f1_gain=0.05,
            max_active_parameters=2371,
            metadata={"best_f1": best_reuse_f1, "sprt_llr": sprt_llr},
        )
