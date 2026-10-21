"""M33 Phase 4 & Phase 6: Learning Strategy Selection and Decision Auditing.

Defines and compares practice & training strategies:
  1. FIXED_PRACTICE: Standard fixed dataset without failure targeting or domain replay.
  2. FAILURE_TARGETED: Failure-targeted practice generated around diagnosed error states, no replay.
  3. FAILURE_TARGETED_WITH_REPLAY: Failure-targeted practice + balanced multi-domain replay
     (prior episode holdouts, elementary carry transitions, and pretraining prose).

Provides dynamic action selection with complete decision auditing and rationale logging.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class LearningStrategy(str, enum.Enum):
    """Available candidate training strategies."""

    FIXED_PRACTICE = "fixed_practice"
    FAILURE_TARGETED = "failure_targeted"
    FAILURE_TARGETED_WITH_REPLAY = "failure_targeted_with_replay"


@dataclass
class StrategyDecision:
    """Audit record explaining why a particular learning strategy was selected."""

    episode_id: str
    target_domain: str
    diagnosed_error: str
    active_prior_skills: list[str]
    selected_strategy: LearningStrategy
    decision_rationale: str
    estimated_tokens: int
    replay_domains_included: list[str] = field(default_factory=list)


class StrategySelector:
    """Autonomous selector that evaluates retention risk and selects training curriculum."""

    def __init__(self, mode: str = "adaptive") -> None:
        """Initialize selector.

        Args:
            mode: "adaptive" (gap & retention risk aware), or fixed policy string for ablation.
        """
        self.mode = mode
        self.decisions_history: list[StrategyDecision] = []

    def select_strategy(
        self,
        episode_id: str,
        target_domain: str,
        diagnosed_error: str,
        active_prior_skills: list[str],
        base_practice_count: int = 48,
    ) -> StrategyDecision:
        """Select the appropriate learning strategy based on error diagnosis and prior skills."""
        if self.mode == "fixed_only":
            selected = LearningStrategy.FIXED_PRACTICE
            rationale = "Ablation policy: forced fixed practice without targeting or replay."
            replay_domains = []
            tokens = base_practice_count * 64

        elif self.mode == "targeted_no_replay":
            selected = LearningStrategy.FAILURE_TARGETED
            rationale = "Ablation policy: failure-targeted practice without retention replay."
            replay_domains = []
            tokens = base_practice_count * 64

        else:
            # Adaptive Mode:
            # If there are prior active skills, replay is mandatory to prevent catastrophic forgetting.
            if len(active_prior_skills) > 0:
                selected = LearningStrategy.FAILURE_TARGETED_WITH_REPLAY
                replay_domains = list(active_prior_skills) + ["elementary_carry", "prose"]
                rationale = (
                    f"Diagnosed error '{diagnosed_error}' on domain '{target_domain}'. "
                    f"Lineage has {len(active_prior_skills)} active prior skills ({', '.join(active_prior_skills)}). "
                    "Selected FAILURE_TARGETED_WITH_REPLAY to eliminate diagnosed deficit while guarding against catastrophic forgetting."
                )
                tokens = (base_practice_count + len(replay_domains) * 24) * 64
            else:
                # First episode: balanced replay of pretrain prose and 200 elementary transitions
                selected = LearningStrategy.FAILURE_TARGETED_WITH_REPLAY
                replay_domains = ["elementary_carry", "prose"]
                rationale = (
                    f"Initial episode '{episode_id}'. Diagnosed error '{diagnosed_error}'. "
                    "Selected FAILURE_TARGETED_WITH_REPLAY with elementary transitions and prose replay to stabilize base representations."
                )
                tokens = (base_practice_count + 48) * 64

        decision = StrategyDecision(
            episode_id=episode_id,
            target_domain=target_domain,
            diagnosed_error=diagnosed_error,
            active_prior_skills=active_prior_skills,
            selected_strategy=selected,
            decision_rationale=rationale,
            estimated_tokens=tokens,
            replay_domains_included=replay_domains,
        )

        self.decisions_history.append(decision)
        logger.info(f"StrategySelector: Episode {episode_id} -> {selected.value} ({rationale})")
        return decision
