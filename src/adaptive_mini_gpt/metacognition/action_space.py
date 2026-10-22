"""M35: Metacognitive Action Vocabulary and Action-Outcome Grounding.

Defines the 7-action metacognitive vocabulary:
  1. ANSWER: Direct generative answer (known facts, simple queries).
  2. VERIFY: Evaluate candidate response against rules / self-consistency check.
  3. RETRIEVE: Query external verified memory bank / documents.
  4. RECURRENT_EXECUTION: Execute procedural multi-step transitions (multi-digit addition, logic chains).
  5. PRACTICE: Generate synthetic failure-targeted practice problems without updating weights.
  6. TRAIN_CANDIDATE: Initiate bounded isolated adapter candidate training on accumulated practice.
  7. ABSTAIN: Decline out-of-distribution, malformed, unanswerable, or contradictory queries.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass
from typing import Any


class LearningAction(str, enum.Enum):
    """The 7 metacognitive actions supported by Adaptive Mini-GPT."""

    ANSWER = "ANSWER"
    VERIFY = "VERIFY"
    RETRIEVE = "RETRIEVE"
    RECURRENT_EXECUTION = "RECURRENT_EXECUTION"
    PRACTICE = "PRACTICE"
    TRAIN_CANDIDATE = "TRAIN_CANDIDATE"
    ABSTAIN = "ABSTAIN"


ACTION_LIST = list(LearningAction)
ACTION_TO_IDX = {act: idx for idx, act in enumerate(ACTION_LIST)}
IDX_TO_ACTION = {idx: act for idx, act in enumerate(ACTION_LIST)}


@dataclass
class ActionOutcomeRecord:
    """Empirical audit record of an action taken on a specific task."""

    task_query: str
    available_evidence: str | None
    active_adapters: list[str]
    candidate_action: LearningAction
    verified_outcome: str  # "SUCCESS", "INCORRECT", "SAFE_ABSTENTION", "FALSE_ABSTENTION", "UNNECESSARY_TRAINING"
    latency_ms: float
    compute_cost: float
    task_solved: bool
    fresh_accuracy_gain: float = 0.0
    skill_regression_detected: bool = False
    utility_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["candidate_action"] = self.candidate_action.value
        return d


# Cost schedule for actions (in relative compute units)
ACTION_COMPUTE_COSTS: dict[LearningAction, float] = {
    LearningAction.ANSWER: 0.01,
    LearningAction.VERIFY: 0.03,
    LearningAction.RETRIEVE: 0.05,
    LearningAction.RECURRENT_EXECUTION: 0.10,
    LearningAction.PRACTICE: 0.25,
    LearningAction.TRAIN_CANDIDATE: 1.00,
    LearningAction.ABSTAIN: 0.005,
}


def compute_action_utility(
    action: LearningAction,
    task_solved: bool,
    is_ground_truth_unanswerable: bool = False,
    is_repeated_procedural_failure: bool = False,
    fresh_accuracy_gain: float = 0.0,
    skill_regression_detected: bool = False,
) -> float:
    """Compute independent empirical utility for an action taken on a task.

    Utility explicitly rewards verified task success and safe abstention while
    penalizing errors, catastrophic skill regression, and wasteful training actions.
    """
    cost = ACTION_COMPUTE_COSTS.get(action, 0.05)

    if skill_regression_detected:
        return -2.5

    # Safe vs False Abstention
    if action == LearningAction.ABSTAIN:
        if is_ground_truth_unanswerable:
            return 1.0 - cost  # High reward for appropriate abstention
        return -0.8  # Penalty for refusing an answerable task

    # If query is actually unanswerable and model attempts an execution action
    if is_ground_truth_unanswerable:
        return -1.0 - cost

    # Training decisions
    if action == LearningAction.TRAIN_CANDIDATE:
        if is_repeated_procedural_failure:
            # Training is justified if it yields fresh accuracy gain
            net_gain_bonus = max(0.0, fresh_accuracy_gain) * 2.0
            return 1.0 + net_gain_bonus - cost
        else:
            # Premature/unnecessary training on easily answerable or retrieved tasks
            return -0.7 - cost

    if action == LearningAction.PRACTICE:
        if is_repeated_procedural_failure:
            return 0.8 - cost
        return -0.2 - cost

    # Immediate task solving actions
    if task_solved:
        return 1.0 - (cost * 0.5)

    # Failed immediate attempt
    return -1.0 - cost
