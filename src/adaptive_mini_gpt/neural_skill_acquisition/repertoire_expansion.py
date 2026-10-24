"""Milestone M80 Phase 4: Dynamic Repertoire Expansion and Plan Space Updating.

Manages:
1. Dynamic registration of newly acquired neural skills into the active primitive repertoire.
2. Expanding the combinatorial plan hypothesis space:
   - 3 primitives -> 10 candidate plans (1 abstain, 3 single, 6 two-stage compositions)
   - 4 primitives -> 17 candidate plans (1 abstain, 4 single, 12 two-stage compositions)
3. Preserving immutable base model weights and checkpoint lineage.
4. Bit-exact rollback removing newly registered primitives if validation fails.
"""

from __future__ import annotations

import copy
import logging

from adaptive_mini_gpt.neural_plan_discovery.plan_space import CandidatePlan, PlanType
from adaptive_mini_gpt.neural_skill_composition.primitive_skills import (
    QualifiedPrimitiveRecord,
)

logger = logging.getLogger(__name__)


class DynamicRepertoireManager:
    """Manages active neural primitives and dynamic candidate plan space."""

    def __init__(self, initial_primitives: dict[str, QualifiedPrimitiveRecord]):
        self._primitives: dict[str, QualifiedPrimitiveRecord] = copy.deepcopy(initial_primitives)
        self._history: list[str] = list(initial_primitives.keys())

    @property
    def primitives(self) -> dict[str, QualifiedPrimitiveRecord]:
        return self._primitives

    def register_new_skill(self, skill_name: str, record: QualifiedPrimitiveRecord) -> None:
        """Register a newly qualified neural primitive into the active repertoire."""
        if skill_name in self._primitives:
            raise ValueError(f"Skill {skill_name} is already registered in repertoire!")
        self._primitives[skill_name] = record
        self._history.append(skill_name)
        logger.info(
            "Registered new neural skill '%s' (hash: %s, params: %d). Total primitives: %d",
            skill_name,
            record.parameter_hash,
            record.trainable_parameters,
            len(self._primitives),
        )

    def rollback_skill(self, skill_name: str) -> None:
        """Roll back a skill candidate completely, restoring previous repertoire state."""
        if skill_name in self._primitives:
            del self._primitives[skill_name]
            if skill_name in self._history:
                self._history.remove(skill_name)
            logger.info("Rolled back skill '%s'. Total primitives restored to %d", skill_name, len(self._primitives))

    def generate_expanded_candidate_plans(self, max_depth: int = 2) -> list[CandidatePlan]:
        """Generate hypothesis space over all currently registered primitives."""
        prim_names = list(self._primitives.keys())
        plans: list[CandidatePlan] = []

        # Abstain plan
        plans.append(
            CandidatePlan(
                plan_id="PLAN_ABSTAIN",
                plan_type=PlanType.ABSTAIN,
                stages=(),
                depth=0,
                description="Abstain / Unsupported task",
            )
        )

        # Depth 1: Single-stage execution
        if max_depth >= 1:
            for p in prim_names:
                plans.append(
                    CandidatePlan(
                        plan_id=f"PLAN_SINGLE_{p}",
                        plan_type=PlanType.SINGLE_PRIMITIVE,
                        stages=(p,),
                        depth=1,
                        description=f"Single-stage primitive: {p}",
                    )
                )

        # Depth 2: Two-stage sequential compositions
        if max_depth >= 2:
            for p1 in prim_names:
                for p2 in prim_names:
                    if p1 != p2:
                        plans.append(
                            CandidatePlan(
                                plan_id=f"PLAN_COMPOSE_{p1}_THEN_{p2}",
                                plan_type=PlanType.COMPOSED_TWO_STAGE,
                                stages=(p1, p2),
                                depth=2,
                                description=f"Two-stage composition: {p1} -> {p2}",
                            )
                        )

        return plans
