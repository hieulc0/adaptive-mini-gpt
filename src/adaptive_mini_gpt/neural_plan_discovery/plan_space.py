"""Phase 2: Candidate Plan Space for Bounded Neural Plan Discovery.

Defines:
- CandidatePlan: data structure representing an executable skill composition sequence.
- PlanType: ABSTAIN, SINGLE_PRIMITIVE, COMPOSED_TWO_STAGE.
- PlanSpace: factory and enumerator generating the bounded hypothesis space over available
  qualified neural primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from adaptive_mini_gpt.neural_skill_composition.primitive_skills import PrimitiveSkillType


class PlanType(StrEnum):
    """Categorical type of composition plan."""

    ABSTAIN = "ABSTAIN"
    SINGLE_PRIMITIVE = "SINGLE_PRIMITIVE"
    COMPOSED_TWO_STAGE = "COMPOSED_TWO_STAGE"


@dataclass(frozen=True)
class CandidatePlan:
    """An explicit execution plan specifying a sequence of neural skills."""

    plan_id: str
    plan_type: PlanType
    stages: tuple[str, ...]
    depth: int
    description: str

    @property
    def is_empty(self) -> bool:
        """Return True if plan abstains from execution."""
        return len(self.stages) == 0


class PlanSpace:
    """Manages generation and enumeration of candidate composition plans."""

    @classmethod
    def generate_candidate_plans(
        cls,
        available_primitives: Sequence[PrimitiveSkillType] | None = None,
        max_depth: int = 2,
        include_abstain: bool = True,
    ) -> list[CandidatePlan]:
        """Enumerate all valid candidate plans up to max_depth."""
        if available_primitives is None:
            available_primitives = [
                PrimitiveSkillType.COMPLEMENT_REFLECT,
                PrimitiveSkillType.CAESAR_SHIFT_MOD3,
                PrimitiveSkillType.AFFINE_TRANSFORM,
            ]

        prim_names = [p.value if isinstance(p, PrimitiveSkillType) else str(p) for p in available_primitives]
        plans: list[CandidatePlan] = []

        if include_abstain:
            plans.append(
                CandidatePlan(
                    plan_id="PLAN_ABSTAIN",
                    plan_type=PlanType.ABSTAIN,
                    stages=(),
                    depth=0,
                    description="Abstain / Incompatible procedure (no composition supported)",
                )
            )

        # Depth 1: Single-step primitive execution
        if max_depth >= 1:
            for p_name in prim_names:
                plans.append(
                    CandidatePlan(
                        plan_id=f"PLAN_SINGLE_{p_name}",
                        plan_type=PlanType.SINGLE_PRIMITIVE,
                        stages=(p_name,),
                        depth=1,
                        description=f"Single-stage primitive execution: {p_name}",
                    )
                )

        # Depth 2: Two-stage sequential composition
        if max_depth >= 2:
            for p1 in prim_names:
                for p2 in prim_names:
                    if p1 != p2:  # Distinct stages
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
