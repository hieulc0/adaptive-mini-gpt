"""M36 Phase 8: Future Structural Evolution Interface.

Defines the formal protocol and interfaces for proposing, evaluating, and safely rolling
back future structural neural modifications (e.g., dynamic topology, gated skip connections,
dimension expansion, or pruning) driven by independent evidence.
"""

from __future__ import annotations

import enum
import hashlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any


class StructuralModificationType(str, enum.Enum):
    """Supported classes of structural architectural proposals."""

    WEIGHT_ADJUSTMENT = "weight_adjustment"
    ADD_UPDATE_ADAPTER = "add_update_adapter"
    GATED_SKIP_CONNECTION = "gated_skip_connection"
    EXPAND_DIMENSIONS = "expand_dimensions"
    PRUNE_CONNECTIONS = "prune_connections"


@dataclass
class StructuralProposal:
    """A formal proposal for architectural modification submitted to the external controller."""

    proposal_id: str
    modification_type: StructuralModificationType
    target_module: str
    evidence_of_need: str
    expected_gain: float
    parameter_delta: int
    compute_overhead_ratio: float
    rollback_hash: str
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    @classmethod
    def create(
        cls,
        modification_type: StructuralModificationType,
        target_module: str,
        evidence_of_need: str,
        expected_gain: float,
        parameter_delta: int,
        compute_overhead_ratio: float,
        current_checkpoint_hash: str,
    ) -> StructuralProposal:
        """Construct a structural proposal with deterministic ID fingerprint."""
        raw_key = f"{modification_type.value}::{target_module}::{current_checkpoint_hash}"
        pid = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
        return cls(
            proposal_id=pid,
            modification_type=modification_type,
            target_module=target_module,
            evidence_of_need=evidence_of_need,
            expected_gain=round(expected_gain, 4),
            parameter_delta=parameter_delta,
            compute_overhead_ratio=round(compute_overhead_ratio, 4),
            rollback_hash=current_checkpoint_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["modification_type"] = self.modification_type.value
        return d


# Alias for backward and forward compatibility
StructuralModificationProposal = StructuralProposal


class StructuralEvolutionGate:
    """External controller gate that evaluates structural modification proposals."""

    def __init__(
        self,
        max_parameter_growth_ratio: float = 0.20,
        min_expected_gain: float = 0.15,
    ) -> None:
        self.max_parameter_growth_ratio = max_parameter_growth_ratio
        self.min_expected_gain = min_expected_gain

    def evaluate_proposal(
        self,
        proposal: StructuralProposal,
        base_parameter_count: int,
    ) -> tuple[bool, str]:
        """Verify whether a structural proposal meets safety, budget, and evidence gates."""
        if proposal.expected_gain < self.min_expected_gain:
            return False, f"Expected gain {proposal.expected_gain} below minimum gate {self.min_expected_gain}"

        growth_ratio = proposal.parameter_delta / max(1, base_parameter_count)
        if growth_ratio > self.max_parameter_growth_ratio:
            return False, f"Parameter growth ratio {growth_ratio:.2%} exceeds ceiling {self.max_parameter_growth_ratio:.2%}"

        if not proposal.rollback_hash:
            return False, "Missing verified rollback hash for state checkpoint"

        return True, "Proposal approved for isolated sandbox evaluation"
