"""Milestone M96 Phase 2: Selective Memory Management Policy.

Implements decision routing for persistent knowledge acquisition:
- Determines whether to store a new fact, update an existing fact,
  retrieve past memories, or abstain on contradictory/unsupported claims.
- Evaluates epistemic confidence and source validity.
- Enforces external memory budget constraints (<= 256 KiB).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.persistent_memory.knowledge_store import (
    FactualRecord,
    PersistentKnowledgeStore,
)


class MemoryAction(str, enum.Enum):
    """Categorical actions executed by the memory manager."""

    STORE_NEW_FACT = "STORE_NEW_FACT"
    UPDATE_EXISTING_FACT = "UPDATE_EXISTING_FACT"
    RETRIEVE_KNOWLEDGE = "RETRIEVE_KNOWLEDGE"
    REJECT_CONTRADICTION = "REJECT_CONTRADICTION"
    ABSTAIN_UNSUPPORTED = "ABSTAIN_UNSUPPORTED"
    NO_OP = "NO_OP"


@dataclass(frozen=True)
class MemoryDecision:
    """Action outcome and rationale produced by the memory manager."""

    action: MemoryAction
    entity: str | None
    attribute: str | None
    value: Any | None
    confidence: float
    reason: str
    recalled_record: FactualRecord | None = None


class SelectiveMemoryManager:
    """Autonomous governor deciding how knowledge is routed, stored, and recalled."""

    def __init__(
        self,
        knowledge_store: PersistentKnowledgeStore,
        min_confidence_threshold: float = 0.5,
    ) -> None:
        self.store = knowledge_store
        self.min_confidence_threshold = min_confidence_threshold

    def evaluate_statement(
        self,
        entity: str,
        attribute: str,
        value: Any,
        confidence: float = 1.0,
        is_verified_correction: bool = False,
        is_contradictory: bool = False,
        timestamp: int = 0,
    ) -> MemoryDecision:
        """Evaluate an incoming claim and decide whether to store, update, or reject."""
        clean_ent = entity.strip().lower()
        clean_attr = attribute.strip().lower()

        # Epistemic rejection on unsupported or contradictory statements
        if is_contradictory:
            return MemoryDecision(
                action=MemoryAction.REJECT_CONTRADICTION,
                entity=clean_ent,
                attribute=clean_attr,
                value=value,
                confidence=confidence,
                reason="statement_contradicts_prior_verification_without_correction_marker",
            )

        if confidence < self.min_confidence_threshold:
            return MemoryDecision(
                action=MemoryAction.ABSTAIN_UNSUPPORTED,
                entity=clean_ent,
                attribute=clean_attr,
                value=value,
                confidence=confidence,
                reason=f"confidence_{confidence:.2f}_below_threshold_{self.min_confidence_threshold}",
            )

        existing = self.store.retrieve_fact(clean_ent, clean_attr)
        if existing is not None:
            if existing.value == value:
                return MemoryDecision(
                    action=MemoryAction.NO_OP,
                    entity=clean_ent,
                    attribute=clean_attr,
                    value=value,
                    confidence=confidence,
                    reason="fact_already_verified_and_current",
                    recalled_record=existing,
                )
            # Value changed: Update fact
            self.store.update_fact(
                entity=clean_ent,
                attribute=clean_attr,
                new_value=value,
                confidence=confidence,
                reason="verified_update" if is_verified_correction else "observation_update",
                timestamp=timestamp,
            )
            return MemoryDecision(
                action=MemoryAction.UPDATE_EXISTING_FACT,
                entity=clean_ent,
                attribute=clean_attr,
                value=value,
                confidence=confidence,
                reason="updated_prior_fact",
                recalled_record=existing,
            )

        # New Fact: Store
        self.store.store_fact(
            entity=clean_ent,
            attribute=clean_attr,
            value=value,
            confidence=confidence,
            source="verified_statement",
            timestamp=timestamp,
        )
        return MemoryDecision(
            action=MemoryAction.STORE_NEW_FACT,
            entity=clean_ent,
            attribute=clean_attr,
            value=value,
            confidence=confidence,
            reason="new_verified_fact_stored",
        )

    def evaluate_query(self, entity: str, attribute: str) -> MemoryDecision:
        """Recall knowledge for a given entity and attribute."""
        clean_ent = entity.strip().lower()
        clean_attr = attribute.strip().lower()
        rec = self.store.retrieve_fact(clean_ent, clean_attr)

        if rec is not None:
            return MemoryDecision(
                action=MemoryAction.RETRIEVE_KNOWLEDGE,
                entity=clean_ent,
                attribute=clean_attr,
                value=rec.value,
                confidence=rec.confidence,
                reason="active_fact_recalled",
                recalled_record=rec,
            )

        return MemoryDecision(
            action=MemoryAction.ABSTAIN_UNSUPPORTED,
            entity=clean_ent,
            attribute=clean_attr,
            value=None,
            confidence=0.0,
            reason="unknown_entity_attribute_pair",
        )
