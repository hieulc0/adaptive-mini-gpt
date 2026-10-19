"""Structured Working Memory with Provenance, Span Tracking, and Grounded Slots (M15/M16/M17).

Implements bounded, in-process working memory representing resolved information:
- Operation name (single-step) / Plan tracking (multi-step)
- Extracted argument values and intermediate step variables
- Exact source-span provenance and variable reference dependencies
- Typed GroundedSlot registry for stable input binding without literal hallucination (M17)
- Bounded multi-turn ConversationMemory with safe cross-turn reference resolution (M17)
- Units and dimensional consistency tracking
- Deterministic verification state and results
- Explicit rejection / abstention reasons

Safety:
- In-process, bounded dictionary/dataclass representation.
- No vector database, persistent external storage, or arbitrary code execution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


class AmbiguousReferenceError(Exception):
    """Raised when a cross-turn conversational reference is ambiguous and cannot be resolved safely."""


@dataclass
class ArgumentProvenance:
    """Provenance tracking for an individual argument slot."""

    slot_name: str
    source_text: str
    start_idx: int
    end_idx: int
    parsed_value: Any
    type_name: str
    extraction_method: str = "span_copy"  # "span_copy" | "regex" | "autoregressive" | "derived"

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_name": self.slot_name,
            "source_text": self.source_text,
            "span": [self.start_idx, self.end_idx],
            "parsed_value": self.parsed_value,
            "type_name": self.type_name,
            "extraction_method": self.extraction_method,
        }


@dataclass
class StepProvenance:
    """Provenance tracking for a derived intermediate variable in a multi-step plan (M16)."""

    var_name: str
    operation: str
    raw_args: dict[str, Any]
    resolved_args: dict[str, Any]
    result: Any
    units: str | None = None
    step_index: int = 1
    verified: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "var_name": self.var_name,
            "operation": self.operation,
            "raw_args": copy_serializable(self.raw_args),
            "resolved_args": copy_serializable(self.resolved_args),
            "result": copy_serializable(self.result),
            "units": self.units,
            "step_index": self.step_index,
            "verified": self.verified,
        }


@dataclass
class GroundedSlot:
    """Typed source-grounded input slot for structural plan execution (M17).

    Separates neural plan structure from deterministic argument values:
    - Stable identifier (e.g. 'distance_1', 'duration_1')
    - Typed value and unit
    - Original source span
    - Extraction provenance and verification status
    """

    slot_id: str
    value: Any
    unit: str | None = None
    source_text: str = ""
    start_idx: int = 0
    end_idx: int = 0
    extraction_method: str = "deterministic"
    verified: bool = True
    type_name: str = "float"

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "value": copy_serializable(self.value),
            "unit": self.unit,
            "source_text": self.source_text,
            "span": [self.start_idx, self.end_idx],
            "extraction_method": self.extraction_method,
            "verified": self.verified,
            "type_name": self.type_name,
        }


@dataclass
class WorkingMemory:
    """Structured in-process working memory for rule reasoning, slot binding, and execution."""

    operation: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, ArgumentProvenance] = field(default_factory=dict)
    verified: bool = False
    result: Any | None = None
    status: str = "empty"  # "empty" | "grounded" | "verified" | "rejected"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Multi-step planning extensions (M16)
    variables: dict[str, Any] = field(default_factory=dict)
    step_provenance: dict[str, StepProvenance] = field(default_factory=dict)
    step_units: dict[str, str | None] = field(default_factory=dict)

    # Grounded Slot Registry extensions (M17)
    slots: dict[str, GroundedSlot] = field(default_factory=dict)

    def bind_operation(self, operation: str) -> None:
        """Set active operation."""
        self.operation = operation.strip().lower()
        self.status = "operation_selected"

    def bind_slot(self, slot: GroundedSlot) -> None:
        """Register a typed source-grounded slot into working memory (M17)."""
        self.slots[slot.slot_id] = slot

    def bind_slot_value(
        self,
        slot_id: str,
        value: Any,
        unit: str | None = None,
        source_text: str = "",
        span: tuple[int, int] = (0, 0),
        type_name: str = "float",
        extraction_method: str = "deterministic",
    ) -> None:
        """Helper to create and register a typed GroundedSlot."""
        slot = GroundedSlot(
            slot_id=slot_id,
            value=value,
            unit=unit,
            source_text=source_text,
            start_idx=span[0],
            end_idx=span[1],
            extraction_method=extraction_method,
            verified=True,
            type_name=type_name,
        )
        self.bind_slot(slot)

    def get_slot(self, slot_id: str) -> GroundedSlot | None:
        """Retrieve grounded slot by identifier."""
        return self.slots.get(slot_id)

    def has_slot(self, slot_id: str) -> bool:
        """Check whether slot identifier exists."""
        return slot_id in self.slots

    def bind_argument(
        self,
        slot_name: str,
        value: Any,
        source_text: str,
        span: tuple[int, int],
        type_name: str,
        extraction_method: str = "span_copy",
    ) -> None:
        """Bind an argument value with full source-span provenance."""
        self.arguments[slot_name] = value
        self.provenance[slot_name] = ArgumentProvenance(
            slot_name=slot_name,
            source_text=source_text,
            start_idx=span[0],
            end_idx=span[1],
            parsed_value=value,
            type_name=type_name,
            extraction_method=extraction_method,
        )
        self.status = "grounded"

    def bind_step_variable(
        self,
        var_name: str,
        value: Any,
        operation: str,
        raw_args: dict[str, Any],
        resolved_args: dict[str, Any],
        units: str | None = None,
        step_index: int = 1,
    ) -> None:
        """Bind intermediate verified variable from plan step execution (M16)."""
        self.variables[var_name] = value
        self.step_units[var_name] = units
        self.step_provenance[var_name] = StepProvenance(
            var_name=var_name,
            operation=operation,
            raw_args=raw_args,
            resolved_args=resolved_args,
            result=value,
            units=units,
            step_index=step_index,
            verified=True,
        )
        self.status = "executing_plan"

    def bind_result(self, result: Any, verified: bool = True) -> None:
        """Record verified execution result."""
        self.result = result
        self.verified = verified
        self.status = "verified" if verified else "unverified"

    def reject(self, reason: str) -> None:
        """Explicitly reject reasoning or execution with justification."""
        self.verified = False
        self.status = "rejected"
        self.error = reason

    def is_complete_for_rule(self, required_slots: list[str]) -> bool:
        """Check if all required slots are bound in working memory."""
        return all(slot in self.arguments for slot in required_slots)

    def to_dict(self) -> dict[str, Any]:
        """Convert working memory to serializable dictionary representation."""
        return {
            "operation": self.operation,
            "arguments": copy_serializable(self.arguments),
            "provenance": {k: v.to_dict() for k, v in self.provenance.items()},
            "variables": copy_serializable(self.variables),
            "step_provenance": {k: v.to_dict() for k, v in self.step_provenance.items()},
            "step_units": self.step_units,
            "slots": {k: v.to_dict() for k, v in self.slots.items()},
            "verified": self.verified,
            "result": copy_serializable(self.result),
            "status": self.status,
            "error": self.error,
            "metadata": copy_serializable(self.metadata),
        }


@dataclass
class ConversationTurn:
    """A single conversational turn within a multi-turn reasoning session (M17 Phase 5)."""

    turn_id: int
    user_message: str
    assistant_response: str | None = None
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_message": self.user_message,
            "assistant_response": self.assistant_response,
            "working_memory": self.working_memory.to_dict(),
            "status": self.status,
        }


@dataclass
class ConversationMemory:
    """Bounded, in-process conversational working memory across consecutive turns (M17 Phase 5).

    Supports:
    - Bounded session memory (up to max_turns, default 5)
    - Cross-turn variable lookup (e.g. 'turn1.d_total', 'turn1.distance_1')
    - Ambiguity detection for pronouns like 'that value' or 'the previous result'
    - Strictly isolated by session; zero external databases or persistence.
    """

    session_id: str
    max_turns: int = 5
    turns: list[ConversationTurn] = field(default_factory=list)

    def start_turn(self, user_message: str) -> ConversationTurn:
        """Start a new conversational turn."""
        if len(self.turns) >= self.max_turns:
            # Shift earliest turn if exceeding bounded capacity
            self.turns.pop(0)

        turn_id = len(self.turns) + 1
        turn = ConversationTurn(turn_id=turn_id, user_message=user_message)
        self.turns.append(turn)
        return turn

    def get_turn(self, turn_id: int) -> ConversationTurn | None:
        """Retrieve turn by 1-indexed turn_id."""
        for t in self.turns:
            if t.turn_id == turn_id:
                return t
        return None

    def get_last_turn(self) -> ConversationTurn | None:
        """Retrieve most recently completed turn, if any."""
        if not self.turns:
            return None
        # If the last turn is currently in progress, return the one before it
        if len(self.turns) >= 2:
            return self.turns[-2]
        return self.turns[-1]

    def resolve_reference(
        self,
        ref_name: str,
        target_unit: str | None = None,
    ) -> tuple[Any, str | None]:
        """Resolve a cross-turn reference to (value, unit) or raise AmbiguousReferenceError / KeyError."""
        clean_ref = ref_name.strip().lstrip("$")

        # Case 1: Explicit syntax 'turn1.var_name' or 'turn1.slot_name'
        if "." in clean_ref:
            turn_part, var_part = clean_ref.split(".", 1)
            m = re.match(r"^turn(\d+)$", turn_part.lower())
            if not m:
                raise KeyError(f"Invalid cross-turn format: '{ref_name}'")
            t_id = int(m.group(1))
            turn = self.get_turn(t_id)
            if not turn:
                raise KeyError(f"Turn {t_id} does not exist in conversational memory")

            # Check variables
            if var_part in turn.working_memory.variables:
                val = turn.working_memory.variables[var_part]
                unit = turn.working_memory.step_units.get(var_part)
                return val, unit

            # Check slots
            if var_part in turn.working_memory.slots:
                slot = turn.working_memory.slots[var_part]
                return slot.value, slot.unit

            raise KeyError(f"Variable or slot '{var_part}' not found in turn {t_id}")

        # Case 2: Implicit references e.g. 'previous_result', 'previous_speed', 'that value'
        last_turn = self.get_last_turn()
        if not last_turn:
            raise KeyError("No prior conversational turn exists to resolve reference")

        ref_lower = clean_ref.lower()

        # Check for explicit speed / distance / duration / result referents
        if "speed" in ref_lower:
            # Look for variables or slots with unit km/h
            speed_candidates = []
            for k, u in last_turn.working_memory.step_units.items():
                if u == "km/h":
                    speed_candidates.append((last_turn.working_memory.variables[k], u))
            for sid, slot in last_turn.working_memory.slots.items():
                if slot.unit == "km/h":
                    speed_candidates.append((slot.value, slot.unit))
            if len(speed_candidates) == 1:
                return speed_candidates[0]
            if len(speed_candidates) > 1:
                # Return the final speed result
                return speed_candidates[-1]
            if last_turn.working_memory.result is not None:
                return last_turn.working_memory.result, "km/h"

        if "distance" in ref_lower:
            candidates = []
            for k, u in last_turn.working_memory.step_units.items():
                if u == "km":
                    candidates.append((last_turn.working_memory.variables[k], u))
            for sid, slot in last_turn.working_memory.slots.items():
                if slot.unit == "km":
                    candidates.append((slot.value, slot.unit))
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                # Ambiguous which distance was intended
                raise AmbiguousReferenceError(
                    f"Ambiguous distance reference: multiple distance quantities ({len(candidates)}) exist in turn {last_turn.turn_id}"
                )

        if "time" in ref_lower or "duration" in ref_lower:
            candidates = []
            for k, u in last_turn.working_memory.step_units.items():
                if u == "h":
                    candidates.append((last_turn.working_memory.variables[k], u))
            for sid, slot in last_turn.working_memory.slots.items():
                if slot.unit == "h":
                    candidates.append((slot.value, slot.unit))
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                raise AmbiguousReferenceError(
                    f"Ambiguous duration reference: multiple time quantities ({len(candidates)}) exist in turn {last_turn.turn_id}"
                )

        # Ambiguous pronoun references like 'that value', 'the number', 'it'
        if any(p in ref_lower for p in ("that value", "that number", "the previous result", "it", "previous")):
            # If the last turn has a single unambiguous result, return it
            if last_turn.working_memory.result is not None:
                # Find its unit if available
                res_unit = None
                for k, v in last_turn.working_memory.variables.items():
                    if v == last_turn.working_memory.result:
                        res_unit = last_turn.working_memory.step_units.get(k)
                        break
                return last_turn.working_memory.result, res_unit
            # If multiple variables and no single final result, raise AmbiguousReferenceError
            if len(last_turn.working_memory.variables) > 1:
                raise AmbiguousReferenceError(
                    f"Ambiguous pronoun '{clean_ref}': prior turn has multiple candidate variables: {list(last_turn.working_memory.variables.keys())}"
                )

        raise KeyError(f"Could not resolve reference '{ref_name}' in conversational memory")

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "max_turns": self.max_turns,
            "num_turns": len(self.turns),
            "turns": [t.to_dict() for t in self.turns],
        }


def copy_serializable(val: Any) -> Any:
    """Helper to ensure JSON-serializable representation of memory attributes."""
    if isinstance(val, (int, float, str, bool)) or val is None:
        return val
    if isinstance(val, list):
        return [copy_serializable(x) for x in val]
    if isinstance(val, dict):
        return {str(k): copy_serializable(v) for k, v in val.items()}
    return str(val)
