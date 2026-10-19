"""Explicit Rule Registry, Schemas, and Precondition/Postcondition Verifiers (M14).

Defines deterministic, bounded mathematical and symbolic operations:
- 'reverse': Sequence reversal with inverse verification
- 'add': Multi-digit arithmetic addition with subtraction inverse check
- 'compare': Transitive relation ordering over entities with acyclicity check
- 'average_speed': Distance/time kinematics calculation with rate consistency check

Safety:
- Disallows arbitrary Python execution (no eval, no exec, no shell).
- Enforces strict argument typing and schema validation.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class RuleError(Exception):
    """Base exception for rule execution and validation."""


class SchemaValidationError(RuleError):
    """Raised when arguments violate the rule's parameter schema."""


class VerificationError(RuleError):
    """Raised when rule result fails deterministic independent postcondition verification."""


@dataclass
class RuleSpecification:
    """Specification of an allowed, deterministic operation."""

    name: str
    description: str
    args_schema: dict[str, str]  # Human-readable argument name -> expected type/format
    execute_fn: Callable[..., Any]
    verify_fn: Callable[[dict[str, Any], Any], bool]


# =====================================================================
# Core Deterministic Implementations & Verifiers
# =====================================================================


def execute_reverse(s: str) -> str:
    """Deterministically reverse an input string."""
    if not isinstance(s, str):
        raise SchemaValidationError(f"Expected string for 's', got {type(s).__name__}")
    if not s:
        raise SchemaValidationError("String 's' must be non-empty")
    return s[::-1]


def verify_reverse(args: dict[str, Any], result: Any) -> bool:
    """Verify that result is exact inverse of input string."""
    s = args.get("s")
    if not isinstance(s, str) or not isinstance(result, str):
        return False
    return len(result) == len(s) and result[::-1] == s


def execute_add(a: int | float, b: int | float) -> int | float:
    """Deterministically compute arithmetic addition."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise SchemaValidationError(
            f"Expected numeric arguments 'a' and 'b', got {type(a).__name__} and {type(b).__name__}"
        )
    return a + b


def verify_add(args: dict[str, Any], result: Any) -> bool:
    """Verify addition result via inverse subtraction."""
    a = args.get("a")
    b = args.get("b")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    if not isinstance(result, (int, float)):
        return False
    return abs((result - b) - a) < 1e-6


def execute_compare(entity_a: str, entity_b: str, relations: list[str]) -> str:
    """Deterministically determine transitive ordering between two entities given relation premises."""
    if not isinstance(entity_a, str) or not entity_a.strip():
        raise SchemaValidationError("entity_a must be a non-empty string")
    if not isinstance(entity_b, str) or not entity_b.strip():
        raise SchemaValidationError("entity_b must be a non-empty string")
    if not isinstance(relations, list) or not relations:
        raise SchemaValidationError("relations must be a non-empty list of premise strings")

    # Build directed graph: u > v creates directed edge u -> v
    adj: dict[str, set[str]] = {}
    for r in relations:
        if not isinstance(r, str):
            raise SchemaValidationError(f"Invalid relation item: {r}")
        parts = [p.strip() for p in r.split(">")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise SchemaValidationError(f"Relation premise must follow 'A > B' format: '{r}'")
        u, v = parts[0], parts[1]
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set())

    # Directed BFS reachability
    def can_reach(src: str, dst: str) -> bool:
        visited: set[str] = set()
        queue = [src]
        while queue:
            curr = queue.pop(0)
            if curr == dst:
                return True
            for nxt in adj.get(curr, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    a_reaches_b = can_reach(entity_a, entity_b)
    b_reaches_a = can_reach(entity_b, entity_a)

    if a_reaches_b and b_reaches_a:
        raise VerificationError(
            f"Contradictory cyclic ordering detected between '{entity_a}' and '{entity_b}'"
        )
    if a_reaches_b:
        return entity_a
    elif b_reaches_a:
        return entity_b
    else:
        raise VerificationError(
            f"Indeterminate: no transitive path connects '{entity_a}' and '{entity_b}'"
        )


def verify_compare(args: dict[str, Any], result: Any) -> bool:
    """Verify transitive comparison output against graph reachability."""
    ea = args.get("entity_a")
    eb = args.get("entity_b")
    rels = args.get("relations")
    if not isinstance(ea, str) or not isinstance(eb, str) or not isinstance(rels, list):
        return False
    if result not in (ea, eb):
        return False
    try:
        expected = execute_compare(ea, eb, rels)
        return result == expected
    except Exception:
        return False


def execute_average_speed(distance_km: float, time_hours: float) -> str:
    """Deterministically compute average speed from distance and duration."""
    if not isinstance(distance_km, (int, float)) or not isinstance(time_hours, (int, float)):
        raise SchemaValidationError("distance_km and time_hours must be numeric")
    if time_hours <= 0:
        raise SchemaValidationError(f"time_hours must be strictly positive (> 0), got {time_hours}")
    if distance_km < 0:
        raise SchemaValidationError(f"distance_km must be non-negative (>= 0), got {distance_km}")

    speed = round(distance_km / time_hours, 2)
    return f"{speed:g} km/h"


def verify_average_speed(args: dict[str, Any], result: Any) -> bool:
    """Verify average speed consistency with distance and duration."""
    d = args.get("distance_km")
    t = args.get("time_hours")
    if not isinstance(d, (int, float)) or not isinstance(t, (int, float)) or t <= 0:
        return False
    if not isinstance(result, str) or not result.endswith(" km/h"):
        return False
    try:
        val = float(result.replace(" km/h", "").strip())
        return abs(val * t - d) < 1e-1
    except Exception:
        return False


# =====================================================================
# Rule Registry
# =====================================================================

RULE_REGISTRY: dict[str, RuleSpecification] = {
    "reverse": RuleSpecification(
        name="reverse",
        description="Reverses an arbitrary character sequence: reverse(s='string') -> 'gnirts'",
        args_schema={"s": "str"},
        execute_fn=execute_reverse,
        verify_fn=verify_reverse,
    ),
    "add": RuleSpecification(
        name="add",
        description="Calculates arithmetic addition of two numbers: add(a=int, b=int) -> sum",
        args_schema={"a": "int | float", "b": "int | float"},
        execute_fn=execute_add,
        verify_fn=verify_add,
    ),
    "compare": RuleSpecification(
        name="compare",
        description="Computes transitive inequality between two entities: compare(entity_a=str, entity_b=str, relations=list[str]) -> winner",
        args_schema={
            "entity_a": "str",
            "entity_b": "str",
            "relations": "list[str] (e.g. ['A > B', 'B > C'])",
        },
        execute_fn=execute_compare,
        verify_fn=verify_compare,
    ),
    "average_speed": RuleSpecification(
        name="average_speed",
        description="Calculates speed from distance and duration: average_speed(distance_km=float, time_hours=float) -> 'X km/h'",
        args_schema={"distance_km": "float", "time_hours": "float"},
        execute_fn=execute_average_speed,
        verify_fn=verify_average_speed,
    ),
}


def get_rule(name: str) -> RuleSpecification:
    """Retrieve rule specification by name, raising SchemaValidationError if unregistered."""
    canonical = name.strip().lower()
    if canonical not in RULE_REGISTRY:
        allowed = list(RULE_REGISTRY.keys())
        raise SchemaValidationError(f"Unknown operation '{name}'. Allowed operations: {allowed}")
    return RULE_REGISTRY[canonical]


def list_allowed_rules() -> list[str]:
    """Return list of all registered rule names."""
    return list(RULE_REGISTRY.keys())
