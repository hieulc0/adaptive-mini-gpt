"""Syntax and safety validator for autonomous experiment proposal."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time


@dataclass
class ValidationResult:
    """Result of syntax and safety validation on a candidate experiment."""
    is_valid: bool
    rejection_reason: str = ""
    source: str = ""
    target: str = ""
    relation: str = ""
    validation_latency_seconds: float = 0.0


@dataclass
class ExperimentValidatorStats:
    """Statistics on generated experiment validation."""
    total_proposed: int = 0
    valid_accepted: int = 0
    rejected_unknown_entity: int = 0
    rejected_self_loop: int = 0
    rejected_redundant: int = 0
    rejected_malformed: int = 0
    total_validation_latency_seconds: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.total_proposed == 0:
            return 0.0
        return self.valid_accepted / self.total_proposed


class ExperimentSyntaxValidator:
    """Validates dynamically proposed experiments against environment constraints."""

    def __init__(self, allowed_entities: list[str], allowed_relation: str) -> None:
        self.allowed_entities: set[str] = set(allowed_entities)
        self.allowed_relation: str = allowed_relation
        self.seen_queries: set[tuple[str, str]] = set()
        self.stats = ExperimentValidatorStats()

    def validate_experiment(self, source: str, relation: str, target: str) -> ValidationResult:
        """Validate candidate experiment tuple (source, relation, target)."""
        t0 = time.perf_counter()
        self.stats.total_proposed += 1

        # Check malformed
        if not source or not target or not isinstance(source, str) or not isinstance(target, str):
            dt = time.perf_counter() - t0
            self.stats.rejected_malformed += 1
            self.stats.total_validation_latency_seconds += dt
            return ValidationResult(False, "MALFORMED_NON_STRING_TOKENS", source, target, relation, dt)

        s_clean = source.strip()
        t_clean = target.strip()
        r_clean = relation.strip() if relation else ""

        # Check relation token
        if r_clean and r_clean != self.allowed_relation:
            dt = time.perf_counter() - t0
            self.stats.rejected_malformed += 1
            self.stats.total_validation_latency_seconds += dt
            return ValidationResult(False, f"INVALID_RELATION_TOKEN_{r_clean}", s_clean, t_clean, r_clean, dt)

        # Check allowed entity membership
        if s_clean not in self.allowed_entities:
            dt = time.perf_counter() - t0
            self.stats.rejected_unknown_entity += 1
            self.stats.total_validation_latency_seconds += dt
            return ValidationResult(False, f"UNKNOWN_SOURCE_ENTITY_{s_clean}", s_clean, t_clean, r_clean, dt)

        if t_clean not in self.allowed_entities:
            dt = time.perf_counter() - t0
            self.stats.rejected_unknown_entity += 1
            self.stats.total_validation_latency_seconds += dt
            return ValidationResult(False, f"UNKNOWN_TARGET_ENTITY_{t_clean}", s_clean, t_clean, r_clean, dt)

        # Check reflexive self loop
        if s_clean == t_clean:
            dt = time.perf_counter() - t0
            self.stats.rejected_self_loop += 1
            self.stats.total_validation_latency_seconds += dt
            return ValidationResult(False, "SELF_LOOP_REFLEXIVE_QUERY", s_clean, t_clean, r_clean, dt)

        # Check redundancy
        query_key = (s_clean, t_clean)
        if query_key in self.seen_queries:
            dt = time.perf_counter() - t0
            self.stats.rejected_redundant += 1
            self.stats.total_validation_latency_seconds += dt
            return ValidationResult(False, "REDUNDANT_REPEATED_QUERY", s_clean, t_clean, r_clean, dt)

        # Valid!
        self.seen_queries.add(query_key)
        self.stats.valid_accepted += 1
        dt = time.perf_counter() - t0
        self.stats.total_validation_latency_seconds += dt
        return ValidationResult(True, "VALID", s_clean, t_clean, self.allowed_relation, dt)

    def reset(self) -> None:
        self.seen_queries.clear()
        self.stats = ExperimentValidatorStats()
