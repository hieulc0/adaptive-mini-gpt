"""Deterministic Rule Executor with Precondition Validation and Postcondition Verification (M14).

Executes registered symbolic/arithmetic operations safely without eval() or shell execution.
Captures timing, input validation errors, and deterministic verifier status.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.rules.registry import (
    RULE_REGISTRY,
    RuleError,
    RuleSpecification,
    SchemaValidationError,
    VerificationError,
    get_rule,
)


@dataclass
class RuleExecutionResult:
    """Outcome of an external deterministic rule execution."""

    rule_name: str
    args: dict[str, Any]
    result: Any | None = None
    success: bool = False
    verified: bool = False
    error: str | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "args": self.args,
            "result": str(self.result) if self.result is not None else None,
            "success": self.success,
            "verified": self.verified,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 3),
        }


class RuleExecutor:
    """Bounded, deterministic rule execution engine with schema checks and verification."""

    def __init__(self, registry: dict[str, RuleSpecification] | None = None) -> None:
        self.registry = registry if registry is not None else RULE_REGISTRY
        self.total_invocations = 0
        self.successful_executions = 0
        self.verified_executions = 0
        self.failed_executions = 0

    def _coerce_and_validate_args(
        self, spec: RuleSpecification, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate and safely coerce argument types to match rule schema."""
        sig = inspect.signature(spec.execute_fn)
        coerced: dict[str, Any] = {}

        # Check for unexpected arguments
        expected_params = set(sig.parameters.keys())
        provided_params = set(args.keys())
        extra = provided_params - expected_params
        if extra:
            raise SchemaValidationError(
                f"Unexpected argument(s) for rule '{spec.name}': {list(extra)}. Expected: {list(expected_params)}"
            )

        # Check for missing arguments
        missing = expected_params - provided_params
        if missing:
            raise SchemaValidationError(
                f"Missing required argument(s) for rule '{spec.name}': {list(missing)}"
            )

        for param_name, param in sig.parameters.items():
            val = args[param_name]
            # Coerce numeric strings if numeric expected
            if spec.name == "add":
                try:
                    if isinstance(val, str) and "." in val:
                        val = float(val)
                    elif isinstance(val, str):
                        val = int(val)
                except ValueError:
                    raise SchemaValidationError(f"Argument '{param_name}' must be numeric, got '{val}'")

            elif spec.name == "average_speed":
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    raise SchemaValidationError(f"Argument '{param_name}' must be numeric, got '{val}'")

            elif spec.name == "reverse":
                if not isinstance(val, str):
                    val = str(val)

            elif spec.name == "compare":
                if param_name in ("entity_a", "entity_b"):
                    val = str(val)
                elif param_name == "relations" and not isinstance(val, list):
                    raise SchemaValidationError("Argument 'relations' must be a list of relation strings")

            coerced[param_name] = val

        return coerced

    def execute(
        self, rule_name: str, args: dict[str, Any], verify: bool = True
    ) -> RuleExecutionResult:
        """Execute a registered rule with strict validation and independent verification."""
        t0 = time.perf_counter()
        self.total_invocations += 1

        try:
            # 1. Lookup rule
            canonical_name = rule_name.strip().lower()
            if canonical_name not in self.registry:
                allowed = list(self.registry.keys())
                raise SchemaValidationError(
                    f"Unknown rule '{rule_name}'. Allowed operations: {allowed}"
                )
            spec = self.registry[canonical_name]

            # 2. Schema check and parameter coercion
            coerced_args = self._coerce_and_validate_args(spec, args)

            # 3. Deterministic execution
            raw_result = spec.execute_fn(**coerced_args)

            # 4. Independent postcondition verification
            is_verified = False
            if verify:
                is_verified = spec.verify_fn(coerced_args, raw_result)
                if not is_verified:
                    raise VerificationError(
                        f"Postcondition verification failed for rule '{spec.name}' with args {coerced_args}"
                    )
            else:
                is_verified = True

            latency = (time.perf_counter() - t0) * 1000.0
            self.successful_executions += 1
            if is_verified:
                self.verified_executions += 1

            return RuleExecutionResult(
                rule_name=canonical_name,
                args=coerced_args,
                result=raw_result,
                success=True,
                verified=is_verified,
                error=None,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000.0
            self.failed_executions += 1
            return RuleExecutionResult(
                rule_name=rule_name,
                args=args,
                result=None,
                success=False,
                verified=False,
                error=f"{type(e).__name__}: {str(e)}",
                latency_ms=latency,
            )
