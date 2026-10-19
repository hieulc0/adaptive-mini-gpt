"""Typed Execution Plan Representation and Verification (M16).

Implements bounded, typed, acyclic multi-step execution plans:
- PlanStep: explicit operation, literal or variable reference arguments ($var), output variable, units, dependencies.
- ExecutionPlan: sequence of up to 5 steps, topological validation, cycle rejection, missing reference checks.
- PlanExecutor: executes steps topologically against WorkingMemory, resolves variable bindings, checks unit consistency, and records provenance.

Safety:
- Disallows arbitrary Python execution (no eval, no exec, no shell).
- Bounded to a maximum of 5 steps per plan.
- All operations must be registered in RULE_REGISTRY.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.rules.executor import RuleExecutionResult, RuleExecutor
from adaptive_mini_gpt.rules.memory import StepProvenance, WorkingMemory
from adaptive_mini_gpt.rules.registry import RULE_REGISTRY, SchemaValidationError


class PlanError(Exception):
    """Base exception for plan construction, validation, and execution errors."""


class PlanValidationError(PlanError):
    """Raised when a plan has invalid structure, cycles, or undefined references."""


class DependencyResolutionError(PlanError):
    """Raised when an execution step references an unverified or missing variable."""


class UnitMismatchError(PlanError):
    """Raised when an operation attempts to combine incompatible physical units."""


@dataclass
class PlanStep:
    """A single deterministic operation step in a multi-step execution plan."""

    output_var: str
    operation: str
    args: dict[str, Any]  # literal constants or variable references like "$dist"
    units: str | None = None
    step_index: int = 1

    def get_referenced_variables(self) -> list[str]:
        """Extract all variable reference names (e.g. '$x' -> 'x') from args."""
        refs: list[str] = []
        for val in self.args.values():
            if isinstance(val, str) and val.startswith("$"):
                refs.append(val[1:].strip())
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and item.startswith("$"):
                        refs.append(item[1:].strip())
        return refs

    def to_string(self) -> str:
        """Format step in human-readable canonical notation."""
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in self.args.items())
        unit_str = f" [{self.units}]" if self.units else ""
        return f"{self.output_var} = {self.operation}({args_str}){unit_str}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_var": self.output_var,
            "operation": self.operation,
            "args": self.args,
            "units": self.units,
            "step_index": self.step_index,
        }


@dataclass
class ExecutionPlan:
    """Bounded, acyclic multi-step execution plan."""

    steps: list[PlanStep] = field(default_factory=list)
    final_var: str | None = None

    def validate(self) -> None:
        """Validate topological ordering, cycle freedom, step bound, and registered operations."""
        if not self.steps:
            raise PlanValidationError("Execution plan cannot be empty")
        if len(self.steps) > 5:
            raise PlanValidationError(f"Plan exceeds maximum limit of 5 steps (got {len(self.steps)})")

        defined_vars: set[str] = set()
        for idx, step in enumerate(self.steps):
            step.step_index = idx + 1
            op_name = step.operation.strip().lower()
            if op_name not in RULE_REGISTRY:
                raise PlanValidationError(
                    f"Step {step.step_index} specifies unregistered operation '{step.operation}'"
                )

            # Check variable naming syntax
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", step.output_var):
                raise PlanValidationError(f"Invalid output variable name '{step.output_var}'")

            if step.output_var in defined_vars:
                raise PlanValidationError(
                    f"Variable '{step.output_var}' is redefined in step {step.step_index} (SSA violation)"
                )

            # Check dependencies: must be defined in prior steps
            refs = step.get_referenced_variables()
            for r in refs:
                if r == step.output_var:
                    raise PlanValidationError(
                        f"Step {step.step_index} ('{step.output_var}') contains self-dependency on '${r}'"
                    )
                if r not in defined_vars:
                    raise PlanValidationError(
                        f"Step {step.step_index} references undefined variable '${r}' (must be defined in prior step)"
                    )

            defined_vars.add(step.output_var)

        # Default final_var to output of last step if unspecified
        if not self.final_var:
            self.final_var = self.steps[-1].output_var
        elif self.final_var not in defined_vars:
            raise PlanValidationError(f"Final variable '{self.final_var}' is not defined in any step")

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "final_var": self.final_var,
            "num_steps": len(self.steps),
        }

    def to_text_format(self) -> str:
        """Serialize plan to readable line-based format."""
        lines = ["Plan:"]
        for s in self.steps:
            lines.append(f"{s.step_index}. {s.to_string()}")
        lines.append(f"Result: ${self.final_var}")
        return "\n".join(lines)


@dataclass
class PlanExecutionResult:
    """Result of executing an ExecutionPlan."""

    success: bool
    final_var: str | None
    final_result: Any | None
    step_results: dict[str, Any] = field(default_factory=dict)
    working_memory: WorkingMemory | None = None
    executed_steps: int = 0
    error: str | None = None
    failed_step_index: int | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "final_var": self.final_var,
            "final_result": self.final_result,
            "step_results": self.step_results,
            "executed_steps": self.executed_steps,
            "error": self.error,
            "failed_step_index": self.failed_step_index,
            "latency_ms": round(self.latency_ms, 2),
        }


class PlanExecutor:
    """Executes validated multi-step plans deterministically with provenance tracking."""

    def __init__(self, rule_executor: RuleExecutor | None = None) -> None:
        self.rule_executor = rule_executor if rule_executor is not None else RuleExecutor()

    def execute(
        self,
        plan: ExecutionPlan,
        working_memory: WorkingMemory | None = None,
    ) -> PlanExecutionResult:
        """Execute all steps in topological order, binding intermediate variables into WorkingMemory."""
        t0 = time.perf_counter()
        mem = working_memory if working_memory is not None else WorkingMemory()

        # Step 1: Validate plan syntax, acyclicity, and step count
        try:
            plan.validate()
        except PlanValidationError as e:
            mem.reject(f"Plan validation failed: {str(e)}")
            latency = (time.perf_counter() - t0) * 1000.0
            return PlanExecutionResult(
                success=False,
                final_var=plan.final_var,
                final_result=None,
                working_memory=mem,
                executed_steps=0,
                error=str(e),
                latency_ms=latency,
            )

        step_results: dict[str, Any] = {}

        # Step 2: Sequential topological execution
        for step_idx, step in enumerate(plan.steps, start=1):
            # Resolve variable references
            resolved_args: dict[str, Any] = {}
            for arg_key, arg_val in step.args.items():
                if isinstance(arg_val, str) and arg_val.startswith("$"):
                    ref_var = arg_val[1:].strip()
                    if ref_var not in mem.variables:
                        err_msg = f"Step {step_idx}: Missing verified variable '${ref_var}'"
                        mem.reject(err_msg)
                        latency = (time.perf_counter() - t0) * 1000.0
                        return PlanExecutionResult(
                            success=False,
                            final_var=plan.final_var,
                            final_result=None,
                            step_results=step_results,
                            working_memory=mem,
                            executed_steps=step_idx - 1,
                            error=err_msg,
                            failed_step_index=step_idx,
                            latency_ms=latency,
                        )
                    resolved_args[arg_key] = mem.variables[ref_var]
                elif isinstance(arg_val, list):
                    resolved_list: list[Any] = []
                    for item in arg_val:
                        if isinstance(item, str) and item.startswith("$"):
                            ref_var = item[1:].strip()
                            if ref_var not in mem.variables:
                                err_msg = f"Step {step_idx}: Missing verified variable '${ref_var}'"
                                mem.reject(err_msg)
                                latency = (time.perf_counter() - t0) * 1000.0
                                return PlanExecutionResult(
                                    success=False,
                                    final_var=plan.final_var,
                                    final_result=None,
                                    step_results=step_results,
                                    working_memory=mem,
                                    executed_steps=step_idx - 1,
                                    error=err_msg,
                                    failed_step_index=step_idx,
                                    latency_ms=latency,
                                )
                            resolved_list.append(mem.variables[ref_var])
                        else:
                            resolved_list.append(item)
                    resolved_args[arg_key] = resolved_list
                else:
                    resolved_args[arg_key] = arg_val

            # Unit consistency checks for arithmetic
            if step.operation == "add":
                # If both arguments reference earlier steps, check their units
                units_used = []
                for val in step.args.values():
                    if isinstance(val, str) and val.startswith("$"):
                        ref_var = val[1:].strip()
                        u = mem.step_units.get(ref_var)
                        if u:
                            units_used.append(u)
                if len(units_used) == 2 and units_used[0] != units_used[1]:
                    err_msg = f"Step {step_idx}: Unit mismatch in addition: '{units_used[0]}' vs '{units_used[1]}'"
                    mem.reject(err_msg)
                    latency = (time.perf_counter() - t0) * 1000.0
                    return PlanExecutionResult(
                        success=False,
                        final_var=plan.final_var,
                        final_result=None,
                        step_results=step_results,
                        working_memory=mem,
                        executed_steps=step_idx - 1,
                        error=err_msg,
                        failed_step_index=step_idx,
                        latency_ms=latency,
                    )

            # Deterministic execution via RuleExecutor
            exec_res = self.rule_executor.execute(step.operation, resolved_args, verify=True)
            if not exec_res.success or not exec_res.verified:
                err_msg = f"Step {step_idx} ({step.operation}) failed verification: {exec_res.error}"
                mem.reject(err_msg)
                latency = (time.perf_counter() - t0) * 1000.0
                return PlanExecutionResult(
                    success=False,
                    final_var=plan.final_var,
                    final_result=None,
                    step_results=step_results,
                    working_memory=mem,
                    executed_steps=step_idx - 1,
                    error=err_msg,
                    failed_step_index=step_idx,
                    latency_ms=latency,
                )

            # Bind step result into WorkingMemory
            step_results[step.output_var] = exec_res.result
            mem.bind_step_variable(
                var_name=step.output_var,
                value=exec_res.result,
                operation=step.operation,
                raw_args=step.args,
                resolved_args=resolved_args,
                units=step.units,
                step_index=step_idx,
            )

        # Retrieve final result
        final_val = mem.variables.get(plan.final_var or "")
        mem.bind_result(final_val, verified=True)
        latency = (time.perf_counter() - t0) * 1000.0

        return PlanExecutionResult(
            success=True,
            final_var=plan.final_var,
            final_result=final_val,
            step_results=step_results,
            working_memory=mem,
            executed_steps=len(plan.steps),
            latency_ms=latency,
        )
