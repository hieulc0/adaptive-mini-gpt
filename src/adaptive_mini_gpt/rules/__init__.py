"""Rules, Deterministic Execution, and Reasoning Loop Package (M14)."""

from adaptive_mini_gpt.rules.dataset import (
    RuleProblem,
    generate_rule_instruction_corpus,
    generate_rule_problems,
    get_rule_split,
)
from adaptive_mini_gpt.rules.executor import (
    RuleExecutionResult,
    RuleExecutor,
)
from adaptive_mini_gpt.rules.loop import (
    AttemptRecord,
    ReasoningController,
    ReasoningResult,
    format_rule_call,
    parse_rule_call,
)
from adaptive_mini_gpt.rules.registry import (
    RULE_REGISTRY,
    RuleError,
    RuleSpecification,
    SchemaValidationError,
    VerificationError,
    execute_add,
    execute_average_speed,
    execute_compare,
    execute_reverse,
    get_rule,
    list_allowed_rules,
    verify_add,
    verify_average_speed,
    verify_compare,
    verify_reverse,
)

__all__ = [
    "AttemptRecord",
    "RULE_REGISTRY",
    "ReasoningController",
    "ReasoningResult",
    "RuleError",
    "RuleExecutionResult",
    "RuleExecutor",
    "RuleProblem",
    "RuleSpecification",
    "SchemaValidationError",
    "VerificationError",
    "execute_add",
    "execute_average_speed",
    "execute_compare",
    "execute_reverse",
    "format_rule_call",
    "generate_rule_instruction_corpus",
    "generate_rule_problems",
    "get_rule",
    "get_rule_split",
    "list_allowed_rules",
    "parse_rule_call",
    "verify_add",
    "verify_average_speed",
    "verify_compare",
    "verify_reverse",
]
