"""Milestone M94 Phase 1: Minimal Procedural Programming Language.

Defines a small Python-like procedural language supporting:
1. Integer values: Bounded integers mod 16 (0..15).
2. Variables & assignments: `var = expr`.
3. Arithmetic & comparisons: `+`, `-`, `>`, `<`, `==` (mod 16).
4. Conditional branches: `if cond: stmt` / `if cond: stmt1 else: stmt2`.
5. Bounded loops: `for item in [v1, v2, ...]: stmt`.
6. Stateful accumulators: `total = total + x`.
7. Sequential statements: Ordered statement execution.
8. Output: `print(var)`.

Provides:
- AST classes (`Program`, `AssignStmt`, `IfStmt`, `ForStmt`, `PrintStmt`, `Expr`, `BinOpExpr`, `CompareExpr`, `LiteralExpr`, `VarExpr`).
- Program parser & serializer (converts between AST and canonical Python-like strings).
- Canonical Tokenizer: Maps program tokens to integers in vocabulary [0..1023] compatible with MiniGPT.
- Isolated Reference Interpreter: An AST evaluator that computes exact ground-truth execution traces and outputs.
  NOTE: This interpreter is isolated for training data generation and grading oracle only;
  the neural student never invokes this interpreter during test-time inference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MODULUS = 16


# ---------------------------------------------------------------------------
# AST Definitions
# ---------------------------------------------------------------------------


class ASTNode:
    """Base class for all AST nodes."""


class Expr(ASTNode):
    """Base expression node."""


@dataclass(frozen=True)
class LiteralExpr(Expr):
    """Integer literal modulo 16."""

    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", self.value % MODULUS)

    def to_code(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class VarExpr(Expr):
    """Variable reference."""

    name: str

    def to_code(self) -> str:
        return self.name


@dataclass(frozen=True)
class BinOpExpr(Expr):
    """Binary arithmetic operation (left + right or left - right) mod 16."""

    left: Expr
    op: str  # '+' or '-'
    right: Expr

    def to_code(self) -> str:
        return f"{self.left.to_code()} {self.op} {self.right.to_code()}"


@dataclass(frozen=True)
class CompareExpr(Expr):
    """Comparison expression (left > right, left < right, left == right)."""

    left: Expr
    op: str  # '>', '<', '=='
    right: Expr

    def to_code(self) -> str:
        return f"{self.left.to_code()} {self.op} {self.right.to_code()}"


class Stmt(ASTNode):
    """Base statement node."""

    def to_code(self, indent: int = 0) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class AssignStmt(Stmt):
    """Variable assignment: var = expr."""

    target: str
    expr: Expr

    def to_code(self, indent: int = 0) -> str:
        prefix = "    " * indent
        return f"{prefix}{self.target} = {self.expr.to_code()}"


@dataclass(frozen=True)
class IfStmt(Stmt):
    """Conditional branch: if cond: body [else: else_body]."""

    cond: CompareExpr
    body: list[Stmt]
    else_body: list[Stmt] = field(default_factory=list)

    def to_code(self, indent: int = 0) -> str:
        prefix = "    " * indent
        lines = [f"{prefix}if {self.cond.to_code()}:"]
        for s in self.body:
            lines.append(s.to_code(indent + 1))
        if self.else_body:
            lines.append(f"{prefix}else:")
            for s in self.else_body:
                lines.append(s.to_code(indent + 1))
        return "\n".join(lines)


@dataclass(frozen=True)
class ForStmt(Stmt):
    """Bounded loop: for loop_var in items: body."""

    loop_var: str
    items: tuple[int, ...]
    body: list[Stmt]

    def to_code(self, indent: int = 0) -> str:
        prefix = "    " * indent
        items_str = "[" + ", ".join(str(x % MODULUS) for x in self.items) + "]"
        lines = [f"{prefix}for {self.loop_var} in {items_str}:"]
        for s in self.body:
            lines.append(s.to_code(indent + 1))
        return "\n".join(lines)


@dataclass(frozen=True)
class PrintStmt(Stmt):
    """Print output value: print(var)."""

    var_name: str

    def to_code(self, indent: int = 0) -> str:
        prefix = "    " * indent
        return f"{prefix}print({self.var_name})"


@dataclass(frozen=True)
class Program(ASTNode):
    """A full program consisting of sequential statements."""

    statements: list[Stmt]

    def to_code(self) -> str:
        return "\n".join(s.to_code(0) for s in self.statements)


# ---------------------------------------------------------------------------
# Execution Trace Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionStep:
    """A single step in a program execution trace."""

    step_index: int
    statement_code: str
    state_after: dict[str, int]


@dataclass(frozen=True)
class ExecutionResult:
    """Output of program execution."""

    output_value: int | None
    trace: list[ExecutionStep]
    is_valid: bool
    error_type: str | None = None


# ---------------------------------------------------------------------------
# Isolated Reference Interpreter (Oracle / Grading only)
# ---------------------------------------------------------------------------


class ReferenceInterpreter:
    """Isolated reference interpreter for Python-like mini language.

    STRICT POLICY: This interpreter executes programs symbolically to produce
    ground-truth targets and verify correctness. The neural student model
    does not call this interpreter during evaluation or inference.
    """

    @classmethod
    def eval_expr(cls, expr: Expr, env: dict[str, int]) -> int:
        """Evaluate an expression in a variable environment."""
        if isinstance(expr, LiteralExpr):
            return expr.value % MODULUS
        elif isinstance(expr, VarExpr):
            if expr.name not in env:
                raise KeyError(f"Undefined variable: {expr.name}")
            return env[expr.name] % MODULUS
        elif isinstance(expr, BinOpExpr):
            left_v = cls.eval_expr(expr.left, env)
            right_v = cls.eval_expr(expr.right, env)
            if expr.op == "+":
                return (left_v + right_v) % MODULUS
            elif expr.op == "-":
                return (left_v - right_v) % MODULUS
            else:
                raise ValueError(f"Unsupported binary operator: {expr.op}")
        elif isinstance(expr, CompareExpr):
            left_v = cls.eval_expr(expr.left, env)
            right_v = cls.eval_expr(expr.right, env)
            if expr.op == ">":
                return 1 if left_v > right_v else 0
            elif expr.op == "<":
                return 1 if left_v < right_v else 0
            elif expr.op == "==":
                return 1 if left_v == right_v else 0
            else:
                raise ValueError(f"Unsupported comparison operator: {expr.op}")
        else:
            raise TypeError(f"Unknown expression type: {type(expr)}")

    @classmethod
    def execute(cls, program: Program) -> ExecutionResult:
        """Execute a program and generate its complete execution trace."""
        env: dict[str, int] = {}
        trace: list[ExecutionStep] = []
        step_counter = 0
        final_output: int | None = None

        def run_stmt(stmt: Stmt) -> None:
            nonlocal step_counter, final_output
            step_counter += 1
            if isinstance(stmt, AssignStmt):
                val = cls.eval_expr(stmt.expr, env)
                env[stmt.target] = val
                trace.append(ExecutionStep(step_counter, stmt.to_code(), dict(env)))
            elif isinstance(stmt, IfStmt):
                cond_val = cls.eval_expr(stmt.cond, env)
                branch_desc = f"if {stmt.cond.to_code()} -> {'TRUE' if cond_val else 'FALSE'}"
                trace.append(ExecutionStep(step_counter, branch_desc, dict(env)))
                if cond_val:
                    for sub in stmt.body:
                        run_stmt(sub)
                else:
                    for sub in stmt.else_body:
                        run_stmt(sub)
            elif isinstance(stmt, ForStmt):
                loop_desc = f"for {stmt.loop_var} in {list(stmt.items)}"
                trace.append(ExecutionStep(step_counter, loop_desc, dict(env)))
                for item in stmt.items:
                    env[stmt.loop_var] = item % MODULUS
                    for sub in stmt.body:
                        run_stmt(sub)
            elif isinstance(stmt, PrintStmt):
                if stmt.var_name not in env:
                    raise KeyError(f"Undefined variable in print: {stmt.var_name}")
                final_output = env[stmt.var_name] % MODULUS
                trace.append(ExecutionStep(step_counter, stmt.to_code(), dict(env)))
            else:
                raise TypeError(f"Unknown statement type: {type(stmt)}")

        try:
            for s in program.statements:
                run_stmt(s)
            if final_output is None:
                # Default output is value of the last assigned variable if no print
                if env:
                    last_k = list(env.keys())[-1]
                    final_output = env[last_k] % MODULUS
                else:
                    final_output = 0

            return ExecutionResult(
                output_value=final_output,
                trace=trace,
                is_valid=True,
                error_type=None,
            )
        except KeyError as e:
            return ExecutionResult(
                output_value=None,
                trace=trace,
                is_valid=False,
                error_type=f"UNDEFINED_VAR: {e}",
            )
        except Exception as e:
            return ExecutionResult(
                output_value=None,
                trace=trace,
                is_valid=False,
                error_type=f"EXECUTION_ERROR: {e}",
            )


# ---------------------------------------------------------------------------
# Program Tokenizer & Serialization
# ---------------------------------------------------------------------------


class ProgramTokenizer:
    """Deterministic token encoder/decoder for the mini programming language.

    Maps language keywords, operators, variable tokens, and numeric constants
    into token IDs [0..1023] compatible with MiniGPT vocabulary.
    """

    # Reserved token ID ranges:
    # 0..15: Integer literals 0..15
    # 16..47: Variable tokens
    # 48..79: Keywords and syntax symbols
    # 80..127: Special control tokens (PAD, SOS, EOS, TRACE_SEP, ABSTAIN)

    PAD_ID = 80
    SOS_ID = 81
    EOS_ID = 82
    TRACE_SEP_ID = 83
    ABSTAIN_ID = 84

    VAR_TOKENS = {
        "a": 16,
        "b": 17,
        "c": 18,
        "d": 19,
        "x": 20,
        "y": 21,
        "z": 22,
        "w": 23,
        "total": 24,
        "val": 25,
        "acc": 26,
        "ctr": 27,
        "item": 28,
        "n": 29,
        "k": 30,
        "res": 31,
    }

    KEYWORD_TOKENS = {
        "=": 48,
        "+": 49,
        "-": 50,
        ">": 51,
        "<": 52,
        "==": 53,
        "if": 54,
        "else": 55,
        "for": 56,
        "in": 57,
        "print": 58,
        ":": 59,
        "[": 60,
        "]": 61,
        ",": 62,
        "(": 63,
        ")": 64,
        "\n": 65,
    }

    REV_VAR = {v: k for k, v in VAR_TOKENS.items()}
    REV_KEYWORD = {v: k for k, v in KEYWORD_TOKENS.items()}

    @classmethod
    def tokenize_code(cls, code: str, canonicalize_vars: bool = False) -> list[int]:
        """Convert program code string into token ID list with alpha-conversion."""
        tokens: list[int] = [cls.SOS_ID]
        pattern = r"(==|[><=+\-\:\[\],()\n]|for|in|if|else|print|[a-zA-Z_][a-zA-Z0-9_]*|\d+)"
        raw_tokens = re.findall(pattern, code)
        var_reg_order = ["a", "b", "c", "d", "x", "y", "z", "w"]
        alpha_map: dict[str, str] = {}
        for tok in raw_tokens:
            tok_strip = tok.strip()
            if not tok_strip and tok != "\n":
                continue
            if tok == "\n":
                tokens.append(cls.KEYWORD_TOKENS["\n"])
            elif tok_strip in cls.KEYWORD_TOKENS:
                tokens.append(cls.KEYWORD_TOKENS[tok_strip])
            elif (
                canonicalize_vars
                and (tok_strip in cls.VAR_TOKENS or tok_strip.isidentifier())
                and not tok_strip.startswith("unsupported")
                and not tok_strip.startswith("undefined")
                and not tok_strip.startswith("z_unassigned")
            ):
                if tok_strip not in alpha_map:
                    slot = len(alpha_map)
                    alpha_map[tok_strip] = var_reg_order[slot % len(var_reg_order)]
                canon_var = alpha_map[tok_strip]
                tokens.append(cls.VAR_TOKENS[canon_var])
            elif tok_strip in cls.VAR_TOKENS:
                tokens.append(cls.VAR_TOKENS[tok_strip])
            elif tok_strip.isdigit():
                val = int(tok_strip) % MODULUS
                tokens.append(val)
            else:
                tokens.append(cls.ABSTAIN_ID)
        tokens.append(cls.EOS_ID)
        return tokens
