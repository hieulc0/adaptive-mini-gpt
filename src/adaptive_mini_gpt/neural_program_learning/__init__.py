"""Milestone M94: Neural Program Learning, Stateful Execution & Algorithmic Generalization."""

from adaptive_mini_gpt.neural_program_learning.audit_m93 import (
    M93ForensicAuditor,
    M93ForensicAuditReport,
)
from adaptive_mini_gpt.neural_program_learning.benchmark_m94 import (
    M94BenchmarkReport,
    M94NeuralProgramBenchmark,
    SeedM94Result,
)
from adaptive_mini_gpt.neural_program_learning.mini_language import (
    AssignStmt,
    BinOpExpr,
    CompareExpr,
    ExecutionResult,
    ExecutionStep,
    ForStmt,
    IfStmt,
    LiteralExpr,
    PrintStmt,
    Program,
    ProgramTokenizer,
    ReferenceInterpreter,
    Stmt,
    VarExpr,
)
from adaptive_mini_gpt.neural_program_learning.neural_program_executor import (
    NeuralExecutionTrace,
    NeuralProgramExecutor,
    ProgramReasoningSubstrate,
)
from adaptive_mini_gpt.neural_program_learning.program_acquisition import (
    AcquisitionResult,
    AutonomousProgramLearner,
)
from adaptive_mini_gpt.neural_program_learning.program_curriculum import (
    M94ProgramCurriculum,
    ProgramCurriculumGenerator,
    ProgramExample,
    ProgramStageSuite,
)

__all__ = [
    "M93ForensicAuditor",
    "M93ForensicAuditReport",
    "M94BenchmarkReport",
    "M94NeuralProgramBenchmark",
    "SeedM94Result",
    "AssignStmt",
    "BinOpExpr",
    "CompareExpr",
    "ExecutionResult",
    "ExecutionStep",
    "ForStmt",
    "IfStmt",
    "LiteralExpr",
    "PrintStmt",
    "Program",
    "ProgramTokenizer",
    "ReferenceInterpreter",
    "Stmt",
    "VarExpr",
    "NeuralExecutionTrace",
    "NeuralProgramExecutor",
    "ProgramReasoningSubstrate",
    "AcquisitionResult",
    "AutonomousProgramLearner",
    "M94ProgramCurriculum",
    "ProgramCurriculumGenerator",
    "ProgramExample",
    "ProgramStageSuite",
]
