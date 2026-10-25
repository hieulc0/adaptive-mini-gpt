"""Milestone M100: Integrated Autonomous Student and Selective Backbone Plasticity."""

from __future__ import annotations

from adaptive_mini_gpt.integrated_autonomous_student import (
    audit_m99,
    autonomous_learning_cycle,
    benchmark_m100,
    future_learning_evaluator_m100,
    integrated_student_m100,
    multi_subject_curriculum,
    structural_evolution_engine,
    two_speed_plasticity_manager,
)
from adaptive_mini_gpt.integrated_autonomous_student.audit_m99 import (
    M99AuditResult,
    M99ForensicAuditor,
)
from adaptive_mini_gpt.integrated_autonomous_student.autonomous_learning_cycle import (
    AutonomousLearningCycleM100,
    CycleStepResult,
)
from adaptive_mini_gpt.integrated_autonomous_student.benchmark_m100 import (
    M100IntegratedAutonomousStudentBenchmark,
)
from adaptive_mini_gpt.integrated_autonomous_student.future_learning_evaluator_m100 import (
    ConditionResultM100,
    FutureLearningEvaluatorM100,
    M100CausalEvaluationResult,
)
from adaptive_mini_gpt.integrated_autonomous_student.integrated_student_m100 import (
    IntegratedAutonomousStudentM100,
)
from adaptive_mini_gpt.integrated_autonomous_student.multi_subject_curriculum import (
    DomainM100,
    IndependentCurriculumVerifier,
    LearningGoalM100,
    MultiSubjectCurriculumGenerator,
)
from adaptive_mini_gpt.integrated_autonomous_student.structural_evolution_engine import (
    AccountingM100,
    CapacityTierM100,
    RankNResidualAdapter,
    StructuralEvolutionEngine,
)
from adaptive_mini_gpt.integrated_autonomous_student.two_speed_plasticity_manager import (
    ConsolidationReport,
    PlasticityMode,
    TwoSpeedPlasticityManager,
)

__all__ = [
    "audit_m99",
    "integrated_student_m100",
    "two_speed_plasticity_manager",
    "structural_evolution_engine",
    "multi_subject_curriculum",
    "autonomous_learning_cycle",
    "future_learning_evaluator_m100",
    "benchmark_m100",
    "M99ForensicAuditor",
    "M99AuditResult",
    "IntegratedAutonomousStudentM100",
    "TwoSpeedPlasticityManager",
    "PlasticityMode",
    "ConsolidationReport",
    "StructuralEvolutionEngine",
    "CapacityTierM100",
    "AccountingM100",
    "RankNResidualAdapter",
    "MultiSubjectCurriculumGenerator",
    "DomainM100",
    "LearningGoalM100",
    "IndependentCurriculumVerifier",
    "AutonomousLearningCycleM100",
    "CycleStepResult",
    "FutureLearningEvaluatorM100",
    "ConditionResultM100",
    "M100CausalEvaluationResult",
    "M100IntegratedAutonomousStudentBenchmark",
]
