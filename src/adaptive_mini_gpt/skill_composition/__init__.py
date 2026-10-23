"""Milestone M55: Representation-Preserving Skill Composition Package."""
from __future__ import annotations

from adaptive_mini_gpt.skill_composition.audit import (
    M54MetricReproduction,
    M54ScientificAuditReport,
    M54ScientificAuditor,
    OracleDiscrepancyAnalysis,
)
from adaptive_mini_gpt.skill_composition.composite_adapter import (
    GatedResidualSkillAdapter,
    ResidualSkillAdapter,
)
from adaptive_mini_gpt.skill_composition.composition_transfer import (
    CompositionArmTrajectory,
    CompositionBudgetStep,
    CompositionTransferEngine,
    MultiWorldCompositionReport,
)
from adaptive_mini_gpt.skill_composition.drift import (
    RepresentationDriftAnalyzer,
    RepresentationDriftMeasurement,
)
from adaptive_mini_gpt.skill_composition.environment import (
    CompositionBenchmarkEnvironments,
    build_m55_environment_suite,
)
from adaptive_mini_gpt.skill_composition.epistemic_dense_diagnostic import (
    DenseCliqueDiagnosticReport,
    DenseStrategyResult,
    EpistemicDenseDiagnosticAuditor,
)
from adaptive_mini_gpt.skill_composition.retention import (
    CompositionRetentionAuditor,
    SourceRetentionAuditResult,
)

__all__ = [
    "M54MetricReproduction",
    "OracleDiscrepancyAnalysis",
    "M54ScientificAuditReport",
    "M54ScientificAuditor",
    "RepresentationDriftMeasurement",
    "RepresentationDriftAnalyzer",
    "ResidualSkillAdapter",
    "GatedResidualSkillAdapter",
    "CompositionBudgetStep",
    "CompositionArmTrajectory",
    "MultiWorldCompositionReport",
    "CompositionTransferEngine",
    "CompositionBenchmarkEnvironments",
    "build_m55_environment_suite",
    "SourceRetentionAuditResult",
    "CompositionRetentionAuditor",
    "DenseStrategyResult",
    "DenseCliqueDiagnosticReport",
    "EpistemicDenseDiagnosticAuditor",
]
