"""Milestone M76: Causal Neural Skill Composition and Reusable Procedural Knowledge.

Exports:
- M75ScientificAuditor: Forensic scientific audit of M75 metrics and hindsight oracle diagnosis.
- PrimitiveSkillType, QualifiedPrimitiveRecord, PrimitiveSkillManager: Atomic procedural primitives.
- CompoundTaskType, CompoundProceduralExample, CompoundTaskDataset, CompoundTaskGenerator: Noncommutative compound tasks.
- NeuralCompositionPipeline, CompositionExecutionReport, CompositionStepDiagnostic: Cascaded two-stage composition.
- CausalCompositionBenchmark, CausalArmResult, CompoundBenchmarkReport: 9-arm causal composition evaluation.
- M76SafeguardsAuditor: Double SHA-256 base invariance, primitive freezing, arithmetic/logic probes, rollback.
"""

from __future__ import annotations

from adaptive_mini_gpt.neural_skill_composition.audit_m75 import (
    M75AuditRecord,
    M75ScientificAuditor,
    PairedComparisonInterval,
)
from adaptive_mini_gpt.neural_skill_composition.causal_composition_benchmark import (
    CausalArmResult,
    CausalCompositionBenchmark,
    CompoundBenchmarkReport,
)
from adaptive_mini_gpt.neural_skill_composition.compound_tasks import (
    CompoundProceduralExample,
    CompoundTaskDataset,
    CompoundTaskGenerator,
    CompoundTaskType,
    apply_primitive_mapping,
)
from adaptive_mini_gpt.neural_skill_composition.neural_composition_pipeline import (
    CompositionExecutionReport,
    CompositionStepDiagnostic,
    NeuralCompositionPipeline,
)
from adaptive_mini_gpt.neural_skill_composition.primitive_skills import (
    PrimitiveSkillManager,
    PrimitiveSkillType,
    QualifiedPrimitiveRecord,
)
from adaptive_mini_gpt.neural_skill_composition.safeguards import M76SafeguardsAuditor

__all__ = [
    "CausalArmResult",
    "CausalCompositionBenchmark",
    "CompositionExecutionReport",
    "CompositionStepDiagnostic",
    "CompoundBenchmarkReport",
    "CompoundProceduralExample",
    "CompoundTaskDataset",
    "CompoundTaskGenerator",
    "CompoundTaskType",
    "M75AuditRecord",
    "M75ScientificAuditor",
    "M76SafeguardsAuditor",
    "NeuralCompositionPipeline",
    "PairedComparisonInterval",
    "PrimitiveSkillManager",
    "PrimitiveSkillType",
    "QualifiedPrimitiveRecord",
    "apply_primitive_mapping",
]
