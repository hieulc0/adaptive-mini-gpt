"""Milestone M80: Autonomous Neural Skill Acquisition and Transfer.

Provides:
- M79 forensic baseline audit and query reconciliation
- Bounded rank-16 and rank-8 evolvable sequence adapters
- Skill gap diagnostic detection (KNOWN_CONFIDENT, KNOWN_AMBIGUOUS, SKILL_GAP_UNSUPPORTED)
- Autonomous adapter acquisition engine with qualification gate (EM >= 0.85)
- Dynamic repertoire expansion with combinatorial plan hypothesis generation and rollback
- Transfer and composition benchmark suite
"""

from __future__ import annotations

from adaptive_mini_gpt.neural_skill_acquisition.adapter_acquisition_engine import (
    AcquisitionAuditResult,
    AcquisitionStepMetric,
    AdapterAcquisitionEngine,
    LearningInitMode,
)
from adaptive_mini_gpt.neural_skill_acquisition.adapter_architecture import (
    EvolvableSkillAdapterFactory,
    Rank8SequenceAdapter,
)
from adaptive_mini_gpt.neural_skill_acquisition.audit_m79 import (
    M79AuditReport,
    M79ForensicAuditor,
)
from adaptive_mini_gpt.neural_skill_acquisition.repertoire_expansion import (
    DynamicRepertoireManager,
)
from adaptive_mini_gpt.neural_skill_acquisition.skill_gap_detector import (
    SkillGapDetector,
    SkillGapDiagnosis,
    SkillGapState,
)
from adaptive_mini_gpt.neural_skill_acquisition.transfer_and_composition_benchmark import (
    BenchmarkTransferTaskGenerator,
    TransferAndCompositionBenchmark,
)

__all__ = [
    "M79AuditReport",
    "M79ForensicAuditor",
    "Rank8SequenceAdapter",
    "EvolvableSkillAdapterFactory",
    "SkillGapState",
    "SkillGapDiagnosis",
    "SkillGapDetector",
    "LearningInitMode",
    "AcquisitionStepMetric",
    "AcquisitionAuditResult",
    "AdapterAcquisitionEngine",
    "DynamicRepertoireManager",
    "BenchmarkTransferTaskGenerator",
    "TransferAndCompositionBenchmark",
]
