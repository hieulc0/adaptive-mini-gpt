"""Milestone M87: Long-Horizon Continual Learning, Memory Management, and Capacity Regulation.

Exports:
- M86ForensicAuditor: Phase 0 audit and baseline formalization.
- LifelongEvaluationMatrix: Formal R[i, j] matrix, ACC, BWT, FWT, F.
- ContinualTaskStream100Generator: 100-episode 4-phase lifelong curriculum.
- AutonomousMemoryManager: Capacity regulator enforcing the 20,000 parameter ceiling.
- LongHorizonContinualStudent: Extended persistent student engine.
- M87LongHorizonBenchmark: Master 6-arm causal benchmark.
"""

from adaptive_mini_gpt.long_horizon.audit_m86 import (
    M86AuditReport,
    M86ForensicAuditor,
)
from adaptive_mini_gpt.long_horizon.benchmark import (
    LongHorizonBenchmarkStrategy,
    M87LongHorizonBenchmark,
)
from adaptive_mini_gpt.long_horizon.evaluation_matrix import (
    ContinualEvaluationMetrics,
    LifelongEvaluationMatrix,
)
from adaptive_mini_gpt.long_horizon.long_horizon_student import (
    LongHorizonContinualStudent,
)
from adaptive_mini_gpt.long_horizon.memory_manager import (
    AutonomousMemoryManager,
    HolisticMemoryAuditReport,
)
from adaptive_mini_gpt.long_horizon.task_stream_100 import (
    ContinualTaskStream100Generator,
    StreamPhaseInfo,
)

__all__ = [
    "AutonomousMemoryManager",
    "ContinualEvaluationMetrics",
    "ContinualTaskStream100Generator",
    "HolisticMemoryAuditReport",
    "LifelongEvaluationMatrix",
    "LongHorizonBenchmarkStrategy",
    "LongHorizonContinualStudent",
    "M86AuditReport",
    "M86ForensicAuditor",
    "M87LongHorizonBenchmark",
    "StreamPhaseInfo",
]
