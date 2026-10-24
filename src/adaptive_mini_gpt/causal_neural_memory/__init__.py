"""Milestone M73: Causal Neural Memory Value and Cost-Efficient Lifelong Learning."""

from __future__ import annotations

from adaptive_mini_gpt.causal_neural_memory.audit_m72 import (
    M72ForensicAuditReport,
    M72ScientificAuditor,
)
from adaptive_mini_gpt.causal_neural_memory.causal_stream_engine import (
    CausalContinualStreamEngine,
    M73EpisodeRecord,
    M73StreamResult,
)
from adaptive_mini_gpt.causal_neural_memory.cost_accounting import (
    CostAndUtilityLedger,
    CostDecompositionBreakdown,
    PhysicalResourceRecord,
    TransparentEpisodeBudgetOutcome,
)
from adaptive_mini_gpt.causal_neural_memory.memory_manager import (
    ConsolidationDiagnosticResult,
    CostEfficientMemoryRegistry,
    ManagedSkillEntry,
    SkillState,
)
from adaptive_mini_gpt.causal_neural_memory.negative_transfer_detector import (
    NegativeTransferAuditor,
    RoutingErrorSummary,
    TaskRoutingComparison,
)
from adaptive_mini_gpt.causal_neural_memory.safeguards import (
    M73SafeguardsAuditor,
    M73SafeguardsReport,
)
from adaptive_mini_gpt.causal_neural_memory.task_stream import (
    build_m72_lifelong_stream_sequence,
    build_m73_open_world_stream_sequence,
    create_hierarchical_subsumption_env,
    create_modular_congruence_mod5_env,
)
from adaptive_mini_gpt.causal_neural_memory.weight_attribution import (
    NeuralWeightAttributionAuditor,
    TaskWeightAttributionReport,
    WeightAttributionRecord,
)

__all__ = [
    "CausalContinualStreamEngine",
    "ConsolidationDiagnosticResult",
    "CostAndUtilityLedger",
    "CostDecompositionBreakdown",
    "CostEfficientMemoryRegistry",
    "M72ForensicAuditReport",
    "M72ScientificAuditor",
    "M73EpisodeRecord",
    "M73SafeguardsAuditor",
    "M73SafeguardsReport",
    "M73StreamResult",
    "ManagedSkillEntry",
    "NegativeTransferAuditor",
    "NeuralWeightAttributionAuditor",
    "PhysicalResourceRecord",
    "RoutingErrorSummary",
    "SkillState",
    "TaskRoutingComparison",
    "TaskWeightAttributionReport",
    "TransparentEpisodeBudgetOutcome",
    "WeightAttributionRecord",
    "build_m72_lifelong_stream_sequence",
    "build_m73_open_world_stream_sequence",
    "create_hierarchical_subsumption_env",
    "create_modular_congruence_mod5_env",
]
