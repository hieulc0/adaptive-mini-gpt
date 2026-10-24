"""Milestone M74: Verified Neural Weight Learning and High-Information Generalization."""

from adaptive_mini_gpt.neural_weight_learning.audit_m73 import (
    M73AuditRecord,
    M73ScientificAuditor,
    TensorLineageItem,
)
from adaptive_mini_gpt.neural_weight_learning.lifelong_procedural_stream import (
    LifelongProceduralStreamEngine,
    PolicyStreamSummary,
    ProceduralStreamPolicy,
    StreamEpisodeResult,
)
from adaptive_mini_gpt.neural_weight_learning.pathway_instrumentation import (
    ForwardExecutionHook,
    NeuralPathwayInstrumenter,
    PathwayAuditRecord,
)
from adaptive_mini_gpt.neural_weight_learning.procedural_tasks import (
    ProceduralExample,
    ProceduralRuleFamily,
    ProceduralTaskDataset,
    ProceduralTaskGenerator,
    apply_procedural_rule,
)
from adaptive_mini_gpt.neural_weight_learning.safeguards import M74SafeguardsAuditor
from adaptive_mini_gpt.neural_weight_learning.sequence_adapter import (
    Rank16SequenceAdapter,
    Rank32SequenceAdapter,
)
from adaptive_mini_gpt.neural_weight_learning.weight_ablation_benchmark import (
    CausalAblationResult,
    CausalWeightAblationBenchmark,
)

__all__ = [
    "CausalAblationResult",
    "CausalWeightAblationBenchmark",
    "ForwardExecutionHook",
    "LifelongProceduralStreamEngine",
    "M73AuditRecord",
    "M73ScientificAuditor",
    "M74SafeguardsAuditor",
    "NeuralPathwayInstrumenter",
    "PathwayAuditRecord",
    "PolicyStreamSummary",
    "ProceduralExample",
    "ProceduralRuleFamily",
    "ProceduralStreamPolicy",
    "ProceduralTaskDataset",
    "ProceduralTaskGenerator",
    "Rank16SequenceAdapter",
    "Rank32SequenceAdapter",
    "StreamEpisodeResult",
    "TensorLineageItem",
    "apply_procedural_rule",
]
