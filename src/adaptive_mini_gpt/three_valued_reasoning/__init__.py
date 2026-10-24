"""Three-Valued Reasoning & Representation Sufficiency Module for Milestone M65."""
from __future__ import annotations

from adaptive_mini_gpt.three_valued_reasoning.architectures import (
    LinearFeatureProbe,
    M65ArchitectureVariant,
    NonlinearResidualHead,
    Rank16ResidualAdapter,
    Rank32ResidualAdapter,
    create_model_for_arm,
)
from adaptive_mini_gpt.three_valued_reasoning.audit import (
    ConfusionMatrixReconciliation,
    ForensicPathwayTrace,
    IdentifiabilityConflict,
    M64ForensicAuditReport,
    M64ForensicAuditor,
)
from adaptive_mini_gpt.three_valued_reasoning.engine import (
    FewShotBudgetPoint,
    ThreeValuedAdaptationTrajectory,
    ThreeValuedReasoningEngine,
    TriStateMetrics,
    compute_standard_horizon_auc,
)
from adaptive_mini_gpt.three_valued_reasoning.evaluator import (
    M65MetaLearningArm,
    ThreeValuedReptileLearner,
    train_three_valued_multitask_adapter,
)
from adaptive_mini_gpt.three_valued_reasoning.safeguards import (
    M65SafeguardAuditor,
    M65SafeguardReport,
)
from adaptive_mini_gpt.three_valued_reasoning.semantics import (
    DatasetValidationReport,
    GroundedThreeValuedEnvironment,
    ObservableEvidenceContext,
    ThreeValuedDatasetValidator,
    TriValuedState,
)
from adaptive_mini_gpt.three_valued_reasoning.smoke_test import (
    PathwaySmokeTestReport,
    ThreeClassLearningPathwayVerifier,
)
from adaptive_mini_gpt.three_valued_reasoning.task_suite import (
    build_m65_new_sealed_suite,
    create_new_sealed_autonomous_rail_grid,
    create_new_sealed_consortium_smart_contract,
    create_new_sealed_epigenetic_chromatin_lattice,
    create_new_sealed_metabolic_inhibitor_tree,
    create_new_sealed_quantum_teleportation_dag,
)

__all__ = [
    "ConfusionMatrixReconciliation",
    "DatasetValidationReport",
    "FewShotBudgetPoint",
    "ForensicPathwayTrace",
    "GroundedThreeValuedEnvironment",
    "IdentifiabilityConflict",
    "LinearFeatureProbe",
    "M64ForensicAuditReport",
    "M64ForensicAuditor",
    "M65ArchitectureVariant",
    "M65MetaLearningArm",
    "M65SafeguardAuditor",
    "M65SafeguardReport",
    "NonlinearResidualHead",
    "ObservableEvidenceContext",
    "PathwaySmokeTestReport",
    "Rank16ResidualAdapter",
    "Rank32ResidualAdapter",
    "ThreeClassLearningPathwayVerifier",
    "ThreeValuedAdaptationTrajectory",
    "ThreeValuedDatasetValidator",
    "ThreeValuedReasoningEngine",
    "ThreeValuedReptileLearner",
    "TriStateMetrics",
    "TriValuedState",
    "build_m65_new_sealed_suite",
    "compute_standard_horizon_auc",
    "create_model_for_arm",
    "create_new_sealed_autonomous_rail_grid",
    "create_new_sealed_consortium_smart_contract",
    "create_new_sealed_epigenetic_chromatin_lattice",
    "create_new_sealed_metabolic_inhibitor_tree",
    "create_new_sealed_quantum_teleportation_dag",
    "train_three_valued_multitask_adapter",
]
