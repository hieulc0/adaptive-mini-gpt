"""Milestone M66: Evidence-Grounded Modular Skill Reuse and Controlled Autonomous Learning."""
from __future__ import annotations

from adaptive_mini_gpt.modular_skill_learning.audit import (
    M65AuditReport,
    M65EvidenceAndLineageAuditor,
)
from adaptive_mini_gpt.modular_skill_learning.candidate_acquisition import (
    CandidateAcquisitionReport,
    ControlledSkillAcquisitionEngine,
)
from adaptive_mini_gpt.modular_skill_learning.hidden_extraction import extract_model_hidden
from adaptive_mini_gpt.modular_skill_learning.learning_policy import (
    LearningPolicy,
    LearningPolicyAblationEngine,
    PolicyEvaluationResult,
)
from adaptive_mini_gpt.modular_skill_learning.repertoire import (
    ModularSkillRepertoire,
    QualifiedSkillRecord,
    RepertoireBenchmarkResult,
    RepertoireRoutingArm,
)
from adaptive_mini_gpt.modular_skill_learning.safeguards import (
    M66SafeguardReport,
    M66SafeguardsAuditor,
)
from adaptive_mini_gpt.modular_skill_learning.sequential_transfer import (
    SequentialEpisodeExperiment,
    SequentialEpisodeOutcome,
    SequentialTransferArm,
)
from adaptive_mini_gpt.modular_skill_learning.task_suite import (
    build_m66_complete_task_suite,
    create_ep1_autonomous_rail_grid,
    create_ep1_epigenetic_chromatin_lattice,
    create_ep2_autonomous_rail_grid,
    create_ep2_epigenetic_chromatin_lattice,
    create_source_consortium_bipartite,
    create_source_dag_transitive,
    create_source_metabolic_inhibitor,
    create_unrelated_control_quantum_dag,
)
from adaptive_mini_gpt.modular_skill_learning.uncertainty_diagnostic import (
    EpistemicSeparationTrace,
    EpistemicUncertaintyDiagnostic,
    UncertaintyCategory,
)

__all__ = [
    "extract_model_hidden",
    "M65EvidenceAndLineageAuditor",
    "M65AuditReport",
    "ModularSkillRepertoire",
    "QualifiedSkillRecord",
    "RepertoireRoutingArm",
    "RepertoireBenchmarkResult",
    "EpistemicUncertaintyDiagnostic",
    "UncertaintyCategory",
    "EpistemicSeparationTrace",
    "ControlledSkillAcquisitionEngine",
    "CandidateAcquisitionReport",
    "SequentialEpisodeExperiment",
    "SequentialTransferArm",
    "SequentialEpisodeOutcome",
    "LearningPolicyAblationEngine",
    "LearningPolicy",
    "PolicyEvaluationResult",
    "M66SafeguardsAuditor",
    "M66SafeguardReport",
    "create_source_dag_transitive",
    "create_source_metabolic_inhibitor",
    "create_source_consortium_bipartite",
    "create_ep1_epigenetic_chromatin_lattice",
    "create_ep2_epigenetic_chromatin_lattice",
    "create_ep1_autonomous_rail_grid",
    "create_ep2_autonomous_rail_grid",
    "create_unrelated_control_quantum_dag",
    "build_m66_complete_task_suite",
]
