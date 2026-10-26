
from adaptive_mini_gpt.natural_language_learning_m103.causal_world_m103_g import (
    CausalItemM103_G,
    CausalObjectM103_G,
    CausalWorldM103_G,
)
from adaptive_mini_gpt.natural_language_learning_m103.neural_state_transition_m103_g import (
    CausalLatentPredictorM103_G,
    ContinuousStateEncoderM103_G,
    NeuralStateTransitionModuleM103_G,
)
from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m103_g import (
    ConceptLearnerM103_G,
    EvaluationScoresM103_G,
    PhaseG1ReportM103_G,
    PhaseG2ReportM103_G,
    PhaseG3ReportM103_G,
    PhaseG4ReportM103_G,
)
from adaptive_mini_gpt.natural_language_learning_m103.benchmark_m103_g import (
    BenchmarkM103_G,
    MilestoneGateM103_G,
    MilestoneReportM103_G,
)
from adaptive_mini_gpt.natural_language_learning_m103.benchmark_m103_f import (
    BenchmarkReportM103_F,
    BenchmarkRunnerM103_F,
    MilestoneGateM103_F,
)
from adaptive_mini_gpt.natural_language_learning_m103.capacity_auditor_m103_e1 import (
    CapacityAuditReportM103_E1,
    DynamicCapacityAuditorM103_E1,
)
from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m103_f import (
    ConceptLearnerM103_F,
    DatasetItemM103_F,
    EvaluationResultM103_F,
    LearningToLearnConditionResultM103_F,
    Phase1OptimizationReportM103_F,
    Phase2CorrectiveReportM103_F,
    Phase3LearningToLearnReportM103_F,
    StateTransitionStudyReportM103_F,
)
"""Milestone M103: Natural-Language Learning and Generalization."""

from adaptive_mini_gpt.natural_language_learning_m103.audit_m103 import (
    CapacityAuditReportM103,
    DynamicCapacityAuditorM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.benchmark_m103 import (
    M103BenchmarkReport,
    M103NaturalLanguageBenchmark,
)
from adaptive_mini_gpt.natural_language_learning_m103.benchmark_m103_b import (
    M103BBenchmarkReport,
    M103BNeuralLanguageBenchmark,
)
from adaptive_mini_gpt.natural_language_learning_m103.benchmark_m103_c import (
    BenchmarkReportM103_C,
    BenchmarkRunnerM103_C,
    MilestoneGateM103_C,
)
from adaptive_mini_gpt.natural_language_learning_m103.benchmark_m103_d import (
    BenchmarkReportM103_D,
    BenchmarkRunnerM103_D,
    MilestoneGateM103_D,
)
from adaptive_mini_gpt.natural_language_learning_m103.benchmark_m103_e import (
    BenchmarkReportM103_E,
    BenchmarkRunnerM103_E,
    MilestoneGateM103_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.capacity_auditor_m103_e import (
    CapacityAuditReportM103_E,
    DynamicCapacityAuditorM103_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.comparator_m103 import (
    ModeEvaluationResultM103,
    MultiModeComparisonReportM103,
    NaturalLanguageComparatorM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m103_d import (
    ConceptLearnerM103_D,
    DatasetItemM103_D,
    EvaluationResultM103_D,
    PhaseD1OptimizationReport,
    PhaseD3SimulationReport,
    PhaseD4CorrectiveReport,
    RepresentationComparisonM103_D,
)
from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m103_e import (
    ConceptLearnerM103_E,
    DatasetItemM103_E,
    EvaluationResultM103_E,
    PhaseE1OptimizationReport,
    PhaseE2SimulationReport,
    PhaseE3CorrectiveReport,
    PhaseE4LearningToLearnReport,
    RepresentationComparisonM103_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.continual_concept_learner_m103_c import (
    ContinualConceptLearnerM103_C,
    ContinualLifelongResultM103_C,
    SingleConceptResultM103_C,
)
from adaptive_mini_gpt.natural_language_learning_m103.continual_language_student_m103_b import (
    ContinualLanguageStudentM103_B,
    ContinualLifelongReportM103_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.convergence_auditor_m103_b import (
    ConvergenceAuditReportM103_B,
    ConvergenceAuditorM103_B,
    ConvergenceRegimeResultM103_B,
    ConvergenceSampleOutputM103_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.curriculum_m103 import (
    QAItemM103,
    RuleGoalM103,
    TextCurriculumGeneratorM103,
    TextRuleDomainM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.generalization_evaluator_m103_c import (
    EvaluationMetricsM103_C,
    GeneralizationDiagnosticReportM103_C,
    GeneralizationEvaluatorM103_C,
)
from adaptive_mini_gpt.natural_language_learning_m103.language_pretrainer_m103_b import (
    LanguagePretrainerM103_B,
    PretrainingReportM103_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.number_tokenizer_m103_d import (
    NumberAwareTokenizerM103_D,
)
from adaptive_mini_gpt.natural_language_learning_m103.progressive_rule_learner_m103_b import (
    ProgressiveConditionResultM103_B,
    ProgressiveLearningReportM103_B,
    ProgressiveRuleLearnerM103_B,
    StepBudgetCheckpointM103_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.routing_diagnostic_m103 import (
    RoutingDiagnosticM103,
    RoutingDiagnosticReportM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.world_simulator_m103_c import (
    ConceptDatasetM103_C,
    ConceptFamilyM103_C,
    SimulatorItemM103_C,
    WorldEntityM103_C,
    WorldSimulatorM103_C,
)

__all__ = [
    "BenchmarkReportM103_F",
    "BenchmarkRunnerM103_F",
    "CapacityAuditReportM103_E1",
    "ConceptLearnerM103_F",
    "DatasetItemM103_F",
    "DynamicCapacityAuditorM103_E1",
    "EvaluationResultM103_F",
    "LearningToLearnConditionResultM103_F",
    "MilestoneGateM103_F",
    "Phase1OptimizationReportM103_F",
    "Phase2CorrectiveReportM103_F",
    "Phase3LearningToLearnReportM103_F",
    "StateTransitionStudyReportM103_F",
    "BenchmarkReportM103_C",
    "BenchmarkReportM103_D",
    "BenchmarkReportM103_E",
    "BenchmarkRunnerM103_C",
    "BenchmarkRunnerM103_D",
    "BenchmarkRunnerM103_E",
    "ByteTextTokenizerM103",
    "CapacityAuditReportM103",
    "CapacityAuditReportM103_E",
    "ConceptDatasetM103_C",
    "ConceptFamilyM103_C",
    "ConceptLearnerM103_D",
    "ConceptLearnerM103_E",
    "ContinualConceptLearnerM103_C",
    "ContinualLanguageStudentM103_B",
    "ContinualLifelongReportM103_B",
    "ContinualLifelongResultM103_C",
    "ConvergenceAuditReportM103_B",
    "ConvergenceAuditorM103_B",
    "ConvergenceRegimeResultM103_B",
    "ConvergenceSampleOutputM103_B",
    "DatasetItemM103_D",
    "DatasetItemM103_E",
    "DynamicCapacityAuditorM103",
    "DynamicCapacityAuditorM103_E",
    "EvaluationMetricsM103_C",
    "EvaluationResultM103_D",
    "EvaluationResultM103_E",
    "GeneralizationDiagnosticReportM103_C",
    "GeneralizationEvaluatorM103_C",
    "LanguagePretrainerM103_B",
    "M103BBenchmarkReport",
    "M103BNeuralLanguageBenchmark",
    "M103BenchmarkReport",
    "M103NaturalLanguageBenchmark",
    "MilestoneGateM103_C",
    "MilestoneGateM103_D",
    "MilestoneGateM103_E",
    "ModeEvaluationResultM103",
    "MultiModeComparisonReportM103",
    "NaturalLanguageComparatorM103",
    "NumberAwareTokenizerM103_D",
    "PhaseD1OptimizationReport",
    "PhaseD3SimulationReport",
    "PhaseD4CorrectiveReport",
    "PhaseE1OptimizationReport",
    "PhaseE2SimulationReport",
    "PhaseE3CorrectiveReport",
    "PhaseE4LearningToLearnReport",
    "PretrainingReportM103_B",
    "ProgressiveConditionResultM103_B",
    "ProgressiveLearningReportM103_B",
    "ProgressiveRuleLearnerM103_B",
    "QAItemM103",
    "RepresentationComparisonM103_D",
    "RepresentationComparisonM103_E",
    "RoutingDiagnosticM103",
    "RoutingDiagnosticReportM103",
    "RuleGoalM103",
    "SimulatorItemM103_C",
    "SingleConceptResultM103_C",
    "StepBudgetCheckpointM103_B",
    "TextCurriculumGeneratorM103",
    "TextRuleDomainM103",
    "TextStudentM103",
    "WorldEntityM103_C",
    "WorldSimulatorM103_C",
]
