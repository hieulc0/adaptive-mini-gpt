"""M36: Autonomous Capability-Gap Discovery and Self-Directed Curriculum Learning."""

from adaptive_mini_gpt.curriculum_discovery.curriculum_generator import (
    VerifiedCurriculumGenerator,
    VerifiedCurriculumItem,
)
from adaptive_mini_gpt.curriculum_discovery.gap_analyzer import (
    CapabilityGapAnalyzer,
)
from adaptive_mini_gpt.curriculum_discovery.multi_episode_learner import (
    EpisodeReport,
    MultiEpisodeLearner,
)
from adaptive_mini_gpt.curriculum_discovery.objective import (
    HypothesizedCause,
    LearningObjective,
)
from adaptive_mini_gpt.curriculum_discovery.objective_selector import (
    LargestGapObjectiveSelector,
    NeuralAssistedObjectiveSelector,
    ObjectiveSelectionPolicy,
    RoundRobinObjectiveSelector,
)
from adaptive_mini_gpt.curriculum_discovery.structural_evolution import (
    StructuralEvolutionGate,
    StructuralModificationProposal,
    StructuralModificationType,
    StructuralProposal,
)
from adaptive_mini_gpt.curriculum_discovery.uncertainty import (
    EvidenceFeatures,
    EnhancedMetacognitivePolicy,
    extract_evidence_features,
    train_enhanced_policy,
)

__all__ = [
    "CapabilityGapAnalyzer",
    "EnhancedMetacognitivePolicy",
    "EpisodeReport",
    "EvidenceFeatures",
    "HypothesizedCause",
    "LargestGapObjectiveSelector",
    "LearningObjective",
    "MultiEpisodeLearner",
    "NeuralAssistedObjectiveSelector",
    "ObjectiveSelectionPolicy",
    "RoundRobinObjectiveSelector",
    "StructuralEvolutionGate",
    "StructuralModificationProposal",
    "StructuralModificationType",
    "StructuralProposal",
    "VerifiedCurriculumGenerator",
    "VerifiedCurriculumItem",
    "extract_evidence_features",
    "train_enhanced_policy",
]
