"""Milestone M90: Foundational Concept Learning & Instruction-Conditioned Neural Transfer."""

from adaptive_mini_gpt.concept_instruction.benchmark_m90 import (
    M90InstructionBenchmark,
)
from adaptive_mini_gpt.concept_instruction.capacity_feasibility import (
    StageACapacityBenchmark,
)
from adaptive_mini_gpt.concept_instruction.concept_adapter import (
    InstructionConceptTrunk,
)
from adaptive_mini_gpt.concept_instruction.curriculum_m90 import (
    InstructionTask,
    M90CurriculumGenerator,
    StageBSuite,
    extract_prompt_features,
)
from adaptive_mini_gpt.concept_instruction.instruction_grammar import (
    ControlToken,
    InstructionGrammar,
    InstructionPrompt,
    Opcode,
    ParaphraseOpcode,
)

__all__ = [
    "Opcode",
    "ParaphraseOpcode",
    "ControlToken",
    "InstructionPrompt",
    "InstructionGrammar",
    "InstructionTask",
    "StageBSuite",
    "M90CurriculumGenerator",
    "extract_prompt_features",
    "InstructionConceptTrunk",
    "StageACapacityBenchmark",
    "M90InstructionBenchmark",
]
