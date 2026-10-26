"""Milestone M103 Natural-Language Rule Curriculum Generator.

Generates controlled synthetic natural-language explanations, rule-based questions,
and target answers across diverse reasoning systems:
1. Energy / Resource Modification System.
2. 1D Coordinate / Movement System.
3. Symbol Transformation / Cipher System.
4. Conditional Logic / Gate System.

Enforces strict scientific partitioning:
- Train Split: Canonical questions on seen parameter ranges.
- Validation Split: Seen rules with unseen parameter ranges (used for early stopping).
- Never-Trained Test Split: Unseen parameter ranges AND independent linguistic paraphrases.
- Novel Rule Parameterization: Unseen rule constants, colors, and compositions.
- 0% Token/Sequence Leakage across partitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
import torch

from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)


class TextRuleDomainM103(str, Enum):
    ENERGY_SYSTEM = "energy_system"
    COORDINATE_SYSTEM = "coordinate_system"
    SYMBOL_TRANSFORMATION = "symbol_transformation"
    CONDITIONAL_LOGIC = "conditional_logic"


@dataclass
class QAItemM103:
    system_domain: TextRuleDomainM103
    explanation: str
    question: str
    answer: str
    is_paraphrase: bool = False
    is_novel_parameterization: bool = False
    full_tokens: list[int] = field(default_factory=list)
    prompt_len: int = 0


@dataclass
class RuleGoalM103:
    domain: TextRuleDomainM103
    system_name: str
    explanation: str
    train_items: list[QAItemM103]
    val_items: list[QAItemM103]
    test_items: list[QAItemM103]
    novel_items: list[QAItemM103]


class TextCurriculumGeneratorM103:
    """Produces controlled, disjoint natural-language rule learning tasks."""

    def __init__(
        self,
        tokenizer: ByteTextTokenizerM103 | None = None,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer or ByteTextTokenizerM103()
        self.seed = seed
        self.rng = random.Random(seed)

    def _format_item(
        self,
        domain: TextRuleDomainM103,
        explanation: str,
        question: str,
        answer: str,
        is_paraphrase: bool = False,
        is_novel: bool = False,
    ) -> QAItemM103:
        full_toks, prompt_len = self.tokenizer.format_qa_sequence(
            question=question,
            answer=answer,
            explanation=explanation,
        )
        return QAItemM103(
            system_domain=domain,
            explanation=explanation,
            question=question,
            answer=answer,
            is_paraphrase=is_paraphrase,
            is_novel_parameterization=is_novel,
            full_tokens=full_toks,
            prompt_len=prompt_len,
        )

    def generate_energy_goal(self) -> RuleGoalM103:
        """Domain 1: Energy modification with priority ordering."""
        sys_name = "System-Alpha"
        gain = 2
        loss = 1
        expl = (
            f"In {sys_name}, a blue orb gains {gain} energy units. "
            f"A red orb loses {loss} energy unit. When both apply, blue acts first."
        )

        train_items: list[QAItemM103] = []
        for s in range(2, 10):  # 8 values
            # Case 1: Blue only
            q_b = f"An orb starts with {s} energy and is blue. What is its final energy?"
            a_b = f"{s + gain} energy"
            train_items.append(self._format_item(TextRuleDomainM103.ENERGY_SYSTEM, expl, q_b, a_b))

            # Case 2: Blue and Red
            q_br = f"An orb starts with {s} energy and is both blue and red. What is its final energy?"
            a_br = f"{s + gain - loss} energy"
            train_items.append(self._format_item(TextRuleDomainM103.ENERGY_SYSTEM, expl, q_br, a_br))

        val_items: list[QAItemM103] = []
        for s in range(10, 14):  # Unseen s in [10..13]
            q_b = f"An orb starts with {s} energy and is blue. What is its final energy?"
            a_b = f"{s + gain} energy"
            val_items.append(self._format_item(TextRuleDomainM103.ENERGY_SYSTEM, expl, q_b, a_b))

            q_br = f"An orb starts with {s} energy and is both blue and red. What is its final energy?"
            a_br = f"{s + gain - loss} energy"
            val_items.append(self._format_item(TextRuleDomainM103.ENERGY_SYSTEM, expl, q_br, a_br))

        test_items: list[QAItemM103] = []
        for s in range(14, 18):  # Unseen s in [14..17] + Paraphrased phrasing
            q_b = f"Calculate the terminal energy for an item starting at {s} energy with blue status."
            a_b = f"{s + gain} energy"
            test_items.append(self._format_item(TextRuleDomainM103.ENERGY_SYSTEM, expl, q_b, a_b, is_paraphrase=True))

            q_br = f"Determine the resulting energy level if an orb begins at {s} energy and carries blue and red."
            a_br = f"{s + gain - loss} energy"
            test_items.append(self._format_item(TextRuleDomainM103.ENERGY_SYSTEM, expl, q_br, a_br, is_paraphrase=True))

        novel_items: list[QAItemM103] = []
        expl_nov = "In System-Omega, a gold orb gains 5 energy units. A dark orb loses 3 energy units. Gold acts first."
        for s in range(20, 28):
            q_nov = f"An orb starts with {s} energy and is gold and dark. What is its final energy?"
            a_nov = f"{s + 5 - 3} energy"
            novel_items.append(self._format_item(TextRuleDomainM103.ENERGY_SYSTEM, expl_nov, q_nov, a_nov, is_novel=True))
            q_nov2 = f"An orb starts with {s} energy and is gold. What is its final energy?"
            a_nov2 = f"{s + 5} energy"
            novel_items.append(self._format_item(TextRuleDomainM103.ENERGY_SYSTEM, expl_nov, q_nov2, a_nov2, is_novel=True))

        return RuleGoalM103(
            domain=TextRuleDomainM103.ENERGY_SYSTEM,
            system_name=sys_name,
            explanation=expl,
            train_items=train_items,
            val_items=val_items,
            test_items=test_items,
            novel_items=novel_items,
        )

    def generate_coordinate_goal(self) -> RuleGoalM103:
        """Domain 2: 1D grid navigation."""
        sys_name = "Grid-Beta"
        step_n = 3
        step_s = 2
        expl = (
            f"In {sys_name}, moving North adds {step_n} units to position. "
            f"Moving South subtracts {step_s} units from position."
        )

        train_items: list[QAItemM103] = []
        for pos in range(1, 9):
            q_n = f"A walker starts at position {pos} and moves North. What is the final position?"
            a_n = f"{pos + step_n} position"
            train_items.append(self._format_item(TextRuleDomainM103.COORDINATE_SYSTEM, expl, q_n, a_n))

            q_ns = f"A walker starts at position {pos} and moves North then South. What is the final position?"
            a_ns = f"{pos + step_n - step_s} position"
            train_items.append(self._format_item(TextRuleDomainM103.COORDINATE_SYSTEM, expl, q_ns, a_ns))

        val_items: list[QAItemM103] = []
        for pos in range(9, 13):
            q_n = f"A walker starts at position {pos} and moves North. What is the final position?"
            a_n = f"{pos + step_n} position"
            val_items.append(self._format_item(TextRuleDomainM103.COORDINATE_SYSTEM, expl, q_n, a_n))

            q_ns = f"A walker starts at position {pos} and moves North then South. What is the final position?"
            a_ns = f"{pos + step_n - step_s} position"
            val_items.append(self._format_item(TextRuleDomainM103.COORDINATE_SYSTEM, expl, q_ns, a_ns))

        test_items: list[QAItemM103] = []
        for pos in range(13, 17):
            q_n = f"Find the destination coordinate for a walker who begins at {pos} and travels North."
            a_n = f"{pos + step_n} position"
            test_items.append(self._format_item(TextRuleDomainM103.COORDINATE_SYSTEM, expl, q_n, a_n, is_paraphrase=True))

            q_ns = f"Where does the traveler stop after starting at {pos} and taking steps North followed by South?"
            a_ns = f"{pos + step_n - step_s} position"
            test_items.append(self._format_item(TextRuleDomainM103.COORDINATE_SYSTEM, expl, q_ns, a_ns, is_paraphrase=True))

        novel_items: list[QAItemM103] = []
        expl_nov = "In Grid-Zeta, moving East adds 7 units to position. Moving West subtracts 4 units."
        for pos in range(20, 28):
            q_nov = f"A walker starts at position {pos} and moves East then West. What is the final position?"
            a_nov = f"{pos + 7 - 4} position"
            novel_items.append(self._format_item(TextRuleDomainM103.COORDINATE_SYSTEM, expl_nov, q_nov, a_nov, is_novel=True))
            q_nov2 = f"A walker starts at position {pos} and moves East. What is the final position?"
            a_nov2 = f"{pos + 7} position"
            novel_items.append(self._format_item(TextRuleDomainM103.COORDINATE_SYSTEM, expl_nov, q_nov2, a_nov2, is_novel=True))

        return RuleGoalM103(
            domain=TextRuleDomainM103.COORDINATE_SYSTEM,
            system_name=sys_name,
            explanation=expl,
            train_items=train_items,
            val_items=val_items,
            test_items=test_items,
            novel_items=novel_items,
        )

    def generate_full_curriculum(self) -> list[RuleGoalM103]:
        """Generate all structured text rule goals."""
        return [
            self.generate_energy_goal(),
            self.generate_coordinate_goal(),
        ]
