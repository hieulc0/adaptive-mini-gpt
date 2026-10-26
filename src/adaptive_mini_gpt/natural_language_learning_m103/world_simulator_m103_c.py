"""Milestone M103-C Ground-Truth World Simulator.

Provides an independent, deterministic simulator of a synthetic physical/symbolic world:
- Entities: Orb, Crystal, Beacon, Relic, Sphere.
- State variables: Energy (int >= 0), Coordinates (int), Status (str).
- Deterministic causal transition rules.
- Counterfactual state generation (what if a constant or rule priority were different?).
- Diverse linguistic frame generation (interrogative, declarative, imperative, conditional).
- Strictly partitioned datasets:
  * train_items (diverse templates, entities, base values s in [2..14])
  * val_canonical (held-out combinations with seen templates and values)
  * unseen_values_items (extrapolation/interpolation values s in [15..28])
  * paraphrase_items (novel syntactic phrasing unseen in training)
  * counterfactual_items (alternative rule constants)
  * novel_combination_items (multi-rule compositions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Any

from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)


class ConceptFamilyM103_C(str, Enum):
    ENERGY_SYSTEM = "energy_system"
    COORDINATE_SYSTEM = "coordinate_system"
    TRANSFORMATION_SYSTEM = "transformation_system"


@dataclass
class WorldEntityM103_C:
    name: str
    kind: str  # "orb", "crystal", "beacon", "relic", "sphere"
    energy: int = 0
    x_pos: int = 0
    active_modifiers: list[str] = field(default_factory=list)


@dataclass
class SimulatorItemM103_C:
    family: ConceptFamilyM103_C
    explanation: str
    question: str
    canonical_answer: str
    unrestricted_answer: str
    expected_numeric_value: int
    entity_kind: str
    split_category: str  # "train", "val_canonical", "unseen_values", "paraphrase", "counterfactual", "novel_combination"
    full_tokens: list[int] = field(default_factory=list)
    prompt_len: int = 0


@dataclass
class ConceptDatasetM103_C:
    family: ConceptFamilyM103_C
    rule_description: str
    train_items: list[SimulatorItemM103_C]
    val_canonical_items: list[SimulatorItemM103_C]
    unseen_values_items: list[SimulatorItemM103_C]
    paraphrase_items: list[SimulatorItemM103_C]
    counterfactual_items: list[SimulatorItemM103_C]
    novel_combination_items: list[SimulatorItemM103_C]


class WorldSimulatorM103_C:
    """Independent, deterministic simulator of synthetic world dynamics."""

    ENTITIES = ["orb", "crystal", "beacon", "relic", "sphere"]

    def __init__(self, seed: int = 42, tokenizer: ByteTextTokenizerM103 | None = None) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.tokenizer = tokenizer or ByteTextTokenizerM103()

    def simulate_energy_step(
        self,
        initial_energy: int,
        is_blue: bool = True,
        is_red: bool = False,
        blue_gain: int = 2,
        red_loss: int = 1,
    ) -> int:
        """Deterministic simulation of energy transitions."""
        val = initial_energy
        if is_blue:
            val += blue_gain
        if is_red:
            val = max(0, val - red_loss)
        return val

    def simulate_coordinate_step(
        self,
        initial_pos: int,
        moves: list[str],
        step_n: int = 3,
        step_s: int = 2,
    ) -> int:
        """Deterministic simulation of coordinate navigation."""
        pos = initial_pos
        for m in moves:
            if m.lower() == "north":
                pos += step_n
            elif m.lower() == "south":
                pos -= step_s
        return pos

    def format_simulator_item(
        self,
        family: ConceptFamilyM103_C,
        explanation: str,
        question: str,
        canonical_answer: str,
        unrestricted_answer: str,
        expected_numeric_value: int,
        entity_kind: str,
        split_category: str,
    ) -> SimulatorItemM103_C:
        """Tokenize and format QA sequence with prompt length tracking."""
        full_toks, prompt_len = self.tokenizer.format_qa_sequence(
            question=question,
            answer=canonical_answer,
            explanation=explanation,
        )
        return SimulatorItemM103_C(
            family=family,
            explanation=explanation,
            question=question,
            canonical_answer=canonical_answer,
            unrestricted_answer=unrestricted_answer,
            expected_numeric_value=expected_numeric_value,
            entity_kind=entity_kind,
            split_category=split_category,
            full_tokens=full_toks,
            prompt_len=prompt_len,
        )

    def generate_energy_dataset(
        self,
        train_val_values: list[int] | None = None,
        unseen_values: list[int] | None = None,
    ) -> ConceptDatasetM103_C:
        """Generates comprehensive dataset for Concept Family 1: Energy Modification."""
        train_val_s = train_val_values or list(range(2, 15))  # 2..14 (13 values)
        unseen_s = unseen_values or list(range(15, 29))       # 15..28 (14 unseen values)

        expl = (
            "In System-Alpha, a blue entity gains 2 energy units. "
            "A red entity loses 1 energy unit. When both apply, blue acts first."
        )

        train_items: list[SimulatorItemM103_C] = []
        val_canonical_items: list[SimulatorItemM103_C] = []

        # Varied training and validation templates
        templates_train = [
            ("An {ent} starts with {s} energy and is blue. What is its final energy?", True, False),
            ("A {ent} begins at {s} energy. Upon blue activation, its energy becomes what?", True, False),
            ("Calculate the energy of a blue {ent} starting with {s} energy.", True, False),
            ("An {ent} starts with {s} energy and is both blue and red. What is its final energy?", True, True),
            ("Given a {ent} with {s} energy having blue and red status, what is the final energy?", True, True),
        ]

        # Populate train items with varied entities and templates
        for idx, s in enumerate(train_val_s):
            for t_idx, (q_tpl, is_b, is_r) in enumerate(templates_train):
                ent = self.ENTITIES[(idx + t_idx) % len(self.ENTITIES)]
                ans_num = self.simulate_energy_step(s, is_blue=is_b, is_red=is_r)
                q_text = q_tpl.format(ent=ent, s=s)
                c_ans = f"{ans_num} energy"
                u_ans = f"The final energy of the {ent} is {ans_num} energy."

                item = self.format_simulator_item(
                    family=ConceptFamilyM103_C.ENERGY_SYSTEM,
                    explanation=expl,
                    question=q_text,
                    canonical_answer=c_ans,
                    unrestricted_answer=u_ans,
                    expected_numeric_value=ans_num,
                    entity_kind=ent,
                    split_category="train",
                )
                if (idx + t_idx) % 5 == 0:
                    item.split_category = "val_canonical"
                    val_canonical_items.append(item)
                else:
                    train_items.append(item)

        # Unseen Values: values in unseen_s with standard phrasing
        unseen_values_items: list[SimulatorItemM103_C] = []
        for s in unseen_s:
            for is_b, is_r in [(True, False), (True, True)]:
                ent = self.ENTITIES[s % len(self.ENTITIES)]
                ans_num = self.simulate_energy_step(s, is_blue=is_b, is_red=is_r)
                status_str = "blue and red" if (is_b and is_r) else "blue"
                q_text = f"An {ent} starts with {s} energy and is {status_str}. What is its final energy?"
                c_ans = f"{ans_num} energy"
                u_ans = f"The final energy of the {ent} is {ans_num} energy."

                unseen_values_items.append(
                    self.format_simulator_item(
                        family=ConceptFamilyM103_C.ENERGY_SYSTEM,
                        explanation=expl,
                        question=q_text,
                        canonical_answer=c_ans,
                        unrestricted_answer=u_ans,
                        expected_numeric_value=ans_num,
                        entity_kind=ent,
                        split_category="unseen_values",
                    )
                )

        # Paraphrase Items: entirely novel linguistic frames, seen value range [2..14]
        paraphrase_templates = [
            ("Determine the resulting energy level if a {ent} begins at {s} energy and carries blue modifier.", True, False),
            ("With an initial energy of {s}, the blue {ent} reaches what energy level?", True, False),
            ("If a {ent} has {s} energy and receives blue power, determine the resulting energy.", True, False),
            ("Find the terminal energy of an {ent} after applying blue then red starting from {s} energy.", True, True),
            ("Operating on a {ent} with {s} initial energy using both blue and red yields what energy?", True, True),
        ]
        paraphrase_items: list[SimulatorItemM103_C] = []
        for s in [3, 5, 7, 9, 11, 13]:
            for p_tpl, is_b, is_r in paraphrase_templates:
                ent = self.ENTITIES[(s + 2) % len(self.ENTITIES)]
                ans_num = self.simulate_energy_step(s, is_blue=is_b, is_red=is_r)
                q_text = p_tpl.format(ent=ent, s=s)
                c_ans = f"{ans_num} energy"
                u_ans = f"The resulting energy level of the {ent} is {ans_num} energy."

                paraphrase_items.append(
                    self.format_simulator_item(
                        family=ConceptFamilyM103_C.ENERGY_SYSTEM,
                        explanation=expl,
                        question=q_text,
                        canonical_answer=c_ans,
                        unrestricted_answer=u_ans,
                        expected_numeric_value=ans_num,
                        entity_kind=ent,
                        split_category="paraphrase",
                    )
                )

        # Counterfactual Items: modified rule parameters (e.g. blue adds 4, red loses 2)
        expl_cf = (
            "Suppose in a hypothetical variation of System-Alpha, blue adds 4 energy units instead of 2, "
            "and red loses 2 energy units. Blue still acts first."
        )
        counterfactual_items: list[SimulatorItemM103_C] = []
        for s in [4, 8, 12, 16, 20]:
            for is_b, is_r in [(True, False), (True, True)]:
                ent = self.ENTITIES[s % len(self.ENTITIES)]
                ans_num = self.simulate_energy_step(s, is_blue=is_b, is_red=is_r, blue_gain=4, red_loss=2)
                status_str = "blue and red" if (is_b and is_r) else "blue"
                q_text = f"Under the modified rule, an {ent} starts with {s} energy and is {status_str}. What is its final energy?"
                c_ans = f"{ans_num} energy"
                u_ans = f"Under the modified rule the final energy is {ans_num} energy."

                counterfactual_items.append(
                    self.format_simulator_item(
                        family=ConceptFamilyM103_C.ENERGY_SYSTEM,
                        explanation=expl_cf,
                        question=q_text,
                        canonical_answer=c_ans,
                        unrestricted_answer=u_ans,
                        expected_numeric_value=ans_num,
                        entity_kind=ent,
                        split_category="counterfactual",
                    )
                )

        # Novel Combination Items: multi-step compound actions (e.g. two blue pulses or blue then amber)
        novel_combination_items: list[SimulatorItemM103_C] = []
        for s in [5, 10, 15, 20]:
            ent = self.ENTITIES[s % len(self.ENTITIES)]
            # Two blue pulses: +2 +2 = +4
            ans_num = s + 4
            q_text = f"An {ent} starts with {s} energy and receives two consecutive blue activations. What is its final energy?"
            c_ans = f"{ans_num} energy"
            u_ans = f"After two activations the {ent} has {ans_num} energy."

            novel_combination_items.append(
                self.format_simulator_item(
                    family=ConceptFamilyM103_C.ENERGY_SYSTEM,
                    explanation=expl,
                    question=q_text,
                    canonical_answer=c_ans,
                    unrestricted_answer=u_ans,
                    expected_numeric_value=ans_num,
                    entity_kind=ent,
                    split_category="novel_combination",
                )
            )

        return ConceptDatasetM103_C(
            family=ConceptFamilyM103_C.ENERGY_SYSTEM,
            rule_description=expl,
            train_items=train_items,
            val_canonical_items=val_canonical_items,
            unseen_values_items=unseen_values_items,
            paraphrase_items=paraphrase_items,
            counterfactual_items=counterfactual_items,
            novel_combination_items=novel_combination_items,
        )

    def generate_coordinate_dataset(self) -> ConceptDatasetM103_C:
        """Generates dataset for Concept Family 2: Coordinate Navigation."""
        expl = (
            "In Grid-Beta, moving North adds 3 units to position. "
            "Moving South subtracts 2 units from position."
        )
        train_items: list[SimulatorItemM103_C] = []
        val_canonical_items: list[SimulatorItemM103_C] = []
        unseen_values_items: list[SimulatorItemM103_C] = []
        paraphrase_items: list[SimulatorItemM103_C] = []
        counterfactual_items: list[SimulatorItemM103_C] = []
        novel_combination_items: list[SimulatorItemM103_C] = []

        # Train: pos in [1..10]
        for pos in range(1, 11):
            for moves, m_name in [(["North"], "North"), (["North", "South"], "North then South")]:
                ans_pos = self.simulate_coordinate_step(pos, moves)
                q_text = f"A traveler starts at coordinate {pos} and moves {m_name}. What is the final coordinate?"
                c_ans = f"{ans_pos} coordinate"
                u_ans = f"The final coordinate is {ans_pos} coordinate."

                item = self.format_simulator_item(
                    family=ConceptFamilyM103_C.COORDINATE_SYSTEM,
                    explanation=expl,
                    question=q_text,
                    canonical_answer=c_ans,
                    unrestricted_answer=u_ans,
                    expected_numeric_value=ans_pos,
                    entity_kind="traveler",
                    split_category="train" if pos != 5 else "val_canonical",
                )
                if pos == 5:
                    val_canonical_items.append(item)
                else:
                    train_items.append(item)

        # Unseen values: pos in [12..20]
        for pos in range(12, 21):
            for moves, m_name in [(["North"], "North"), (["North", "South"], "North then South")]:
                ans_pos = self.simulate_coordinate_step(pos, moves)
                q_text = f"A traveler starts at coordinate {pos} and moves {m_name}. What is the final coordinate?"
                c_ans = f"{ans_pos} coordinate"
                u_ans = f"The final coordinate is {ans_pos} coordinate."

                unseen_values_items.append(
                    self.format_simulator_item(
                        family=ConceptFamilyM103_C.COORDINATE_SYSTEM,
                        explanation=expl,
                        question=q_text,
                        canonical_answer=c_ans,
                        unrestricted_answer=u_ans,
                        expected_numeric_value=ans_pos,
                        entity_kind="traveler",
                        split_category="unseen_values",
                    )
                )

        # Paraphrase
        for pos in [2, 4, 6, 8]:
            ans_pos = self.simulate_coordinate_step(pos, ["North"])
            q_text = f"Determine the destination coordinate if an agent departs location {pos} heading North."
            c_ans = f"{ans_pos} coordinate"
            u_ans = f"The destination is coordinate {ans_pos} coordinate."
            paraphrase_items.append(
                self.format_simulator_item(
                    family=ConceptFamilyM103_C.COORDINATE_SYSTEM,
                    explanation=expl,
                    question=q_text,
                    canonical_answer=c_ans,
                    unrestricted_answer=u_ans,
                    expected_numeric_value=ans_pos,
                    entity_kind="agent",
                    split_category="paraphrase",
                )
            )

        return ConceptDatasetM103_C(
            family=ConceptFamilyM103_C.COORDINATE_SYSTEM,
            rule_description=expl,
            train_items=train_items,
            val_canonical_items=val_canonical_items,
            unseen_values_items=unseen_values_items,
            paraphrase_items=paraphrase_items,
            counterfactual_items=counterfactual_items,
            novel_combination_items=novel_combination_items,
        )
