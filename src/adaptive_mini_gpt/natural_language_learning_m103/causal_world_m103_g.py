"""Milestone M103-G Non-Numerical Causal World Simulator.

Simulates an Alchemy/Elemental world with objects, discrete attributes,
states, and causal actions.

Designed to evaluate non-numerical conceptual learning independently from
arithmetic calculation:
- Objects: Torch, Crucible, Glowstone, Rune, Cauldron, Prism (Familiar);
  Amulet, Mirror (Novel Held-Out Entities).
- Attributes: flammable, metal, heat_tolerant, crystalline, radiant, magical, container.
- Actions: Apply_Fire, Apply_Frost, Apply_Spark, Apply_Solvent, Apply_Essence.
- Outcomes: Ignited, Tempered, Frozen, Charged, Resonating, Purified, Awakened.
- Sequential Multi-Step: Apply_Fire then Apply_Frost -> Extinguished / Brittle.
- Counterfactual Shadow Realm: Apply_Spark -> Dimmed, Apply_Fire -> Ash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any


@dataclass
class CausalObjectM103_G:
    name: str
    tags: set[str]
    description: str
    is_novel: bool = False


@dataclass
class CausalItemM103_G:
    scenario_id: str
    category: str  # "train", "val_held_out_pair", "novel_entity", "multistep", "counterfactual"
    object_name: str
    action_name: str
    second_action: str | None
    initial_state: str
    expected_state: str
    rule_explanation: str
    question_text: str
    canonical_answer: str
    is_counterfactual: bool = False


class CausalWorldM103_G:
    """Deterministic causal world simulator and dataset generator for M103-G."""

    OBJECTS: dict[str, CausalObjectM103_G] = {
        "Torch": CausalObjectM103_G(
            name="Torch",
            tags={"flammable", "wood", "tool"},
            description="A wooden torch wrapped in porous hemp.",
            is_novel=False,
        ),
        "Crucible": CausalObjectM103_G(
            name="Crucible",
            tags={"metal", "heat_tolerant", "container"},
            description="A dense iron crucible forged for high heat.",
            is_novel=False,
        ),
        "Glowstone": CausalObjectM103_G(
            name="Glowstone",
            tags={"crystalline", "radiant", "mineral"},
            description="A glowing mineral crystal that emits soft luminescence.",
            is_novel=False,
        ),
        "Rune": CausalObjectM103_G(
            name="Rune",
            tags={"inscribed", "magical", "stone"},
            description="An ancient runic tablet carved with arcane sigils.",
            is_novel=False,
        ),
        "Cauldron": CausalObjectM103_G(
            name="Cauldron",
            tags={"metal", "container"},
            description="A large iron cauldron used for blending solutions.",
            is_novel=False,
        ),
        "Prism": CausalObjectM103_G(
            name="Prism",
            tags={"crystalline", "optical"},
            description="A polished crystalline prism refracting ambient light.",
            is_novel=False,
        ),
        # Genuinely UNSEEN Novel Entities
        "Amulet": CausalObjectM103_G(
            name="Amulet",
            tags={"metal", "magical"},
            description="A silver amulet inscribed with protective enchantments.",
            is_novel=True,
        ),
        "Mirror": CausalObjectM103_G(
            name="Mirror",
            tags={"crystalline", "optical"},
            description="A reflective obsidian mirror framed in crystal.",
            is_novel=True,
        ),
    }

    ACTIONS: list[str] = [
        "Apply_Fire",
        "Apply_Frost",
        "Apply_Spark",
        "Apply_Solvent",
        "Apply_Essence",
    ]

    STATES: list[str] = [
        "Dormant",
        "Ignited",
        "Tempered",
        "Frozen",
        "Charged",
        "Resonating",
        "Purified",
        "Awakened",
        "Extinguished",
        "Brittle",
        "Discharged",
        "Dimmed",
        "Ash",
    ]

    STANDARD_RULE_TEXT: str = (
        "Fire ignites flammables, tempers metal. Frost freezes all. Sparks charge relics, resonate crystals. Solvent purifies containers. Essence awakens."
    )

    SHADOW_RULE_TEXT: str = (
        "In Shadow Realm: Fire burns flammables to ash. Sparks dim relics. Frost freezes all."
    )

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def transition_state(
        self,
        obj_name: str,
        action: str,
        current_state: str = "Dormant",
        realm: str = "Alchemy",
    ) -> str:
        """Deterministic causal transition function (ground truth)."""
        obj = self.OBJECTS[obj_name]
        tags = obj.tags

        # Counterfactual Shadow Realm
        if realm == "Shadow":
            if action == "Apply_Fire":
                return "Ash" if "flammable" in tags else "Tempered"
            if action == "Apply_Spark":
                return "Dimmed" if ("magical" in tags or "radiant" in tags) else "Dormant"
            if action == "Apply_Frost":
                return "Frozen"
            return current_state

        # Multi-step sequential transitions
        if current_state == "Ignited" and action == "Apply_Frost":
            return "Extinguished"
        if current_state == "Tempered" and action == "Apply_Frost":
            return "Brittle"
        if current_state == "Charged" and action == "Apply_Solvent":
            return "Discharged"

        # Standard One-Step Transitions
        if action == "Apply_Fire":
            if "flammable" in tags:
                return "Ignited"
            if "metal" in tags:
                return "Tempered"
            return "Dormant"
        if action == "Apply_Frost":
            return "Frozen"
        if action == "Apply_Spark":
            if "magical" in tags or "radiant" in tags:
                return "Charged"
            if "crystalline" in tags:
                return "Resonating"
            return "Dormant"
        if action == "Apply_Solvent":
            if "metal" in tags or "container" in tags:
                return "Purified"
            return "Dormant"
        if action == "Apply_Essence":
            return "Awakened"

        return current_state

    def generate_datasets(self) -> dict[str, list[CausalItemM103_G]]:
        """Constructs strictly partitioned datasets for G2 causal evaluation.

        Splits:
        - train: 18 standard (familiar_object, action) pairs.
        - val_held_out_pair: 6 combinations where both object and action are seen,
          but the specific pair was strictly held out.
        - novel_entity: 6 pairs testing unseen objects (Amulet, Mirror) with their descriptions.
        - multistep: 6 two-step causal chains.
        - counterfactual: 6 items in Shadow Realm with altered causal laws.
        """
        # Held-out pairs: strictly disjoint from train
        held_out_pairs: set[tuple[str, str]] = {
            ("Crucible", "Apply_Frost"),   # Metal container + frost
            ("Torch", "Apply_Solvent"),     # Wood tool + solvent
            ("Rune", "Apply_Frost"),        # Magical stone + frost
            ("Glowstone", "Apply_Essence"), # Radiant crystal + essence
            ("Cauldron", "Apply_Fire"),     # Metal container + fire
            ("Prism", "Apply_Spark"),       # Crystal + spark
        }

        familiar_objects = ["Torch", "Crucible", "Glowstone", "Rune", "Cauldron", "Prism"]
        actions = ["Apply_Fire", "Apply_Frost", "Apply_Spark", "Apply_Solvent", "Apply_Essence"]

        train_items: list[CausalItemM103_G] = []
        val_pairs_items: list[CausalItemM103_G] = []

        item_idx = 0
        for obj_name in familiar_objects:
            for act in actions:
                pair = (obj_name, act)
                target = self.transition_state(obj_name, act, "Dormant", "Alchemy")
                obj = self.OBJECTS[obj_name]

                item = CausalItemM103_G(
                    scenario_id=f"g2_scenario_{item_idx}",
                    category="train" if pair not in held_out_pairs else "val_held_out_pair",
                    object_name=obj_name,
                    action_name=act,
                    second_action=None,
                    initial_state="Dormant",
                    expected_state=target,
                    rule_explanation=self.STANDARD_RULE_TEXT,
                    question_text=f"{obj.description} We apply {act}. What is the resulting state?",
                    canonical_answer=target,
                    is_counterfactual=False,
                )
                item_idx += 1

                if pair in held_out_pairs:
                    val_pairs_items.append(item)
                else:
                    train_items.append(item)

        # Novel Entity Items (Amulet, Mirror)
        novel_items: list[CausalItemM103_G] = []
        for obj_name in ["Amulet", "Mirror"]:
            for act in ["Apply_Fire", "Apply_Frost", "Apply_Spark"]:
                target = self.transition_state(obj_name, act, "Dormant", "Alchemy")
                obj = self.OBJECTS[obj_name]
                novel_items.append(
                    CausalItemM103_G(
                        scenario_id=f"g2_scenario_{item_idx}",
                        category="novel_entity",
                        object_name=obj_name,
                        action_name=act,
                        second_action=None,
                        initial_state="Dormant",
                        expected_state=target,
                        rule_explanation=self.STANDARD_RULE_TEXT,
                        question_text=f"{obj.description} We apply {act}. What is the resulting state?",
                        canonical_answer=target,
                        is_counterfactual=False,
                    )
                )
                item_idx += 1

        # Multi-step Items
        multistep_chains = [
            ("Torch", "Apply_Fire", "Apply_Frost"),      # Ignited -> Extinguished
            ("Crucible", "Apply_Fire", "Apply_Frost"),   # Tempered -> Brittle
            ("Rune", "Apply_Spark", "Apply_Solvent"),    # Charged -> Discharged
            ("Cauldron", "Apply_Solvent", "Apply_Frost"),# Purified -> Frozen
            ("Glowstone", "Apply_Spark", "Apply_Essence"),# Resonating -> Awakened
            ("Prism", "Apply_Frost", "Apply_Essence"),   # Frozen -> Awakened
        ]
        multistep_items: list[CausalItemM103_G] = []
        for obj_name, act1, act2 in multistep_chains:
            intermediate = self.transition_state(obj_name, act1, "Dormant", "Alchemy")
            final_st = self.transition_state(obj_name, act2, intermediate, "Alchemy")
            obj = self.OBJECTS[obj_name]
            multistep_items.append(
                CausalItemM103_G(
                    scenario_id=f"g2_scenario_{item_idx}",
                    category="multistep",
                    object_name=obj_name,
                    action_name=act1,
                    second_action=act2,
                    initial_state="Dormant",
                    expected_state=final_st,
                    rule_explanation=self.STANDARD_RULE_TEXT,
                    question_text=f"{obj.description} First we apply {act1}, then we apply {act2}. What is the final state?",
                    canonical_answer=final_st,
                    is_counterfactual=False,
                )
            )
            item_idx += 1

        # Counterfactual Shadow Realm Items
        cf_pairs = [
            ("Torch", "Apply_Fire"),      # Ash
            ("Crucible", "Apply_Fire"),   # Tempered
            ("Rune", "Apply_Spark"),      # Dimmed
            ("Glowstone", "Apply_Spark"), # Dimmed
            ("Torch", "Apply_Frost"),     # Frozen
            ("Rune", "Apply_Frost"),      # Frozen
        ]
        cf_items: list[CausalItemM103_G] = []
        for obj_name, act in cf_pairs:
            target = self.transition_state(obj_name, act, "Dormant", "Shadow")
            obj = self.OBJECTS[obj_name]
            cf_items.append(
                CausalItemM103_G(
                    scenario_id=f"g2_scenario_{item_idx}",
                    category="counterfactual",
                    object_name=obj_name,
                    action_name=act,
                    second_action=None,
                    initial_state="Dormant",
                    expected_state=target,
                    rule_explanation=self.SHADOW_RULE_TEXT,
                    question_text=f"In the Shadow Realm: {obj.description} We apply {act}. What is the resulting state?",
                    canonical_answer=target,
                    is_counterfactual=True,
                )
            )
            item_idx += 1

        return {
            "train": train_items,
            "val_held_out_pair": val_pairs_items,
            "novel_entity": novel_items,
            "multistep": multistep_items,
            "counterfactual": cf_items,
        }
