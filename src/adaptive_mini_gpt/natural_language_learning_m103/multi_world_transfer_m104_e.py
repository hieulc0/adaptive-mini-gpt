"""Milestone M104-E Multi-World Forward Transfer Benchmark with Right-Censored Survival Accounting.

Features:
1. Five Procedural Transfer World Families with Compositional Multi-Step Dynamics:
   - World T1: Elemental Realm (Pyromancy, Hydromancy, Cryomancy, Electromancy)
   - World T2: Celestial Reliquary (Solar, Lunar, Astral, Umbral)
   - World T3: Biomechanical Fabric (Chitin, Bio-Polymer, Fungal, Neural)
   - World T4: Chrono-Crystalline (Temporal, Resonant, Stasis, Harmonic)
   - World T5: Subterranean Geode (Ferromagnetic, Geothermal, Basaltic, Tectonic)
2. Granular Step-Zero and Early Measurement Evaluation:
   - Evaluates held-out accuracy at steps [0, 1, 2, 5, 10, 20, 40, 80].
   - Captures pre-adaptation (step 0) zero-shot transfer and early adaptation dynamics.
3. Rigorous Survival Analysis:
   - Tracks time-to-target accuracy with explicit `is_censored` status.
   - Prevents saturation artifacts where both arms appear tied at later step milestones.
4. Independent 4-Arm Comparison:
   - Experienced Persistent Student (pre-trained on World A foundational experience)
   - Fresh Student (initialized from scratch)
   - Language-Pretrained Control (pre-trained on language corpora without causal transitions)
   - Frozen Backbone Control (World A representation frozen, only adapter adapted)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.natural_language_learning_m103.causal_world_m104_b import (
    CausalItemM104_B,
    CausalObjectM104_B,
)


@dataclass
class WorldFamilySpecM104_E:
    world_id: str
    realm_name: str
    materials: list[tuple[str, list[str]]]
    forms: list[tuple[str, str, list[str]]]
    actions: list[str]
    states: list[str]
    rules_text: str


WORLD_FAMILIES_M104_E: list[WorldFamilySpecM104_E] = [
    WorldFamilySpecM104_E(
        world_id="world_t1_elemental",
        realm_name="Elemental Realm",
        materials=[
            ("Obsidian_Core", ["heat_tolerant", "magical", "stone"]),
            ("Glacial_Ice", ["crystalline", "organic"]),
            ("Copper_Coil", ["conductive", "metal"]),
            ("Sulfur_Ore", ["flammable", "mineral"]),
        ],
        forms=[
            ("Orb", "spherical orb", ["vessel"]),
            ("Rod", "cylindrical rod", ["conduit"]),
            ("Plate", "flat inscribed plate", ["inscribed"]),
        ],
        actions=["Kindle", "Submerge", "Frostbite", "Electrify"],
        states=["Vaporized", "Frozen_Solid", "Supercharged", "Extinguished", "Dormant", "Molten"],
        rules_text=(
            "Elemental Laws: Kindle ignites flammable minerals into Molten. "
            "Frostbite freezes crystalline items into Frozen_Solid. "
            "Electrify supercharges conductive metals into Supercharged. "
            "Submerge extinguishes molten items and dissolves organic matter."
        ),
    ),
    WorldFamilySpecM104_E(
        world_id="world_t2_celestial",
        realm_name="Celestial Reliquary",
        materials=[
            ("Sunstone", ["radiant", "mineral", "magical"]),
            ("Moon_Pearl", ["optical", "organic", "crystalline"]),
            ("Star_Metal", ["conductive", "silver", "heat_tolerant"]),
            ("Eclipse_Glass", ["glass", "shadow_infused"]),
        ],
        forms=[
            ("Prism", "faceted refractive prism", ["optical"]),
            ("Amulet", "ornate celestial amulet", ["relic"]),
            ("Mirror", "polished circular mirror", ["tool"]),
        ],
        actions=["Channel_Sunlight", "Invoke_Moonlight", "Astral_Pulse", "Shadow_Veil"],
        states=["Luminous", "Phase_Shifted", "Celestial_Harmonized", "Shrouded", "Dormant"],
        rules_text=(
            "Celestial Laws: Channel_Sunlight turns radiant sunstones Luminous. "
            "Shadow_Veil shrouds eclipse glass into Shrouded. "
            "Astral_Pulse harmonizes conductive star metal into Celestial_Harmonized. "
            "Invoke_Moonlight phase shifts optical crystals."
        ),
    ),
    WorldFamilySpecM104_E(
        world_id="world_t3_biomechanical",
        realm_name="Biomechanical Fabric",
        materials=[
            ("Chitin_Plate", ["organic", "heat_tolerant"]),
            ("Bio_Polymer", ["conductive", "flammable"]),
            ("Fungal_Network", ["organic", "alchemical"]),
            ("Neural_Fiber", ["conductive", "inscribed", "magical"]),
        ],
        forms=[
            ("Carapace", "segmented protective carapace", ["container"]),
            ("Sinew", "flexible contracting sinew", ["conduit"]),
            ("Node", "pulsing bio-node", ["vessel"]),
        ],
        actions=["Graft", "Inoculate", "Synthesize", "Spore_Burst"],
        states=["Metabolized", "Bio_Fused", "Spore_Infected", "Desiccated", "Dormant"],
        rules_text=(
            "Biomechanical Laws: Graft fuses heat_tolerant chitin into Bio_Fused. "
            "Spore_Burst infects organic fungal networks into Spore_Infected. "
            "Synthesize metabolizes conductive biopolymers into Metabolized. "
            "Inoculate desiccates flammable tissue."
        ),
    ),
    WorldFamilySpecM104_E(
        world_id="world_t4_chronocrystal",
        realm_name="Chrono-Crystalline",
        materials=[
            ("Chrono_Quartz", ["crystalline", "magical", "optical"]),
            ("Tuning_Bronze", ["metal", "conductive"]),
            ("Phase_Crystal", ["crystalline", "radiant"]),
            ("Resonance_Bell", ["metal", "tool"]),
        ],
        forms=[
            ("Hourglass", "sand-filled glass timer", ["vessel"]),
            ("Pendulum", "weighted swinging pendulum", ["conduit"]),
            ("Diapason", "dual-pronged resonator", ["tool"]),
        ],
        actions=["Accelerate_Time", "Harmonic_Resonance", "Stasis_Lock", "Echo_Pulse"],
        states=["Time_Dilated", "Harmonized", "Stasis_Frozen", "Shattered_Echo", "Dormant"],
        rules_text=(
            "Chrono Laws: Accelerate_Time dilates crystalline chrono quartz into Time_Dilated. "
            "Harmonic_Resonance harmonizes conductive tuning bronze into Harmonized. "
            "Stasis_Lock freezes radiant phase crystals into Stasis_Frozen. "
            "Echo_Pulse shatters ungrounded tools."
        ),
    ),
    WorldFamilySpecM104_E(
        world_id="world_t5_subterranean",
        realm_name="Subterranean Geode",
        materials=[
            ("Lodestone", ["metal", "conductive", "mineral"]),
            ("Magma_Slag", ["stone", "heat_tolerant", "flammable"]),
            ("Deep_Granite", ["stone", "mineral"]),
            ("Quartz_Vein", ["crystalline", "optical"]),
        ],
        forms=[
            ("Geode", "hollow hollow mineral nodule", ["container"]),
            ("Spire", "tapered basaltic spire", ["conduit"]),
            ("Anvil", "heavy forged block", ["tool"]),
        ],
        actions=["Magnetic_Pulse", "Magma_Infuse", "Pressure_Crush", "Tectonic_Shift"],
        states=["Magnetized", "Molten", "Fissured", "Crystallized", "Dormant"],
        rules_text=(
            "Subterranean Laws: Magnetic_Pulse magnetizes conductive lodestone into Magnetized. "
            "Magma_Infuse melts heat_tolerant magma slag into Molten. "
            "Pressure_Crush fissures mineral stone into Fissured. "
            "Tectonic_Shift crystallizes quartz veins into Crystallized."
        ),
    ),
]


class ProceduralTransferWorldM104_E:
    """Instantiates a procedurally generated transfer world with compositional and multi-step dynamics."""

    def __init__(self, spec: WorldFamilySpecM104_E, seed: int = 42) -> None:
        self.spec = spec
        self.seed = seed
        self.rng = random.Random(seed)
        self.world_id = spec.world_id
        self.objects: dict[str, CausalObjectM104_B] = {}
        self.actions = list(spec.actions)
        self.states = list(spec.states)
        self.rules = spec.rules_text
        self._build_world()

    def _build_world(self) -> None:
        """Procedurally builds familiar and novel objects for this world family."""
        for mat_name, mat_tags in self.spec.materials:
            for form_name, form_desc, form_tags in self.spec.forms:
                name = f"{mat_name}_{form_name}"
                all_tags = set(mat_tags) | set(form_tags)
                desc = f"A {mat_name.lower().replace('_', ' ')} {form_desc}."
                self.objects[name] = CausalObjectM104_B(
                    name=name,
                    tags=all_tags,
                    description=desc,
                    is_novel=False,
                    family=self.spec.world_id,
                )

        exotic_prefixes = ["Prismatic", "Abyssal", "Hyperborean", "Titan", "Ethereal", "Vortex"]
        for pfix in exotic_prefixes:
            mat_name, mat_tags = self.spec.materials[self.rng.randint(0, len(self.spec.materials) - 1)]
            form_name, form_desc, form_tags = self.spec.forms[self.rng.randint(0, len(self.spec.forms) - 1)]
            name = f"{pfix}_{mat_name}_{form_name}"
            all_tags = set(mat_tags) | set(form_tags) | {"magical"}
            desc = f"A legendary {pfix.lower()} {mat_name.lower().replace('_', ' ')} {form_desc}."
            self.objects[name] = CausalObjectM104_B(
                name=name,
                tags=all_tags,
                description=desc,
                is_novel=True,
                family=f"{self.spec.world_id}_novel",
            )

    def transition_state(self, obj_name: str, action: str, second_action: str | None = None) -> str:
        """Compositional and multi-step causal transition logic."""
        obj = self.objects.get(obj_name)
        if not obj:
            return self.states[0]

        tags = obj.tags
        inter_state = "Dormant"

        # Step 1: First action transition
        if "Kindle" in action:
            inter_state = "Molten" if ("flammable" in tags or "organic" in tags) else "Extinguished"
        elif "Frostbite" in action:
            inter_state = "Frozen_Solid" if "crystalline" in tags else "Dormant"
        elif "Electrify" in action:
            inter_state = "Supercharged" if "conductive" in tags else "Dormant"
        elif "Sunlight" in action:
            inter_state = "Luminous" if "radiant" in tags else "Dormant"
        elif "Shadow" in action:
            inter_state = "Shrouded" if "shadow_infused" in tags else "Dormant"
        elif "Astral" in action:
            inter_state = "Celestial_Harmonized" if ("conductive" in tags or "silver" in tags) else "Phase_Shifted"
        elif "Graft" in action:
            inter_state = "Bio_Fused" if "heat_tolerant" in tags else "Desiccated"
        elif "Spore" in action:
            inter_state = "Spore_Infected" if "organic" in tags else "Dormant"
        elif "Synthesize" in action:
            inter_state = "Metabolized" if "conductive" in tags else "Dormant"
        elif "Time" in action:
            inter_state = "Time_Dilated" if "crystalline" in tags else "Dormant"
        elif "Resonance" in action:
            inter_state = "Harmonized" if "metal" in tags else "Shattered_Echo"
        elif "Stasis" in action:
            inter_state = "Stasis_Frozen" if ("optical" in tags or "radiant" in tags) else "Dormant"
        elif "Magnetic" in action:
            inter_state = "Magnetized" if ("conductive" in tags or "metal" in tags) else "Dormant"
        elif "Magma" in action:
            inter_state = "Molten" if ("flammable" in tags or "stone" in tags) else "Dormant"
        elif "Pressure" in action:
            inter_state = "Fissured" if "stone" in tags else "Crystallized"
        else:
            inter_state = self.states[0]

        # Step 2: Second action if multi-step sequence
        if second_action:
            if inter_state == "Molten" and "Frostbite" in second_action:
                return "Extinguished"
            if inter_state == "Supercharged" and "Submerge" in second_action:
                return "Extinguished"
            if inter_state == "Luminous" and "Shadow" in second_action:
                return "Shrouded"
            if inter_state == "Phase_Shifted" and "Astral" in second_action:
                return "Celestial_Harmonized"
            if inter_state == "Bio_Fused" and "Spore" in second_action:
                return "Spore_Infected"
            if inter_state == "Time_Dilated" and "Stasis" in second_action:
                return "Stasis_Frozen"
            if inter_state == "Molten" and "Magnetic" in second_action:
                return "Magnetized"
            return inter_state

        return inter_state

    def generate_datasets(self) -> dict[str, list[CausalItemM104_B]]:
        """Generates train, held_out_pair, novel_entity, and multistep splits."""
        all_items: list[CausalItemM104_B] = []
        item_id = 0

        # 1. Single-step transitions
        for obj_name, obj in self.objects.items():
            for act in self.actions:
                tgt = self.transition_state(obj_name, act)
                item_id += 1
                all_items.append(
                    CausalItemM104_B(
                        scenario_id=f"{self.world_id}_item_{item_id}",
                        category="novel_entity" if obj.is_novel else "familiar",
                        object_name=obj_name,
                        action_name=act,
                        second_action=None,
                        initial_state="Dormant",
                        expected_state=tgt,
                        rule_explanation=self.rules,
                        question_text=f"{obj.description} When we execute {act}, what condition is reached?",
                        canonical_answer=tgt,
                        family=self.world_id,
                        is_counterfactual=False,
                        paraphrased_question=f"Regarding {obj.description}, applying {act} results in what state?",
                        world_id=self.world_id,
                    )
                )

        # 2. Compositional multi-step transitions (Action 1 followed by Action 2)
        multistep_items: list[CausalItemM104_B] = []
        for obj_name, obj in self.objects.items():
            for i in range(len(self.actions) - 1):
                act1 = self.actions[i]
                act2 = self.actions[(i + 1) % len(self.actions)]
                tgt2 = self.transition_state(obj_name, act1, second_action=act2)
                item_id += 1
                multistep_items.append(
                    CausalItemM104_B(
                        scenario_id=f"{self.world_id}_multi_{item_id}",
                        category="multistep",
                        object_name=obj_name,
                        action_name=act1,
                        second_action=act2,
                        initial_state="Dormant",
                        expected_state=tgt2,
                        rule_explanation=self.rules,
                        question_text=f"{obj.description} First {act1} is applied, then {act2}. What final state is achieved?",
                        canonical_answer=tgt2,
                        family=self.world_id,
                        is_counterfactual=False,
                        paraphrased_question=f"Sequential process on {obj.description}: {act1} followed by {act2}. What is the outcome?",
                        world_id=self.world_id,
                    )
                )

        familiar = [it for it in all_items if it.category == "familiar"]
        novel = [it for it in all_items if it.category == "novel_entity"]
        self.rng.shuffle(familiar)

        # 70% train familiar, 30% held-out pairs
        n_held_out = max(6, int(len(familiar) * 0.30))
        val_held_out = familiar[:n_held_out]
        train_items = familiar[n_held_out:]

        return {
            "train": train_items,
            "val_held_out_pair": val_held_out,
            "novel_entity": novel,
            "multistep": multistep_items,
        }


@dataclass
class SurvivalRunRecordM104_E:
    arm_name: str
    target_accuracy_pct: float
    target_reached: bool
    steps_to_target: int
    is_censored: bool
    step_zero_acc_pct: float
    final_accuracy_pct: float
    learning_curve: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SingleWorldTransferResultM104_E:
    world_id: str
    realm_name: str
    target_accuracy_pct: float
    experienced_record: SurvivalRunRecordM104_E
    fresh_record: SurvivalRunRecordM104_E
    language_pretrained_record: SurvivalRunRecordM104_E
    frozen_backbone_record: SurvivalRunRecordM104_E
    speedup_ratio: float | None
    speedup_status: str  # "OBSERVED", "CENSORED_LOWER_BOUND", or "BOTH_CENSORED"


@dataclass
class MultiWorldTransferReportM104_E:
    num_worlds_evaluated: int
    world_results: list[SingleWorldTransferResultM104_E]
    mean_experienced_step_zero_acc: float
    mean_fresh_step_zero_acc: float
    mean_experienced_final_acc: float
    mean_fresh_final_acc: float
    mean_language_pretrained_final_acc: float
    mean_frozen_backbone_final_acc: float
    mean_experienced_steps_to_target: float
    mean_fresh_steps_to_target: float
    experienced_target_reach_rate_pct: float
    fresh_target_reach_rate_pct: float
    overall_transfer_advantage_demonstrated: bool
    scientific_summary: str
