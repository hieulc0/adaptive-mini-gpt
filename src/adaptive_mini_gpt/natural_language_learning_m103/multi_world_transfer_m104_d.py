"""Milestone M104-D Multi-World Forward Transfer Benchmark with Right-Censored Survival Accounting.

Features:
1. Five Procedural Transfer World Families:
   - World T1: Elemental Realm (Pyromancy, Hydromancy, Cryomancy, Electromancy)
   - World T2: Celestial Reliquary (Solar, Lunar, Astral, Umbral)
   - World T3: Biomechanical Fabric (Chitin, Bio-Polymer, Fungal, Neural)
   - World T4: Chrono-Crystalline (Temporal, Resonant, Stasis, Harmonic)
   - World T5: Subterranean Geode (Ferromagnetic, Geothermal, Basaltic, Tectonic)
2. Right-Censored Survival Accounting:
   - Tracks time-to-target accuracy with explicit `is_censored` status.
   - Refuses to report false empirical ratios when controls time out without reaching target.
3. Multi-Arm Benchmark:
   - Experienced Persistent Student (trained on World A)
   - Matched Fresh Student (scratch baseline)
   - Language-Pretrained Control (corpus text pretraining without causal experience)
   - Frozen Backbone Control (World A weights frozen, adapter only)
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
class WorldFamilySpecM104_D:
    world_id: str
    realm_name: str
    materials: list[tuple[str, list[str]]]
    forms: list[tuple[str, str, list[str]]]
    actions: list[str]
    states: list[str]
    rules_text: str


WORLD_FAMILIES_M104_D: list[WorldFamilySpecM104_D] = [
    WorldFamilySpecM104_D(
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
        rules_text="Kindle burns flammable minerals into Molten. Frostbite freezes crystalline into Frozen_Solid. Electrify charges conductive into Supercharged.",
    ),
    WorldFamilySpecM104_D(
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
        rules_text="Channel_Sunlight turns radiant sunstones Luminous. Shadow_Veil shrouds eclipse glass into Shrouded. Astral_Pulse harmonizes star metal.",
    ),
    WorldFamilySpecM104_D(
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
        rules_text="Graft fuses heat_tolerant chitin into Bio_Fused. Spore_Burst infects organic fungal networks. Synthesize metabolizes conductive biopolymers.",
    ),
    WorldFamilySpecM104_D(
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
        rules_text="Accelerate_Time dilates chrono quartz. Harmonic_Resonance harmonizes tuning metal into Harmonized. Stasis_Lock freezes phase crystals.",
    ),
    WorldFamilySpecM104_D(
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
        rules_text="Magnetic_Pulse magnetizes conductive lodestone. Magma_Infuse melts heat_tolerant slag into Molten. Pressure_Crush fissures deep stone.",
    ),
]


class ProceduralTransferWorldM104_D:
    """Instantiates a procedurally generated transfer world based on a specific family specification."""

    def __init__(self, spec: WorldFamilySpecM104_D, seed: int = 42) -> None:
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

        # Generate novel objects by combining exotic prefixes
        exotic_prefixes = ["Prismatic", "Abyssal", "Hyperborean", "Titan"]
        for pfix in exotic_prefixes:
            mat_name, mat_tags = self.spec.materials[self.rng.randint(0, len(self.spec.materials) - 1)]
            form_name, form_desc, form_tags = self.spec.forms[self.rng.randint(0, len(self.spec.forms) - 1)]
            name = f"{pfix}_{mat_name}_{form_name}"
            all_tags = set(mat_tags) | set(form_tags) | {"magical"}
            desc = f"A legendary {pfix.lower()} {mat_name.lower()} {form_desc}."
            self.objects[name] = CausalObjectM104_B(
                name=name,
                tags=all_tags,
                description=desc,
                is_novel=True,
                family=f"{self.spec.world_id}_novel",
            )

    def transition_state(self, obj_name: str, action: str) -> str:
        """Deterministic causal outcome function based on object tags and action."""
        obj = self.objects.get(obj_name)
        if not obj:
            return self.states[0]

        tags = obj.tags
        # World T1
        if "Kindle" in action:
            return "Molten" if "flammable" in tags else "Extinguished"
        if "Frostbite" in action:
            return "Frozen_Solid" if "crystalline" in tags else "Dormant"
        if "Electrify" in action:
            return "Supercharged" if "conductive" in tags else "Dormant"

        # World T2
        if "Sunlight" in action:
            return "Luminous" if "radiant" in tags else "Dormant"
        if "Shadow" in action:
            return "Shrouded" if "shadow_infused" in tags else "Dormant"
        if "Astral" in action:
            return "Celestial_Harmonized" if "conductive" in tags or "silver" in tags else "Phase_Shifted"

        # World T3
        if "Graft" in action:
            return "Bio_Fused" if "heat_tolerant" in tags else "Desiccated"
        if "Spore" in action:
            return "Spore_Infected" if "organic" in tags else "Dormant"
        if "Synthesize" in action:
            return "Metabolized" if "conductive" in tags else "Dormant"

        # World T4
        if "Time" in action:
            return "Time_Dilated" if "crystalline" in tags else "Dormant"
        if "Resonance" in action:
            return "Harmonized" if "metal" in tags else "Shattered_Echo"
        if "Stasis" in action:
            return "Stasis_Frozen" if "optical" in tags or "radiant" in tags else "Dormant"

        # World T5
        if "Magnetic" in action:
            return "Magnetized" if "conductive" in tags or "metal" in tags else "Dormant"
        if "Magma" in action:
            return "Molten" if "flammable" in tags or "stone" in tags else "Dormant"
        if "Pressure" in action:
            return "Fissured" if "stone" in tags else "Crystallized"

        return self.states[0]

    def generate_datasets(self) -> dict[str, list[CausalItemM104_B]]:
        """Generates train, held_out_pair, and novel_entity splits."""
        all_items: list[CausalItemM104_B] = []
        item_id = 0

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
        }


@dataclass
class SurvivalRunRecordM104_D:
    arm_name: str
    target_accuracy_pct: float
    target_reached: bool
    steps_to_target: int
    is_censored: bool
    final_accuracy_pct: float
    learning_curve: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SingleWorldTransferResultM104_D:
    world_id: str
    realm_name: str
    target_accuracy_pct: float
    experienced_record: SurvivalRunRecordM104_D
    fresh_record: SurvivalRunRecordM104_D
    language_pretrained_record: SurvivalRunRecordM104_D
    frozen_backbone_record: SurvivalRunRecordM104_D
    speedup_ratio: float | None
    speedup_status: str  # "OBSERVED", "CENSORED_LOWER_BOUND", or "BOTH_CENSORED"


@dataclass
class MultiWorldTransferReportM104_D:
    num_worlds_evaluated: int
    world_results: list[SingleWorldTransferResultM104_D]
    mean_experienced_final_acc: float
    mean_fresh_final_acc: float
    mean_language_pretrained_final_acc: float
    mean_frozen_backbone_final_acc: float
    experienced_target_reach_rate_pct: float
    fresh_target_reach_rate_pct: float
    overall_transfer_advantage_demonstrated: bool
    scientific_summary: str
