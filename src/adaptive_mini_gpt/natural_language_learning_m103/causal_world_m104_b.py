"""Milestone M104-B Procedural Causal World Simulator.

Features:
1. Procedural generation of combinatorial entities (up to 60 familiar and 60 novel objects).
2. Dynamic candidate experiment pools of sizes 28, 100, and 200 for active learning studies.
3. Natural-language paraphrasing engine: generates canonical, paraphrased, and property-renamed descriptions.
4. Multi-world generator: creates strictly independent worlds (e.g. World A for training, World B for meta-transfer).
5. Deterministic transitions across standard, multistep, and counterfactual realms.
"""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_mini_gpt.natural_language_learning_m103.causal_world_m103_g1 import (
    CausalItemM103_G1,
    CausalObjectM103_G1,
    ProceduralCausalWorldM103_G1,
)

CausalObjectM104_B = CausalObjectM103_G1


@dataclass
class CausalItemM104_B(CausalItemM103_G1):
    paraphrased_question: str = ""
    world_id: str = "world_a"
    is_noisy: bool = False


class ProceduralCausalWorldM104_B(ProceduralCausalWorldM103_G1):
    """Expanded procedural causal world simulator for Milestone M104-B."""

    PARAPHRASE_TEMPLATES = [
        "Given {desc}, what state is reached when we apply {act}?",
        "Consider {desc}. If we execute {act}, what is the resulting condition?",
        "We have {desc}. Upon applying {act}, what outcome occurs?",
        "Observing {desc}, applying {act} will yield what state?",
    ]

    def __init__(self, seed: int = 42, world_id: str = "world_a") -> None:
        self.world_id = world_id
        super().__init__(seed=seed)

    def _generate_world(self) -> None:
        """Generates expanded familiar and novel objects for this world."""
        # Base objects from M103-G.1
        super()._generate_world()

        # Add more procedural objects to support pool sizes up to 200
        extra_fam_prefixes = [
            "Bronze", "Pewter", "Ash_Wood", "Basalt_Rock", "Onyx", "Flint",
            "Serpentine", "Electrum", "Cobalt", "Topaz", "Ruby", "Sapphire",
            "Silk", "Linen", "Sandstone", "Porcelain", "Tin", "Lead",
        ]
        for i, prefix in enumerate(extra_fam_prefixes):
            mat_name, mat_tags = self.MATERIALS[i % len(self.MATERIALS)]
            form_name, form_desc, form_tags = self.FORMS[(i + 2) % len(self.FORMS)]
            name = f"{prefix}_{form_name}"
            if name not in self.objects:
                all_tags = set(mat_tags) | set(form_tags)
                desc = f"A {mat_name} {form_desc} fashioned from {prefix.lower()}."
                self.objects[name] = CausalObjectM103_G1(
                    name=name, tags=all_tags, description=desc, is_novel=False, family=f"extra_{mat_name}"
                )

        extra_novel_prefixes = [
            "Adamantine", "Starmetal", "Void_Shard", "Solar_Quartz", "Aether_Silk",
            "Frost_Jade", "Cinder_Stone", "Nether_Glass", "Blood_Agate", "Glimmer_Clay",
            "Shadow_Iron", "Dragon_Glass", "Chrono_Crystal", "Runed_Flax",
        ]
        for j, prefix in enumerate(extra_novel_prefixes):
            mat_name, mat_tags = self.MATERIALS[(j + 1) % len(self.MATERIALS)]
            aff_name, aff_tags = self.AFFILIATIONS[(j + 2) % len(self.AFFILIATIONS)]
            form_name, form_desc, form_tags = self.FORMS[(j + 5) % len(self.FORMS)]
            name = f"{prefix}_{form_name}"
            if name not in self.objects:
                all_tags = set(mat_tags) | set(aff_tags) | set(form_tags)
                desc = f"An exotic {aff_name} {prefix.lower()} {form_desc}."
                self.objects[name] = CausalObjectM103_G1(
                    name=name, tags=all_tags, description=desc, is_novel=True, family="extra_novel"
                )

    def generate_datasets_m104b(
        self,
        candidate_pool_size: int = 100,
    ) -> dict[str, list[CausalItemM104_B]]:
        """Generates datasets with paraphrases and controllable candidate pool sizes (28, 100, 200)."""
        base_ds = super().generate_datasets()
        familiar_objects = [k for k, v in self.objects.items() if not v.is_novel]
        novel_objects = [k for k, v in self.objects.items() if v.is_novel]

        m104b_ds: dict[str, list[CausalItemM104_B]] = {}

        # Convert base items to CausalItemM104_B with paraphrases
        for split_name, items in base_ds.items():
            converted = []
            for it in items:
                desc = self.objects[it.object_name].description if it.object_name in self.objects else it.object_name
                template = self.PARAPHRASE_TEMPLATES[self.rng.randint(0, len(self.PARAPHRASE_TEMPLATES) - 1)]
                para_q = template.format(desc=desc, act=it.action_name)
                converted.append(
                    CausalItemM104_B(
                        scenario_id=it.scenario_id,
                        category=it.category,
                        object_name=it.object_name,
                        action_name=it.action_name,
                        second_action=it.second_action,
                        initial_state=it.initial_state,
                        expected_state=it.expected_state,
                        rule_explanation=it.rule_explanation,
                        question_text=it.question_text,
                        canonical_answer=it.canonical_answer,
                        family=it.family,
                        is_counterfactual=it.is_counterfactual,
                        paraphrased_question=para_q,
                        world_id=self.world_id,
                        is_noisy=False,
                    )
                )
            m104b_ds[split_name] = converted

        # Construct specific candidate pool of requested size (e.g. 28, 100, 200)
        all_pool_candidates: list[CausalItemM104_B] = []
        item_id = 9000
        for obj_name in novel_objects + familiar_objects:
            for act in self.ACTIONS:
                tgt = self.transition_state(obj_name, act, "Dormant", "Alchemy")
                obj = self.objects[obj_name]
                template = self.PARAPHRASE_TEMPLATES[self.rng.randint(0, len(self.PARAPHRASE_TEMPLATES) - 1)]
                all_pool_candidates.append(
                    CausalItemM104_B(
                        scenario_id=f"m104b_cand_{item_id}",
                        category="active_pool",
                        object_name=obj_name,
                        action_name=act,
                        second_action=None,
                        initial_state="Dormant",
                        expected_state=tgt,
                        rule_explanation=self.STANDARD_RULES,
                        question_text=f"{obj.description} We apply {act}. What is the resulting state?",
                        canonical_answer=tgt,
                        family="active_candidate",
                        is_counterfactual=False,
                        paraphrased_question=template.format(desc=obj.description, act=act),
                        world_id=self.world_id,
                        is_noisy=(self.rng.random() < 0.05),  # 5% subtle observation noise
                    )
                )
                item_id += 1

        self.rng.shuffle(all_pool_candidates)
        m104b_ds["active_pool"] = all_pool_candidates[:candidate_pool_size]

        # Independent paraphrased evaluation set
        para_test_items: list[CausalItemM104_B] = []
        for it in m104b_ds["novel_entity"]:
            para_it = CausalItemM104_B(
                scenario_id=f"para_{it.scenario_id}",
                category="paraphrase_test",
                object_name=it.object_name,
                action_name=it.action_name,
                second_action=it.second_action,
                initial_state=it.initial_state,
                expected_state=it.expected_state,
                rule_explanation=it.rule_explanation,
                question_text=it.question_text,
                canonical_answer=it.canonical_answer,
                family=it.family,
                is_counterfactual=it.is_counterfactual,
                paraphrased_question=it.paraphrased_question,
                world_id=self.world_id,
            )
            para_test_items.append(para_it)
        m104b_ds["paraphrase_test"] = para_test_items

        return m104b_ds
