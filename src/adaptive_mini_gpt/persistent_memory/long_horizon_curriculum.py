"""Milestone M96 Phase 3: Long-Horizon Multi-Domain Curriculum.

Constructs a 50-100 episode sequential lifelong learning stream spanning:
1. Factual Knowledge Acquisition (Entities, founding dates, locations, properties).
2. Factual Updates & Corrections (Revised attributes, superseding prior values).
3. Delayed Factual Recall (Querying knowledge learned 10-50 episodes earlier).
4. Foundational Neural Procedural Skills (Arithmetic, Conditions, Loops from M91-M95).
5. Fact-Skill Compositional Generalization (Retrieving a fact and applying neural execution).
6. Epistemic Contradiction & Abstention (Unsupported entities, contradictory statements).
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.cross_domain_transfer.cross_domain_curriculum import (
    ComputationalConcept,
    CrossDomainCurriculumGenerator,
    CrossDomainExample,
    Domain,
)

MODULUS = 16


class EpisodeType(str, enum.Enum):
    """Categorical type of long-horizon episode."""

    FACT_LEARN = "FACT_LEARN"
    FACT_UPDATE = "FACT_UPDATE"
    FACT_QUERY = "FACT_QUERY"
    DELAYED_FACT_QUERY = "DELAYED_FACT_QUERY"
    PROCEDURAL_SKILL = "PROCEDURAL_SKILL"
    FACT_SKILL_COMPOSITION = "FACT_SKILL_COMPOSITION"
    UNSUPPORTED_QUERY = "UNSUPPORTED_QUERY"


@dataclass(frozen=True)
class LongHorizonEpisode:
    """An individual episode in the lifelong learning stream."""

    episode_id: int
    episode_type: EpisodeType
    prompt_text: str
    target_output: Any
    entity: str | None = None
    attribute: str | None = None
    input_value: Any | None = None
    is_correction: bool = False
    is_contradictory: bool = False
    requires_abstention: bool = False
    procedural_example: CrossDomainExample | None = None


class LongHorizonCurriculumGenerator:
    """Generates continuous multi-domain episodes over extended horizons."""

    def __init__(self, seed: int = 42, num_episodes: int = 60) -> None:
        self.seed = seed
        self.num_episodes = num_episodes
        self.rng = random.Random(seed)
        self.prog_gen = CrossDomainCurriculumGenerator(seed=seed)

    def generate_curriculum(self) -> list[LongHorizonEpisode]:
        """Generate a sequential 60-episode lifelong learning stream."""
        episodes: list[LongHorizonEpisode] = []

        # Canonical factual entities
        entities = [
            ("atlas_labs", "founding_year", 2024, 2025),
            ("hyperscale", "headquarters", "zurich", "geneva"),
            ("alphatech", "initial_register", 12, 14),
            ("novacore", "datacenter_count", 8, 10),
            ("tensorworks", "lead_architect", "elena", "marcus"),
            ("deepflow", "primary_compiler", "mlir", "triton"),
        ]

        # Tracking state for delayed queries
        learned_facts: dict[tuple[str, str], Any] = {}
        updated_facts: set[tuple[str, str]] = set()

        ep_id = 1

        # Phase A: Initial Factual Acquisition (Episodes 1-10)
        for ent, attr, val1, _ in entities:
            episodes.append(
                LongHorizonEpisode(
                    episode_id=ep_id,
                    episode_type=EpisodeType.FACT_LEARN,
                    prompt_text=f"Learn: {ent.replace('_', ' ').title()} {attr.replace('_', ' ')} is {val1}.",
                    target_output=val1,
                    entity=ent,
                    attribute=attr,
                    input_value=val1,
                )
            )
            learned_facts[(ent, attr)] = val1
            ep_id += 1

        # Immediate Factual Queries (Episodes 11-14)
        for ent, attr, val1, _ in entities[:4]:
            episodes.append(
                LongHorizonEpisode(
                    episode_id=ep_id,
                    episode_type=EpisodeType.FACT_QUERY,
                    prompt_text=f"Query: What is the {attr.replace('_', ' ')} of {ent.replace('_', ' ').title()}?",
                    target_output=val1,
                    entity=ent,
                    attribute=attr,
                )
            )
            ep_id += 1

        # Phase B: Procedural Neural Skills (Episodes 15-24)
        domains = [Domain.PROGRAM, Domain.NATURAL_LANGUAGE, Domain.PLANNING]
        for i in range(10):
            dom = domains[i % 3]
            # Alternate arithmetic, condition, loop, nested
            if i % 4 == 0:
                ex = self.prog_gen._make_arithmetic_example(dom, f"p_ar_{i}", (i * 3) % MODULUS, "+", 4)
            elif i % 4 == 1:
                ex = self.prog_gen._make_condition_example(dom, f"p_cond_{i}", (i * 2) % MODULUS, ">", 3)
            elif i % 4 == 2:
                ex = self.prog_gen._make_loop_example(dom, f"p_loop_{i}", 1, 3, 2)
            else:
                ex = self.prog_gen._make_nested_example(dom, f"p_nest_{i}", 2, 3, 2)

            episodes.append(
                LongHorizonEpisode(
                    episode_id=ep_id,
                    episode_type=EpisodeType.PROCEDURAL_SKILL,
                    prompt_text=f"Skill Execution ({dom.value}): {ex.raw_text}",
                    target_output=ex.expected_output,
                    procedural_example=ex,
                )
            )
            ep_id += 1

        # Phase C: Fact Updates & Corrections (Episodes 25-30)
        for ent, attr, _, val2 in entities[:4]:
            episodes.append(
                LongHorizonEpisode(
                    episode_id=ep_id,
                    episode_type=EpisodeType.FACT_UPDATE,
                    prompt_text=f"Correction: {ent.replace('_', ' ').title()} {attr.replace('_', ' ')} is updated to {val2}.",
                    target_output=val2,
                    entity=ent,
                    attribute=attr,
                    input_value=val2,
                    is_correction=True,
                )
            )
            learned_facts[(ent, attr)] = val2
            updated_facts.add((ent, attr))
            ep_id += 1

        # Phase D: Updated Fact Verification (Episodes 31-34)
        for ent, attr, _, val2 in entities[:4]:
            episodes.append(
                LongHorizonEpisode(
                    episode_id=ep_id,
                    episode_type=EpisodeType.FACT_QUERY,
                    prompt_text=f"Query: What is the current {attr.replace('_', ' ')} of {ent.replace('_', ' ').title()}?",
                    target_output=val2,
                    entity=ent,
                    attribute=attr,
                )
            )
            ep_id += 1

        # Phase E: Fact-Skill Composition (Episodes 35-44)
        # Model retrieves factual register value and executes neural arithmetic on it!
        for i in range(10):
            ent, attr = ("alphatech", "initial_register") if i % 2 == 0 else ("novacore", "datacenter_count")
            fact_val = learned_facts[(ent, attr)]
            step_add = (i + 2) % MODULUS
            reps = 2
            # Target output is ((fact_val + step_add * reps) % MODULUS)
            curr = fact_val % MODULUS
            for _ in range(reps):
                curr = (curr + step_add) % MODULUS

            prompt = (
                f"Fact-Skill Composition: Retrieve {attr} of {ent} and execute {reps} repeated additions of {step_add}."
            )
            episodes.append(
                LongHorizonEpisode(
                    episode_id=ep_id,
                    episode_type=EpisodeType.FACT_SKILL_COMPOSITION,
                    prompt_text=prompt,
                    target_output=curr,
                    entity=ent,
                    attribute=attr,
                    input_value=step_add,
                )
            )
            ep_id += 1

        # Phase F: Delayed Factual Recall (Episodes 45-50)
        # Querying facts learned at Episode 1-6 (delayed by ~40-45 episodes!)
        for ent, attr, val1, val2 in entities:
            expected = val2 if (ent, attr) in updated_facts else val1
            episodes.append(
                LongHorizonEpisode(
                    episode_id=ep_id,
                    episode_type=EpisodeType.DELAYED_FACT_QUERY,
                    prompt_text=f"Delayed Recall: What is the verified {attr.replace('_', ' ')} of {ent.replace('_', ' ').title()}?",
                    target_output=expected,
                    entity=ent,
                    attribute=attr,
                )
            )
            ep_id += 1

        # Phase G: Epistemic Contradiction & Unsupported Queries (Episodes 51-55)
        unsupported_items = [
            ("omnicorp", "market_cap"),
            ("quantum_vault", "encryption_key"),
            ("atlantis", "population"),
            ("time_machine", "speed_of_light"),
            ("null_entity", "undefined_val"),
        ]
        for ent, attr in unsupported_items:
            episodes.append(
                LongHorizonEpisode(
                    episode_id=ep_id,
                    episode_type=EpisodeType.UNSUPPORTED_QUERY,
                    prompt_text=f"Query: What is the {attr} of {ent}?",
                    target_output="ABSTAIN",
                    entity=ent,
                    attribute=attr,
                    requires_abstention=True,
                )
            )
            ep_id += 1

        # Phase H: Final Long-Delayed Neural Skill Revisit (Episodes 56-60)
        for i in range(5):
            dom = domains[i % 3]
            ex = self.prog_gen._make_arithmetic_example(dom, f"revisit_{i}", (i + 4) % MODULUS, "+", 5)
            episodes.append(
                LongHorizonEpisode(
                    episode_id=ep_id,
                    episode_type=EpisodeType.PROCEDURAL_SKILL,
                    prompt_text=f"Long-Delayed Skill Revisit: {ex.raw_text}",
                    target_output=ex.expected_output,
                    procedural_example=ex,
                )
            )
            ep_id += 1

        return episodes
