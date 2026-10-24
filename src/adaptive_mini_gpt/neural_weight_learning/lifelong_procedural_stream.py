"""Phase 6 & 7: Lifelong Procedural Stream Engine & Autonomous Skill Retention.

Evaluates continual procedural skill learning across a sequence of episodes featuring:
- Novel task learning with candidate training and held-out qualification.
- Skill retention in modular repertoire.
- Returning task encounters separated by unrelated intervening episodes.
- Adversarial rule inversions (e.g. INVERTED_CYCLE_PERM).
- Direct comparison across 5 policies:
    1. SELECTIVE_NO_EXTRA_PROBES: Prompt-observable rule routing with zero-shot reuse and 0 extra probe queries.
    2. SELECTIVE_RETAIN: Evidence-probed routing on dev set.
    3. RESET: Wipe acquired skills after every episode.
    4. FRESH_ADAPT: Perform fresh gradient adaptation on every task.
    5. ALWAYS_REUSE: Reuse only initial source skill repertoire.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.neural_weight_learning.pathway_instrumentation import (
    NeuralPathwayInstrumenter,
)
from adaptive_mini_gpt.neural_weight_learning.procedural_tasks import (
    ProceduralRuleFamily,
    ProceduralTaskGenerator,
)
from adaptive_mini_gpt.neural_weight_learning.sequence_adapter import Rank16SequenceAdapter

logger = logging.getLogger(__name__)


class ProceduralStreamPolicy(StrEnum):
    """Evaluated lifelong continual learning policies."""

    SELECTIVE_NO_EXTRA_PROBES = "SELECTIVE_NO_EXTRA_PROBES"
    SELECTIVE_RETAIN = "SELECTIVE_RETAIN"
    RESET = "RESET"
    FRESH_ADAPT = "FRESH_ADAPT"
    ALWAYS_REUSE = "ALWAYS_REUSE"


@dataclass
class StreamEpisodeResult:
    """Performance and cost metrics for a single episode under a policy."""

    episode_index: int
    rule_family: str
    is_repeated_encounter: bool
    action_taken: str
    selected_skill_id: str | None
    zero_shot_exact_match: float
    final_exact_match: float
    final_sequence_loss: float
    training_steps_incurred: int
    probe_queries_incurred: int
    stored_parameters: int
    active_parameters: int


@dataclass
class PolicyStreamSummary:
    """Aggregated stream summary across all episodes for a policy."""

    policy_name: str
    macro_exact_match: float
    repeated_zero_shot_exact_match: float
    total_training_steps: int
    total_probe_queries: int
    physical_compute_utility: float
    final_stored_parameters: int
    successful_reuses: int
    episodes: list[StreamEpisodeResult]


class LifelongProceduralStreamEngine:
    """Execution engine for multi-seed lifelong procedural task streams."""

    @classmethod
    def run_stream(
        cls,
        model: MiniGPT,
        policy: ProceduralStreamPolicy,
        seed: int,
        device: torch.device,
        num_episodes: int = 10,
    ) -> PolicyStreamSummary:
        """Run a sequential stream of procedural episodes under a specified policy."""
        task_gen = ProceduralTaskGenerator(vocab_size=16)

        # Canonical sequence schedule with recurring tasks
        # 0: CYCLE_PERM (novel)
        # 1: REVERSE_SEQ (novel)
        # 2: CAESAR_SHIFT_MOD3 (novel)
        # 3: ALT_SWAP (novel)
        # 4: CYCLE_PERM (1st return, 3 intervening eps)
        # 5: REVERSE_SEQ (1st return, 3 intervening eps)
        # 6: INVERTED_CYCLE_PERM (adversarial)
        # 7: CAESAR_SHIFT_MOD3 (1st return, 4 intervening eps)
        # 8: CYCLE_PERM (2nd return)
        # 9: ALT_SWAP (1st return, 5 intervening eps)
        schedule = [
            (ProceduralRuleFamily.CAESAR_SHIFT_MOD3, False),
            (ProceduralRuleFamily.CAESAR_SHIFT_MOD5, False),
            (ProceduralRuleFamily.AFFINE_TRANSFORM, False),
            (ProceduralRuleFamily.COMPLEMENT_REFLECT, False),
            (ProceduralRuleFamily.CAESAR_SHIFT_MOD3, True),
            (ProceduralRuleFamily.CAESAR_SHIFT_MOD5, True),
            (ProceduralRuleFamily.NONLINEAR_SBOX, False),
            (ProceduralRuleFamily.AFFINE_TRANSFORM, True),
            (ProceduralRuleFamily.CAESAR_SHIFT_MOD3, True),
            (ProceduralRuleFamily.COMPLEMENT_REFLECT, True),
        ][:num_episodes]

        # Repertoire registry: skill_id -> (rule_family, adapter)
        repertoire: dict[str, tuple[ProceduralRuleFamily, Rank16SequenceAdapter]] = {}

        episode_results: list[StreamEpisodeResult] = []
        total_steps = 0
        total_queries = 0
        successful_reuses = 0

        for ep_idx, (rule_fam, is_repeat) in enumerate(schedule):
            task = task_gen.generate_task(
                task_id=f"ep{ep_idx}_{rule_fam.value}",
                rule_family=rule_fam,
                seed=seed * 1000 + ep_idx,
            )

            test_h, test_y = NeuralPathwayInstrumenter.extract_symbol_hidden_states(
                model, task.sealed_test_examples, device
            )
            dev_h, dev_y = NeuralPathwayInstrumenter.extract_symbol_hidden_states(
                model, task.dev_examples, device
            )

            # Determine routing / adaptation action
            chosen_adapter: Rank16SequenceAdapter | None = None
            action_taken = "UNKNOWN"
            steps_incurred = 0
            queries_incurred = 0
            selected_id: str | None = None
            zero_shot_em = 0.0

            if policy == ProceduralStreamPolicy.SELECTIVE_NO_EXTRA_PROBES:
                # Prompt-observable rule matching
                # Check if we already retain a qualified skill for this rule family
                for sk_id, (fam, ad) in repertoire.items():
                    if fam == rule_fam:
                        chosen_adapter = ad
                        selected_id = sk_id
                        action_taken = "ZERO_SHOT_REUSE"
                        break

                if chosen_adapter is not None:
                    # Evaluate zero-shot
                    chosen_adapter.eval()
                    with torch.no_grad():
                        logits = chosen_adapter(test_h)
                        preds = torch.argmax(logits, dim=-1)
                        zero_shot_em = float((preds == test_y).all(dim=-1).float().mean().item())
                    successful_reuses += 1
                else:
                    # Train fresh skill
                    action_taken = "TRAIN_AND_RETAIN"
                    new_ad = Rank16SequenceAdapter().to(device)
                    new_ad, _ = NeuralPathwayInstrumenter.train_and_instrument(
                        model=model,
                        adapter=new_ad,
                        task=task,
                        adapter_id=f"skill_{rule_fam.value}",
                        device=device,
                        steps=50,
                    )
                    steps_incurred = 50
                    queries_incurred = 0  # No extra probe queries
                    # Qualify on held-out qualification split
                    qual_h, qual_y = NeuralPathwayInstrumenter.extract_symbol_hidden_states(
                        model, task.qualification_examples, device
                    )
                    new_ad.eval()
                    with torch.no_grad():
                        q_preds = torch.argmax(new_ad(qual_h), dim=-1)
                        q_em = float((q_preds == qual_y).all(dim=-1).float().mean().item())
                    if q_em >= 0.80:
                        sk_id = f"skill_{rule_fam.value}"
                        repertoire[sk_id] = (rule_fam, new_ad)
                        chosen_adapter = new_ad
                        selected_id = sk_id
                    else:
                        chosen_adapter = new_ad

            elif policy == ProceduralStreamPolicy.SELECTIVE_RETAIN:
                # Probe existing repertoire on dev examples
                best_dev_em = -1.0
                best_candidate = None
                best_id = None

                for sk_id, (_fam, ad) in repertoire.items():
                    queries_incurred += len(task.dev_examples)
                    ad.eval()
                    with torch.no_grad():
                        d_preds = torch.argmax(ad(dev_h), dim=-1)
                        d_em = float((d_preds == dev_y).all(dim=-1).float().mean().item())
                    if d_em > best_dev_em:
                        best_dev_em = d_em
                        best_candidate = ad
                        best_id = sk_id

                if best_dev_em >= 0.80 and best_candidate is not None:
                    chosen_adapter = best_candidate
                    selected_id = best_id
                    action_taken = "ZERO_SHOT_REUSE"
                    successful_reuses += 1
                    with torch.no_grad():
                        z_preds = torch.argmax(chosen_adapter(test_h), dim=-1)
                        zero_shot_em = float((z_preds == test_y).all(dim=-1).float().mean().item())
                else:
                    action_taken = "TRAIN_AND_RETAIN"
                    new_ad = Rank16SequenceAdapter().to(device)
                    new_ad, _ = NeuralPathwayInstrumenter.train_and_instrument(
                        model=model,
                        adapter=new_ad,
                        task=task,
                        adapter_id=f"skill_{rule_fam.value}",
                        device=device,
                        steps=50,
                    )
                    steps_incurred = 50
                    sk_id = f"skill_{rule_fam.value}_{ep_idx}"
                    repertoire[sk_id] = (rule_fam, new_ad)
                    chosen_adapter = new_ad
                    selected_id = sk_id

            elif policy == ProceduralStreamPolicy.RESET:
                # Always discard repertoire
                repertoire.clear()
                action_taken = "RESET_AND_TRAIN"
                new_ad = Rank16SequenceAdapter().to(device)
                new_ad, _ = NeuralPathwayInstrumenter.train_and_instrument(
                    model=model,
                    adapter=new_ad,
                    task=task,
                    adapter_id="temp_skill",
                    device=device,
                    steps=50,
                )
                steps_incurred = 50
                chosen_adapter = new_ad

            elif policy == ProceduralStreamPolicy.FRESH_ADAPT:
                # Always train fresh on every task
                action_taken = "FRESH_ADAPT"
                new_ad = Rank16SequenceAdapter().to(device)
                new_ad, _ = NeuralPathwayInstrumenter.train_and_instrument(
                    model=model,
                    adapter=new_ad,
                    task=task,
                    adapter_id="fresh_skill",
                    device=device,
                    steps=50,
                )
                steps_incurred = 50
                chosen_adapter = new_ad

            elif policy == ProceduralStreamPolicy.ALWAYS_REUSE:
                # If repertoire has any skill, reuse the first one; never train
                if len(repertoire) > 0:
                    first_k = next(iter(repertoire.keys()))
                    chosen_adapter = repertoire[first_k][1]
                    selected_id = first_k
                    action_taken = "ALWAYS_REUSE_EXISTING"
                else:
                    # Train initial seed skill
                    new_ad = Rank16SequenceAdapter().to(device)
                    new_ad, _ = NeuralPathwayInstrumenter.train_and_instrument(
                        model=model,
                        adapter=new_ad,
                        task=task,
                        adapter_id="initial_skill",
                        device=device,
                        steps=50,
                    )
                    repertoire["init_skill"] = (rule_fam, new_ad)
                    chosen_adapter = new_ad
                    selected_id = "init_skill"
                    steps_incurred = 50
                    action_taken = "INIT_FIRST_SKILL"

            # Final evaluation on sealed test set
            if chosen_adapter is not None:
                chosen_adapter.eval()
                with torch.no_grad():
                    logits = chosen_adapter(test_h)
                    loss = float(F.cross_entropy(logits.view(-1, 16), test_y.view(-1)).item())
                    preds = torch.argmax(logits, dim=-1)
                    em = float((preds == test_y).all(dim=-1).float().mean().item())
            else:
                loss = 5.0
                em = 0.0

            stored_params = len(repertoire) * 2592
            active_params = 2592

            total_steps += steps_incurred
            total_queries += queries_incurred

            episode_results.append(
                StreamEpisodeResult(
                    episode_index=ep_idx,
                    rule_family=rule_fam.value,
                    is_repeated_encounter=is_repeat,
                    action_taken=action_taken,
                    selected_skill_id=selected_id,
                    zero_shot_exact_match=round(zero_shot_em, 4),
                    final_exact_match=round(em, 4),
                    final_sequence_loss=round(loss, 4),
                    training_steps_incurred=steps_incurred,
                    probe_queries_incurred=queries_incurred,
                    stored_parameters=stored_params,
                    active_parameters=active_params,
                )
            )

        macro_em = float(np.mean([ep.final_exact_match for ep in episode_results]))
        repeat_eps = [ep for ep in episode_results if ep.is_repeated_encounter]
        repeat_zero_shot = (
            float(np.mean([ep.zero_shot_exact_match for ep in repeat_eps])) if repeat_eps else 0.0
        )

        # Physical Compute Utility:
        # Penalizes training steps (1e-4 per step) and probe queries (5e-5 per query)
        compute_penalty = (total_steps * 1e-4) + (total_queries * 5e-5)
        physical_utility = max(0.0, macro_em - compute_penalty)

        return PolicyStreamSummary(
            policy_name=policy.value,
            macro_exact_match=round(macro_em, 4),
            repeated_zero_shot_exact_match=round(repeat_zero_shot, 4),
            total_training_steps=total_steps,
            total_probe_queries=total_queries,
            physical_compute_utility=round(physical_utility, 4),
            final_stored_parameters=len(repertoire) * 2592,
            successful_reuses=successful_reuses,
            episodes=episode_results,
        )
