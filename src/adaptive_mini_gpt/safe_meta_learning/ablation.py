"""Phase 3 & 4: Safe Causal Meta-Learning & Adaptation Engine for M61.

Evaluates 8 comparative arms:
- Arm A: NO_HISTORY (Static standard optimization)
- Arm B: RELEVANT_HISTORY (Raw kNN history without progress safeguard — M60 behavior)
- Arm C: RANDOM_UNRELATED_HISTORY (Random synthetic traces)
- Arm D: SHUFFLED_OUTCOMES_HISTORY (Permuted historical traces)
- Arm E: INCOMPATIBLE_HISTORY (Adversarial/conflicting traces)
- Arm F: SAFE_EXPERIENTIAL_PROGRESS (Relevant history WITH real-time progress checks & minimum budget guard)
- Arm G: FIXED_10, FIXED_20, FIXED_40 (Static baselines)
- Arm H: PROGRESS_COST_NO_MEMORY (Compute-matched progress early stopping without memory)

Measures learning curve AUC, final Macro F1, structural paired correctness,
compute efficiency, and history veto counts.
"""
from __future__ import annotations

import copy
import enum
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.causal_meta_learning.task_suite import (
    ObservableTaskComplexity,
    compute_observable_complexity,
)
from adaptive_mini_gpt.generalizable_learning.experiential_memory import (
    ExperientialMemoryStore,
    LearningEpisodeRecord,
)
from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.reliable_transfer.composition_models import ResidualSkillAdapter
from adaptive_mini_gpt.safe_meta_learning.controller import (
    ControllerAction,
    SafeControlDecision,
    SafeLearningProgressController,
)

logger = logging.getLogger(__name__)


class M61AblationArm(str, enum.Enum):
    ARM_A_NO_HISTORY = "Arm_A_No_History"
    ARM_B_RELEVANT_HISTORY_UNGUARDED = "Arm_B_Relevant_History_Unguarded"
    ARM_C_RANDOM_UNRELATED_HISTORY = "Arm_C_Random_Unrelated_History"
    ARM_D_SHUFFLED_OUTCOMES_HISTORY = "Arm_D_Shuffled_Outcomes_History"
    ARM_E_INCOMPATIBLE_HISTORY = "Arm_E_Incompatible_History"
    ARM_F_SAFE_EXPERIENTIAL_PROGRESS = "Arm_F_Safe_Experiential_Progress"
    ARM_G_FIXED_10_STEPS = "Arm_G_Fixed_10_Steps"
    ARM_G_FIXED_20_STEPS = "Arm_G_Fixed_20_Steps"
    ARM_G_FIXED_40_STEPS = "Arm_G_Fixed_40_Steps"
    ARM_H_PROGRESS_COST_NO_MEMORY = "Arm_H_Progress_Cost_No_Memory"


@dataclass
class M61AdaptationResult:
    """Outcome of an adaptation trajectory under M61 safe meta-control."""
    arm: M61AblationArm
    env_id: str
    seed: int
    total_steps_taken: int
    final_loss: float
    final_f1: float
    paired_correctness: float
    learning_curve_auc: float
    wall_clock_time_ms: float
    memory_lookup_overhead_ms: float
    compute_adjusted_efficiency: float
    step_history: list[int]
    loss_history: list[float]
    f1_history: list[float]
    decisions: list[SafeControlDecision]
    premature_stop_prevented: bool
    reset_triggered: bool


def build_m61_memory_store(arm: M61AblationArm, seed: int = 42) -> ExperientialMemoryStore:
    """Build experiential memory store for the designated ablation arm."""
    rng = random.Random(seed)
    store = ExperientialMemoryStore()

    if arm in (
        M61AblationArm.ARM_A_NO_HISTORY,
        M61AblationArm.ARM_G_FIXED_10_STEPS,
        M61AblationArm.ARM_G_FIXED_20_STEPS,
        M61AblationArm.ARM_G_FIXED_40_STEPS,
        M61AblationArm.ARM_H_PROGRESS_COST_NO_MEMORY,
    ):
        return store

    # High quality relevant historical records from meta-training tasks
    relevant_episodes = [
        LearningEpisodeRecord(
            episode_id="exp_chain_shallow",
            task_id="train_genomics_hierarchy",
            num_nodes=6,
            graph_diameter=5,
            graph_density=0.167,
            structural_complexity_index=7.4,
            prior_skill_id="source_hierarchy",
            initial_sample_loss=0.693,
            initial_sample_acc=0.55,
            steps_spent=25,
            final_loss=0.485,
            final_f1=0.75,
            learning_gain_f1=0.20,
            was_negative_transfer=False,
            recommended_action="TRAIN",
            recommended_budget_steps=25,
        ),
        LearningEpisodeRecord(
            episode_id="exp_mesh_cluster",
            task_id="train_mesh_cluster",
            num_nodes=6,
            graph_diameter=2,
            graph_density=0.40,
            structural_complexity_index=5.1,
            prior_skill_id="source_hierarchy",
            initial_sample_loss=0.710,
            initial_sample_acc=0.50,
            steps_spent=15,
            final_loss=0.520,
            final_f1=0.71,
            learning_gain_f1=0.21,
            was_negative_transfer=False,
            recommended_action="KEEP",
            recommended_budget_steps=15,
        ),
        LearningEpisodeRecord(
            episode_id="exp_deep_dag",
            task_id="train_deep_causal_chain",
            num_nodes=7,
            graph_diameter=6,
            graph_density=0.143,
            structural_complexity_index=8.6,
            prior_skill_id="source_hierarchy",
            initial_sample_loss=0.705,
            initial_sample_acc=0.52,
            steps_spent=35,
            final_loss=0.460,
            final_f1=0.78,
            learning_gain_f1=0.26,
            was_negative_transfer=False,
            recommended_action="TRAIN",
            recommended_budget_steps=35,
        ),
    ]

    if arm in (M61AblationArm.ARM_B_RELEVANT_HISTORY_UNGUARDED, M61AblationArm.ARM_F_SAFE_EXPERIENTIAL_PROGRESS):
        for ep in relevant_episodes:
            store.add_record(ep)
        return store

    if arm == M61AblationArm.ARM_C_RANDOM_UNRELATED_HISTORY:
        for i in range(4):
            rand_nodes = rng.randint(20, 50)
            rand_steps = rng.choice([5, 10, 45, 50])
            ep = LearningEpisodeRecord(
                episode_id=f"exp_random_{i}",
                task_id=f"synthetic_random_graph_{i}",
                num_nodes=rand_nodes,
                graph_diameter=rng.randint(15, 30),
                graph_density=rng.uniform(0.6, 0.9),
                structural_complexity_index=rng.uniform(25.0, 50.0),
                prior_skill_id="source_hierarchy",
                initial_sample_loss=rng.uniform(0.6, 1.2),
                initial_sample_acc=rng.uniform(0.4, 0.6),
                steps_spent=rand_steps,
                final_loss=rng.uniform(0.4, 0.9),
                final_f1=rng.uniform(0.5, 0.8),
                learning_gain_f1=rng.uniform(0.1, 0.3),
                was_negative_transfer=False,
                recommended_action=rng.choice(["KEEP", "TRAIN", "RESET"]),
                recommended_budget_steps=rand_steps,
            )
            store.add_record(ep)
        return store

    if arm == M61AblationArm.ARM_D_SHUFFLED_OUTCOMES_HISTORY:
        steps_pool = [10, 15, 25, 35, 40]
        rng.shuffle(steps_pool)
        for i, ep in enumerate(relevant_episodes):
            ep_shuff = copy.deepcopy(ep)
            ep_shuff.episode_id = f"exp_shuffled_{i}"
            ep_shuff.steps_spent = steps_pool[i % len(steps_pool)]
            ep_shuff.recommended_budget_steps = ep_shuff.steps_spent
            ep_shuff.learning_gain_f1 = rng.uniform(-0.15, 0.30)
            ep_shuff.was_negative_transfer = rng.choice([True, False])
            store.add_record(ep_shuff)
        return store

    if arm == M61AblationArm.ARM_E_INCOMPATIBLE_HISTORY:
        for i in range(3):
            ep_incomp = LearningEpisodeRecord(
                episode_id=f"exp_incomp_{i}",
                task_id=f"incompatible_parity_{i}",
                num_nodes=8,
                graph_diameter=4,
                graph_density=0.14,
                structural_complexity_index=6.9,
                prior_skill_id="source_hierarchy",
                initial_sample_loss=1.15,
                initial_sample_acc=0.35,
                steps_spent=40,
                final_loss=1.45,
                final_f1=0.28,
                learning_gain_f1=-0.07,
                was_negative_transfer=True,
                recommended_action="RESET",
                recommended_budget_steps=10,
            )
            store.add_record(ep_incomp)
        return store

    return store


class SafeCausalAdaptationEngine:
    """Executes safe, evidence-governed adaptation trajectories across ablation arms."""

    CLASS_LABELS = ["NO", "YES"]

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: Any,
        device: torch.device,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def _extract_hidden(self, text: str) -> torch.Tensor:
        toks = self.tokenizer.encode(text)
        inp = torch.tensor([toks], dtype=torch.long, device=self.device)
        b, t = inp.size()
        tok_emb = self.base_model.wte(inp)
        pos = torch.arange(0, t, dtype=torch.long, device=self.device)
        x = self.base_model.drop(tok_emb + self.base_model.wpe(pos))
        for block in self.base_model.blocks:
            x, _ = block(x)
        return self.base_model.ln_f(x)

    def _evaluate_split(
        self, adapter: torch.nn.Module, queries: list[Any]
    ) -> tuple[float, float]:
        adapter.eval()
        losses: list[float] = []
        correct = 0
        total = len(queries)
        if total == 0:
            return 0.0, 0.0

        with torch.no_grad():
            for q in queries:
                h = self._extract_hidden(q.question_text)
                logits = adapter(h)
                target_idx = 1 if q.expected_answer == "YES" else 0
                tgt = torch.tensor([target_idx], dtype=torch.long, device=self.device)
                loss = F.cross_entropy(logits, tgt)
                losses.append(float(loss.item()))
                pred = int(torch.argmax(logits, dim=-1).item())
                if pred == target_idx:
                    correct += 1

        mean_loss = float(np.mean(losses))
        acc = float(correct / total)
        return mean_loss, acc

    def _evaluate_paired_correctness(
        self, adapter: torch.nn.Module, env: AutonomousEnvironment
    ) -> float:
        adapter.eval()
        pairs = env.minimal_pairs
        if not pairs:
            return 0.0

        joint_correct = 0
        with torch.no_grad():
            for mp in pairs:
                h_base = self._extract_hidden(mp.base_query.question_text)
                h_pert = self._extract_hidden(mp.perturbed_query.question_text)
                l_base = adapter(h_base)
                l_pert = adapter(h_pert)

                p_base = self.CLASS_LABELS[int(torch.argmax(l_base, dim=-1).item())]
                p_pert = self.CLASS_LABELS[int(torch.argmax(l_pert, dim=-1).item())]

                if p_base == mp.base_query.expected_answer and p_pert == mp.perturbed_query.expected_answer:
                    joint_correct += 1

        return float(joint_correct / len(pairs))

    def _instantiate_target_adapter(
        self, source_adapter: ResidualSkillAdapter | None
    ) -> torch.nn.Module:
        d_model = getattr(self.base_model.config, "d_model", getattr(self.base_model.config, "n_embd", 128))
        if source_adapter is not None:
            return ResidualSkillAdapter(
                source_adapter=source_adapter,
                d_model=d_model,
                residual_bottleneck_dim=16,
                num_classes=2,
            ).to(self.device)
        return PersistentCandidateAdapter(
            d_model=d_model,
            bottleneck_dim=16,
            num_classes=2,
        ).to(self.device)

    def run_safe_adaptation(
        self,
        env: AutonomousEnvironment,
        arm: M61AblationArm,
        source_adapter: ResidualSkillAdapter | None = None,
        seed: int = 42,
    ) -> M61AdaptationResult:
        """Run single adaptation trajectory governed by the assigned ablation arm."""
        t_start = time.perf_counter()
        torch.manual_seed(seed)
        np.random.seed(seed)

        target_adapter = self._instantiate_target_adapter(source_adapter)
        optimizer = torch.optim.AdamW(
            target_adapter.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        train_queries = [mp.base_query for mp in env.minimal_pairs] + [mp.perturbed_query for mp in env.minimal_pairs]
        if not train_queries:
            train_queries = env.held_out_test_queries[:16]
        val_queries = env.held_out_test_queries

        # Setup Memory Store and Query Recommendation
        t_lookup_start = time.perf_counter()
        memory_store = build_m61_memory_store(arm, seed=seed)
        complexity = compute_observable_complexity(env)
        init_loss, init_f1 = self._evaluate_split(target_adapter, train_queries[:4])
        rec = memory_store.query_recommendation(
            complexity=complexity, sample_loss=init_loss, sample_acc=init_f1
        )
        lookup_overhead_ms = (time.perf_counter() - t_lookup_start) * 1000.0

        # Determine Hard Optimization Cap based on Arm
        if arm == M61AblationArm.ARM_G_FIXED_10_STEPS:
            hard_max_steps = 10
        elif arm == M61AblationArm.ARM_G_FIXED_20_STEPS:
            hard_max_steps = 20
        elif arm in (M61AblationArm.ARM_A_NO_HISTORY, M61AblationArm.ARM_G_FIXED_40_STEPS):
            hard_max_steps = 40
        elif arm in (
            M61AblationArm.ARM_B_RELEVANT_HISTORY_UNGUARDED,
            M61AblationArm.ARM_C_RANDOM_UNRELATED_HISTORY,
            M61AblationArm.ARM_D_SHUFFLED_OUTCOMES_HISTORY,
            M61AblationArm.ARM_E_INCOMPATIBLE_HISTORY,
        ):
            # Unguarded historical budgeting (M60 behavior: truncates immediately at advice)
            hard_max_steps = 15 if (rec.negative_transfer_warning or rec.recommended_action == "RESET") else rec.recommended_budget_steps
            hard_max_steps = max(10, min(40, hard_max_steps))
        elif arm in (M61AblationArm.ARM_F_SAFE_EXPERIENTIAL_PROGRESS, M61AblationArm.ARM_H_PROGRESS_COST_NO_MEMORY):
            # Dynamic block controller up to 40 steps max
            hard_max_steps = 40

        # Initialize Safe Progress Controller
        controller = SafeLearningProgressController(
            min_steps=20 if arm == M61AblationArm.ARM_F_SAFE_EXPERIENTIAL_PROGRESS else 10,
            max_steps=hard_max_steps,
            block_size=10,
        )

        full_init_loss, full_init_f1 = self._evaluate_split(target_adapter, val_queries)
        controller.record_initial_state(full_init_loss, full_init_f1)

        step_history: list[int] = [0]
        loss_history: list[float] = [full_init_loss]
        f1_history: list[float] = [full_init_f1]
        decisions: list[SafeControlDecision] = []

        total_steps = 0
        step = 0
        reset_triggered = False
        premature_stop_prevented = False

        while step < hard_max_steps:
            # Execute one 10-step optimization block (or remaining steps)
            block_steps = min(10, hard_max_steps - step)
            target_adapter.train()
            for _ in range(block_steps):
                optimizer.zero_grad()
                q = train_queries[step % len(train_queries)]
                h = self._extract_hidden(q.question_text)
                logits = target_adapter(h)
                target_idx = 1 if q.expected_answer == "YES" else 0
                tgt = torch.tensor([target_idx], dtype=torch.long, device=self.device)
                loss = F.cross_entropy(logits, tgt)
                loss.backward()
                optimizer.step()
                step += 1
                total_steps += 1

            cur_loss, cur_f1 = self._evaluate_split(target_adapter, val_queries)
            step_history.append(step)
            loss_history.append(cur_loss)
            f1_history.append(cur_f1)

            # Controller evaluation at block boundary for dynamic arms
            if arm in (M61AblationArm.ARM_F_SAFE_EXPERIENTIAL_PROGRESS, M61AblationArm.ARM_H_PROGRESS_COST_NO_MEMORY):
                hist_action = rec.recommended_action if arm == M61AblationArm.ARM_F_SAFE_EXPERIENTIAL_PROGRESS else "TRAIN"
                hist_steps = rec.recommended_budget_steps if arm == M61AblationArm.ARM_F_SAFE_EXPERIENTIAL_PROGRESS else 40
                dec = controller.evaluate_checkpoint(
                    current_step=step,
                    val_loss=cur_loss,
                    val_f1=cur_f1,
                    history_recommended_action=hist_action,
                    history_recommended_steps=hist_steps,
                )
                decisions.append(dec)

                if dec.history_vetoed_by_progress:
                    premature_stop_prevented = True

                if dec.action == ControllerAction.RESET and not reset_triggered:
                    reset_triggered = True
                    # Reinitialize to fresh adapter
                    target_adapter = self._instantiate_target_adapter(source_adapter=None)
                    optimizer = torch.optim.AdamW(
                        target_adapter.parameters(),
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay,
                    )
                elif dec.action == ControllerAction.STOP:
                    break

        # Compute AUC normalized over 40 steps
        norm_steps = [s / 40.0 for s in step_history]
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        if trapz_fn is not None:
            auc = float(trapz_fn(f1_history, norm_steps))
        else:
            auc = float(sum((norm_steps[i] - norm_steps[i - 1]) * (f1_history[i] + f1_history[i - 1]) / 2.0 for i in range(1, len(norm_steps))))

        final_loss = loss_history[-1]
        final_f1 = f1_history[-1]
        paired_corr = self._evaluate_paired_correctness(target_adapter, env)
        wall_clock_ms = (time.perf_counter() - t_start) * 1000.0

        # Compute-adjusted efficiency: AUC / (steps / 40)
        norm_cost = max(0.25, total_steps / 40.0)
        comp_eff = float(auc / norm_cost)

        return M61AdaptationResult(
            arm=arm,
            env_id=env.env_id,
            seed=seed,
            total_steps_taken=total_steps,
            final_loss=final_loss,
            final_f1=final_f1,
            paired_correctness=paired_corr,
            learning_curve_auc=auc,
            wall_clock_time_ms=wall_clock_ms,
            memory_lookup_overhead_ms=lookup_overhead_ms,
            compute_adjusted_efficiency=comp_eff,
            step_history=step_history,
            loss_history=loss_history,
            f1_history=f1_history,
            decisions=decisions,
            premature_stop_prevented=premature_stop_prevented,
            reset_triggered=reset_triggered,
        )
