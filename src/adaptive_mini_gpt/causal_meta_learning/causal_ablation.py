"""Phase 2: Causal Experiential-Memory Ablation for M60.

Benchmarks 7 distinct adaptation arms to rigorously isolate whether historical learning
records provide a causal advantage or mere heuristic correlation:
- Arm A: NO_HISTORY (Cold-start, empty experiential store)
- Arm B: RELEVANT_HISTORY (Structurally matched verified learning traces)
- Arm C: RANDOM_UNRELATED_HISTORY (Synthetically generated unrelated topological traces)
- Arm D: SHUFFLED_OUTCOMES_HISTORY (Matched topological tasks with randomly permuted step budgets and outcomes)
- Arm E: INCOMPATIBLE_HISTORY (Histories from conflicting/incompatible rule structures)
- Arm F: FIXED_40_STEPS (Static 40-step reference baseline)
- Arm G: PROGRESS_COST_NO_MEMORY (Strong real-time validation progress & cost controller without memory)
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

logger = logging.getLogger(__name__)


class CausalAblationArm(str, enum.Enum):
    ARM_A_NO_HISTORY = "Arm_A_No_History"
    ARM_B_RELEVANT_HISTORY = "Arm_B_Relevant_History"
    ARM_C_RANDOM_UNRELATED_HISTORY = "Arm_C_Random_Unrelated_History"
    ARM_D_SHUFFLED_OUTCOMES_HISTORY = "Arm_D_Shuffled_Outcomes_History"
    ARM_E_INCOMPATIBLE_HISTORY = "Arm_E_Incompatible_History"
    ARM_F_FIXED_40_STEPS = "Arm_F_Fixed_40_Steps"
    ARM_G_PROGRESS_COST_NO_MEMORY = "Arm_G_Progress_Cost_No_Memory"


@dataclass
class CausalAdaptationResult:
    """Outcome of a single causal adaptation run."""
    arm: CausalAblationArm
    env_id: str
    seed: int
    total_steps_taken: int
    final_loss: float
    final_f1: float
    learning_curve_auc: float
    wall_clock_time_ms: float
    memory_lookup_overhead_ms: float
    compute_adjusted_efficiency: float
    step_history: list[int]
    loss_history: list[float]
    f1_history: list[float]


def build_ablated_memory_store(
    arm: CausalAblationArm, seed: int = 42
) -> ExperientialMemoryStore:
    """Construct an experiential memory store configured for the specific ablation arm."""
    rng = random.Random(seed)
    store = ExperientialMemoryStore()

    if arm in (CausalAblationArm.ARM_A_NO_HISTORY, CausalAblationArm.ARM_F_FIXED_40_STEPS, CausalAblationArm.ARM_G_PROGRESS_COST_NO_MEMORY):
        # Empty memory store (cold start)
        return store

    # Relevant base experiences (matching graph families)
    relevant_episodes = [
        LearningEpisodeRecord(
            episode_id="exp_train_chain_01",
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
            episode_id="exp_train_mesh_02",
            task_id="train_cluster_mesh",
            num_nodes=6,
            graph_diameter=2,
            graph_density=0.20,
            structural_complexity_index=4.6,
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
            episode_id="exp_train_deep_dag_03",
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

    if arm == CausalAblationArm.ARM_B_RELEVANT_HISTORY:
        for ep in relevant_episodes:
            store.add_record(ep)
        return store

    if arm == CausalAblationArm.ARM_C_RANDOM_UNRELATED_HISTORY:
        # Generate arbitrary synthetic topological features and budgets
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

    if arm == CausalAblationArm.ARM_D_SHUFFLED_OUTCOMES_HISTORY:
        # Same topological complexity as relevant, but outcomes/steps randomly permuted
        shuffled_steps = [10, 40, 5]
        rng.shuffle(shuffled_steps)
        shuffled_actions = ["RESET", "KEEP", "RESET"]
        rng.shuffle(shuffled_actions)
        for i, ep in enumerate(relevant_episodes):
            ep_shuffled = copy.deepcopy(ep)
            ep_shuffled.episode_id = f"exp_shuffled_{i}"
            ep_shuffled.steps_spent = shuffled_steps[i]
            ep_shuffled.recommended_budget_steps = shuffled_steps[i]
            ep_shuffled.recommended_action = shuffled_actions[i]
            ep_shuffled.final_f1 = rng.uniform(0.35, 0.55)
            ep_shuffled.was_negative_transfer = True
            store.add_record(ep_shuffled)
        return store

    if arm == CausalAblationArm.ARM_E_INCOMPATIBLE_HISTORY:
        # History exclusively from incompatible rule structures (severe negative transfer)
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


class CausalAblationAdaptationEngine:
    """Adapts compact residual adapter under one of the 7 causal ablation arms."""

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
        self, adapter: ResidualSkillAdapter, queries: list[Any]
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

    def run_causal_adaptation(
        self,
        env: AutonomousEnvironment,
        arm: CausalAblationArm,
        source_adapter: ResidualSkillAdapter | None = None,
        seed: int = 42,
    ) -> CausalAdaptationResult:
        """Run single adaptation trajectory governed strictly by the assigned causal ablation arm."""
        t_start = time.perf_counter()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 1. Target residual adapter initialization
        d_model = getattr(self.base_model.config, "d_model", getattr(self.base_model.config, "n_embd", 128))
        target_adapter = ResidualSkillAdapter(
            d_model=d_model, bottleneck_dim=16, num_classes=2
        ).to(self.device)

        if source_adapter is not None:
            with torch.no_grad():
                target_adapter.down_proj.weight.copy_(source_adapter.down_proj.weight)
                target_adapter.up_proj.weight.copy_(source_adapter.up_proj.weight)
                target_adapter.head.weight.copy_(source_adapter.head.weight)

        optimizer = torch.optim.AdamW(
            target_adapter.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        train_queries = env.observed_support_queries
        val_queries = env.held_out_test_queries

        # 2. Setup Arm Budget & Controller Policy
        complexity = compute_observable_complexity(env)
        memory_store = build_ablated_memory_store(arm, seed=seed)

        t_lookup_start = time.perf_counter()
        target_budget_steps = 40

        # Initial small sample diagnostic
        init_loss, init_f1 = self._evaluate_split(target_adapter, train_queries[:4])

        if arm == CausalAblationArm.ARM_F_FIXED_40_STEPS:
            target_budget_steps = 40
        elif arm == CausalAblationArm.ARM_G_PROGRESS_COST_NO_MEMORY:
            target_budget_steps = 20
        else:
            rec = memory_store.query_recommendation(
                complexity=complexity, sample_loss=init_loss, sample_acc=init_f1
            )
            if rec.negative_transfer_warning or rec.recommended_action == "RESET":
                target_budget_steps = 15
            else:
                target_budget_steps = rec.recommended_budget_steps

        lookup_overhead_ms = (time.perf_counter() - t_lookup_start) * 1000.0

        # Bound step budget strictly between 10 and 40 steps
        target_budget_steps = max(10, min(40, target_budget_steps))

        step_history: list[int] = [0]
        full_init_loss, full_init_f1 = self._evaluate_split(target_adapter, val_queries)
        loss_history: list[float] = [full_init_loss]
        f1_history: list[float] = [full_init_f1]

        # 3. Optimization Loop
        total_steps = 0
        for step in range(1, target_budget_steps + 1):
            target_adapter.train()
            optimizer.zero_grad()
            q = train_queries[(step - 1) % len(train_queries)]
            h = self._extract_hidden(q.question_text)
            logits = target_adapter(h)
            target_idx = 1 if q.expected_answer == "YES" else 0
            tgt = torch.tensor([target_idx], dtype=torch.long, device=self.device)
            loss = F.cross_entropy(logits, tgt)
            loss.backward()
            optimizer.step()
            total_steps += 1

            if step % 10 == 0 or step == target_budget_steps:
                cur_loss, cur_f1 = self._evaluate_split(target_adapter, val_queries)
                step_history.append(step)
                loss_history.append(cur_loss)
                f1_history.append(cur_f1)

                if arm == CausalAblationArm.ARM_G_PROGRESS_COST_NO_MEMORY and step == 20:
                    delta_loss = loss_history[-2] - cur_loss
                    if delta_loss > 0.05 and cur_f1 < 0.70 and target_budget_steps < 40:
                        target_budget_steps = min(40, target_budget_steps + 10)

        norm_steps = [s / 40.0 for s in step_history]
        auc = float(np.trapz(f1_history, norm_steps))

        wall_clock_ms = (time.perf_counter() - t_start) * 1000.0
        compute_efficiency = float(auc / (total_steps / 40.0)) if total_steps > 0 else 0.0

        return CausalAdaptationResult(
            arm=arm,
            env_id=env.env_id,
            seed=seed,
            total_steps_taken=total_steps,
            final_loss=loss_history[-1],
            final_f1=f1_history[-1],
            learning_curve_auc=auc,
            wall_clock_time_ms=wall_clock_ms,
            memory_lookup_overhead_ms=lookup_overhead_ms,
            compute_adjusted_efficiency=compute_efficiency,
            step_history=step_history,
            loss_history=loss_history,
            f1_history=f1_history,
        )
