"""Phase 4: Minimal Learned Policy (REINFORCE / Contextual Bandit) and 5-Arm Exploration Suite."""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment
from adaptive_mini_gpt.guided_experiment.ranking_policy import ArmExecutionResult, NeuralProbeRankingHead
from adaptive_mini_gpt.guided_experiment.structural_probe_generator import (
    StructuralProbe,
    StructuralProbeGenerator,
)
from adaptive_mini_gpt.learning_driven_experiment.epistemic_tracker import NormalizedEpistemicTracker
from adaptive_mini_gpt.learning_driven_experiment.evaluator import LearningOutcomeEvaluator


class LearningOutcomePolicyHead(nn.Module):
    """Minimal policy head predicting action preference logits for structural candidate probes."""

    def __init__(self, in_features: int = 8, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute scalar preference logit for each candidate probe."""
        return self.net(features).squeeze(-1)

    @staticmethod
    def extract_features(
        probe: StructuralProbe,
        current_entropy: float,
        step: int,
        max_steps: int,
        tracker: NormalizedEpistemicTracker,
    ) -> list[float]:
        """Construct normalized feature vector for candidate probe."""
        f1, f2, f3, f4 = probe.structural_features
        f5 = min(1.0, current_entropy / 3.1223)
        f6 = float(step) / max(1.0, float(max_steps))
        top_h, top_prob = tracker.get_top_hypothesis()
        f7 = float(top_prob)
        f8 = float(tracker.is_unknown_rule_detected())
        return [f1, f2, f3, f4, f5, f6, f7, f8]


def train_learning_outcome_policy(
    policy_head: LearningOutcomePolicyHead,
    evaluator: LearningOutcomeEvaluator,
    train_environments: list[AutonomousEnvironment],
    dev_environments: list[AutonomousEnvironment],
    num_episodes: int = 20,
    steps_per_episode: int = 4,
    lr: float = 2e-3,
    seed: int = 42,
) -> float:
    """Train minimal policy via REINFORCE where reward is downstream neural adaptation gain on dev environments."""
    torch.manual_seed(seed)
    random.seed(seed)
    device = next(policy_head.parameters()).device
    optimizer = optim.Adam(policy_head.parameters(), lr=lr, weight_decay=1e-4)

    policy_head.train()
    running_baseline = 0.0
    total_policy_loss = 0.0
    num_updates = 0

    for ep in range(num_episodes):
        env = random.choice(train_environments)
        tracker = NormalizedEpistemicTracker(base_edges=env.base_edges)
        generator = StructuralProbeGenerator(env.entities, env.base_edges, env.relation_name)
        executed_obs: list[tuple[str, str, str]] = []

        episode_log_probs: list[torch.Tensor] = []
        episode_rewards: list[float] = []

        for step in range(steps_per_episode):
            candidates, stats = generator.generate_candidate_probes(max_probes=16)
            # Filter already executed
            available = [p for p in candidates if (p.source, p.target) not in [(o[0], o[1]) for o in executed_obs]]
            if not available:
                break

            feats = [
                LearningOutcomePolicyHead.extract_features(
                    p, tracker.compute_entropy(), step, steps_per_episode, tracker
                )
                for p in available
            ]
            x_t = torch.tensor(feats, dtype=torch.float32, device=device)
            logits = policy_head(x_t)
            probs = torch.softmax(logits, dim=-1)

            # Sample action via Categorical distribution
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)

            chosen_probe = available[action_idx.item()]
            ans = env.oracle_evaluate_query(chosen_probe.source, chosen_probe.target)
            cand_obs = (chosen_probe.source, chosen_probe.target, ans)

            # Measure learning benefit reward on DEV environments
            reward_res = evaluator.evaluate_fixed_budget_dev_improvement(
                base_observations=executed_obs,
                candidate_observation=cand_obs,
                dev_envs=dev_environments,
                fixed_budget=8,
            )

            executed_obs.append(cand_obs)
            tracker.update(chosen_probe.source, chosen_probe.target, ans)

            episode_log_probs.append(log_prob)
            episode_rewards.append(reward_res.total_reward)

        # REINFORCE policy gradient update with baseline subtraction
        if episode_log_probs:
            returns = []
            g = 0.0
            for r in reversed(episode_rewards):
                g = r + 0.95 * g
                returns.insert(0, g)

            loss = torch.tensor(0.0, device=device)
            for lp, ret in zip(episode_log_probs, returns):
                advantage = ret - running_baseline
                loss = loss - lp * advantage
                running_baseline = 0.90 * running_baseline + 0.10 * ret

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += float(loss.item())
            num_updates += 1

    return float(total_policy_loss / max(1, num_updates))


class LearningDrivenArmsRunner:
    """Executes matched query budget Q across 5 exploration arms."""

    def __init__(
        self,
        env: AutonomousEnvironment,
        m52_policy: LearningOutcomePolicyHead,
        m51_ranking_head: NeuralProbeRankingHead | None = None,
        query_budget: int = 6,
        seed: int = 42,
    ) -> None:
        self.env = env
        self.m52_policy = m52_policy
        self.m51_ranking_head = m51_ranking_head
        self.query_budget = query_budget
        self.seed = seed

    def run_arm(self, arm_name: str) -> ArmExecutionResult:
        """Run exploration arm up to query budget on environment."""
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        device = next(self.m52_policy.parameters()).device

        env_copy = copy.deepcopy(self.env)
        env_copy.reset_oracle_budget()

        tracker = NormalizedEpistemicTracker(
            base_edges=env_copy.base_edges,
            entity_levels=env_copy.entity_levels,
            blocked_nodes=env_copy.blocked_nodes,
            negated_edges=env_copy.negated_edges,
        )
        generator = StructuralProbeGenerator(env_copy.entities, env_copy.base_edges, env_copy.relation_name)

        executed_queries: list[tuple[str, str, str]] = []
        entropy_history: list[float] = [tracker.compute_entropy()]
        cumulative_ig = 0.0

        t0 = time.perf_counter()

        for step in range(self.query_budget):
            candidates, _ = generator.generate_candidate_probes(max_probes=20)
            executed_set = {(u, v) for u, v, _ in executed_queries}
            available = [p for p in candidates if (p.source, p.target) not in executed_set]
            if not available:
                break

            chosen_probe: StructuralProbe

            if arm_name == "ARM_A_RANDOM":
                chosen_probe = random.choice(available)

            elif arm_name == "ARM_B_INFOGAIN":
                best_eig = -1.0
                best_p = available[0]
                for p in available:
                    eig = tracker.compute_expected_information_gain(p)
                    if eig > best_eig:
                        best_eig = eig
                        best_p = p
                chosen_probe = best_p

            elif arm_name == "ARM_C_M51_NEURAL":
                if self.m51_ranking_head is None:
                    chosen_probe = random.choice(available)
                else:
                    self.m51_ranking_head.eval()
                    feats = [
                        NeuralProbeRankingHead.extract_features(
                            p, tracker.compute_entropy(), step, self.query_budget, tracker  # type: ignore
                        )
                        for p in available
                    ]
                    with torch.no_grad():
                        scores = self.m51_ranking_head(torch.tensor(feats, dtype=torch.float32, device=device))
                        best_idx = int(torch.argmax(scores).item())
                    chosen_probe = available[best_idx]

            elif arm_name == "ARM_D_M52_LEARNING_OUTCOME":
                self.m52_policy.eval()
                feats = [
                    LearningOutcomePolicyHead.extract_features(
                        p, tracker.compute_entropy(), step, self.query_budget, tracker
                    )
                    for p in available
                ]
                with torch.no_grad():
                    logits = self.m52_policy(torch.tensor(feats, dtype=torch.float32, device=device))
                    best_idx = int(torch.argmax(logits).item())
                chosen_probe = available[best_idx]

            elif arm_name == "ARM_E_POLICY_ABLATION":
                # Identical policy architecture with shuffled scores (control ablation)
                self.m52_policy.eval()
                feats = [
                    LearningOutcomePolicyHead.extract_features(
                        p, tracker.compute_entropy(), step, self.query_budget, tracker
                    )
                    for p in available
                ]
                with torch.no_grad():
                    logits = self.m52_policy(torch.tensor(feats, dtype=torch.float32, device=device))
                    # Deterministically shuffle indices to ablate learned ordering
                    shuffled_indices = list(range(len(available)))
                    random.Random(self.seed + step).shuffle(shuffled_indices)
                    chosen_probe = available[shuffled_indices[0]]

            else:
                raise ValueError(f"Unknown exploration arm: {arm_name}")

            # Oracle interaction
            ans = env_copy.oracle_evaluate_query(chosen_probe.source, chosen_probe.target)
            step_ig = tracker.update(chosen_probe.source, chosen_probe.target, ans)
            cumulative_ig += step_ig
            executed_queries.append((chosen_probe.source, chosen_probe.target, ans))
            entropy_history.append(tracker.compute_entropy())

        duration = time.perf_counter() - t0
        top_h, top_prob = tracker.get_top_hypothesis()

        return ArmExecutionResult(
            arm_name=arm_name,
            env_id=self.env.env_id,
            queries_executed=executed_queries,
            total_information_gain=float(cumulative_ig),
            final_entropy=float(tracker.compute_entropy()),
            entropy_history=entropy_history,
            top_identified_hypothesis=top_h,
            top_hypothesis_probability=float(top_prob),
            is_unknown_detected=tracker.is_unknown_rule_detected(),
            validity_rate=1.00,  # Valid by construction
            duration_seconds=float(duration),
        )
