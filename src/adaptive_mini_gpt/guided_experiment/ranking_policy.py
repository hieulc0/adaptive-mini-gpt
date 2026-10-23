"""Phase 3: Neural Experiment Ranking Head, Exploration Policies, and Ablation Arms."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim

from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment
from adaptive_mini_gpt.guided_experiment.hypothesis_tracker import GuidedHypothesisTracker
from adaptive_mini_gpt.guided_experiment.structural_probe_generator import (
    StructuralProbe,
    StructuralProbeGenerator,
)


class NeuralProbeRankingHead(nn.Module):
    """Compact neural head predicting expected probe utility from structural and context features."""

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
        """Forward pass outputting scalar utility score."""
        return self.net(features).squeeze(-1)

    @staticmethod
    def extract_features(
        probe: StructuralProbe,
        current_entropy: float,
        step: int,
        max_steps: int,
        tracker: GuidedHypothesisTracker,
    ) -> list[float]:
        """Construct normalized feature vector for candidate probe."""
        f1, f2, f3, f4 = probe.structural_features
        f5 = min(1.0, current_entropy / 3.0)
        f6 = float(step) / max(1.0, float(max_steps))
        top_h, top_prob = tracker.get_top_hypothesis()
        f7 = float(top_prob)
        f8 = float(tracker.is_unknown_rule_detected())
        return [f1, f2, f3, f4, f5, f6, f7, f8]


def train_neural_ranking_head(
    ranking_head: NeuralProbeRankingHead,
    training_environments: list[AutonomousEnvironment],
    num_epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 42,
) -> float:
    """Supervise ranking head on candidate probes using research oracle EIG across training worlds."""
    torch.manual_seed(seed)
    device = next(ranking_head.parameters()).device
    optimizer = optim.Adam(ranking_head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    ranking_head.train()
    total_loss = 0.0
    num_batches = 0

    for epoch in range(num_epochs):
        for env in training_environments:
            tracker = GuidedHypothesisTracker(base_edges=env.base_edges)
            generator = StructuralProbeGenerator(env.entities, env.base_edges, env.relation_name)
            candidates, _ = generator.generate_candidate_probes(max_probes=20)
            if not candidates:
                continue

            feats = []
            targets = []
            for probe in candidates:
                eig = tracker.compute_expected_information_gain(probe)
                feat = NeuralProbeRankingHead.extract_features(
                    probe, tracker.compute_entropy(), step=0, max_steps=6, tracker=tracker
                )
                feats.append(feat)
                targets.append(eig)

            x = torch.tensor(feats, dtype=torch.float32, device=device)
            y = torch.tensor(targets, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            preds = ranking_head(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

    ranking_head.eval()
    return total_loss / max(1, num_batches)


@dataclass
class ArmExecutionResult:
    """Outcome of running an exploration arm in an environment."""
    arm_name: str
    env_id: str
    total_information_gain: float
    mean_information_gain_per_query: float
    queries_executed: list[dict[str, Any]]
    final_entropy: float
    unknown_rule_detected: bool
    top_hypothesis: str
    top_hypothesis_prob: float
    validity_rate: float = 1.00


class GuidedArmsRunner:
    """Executes Arms A, B, C, D, and Filter Ablation in an environment."""

    def __init__(
        self,
        environment: AutonomousEnvironment,
        ranking_head: NeuralProbeRankingHead,
        query_budget: int = 6,
        seed: int = 42,
    ) -> None:
        self.environment = environment
        self.ranking_head = ranking_head
        self.query_budget = query_budget
        self.seed = seed
        self.rng = random.Random(seed)

    def run_arm(self, arm_name: str) -> ArmExecutionResult:
        """Run specified arm: 'ARM_A_RANDOM', 'ARM_B_INFOGAIN', 'ARM_C_NEURAL', 'ARM_D_NEURAL_FILTERED', or 'ARM_ABLATION_FILTER_ONLY'."""
        device = next(self.ranking_head.parameters()).device
        tracker = GuidedHypothesisTracker(
            base_edges=self.environment.base_edges,
            entity_levels=self.environment.entity_levels,
            blocked_nodes=self.environment.blocked_nodes,
            negated_edges=self.environment.negated_edges,
        )
        generator = StructuralProbeGenerator(
            self.environment.entities,
            self.environment.base_edges,
            self.environment.relation_name,
        )
        candidate_pool, _ = generator.generate_candidate_probes(max_probes=32)

        asked_pairs: set[tuple[str, str]] = set()
        executed_records: list[dict[str, Any]] = []
        total_ig = 0.0

        for step in range(self.query_budget):
            remaining_candidates = [
                p for p in candidate_pool if (p.source, p.target) not in asked_pairs
            ]
            if not remaining_candidates:
                break

            chosen_probe: StructuralProbe | None = None

            if arm_name == "ARM_A_RANDOM":
                chosen_probe = self.rng.choice(remaining_candidates)

            elif arm_name == "ARM_B_INFOGAIN":
                # Deterministic argmax EIG
                best_eig = -1.0
                best_p = remaining_candidates[0]
                for p in remaining_candidates:
                    eig = tracker.compute_expected_information_gain(p)
                    if eig > best_eig:
                        best_eig = eig
                        best_p = p
                chosen_probe = best_p

            elif arm_name == "ARM_C_NEURAL":
                # Neural ranking argmax with NO deterministic filtering
                self.ranking_head.eval()
                with torch.no_grad():
                    feats = [
                        NeuralProbeRankingHead.extract_features(
                            p, tracker.compute_entropy(), step, self.query_budget, tracker
                        )
                        for p in remaining_candidates
                    ]
                    x = torch.tensor(feats, dtype=torch.float32, device=device)
                    scores = self.ranking_head(x).tolist()
                    if isinstance(scores, float):
                        scores = [scores]

                best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
                chosen_probe = remaining_candidates[best_idx]

            elif arm_name == "ARM_D_NEURAL_FILTERED":
                # Neural proposal with deterministic validity and non-zero EIG filter gate
                self.ranking_head.eval()
                with torch.no_grad():
                    feats = [
                        NeuralProbeRankingHead.extract_features(
                            p, tracker.compute_entropy(), step, self.query_budget, tracker
                        )
                        for p in remaining_candidates
                    ]
                    x = torch.tensor(feats, dtype=torch.float32, device=device)
                    scores = self.ranking_head(x).tolist()
                    if isinstance(scores, float):
                        scores = [scores]

                ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                top_candidate = remaining_candidates[ranked_indices[0]]

                # Deterministic filter check
                cand_eig = tracker.compute_expected_information_gain(top_candidate)
                if cand_eig > 1e-4:
                    chosen_probe = top_candidate
                else:
                    # Filter rejects uninformative candidate: fall back to probe with maximum EIG
                    best_eig = -1.0
                    fallback_p = top_candidate
                    for p in remaining_candidates:
                        eig = tracker.compute_expected_information_gain(p)
                        if eig > best_eig:
                            best_eig = eig
                            fallback_p = p
                    chosen_probe = fallback_p

            elif arm_name == "ARM_ABLATION_FILTER_ONLY":
                # Filter alone without neural ranking scores: select uniformly at random among probes passing EIG > 0
                informative_probes = [
                    p for p in remaining_candidates
                    if tracker.compute_expected_information_gain(p) > 1e-4
                ]
                if informative_probes:
                    chosen_probe = self.rng.choice(informative_probes)
                else:
                    chosen_probe = self.rng.choice(remaining_candidates)

            else:
                raise ValueError(f"Unknown arm name: {arm_name}")

            asked_pairs.add((chosen_probe.source, chosen_probe.target))
            answer = self.environment.oracle_query(chosen_probe.source, chosen_probe.target)
            step_ig = tracker.update(chosen_probe.source, chosen_probe.target, answer)
            total_ig += step_ig

            executed_records.append({
                "step": step + 1,
                "probe_id": chosen_probe.probe_id,
                "source": chosen_probe.source,
                "target": chosen_probe.target,
                "probe_type": chosen_probe.probe_type,
                "answer": answer,
                "information_gain": step_ig,
                "current_entropy": tracker.compute_entropy(),
            })

        final_ent = tracker.compute_entropy()
        top_h, top_p = tracker.get_top_hypothesis()
        mean_ig = total_ig / max(1, len(executed_records))

        return ArmExecutionResult(
            arm_name=arm_name,
            env_id=self.environment.env_id,
            total_information_gain=float(total_ig),
            mean_information_gain_per_query=float(mean_ig),
            queries_executed=executed_records,
            final_entropy=float(final_ent),
            unknown_rule_detected=tracker.is_unknown_rule_detected(),
            top_hypothesis=top_h,
            top_hypothesis_prob=float(top_p),
            validity_rate=1.00,
        )
