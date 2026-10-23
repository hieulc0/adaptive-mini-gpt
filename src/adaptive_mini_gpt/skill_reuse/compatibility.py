"""Phase 3: Evidence-Based Skill Compatibility Evaluator and Neural Selector."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment, EnvironmentQuery
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.skill_reuse.registry import SkillRepertoireRegistry

logger = logging.getLogger(__name__)


@dataclass
class TargetObservation:
    """A single verified question-answer observation from an unfamiliar target environment."""
    query_id: str
    question_text: str
    observed_answer: str
    token_ids: list[int]


@dataclass
class SkillCompatibilityScore:
    """Quantitative compatibility assessment for a single candidate skill."""
    skill_id: str
    log_likelihood: float
    accuracy_on_evidence: float
    predicted_neural_score: float


@dataclass
class SelectionDecision:
    """Outcome of skill selection for an unfamiliar target task."""
    arm_name: str  # "fresh", "oracle", "deterministic_evidence", "neural_evidence", "incompatible_control"
    selected_skill_id: str  # "fresh" or registered skill ID
    scores_by_skill: dict[str, SkillCompatibilityScore]
    selection_overhead_ms: float
    selection_rationale: str


class NeuralSkillSelectorHead(nn.Module):
    """Compact neural compatibility head predicting adapter applicability from pooled hidden states."""

    def __init__(self, d_model: int = 128, num_skills: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_skills = num_skills
        self.skill_embeddings = nn.Embedding(num_skills, hidden_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled_obs_hidden: torch.Tensor, skill_indices: torch.Tensor) -> torch.Tensor:
        """Score compatibility between pooled target observations and candidate skills.
        
        Args:
            pooled_obs_hidden: (B, d_model) representation of target evidence.
            skill_indices: (K,) or (B, K) tensor of skill category IDs.
            
        Returns:
            (B, K) compatibility logits.
        """
        b = pooled_obs_hidden.size(0)
        h_obs = self.obs_encoder(pooled_obs_hidden)  # (B, hidden_dim)
        e_skills = self.skill_embeddings(skill_indices)  # (K, hidden_dim)
        if e_skills.dim() == 2:
            e_skills = e_skills.unsqueeze(0).expand(b, -1, -1)  # (B, K, hidden_dim)
        k = e_skills.size(1)
        h_obs_expanded = h_obs.unsqueeze(1).expand(-1, k, -1)  # (B, K, hidden_dim)
        combined = torch.cat([h_obs_expanded, e_skills], dim=-1)  # (B, K, 2*hidden_dim)
        scores = self.scorer(combined).squeeze(-1)  # (B, K)
        return scores


class SkillCompatibilityEvaluator:
    """Evaluates candidate skill compatibility on target environments without revealing hidden rule names."""

    CLASS_LABELS = ["YES", "NO", "UNKNOWN"]
    LABEL_TO_IDX = {"YES": 0, "NO": 1, "UNKNOWN": 2}

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: Any,
        registry: SkillRepertoireRegistry,
        neural_selector: NeuralSkillSelectorHead | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.registry = registry
        self.device = device or torch.device("cpu")
        self.neural_selector = neural_selector
        if self.neural_selector is not None:
            self.neural_selector.to(self.device)

    def _extract_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract final layer hidden representation from base model."""
        b, t = input_ids.size()
        tok_emb = self.base_model.wte(input_ids)
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device)
        pos_emb = self.base_model.wpe(pos)
        x = self.base_model.drop(tok_emb + pos_emb)
        for block in self.base_model.blocks:
            x, _ = block(x)
        return self.base_model.ln_f(x)

    def sample_verified_observations(
        self,
        env: AutonomousEnvironment,
        budget: int = 6,
        seed: int = 42,
    ) -> list[TargetObservation]:
        """Sample a small budget of verified observations without disclosing rule names."""
        rng = np.random.RandomState(seed)
        entities = env.entities
        pairs = [(u, v) for u in entities for v in entities if u != v]
        rng.shuffle(pairs)

        observations: list[TargetObservation] = []
        counts = {"YES": 0, "NO": 0, "UNKNOWN": 0}
        target_per_class = max(1, budget // 3)

        for u, v in pairs:
            ans = env.evaluate_ground_truth(u, v)
            if counts[ans] < target_per_class or len(observations) < budget:
                q_text = f"Does {u} {env.domain_name} {v}?"
                toks = self.tokenizer.encode(q_text)
                obs = TargetObservation(
                    query_id=f"obs_{u}_{v}",
                    question_text=q_text,
                    observed_answer=ans,
                    token_ids=toks,
                )
                observations.append(obs)
                counts[ans] += 1
                if len(observations) >= budget:
                    break
        return observations

    def evaluate_deterministic_compatibility(
        self,
        observations: list[TargetObservation],
        fallback_threshold: float = -1.5,
    ) -> dict[str, SkillCompatibilityScore]:
        """Compute log-likelihood and empirical accuracy on evidence for each candidate skill."""
        scores: dict[str, SkillCompatibilityScore] = {}
        self.base_model.eval()

        for sid in self.registry.list_skills():
            adapter = self.registry.get_adapter(sid)
            adapter.to(self.device)
            adapter.eval()

            log_lik_sum = 0.0
            correct = 0

            with torch.no_grad():
                for obs in observations:
                    t_tensor = torch.tensor([obs.token_ids], dtype=torch.long, device=self.device)
                    h = self._extract_hidden(t_tensor)
                    logits = adapter(h)
                    probs = F.softmax(logits, dim=-1)[0]
                    tgt_idx = self.LABEL_TO_IDX[obs.observed_answer]
                    p = max(float(probs[tgt_idx].item()), 1e-7)
                    log_lik_sum += np.log(p)

                    pred_idx = int(torch.argmax(logits, dim=-1).item())
                    if pred_idx == tgt_idx:
                        correct += 1

            n = max(1, len(observations))
            mean_ll = log_lik_sum / n
            acc = correct / n
            scores[sid] = SkillCompatibilityScore(
                skill_id=sid,
                log_likelihood=float(mean_ll),
                accuracy_on_evidence=float(acc),
                predicted_neural_score=0.0,
            )
        return scores

    def evaluate_neural_compatibility(
        self,
        observations: list[TargetObservation],
    ) -> dict[str, float]:
        """Predict compatibility using NeuralSkillSelectorHead."""
        if self.neural_selector is None:
            return {sid: 0.0 for sid in self.registry.list_skills()}

        self.base_model.eval()
        self.neural_selector.eval()

        # Pool hidden states across observations
        h_list = []
        with torch.no_grad():
            for obs in observations:
                t_tensor = torch.tensor([obs.token_ids], dtype=torch.long, device=self.device)
                h = self._extract_hidden(t_tensor)
                h_mean = h.mean(dim=1)  # (1, d_model)
                h_list.append(h_mean)

            h_obs_pooled = torch.stack(h_list).mean(dim=0)  # (1, d_model)
            skills = self.registry.list_skills()
            skill_indices = torch.arange(len(skills), dtype=torch.long, device=self.device)
            logits = self.neural_selector(h_obs_pooled, skill_indices)[0]  # (K,)
            return {skills[i]: float(logits[i].item()) for i in range(len(skills))}

    def select_skill(
        self,
        arm_name: str,
        env: AutonomousEnvironment,
        oracle_skill_id: str,
        incompatible_skill_id: str,
        budget: int = 6,
        seed: int = 42,
    ) -> SelectionDecision:
        """Select a candidate skill adapter according to the specified experimental arm."""
        start_t = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        end_t = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        if start_t:
            start_t.record()
        t0 = np.datetime64("now")

        observations = self.sample_verified_observations(env, budget=budget, seed=seed)
        det_scores = self.evaluate_deterministic_compatibility(observations)
        neural_scores = self.evaluate_neural_compatibility(observations)

        # Merge scores
        full_scores: dict[str, SkillCompatibilityScore] = {}
        for sid, score in det_scores.items():
            score.predicted_neural_score = neural_scores.get(sid, 0.0)
            full_scores[sid] = score

        selected_id = "fresh"
        rationale = ""

        if arm_name == "fresh":
            selected_id = "fresh"
            rationale = "Baseline arm: initialize fresh candidate adapter with random normal weights."
        elif arm_name == "oracle":
            selected_id = oracle_skill_id
            rationale = f"Oracle upper-bound arm: selected ground-truth compatible skill '{oracle_skill_id}'."
        elif arm_name == "deterministic_evidence":
            # Select skill with highest evidence log-likelihood
            best_sid = max(full_scores.keys(), key=lambda s: full_scores[s].log_likelihood)
            if full_scores[best_sid].log_likelihood < -1.8:
                selected_id = "fresh"
                rationale = f"Deterministic fallback to fresh (max LL {full_scores[best_sid].log_likelihood:.3f} < threshold)."
            else:
                selected_id = best_sid
                rationale = (
                    f"Selected '{best_sid}' based on evidence log-likelihood {full_scores[best_sid].log_likelihood:.3f} "
                    f"and evidence accuracy {full_scores[best_sid].accuracy_on_evidence * 100:.1f}% on {len(observations)} observations."
                )
        elif arm_name == "neural_evidence":
            # Select skill with highest neural predicted score
            best_sid = max(full_scores.keys(), key=lambda s: full_scores[s].predicted_neural_score)
            selected_id = best_sid
            rationale = (
                f"Selected '{best_sid}' based on neural compatibility head logit {full_scores[best_sid].predicted_neural_score:.3f} "
                f"derived from {len(observations)} target evidence embeddings."
            )
        elif arm_name == "incompatible_control":
            selected_id = incompatible_skill_id
            rationale = f"Negative-transfer control arm: deliberately selected incompatible skill '{incompatible_skill_id}'."
        else:
            raise ValueError(f"Unknown arm name: {arm_name}")

        overhead_ms = 0.0
        if start_t and end_t:
            end_t.record()
            torch.cuda.synchronize()
            overhead_ms = float(start_t.elapsed_time(end_t))
        else:
            t1 = np.datetime64("now")
            overhead_ms = float((t1 - t0) / np.timedelta64(1, "ms"))

        return SelectionDecision(
            arm_name=arm_name,
            selected_skill_id=selected_id,
            scores_by_skill=full_scores,
            selection_overhead_ms=overhead_ms,
            selection_rationale=rationale,
        )
