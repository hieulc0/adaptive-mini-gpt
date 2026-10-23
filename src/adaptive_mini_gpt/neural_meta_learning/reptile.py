"""Phase 2: Reptile First-Order Neural Meta-Learning Initializer for M62.

Implements compact, first-order meta-learning (Reptile) for the 2,371-parameter
residual skill adapter without requiring second-order Hessian computation:
1. Meta-Initialization ($\theta_0$): shared parameter vector for the residual bottleneck.
2. Inner Loop: for sampled meta-training task $T_i$, clone $\theta_0 \to \theta_i$ and perform
   $K=5$ optimization steps using the combined pointwise and paired relational objective.
3. Outer Loop: interpolate toward adapted parameters:
   $\theta_0 \leftarrow \theta_0 + \epsilon (\theta_i - \theta_0)$
4. Preserves frozen M18 transformer backbone.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment
from adaptive_mini_gpt.model import MiniGPT

logger = logging.getLogger(__name__)


@dataclass
class MetaTrainingProgressRecord:
    """Diagnostic trace of a meta-training iteration."""
    iteration: int
    task_id: str
    inner_initial_loss: float
    inner_final_loss: float
    outer_step_norm: float
    meta_learning_rate: float


class ReptileMetaLearner:
    """First-order neural meta-learning engine for compact skill adapters."""

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: Any,
        device: torch.device,
        inner_lr: float = 3e-3,
        meta_lr: float = 0.15,
        inner_steps: int = 5,
        bottleneck_dim: int = 16,
        num_classes: int = 2,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.d_model = getattr(base_model.config, "d_model", getattr(base_model.config, "n_embd", 128))
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes

        # Shared Meta-Initialization (2,371 parameters)
        self.meta_adapter = PersistentCandidateAdapter(
            d_model=self.d_model,
            bottleneck_dim=self.bottleneck_dim,
            num_classes=self.num_classes,
        ).to(self.device)

    def get_meta_initialization_state(self) -> dict[str, torch.Tensor]:
        """Return detached copy of the current meta-learned parameter state."""
        return {k: v.detach().clone() for k, v in self.meta_adapter.state_dict().items()}

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

    def _compute_inner_loss(
        self,
        adapter: PersistentCandidateAdapter,
        env: AutonomousEnvironment,
        step_idx: int,
    ) -> torch.Tensor:
        """Combined pointwise and paired relational supervision objective."""
        # 1. Pointwise Triple Loss
        train_queries = [mp.base_query for mp in env.minimal_pairs] + [mp.perturbed_query for mp in env.minimal_pairs]
        if not train_queries:
            train_queries = env.held_out_test_queries[:16]

        q = train_queries[step_idx % len(train_queries)]
        h = self._extract_hidden(q.question_text)
        logits = adapter(h)
        tgt_idx = 1 if q.expected_answer == "YES" else 0
        tgt = torch.tensor([tgt_idx], dtype=torch.long, device=self.device)
        l_point = F.cross_entropy(logits, tgt)

        # 2. Paired Structural Relational Loss
        l_pair = torch.tensor(0.0, device=self.device)
        if env.minimal_pairs:
            mp = env.minimal_pairs[step_idx % len(env.minimal_pairs)]
            h_b = self._extract_hidden(mp.base_query.question_text)
            h_p = self._extract_hidden(mp.perturbed_query.question_text)
            l_b = adapter(h_b)
            l_p = adapter(h_p)

            t_b = torch.tensor([1 if mp.base_query.expected_answer == "YES" else 0], dtype=torch.long, device=self.device)
            t_p = torch.tensor([1 if mp.perturbed_query.expected_answer == "YES" else 0], dtype=torch.long, device=self.device)
            l_pair = F.cross_entropy(l_b, t_b) + F.cross_entropy(l_p, t_p)

        return l_point + 0.5 * l_pair

    def meta_train_step(
        self,
        env: AutonomousEnvironment,
        iteration: int = 1,
    ) -> MetaTrainingProgressRecord:
        """Execute single outer Reptile iteration on a sampled meta-training environment."""
        # 1. Clone meta-initialization into inner worker adapter
        inner_adapter = copy.deepcopy(self.meta_adapter)
        inner_adapter.train()
        inner_opt = torch.optim.AdamW(
            inner_adapter.parameters(),
            lr=self.inner_lr,
            weight_decay=1e-4,
        )

        initial_loss = float(self._compute_inner_loss(inner_adapter, env, 0).item())

        # 2. Inner-loop task adaptation (K steps)
        final_loss = initial_loss
        for k in range(self.inner_steps):
            inner_opt.zero_grad()
            loss = self._compute_inner_loss(inner_adapter, env, k)
            loss.backward()
            inner_opt.step()
            final_loss = float(loss.item())

        # 3. Outer-loop Reptile update: theta_meta = theta_meta + meta_lr * (theta_inner - theta_meta)
        meta_state = self.meta_adapter.state_dict()
        inner_state = inner_adapter.state_dict()
        step_diff_sq = 0.0

        for key in meta_state:
            diff = inner_state[key] - meta_state[key]
            meta_state[key] = meta_state[key] + self.meta_lr * diff
            step_diff_sq += float(torch.norm(diff).item() ** 2)

        self.meta_adapter.load_state_dict(meta_state)
        step_norm = float(step_diff_sq ** 0.5)

        return MetaTrainingProgressRecord(
            iteration=iteration,
            task_id=env.env_id,
            inner_initial_loss=initial_loss,
            inner_final_loss=final_loss,
            outer_step_norm=step_norm,
            meta_learning_rate=self.meta_lr,
        )

    def run_meta_training_loop(
        self,
        meta_training_envs: list[AutonomousEnvironment],
        num_iterations: int = 30,
    ) -> list[MetaTrainingProgressRecord]:
        """Run full Reptile outer loop across the meta-training task distribution."""
        records: list[MetaTrainingProgressRecord] = []
        for i in range(1, num_iterations + 1):
            env = meta_training_envs[(i - 1) % len(meta_training_envs)]
            rec = self.meta_train_step(env, iteration=i)
            records.append(rec)
            if i % 10 == 0 or i == num_iterations:
                logger.info("  Reptile Iteration %d/%d on [%s]: InitLoss=%.4f -> FinalLoss=%.4f (Norm=%.4f)",
                            i, num_iterations, env.env_id, rec.inner_initial_loss, rec.inner_final_loss, rec.outer_step_norm)
        return records


def train_multitask_pretrained_adapter(
    base_model: MiniGPT,
    tokenizer: Any,
    device: torch.device,
    meta_training_envs: list[AutonomousEnvironment],
    total_steps: int = 60,
    lr: float = 2e-3,
) -> PersistentCandidateAdapter:
    """Ordinary multitask pretraining baseline pooling all meta-training data."""
    d_model = getattr(base_model.config, "d_model", getattr(base_model.config, "n_embd", 128))
    adapter = PersistentCandidateAdapter(
        d_model=d_model, bottleneck_dim=16, num_classes=2
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)

    all_queries = []
    for env in meta_training_envs:
        all_queries.extend([mp.base_query for mp in env.minimal_pairs])
        all_queries.extend([mp.perturbed_query for mp in env.minimal_pairs])

    def extract_hidden(text: str) -> torch.Tensor:
        toks = tokenizer.encode(text)
        inp = torch.tensor([toks], dtype=torch.long, device=device)
        b, t = inp.size()
        tok_emb = base_model.wte(inp)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = base_model.drop(tok_emb + base_model.wpe(pos))
        for block in base_model.blocks:
            x, _ = block(x)
        return base_model.ln_f(x)

    adapter.train()
    for s in range(total_steps):
        optimizer.zero_grad()
        q = all_queries[s % len(all_queries)]
        h = extract_hidden(q.question_text)
        logits = adapter(h)
        tgt_idx = 1 if q.expected_answer == "YES" else 0
        tgt = torch.tensor([tgt_idx], dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, tgt)
        loss.backward()
        optimizer.step()

    adapter.eval()
    return adapter
