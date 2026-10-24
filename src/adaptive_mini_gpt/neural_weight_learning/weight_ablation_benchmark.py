"""Phase 3: 8-Arm Causal Weight-Ablation Benchmark.

Compares identical architectures and inference routes under 8 causal weight conditions:
1. TRAINED_ADAPTER: Properly trained on genuine task examples.
2. FRESH_RANDOM_ADAPTER: Untrained, freshly initialized weights.
3. RESET_ADAPTER: Re-initialized weights with different random seed.
4. SHUFFLED_TRAINED_WEIGHTS: Trained adapter with permuted weight elements.
5. UNRELATED_TRAINED_ADAPTER: Adapter trained on an unrelated procedural rule.
6. DISABLED_ADAPTER: Zero-output adapter (residual bypassed).
7. TRAINED_WITH_SHUFFLED_TARGETS: Adapter trained on scrambled training targets.
8. FROZEN_BASE_ONLY: Backbone representation with zero adapter adaptation.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.neural_weight_learning.pathway_instrumentation import (
    NeuralPathwayInstrumenter,
)
from adaptive_mini_gpt.neural_weight_learning.procedural_tasks import (
    ProceduralExample,
    ProceduralTaskDataset,
)
from adaptive_mini_gpt.neural_weight_learning.sequence_adapter import Rank16SequenceAdapter

logger = logging.getLogger(__name__)


@dataclass
class CausalAblationResult:
    """Metrics achieved by a specific causal weight condition on held-out sealed evaluation."""

    arm_name: str
    exact_match: float
    token_accuracy: float
    sequence_loss: float
    logit_divergence_vs_trained: float
    stored_parameters: int
    active_parameters: int
    inference_latency_ms: float
    causal_advantage_vs_shuffled: float


class CausalWeightAblationBenchmark:
    """Evaluates 8 causal weight conditions holding router and test inputs fixed."""

    @classmethod
    def evaluate_condition(
        cls,
        adapter: nn.Module | None,
        test_h: torch.Tensor,
        test_y: torch.Tensor,
        trained_logits: torch.Tensor | None = None,
    ) -> tuple[float, float, float, float, float, torch.Tensor]:
        """Evaluate a specific adapter on held-out hidden states."""
        t0 = time.perf_counter()
        if adapter is not None:
            adapter.eval()
            with torch.no_grad():
                logits = adapter(test_h)
        else:
            # FROZEN_BASE_ONLY: uniform logit baseline across the 16 symbols
            logits = torch.zeros(test_h.size(0), test_h.size(1), 16, device=test_h.device)

        loss = float(F.cross_entropy(logits.view(-1, 16), test_y.view(-1)).item())
        preds = torch.argmax(logits, dim=-1)
        em = float((preds == test_y).all(dim=-1).float().mean().item())
        tok_acc = float((preds == test_y).float().mean().item())

        if trained_logits is not None:
            div = float(torch.norm(logits - trained_logits).item())
        else:
            div = 0.0

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return em, tok_acc, loss, div, latency_ms, logits

    @classmethod
    def run_benchmark(
        cls,
        model: MiniGPT,
        task: ProceduralTaskDataset,
        unrelated_task: ProceduralTaskDataset,
        device: torch.device,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run complete 8-arm causal weight ablation on sealed test partition."""
        test_h, test_y = NeuralPathwayInstrumenter.extract_symbol_hidden_states(
            model, task.sealed_test_examples, device
        )

        # 1. Arm A: TRAINED_ADAPTER
        ad_trained = Rank16SequenceAdapter().to(device)
        ad_trained, _ = NeuralPathwayInstrumenter.train_and_instrument(
            model=model,
            adapter=ad_trained,
            task=task,
            adapter_id="cand_trained",
            device=device,
            steps=50,
        )
        em_a, tok_a, loss_a, _, lat_a, logits_a = cls.evaluate_condition(
            ad_trained, test_h, test_y, None
        )

        # 2. Arm B: FRESH_RANDOM_ADAPTER
        ad_fresh = ad_trained.reset_weights(seed=seed + 100).to(device)
        em_b, tok_b, loss_b, div_b, lat_b, _ = cls.evaluate_condition(
            ad_fresh, test_h, test_y, logits_a
        )

        # 3. Arm C: RESET_ADAPTER
        ad_reset = ad_trained.reset_weights(seed=seed + 200).to(device)
        em_c, tok_c, loss_c, div_c, lat_c, _ = cls.evaluate_condition(
            ad_reset, test_h, test_y, logits_a
        )

        # 4. Arm D: SHUFFLED_TRAINED_WEIGHTS
        ad_shuffled = ad_trained.shuffle_weights(seed=seed + 300).to(device)
        em_d, tok_d, loss_d, div_d, lat_d, _ = cls.evaluate_condition(
            ad_shuffled, test_h, test_y, logits_a
        )

        # 5. Arm E: UNRELATED_TRAINED_ADAPTER (trained on unrelated procedural rule)
        ad_unrelated = Rank16SequenceAdapter().to(device)
        ad_unrelated, _ = NeuralPathwayInstrumenter.train_and_instrument(
            model=model,
            adapter=ad_unrelated,
            task=unrelated_task,
            adapter_id="unrelated_rule",
            device=device,
            steps=50,
        )
        em_e, tok_e, loss_e, div_e, lat_e, _ = cls.evaluate_condition(
            ad_unrelated, test_h, test_y, logits_a
        )

        # 6. Arm F: DISABLED_ADAPTER (weights zeroed out)
        ad_disabled = copy.deepcopy(ad_trained)
        with torch.no_grad():
            for p in ad_disabled.parameters():
                p.zero_()
        em_f, tok_f, loss_f, div_f, lat_f, _ = cls.evaluate_condition(
            ad_disabled, test_h, test_y, logits_a
        )

        # 7. Arm G: TRAINED_WITH_SHUFFLED_TARGETS
        scrambled_task = copy.deepcopy(task)
        rng = np.random.default_rng(seed)
        shuffled_train_exs = []
        for ex in task.train_examples:
            scrambled_tgt = tuple(int(x) for x in rng.permutation(ex.target_tokens))
            shuffled_train_exs.append(
                ProceduralExample(
                    example_id=ex.example_id,
                    rule_family=ex.rule_family,
                    input_tokens=ex.input_tokens,
                    target_tokens=scrambled_tgt,
                    input_text=ex.input_text,
                    target_text=" ".join(f"S{s}" for s in scrambled_tgt),
                )
            )
        scrambled_task.train_examples = shuffled_train_exs
        ad_scrambled = Rank16SequenceAdapter().to(device)
        ad_scrambled, _ = NeuralPathwayInstrumenter.train_and_instrument(
            model=model,
            adapter=ad_scrambled,
            task=scrambled_task,
            adapter_id="scrambled_targets",
            device=device,
            steps=50,
        )
        em_g, tok_g, loss_g, div_g, lat_g, _ = cls.evaluate_condition(
            ad_scrambled, test_h, test_y, logits_a
        )

        # 8. Arm H: FROZEN_BASE_ONLY
        em_h, tok_h, loss_h, div_h, lat_h, _ = cls.evaluate_condition(
            None, test_h, test_y, logits_a
        )

        param_count = ad_trained.count_parameters()
        results = [
            CausalAblationResult(
                "A_TRAINED_ADAPTER",
                em_a,
                tok_a,
                loss_a,
                0.0,
                param_count,
                param_count,
                round(lat_a, 2),
                em_a - em_d,
            ),
            CausalAblationResult(
                "B_FRESH_RANDOM_ADAPTER",
                em_b,
                tok_b,
                loss_b,
                div_b,
                param_count,
                param_count,
                round(lat_b, 2),
                em_a - em_b,
            ),
            CausalAblationResult(
                "C_RESET_ADAPTER",
                em_c,
                tok_c,
                loss_c,
                div_c,
                param_count,
                param_count,
                round(lat_c, 2),
                em_a - em_c,
            ),
            CausalAblationResult(
                "D_SHUFFLED_TRAINED_WEIGHTS",
                em_d,
                tok_d,
                loss_d,
                div_d,
                param_count,
                param_count,
                round(lat_d, 2),
                em_a - em_d,
            ),
            CausalAblationResult(
                "E_UNRELATED_TRAINED_ADAPTER",
                em_e,
                tok_e,
                loss_e,
                div_e,
                param_count,
                param_count,
                round(lat_e, 2),
                em_a - em_e,
            ),
            CausalAblationResult(
                "F_DISABLED_ADAPTER",
                em_f,
                tok_f,
                loss_f,
                div_f,
                param_count,
                0,
                round(lat_f, 2),
                em_a - em_f,
            ),
            CausalAblationResult(
                "G_TRAINED_WITH_SHUFFLED_TARGETS",
                em_g,
                tok_g,
                loss_g,
                div_g,
                param_count,
                param_count,
                round(lat_g, 2),
                em_a - em_g,
            ),
            CausalAblationResult(
                "H_FROZEN_BASE_ONLY", em_h, tok_h, loss_h, div_h, 0, 0, round(lat_h, 2), em_a - em_h
            ),
        ]

        # Weight attribution is verified if TRAINED beats SHUFFLED and RESET by >= 0.50 Exact Match
        attribution_verified = (em_a - em_d >= 0.50) and (em_a - em_c >= 0.50)

        return {
            "task_id": task.task_id,
            "rule_family": task.rule_family.value,
            "attribution_verified": attribution_verified,
            "trained_exact_match": em_a,
            "shuffled_exact_match": em_d,
            "exact_match_delta_vs_shuffled": round(em_a - em_d, 4),
            "loss_delta_vs_shuffled": round(loss_d - loss_a, 4),
            "results": [asdict(r) for r in results],
        }
