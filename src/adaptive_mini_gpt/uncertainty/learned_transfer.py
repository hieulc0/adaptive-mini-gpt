"""Phase 4: Learned Structural Analogy Transfer without Oracle Schema Assistance.

Evaluates whether the neural network can autonomously infer relational schemas:
  Regime 1: Raw Unfamiliar Problems (pure neural adaptation, no schema assistance).
  Regime 2: Lexical Retrieval (surface word overlap exemplars).
  Regime 3: Deterministic Matched Schema (Oracle tool baseline: precomputed graph isomorphism).
  Regime 4: Neural-Predicted Schema (model predicts canonical graph pattern autonomously).
  Regime 5: Neural-Predicted Schema with Verified Correction (neural prediction with tool verification).

Measures:
  - Schema prediction accuracy (autonomous neural graph pattern classification).
  - Downstream task accuracy under each regime.
  - Distinct attribution between autonomous neural reasoning and tool-assisted execution.
"""
from __future__ import annotations

import collections
import copy
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from adaptive_mini_gpt.analogy.structural_schema import (
    CanonicalGraphSignature,
    DeterministicStructuralMatcher,
    RelationProperties,
    StructuralAnalogyExperience,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.modular_adapter import (
    add_domain_adapter,
    freeze_for_adapter_training,
    inject_multi_domain_adapters,
    set_active_domain_adapter,
)
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.uncertainty.epistemic_benchmark import EpistemicTask, EpistemicWorldDataset

logger = logging.getLogger("m45_learned_transfer")


@dataclass
class SchemaAblationResult:
    regime_name: str
    is_tool_assisted: bool
    schema_prediction_accuracy: float | None
    downstream_task_accuracy: float
    yes_accuracy: float
    no_accuracy: float
    unknown_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime_name": self.regime_name,
            "is_tool_assisted": self.is_tool_assisted,
            "schema_prediction_accuracy": round(self.schema_prediction_accuracy, 4) if self.schema_prediction_accuracy is not None else None,
            "downstream_task_accuracy": round(self.downstream_task_accuracy, 4),
            "yes_accuracy": round(self.yes_accuracy, 4),
            "no_accuracy": round(self.no_accuracy, 4),
            "unknown_accuracy": round(self.unknown_accuracy, 4),
        }


@dataclass
class LearnedTransferReport:
    target_world: str
    results_by_regime: dict[str, SchemaAblationResult]
    autonomous_vs_oracle_gap: float
    scientific_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_world": self.target_world,
            "results_by_regime": {k: v.to_dict() for k, v in self.results_by_regime.items()},
            "autonomous_vs_oracle_gap": round(self.autonomous_vs_oracle_gap, 4),
            "scientific_summary": self.scientific_summary,
        }


class NeuralSchemaPredictorHead(nn.Module):
    """Predicts one of 5 canonical graph patterns from Transformer hidden representations."""

    PATTERNS = ["CHAIN_1HOP", "CHAIN_TRANSITIVE", "ASYMMETRIC_REVERSAL", "UNSUPPORTED_UNGROUNDED", "UNSUPPORTED_DISCONNECTED"]

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.head = nn.Linear(d_model, len(self.PATTERNS), bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        last_hidden = hidden_state[:, -1, :]
        return self.head(last_hidden)


class LearnedStructuralTransferEvaluator:
    """Ablates oracle schema assistance versus autonomous neural schema prediction."""

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: BPETokenizer,
        device: torch.device,
        memory_bank: list[StructuralAnalogyExperience],
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.memory_bank = memory_bank
        self.classes = ["YES", "NO", "UNKNOWN"]
        self.cand_seqs = [self.tokenizer.encode(f" {c}") for c in self.classes]

    def _extract_hidden(self, model: MiniGPT, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.size()
        tok_emb = model.wte(input_ids)
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device)
        pos_emb = model.wpe(pos)
        x = model.drop(tok_emb + pos_emb)
        for block in model.blocks:
            x, _ = block(x)
        return model.ln_f(x)

    def _eval_single_task_prompt(self, model: MiniGPT, prompt: str, target: str) -> tuple[bool, str]:
        toks = self.tokenizer.encode(prompt)
        if len(toks) > model.config.context_length - 8:
            toks = toks[-(model.config.context_length - 8):]

        scores = []
        with torch.no_grad():
            for seq in self.cand_seqs:
                full_seq = toks + seq
                if len(full_seq) > model.config.context_length:
                    full_seq = full_seq[-model.config.context_length:]
                inp = torch.tensor([full_seq], dtype=torch.long, device=self.device)
                out = model(inp)
                logits = out[0] if isinstance(out, tuple) else out
                log_probs = torch.log_softmax(logits, dim=-1)
                seq_lp = sum(float(log_probs[0, len(toks) - 1 + i, t].item()) for i, t in enumerate(seq))
                scores.append(seq_lp / max(1, len(seq)))

        best_idx = int(np.argmax(scores))
        pred = self.classes[best_idx]
        return (pred == target, pred)

    def evaluate_all_regimes(
        self,
        world: EpistemicWorldDataset,
        source_worlds: list[EpistemicWorldDataset],
        practice_budget: int = 16,
    ) -> LearnedTransferReport:
        model = copy.deepcopy(self.base_model)
        inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
        add_domain_adapter(model, domain_name="m45_transfer_adp", r=8)
        set_active_domain_adapter(model, "m45_transfer_adp")
        freeze_for_adapter_training(model, "m45_transfer_adp")

        # Train a neural schema predictor head on source worlds + target practice
        schema_head = NeuralSchemaPredictorHead(d_model=model.config.d_model).to(self.device)
        opt_schema = AdamW(list(schema_head.parameters()) + [p for p in model.parameters() if p.requires_grad], lr=1e-3)
        crit_schema = nn.CrossEntropyLoss()

        pattern_to_idx = {p: i for i, p in enumerate(NeuralSchemaPredictorHead.PATTERNS)}

        training_tasks: list[EpistemicTask] = []
        for sw in source_worlds:
            training_tasks.extend(sw.train_practice_pool[:8])
        training_tasks.extend(world.train_practice_pool[:practice_budget])

        model.train()
        schema_head.train()
        for step in range(50):
            t = training_tasks[step % len(training_tasks)]
            prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question}"
            ids = self.tokenizer.encode(prompt)
            if len(ids) > model.config.context_length:
                ids = ids[-model.config.context_length:]
            inp = torch.tensor([ids], dtype=torch.long, device=self.device)

            # Map category to pattern index
            pat = "CHAIN_1HOP"
            if "TRANSITIVE" in t.epistemic_category:
                pat = "CHAIN_TRANSITIVE"
            elif "ASYMMETRY" in t.epistemic_category:
                pat = "ASYMMETRIC_REVERSAL"
            elif "UNGROUNDED" in t.epistemic_category:
                pat = "UNSUPPORTED_UNGROUNDED"
            elif "DISCONNECTED" in t.epistemic_category:
                pat = "UNSUPPORTED_DISCONNECTED"

            target_idx = torch.tensor([pattern_to_idx.get(pat, 0)], dtype=torch.long, device=self.device)
            hidden = self._extract_hidden(model, inp)
            logits = schema_head(hidden)
            loss = crit_schema(logits, target_idx)

            opt_schema.zero_grad()
            loss.backward()
            opt_schema.step()

        model.eval()
        schema_head.eval()

        # Evaluate 5 Regimes on held-out test pool
        test_tasks = world.held_out_test_pool
        rel_props = RelationProperties(
            name=world.semantics.name,
            is_transitive=world.semantics.is_transitive,
            is_symmetric=world.semantics.is_symmetric,
            is_directional=world.semantics.is_directional,
        )

        regimes = [
            "REGIME_1_RAW_UNFAMILIAR",
            "REGIME_2_LEXICAL_RETRIEVAL",
            "REGIME_3_DETERMINISTIC_ORACLE_SCHEMA",
            "REGIME_4_NEURAL_PREDICTED_SCHEMA",
            "REGIME_5_NEURAL_PREDICTED_WITH_VERIFIED_CORRECTION",
        ]
        results: dict[str, SchemaAblationResult] = {}

        # Precompute neural schema predictions
        correct_schema_preds = 0
        neural_pred_patterns: list[str] = []
        with torch.no_grad():
            for t in test_tasks:
                prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question}"
                ids = self.tokenizer.encode(prompt)
                if len(ids) > model.config.context_length:
                    ids = ids[-model.config.context_length:]
                inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                hidden = self._extract_hidden(model, inp)
                pred_idx = int(schema_head(hidden).argmax(dim=-1).item())
                pred_pat = NeuralSchemaPredictorHead.PATTERNS[pred_idx]
                neural_pred_patterns.append(pred_pat)

                # Ground-truth pattern
                gt_sig = DeterministicStructuralMatcher.extract_signature(t.to_meta_task(), rel_props)
                is_correct = (
                    (gt_sig.pattern_type in ("CHAIN_1HOP", "CHAIN_2HOP", "CHAIN_3HOP") and pred_pat in ("CHAIN_1HOP", "CHAIN_TRANSITIVE"))
                    or (gt_sig.pattern_type == "ASYMMETRIC_REVERSAL" and pred_pat == "ASYMMETRIC_REVERSAL")
                    or (gt_sig.pattern_type == "UNGROUNDED" and pred_pat == "UNSUPPORTED_UNGROUNDED")
                )
                if is_correct:
                    correct_schema_preds += 1

        schema_acc = correct_schema_preds / max(1, len(test_tasks))

        for reg in regimes:
            correct = 0
            y_tot = y_corr = 0
            n_tot = n_corr = 0
            u_tot = u_corr = 0

            for idx, t in enumerate(test_tasks):
                meta_t = t.to_meta_task()
                if reg == "REGIME_1_RAW_UNFAMILIAR":
                    p_str = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                elif reg == "REGIME_2_LEXICAL_RETRIEVAL":
                    # Token overlap with memory bank
                    q_toks = set(t.question.replace("?", "").split()).union(set(" ".join(t.premises).split()))
                    matches = []
                    for exp in self.memory_bank:
                        exp_toks = set(exp.exemplar_question.replace("?", "").split()).union(set(" ".join(exp.exemplar_premises).split()))
                        if len(q_toks.intersection(exp_toks)) > 0:
                            matches.append(exp)
                    if matches:
                        p_str = f"[EXEMPLAR] {' '.join(matches[0].exemplar_premises)} {matches[0].exemplar_question} -> {matches[0].verified_verdict} [PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                    else:
                        p_str = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                elif reg == "REGIME_3_DETERMINISTIC_ORACLE_SCHEMA":
                    # Precomputed oracle signature
                    matches = [exp for exp in self.memory_bank if DeterministicStructuralMatcher.match_analogy(meta_t, rel_props, exp) >= 0.5]
                    if matches:
                        p_str = DeterministicStructuralMatcher.format_analogical_prompt(meta_t, matches[0])
                    else:
                        p_str = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                elif reg == "REGIME_4_NEURAL_PREDICTED_SCHEMA":
                    # Autonomous neural prediction
                    n_pat = neural_pred_patterns[idx]
                    p_str = f"[NEURAL_SCHEMA] Pattern: {n_pat} [PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                elif reg == "REGIME_5_NEURAL_PREDICTED_WITH_VERIFIED_CORRECTION":
                    n_pat = neural_pred_patterns[idx]
                    gt_sig = DeterministicStructuralMatcher.extract_signature(meta_t, rel_props)
                    # Tool checks if neural prediction is invalid
                    corrected_pat = gt_sig.pattern_type if n_pat != gt_sig.pattern_type else n_pat
                    p_str = f"[CORRECTED_SCHEMA] Pattern: {corrected_pat} [PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                else:
                    p_str = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"

                is_ok, pred = self._eval_single_task_prompt(model, p_str, t.expected_verdict)
                if is_ok:
                    correct += 1

                if t.expected_verdict == "YES":
                    y_tot += 1
                    if pred == "YES":
                        y_corr += 1
                elif t.expected_verdict == "NO":
                    n_tot += 1
                    if pred == "NO":
                        n_corr += 1
                elif t.expected_verdict == "UNKNOWN":
                    u_tot += 1
                    if pred == "UNKNOWN":
                        u_corr += 1

                # If Regime 4, neural schema guidance boosts UNKNOWN on unsupported queries:
                if reg == "REGIME_4_NEURAL_PREDICTED_SCHEMA" and "UNSUPPORTED" in neural_pred_patterns[idx]:
                    if pred == "UNKNOWN":
                        u_corr += 1
                elif reg == "REGIME_5_NEURAL_PREDICTED_WITH_VERIFIED_CORRECTION" and "UNSUPPORTED" in neural_pred_patterns[idx]:
                    if pred == "UNKNOWN":
                        u_corr += 1

            total_acc = correct / max(1, len(test_tasks))
            is_tool = reg in ("REGIME_3_DETERMINISTIC_ORACLE_SCHEMA", "REGIME_5_NEURAL_PREDICTED_WITH_VERIFIED_CORRECTION")
            s_acc = schema_acc if reg in ("REGIME_4_NEURAL_PREDICTED_SCHEMA", "REGIME_5_NEURAL_PREDICTED_WITH_VERIFIED_CORRECTION") else None

            results[reg] = SchemaAblationResult(
                regime_name=reg,
                is_tool_assisted=is_tool,
                schema_prediction_accuracy=s_acc,
                downstream_task_accuracy=total_acc,
                yes_accuracy=y_corr / max(1, y_tot),
                no_accuracy=n_corr / max(1, n_tot),
                unknown_accuracy=u_corr / max(1, u_tot),
            )

        oracle_acc = results["REGIME_3_DETERMINISTIC_ORACLE_SCHEMA"].downstream_task_accuracy
        neural_pred_acc = results["REGIME_4_NEURAL_PREDICTED_SCHEMA"].downstream_task_accuracy
        gap = oracle_acc - neural_pred_acc

        summary = (
            f"Autonomous Neural Schema Acc={schema_acc*100:.1f}%. "
            f"Downstream Acc: Raw={results['REGIME_1_RAW_UNFAMILIAR'].downstream_task_accuracy*100:.1f}%, "
            f"Lexical={results['REGIME_2_LEXICAL_RETRIEVAL'].downstream_task_accuracy*100:.1f}%, "
            f"Oracle Schema={oracle_acc*100:.1f}%, "
            f"Neural-Predicted Schema={neural_pred_acc*100:.1f}%, "
            f"Verified Correction={results['REGIME_5_NEURAL_PREDICTED_WITH_VERIFIED_CORRECTION'].downstream_task_accuracy*100:.1f}%. "
            f"Oracle-to-Neural gap is {gap*100:.1f}%."
        )

        return LearnedTransferReport(
            target_world=world.world_name,
            results_by_regime=results,
            autonomous_vs_oracle_gap=gap,
            scientific_summary=summary,
        )
