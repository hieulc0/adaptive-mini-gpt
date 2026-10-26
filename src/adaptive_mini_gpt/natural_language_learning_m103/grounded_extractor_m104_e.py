"""Milestone M104-E Grounded Concept Extractor and Comparative Diagnostic Auditor.

Features:
1. Three Neural Extraction Strategies:
   - Question-Only: Pools representations strictly from question/entity tokens.
   - Full-Context: Pools representations over all non-padding prompt tokens.
   - Entity-Conditioned Cross-Attention (ECA): Uses entity/question representation as query
     to attend selectively over the full prompt, extracting relevant entity properties while
     suppressing rule-text leakage.
2. Trivial Lexical / Bag-of-Words Baseline:
   - Evaluates on both question_only and full_context to isolate the precise impact of rule contamination.
   - Proves whether property descriptions contain sufficient information for attribute classification.
3. Decoupled Stage Metrics:
   - Micro-F1, Macro-F1, Precision, Recall, Exact Attribute Set Match %.
   - Full 26-class true positive, false positive, and false negative analysis.
4. Tiny-Batch Convergence Audit:
   - Tests supervised text-to-attribute optimization on tiny batches to verify gradient flow
     and rule-learning convergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch
import torch.nn as nn

from adaptive_mini_gpt.natural_language_learning_m103.causal_world_m104_b import (
    CausalItemM104_B,
    ProceduralCausalWorldM104_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)


class ExtractionStrategyTypeM104_E(StrEnum):
    QUESTION_ONLY = "question_only"
    FULL_CONTEXT = "full_context"
    ENTITY_CONDITIONED_ATTENTION = "entity_conditioned_attention"


@dataclass
class AttributeClassMetricM104_E:
    tag_name: str
    tag_id: int
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


@dataclass
class ExtractionEvaluationReportM104_E:
    strategy_name: str
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_f1: float
    exact_tag_set_match_pct: float
    optimal_threshold: float
    sample_count: int
    per_attribute_metrics: dict[str, AttributeClassMetricM104_E] = field(default_factory=dict)
    top_false_positive_classes: list[str] = field(default_factory=list)


@dataclass
class ExtractionStrategyComparisonReportM104_E:
    lexical_baseline: ExtractionEvaluationReportM104_E
    lexical_full_context: ExtractionEvaluationReportM104_E
    question_only: ExtractionEvaluationReportM104_E
    full_context: ExtractionEvaluationReportM104_E
    entity_conditioned_attention: ExtractionEvaluationReportM104_E
    best_strategy: str
    best_f1: float
    lexical_vs_neural_gap_f1: float
    scientific_summary: str


class LexicalKeywordBaselineM104_E:
    """Trivial lexical / rule-based extractor using precise keyword matching."""

    def predict_tags(self, text: str) -> set[str]:
        """Extracts tags from raw text using substring/keyword matching."""
        lower = f" {text.lower()} ".replace(".", " ").replace(",", " ").replace("-", "_")
        tags: set[str] = set()

        # Material mappings
        if " wooden " in lower or " wood " in lower:
            tags.update(["wood", "flammable", "organic"])
        if " metal " in lower or " dense iron " in lower or " iron " in lower or " copper " in lower or " steel " in lower or " bronze " in lower or " electrum " in lower:
            tags.update(["metal", "heat_tolerant", "conductive"])
        if " stone " in lower or " granite " in lower or " basalt " in lower or " marble " in lower:
            tags.update(["stone", "heat_tolerant"])
        if " crystal " in lower or " crystalline " in lower or " quartz " in lower or " beryl " in lower:
            tags.update(["crystalline", "optical"])
        if " glass " in lower:
            tags.update(["glass", "optical"])
        if " cloth " in lower:
            tags.update(["cloth", "flammable", "organic"])
        if " clay " in lower or " terracotta " in lower:
            tags.update(["clay", "heat_tolerant"])
        if " ice " in lower:
            tags.update(["ice", "cold"])
        if " obsidian " in lower:
            tags.update(["stone", "crystalline", "heat_tolerant"])
        if " silver " in lower:
            tags.update(["metal", "conductive", "magical"])

        # Affiliation / Metaphysical mappings
        if " magical " in lower or " arcane " in lower:
            tags.add("magical")
        if " radiant " in lower or " luminescence " in lower or " glowing " in lower:
            tags.update(["radiant", "optical"])
        if " inscribed " in lower or " runic " in lower:
            tags.update(["inscribed", "relic"])
        if " alchemical " in lower:
            tags.add("alchemical")
        if " shadow " in lower or " umbral " in lower:
            tags.add("shadow_infused")

        # Form mappings
        if " tool " in lower or " torch " in lower or " diapason " in lower or " mirror " in lower:
            tags.add("tool")
        if " container " in lower or " crucible " in lower:
            tags.update(["container", "vessel"])
        if " mineral " in lower or " glowstone " in lower:
            tags.update(["mineral", "radiant", "optical"])
        if " tablet " in lower or " rune " in lower or " sigil " in lower:
            tags.update(["relic", "magical", "inscribed"])
        if " vessel " in lower or " cauldron " in lower or " chalice " in lower or " flask " in lower:
            tags.update(["container", "vessel"])
        if " optical " in lower or " prism " in lower or " reflector " in lower:
            tags.add("optical")
        if " instrument " in lower or " bell " in lower or " jewelry " in lower or " amulet " in lower:
            tags.add("relic")
        if " conduit " in lower or " rod " in lower or " spire " in lower:
            tags.update(["conduit", "tool"])

        return tags


class EntityConditionedAttentionExtractorM104_E(nn.Module):
    """Compact Entity-Conditioned Cross-Attention (ECA) module.

    Projects entity query and sequence tokens into a compact latent attention space,
    cross-attends, and projects back into representation space.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_attn: int = 32,
        num_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_attn = d_attn

        self.q_proj = nn.Linear(d_model, d_attn)
        self.k_proj = nn.Linear(d_model, d_attn)
        self.v_proj = nn.Linear(d_model, d_attn)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_attn,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_attn, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query_repr: torch.Tensor,
        seq_hidden_states: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query_repr: (B, 1, d_model) entity query.
            seq_hidden_states: (B, L, d_model) sequence states.
            key_padding_mask: (B, L) padding mask.

        Returns:
            attended_repr: (B, d_model)
        """
        q = self.q_proj(query_repr)
        k = self.k_proj(seq_hidden_states)
        v = self.v_proj(seq_hidden_states)

        attn_out, _ = self.cross_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
        )
        delta = self.out_proj(attn_out.squeeze(1))
        return self.norm(query_repr.squeeze(1) + delta)


class GroundedBottleneckAuditorM104_E:
    """Audits text-to-attribute extraction across multiple strategies and baselines."""

    TAG_VOCAB: dict[str, int] = {
        "wood": 0, "metal": 1, "stone": 2, "crystalline": 3, "glass": 4, "cloth": 5,
        "clay": 6, "ice": 7, "obsidian": 8, "silver": 9, "flammable": 10, "heat_tolerant": 11,
        "magical": 12, "radiant": 13, "inscribed": 14, "conductive": 15, "organic": 16,
        "shadow_infused": 17, "alchemical": 18, "container": 19, "tool": 20, "relic": 21,
        "optical": 22, "vessel": 23, "conduit": 24, "mineral": 25,
    }
    REV_TAG_VOCAB: dict[int, str] = {v: k for k, v in TAG_VOCAB.items()}

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.action_vocab = {
            act: i for i, act in enumerate(ProceduralCausalWorldM104_B.ACTIONS + ProceduralCausalWorldM104_B.EXTENDED_ACTIONS)
        }
        self.state_vocab = {st: i for i, st in enumerate(ProceduralCausalWorldM104_B.STATES)}
        self.lexical_baseline = LexicalKeywordBaselineM104_E()

    def evaluate_lexical_baseline(
        self,
        items: list[CausalItemM104_B],
        objects: dict[str, Any],
        use_question_only: bool = True,
    ) -> ExtractionEvaluationReportM104_E:
        """Evaluates trivial lexical keyword matching on items."""
        class_metrics = {
            t_name: AttributeClassMetricM104_E(tag_name=t_name, tag_id=t_id)
            for t_name, t_id in self.TAG_VOCAB.items()
        }

        total_tp = 0
        total_fp = 0
        total_fn = 0
        exact_matches = 0
        total_items = len(items)

        for it in items:
            obj = objects.get(it.object_name)
            true_tags = obj.tags if obj else set()
            true_tag_indices = {self.TAG_VOCAB[t] for t in true_tags if t in self.TAG_VOCAB}

            # Prompt text containing description
            if use_question_only:
                desc_text = it.question_text
            else:
                desc_text = f"{it.rule_explanation} {it.question_text}"

            pred_tag_names = self.lexical_baseline.predict_tags(desc_text)
            pred_indices = {self.TAG_VOCAB[t] for t in pred_tag_names if t in self.TAG_VOCAB}

            for t_name, t_id in self.TAG_VOCAB.items():
                is_true = t_id in true_tag_indices
                is_pred = t_id in pred_indices
                if is_true and is_pred:
                    class_metrics[t_name].true_positives += 1
                elif not is_true and is_pred:
                    class_metrics[t_name].false_positives += 1
                elif is_true and not is_pred:
                    class_metrics[t_name].false_negatives += 1
                else:
                    class_metrics[t_name].true_negatives += 1

            tp = len(pred_indices.intersection(true_tag_indices))
            fp = len(pred_indices.difference(true_tag_indices))
            fn = len(true_tag_indices.difference(pred_indices))
            total_tp += tp
            total_fp += fp
            total_fn += fn
            if pred_indices == true_tag_indices:
                exact_matches += 1

        macro_f1_sum = 0.0
        active_classes = 0
        for m in class_metrics.values():
            m.precision = (m.true_positives / max(1, m.true_positives + m.false_positives)) * 100.0
            m.recall = (m.true_positives / max(1, m.true_positives + m.false_negatives)) * 100.0
            m.f1_score = (2 * m.precision * m.recall) / max(1e-9, m.precision + m.recall)
            if (m.true_positives + m.false_negatives) > 0:
                macro_f1_sum += m.f1_score
                active_classes += 1

        macro_f1 = macro_f1_sum / max(1, active_classes)
        micro_prec = (total_tp / max(1, total_tp + total_fp)) * 100.0
        micro_rec = (total_tp / max(1, total_tp + total_fn)) * 100.0
        micro_f1 = (2 * micro_prec * micro_rec) / max(1e-9, micro_prec + micro_rec)
        exact_match_pct = (exact_matches / max(1, total_items)) * 100.0

        sorted_fp = sorted(
            class_metrics.keys(),
            key=lambda k: class_metrics[k].false_positives,
            reverse=True,
        )[:5]

        strategy_lbl = "lexical_question_only" if use_question_only else "lexical_full_context"
        return ExtractionEvaluationReportM104_E(
            strategy_name=strategy_lbl,
            micro_precision=micro_prec,
            micro_recall=micro_rec,
            micro_f1=micro_f1,
            macro_f1=macro_f1,
            exact_tag_set_match_pct=exact_match_pct,
            optimal_threshold=0.50,
            sample_count=total_items,
            per_attribute_metrics=class_metrics,
            top_false_positive_classes=sorted_fp,
        )

    def evaluate_neural_extractor(
        self,
        student_model: Any,
        items: list[CausalItemM104_B],
        objects: dict[str, Any],
        tokenizer: ByteTextTokenizerM103,
        strategy_name: str = "question_only",
        decision_threshold: float = 0.50,
    ) -> ExtractionEvaluationReportM104_E:
        """Evaluates neural attribute extraction for a specific pooling/attention strategy."""
        student_model.eval()

        class_metrics = {
            t_name: AttributeClassMetricM104_E(tag_name=t_name, tag_id=t_id)
            for t_name, t_id in self.TAG_VOCAB.items()
        }

        total_tp = 0
        total_fp = 0
        total_fn = 0
        exact_matches = 0
        total_items = len(items)

        with torch.no_grad():
            for it in items:
                obj = objects.get(it.object_name)
                true_tags = obj.tags if obj else set()
                true_tag_indices = {self.TAG_VOCAB[t] for t in true_tags if t in self.TAG_VOCAB}

                prompt_ids = tokenizer.encode_prompt(
                    question=it.question_text, explanation=it.rule_explanation
                )
                p_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
                act_id = torch.tensor([self.action_vocab.get(it.action_name, 0)], dtype=torch.long, device=self.device)

                if hasattr(student_model, "forward_with_strategy"):
                    out = student_model.forward_with_strategy(p_ids, act_id, strategy=strategy_name)
                else:
                    out = student_model(p_ids, act_id)

                tag_logits = out[1] if len(out) > 1 and out[1] is not None else None
                if tag_logits is not None:
                    probs = torch.sigmoid(tag_logits[0])
                    pred_indices = set(torch.where(probs >= decision_threshold)[0].tolist())
                else:
                    pred_indices = set()

                for t_name, t_id in self.TAG_VOCAB.items():
                    is_true = t_id in true_tag_indices
                    is_pred = t_id in pred_indices
                    if is_true and is_pred:
                        class_metrics[t_name].true_positives += 1
                    elif not is_true and is_pred:
                        class_metrics[t_name].false_positives += 1
                    elif is_true and not is_pred:
                        class_metrics[t_name].false_negatives += 1
                    else:
                        class_metrics[t_name].true_negatives += 1

                tp = len(pred_indices.intersection(true_tag_indices))
                fp = len(pred_indices.difference(true_tag_indices))
                fn = len(true_tag_indices.difference(pred_indices))
                total_tp += tp
                total_fp += fp
                total_fn += fn
                if pred_indices == true_tag_indices:
                    exact_matches += 1

        macro_f1_sum = 0.0
        active_classes = 0
        for m in class_metrics.values():
            m.precision = (m.true_positives / max(1, m.true_positives + m.false_positives)) * 100.0
            m.recall = (m.true_positives / max(1, m.true_positives + m.false_negatives)) * 100.0
            m.f1_score = (2 * m.precision * m.recall) / max(1e-9, m.precision + m.recall)
            if (m.true_positives + m.false_negatives) > 0:
                macro_f1_sum += m.f1_score
                active_classes += 1

        macro_f1 = macro_f1_sum / max(1, active_classes)
        micro_prec = (total_tp / max(1, total_tp + total_fp)) * 100.0
        micro_rec = (total_tp / max(1, total_tp + total_fn)) * 100.0
        micro_f1 = (2 * micro_prec * micro_rec) / max(1e-9, micro_prec + micro_rec)
        exact_match_pct = (exact_matches / max(1, total_items)) * 100.0

        sorted_fp = sorted(
            class_metrics.keys(),
            key=lambda k: class_metrics[k].false_positives,
            reverse=True,
        )[:5]

        return ExtractionEvaluationReportM104_E(
            strategy_name=strategy_name,
            micro_precision=micro_prec,
            micro_recall=micro_rec,
            micro_f1=micro_f1,
            macro_f1=macro_f1,
            exact_tag_set_match_pct=exact_match_pct,
            optimal_threshold=decision_threshold,
            sample_count=total_items,
            per_attribute_metrics=class_metrics,
            top_false_positive_classes=sorted_fp,
        )

    def calibrate_optimal_threshold(
        self,
        student_model: Any,
        items: list[CausalItemM104_B],
        objects: dict[str, Any],
        tokenizer: ByteTextTokenizerM103,
        strategy_name: str = "entity_conditioned_attention",
        threshold_candidates: list[float] | None = None,
    ) -> float:
        """Finds optimal decision threshold tau that maximizes micro-F1."""
        if threshold_candidates is None:
            threshold_candidates = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

        student_model.eval()
        all_probs: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        with torch.no_grad():
            for it in items:
                obj = objects.get(it.object_name)
                true_tags = obj.tags if obj else set()
                tgt = torch.zeros(len(self.TAG_VOCAB), dtype=torch.float32)
                for t in true_tags:
                    if t in self.TAG_VOCAB:
                        tgt[self.TAG_VOCAB[t]] = 1.0
                all_targets.append(tgt)

                prompt_ids = tokenizer.encode_prompt(
                    question=it.question_text, explanation=it.rule_explanation
                )
                p_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
                act_id = torch.tensor([self.action_vocab.get(it.action_name, 0)], dtype=torch.long, device=self.device)

                if hasattr(student_model, "forward_with_strategy"):
                    out = student_model.forward_with_strategy(p_ids, act_id, strategy=strategy_name)
                else:
                    out = student_model(p_ids, act_id)

                tag_logits = out[1] if len(out) > 1 and out[1] is not None else None
                if tag_logits is not None:
                    probs = torch.sigmoid(tag_logits[0]).cpu()
                else:
                    probs = torch.zeros(len(self.TAG_VOCAB))
                all_probs.append(probs)

        if not all_probs:
            return 0.50

        best_tau = 0.50
        best_f1 = -1.0
        for tau in threshold_candidates:
            tp, fp, fn = 0, 0, 0
            for prob, tgt in zip(all_probs, all_targets, strict=False):
                pred = (prob >= tau).float()
                tp += int((pred * tgt).sum().item())
                fp += int((pred * (1.0 - tgt)).sum().item())
                fn += int(((1.0 - pred) * tgt).sum().item())
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = (2 * prec * rec) / max(1e-9, prec + rec)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau

        return best_tau
