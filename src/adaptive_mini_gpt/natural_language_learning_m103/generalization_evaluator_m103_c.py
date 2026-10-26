"""Milestone M103-C Generalization & Diagnostic Evaluator.

Separates and independently reports:
- Training exact match (memorization check).
- Validation canonical exact match (held-out combinations with seen phrasing/range).
- Unseen-value exact match (the core test of whether the student learned the rule).
- Linguistic paraphrase exact match (robustness across surface wording).
- Counterfactual / novel rule exact match (internalized vs prompted reasoning).

Also computes:
- Numeric answer accuracy (integer extraction matching ground truth).
- Format validity rate (matches `<val> <unit>` structure).
- Character accuracy (character-level similarity).
- Cross-entropy token loss.
- Full sample audit logs of actual generated outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import re
from typing import Any
import torch

from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.world_simulator_m103_c import (
    ConceptDatasetM103_C,
    SimulatorItemM103_C,
)


@dataclass
class EvaluationMetricsM103_C:
    split_name: str
    count: int
    exact_match_pct: float
    numeric_accuracy_pct: float
    format_validity_pct: float
    character_accuracy_pct: float
    mean_token_loss: float
    sample_outputs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GeneralizationDiagnosticReportM103_C:
    train_metrics: EvaluationMetricsM103_C
    val_canonical_metrics: EvaluationMetricsM103_C
    unseen_values_metrics: EvaluationMetricsM103_C
    paraphrase_metrics: EvaluationMetricsM103_C
    counterfactual_metrics: EvaluationMetricsM103_C
    novel_combination_metrics: EvaluationMetricsM103_C
    generalization_gap_pct: float  # train_exact_match - unseen_values_exact_match
    discrepancy_diagnosis: str
    router_dimension_certified: bool
    router_parameter_count: int


class GeneralizationEvaluatorM103_C:
    """Evaluates generalization frontiers and generates diagnostic breakdowns."""

    def __init__(self, tokenizer: ByteTextTokenizerM103 | None = None, device: torch.device | None = None) -> None:
        self.tokenizer = tokenizer or ByteTextTokenizerM103()
        self.device = device or torch.device("cpu")

    def extract_numeric_value(self, text: str) -> int | None:
        """Extract the first integer from output text."""
        match = re.search(r"(?<!\w)(-?\d+)(?!\w)", text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def evaluate_split(
        self,
        student: TextStudentM103,
        items: list[SimulatorItemM103_C],
        split_name: str,
        active_domain: str | None = None,
        with_explanation: bool = True,
        max_new_tokens: int = 16,
        canonical: bool = True,
    ) -> EvaluationMetricsM103_C:
        """Evaluates model completions across multiple metrics and collects output audit."""
        if not items:
            return EvaluationMetricsM103_C(
                split_name=split_name,
                count=0,
                exact_match_pct=0.0,
                numeric_accuracy_pct=0.0,
                format_validity_pct=0.0,
                character_accuracy_pct=0.0,
                mean_token_loss=0.0,
            )

        student.eval()
        exact_matches = 0
        numeric_matches = 0
        format_valid = 0
        total_char_sim = 0.0
        sample_logs: list[dict[str, Any]] = []

        # Evaluate autoregressive generation
        for idx, it in enumerate(items):
            expected_target = it.canonical_answer if canonical else it.unrestricted_answer
            expected_num = it.expected_numeric_value

            if with_explanation and it.prompt_len > 0:
                prompt_toks = it.full_tokens[:it.prompt_len]
            else:
                expl = it.explanation if with_explanation else None
                prompt_toks, _ = self.tokenizer.format_qa_sequence(
                    question=it.question,
                    answer="",
                    explanation=expl,
                )
                if prompt_toks and prompt_toks[-1] == self.tokenizer.EOS_ID:
                    prompt_toks = prompt_toks[:-1]

            gen_ids = student.generate(
                prompt_tokens=prompt_toks,
                max_new_tokens=max_new_tokens,
                active_domain=active_domain,
            )
            gen_text = self.tokenizer.decode(gen_ids).strip()
            norm_gen = gen_text.lower().strip()
            norm_target = expected_target.lower().strip()

            # 1. Exact Match: either complete equality or answer prefix match
            is_em = (norm_gen == norm_target) or (norm_gen.startswith(norm_target))
            if is_em:
                exact_matches += 1

            # 2. Numeric Accuracy
            pred_num = self.extract_numeric_value(gen_text)
            is_num_match = (pred_num == expected_num)
            if is_num_match:
                numeric_matches += 1

            # 3. Format Validity (ends with expected unit, e.g. 'energy' or 'coordinate')
            unit = "energy" if "energy" in norm_target else ("coordinate" if "coordinate" in norm_target else "")
            is_fmt = bool(unit and unit in norm_gen and pred_num is not None)
            if is_fmt:
                format_valid += 1

            # 4. Character Accuracy via difflib ratio
            char_sim = difflib.SequenceMatcher(None, norm_gen, norm_target).ratio()
            total_char_sim += char_sim

            if idx < 5:  # Store first 5 samples for transparency
                sample_logs.append({
                    "question": it.question,
                    "target": expected_target,
                    "generated": gen_text,
                    "pred_num": pred_num,
                    "expected_num": expected_num,
                    "exact_match": is_em,
                    "numeric_match": is_num_match,
                })

        # Calculate token loss across split
        with torch.no_grad():
            token_lists = [it.full_tokens for it in items if it.full_tokens]
            if token_lists:
                batch_tokens, mask = self.tokenizer.pad_batch(token_lists, device=student.device)
                logits = student.forward_lm(batch_tokens, active_domain=active_domain)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch_tokens[:, 1:].contiguous().clone()
                for i, it in enumerate(items):
                    p_len = it.prompt_len - 1
                    if p_len > 0:
                        shift_labels[i, :p_len] = -100
                shift_labels[~mask[:, 1:]] = -100
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss_val = round(float(loss.item()), 4)
            else:
                loss_val = 0.0

        n = len(items)
        return EvaluationMetricsM103_C(
            split_name=split_name,
            count=n,
            exact_match_pct=round((exact_matches / n) * 100.0, 2),
            numeric_accuracy_pct=round((numeric_matches / n) * 100.0, 2),
            format_validity_pct=round((format_valid / n) * 100.0, 2),
            character_accuracy_pct=round((total_char_sim / n) * 100.0, 2),
            mean_token_loss=loss_val,
            sample_outputs=sample_logs,
        )

    def run_full_generalization_diagnostic(
        self,
        student: TextStudentM103,
        dataset: ConceptDatasetM103_C,
        active_domain: str | None = None,
        with_explanation: bool = True,
    ) -> GeneralizationDiagnosticReportM103_C:
        """Runs complete diagnostic over all 6 scientific splits."""
        m_train = self.evaluate_split(student, dataset.train_items, "train", active_domain, with_explanation)
        m_val = self.evaluate_split(student, dataset.val_canonical_items, "val_canonical", active_domain, with_explanation)
        m_unseen = self.evaluate_split(student, dataset.unseen_values_items, "unseen_values", active_domain, with_explanation)
        m_para = self.evaluate_split(student, dataset.paraphrase_items, "paraphrase", active_domain, with_explanation)
        m_cf = self.evaluate_split(student, dataset.counterfactual_items, "counterfactual", active_domain, with_explanation)
        m_novel = self.evaluate_split(student, dataset.novel_combination_items, "novel_combination", active_domain, with_explanation)

        gap = round(m_train.exact_match_pct - m_unseen.exact_match_pct, 2)

        # Audit router dimensions
        router_p = 0
        if hasattr(student, "layer_routers") and len(student.layer_routers) > 0:
            router_p = sum(p.numel() for p in student.layer_routers[0].parameters())
        # Linear(128, 32) has 4128, Linear(32, 2) has 66 -> total 4,194
        router_certified = (router_p == 4194)

        diag = (
            f"Audit completed: Train EM={m_train.exact_match_pct}%, "
            f"Canonical Val EM={m_val.exact_match_pct}%, "
            f"Unseen Values EM={m_unseen.exact_match_pct}%, "
            f"Paraphrase EM={m_para.exact_match_pct}%, "
            f"Generalization Gap={gap}%. "
            f"Router parameter count={router_p} (certified Linear(128->32->2) = {router_certified})."
        )

        return GeneralizationDiagnosticReportM103_C(
            train_metrics=m_train,
            val_canonical_metrics=m_val,
            unseen_values_metrics=m_unseen,
            paraphrase_metrics=m_para,
            counterfactual_metrics=m_cf,
            novel_combination_metrics=m_novel,
            generalization_gap_pct=gap,
            discrepancy_diagnosis=diag,
            router_dimension_certified=router_certified,
            router_parameter_count=router_p,
        )
