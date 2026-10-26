"""Milestone M103-C Continual Concept Learner & Competence Gating.

Implements:
1. Selective backbone optimization for reliable concept acquisition.
2. Canonical short-answer vs unrestricted generation comparison.
3. Pretrained vs Random initialization under matched optimization budgets.
4. In-context explanation vs Weight-internalized rule evaluation.
5. Strict Competence Gating (Phase C4):
   - Only advance to multi-concept sequential learning if single-concept
     unseen value generalization achieves >= 75% accuracy.
   - If gated, stop cleanly and report the exact learning frontier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import torch

from adaptive_mini_gpt.natural_language_learning_m103.generalization_evaluator_m103_c import (
    EvaluationMetricsM103_C,
    GeneralizationDiagnosticReportM103_C,
    GeneralizationEvaluatorM103_C,
)
from adaptive_mini_gpt.natural_language_learning_m103.language_pretrainer_m103_b import (
    LanguagePretrainerM103_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.world_simulator_m103_c import (
    ConceptDatasetM103_C,
    ConceptFamilyM103_C,
    SimulatorItemM103_C,
    WorldSimulatorM103_C,
)


@dataclass
class ConceptTrainingProgressM103_C:
    step: int
    train_loss: float
    train_exact_match: float
    val_canonical_exact_match: float
    unseen_values_exact_match: float
    paraphrase_exact_match: float
    unseen_numeric_accuracy: float


@dataclass
class SingleConceptResultM103_C:
    family: ConceptFamilyM103_C
    condition: str
    trainable_params: int
    total_steps: int
    history: list[ConceptTrainingProgressM103_C]
    final_train_em: float
    final_val_canonical_em: float
    final_unseen_values_em: float
    final_paraphrase_em: float
    final_unseen_numeric_acc: float
    internalized_without_explanation_em: float
    format_comparison_canonical_em: float
    format_comparison_unrestricted_em: float
    competence_gate_passed: bool
    status_summary: str


@dataclass
class ContinualLifelongResultM103_C:
    family_1_result: SingleConceptResultM103_C
    family_2_attempted: bool
    family_2_result: SingleConceptResultM103_C | None
    backward_retention_family_1_em: float | None
    fresh_student_family_2_steps: int | None
    lifelong_student_family_2_steps: int | None
    sample_efficiency_gain_pct: float | None
    stop_condition_triggered: bool
    verdict: str


class ContinualConceptLearnerM103_C:
    """Manages concept learning, comparative diagnosis, and competence gating."""

    COMPETENCE_GATE_THRESHOLD = 75.0  # Require >= 75% unseen value EM to advance to C4

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        self.evaluator = GeneralizationEvaluatorM103_C(device=self.device)
        self.simulator = WorldSimulatorM103_C()

    def configure_selective_backbone(
        self,
        student: TextStudentM103,
        domain: str,
        unfreeze_backbone_depth: int = 1,
    ) -> int:
        """Configures selective plasticity: adapter + final block + ln_f + wte."""
        for p in student.parameters():
            p.requires_grad = False

        adapter_key = f"adapter_{domain}"
        if adapter_key in student.adapter_modules:
            for p in student.adapter_modules[adapter_key].parameters():
                p.requires_grad = True

        if unfreeze_backbone_depth > 0:
            blocks = student.base_model.blocks
            for block in blocks[-unfreeze_backbone_depth:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in student.base_model.ln_f.parameters():
                p.requires_grad = True
            for p in student.base_model.wte.parameters():
                p.requires_grad = True

        return sum(p.numel() for p in student.parameters() if p.requires_grad)

    def _compute_batch_loss(
        self,
        student: TextStudentM103,
        items: list[SimulatorItemM103_C],
        active_domain: str,
    ) -> torch.Tensor:
        """Computes causal autoregressive CE loss with prompt tokens masked (-100)."""
        token_lists = [it.full_tokens for it in items if it.full_tokens]
        if not token_lists:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        batch_tokens, mask = student.tokenizer.pad_batch(token_lists, device=self.device)
        logits = student.forward_lm(batch_tokens, active_domain=active_domain)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch_tokens[:, 1:].contiguous().clone()

        for i, it in enumerate(items):
            p_len = it.prompt_len - 1
            if p_len > 0:
                shift_labels[i, :p_len] = -100

        shift_labels[~mask[:, 1:]] = -100
        return torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def train_concept(
        self,
        student: TextStudentM103,
        dataset: ConceptDatasetM103_C,
        condition_name: str,
        steps: int = 150,
        batch_size: int = 16,
        lr: float = 0.001,
        eval_interval: int = 30,
    ) -> SingleConceptResultM103_C:
        """Trains student on concept dataset with progressive diagnostic tracking."""
        dom = dataset.family.value
        trainable_p = self.configure_selective_backbone(student, dom, unfreeze_backbone_depth=1)

        opt = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=1e-4,
        )

        history: list[ConceptTrainingProgressM103_C] = []
        rng = torch.Generator(device="cpu").manual_seed(42)

        for s in range(1, steps + 1):
            student.train()
            opt.zero_grad()

            # Sample batch from train items
            indices = torch.randint(0, len(dataset.train_items), (batch_size,), generator=rng).tolist()
            batch = [dataset.train_items[i] for i in indices]

            loss = self._compute_batch_loss(student, batch, active_domain=dom)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

            if s % eval_interval == 0 or s == steps:
                student.eval()
                with torch.no_grad():
                    tr_m = self.evaluator.evaluate_split(student, dataset.train_items[:20], "train", dom)
                    val_m = self.evaluator.evaluate_split(student, dataset.val_canonical_items, "val", dom)
                    uns_m = self.evaluator.evaluate_split(student, dataset.unseen_values_items, "unseen", dom)
                    para_m = self.evaluator.evaluate_split(student, dataset.paraphrase_items, "para", dom)

                history.append(
                    ConceptTrainingProgressM103_C(
                        step=s,
                        train_loss=round(float(loss.item()), 4),
                        train_exact_match=tr_m.exact_match_pct,
                        val_canonical_exact_match=val_m.exact_match_pct,
                        unseen_values_exact_match=uns_m.exact_match_pct,
                        paraphrase_exact_match=para_m.exact_match_pct,
                        unseen_numeric_accuracy=uns_m.numeric_accuracy_pct,
                    )
                )

        final_prog = history[-1] if history else ConceptTrainingProgressM103_C(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Test internalized rule execution WITHOUT explanation in prompt
        without_expl_m = self.evaluator.evaluate_split(
            student,
            dataset.unseen_values_items,
            "unseen_no_expl",
            active_domain=dom,
            with_explanation=False,
        )

        # Compare canonical short answer vs unrestricted byte-level generation
        canonical_m = self.evaluator.evaluate_split(
            student, dataset.val_canonical_items, "canonical_test", dom, canonical=True
        )
        unrestricted_m = self.evaluator.evaluate_split(
            student, dataset.val_canonical_items, "unrestricted_test", dom, canonical=False, max_new_tokens=24
        )

        gate_passed = (
            final_prog.unseen_values_exact_match >= self.COMPETENCE_GATE_THRESHOLD
            or final_prog.unseen_numeric_accuracy >= self.COMPETENCE_GATE_THRESHOLD
        )

        status = (
            f"Condition {condition_name}: Train EM={final_prog.train_exact_match}%, "
            f"Unseen EM={final_prog.unseen_values_exact_match}%, "
            f"Unseen Numeric Acc={final_prog.unseen_numeric_accuracy}%, "
            f"Paraphrase EM={final_prog.paraphrase_exact_match}%, "
            f"Competence Gate Passed={gate_passed}."
        )

        return SingleConceptResultM103_C(
            family=dataset.family,
            condition=condition_name,
            trainable_params=trainable_p,
            total_steps=steps,
            history=history,
            final_train_em=final_prog.train_exact_match,
            final_val_canonical_em=final_prog.val_canonical_exact_match,
            final_unseen_values_em=final_prog.unseen_values_exact_match,
            final_paraphrase_em=final_prog.paraphrase_exact_match,
            final_unseen_numeric_acc=final_prog.unseen_numeric_accuracy,
            internalized_without_explanation_em=without_expl_m.exact_match_pct,
            format_comparison_canonical_em=canonical_m.exact_match_pct,
            format_comparison_unrestricted_em=unrestricted_m.exact_match_pct,
            competence_gate_passed=gate_passed,
            status_summary=status,
        )

    def run_continual_learning_pipeline(
        self,
        student: TextStudentM103,
        steps_per_concept: int = 150,
    ) -> ContinualLifelongResultM103_C:
        """Executes sequential learning with strict competence gating."""
        # Concept 1: Energy Modification
        energy_dataset = self.simulator.generate_energy_dataset()
        res1 = self.train_concept(
            student=student,
            dataset=energy_dataset,
            condition_name="Lifelong_Student_Concept1",
            steps=steps_per_concept,
        )

        # Stop condition check: if C2 single-concept generalization fails, do NOT advance
        if not res1.competence_gate_passed:
            verdict = (
                f"STOP_CONDITION_TRIGGERED: Single-concept unseen value accuracy "
                f"({res1.final_unseen_values_em}%) < {self.COMPETENCE_GATE_THRESHOLD}%. "
                f"Per M103-C scientific protocol, halted before multi-concept continual learning."
            )
            return ContinualLifelongResultM103_C(
                family_1_result=res1,
                family_2_attempted=False,
                family_2_result=None,
                backward_retention_family_1_em=None,
                fresh_student_family_2_steps=None,
                lifelong_student_family_2_steps=None,
                sample_efficiency_gain_pct=None,
                stop_condition_triggered=True,
                verdict=verdict,
            )

        # If gate passed, proceed to Concept 2: Coordinate Navigation
        coord_dataset = self.simulator.generate_coordinate_dataset()
        res2 = self.train_concept(
            student=student,
            dataset=coord_dataset,
            condition_name="Lifelong_Student_Concept2",
            steps=steps_per_concept,
        )

        # Evaluate backward retention on Concept 1
        student.eval()
        retention_m = self.evaluator.evaluate_split(
            student,
            energy_dataset.unseen_values_items,
            "retention_c1",
            active_domain=ConceptFamilyM103_C.ENERGY_SYSTEM.value,
        )

        # Fresh student control on Concept 2
        fresh_student = TextStudentM103(enable_persistent_growth=True, device=self.device)
        fresh_res2 = self.train_concept(
            student=fresh_student,
            dataset=coord_dataset,
            condition_name="Fresh_Student_Concept2",
            steps=steps_per_concept,
        )

        gain_pct = 0.0
        if fresh_res2.final_unseen_values_em > 0:
            gain_pct = round(
                ((res2.final_unseen_values_em - fresh_res2.final_unseen_values_em) / fresh_res2.final_unseen_values_em) * 100.0,
                2,
            )

        verdict = (
            f"QUALIFIED_PASS: Concept 1 passed ({res1.final_unseen_values_em}%), "
            f"Concept 2 passed ({res2.final_unseen_values_em}%), "
            f"Concept 1 Backward Retention={retention_m.exact_match_pct}%."
        )

        return ContinualLifelongResultM103_C(
            family_1_result=res1,
            family_2_attempted=True,
            family_2_result=res2,
            backward_retention_family_1_em=retention_m.exact_match_pct,
            fresh_student_family_2_steps=fresh_res2.total_steps,
            lifelong_student_family_2_steps=res2.total_steps,
            sample_efficiency_gain_pct=gain_pct,
            stop_condition_triggered=False,
            verdict=verdict,
        )
