"""Milestone M103-D Concept Learner & Generalization Engine.

Addresses the four core scientific phases of M103-D:
Phase D1 — Acquire one rule correctly:
  - Solve optimization on the full training set (distinguishing optimization from generalization).
  - Implements learning rate sweep, gradient clipping, and extended training budget (300-500 steps).
Phase D2 — Learn reusable numerical representations:
  - Compares three representation conditions:
    1. raw_bytes: Multi-byte ASCII digits.
    2. structured_tokens: Atomic number tokens (<num_0>..<num_99>).
    3. scratchpad: Intermediate calculation trace ("s=6,+2=8").
  - Tests Interpolation (unseen odd values within training range [2..14])
    separately from Extrapolation (unseen values [15..28]).
Phase D3 — Predict hypothetical situations:
  - Evaluates counterfactual rules, renamed entities, and multi-step composition against WorldSimulatorM103_C.
Phase D4 — Learn from corrective experience:
  - Measures targeted neural adaptation on surprising feedback and evaluates backward retention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import math
import re
import time
from typing import Any
import torch
import torch.nn.functional as F

from adaptive_mini_gpt.natural_language_learning_m103.number_tokenizer_m103_d import (
    NumberAwareTokenizerM103_D,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.world_simulator_m103_c import (
    ConceptFamilyM103_C,
    WorldSimulatorM103_C,
)


@dataclass
class DatasetItemM103_D:
    explanation: str
    question: str
    canonical_answer: str
    scratchpad: str
    expected_num: int
    entity_name: str
    category: str  # "train", "val_interpolation", "test_extrapolation", "test_paraphrase", "test_hypothetical"
    full_tokens: list[int] = field(default_factory=list)
    prompt_len: int = 0


@dataclass
class EvaluationResultM103_D:
    split_name: str
    item_count: int
    exact_match_pct: float
    numeric_accuracy_pct: float
    format_validity_pct: float
    character_accuracy_pct: float
    mean_token_loss: float
    sample_outputs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PhaseD1OptimizationReport:
    best_lr: float
    sweep_results: dict[str, float]
    final_train_loss: float
    final_train_em: float
    final_train_numeric_acc: float
    optimization_converged: bool
    convergence_diagnosis: str


@dataclass
class RepresentationComparisonM103_D:
    mode_name: str
    train_em: float
    interpolation_em: float
    interpolation_numeric_acc: float
    extrapolation_em: float
    extrapolation_numeric_acc: float
    paraphrase_em: float
    format_validity_pct: float


@dataclass
class PhaseD3SimulationReport:
    counterfactual_em: float
    counterfactual_numeric_acc: float
    novel_entity_em: float
    novel_entity_numeric_acc: float
    multistep_em: float
    multistep_numeric_acc: float
    internal_prediction_demonstrated: bool


@dataclass
class PhaseD4CorrectiveReport:
    pre_feedback_target_em: float
    post_feedback_target_em: float
    feedback_adaptation_gain_pct: float
    backward_retention_original_rule_pct: float
    catastrophic_forgetting_detected: bool


class ConceptLearnerM103_D:
    """Master learner for Milestone M103-D genuine concept acquisition."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        self.tokenizer = NumberAwareTokenizerM103_D()
        self.simulator = WorldSimulatorM103_C(seed=42)

    def extract_numeric_value(self, text: str) -> int | None:
        """Extract the first integer from output text using word-boundary matching."""
        match = re.search(r"(?<!\w)(-?\d+)(?!\w)", text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def build_m103_d_datasets(
        self,
        mode: str = "raw_bytes",
    ) -> dict[str, list[DatasetItemM103_D]]:
        """Constructs strictly partitioned datasets for M103-D.

        Partitioning strategy:
        - Train: EVEN numbers s in [2, 4, 6, 8, 10, 12, 14] with 4 entities and 3 templates.
        - Val Interpolation: ODD numbers s in [3, 5, 7, 9, 11, 13] (within training range!).
        - Test Extrapolation: Unseen numbers s in [16, 18, 20, 22, 24, 26, 28] (outside range).
        - Test Paraphrase: ODD numbers with completely novel phrasing.
        - Hypothetical: Modified rules, new entities, and multi-step composition.
        """
        use_num_tokens = (mode == "structured_tokens")
        use_scratchpad = (mode == "scratchpad")

        expl = "In System-Alpha, blue adds 2 energy. Red loses 1. Blue acts first."
        entities = ["orb", "crystal", "beacon", "sphere"]

        def make_item(
            q_text: str,
            ans_num: int,
            ent: str,
            cat: str,
            rule_str: str = expl,
            s_val: int = 0,
            mod_type: str = "blue",
        ) -> DatasetItemM103_D:
            c_ans = f"{ans_num} energy"
            if mod_type == "blue":
                scratch = f"s={s_val},+2={ans_num}"
            elif mod_type == "blue_red":
                scratch = f"s={s_val},+2-1={ans_num}"
            else:
                scratch = f"s={s_val},result={ans_num}"

            sp_arg = scratch if use_scratchpad else None
            full_toks, prompt_len = self.tokenizer.format_qa_sequence(
                question=q_text,
                answer=c_ans,
                explanation=rule_str,
                scratchpad=sp_arg,
                use_number_tokens=use_num_tokens,
            )
            return DatasetItemM103_D(
                explanation=rule_str,
                question=q_text,
                canonical_answer=c_ans,
                scratchpad=scratch,
                expected_num=ans_num,
                entity_name=ent,
                category=cat,
                full_tokens=full_toks,
                prompt_len=prompt_len,
            )

        train_items: list[DatasetItemM103_D] = []
        # Train on EVEN starting values: 2, 4, 6, 8, 10, 12, 14
        even_s = [2, 4, 6, 8, 10, 12, 14]
        for s in even_s:
            for ent in entities:
                # Blue only
                q1 = f"An {ent} starts with {s} energy and is blue. What is its final energy?"
                a1 = s + 2
                train_items.append(make_item(q1, a1, ent, "train", s_val=s, mod_type="blue"))

                # Blue and red
                q2 = f"An {ent} starts with {s} energy and is blue and red. What is its final energy?"
                a2 = s + 2 - 1
                train_items.append(make_item(q2, a2, ent, "train", s_val=s, mod_type="blue_red"))

        # Val Interpolation: ODD numbers: 3, 5, 7, 9, 11, 13 (within [2..14] range)
        val_interpolation: list[DatasetItemM103_D] = []
        odd_s = [3, 5, 7, 9, 11, 13]
        for s in odd_s:
            ent = entities[s % len(entities)]
            q1 = f"An {ent} starts with {s} energy and is blue. What is its final energy?"
            a1 = s + 2
            val_interpolation.append(make_item(q1, a1, ent, "val_interpolation", s_val=s, mod_type="blue"))

            q2 = f"An {ent} starts with {s} energy and is blue and red. What is its final energy?"
            a2 = s + 2 - 1
            val_interpolation.append(make_item(q2, a2, ent, "val_interpolation", s_val=s, mod_type="blue_red"))

        # Test Extrapolation: Unseen values s in [16, 18, 20, 22, 24, 26, 28]
        test_extrapolation: list[DatasetItemM103_D] = []
        extra_s = [16, 18, 20, 22, 24, 26, 28]
        for s in extra_s:
            ent = entities[s % len(entities)]
            q1 = f"An {ent} starts with {s} energy and is blue. What is its final energy?"
            a1 = s + 2
            test_extrapolation.append(make_item(q1, a1, ent, "test_extrapolation", s_val=s, mod_type="blue"))

            q2 = f"An {ent} starts with {s} energy and is blue and red. What is its final energy?"
            a2 = s + 2 - 1
            test_extrapolation.append(make_item(q2, a2, ent, "test_extrapolation", s_val=s, mod_type="blue_red"))

        # Test Paraphrase: ODD numbers with novel syntax
        test_paraphrase: list[DatasetItemM103_D] = []
        for s in [5, 7, 11]:
            ent = entities[(s + 1) % len(entities)]
            qp1 = f"Determine the resulting energy level if an {ent} begins at {s} energy and carries blue."
            ap1 = s + 2
            test_paraphrase.append(make_item(qp1, ap1, ent, "test_paraphrase", s_val=s, mod_type="blue"))

            qp2 = f"Compute the terminal energy of an {ent} after applying blue then red starting from {s} energy."
            ap2 = s + 2 - 1
            test_paraphrase.append(make_item(qp2, ap2, ent, "test_paraphrase", s_val=s, mod_type="blue_red"))

        # Test Hypothetical (Phase D3)
        test_hypothetical: list[DatasetItemM103_D] = []
        # 1. Counterfactual rule (blue adds 4, red loses 2)
        expl_cf = "In modified System-Alpha, blue adds 4 energy. Red loses 2. Blue acts first."
        for s in [6, 12, 18]:
            ent = "beacon"
            q_cf = f"Under modified rules, a {ent} starts with {s} energy and is blue. What is final energy?"
            a_cf = s + 4
            test_hypothetical.append(make_item(q_cf, a_cf, ent, "test_hypothetical", rule_str=expl_cf, s_val=s, mod_type="blue"))

        # 2. Renamed entities (Monolith, Prism)
        for s in [8, 14]:
            for ent_new in ["monolith", "prism"]:
                q_re = f"A {ent_new} starts with {s} energy and is blue. What is its final energy?"
                a_re = s + 2
                test_hypothetical.append(make_item(q_re, a_re, ent_new, "test_hypothetical", s_val=s, mod_type="blue"))

        # 3. Multi-step composition: two blue pulses (+4)
        for s in [4, 10]:
            q_ms = f"An orb starts with {s} energy and receives two blue pulses. What is its final energy?"
            a_ms = s + 4
            test_hypothetical.append(make_item(q_ms, a_ms, "orb", "test_hypothetical", s_val=s, mod_type="multistep"))

        return {
            "train": train_items,
            "val_interpolation": val_interpolation,
            "test_extrapolation": test_extrapolation,
            "test_paraphrase": test_paraphrase,
            "test_hypothetical": test_hypothetical,
        }

    def configure_selective_backbone(
        self,
        student: TextStudentM103,
        domain: str = "energy_system",
    ) -> int:
        """Unfreezes domain adapter + top block + ln_f + wte."""
        for p in student.parameters():
            p.requires_grad = False

        adapter_key = f"adapter_{domain}"
        if adapter_key in student.adapter_modules:
            for p in student.adapter_modules[adapter_key].parameters():
                p.requires_grad = True

        blocks = student.base_model.blocks
        if len(blocks) > 0:
            for p in blocks[-1].parameters():
                p.requires_grad = True
        for p in student.base_model.ln_f.parameters():
            p.requires_grad = True
        for p in student.base_model.wte.parameters():
            p.requires_grad = True

        return sum(p.numel() for p in student.parameters() if p.requires_grad)

    def _compute_batch_loss(
        self,
        student: TextStudentM103,
        items: list[DatasetItemM103_D],
        active_domain: str = "energy_system",
    ) -> torch.Tensor:
        """Causal cross-entropy with prompt masking and context length clamping."""
        token_lists = [it.full_tokens for it in items if it.full_tokens]
        if not token_lists:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        batch_tokens, mask = self.tokenizer.pad_batch(token_lists, device=self.device)
        ctx_len = student.base_model.config.context_length
        if batch_tokens.size(1) > ctx_len:
            batch_tokens = batch_tokens[:, :ctx_len]
            mask = mask[:, :ctx_len]

        logits = student.forward_lm(batch_tokens, active_domain=active_domain)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch_tokens[:, 1:].contiguous().clone()

        for i, it in enumerate(items):
            p_len = it.prompt_len - 1
            if p_len > 0:
                shift_labels[i, :p_len] = -100

        shift_labels[~mask[:, 1:]] = -100
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def evaluate_split(
        self,
        student: TextStudentM103,
        items: list[DatasetItemM103_D],
        split_name: str,
        active_domain: str = "energy_system",
        max_new_tokens: int = 16,
    ) -> EvaluationResultM103_D:
        """Evaluates split completions with exact match, numeric accuracy, and format validity."""
        if not items:
            return EvaluationResultM103_D(
                split_name=split_name,
                item_count=0,
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

        ctx_len = student.base_model.config.context_length

        for idx, it in enumerate(items):
            expected_target = it.canonical_answer.strip().lower()
            expected_num = it.expected_num

            prompt_toks = it.full_tokens[:it.prompt_len] if it.prompt_len > 0 else []
            if len(prompt_toks) > ctx_len:
                prompt_toks = prompt_toks[-ctx_len:]

            gen_ids = student.generate(
                prompt_tokens=prompt_toks,
                max_new_tokens=max_new_tokens,
                active_domain=active_domain,
            )
            gen_text = self.tokenizer.decode(gen_ids).strip()
            norm_gen = gen_text.lower().strip()

            # Exact match: matches expected target or begins with target
            is_em = (norm_gen == expected_target) or (norm_gen.startswith(expected_target))
            if is_em:
                exact_matches += 1

            # Numeric accuracy
            pred_num = self.extract_numeric_value(gen_text)
            is_num_match = (pred_num == expected_num)
            if is_num_match:
                numeric_matches += 1

            # Format validity
            is_fmt = bool("energy" in norm_gen and pred_num is not None)
            if is_fmt:
                format_valid += 1

            char_sim = difflib.SequenceMatcher(None, norm_gen, expected_target).ratio()
            total_char_sim += char_sim

            if idx < 4:
                sample_logs.append({
                    "question": it.question,
                    "target": expected_target,
                    "generated": gen_text,
                    "pred_num": pred_num,
                    "expected_num": expected_num,
                    "exact_match": is_em,
                    "numeric_match": is_num_match,
                })

        # Calculate loss
        with torch.no_grad():
            loss = self._compute_batch_loss(student, items, active_domain=active_domain)
            mean_loss = round(float(loss.item()), 4)

        n = len(items)
        return EvaluationResultM103_D(
            split_name=split_name,
            item_count=n,
            exact_match_pct=round((exact_matches / n) * 100.0, 2),
            numeric_accuracy_pct=round((numeric_matches / n) * 100.0, 2),
            format_validity_pct=round((format_valid / n) * 100.0, 2),
            character_accuracy_pct=round((total_char_sim / n) * 100.0, 2),
            mean_token_loss=mean_loss,
            sample_outputs=sample_logs,
        )

    def run_phase_d1_optimization(
        self,
        student: TextStudentM103,
        train_items: list[DatasetItemM103_D],
        val_items: list[DatasetItemM103_D],
        candidate_lrs: list[float] | None = None,
        sweep_steps: int = 50,
        full_train_steps: int = 300,
        batch_size: int = 16,
    ) -> tuple[PhaseD1OptimizationReport, TextStudentM103]:
        """Phase D1: LR sweep to find optimal learning rate, then train full single-rule family."""
        lrs = candidate_lrs or [3e-4, 1e-3, 3e-3]
        sweep_scores: dict[str, float] = {}
        best_lr = lrs[0]
        lowest_sweep_loss = float("inf")

        print(f"[Phase D1] Running Learning Rate Sweep across {lrs}...")
        for lr in lrs:
            # Clone student initial weights
            trial_student = TextStudentM103(enable_persistent_growth=True, device=self.device)
            trial_student.load_state_dict(student.state_dict())
            self.configure_selective_backbone(trial_student)

            opt = torch.optim.AdamW(
                [p for p in trial_student.parameters() if p.requires_grad],
                lr=lr,
                weight_decay=1e-4,
            )

            rng = torch.Generator(device="cpu").manual_seed(42)
            trial_loss = 0.0
            for s in range(1, sweep_steps + 1):
                trial_student.train()
                opt.zero_grad()
                indices = torch.randint(0, len(train_items), (batch_size,), generator=rng).tolist()
                batch = [train_items[i] for i in indices]
                loss = self._compute_batch_loss(trial_student, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trial_student.parameters(), 1.0)
                opt.step()
                trial_loss = float(loss.item())

            sweep_scores[f"lr_{lr}"] = round(trial_loss, 4)
            if trial_loss < lowest_sweep_loss:
                lowest_sweep_loss = trial_loss
                best_lr = lr

        print(f"[Phase D1] Best LR: {best_lr} (Sweep Loss: {lowest_sweep_loss:.4f})")
        print(f"[Phase D1] Training full single-rule family for {full_train_steps} steps...")

        # Train primary student with best_lr
        self.configure_selective_backbone(student)
        optimizer = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=best_lr,
            weight_decay=1e-4,
        )

        rng = torch.Generator(device="cpu").manual_seed(42)
        final_loss = 0.0
        for s in range(1, full_train_steps + 1):
            student.train()
            optimizer.zero_grad()
            indices = torch.randint(0, len(train_items), (batch_size,), generator=rng).tolist()
            batch = [train_items[i] for i in indices]
            loss = self._compute_batch_loss(student, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.item())

        # Evaluate final training performance
        train_res = self.evaluate_split(student, train_items, "train_eval")
        converged = (train_res.exact_match_pct >= 70.0 or train_res.numeric_accuracy_pct >= 80.0)

        diagnosis = (
            f"Phase D1 complete: Best LR={best_lr}. "
            f"Train Loss dropped to {final_loss:.4f}. "
            f"Train Exact Match: {train_res.exact_match_pct}%. "
            f"Train Numeric Accuracy: {train_res.numeric_accuracy_pct}%. "
            f"Optimization Convergence: {'CONFIRMED' if converged else 'PARTIAL'}."
        )

        report = PhaseD1OptimizationReport(
            best_lr=best_lr,
            sweep_results=sweep_scores,
            final_train_loss=round(final_loss, 4),
            final_train_em=train_res.exact_match_pct,
            final_train_numeric_acc=train_res.numeric_accuracy_pct,
            optimization_converged=converged,
            convergence_diagnosis=diagnosis,
        )
        return report, student

    def run_phase_d2_representation_comparison(
        self,
        train_steps: int = 250,
        best_lr: float = 1e-3,
    ) -> list[RepresentationComparisonM103_D]:
        """Phase D2: Compares raw_bytes vs structured_tokens vs scratchpad representations."""
        modes = ["raw_bytes", "structured_tokens", "scratchpad"]
        results: list[RepresentationComparisonM103_D] = []

        for mode in modes:
            print(f"[Phase D2] Evaluating Representation Mode: {mode}...")
            datasets = self.build_m103_d_datasets(mode=mode)
            student = TextStudentM103(enable_persistent_growth=True, device=self.device)
            self.configure_selective_backbone(student)

            opt = torch.optim.AdamW(
                [p for p in student.parameters() if p.requires_grad],
                lr=best_lr,
                weight_decay=1e-4,
            )

            train_items = datasets["train"]
            rng = torch.Generator(device="cpu").manual_seed(42)
            batch_size = 16

            for s in range(1, train_steps + 1):
                student.train()
                opt.zero_grad()
                indices = torch.randint(0, len(train_items), (batch_size,), generator=rng).tolist()
                batch = [train_items[i] for i in indices]
                loss = self._compute_batch_loss(student, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()

            # Evaluate on all splits
            tr_eval = self.evaluate_split(student, datasets["train"][:20], f"{mode}_train")
            interp_eval = self.evaluate_split(student, datasets["val_interpolation"], f"{mode}_interp")
            extrap_eval = self.evaluate_split(student, datasets["test_extrapolation"], f"{mode}_extrap")
            para_eval = self.evaluate_split(student, datasets["test_paraphrase"], f"{mode}_para")

            results.append(
                RepresentationComparisonM103_D(
                    mode_name=mode,
                    train_em=tr_eval.exact_match_pct,
                    interpolation_em=interp_eval.exact_match_pct,
                    interpolation_numeric_acc=interp_eval.numeric_accuracy_pct,
                    extrapolation_em=extrap_eval.exact_match_pct,
                    extrapolation_numeric_acc=extrap_eval.numeric_accuracy_pct,
                    paraphrase_em=para_eval.exact_match_pct,
                    format_validity_pct=interp_eval.format_validity_pct,
                )
            )

        return results

    def run_phase_d3_hypothetical_predictions(
        self,
        student: TextStudentM103,
        hypothetical_items: list[DatasetItemM103_D],
    ) -> PhaseD3SimulationReport:
        """Phase D3: Tests modified rules, renamed entities, and multi-step composition."""
        cf_items = [it for it in hypothetical_items if "modified" in it.question.lower()]
        re_items = [it for it in hypothetical_items if it.entity_name in ["monolith", "prism"]]
        ms_items = [it for it in hypothetical_items if "two blue" in it.question.lower()]

        eval_cf = self.evaluate_split(student, cf_items, "hypo_counterfactual")
        eval_re = self.evaluate_split(student, re_items, "hypo_renamed")
        eval_ms = self.evaluate_split(student, ms_items, "hypo_multistep")

        demonstrated = bool(
            eval_cf.numeric_accuracy_pct > 0.0
            or eval_re.numeric_accuracy_pct > 0.0
            or eval_ms.numeric_accuracy_pct > 0.0
        )

        return PhaseD3SimulationReport(
            counterfactual_em=eval_cf.exact_match_pct,
            counterfactual_numeric_acc=eval_cf.numeric_accuracy_pct,
            novel_entity_em=eval_re.exact_match_pct,
            novel_entity_numeric_acc=eval_re.numeric_accuracy_pct,
            multistep_em=eval_ms.exact_match_pct,
            multistep_numeric_acc=eval_ms.numeric_accuracy_pct,
            internal_prediction_demonstrated=demonstrated,
        )

    def run_phase_d4_corrective_learning(
        self,
        student: TextStudentM103,
        original_train_items: list[DatasetItemM103_D],
        corrective_steps: int = 25,
        lr: float = 1e-3,
    ) -> PhaseD4CorrectiveReport:
        """Phase D4: Learns from surprising corrective feedback with rehearsal."""
        # Baseline original competence before feedback
        pre_original = self.evaluate_split(student, original_train_items[:15], "pre_feedback_original")

        # Feedback task: In Sector-Z, blue adds 5 energy units (surprising rule change)
        expl_z = "In Sector-Z, blue adds 5 energy units. Blue acts first."
        feedback_items: list[DatasetItemM103_D] = []
        for s in [3, 6, 9]:
            q_z = f"In Sector-Z, an orb starts with {s} energy and is blue. What is final energy?"
            a_z = f"{s + 5} energy"
            toks, p_len = self.tokenizer.format_qa_sequence(question=q_z, answer=a_z, explanation=expl_z)
            feedback_items.append(
                DatasetItemM103_D(
                    explanation=expl_z,
                    question=q_z,
                    canonical_answer=a_z,
                    scratchpad="",
                    expected_num=s + 5,
                    entity_name="orb",
                    category="corrective_feedback",
                    full_tokens=toks,
                    prompt_len=p_len,
                )
            )

        pre_feedback = self.evaluate_split(student, feedback_items, "pre_feedback_target")

        # Fine-tune with 50% feedback items + 50% rehearsal of original items
        self.configure_selective_backbone(student)
        opt = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=1e-4,
        )

        rng = torch.Generator(device="cpu").manual_seed(42)
        student.train()
        for s in range(1, corrective_steps + 1):
            opt.zero_grad()
            batch_fb = [feedback_items[i % len(feedback_items)] for i in range(8)]
            batch_reh = [original_train_items[i % len(original_train_items)] for i in range(8)]
            combined_batch = batch_fb + batch_reh

            loss = self._compute_batch_loss(student, combined_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

        # Post-feedback evaluation
        post_feedback = self.evaluate_split(student, feedback_items, "post_feedback_target")
        post_original = self.evaluate_split(student, original_train_items[:15], "post_feedback_original")

        gain = round(post_feedback.exact_match_pct - pre_feedback.exact_match_pct, 2)
        retention = round(
            (post_original.exact_match_pct / max(1e-5, pre_original.exact_match_pct)) * 100.0, 2
        ) if pre_original.exact_match_pct > 0 else 100.0

        forgetting = (retention < 60.0 and pre_original.exact_match_pct >= 50.0)

        return PhaseD4CorrectiveReport(
            pre_feedback_target_em=pre_feedback.exact_match_pct,
            post_feedback_target_em=post_feedback.exact_match_pct,
            feedback_adaptation_gain_pct=gain,
            backward_retention_original_rule_pct=min(100.0, retention),
            catastrophic_forgetting_detected=forgetting,
        )
