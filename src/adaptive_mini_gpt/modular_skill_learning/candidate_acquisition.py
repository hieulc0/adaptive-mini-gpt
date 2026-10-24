"""Phase 4 & 5: Controlled Research Candidate Skill Acquisition & Independent Qualification."""
from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.modular_skill_learning.hidden_extraction import extract_model_hidden
from adaptive_mini_gpt.modular_skill_learning.repertoire import (
    ModularSkillRepertoire,
    QualifiedSkillRecord,
)
from adaptive_mini_gpt.three_valued_reasoning.architectures import Rank16ResidualAdapter
from adaptive_mini_gpt.three_valued_reasoning.semantics import GroundedThreeValuedEnvironment

logger = logging.getLogger(__name__)


@dataclass
class CandidateAcquisitionReport:
    """Detailed audit of candidate adapter training and qualification outcomes."""
    candidate_id: str
    target_env_id: str
    rule_family: str
    budget_n: int
    train_steps: int
    train_loss: float
    train_accuracy: float
    independent_eval_macro_f1: float
    independent_eval_accuracy: float
    independent_eval_recalls: dict[str, float]
    paired_correctness: float
    best_existing_f1: float
    f1_gain_over_best_existing: float
    qualification_passed: bool
    retained_in_research_repertoire: bool
    parameter_footprint: int
    diagnostics: str
    candidate: Rank16ResidualAdapter | None = field(default=None, repr=False)

    def __iter__(self):
        """Allow unpacking: report, candidate = engine.acquire_and_qualify_candidate(...)."""
        return iter((self, self.candidate))


class ControlledSkillAcquisitionEngine:
    """Trains compact rank-16 candidate adapters and qualifies them on independent held-out worlds."""

    def __init__(
        self,
        d_model: int = 128,
        bottleneck_dim: int = 16,
        num_classes: int = 3,
        lr: float = 5e-4,
        max_steps: int = 40,
        qualification_threshold_f1: float = 0.40,
    ) -> None:
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes
        self.lr = lr
        self.max_steps = max_steps
        self.qualification_threshold_f1 = qualification_threshold_f1

    def train_candidate_adapter(
        self,
        candidate_id: str,
        train_env: GroundedThreeValuedEnvironment,
        model: MiniGPT,
        budget_n: int,
        device: torch.device,
        seed: int = 42,
    ) -> tuple[Rank16ResidualAdapter, float, float]:
        """Train a candidate rank-16 adapter from fresh random initialization on budget N examples."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        candidate = Rank16ResidualAdapter(
            d_model=self.d_model, bottleneck_dim=self.bottleneck_dim, num_classes=self.num_classes
        ).to(device)

        optimizer = torch.optim.AdamW(candidate.parameters(), lr=self.lr, weight_decay=1e-4)

        # Verified practice examples limited to budget N
        practice_queries = train_env.held_out_test_queries[:budget_n]
        label_to_idx = {"NO": 0, "YES": 1, "UNKNOWN": 2}

        encoded_dataset = []
        for q in practice_queries:
            tokens = model.tokenizer.encode(q.question_text)
            tgt = label_to_idx.get(q.expected_answer, 2)
            encoded_dataset.append((tokens, tgt))

        if not encoded_dataset:
            return candidate, 0.0, 0.0

        candidate.train()
        final_loss = 0.0
        correct_tokens = 0
        total_tokens = 0

        steps = min(self.max_steps, max(10, budget_n * 2))
        for step in range(steps):
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            batch_correct = 0

            # Mini-batch of size min(4, len)
            batch_size = min(4, len(encoded_dataset))
            indices = np.random.choice(len(encoded_dataset), size=batch_size, replace=True)

            for idx in indices:
                tokens, tgt = encoded_dataset[idx]
                inp = torch.tensor([tokens], dtype=torch.long, device=device)
                with torch.no_grad():
                    hidden = extract_model_hidden(model, inp)
                logits = candidate(hidden)
                loss = F.cross_entropy(logits, torch.tensor([tgt], dtype=torch.long, device=device))
                batch_loss = batch_loss + loss
                pred = logits.argmax(dim=-1).item()
                if pred == tgt:
                    batch_correct += 1

            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), max_norm=1.0)
            optimizer.step()

            final_loss = batch_loss.item()
            correct_tokens = batch_correct
            total_tokens = batch_size

        train_acc = float(correct_tokens / total_tokens) if total_tokens > 0 else 0.0
        return candidate, final_loss, train_acc

    def evaluate_candidate_on_independent_world(
        self,
        candidate: Rank16ResidualAdapter,
        eval_env: GroundedThreeValuedEnvironment,
        model: MiniGPT,
        device: torch.device,
    ) -> tuple[float, float, dict[str, float], float]:
        """Independently evaluate candidate adapter on a brand new, unseen environment graph."""
        candidate.eval()
        label_to_idx = {"NO": 0, "YES": 1, "UNKNOWN": 2}
        idx_to_label = {0: "NO", 1: "YES", 2: "UNKNOWN"}

        conf_mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        correct_count = 0
        total = len(eval_env.held_out_test_queries)

        with torch.no_grad():
            for q in eval_env.held_out_test_queries:
                tokens = model.tokenizer.encode(q.question_text)
                inp = torch.tensor([tokens], dtype=torch.long, device=device)
                hidden = extract_model_hidden(model, inp)
                pred_idx = candidate(hidden).argmax(dim=-1).item()
                true_idx = label_to_idx.get(q.expected_answer, 2)
                conf_mat[true_idx][pred_idx] += 1
                if pred_idx == true_idx:
                    correct_count += 1

        recalls = {}
        f1_list = []
        for c, name in idx_to_label.items():
            tp = conf_mat[c][c]
            fp = sum(conf_mat[r][c] for r in range(3) if r != c)
            fn = sum(conf_mat[c][p] for p in range(3) if p != c)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            recalls[name] = float(rec)
            f1_list.append(f1)

        macro_f1 = float(np.mean(f1_list))
        acc = float(correct_count / total) if total > 0 else 0.0

        # Paired correctness
        paired_correct = 0
        for mp in eval_env.minimal_pairs:
            t1 = model.tokenizer.encode(mp.base_query.question_text)
            t2 = model.tokenizer.encode(mp.perturbed_query.question_text)
            with torch.no_grad():
                h1 = extract_model_hidden(model, torch.tensor([t1], dtype=torch.long, device=device))
                h2 = extract_model_hidden(model, torch.tensor([t2], dtype=torch.long, device=device))
                p1 = idx_to_label.get(candidate(h1).argmax(dim=-1).item())
                p2 = idx_to_label.get(candidate(h2).argmax(dim=-1).item())
                if p1 == mp.base_query.expected_answer and p2 == mp.perturbed_query.expected_answer:
                    paired_correct += 1
        paired_rate = float(paired_correct / len(eval_env.minimal_pairs)) if eval_env.minimal_pairs else 0.0

        return macro_f1, acc, recalls, paired_rate

    def acquire_and_qualify_candidate(
        self,
        candidate_id: str,
        rule_family: str,
        train_env: GroundedThreeValuedEnvironment,
        eval_env: GroundedThreeValuedEnvironment,
        model: MiniGPT,
        repertoire: ModularSkillRepertoire,
        budget_n: int,
        device: torch.device,
        seed: int = 42,
    ) -> CandidateAcquisitionReport:
        """Complete workflow: acquire candidate, independently qualify, and retain or discard."""
        # 1. Measure best existing performance in repertoire on evaluation environment
        best_existing_f1 = 0.0
        for sid in repertoire.list_skills():
            f1, _, _, _ = repertoire.evaluate_skill_on_environment(sid, model, eval_env, device)
            if f1 > best_existing_f1:
                best_existing_f1 = f1

        # 2. Train candidate adapter on verified examples
        candidate, loss, tr_acc = self.train_candidate_adapter(
            candidate_id=candidate_id,
            train_env=train_env,
            model=model,
            budget_n=budget_n,
            device=device,
            seed=seed,
        )

        # 3. Independent qualification on unseen world
        cand_f1, cand_acc, cand_recs, paired = self.evaluate_candidate_on_independent_world(
            candidate=candidate,
            eval_env=eval_env,
            model=model,
            device=device,
        )

        f1_gain = cand_f1 - best_existing_f1

        # Qualification rule: Must beat best existing skill AND meet minimum threshold
        qualification_passed = bool(cand_f1 >= self.qualification_threshold_f1 and f1_gain >= 0.0)

        retained = False
        if qualification_passed:
            repertoire.register_skill(
                skill_id=candidate_id,
                rule_family=rule_family,
                source_env_id=train_env.env_id,
                adapter=candidate,
                macro_f1=cand_f1,
                accuracy=cand_acc,
                classwise_recall=cand_recs,
            )
            retained = True
            diag = f"QUALIFIED: Candidate achieved F1={cand_f1:.4f} (gain: +{f1_gain:.4f} over best existing {best_existing_f1:.4f}). Retained in research repertoire."
        else:
            diag = f"REJECTED: Candidate achieved F1={cand_f1:.4f} (best existing: {best_existing_f1:.4f}, threshold: {self.qualification_threshold_f1}). Discarded."

        return CandidateAcquisitionReport(
            candidate_id=candidate_id,
            target_env_id=eval_env.env_id,
            rule_family=rule_family,
            budget_n=budget_n,
            train_steps=min(self.max_steps, max(10, budget_n * 2)),
            train_loss=float(loss),
            train_accuracy=float(tr_acc),
            independent_eval_macro_f1=float(cand_f1),
            independent_eval_accuracy=float(cand_acc),
            independent_eval_recalls=cand_recs,
            paired_correctness=float(paired),
            best_existing_f1=float(best_existing_f1),
            f1_gain_over_best_existing=float(f1_gain),
            qualification_passed=qualification_passed,
            retained_in_research_repertoire=retained,
            parameter_footprint=candidate.count_parameters(),
            diagnostics=diag,
            candidate=candidate,
        )
