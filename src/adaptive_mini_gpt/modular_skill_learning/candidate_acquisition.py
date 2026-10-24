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
    is_qualified: bool
    diagnostics: str


class ControlledSkillAcquisitionEngine:
    """Manages bounded candidate skill creation, training, and independent qualification."""

    def __init__(
        self,
        d_model: int = 128,
        bottleneck_dim: int = 16,
        num_classes: int = 3,
        lr: float = 5e-4,
        max_steps: int = 40,
        qualification_threshold_f1: float = 0.50,
        min_improvement_margin: float = 0.05,
    ) -> None:
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes
        self.lr = lr
        self.max_steps = max_steps
        self.qualification_threshold_f1 = qualification_threshold_f1
        self.min_improvement_margin = min_improvement_margin

    def train_candidate_adapter(
        self,
        model: MiniGPT,
        env: GroundedThreeValuedEnvironment,
        budget_n: int,
        device: torch.device,
        seed: int = 42,
    ) -> tuple[Rank16ResidualAdapter, float, float]:
        """Train a fresh rank-16 candidate adapter on N verified practice examples."""
        torch.manual_seed(seed)
        candidate = Rank16ResidualAdapter(
            d_model=self.d_model, bottleneck_dim=self.bottleneck_dim, num_classes=self.num_classes
        ).to(device)

        # Freeze base model completely
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(candidate.parameters(), lr=self.lr, weight_decay=1e-4)
        label_to_idx = {"NO": 0, "YES": 1, "UNKNOWN": 2}

        # Select budget N practice examples
        practice_queries = env.held_out_test_queries[:budget_n]
        if not practice_queries:
            return candidate, 0.0, 0.0

        encoded_dataset = []
        for q in practice_queries:
            tokens = model.tokenizer.encode(q.question_text)
            target = label_to_idx.get(q.expected_answer, 2)
            encoded_dataset.append((tokens, target))

        candidate.train()
        final_loss = 0.0
        correct_tokens = 0
        total_tokens = 0

        for step in range(self.max_steps):
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
                    out = model(inp)
                    hidden = out.hidden_states if hasattr(out, "hidden_states") else out.last_hidden_state
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

    def evaluate_on_independent_world(
        self,
        candidate: nn.Module,
        model: MiniGPT,
        eval_env: GroundedThreeValuedEnvironment,
        device: torch.device,
    ) -> tuple[float, float, dict[str, float], float]:
        """Evaluate candidate on an independent held-out evaluation world."""
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
                out = model(inp)
                hidden = out.hidden_states if hasattr(out, "hidden_states") else out.last_hidden_state
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
                h1 = model(torch.tensor([t1], dtype=torch.long, device=device))
                h2 = model(torch.tensor([t2], dtype=torch.long, device=device))
                hd1 = h1.hidden_states if hasattr(h1, "hidden_states") else h1.last_hidden_state
                hd2 = h2.hidden_states if hasattr(h2, "hidden_states") else h2.last_hidden_state
                p1 = idx_to_label.get(candidate(hd1).argmax(dim=-1).item())
                p2 = idx_to_label.get(candidate(hd2).argmax(dim=-1).item())
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
    ) -> tuple[CandidateAcquisitionReport, Rank16ResidualAdapter | None]:
        """Train candidate, independently qualify it against existing repertoire, and retain or roll back."""
        # Check performance of best existing skill on eval_env
        best_existing_f1 = 0.0
        for sk_id in repertoire.get_skill_ids():
            sk_f1, _, _, _ = repertoire.evaluate_skill_on_env(sk_id, model, eval_env, device)
            if sk_f1 > best_existing_f1:
                best_existing_f1 = sk_f1

        # Train candidate
        candidate, train_loss, train_acc = self.train_candidate_adapter(
            model=model, env=train_env, budget_n=budget_n, device=device, seed=seed
        )

        # Independent qualification on separate evaluation environment
        eval_f1, eval_acc, recalls, paired = self.evaluate_on_independent_world(
            candidate=candidate, model=model, eval_env=eval_env, device=device
        )

        gain = eval_f1 - best_existing_f1
        # Qualified iff meets absolute threshold or beats best existing by margin
        is_qualified = (
            (eval_f1 >= self.qualification_threshold_f1)
            and (gain >= self.min_improvement_margin)
            and (recalls.get("UNKNOWN", 0.0) >= 0.20)  # Must predict UNKNOWN meaningfully
        )

        if is_qualified:
            # Register into research repertoire
            repertoire.register_skill(
                skill_id=candidate_id,
                rule_family=rule_family,
                source_env_id=train_env.env_id,
                adapter=candidate,
                macro_f1=eval_f1,
                accuracy=eval_acc,
                classwise_recall=recalls,
            )
            diag = (
                f"QUALIFIED: Candidate '{candidate_id}' achieved F1={eval_f1:.4f} "
                f"(gain +{gain:.4f} over best existing {best_existing_f1:.4f}). Retained for research reuse."
            )
            retained_candidate = candidate
        else:
            diag = (
                f"REJECTED: Candidate '{candidate_id}' achieved F1={eval_f1:.4f} "
                f"(gain {gain:.4f} vs threshold {self.qualification_threshold_f1:.4f}). Rolled back."
            )
            retained_candidate = None

        report = CandidateAcquisitionReport(
            candidate_id=candidate_id,
            target_env_id=train_env.env_id,
            rule_family=rule_family,
            budget_n=budget_n,
            train_steps=self.max_steps,
            train_loss=float(train_loss),
            train_accuracy=float(train_acc),
            independent_eval_macro_f1=float(eval_f1),
            independent_eval_accuracy=float(eval_acc),
            independent_eval_recalls=recalls,
            paired_correctness=float(paired),
            best_existing_f1=float(best_existing_f1),
            f1_gain_over_best_existing=float(gain),
            is_qualified=is_qualified,
            diagnostics=diag,
        )
        return report, retained_candidate
