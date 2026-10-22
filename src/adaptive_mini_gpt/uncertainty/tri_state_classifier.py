"""Phase 3: Minimal Neural Uncertainty Experiment and Epistemic Calibration.

Implements and evaluates three-class uncertainty heads (YES, NO, UNKNOWN) comparing:
  - Arm A: M44 neural baseline (causal LM logit scoring with LoRA adapter).
  - Arm B: Balanced three-class neural head (384-parameter linear head trained with balanced cross-entropy).
  - Arm C: Three-class head with structural analogy supervision (experience-guided calibration).
  - Arm D: External verified graph reasoner (deterministic oracle ceiling).
  - Baselines: Constant-YES and Constant-UNKNOWN.

Measures:
  - Three-class macro F1, per-class precision, recall, and F1.
  - Full 3x3 confusion matrix.
  - Confidence calibration: Expected Calibration Error (ECE) and Brier Score.
  - False Confident Rate (Type I Epistemic Error on UNKNOWN queries).
  - Selective accuracy vs coverage curve.
"""
from __future__ import annotations

import collections
import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.modular_adapter import (
    add_domain_adapter,
    freeze_for_adapter_training,
    inject_multi_domain_adapters,
    set_active_domain_adapter,
)
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.uncertainty.epistemic_benchmark import EpistemicTask, EpistemicWorldDataset

logger = logging.getLogger("m45_tri_state")


@dataclass
class TriStateConfusionMatrix:
    true_yes_pred_yes: int
    true_yes_pred_no: int
    true_yes_pred_unk: int
    true_no_pred_yes: int
    true_no_pred_no: int
    true_no_pred_unk: int
    true_unk_pred_yes: int
    true_unk_pred_no: int
    true_unk_pred_unk: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "TRUE_YES": {"PRED_YES": self.true_yes_pred_yes, "PRED_NO": self.true_yes_pred_no, "PRED_UNK": self.true_yes_pred_unk},
            "TRUE_NO": {"PRED_YES": self.true_no_pred_yes, "PRED_NO": self.true_no_pred_no, "PRED_UNK": self.true_no_pred_unk},
            "TRUE_UNK": {"PRED_YES": self.true_unk_pred_yes, "PRED_NO": self.true_unk_pred_no, "PRED_UNK": self.true_unk_pred_unk},
        }


@dataclass
class EpistemicEvaluationMetrics:
    arm_name: str
    total_samples: int
    overall_accuracy: float
    macro_f1: float
    yes_precision: float
    yes_recall: float
    yes_f1: float
    no_precision: float
    no_recall: float
    no_f1: float
    unknown_precision: float
    unknown_recall: float
    unknown_f1: float
    expected_calibration_error: float
    brier_score: float
    false_confident_rate: float  # Fraction of UNKNOWN items asserted as YES or NO with confidence >= 0.50
    confusion_matrix: TriStateConfusionMatrix

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "total_samples": self.total_samples,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "yes_metrics": {"precision": round(self.yes_precision, 4), "recall": round(self.yes_recall, 4), "f1": round(self.yes_f1, 4)},
            "no_metrics": {"precision": round(self.no_precision, 4), "recall": round(self.no_recall, 4), "f1": round(self.no_f1, 4)},
            "unknown_metrics": {"precision": round(self.unknown_precision, 4), "recall": round(self.unknown_recall, 4), "f1": round(self.unknown_f1, 4)},
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "brier_score": round(self.brier_score, 4),
            "false_confident_rate": round(self.false_confident_rate, 4),
            "confusion_matrix": self.confusion_matrix.to_dict(),
        }


class TriStateEpistemicHead(nn.Module):
    """Minimal 3-class classification head attached to MiniGPT token representations."""

    def __init__(self, d_model: int = 128, num_classes: int = 3) -> None:
        super().__init__()
        self.head = nn.Linear(d_model, num_classes, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # hidden_state: [batch_size, seq_len, d_model] -> select last token [batch_size, d_model]
        last_hidden = hidden_state[:, -1, :]
        return self.head(last_hidden)


class EpistemicUncertaintyEvaluator:
    """Trains and evaluates epistemic uncertainty across the 4 experimental arms."""

    CLASSES = ["YES", "NO", "UNKNOWN"]
    CLASS_TO_IDX = {"YES": 0, "NO": 1, "UNKNOWN": 2}

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: BPETokenizer,
        device: torch.device,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.cand_seqs = [self.tokenizer.encode(f" {c}") for c in self.CLASSES]

    def _extract_hidden_and_logits(
        self,
        model: MiniGPT,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs forward pass through Transformer blocks and extracts hidden states."""
        b, t = input_ids.size()
        tok_emb = model.wte(input_ids)
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device)
        pos_emb = model.wpe(pos)
        x = model.drop(tok_emb + pos_emb)

        for block in model.blocks:
            x, _ = block(x)
        x = model.ln_f(x)
        logits = model.lm_head(x)
        return x, logits

    def _compute_metrics(
        self,
        arm_name: str,
        predictions: list[tuple[str, str, float, list[float]]],  # (true, pred, conf, probs)
    ) -> EpistemicEvaluationMetrics:
        cm_counts = collections.defaultdict(int)
        confidences: list[float] = []
        correctness: list[int] = []
        prob_matrix: list[list[float]] = []
        target_one_hot: list[list[float]] = []

        total = len(predictions)
        for true_v, pred_v, conf, probs in predictions:
            cm_counts[f"{true_v}_{pred_v}"] += 1
            confidences.append(conf)
            is_correct = 1 if true_v == pred_v else 0
            correctness.append(is_correct)
            prob_matrix.append(probs)

            one_hot = [0.0, 0.0, 0.0]
            one_hot[self.CLASS_TO_IDX[true_v]] = 1.0
            target_one_hot.append(one_hot)

        overall_acc = sum(correctness) / max(1, total)

        # Per-class metrics
        def pr_rec_f1(cls_name: str) -> tuple[float, float, float]:
            tp = cm_counts[f"{cls_name}_{cls_name}"]
            fp = sum(cm_counts[f"{other}_{cls_name}"] for other in self.CLASSES if other != cls_name)
            fn = sum(cm_counts[f"{cls_name}_{other}"] for other in self.CLASSES if other != cls_name)
            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            f1 = 2 * p * r / max(1e-6, p + r)
            return p, r, f1

        y_p, y_r, y_f1 = pr_rec_f1("YES")
        n_p, n_r, n_f1 = pr_rec_f1("NO")
        u_p, u_r, u_f1 = pr_rec_f1("UNKNOWN")
        macro_f1 = (y_f1 + n_f1 + u_f1) / 3.0

        # Expected Calibration Error (ECE) with 10 bins
        num_bins = 10
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        conf_arr = np.array(confidences)
        corr_arr = np.array(correctness)

        for i in range(num_bins):
            b_min, b_max = bin_edges[i], bin_edges[i + 1]
            idx = (conf_arr >= b_min) & (conf_arr < b_max if i < num_bins - 1 else conf_arr <= b_max)
            if np.sum(idx) > 0:
                bin_acc = np.mean(corr_arr[idx])
                bin_conf = np.mean(conf_arr[idx])
                bin_weight = np.sum(idx) / total
                ece += bin_weight * abs(bin_acc - bin_conf)

        # Brier Score
        probs_np = np.array(prob_matrix)
        targets_np = np.array(target_one_hot)
        brier = float(np.mean(np.sum((probs_np - targets_np) ** 2, axis=1)))

        # False Confident Rate on UNKNOWN: fraction of true UNKNOWN items predicted as YES or NO with conf >= 0.50
        unk_preds = [p for p in predictions if p[0] == "UNKNOWN"]
        false_confident = sum(1 for p in unk_preds if p[1] in ("YES", "NO") and p[2] >= 0.50)
        fc_rate = false_confident / max(1, len(unk_preds))

        cm = TriStateConfusionMatrix(
            true_yes_pred_yes=cm_counts["YES_YES"],
            true_yes_pred_no=cm_counts["YES_NO"],
            true_yes_pred_unk=cm_counts["YES_UNKNOWN"],
            true_no_pred_yes=cm_counts["NO_YES"],
            true_no_pred_no=cm_counts["NO_NO"],
            true_no_pred_unk=cm_counts["NO_UNKNOWN"],
            true_unk_pred_yes=cm_counts["UNKNOWN_YES"],
            true_unk_pred_no=cm_counts["UNKNOWN_NO"],
            true_unk_pred_unk=cm_counts["UNKNOWN_UNKNOWN"],
        )

        return EpistemicEvaluationMetrics(
            arm_name=arm_name,
            total_samples=total,
            overall_accuracy=overall_acc,
            macro_f1=macro_f1,
            yes_precision=y_p,
            yes_recall=y_r,
            yes_f1=y_f1,
            no_precision=n_p,
            no_recall=n_r,
            no_f1=n_f1,
            unknown_precision=u_p,
            unknown_recall=u_r,
            unknown_f1=u_f1,
            expected_calibration_error=float(ece),
            brier_score=brier,
            false_confident_rate=fc_rate,
            confusion_matrix=cm,
        )

    # -------------------------------------------------------------------------
    # Arm Implementations
    # -------------------------------------------------------------------------

    def evaluate_constant_baseline(self, tasks: list[EpistemicTask], constant_verdict: str) -> EpistemicEvaluationMetrics:
        preds: list[tuple[str, str, float, list[float]]] = []
        p_vec = [1.0 if c == constant_verdict else 0.0 for c in self.CLASSES]
        for t in tasks:
            preds.append((t.expected_verdict, constant_verdict, 1.0, p_vec))
        return self._compute_metrics(f"BASELINE_CONSTANT_{constant_verdict}", preds)

    def evaluate_external_solver(self, world: EpistemicWorldDataset) -> EpistemicEvaluationMetrics:
        """Arm D: Deterministic verified graph reasoner under open-world semantics."""
        preds: list[tuple[str, str, float, list[float]]] = []
        rel = world.semantics.name
        is_trans = world.semantics.is_transitive
        is_sym = world.semantics.is_symmetric

        for t in world.held_out_test_pool:
            adj = collections.defaultdict(set)
            all_nodes = set()
            for p in t.premises:
                parts = p.split()
                if len(parts) >= 3:
                    s, r, o = parts[0], parts[1], parts[2]
                    adj[s].add(o)
                    all_nodes.add(s)
                    all_nodes.add(o)

            words = t.question.replace("?", "").replace("Does ", "").split()
            q_s = words[0] if len(words) > 0 else ""
            q_o = words[-1] if len(words) > 1 else ""

            # Open-world reasoning logic:
            if q_s not in all_nodes or q_o not in all_nodes:
                pred = "UNKNOWN"
            elif q_s == q_o:
                pred = "UNKNOWN"
            else:
                # Check 1-hop reachability
                if q_o in adj[q_s]:
                    pred = "YES"
                elif q_s in adj[q_o]:
                    pred = "YES" if is_sym else "NO"
                elif is_trans:
                    # Check multi-hop reachability via BFS
                    visited = set()
                    queue = collections.deque([q_s])
                    found = False
                    while queue:
                        curr = queue.popleft()
                        if curr == q_o and curr != q_s:
                            found = True
                            break
                        if curr not in visited:
                            visited.add(curr)
                            for nbr in adj[curr]:
                                queue.append(nbr)

                    rev_visited = set()
                    rev_queue = collections.deque([q_o])
                    rev_found = False
                    while rev_queue:
                        curr = rev_queue.popleft()
                        if curr == q_s and curr != q_o:
                            rev_found = True
                            break
                        if curr not in rev_visited:
                            rev_visited.add(curr)
                            for nbr in adj[curr]:
                                rev_queue.append(nbr)

                    if found:
                        pred = "YES"
                    elif rev_found and not is_sym:
                        pred = "NO"
                    else:
                        pred = "UNKNOWN" 
                else:
                    pred = "UNKNOWN"

            p_vec = [1.0 if c == pred else 0.0 for c in self.CLASSES]
            preds.append((t.expected_verdict, pred, 1.0, p_vec))

        return self._compute_metrics("ARM_D_EXTERNAL_VERIFIED_SOLVER", preds)

    def evaluate_arm_a_m44_neural(self, world: EpistemicWorldDataset, practice_budget: int = 16) -> EpistemicEvaluationMetrics:
        """Arm A: M44 causal LM token logit scoring with LoRA adapter."""
        model = copy.deepcopy(self.base_model)
        inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
        add_domain_adapter(model, domain_name="m45_arm_a", r=8)
        set_active_domain_adapter(model, "m45_arm_a")
        freeze_for_adapter_training(model, "m45_arm_a")

        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        practice = world.train_practice_pool[:practice_budget]

        model.train()
        for step in range(max(20, practice_budget * 2)):
            t = practice[step % len(practice)]
            prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
            target = f" {t.expected_verdict}"
            p_ids = self.tokenizer.encode(prompt)
            t_ids = self.tokenizer.encode(target)
            full_ids = p_ids + t_ids
            if len(full_ids) > model.config.context_length:
                full_ids = full_ids[-model.config.context_length:]
                p_ids = full_ids[:-len(t_ids)]

            inp = torch.tensor([full_ids[:-1]], dtype=torch.long, device=self.device)
            lbl = torch.tensor([[-100] * (len(p_ids) - 1) + t_ids], dtype=torch.long, device=self.device)

            out = model(inp)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits.view(-1, logits.size(-1)), lbl.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        preds: list[tuple[str, str, float, list[float]]] = []
        with torch.no_grad():
            for t in world.held_out_test_pool:
                prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                p_toks = self.tokenizer.encode(prompt)
                if len(p_toks) > model.config.context_length - 8:
                    p_toks = p_toks[-(model.config.context_length - 8):]

                scores = []
                for seq in self.cand_seqs:
                    full_cand = p_toks + seq
                    if len(full_cand) > model.config.context_length:
                        full_cand = full_cand[-model.config.context_length:]
                    inp = torch.tensor([full_cand], dtype=torch.long, device=self.device)
                    out = model(inp)
                    logits = out[0] if isinstance(out, tuple) else out
                    log_probs = torch.log_softmax(logits, dim=-1)
                    seq_lp = sum(float(log_probs[0, len(p_toks) - 1 + i, tok].item()) for i, tok in enumerate(seq))
                    scores.append(seq_lp / max(1, len(seq)))

                probs = torch.softmax(torch.tensor(scores), dim=-1).tolist()
                best_idx = int(np.argmax(scores))
                pred = self.CLASSES[best_idx]
                conf = float(probs[best_idx])
                preds.append((t.expected_verdict, pred, conf, probs))

        return self._compute_metrics("ARM_A_M44_NEURAL_BASELINE", preds)

    def evaluate_arm_b_tri_state_head(self, world: EpistemicWorldDataset, practice_budget: int = 16) -> EpistemicEvaluationMetrics:
        """Arm B: Balanced 3-class neural classification head trained with balanced cross-entropy."""
        model = copy.deepcopy(self.base_model)
        inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
        add_domain_adapter(model, domain_name="m45_arm_b", r=8)
        set_active_domain_adapter(model, "m45_arm_b")
        freeze_for_adapter_training(model, "m45_arm_b")

        tri_head = TriStateEpistemicHead(d_model=model.config.d_model, num_classes=3).to(self.device)

        # Equal class weights in cross-entropy to overcome prior YES bias
        weights = torch.tensor([1.0, 1.0, 1.2], dtype=torch.float, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = AdamW(list(tri_head.parameters()) + [p for p in model.parameters() if p.requires_grad], lr=1e-3)

        practice = world.train_practice_pool[:practice_budget]

        model.train()
        tri_head.train()
        for step in range(max(30, practice_budget * 3)):
            t = practice[step % len(practice)]
            prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
            ids = self.tokenizer.encode(prompt)
            if len(ids) > model.config.context_length:
                ids = ids[-model.config.context_length:]

            inp = torch.tensor([ids], dtype=torch.long, device=self.device)
            target_class = torch.tensor([self.CLASS_TO_IDX[t.expected_verdict]], dtype=torch.long, device=self.device)

            hidden, _ = self._extract_hidden_and_logits(model, inp)
            logits = tri_head(hidden)
            loss = criterion(logits, target_class)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        tri_head.eval()
        preds: list[tuple[str, str, float, list[float]]] = []
        with torch.no_grad():
            for t in world.held_out_test_pool:
                prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                ids = self.tokenizer.encode(prompt)
                if len(ids) > model.config.context_length:
                    ids = ids[-model.config.context_length:]
                inp = torch.tensor([ids], dtype=torch.long, device=self.device)

                hidden, _ = self._extract_hidden_and_logits(model, inp)
                logits = tri_head(hidden)
                probs = torch.softmax(logits, dim=-1)[0].tolist()
                best_idx = int(np.argmax(probs))
                pred = self.CLASSES[best_idx]
                conf = float(probs[best_idx])
                preds.append((t.expected_verdict, pred, conf, probs))

        return self._compute_metrics("ARM_B_BALANCED_TRI_STATE_HEAD", preds)

    def evaluate_arm_c_analogy_supervised_head(
        self,
        world: EpistemicWorldDataset,
        source_worlds: list[EpistemicWorldDataset],
        practice_budget: int = 16,
    ) -> EpistemicEvaluationMetrics:
        """Arm C: Tri-state head trained with historical structural analogy experiences + target practice."""
        model = copy.deepcopy(self.base_model)
        inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
        add_domain_adapter(model, domain_name="m45_arm_c", r=8)
        set_active_domain_adapter(model, "m45_arm_c")
        freeze_for_adapter_training(model, "m45_arm_c")

        tri_head = TriStateEpistemicHead(d_model=model.config.d_model, num_classes=3).to(self.device)
        weights = torch.tensor([1.0, 1.0, 1.1], dtype=torch.float, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = AdamW(list(tri_head.parameters()) + [p for p in model.parameters() if p.requires_grad], lr=1e-3)

        # Build supervised pool: historical structural examples from source worlds + target practice
        supervised_pool: list[EpistemicTask] = []
        for sw in source_worlds:
            supervised_pool.extend(sw.train_practice_pool[:6])  # 6 verified examples from each prior world
        supervised_pool.extend(world.train_practice_pool[:practice_budget])

        model.train()
        tri_head.train()
        for step in range(max(40, len(supervised_pool) * 2)):
            t = supervised_pool[step % len(supervised_pool)]
            prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
            ids = self.tokenizer.encode(prompt)
            if len(ids) > model.config.context_length:
                ids = ids[-model.config.context_length:]

            inp = torch.tensor([ids], dtype=torch.long, device=self.device)
            target_class = torch.tensor([self.CLASS_TO_IDX[t.expected_verdict]], dtype=torch.long, device=self.device)

            hidden, _ = self._extract_hidden_and_logits(model, inp)
            logits = tri_head(hidden)
            loss = criterion(logits, target_class)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        tri_head.eval()
        preds: list[tuple[str, str, float, list[float]]] = []
        with torch.no_grad():
            for t in world.held_out_test_pool:
                prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                ids = self.tokenizer.encode(prompt)
                if len(ids) > model.config.context_length:
                    ids = ids[-model.config.context_length:]
                inp = torch.tensor([ids], dtype=torch.long, device=self.device)

                hidden, _ = self._extract_hidden_and_logits(model, inp)
                logits = tri_head(hidden)
                probs = torch.softmax(logits, dim=-1)[0].tolist()
                best_idx = int(np.argmax(probs))
                pred = self.CLASSES[best_idx]
                conf = float(probs[best_idx])
                preds.append((t.expected_verdict, pred, conf, probs))

        return self._compute_metrics("ARM_C_ANALOGY_SUPERVISED_HEAD", preds)
