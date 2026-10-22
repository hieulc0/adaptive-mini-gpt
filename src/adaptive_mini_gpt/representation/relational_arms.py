"""Phase 4: Minimal Controlled Relational Comparison Across 4 Arms.

Arms:
  - Arm A: Existing M45 three-class head with unstratified slice (Baseline).
  - Arm B: Balanced Supervised Relational Training (Stratified Sampling + Balanced Loss).
  - Arm C: Arm B + Graph-Contrastive Representation Objective (SupCon on latent states).
  - Arm D: External Verified Graph Reasoner (Programmatic Reference Comparator).
"""
from __future__ import annotations

import collections
import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from adaptive_mini_gpt.data.bpe_scaling import BPETokenizer
from adaptive_mini_gpt.model.modular_adapter import (
    add_domain_adapter,
    freeze_for_adapter_training,
    inject_multi_domain_adapters,
    set_active_domain_adapter,
)
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.representation.positive_control import TriStateClassificationHead
from adaptive_mini_gpt.representation.structural_benchmark import RelationalGraphDataset
from adaptive_mini_gpt.uncertainty.epistemic_benchmark import EpistemicTask
from adaptive_mini_gpt.uncertainty.tri_state_classifier import EpistemicEvaluationMetrics, TriStateConfusionMatrix


class GraphContrastiveLoss(nn.Module):
    """Supervised contrastive loss over normalized hidden representations.

    Pulls representations of identical epistemic classes (YES-YES, NO-NO, UNK-UNK)
    together in angular cosine space while repelling incompatible classes.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: [batch_size, d_model], labels: [batch_size]
        if features.size(0) < 2:
            return torch.tensor(0.0, device=features.device)

        norm_feat = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.matmul(norm_feat, norm_feat.T) / self.temperature

        # Create positive mask: 1 where labels match and i != j
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        diag = torch.eye(features.size(0), dtype=torch.float, device=features.device)
        mask = mask * (1.0 - diag)

        # Log-sum-exp over non-self comparisons
        exp_sim = torch.exp(sim_matrix) * (1.0 - diag)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean over positive pairs
        pos_count = mask.sum(dim=1)
        valid_idx = pos_count > 0
        if not valid_idx.any():
            return torch.tensor(0.0, device=features.device)

        mean_log_prob = (mask * log_prob).sum(dim=1)[valid_idx] / pos_count[valid_idx]
        return -mean_log_prob.mean()


class RelationalArmsBenchmark:
    """Executes the 4 experimental arms on relational graph datasets."""

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

    def _extract_hidden(self, model: MiniGPT, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.size()
        tok_emb = model.wte(input_ids)
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device)
        pos_emb = model.wpe(pos)
        x = model.drop(tok_emb + pos_emb)
        for block in model.blocks:
            x, _ = block(x)
        return model.ln_f(x)

    def _compute_metrics(
        self,
        arm_name: str,
        predictions: list[tuple[str, str, float, list[float]]],
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

        # ECE with 10 bins
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

        probs_np = np.array(prob_matrix)
        targets_np = np.array(target_one_hot)
        brier = float(np.mean(np.sum((probs_np - targets_np) ** 2, axis=1)))

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

    def evaluate_arm_a_m45_baseline(
        self, world: RelationalGraphDataset, practice_budget: int = 16
    ) -> EpistemicEvaluationMetrics:
        """Arm A: M45 baseline reproducing unstratified sequential practice slice."""
        model = copy.deepcopy(self.base_model)
        inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
        add_domain_adapter(model, domain_name="m46_arm_a", r=8)
        set_active_domain_adapter(model, "m46_arm_a")
        freeze_for_adapter_training(model, "m46_arm_a")

        tri_head = TriStateClassificationHead(d_model=model.config.d_model, num_classes=3).to(self.device)
        optimizer = AdamW(list(tri_head.parameters()) + [p for p in model.parameters() if p.requires_grad], lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Unstratified sequential practice: all YES, then NO, no UNK
        raw_pool = [t for t in world.train_practice_pool if t.expected_verdict == "YES"] + \
                   [t for t in world.train_practice_pool if t.expected_verdict == "NO"] + \
                   [t for t in world.train_practice_pool if t.expected_verdict == "UNKNOWN"]
        practice = raw_pool[:practice_budget]

        model.train()
        tri_head.train()
        for step in range(max(30, practice_budget * 2)):
            t = practice[step % len(practice)]
            prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
            ids = self.tokenizer.encode(prompt)
            if len(ids) > model.config.context_length:
                ids = ids[-model.config.context_length:]
            inp = torch.tensor([ids], dtype=torch.long, device=self.device)
            target = torch.tensor([self.CLASS_TO_IDX[t.expected_verdict]], dtype=torch.long, device=self.device)

            hidden = self._extract_hidden(model, inp)
            logits = tri_head(hidden)
            loss = criterion(logits, target)

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
                hidden = self._extract_hidden(model, inp)
                logits = tri_head(hidden)
                probs = torch.softmax(logits, dim=-1)[0].tolist()
                pred_idx = int(torch.argmax(logits, dim=-1).item())
                preds.append((t.expected_verdict, self.CLASSES[pred_idx], float(probs[pred_idx]), probs))

        return self._compute_metrics("ARM_A_M45_BASELINE", preds)

    def evaluate_arm_b_balanced_supervised(
        self, world: RelationalGraphDataset, practice_budget: int = 16, epochs: int = 25
    ) -> EpistemicEvaluationMetrics:
        """Arm B: Stratified balanced supervised training with calibrated step count."""
        model = copy.deepcopy(self.base_model)
        inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
        add_domain_adapter(model, domain_name="m46_arm_b", r=8)
        set_active_domain_adapter(model, "m46_arm_b")
        freeze_for_adapter_training(model, "m46_arm_b")

        tri_head = TriStateClassificationHead(d_model=model.config.d_model, num_classes=3).to(self.device)
        optimizer = AdamW(list(tri_head.parameters()) + [p for p in model.parameters() if p.requires_grad], lr=2e-3)
        criterion = nn.CrossEntropyLoss()

        # Stratified balanced practice pool: take practice_budget items evenly distributed across classes
        by_class: dict[str, list[EpistemicTask]] = {"YES": [], "NO": [], "UNKNOWN": []}
        for t in world.train_practice_pool:
            by_class[t.expected_verdict].append(t)

        per_class = max(1, practice_budget // 3)
        balanced_practice: list[EpistemicTask] = []
        for c in self.CLASSES:
            balanced_practice.extend(by_class[c][:per_class])

        model.train()
        tri_head.train()
        for epoch in range(epochs):
            for t in balanced_practice:
                prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                ids = self.tokenizer.encode(prompt)
                if len(ids) > model.config.context_length:
                    ids = ids[-model.config.context_length:]
                inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                target = torch.tensor([self.CLASS_TO_IDX[t.expected_verdict]], dtype=torch.long, device=self.device)

                hidden = self._extract_hidden(model, inp)
                logits = tri_head(hidden)
                loss = criterion(logits, target)

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
                hidden = self._extract_hidden(model, inp)
                logits = tri_head(hidden)
                probs = torch.softmax(logits, dim=-1)[0].tolist()
                pred_idx = int(torch.argmax(logits, dim=-1).item())
                preds.append((t.expected_verdict, self.CLASSES[pred_idx], float(probs[pred_idx]), probs))

        return self._compute_metrics("ARM_B_BALANCED_SUPERVISED", preds)

    def evaluate_arm_c_contrastive_head(
        self, world: RelationalGraphDataset, practice_budget: int = 16, epochs: int = 25, lambda_contrast: float = 0.25
    ) -> EpistemicEvaluationMetrics:
        """Arm C: Stratified balanced training + Graph Contrastive Representation Objective."""
        model = copy.deepcopy(self.base_model)
        inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
        add_domain_adapter(model, domain_name="m46_arm_c", r=8)
        set_active_domain_adapter(model, "m46_arm_c")
        freeze_for_adapter_training(model, "m46_arm_c")

        tri_head = TriStateClassificationHead(d_model=model.config.d_model, num_classes=3).to(self.device)
        contrastive_loss_fn = GraphContrastiveLoss(temperature=0.15)
        optimizer = AdamW(list(tri_head.parameters()) + [p for p in model.parameters() if p.requires_grad], lr=2e-3)
        ce_criterion = nn.CrossEntropyLoss()

        by_class: dict[str, list[EpistemicTask]] = {"YES": [], "NO": [], "UNKNOWN": []}
        for t in world.train_practice_pool:
            by_class[t.expected_verdict].append(t)

        per_class = max(1, practice_budget // 3)
        balanced_practice: list[EpistemicTask] = []
        for c in self.CLASSES:
            balanced_practice.extend(by_class[c][:per_class])

        model.train()
        tri_head.train()
        for epoch in range(epochs):
            # Form mini-batches containing mixed classes to compute contrastive pairings
            np.random.shuffle(balanced_practice)
            batch_size = min(6, len(balanced_practice))

            for i in range(0, len(balanced_practice), batch_size):
                batch_tasks = balanced_practice[i : i + batch_size]
                if len(batch_tasks) < 2:
                    continue

                batch_hiddens = []
                batch_targets = []

                for t in batch_tasks:
                    prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                    ids = self.tokenizer.encode(prompt)
                    if len(ids) > model.config.context_length:
                        ids = ids[-model.config.context_length:]
                    inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                    h = self._extract_hidden(model, inp)
                    # Extract last token vector: [1, d_model]
                    last_h = h[:, -1, :]
                    batch_hiddens.append(last_h)
                    batch_targets.append(self.CLASS_TO_IDX[t.expected_verdict])

                all_h = torch.cat(batch_hiddens, dim=0)  # [batch_size, d_model]
                targets = torch.tensor(batch_targets, dtype=torch.long, device=self.device)

                logits = tri_head.head(all_h)
                loss_ce = ce_criterion(logits, targets)
                loss_contrast = contrastive_loss_fn(all_h, targets)
                loss_total = loss_ce + lambda_contrast * loss_contrast

                optimizer.zero_grad()
                loss_total.backward()
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
                hidden = self._extract_hidden(model, inp)
                logits = tri_head(hidden)
                probs = torch.softmax(logits, dim=-1)[0].tolist()
                pred_idx = int(torch.argmax(logits, dim=-1).item())
                preds.append((t.expected_verdict, self.CLASSES[pred_idx], float(probs[pred_idx]), probs))

        return self._compute_metrics("ARM_C_CONTRASTIVE_SUPERVISED", preds)

    def evaluate_arm_d_external_reasoner(
        self, world: RelationalGraphDataset
    ) -> EpistemicEvaluationMetrics:
        """Arm D: Deterministic verified graph reasoner (Programmatic Reference Comparator)."""
        preds: list[tuple[str, str, float, list[float]]] = []
        adj = collections.defaultdict(set)
        rel = world.semantics.name
        is_trans = world.semantics.is_transitive
        is_sym = world.semantics.is_symmetric

        for p in world.premises:
            parts = p.split()
            if len(parts) >= 3:
                s, r, o = parts[0], parts[1], parts[2]
                adj[s].add(o)
                if is_sym:
                    adj[o].add(s)

        for t in world.held_out_test_pool:
            words = t.question.replace("?", "").replace("Does ", "").split()
            q_s = words[0] if len(words) > 0 else ""
            q_o = words[-1] if len(words) > 1 else ""

            # Open-world reachability check
            visited = set()
            queue = collections.deque([q_s])
            forward = False
            while queue:
                curr = queue.popleft()
                if curr == q_o and curr != q_s:
                    forward = True
                    break
                if curr not in visited:
                    visited.add(curr)
                    for nbr in adj[curr]:
                        queue.append(nbr)

            rev_visited = set()
            rev_queue = collections.deque([q_o])
            backward = False
            while rev_queue:
                curr = rev_queue.popleft()
                if curr == q_s and curr != q_o:
                    backward = True
                    break
                if curr not in rev_visited:
                    rev_visited.add(curr)
                    for nbr in adj[curr]:
                        rev_queue.append(nbr)

            if forward:
                pred = "YES"
            elif backward and not is_sym:
                pred = "NO"
            else:
                pred = "UNKNOWN"

            p_vec = [1.0 if c == pred else 0.0 for c in self.CLASSES]
            preds.append((t.expected_verdict, pred, 1.0, p_vec))

        return self._compute_metrics("ARM_D_EXTERNAL_VERIFIED_REASONER", preds)
