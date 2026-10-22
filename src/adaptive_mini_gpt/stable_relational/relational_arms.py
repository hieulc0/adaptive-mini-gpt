"""Phase 3: Controlled Relational Learning Arms Benchmark.

Compares:
  Arm A: M46 Contrastive Baseline (Unstratified mini-batches).
  Arm B: Balanced Supervised Training with Stratified Replay.
  Arm C: Arm B + Hard-Negative Structural Contrastive Objective.
  Arm D: External Verified Graph Reasoner (Programmatic Reference Ceiling).
"""
from __future__ import annotations

import collections
import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.modular_adapter import (
    add_domain_adapter,
    freeze_for_adapter_training,
    inject_multi_domain_adapters,
    set_active_domain_adapter,
)
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.representation.positive_control import TriStateClassificationHead
from adaptive_mini_gpt.stable_relational.hard_negatives import (
    HardNegativeGenerator,
    StructuralContrastiveLoss,
)
from adaptive_mini_gpt.stable_relational.stratified_replay import StratifiedRelationalReplayBuffer
from adaptive_mini_gpt.stable_relational.structural_benchmark import EnhancedRelationalDataset
from adaptive_mini_gpt.uncertainty.epistemic_benchmark import EpistemicTask
from adaptive_mini_gpt.uncertainty.tri_state_classifier import (
    EpistemicEvaluationMetrics,
    TriStateConfusionMatrix,
)


class M47RelationalArmsRunner:
    """Executes the 4 controlled experimental arms on enhanced relational datasets."""

    CLASSES = ["YES", "NO", "UNKNOWN"]
    CLASS_TO_IDX = {"YES": 0, "NO": 1, "UNKNOWN": 2}

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: BPETokenizer,
        device: torch.device,
        seed: int = 42,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.seed = seed

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
        false_confident = 0
        for p in unk_preds:
            if p[1] in ("YES", "NO") and p[2] >= 0.50:
                false_confident += 1
        false_conf_rate = false_confident / max(1, len(unk_preds))

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
            expected_calibration_error=ece,
            brier_score=brier,
            false_confident_rate=false_conf_rate,
            confusion_matrix=cm,
        )

    def evaluate_arm_a_m46_baseline(
        self, world: EnhancedRelationalDataset, practice_budget: int = 16, epochs: int = 25
    ) -> EpistemicEvaluationMetrics:
        """Arm A: M46 Contrastive Baseline (Unstratified mini-batches)."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        model = copy.deepcopy(self.base_model)
        inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
        add_domain_adapter(model, domain_name="m47_arm_a", r=8)
        set_active_domain_adapter(model, "m47_arm_a")
        freeze_for_adapter_training(model, "m47_arm_a")

        tri_head = TriStateClassificationHead(d_model=model.config.d_model, num_classes=3).to(self.device)
        optimizer = AdamW(list(tri_head.parameters()) + [p for p in model.parameters() if p.requires_grad], lr=2e-3)
        ce_criterion = nn.CrossEntropyLoss()

        practice_tasks = world.train_practice_pool[:practice_budget]

        model.train()
        tri_head.train()
        for epoch in range(epochs):
            for t in practice_tasks:
                prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                ids = self.tokenizer.encode(prompt)
                if len(ids) > model.config.context_length:
                    ids = ids[-model.config.context_length:]
                inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                target = torch.tensor([self.CLASS_TO_IDX[t.expected_verdict]], dtype=torch.long, device=self.device)

                h = self._extract_hidden(model, inp)
                logits = tri_head(h)
                loss = ce_criterion(logits, target)

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

        return self._compute_metrics("ARM_A_M46_BASELINE", preds)

    def evaluate_arm_b_balanced_replay(
        self, world: EnhancedRelationalDataset, practice_budget: int = 16, epochs: int = 25
    ) -> EpistemicEvaluationMetrics:
        """Arm B: Balanced Supervised Training with Stratified Replay Buffer."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        model = copy.deepcopy(self.base_model)
        inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
        add_domain_adapter(model, domain_name="m47_arm_b", r=8)
        set_active_domain_adapter(model, "m47_arm_b")
        freeze_for_adapter_training(model, "m47_arm_b")

        tri_head = TriStateClassificationHead(d_model=model.config.d_model, num_classes=3).to(self.device)
        optimizer = AdamW(list(tri_head.parameters()) + [p for p in model.parameters() if p.requires_grad], lr=2e-3)
        ce_criterion = nn.CrossEntropyLoss()

        replay_buffer = StratifiedRelationalReplayBuffer(capacity_per_class=50, seed=self.seed)
        # Seed replay buffer with stratified practice pool
        replay_buffer.add_batch(world.train_practice_pool[:practice_budget])

        model.train()
        tri_head.train()
        for epoch in range(epochs):
            # Sample exactly balanced batch of size 6 (2 YES, 2 NO, 2 UNK)
            batch = replay_buffer.sample_stratified_batch(batch_size=6)
            for t in batch:
                prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                ids = self.tokenizer.encode(prompt)
                if len(ids) > model.config.context_length:
                    ids = ids[-model.config.context_length:]
                inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                target = torch.tensor([self.CLASS_TO_IDX[t.expected_verdict]], dtype=torch.long, device=self.device)

                h = self._extract_hidden(model, inp)
                logits = tri_head(h)
                loss = ce_criterion(logits, target)

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

        return self._compute_metrics("ARM_B_BALANCED_REPLAY", preds)

    def evaluate_arm_c_hard_negative_contrastive(
        self,
        world: EnhancedRelationalDataset,
        practice_budget: int = 16,
        epochs: int = 25,
        lambda_contrast: float = 0.30,
    ) -> EpistemicEvaluationMetrics:
        """Arm C: Balanced Supervised Training + Stratified Replay + Hard-Negative Contrastive Loss."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        model = copy.deepcopy(self.base_model)
        inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
        add_domain_adapter(model, domain_name="m47_arm_c", r=8)
        set_active_domain_adapter(model, "m47_arm_c")
        freeze_for_adapter_training(model, "m47_arm_c")

        tri_head = TriStateClassificationHead(d_model=model.config.d_model, num_classes=3).to(self.device)
        contrast_fn = StructuralContrastiveLoss(temperature=0.15, hard_negative_weight=1.5)
        optimizer = AdamW(list(tri_head.parameters()) + [p for p in model.parameters() if p.requires_grad], lr=2e-3)
        ce_criterion = nn.CrossEntropyLoss()

        replay_buffer = StratifiedRelationalReplayBuffer(capacity_per_class=50, seed=self.seed)
        replay_buffer.add_batch(world.train_practice_pool[:practice_budget])

        # Generate structural hard negatives
        neg_gen = HardNegativeGenerator(world, seed=self.seed)
        reversals = neg_gen.generate_directional_reversals()
        if reversals:
            replay_buffer.add_batch(reversals[:6])

        model.train()
        tri_head.train()
        for epoch in range(epochs):
            # Sample balanced batch (6 tasks: 2 YES, 2 NO, 2 UNKNOWN)
            batch = replay_buffer.sample_stratified_batch(batch_size=6)
            if len(batch) < 3:
                continue

            batch_hiddens = []
            batch_targets = []

            for t in batch:
                prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                ids = self.tokenizer.encode(prompt)
                if len(ids) > model.config.context_length:
                    ids = ids[-model.config.context_length:]
                inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                h = self._extract_hidden(model, inp)
                last_h = h[:, -1, :]
                batch_hiddens.append(last_h)
                batch_targets.append(self.CLASS_TO_IDX[t.expected_verdict])

            all_h = torch.cat(batch_hiddens, dim=0)  # [B, d_model]
            targets = torch.tensor(batch_targets, dtype=torch.long, device=self.device)

            logits = tri_head.head(all_h)
            loss_ce = ce_criterion(logits, targets)
            loss_contrast = contrast_fn(all_h, targets)
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

        return self._compute_metrics("ARM_C_HARD_NEGATIVE_CONTRASTIVE", preds)

    def evaluate_arm_d_external_reasoner(
        self, world: EnhancedRelationalDataset
    ) -> EpistemicEvaluationMetrics:
        """Arm D: Deterministic verified graph reasoner (Programmatic Reference Ceiling)."""
        preds: list[tuple[str, str, float, list[float]]] = []
        adj: dict[str, set[str]] = collections.defaultdict(set)
        refutations: set[tuple[str, str]] = set()

        rel = world.semantics.name
        is_sym = world.semantics.is_symmetric
        is_trans = world.semantics.is_transitive

        for p in world.premises:
            parts = p.split()
            if len(parts) >= 3:
                s, r, o = parts[0], parts[1], parts[2]
                if r == rel:
                    adj[s].add(o)
                    if is_sym:
                        adj[o].add(s)
                else:
                    refutations.add((s, o))
                    if is_sym:
                        refutations.add((o, s))

        for t in world.held_out_test_pool:
            words = t.question.replace("?", "").replace("Does ", "").split()
            q_s = words[0] if len(words) > 0 else ""
            q_o = words[-1] if len(words) > 1 else ""

            # Check explicit refutation
            if (q_s, q_o) in refutations or (q_o, q_s) in refutations:
                pred = "NO"
            else:
                # Forward reachability check
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

                # Backward reachability check
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

            probs = [1.0 if c == pred else 0.0 for c in self.CLASSES]
            preds.append((t.expected_verdict, pred, 1.0, probs))

        return self._compute_metrics("ARM_D_EXTERNAL_VERIFIED_REASONER", preds)
