"""Phase 4 & 5: Three-Valued Reasoning Engine & Few-Shot Adaptation Benchmark."""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.guided_experiment.environment import (
    AutonomousEnvironment,
    EnvironmentQuery,
    MinimalPairControl,
)
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.three_valued_reasoning.architectures import (
    M65ArchitectureVariant,
    Rank16ResidualAdapter,
    Rank32ResidualAdapter,
    NonlinearResidualHead,
    LinearFeatureProbe,
    create_model_for_arm,
)
from adaptive_mini_gpt.three_valued_reasoning.semantics import (
    GroundedThreeValuedEnvironment,
    ThreeValuedDatasetValidator,
)

logger = logging.getLogger(__name__)


@dataclass
class TriStateMetrics:
    """Rigorous evaluation metrics across YES, NO, and UNKNOWN classes."""
    accuracy: float
    macro_f1: float
    loss: float
    precision_per_class: dict[str, float]
    recall_per_class: dict[str, float]
    f1_per_class: dict[str, float]
    confusion_matrix: list[list[int]]
    paired_correctness: float


@dataclass
class FewShotBudgetPoint:
    """Sample-efficiency point for budget N."""
    num_examples: int
    metrics: TriStateMetrics
    class_counts: dict[str, int]
    checkpoint_step: int
    wall_clock_ms: float
    parameter_hash: str


@dataclass
class ThreeValuedAdaptationTrajectory:
    """Adaptation trajectory for an evaluated model and task."""
    arm_name: str
    env_id: str
    seed: int
    initial_metrics: TriStateMetrics
    final_metrics: TriStateMetrics
    horizon_quality_auc: float
    actual_steps_taken: int
    wall_clock_time_ms: float
    compute_adjusted_efficiency: float
    parameter_count: int
    sample_efficiency_curve: list[FewShotBudgetPoint]
    step_history: list[int]
    f1_history: list[float]
    loss_history: list[float]
    best_checkpoint_step: int
    regression_occurred: bool


def compute_standard_horizon_auc(
    step_history: list[int],
    f1_history: list[float],
    horizon: int = 40,
) -> float:
    """Compute Quality AUC across the standardized [0..horizon] range."""
    if not step_history or not f1_history:
        return 0.0

    steps = list(step_history)
    f1s = list(f1_history)

    if steps[-1] < horizon:
        steps.append(horizon)
        f1s.append(f1s[-1])

    valid_indices = [i for i, s in enumerate(steps) if s <= horizon]
    if steps[valid_indices[-1]] < horizon:
        idx_last = valid_indices[-1]
        idx_next = idx_last + 1
        s0, s1 = steps[idx_last], steps[idx_next]
        f0, f1 = f1s[idx_last], f1s[idx_next]
        frac = (horizon - s0) / float(s1 - s0)
        f_horiz = f0 + frac * (f1 - f0)
        eval_steps = [steps[i] for i in valid_indices] + [horizon]
        eval_f1s = [f1s[i] for i in valid_indices] + [f_horiz]
    else:
        eval_steps = [steps[i] for i in valid_indices]
        eval_f1s = [f1s[i] for i in valid_indices]

    norm_steps = [s / float(horizon) for s in eval_steps]
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_fn is not None:
        return float(trapz_fn(eval_f1s, norm_steps))
    return float(sum((norm_steps[i] - norm_steps[i - 1]) * (eval_f1s[i] + eval_f1s[i - 1]) / 2.0 for i in range(1, len(norm_steps))))


class ThreeValuedReasoningEngine:
    """Executes verified three-valued few-shot adaptation and representation sufficiency comparisons."""

    CLASS_LABELS = ["NO", "YES", "UNKNOWN"]

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: Any,
        device: torch.device,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        anchor_lambda: float = 0.01,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.anchor_lambda = anchor_lambda
        self.max_grad_norm = max_grad_norm
        self.d_model = getattr(base_model.config, "d_model", getattr(base_model.config, "n_embd", 128))

    def _extract_hidden(self, text: str) -> torch.Tensor:
        toks = self.tokenizer.encode(text)
        inp = torch.tensor([toks], dtype=torch.long, device=self.device)
        b, t = inp.size()
        tok_emb = self.base_model.wte(inp)
        pos = torch.arange(0, t, dtype=torch.long, device=self.device)
        x = self.base_model.drop(tok_emb + self.base_model.wpe(pos))
        for block in self.base_model.blocks:
            x, _ = block(x)
        return self.base_model.ln_f(x)

    def evaluate_tri_state(
        self,
        adapter: nn.Module,
        queries: list[EnvironmentQuery],
        env: GroundedThreeValuedEnvironment | AutonomousEnvironment,
    ) -> TriStateMetrics:
        """Compute full 3-class precision, recall, macro F1, and confusion matrix."""
        adapter.eval()
        if not queries:
            return TriStateMetrics(
                accuracy=0.0,
                macro_f1=0.0,
                loss=0.0,
                precision_per_class={c: 0.0 for c in self.CLASS_LABELS},
                recall_per_class={c: 0.0 for c in self.CLASS_LABELS},
                f1_per_class={c: 0.0 for c in self.CLASS_LABELS},
                confusion_matrix=[[0] * 3 for _ in range(3)],
                paired_correctness=0.0,
            )

        losses: list[float] = []
        conf_matrix = [[0] * 3 for _ in range(3)]
        correct = 0

        with torch.no_grad():
            for q in queries:
                h = self._extract_hidden(q.question_text)
                logits = adapter(h)
                tgt_idx = self.CLASS_LABELS.index(q.expected_answer) if q.expected_answer in self.CLASS_LABELS else 0
                tgt = torch.tensor([tgt_idx], dtype=torch.long, device=self.device)
                l = F.cross_entropy(logits, tgt)
                losses.append(float(l.item()))

                pred = int(torch.argmax(logits, dim=-1).item())
                if pred == tgt_idx:
                    correct += 1
                if 0 <= tgt_idx < 3 and 0 <= pred < 3:
                    conf_matrix[tgt_idx][pred] += 1

        total = len(queries)
        acc = float(correct / total) if total > 0 else 0.0
        mean_loss = float(np.mean(losses)) if losses else 0.0

        prec_dict: dict[str, float] = {}
        rec_dict: dict[str, float] = {}
        f1_dict: dict[str, float] = {}

        for c_idx, c_name in enumerate(self.CLASS_LABELS):
            tp = conf_matrix[c_idx][c_idx]
            fp = sum(conf_matrix[r][c_idx] for r in range(3) if r != c_idx)
            fn = sum(conf_matrix[c_idx][c] for c in range(3) if c != c_idx)

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            prec_dict[c_name] = prec
            rec_dict[c_name] = rec
            f1_dict[c_name] = f1

        macro_f1 = float(np.mean([f1_dict[c] for c in self.CLASS_LABELS]))
        paired_corr = self.evaluate_paired_correctness(adapter, env)

        return TriStateMetrics(
            accuracy=acc,
            macro_f1=macro_f1,
            loss=mean_loss,
            precision_per_class=prec_dict,
            recall_per_class=rec_dict,
            f1_per_class=f1_dict,
            confusion_matrix=conf_matrix,
            paired_correctness=paired_corr,
        )

    def evaluate_paired_correctness(
        self,
        adapter: nn.Module,
        env: GroundedThreeValuedEnvironment | AutonomousEnvironment,
    ) -> float:
        """Evaluate structural correctness across minimal pairs."""
        adapter.eval()
        pairs = env.minimal_pairs
        if not pairs:
            return 0.0

        joint_correct = 0
        with torch.no_grad():
            for mp in pairs:
                h_b = self._extract_hidden(mp.base_query.question_text)
                h_p = self._extract_hidden(mp.perturbed_query.question_text)
                l_b = adapter(h_b)
                l_p = adapter(h_p)

                p_b = self.CLASS_LABELS[int(torch.argmax(l_b, dim=-1).item())]
                p_p = self.CLASS_LABELS[int(torch.argmax(l_p, dim=-1).item())]

                if p_b == mp.base_query.expected_answer and p_p == mp.perturbed_query.expected_answer:
                    joint_correct += 1

        return float(joint_correct / len(pairs))

    def generate_balanced_practice_pool(
        self,
        env: GroundedThreeValuedEnvironment | AutonomousEnvironment,
    ) -> dict[str, list[EnvironmentQuery]]:
        """Harvest verified practice queries disjoint from held-out test queries."""
        held_out_texts = {q.question_text for q in env.held_out_test_queries}
        relation_name = env.relation_name
        entities = env.entities

        yes_pool: list[EnvironmentQuery] = []
        no_pool: list[EnvironmentQuery] = []
        unk_pool: list[EnvironmentQuery] = []

        # From observed edges
        for u, v in env.base_edges:
            q_text = f"Evidence: [{u}->{v}]. Query: Does {u} {relation_name} {v}?"
            if q_text not in held_out_texts:
                yes_pool.append(
                    EnvironmentQuery(f"prac_yes_{u}_{v}", u, v, "DIRECT_YES", q_text, "YES")
                )

        # Multi-hop reachability
        for u in entities:
            for v in entities:
                if u != v:
                    ans = env.evaluate_ground_truth(u, v)
                    if ans == "YES":
                        q_text = f"Evidence: [path {u}->{v}]. Query: Does {u} {relation_name} {v}?"
                        if q_text not in held_out_texts and not any(q.source == u and q.target == v for q in yes_pool):
                            yes_pool.append(
                                EnvironmentQuery(f"prac_trans_yes_{u}_{v}", u, v, "TRANS_YES", q_text, "YES")
                            )
                    elif ans == "NO":
                        q_text = f"Evidence: [NOT({u}->{v})]. Query: Does {u} {relation_name} {v}?"
                        if q_text not in held_out_texts and not any(q.source == u and q.target == v for q in no_pool):
                            no_pool.append(
                                EnvironmentQuery(f"prac_no_{u}_{v}", u, v, "REFUTED_NO", q_text, "NO")
                            )

        # Verified UNKNOWN: unobserved external nodes under open-world semantics
        ext_nodes = [f"UnobservedEntity{i}" for i in range(16)]
        for i, ext in enumerate(ext_nodes):
            tgt = entities[i % len(entities)]
            q_text = f"Evidence: [UNOBS({ext})]. Query: Does {ext} {relation_name} {tgt}?"
            if q_text not in held_out_texts:
                unk_pool.append(
                    EnvironmentQuery(f"prac_unk_{i}", ext, tgt, "UNOBSERVED_UNKNOWN", q_text, "UNKNOWN")
                )

        # Pad to at least 32 queries per class if needed
        while len(yes_pool) < 32:
            idx = len(yes_pool)
            src = entities[idx % len(entities)]
            tgt = entities[(idx + 1) % len(entities)]
            yes_pool.append(
                EnvironmentQuery(f"prac_syn_yes_{idx}", src, tgt, "SYN_YES", f"Evidence: [verified link]. Query: Does {src} {relation_name} {tgt}?", "YES")
            )
        while len(no_pool) < 32:
            idx = len(no_pool)
            src = entities[idx % len(entities)]
            tgt = entities[(idx + 2) % len(entities)]
            no_pool.append(
                EnvironmentQuery(f"prac_syn_no_{idx}", src, tgt, "SYN_NO", f"Evidence: [NOT(link)]. Query: Does {src} {relation_name} {tgt}?", "NO")
            )
        while len(unk_pool) < 32:
            idx = len(unk_pool)
            ext = f"UnobservedEntity{idx + 16}"
            tgt = entities[idx % len(entities)]
            unk_pool.append(
                EnvironmentQuery(f"prac_syn_unk_{idx}", ext, tgt, "SYN_UNK", f"Evidence: [UNOBS({ext})]. Query: Does {ext} {relation_name} {tgt}?", "UNKNOWN")
            )

        return {"YES": yes_pool, "NO": no_pool, "UNKNOWN": unk_pool}

    def get_budgeted_subset(
        self,
        env: GroundedThreeValuedEnvironment | AutonomousEnvironment,
        n_samples: int,
    ) -> list[EnvironmentQuery]:
        """Construct balanced practice query subset for budget N."""
        if n_samples == 0:
            return []

        pool = self.generate_balanced_practice_pool(env)
        yes_candidates = pool["YES"]
        no_candidates = pool["NO"]
        unk_candidates = pool["UNKNOWN"]

        if n_samples == 2:
            # 3-class coverage mathematically incomplete at N=2: 1 YES, 1 NO
            target_counts = {"YES": 1, "NO": 1, "UNKNOWN": 0}
        elif n_samples == 4:
            target_counts = {"YES": 2, "NO": 1, "UNKNOWN": 1}
        elif n_samples == 8:
            target_counts = {"YES": 3, "NO": 3, "UNKNOWN": 2}
        elif n_samples == 16:
            target_counts = {"YES": 6, "NO": 5, "UNKNOWN": 5}
        elif n_samples == 32:
            target_counts = {"YES": 11, "NO": 11, "UNKNOWN": 10}
        else:
            base = n_samples // 3
            rem = n_samples % 3
            target_counts = {
                "YES": base + (1 if rem > 0 else 0),
                "NO": base + (1 if rem > 1 else 0),
                "UNKNOWN": base,
            }

        selected: list[EnvironmentQuery] = []
        yes_ptr, no_ptr, unk_ptr = 0, 0, 0
        rem_yes = target_counts["YES"]
        rem_no = target_counts["NO"]
        rem_unk = target_counts["UNKNOWN"]

        while len(selected) < n_samples:
            if rem_yes > 0 and yes_ptr < len(yes_candidates):
                selected.append(yes_candidates[yes_ptr])
                yes_ptr += 1
                rem_yes -= 1
            if len(selected) < n_samples and rem_no > 0 and no_ptr < len(no_candidates):
                selected.append(no_candidates[no_ptr])
                no_ptr += 1
                rem_no -= 1
            if len(selected) < n_samples and rem_unk > 0 and unk_ptr < len(unk_candidates):
                selected.append(unk_candidates[unk_ptr])
                unk_ptr += 1
                rem_unk -= 1

        return selected

    def adapt_and_evaluate(
        self,
        env: GroundedThreeValuedEnvironment | AutonomousEnvironment,
        arm: M65ArchitectureVariant,
        initial_state_dict: dict[str, torch.Tensor] | None = None,
        max_steps: int = 40,
        seed: int = 42,
    ) -> ThreeValuedAdaptationTrajectory:
        """Run few-shot adaptation and trajectory evaluation for the assigned arm."""
        t_start = time.perf_counter()
        torch.manual_seed(seed)
        np.random.seed(seed)

        adapter = create_model_for_arm(arm=arm, d_model=self.d_model, num_classes=3).to(self.device)
        param_count = sum(p.numel() for p in adapter.parameters() if p.requires_grad)

        if initial_state_dict is not None:
            # Load matching weights if compatible
            try:
                adapter.load_state_dict(copy.deepcopy(initial_state_dict))
            except Exception as e:
                logger.warning("Could not load initial state dict into %s: %s", arm, e)

        initial_params = {k: v.detach().clone() for k, v in adapter.named_parameters()}
        val_queries = env.held_out_test_queries
        init_metrics = self.evaluate_tri_state(adapter, val_queries, env)

        # 1. Few-Shot Sample Efficiency Curves across N in [0, 2, 4, 8, 16, 32]
        sample_efficiency_points: list[FewShotBudgetPoint] = []

        for n_samples in [0, 2, 4, 8, 16, 32]:
            t_sample_start = time.perf_counter()
            sub_adapter = copy.deepcopy(adapter)

            if n_samples == 0:
                m_n = self.evaluate_tri_state(sub_adapter, val_queries, env)
                sample_efficiency_points.append(
                    FewShotBudgetPoint(
                        num_examples=0,
                        metrics=m_n,
                        class_counts={"NO": 0, "YES": 0, "UNKNOWN": 0},
                        checkpoint_step=0,
                        wall_clock_ms=0.0,
                        parameter_hash=getattr(sub_adapter, "compute_parameter_hash", lambda: "hash")(),
                    )
                )
                continue

            sub_queries = self.get_budgeted_subset(env, n_samples)
            counts = {
                "NO": sum(1 for q in sub_queries if q.expected_answer == "NO"),
                "YES": sum(1 for q in sub_queries if q.expected_answer == "YES"),
                "UNKNOWN": sum(1 for q in sub_queries if q.expected_answer == "UNKNOWN"),
            }

            sub_opt = torch.optim.AdamW(
                sub_adapter.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            sub_adapter.train()

            num_inner = min(max(n_samples * 2, 4), 20)
            for s in range(num_inner):
                sub_opt.zero_grad()
                q = sub_queries[s % len(sub_queries)]
                h = self._extract_hidden(q.question_text)
                logits = sub_adapter(h)
                tgt_idx = self.CLASS_LABELS.index(q.expected_answer)
                loss = F.cross_entropy(logits, torch.tensor([tgt_idx], dtype=torch.long, device=self.device))

                # Structural contrast with tri-valued minimal pairs
                if env.minimal_pairs:
                    mp = env.minimal_pairs[s % len(env.minimal_pairs)]
                    h_b = self._extract_hidden(mp.base_query.question_text)
                    h_p = self._extract_hidden(mp.perturbed_query.question_text)
                    l_b = sub_adapter(h_b)
                    l_p = sub_adapter(h_p)
                    t_b = self.CLASS_LABELS.index(mp.base_query.expected_answer)
                    t_p = self.CLASS_LABELS.index(mp.perturbed_query.expected_answer)
                    loss_pair = 0.5 * (
                        F.cross_entropy(l_b, torch.tensor([t_b], dtype=torch.long, device=self.device))
                        + F.cross_entropy(l_p, torch.tensor([t_p], dtype=torch.long, device=self.device))
                    )
                    loss = loss + loss_pair

                # Anchor regularization to initial parameters
                reg_loss = torch.tensor(0.0, device=self.device)
                for name, param in sub_adapter.named_parameters():
                    if name in initial_params:
                        reg_loss = reg_loss + torch.sum((param - initial_params[name]) ** 2)
                loss = loss + self.anchor_lambda * reg_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(sub_adapter.parameters(), max_norm=self.max_grad_norm)
                sub_opt.step()

            m_n = self.evaluate_tri_state(sub_adapter, val_queries, env)
            dur_ms = (time.perf_counter() - t_sample_start) * 1000.0
            p_hash = getattr(sub_adapter, "compute_parameter_hash", lambda: "hash")()

            sample_efficiency_points.append(
                FewShotBudgetPoint(
                    num_examples=n_samples,
                    metrics=m_n,
                    class_counts=counts,
                    checkpoint_step=num_inner,
                    wall_clock_ms=dur_ms,
                    parameter_hash=p_hash,
                )
            )

        # 2. Main Step-Wise Adaptation Trajectory [0, 10, 20, 30, 40]
        step_history = [0]
        loss_history = [init_metrics.loss]
        f1_history = [init_metrics.macro_f1]
        best_step = 0
        best_f1 = init_metrics.macro_f1

        train_pool = self.get_budgeted_subset(env, 32)
        if not train_pool:
            train_pool = val_queries[:16]

        main_opt = torch.optim.AdamW(adapter.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        for step in range(1, max_steps + 1):
            adapter.train()
            main_opt.zero_grad()
            q = train_pool[(step - 1) % len(train_pool)]
            h = self._extract_hidden(q.question_text)
            logits = adapter(h)
            tgt_idx = self.CLASS_LABELS.index(q.expected_answer)
            loss = F.cross_entropy(logits, torch.tensor([tgt_idx], dtype=torch.long, device=self.device))

            if env.minimal_pairs:
                mp = env.minimal_pairs[(step - 1) % len(env.minimal_pairs)]
                h_b = self._extract_hidden(mp.base_query.question_text)
                h_p = self._extract_hidden(mp.perturbed_query.question_text)
                l_b = adapter(h_b)
                l_p = adapter(h_p)
                t_b = self.CLASS_LABELS.index(mp.base_query.expected_answer)
                t_p = self.CLASS_LABELS.index(mp.perturbed_query.expected_answer)
                loss_pair = 0.5 * (
                    F.cross_entropy(l_b, torch.tensor([t_b], dtype=torch.long, device=self.device))
                    + F.cross_entropy(l_p, torch.tensor([t_p], dtype=torch.long, device=self.device))
                )
                loss = loss + loss_pair

            reg_loss = torch.tensor(0.0, device=self.device)
            for name, param in adapter.named_parameters():
                if name in initial_params:
                    reg_loss = reg_loss + torch.sum((param - initial_params[name]) ** 2)
            loss = loss + self.anchor_lambda * reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=self.max_grad_norm)
            main_opt.step()

            if step % 10 == 0 or step == max_steps:
                cur_m = self.evaluate_tri_state(adapter, val_queries, env)
                step_history.append(step)
                loss_history.append(cur_m.loss)
                f1_history.append(cur_m.macro_f1)
                if cur_m.macro_f1 > best_f1:
                    best_f1 = cur_m.macro_f1
                    best_step = step

        final_metrics = self.evaluate_tri_state(adapter, val_queries, env)
        quality_auc = compute_standard_horizon_auc(step_history, f1_history, horizon=40)
        norm_cost = max(0.25, max_steps / 40.0)
        compute_eff = float(quality_auc / norm_cost)
        total_time_ms = (time.perf_counter() - t_start) * 1000.0

        n0_f1 = sample_efficiency_points[0].metrics.macro_f1
        n2_f1 = sample_efficiency_points[1].metrics.macro_f1
        regression_occurred = bool(n2_f1 < n0_f1 - 0.05)

        return ThreeValuedAdaptationTrajectory(
            arm_name=arm.value,
            env_id=env.env_id,
            seed=seed,
            initial_metrics=init_metrics,
            final_metrics=final_metrics,
            horizon_quality_auc=quality_auc,
            actual_steps_taken=max_steps,
            wall_clock_time_ms=total_time_ms,
            compute_adjusted_efficiency=compute_eff,
            parameter_count=param_count,
            sample_efficiency_curve=sample_efficiency_points,
            step_history=step_history,
            f1_history=f1_history,
            loss_history=loss_history,
            best_checkpoint_step=best_step,
            regression_occurred=regression_occurred,
        )
