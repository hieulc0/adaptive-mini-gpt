"""Learned Action Policy, Candidate Scoring Head, and Residual Adapters (M19.2 & M19.3).

Supports controlled representation adaptation arms:
- Arm A: Frozen backbone + Action Head
- Arm B: Frozen backbone + Trainable Residual Bottleneck Adapters + Action Head
- Arm C: Fully Fine-tuned Backbone + Action Head

Guarantees:
- Single-pass context representation h_ctx from Mini-GPT backbone.
- Complete action scoring avoiding inconsistent decomposed predictions.
- Soft-target multi-label cross entropy supporting commutative and parallel DAG equivalence.
- Strictly isolated baseline comparison: Uniform vs Deterministic vs Learned Policy vs Oracle.
- Complete rollout evaluation measuring free-running plan success.
- Unrounded language loss retention tracking to measure representation drift.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.action_dataset import ActionTrace, ActionTraceDataset
from adaptive_mini_gpt.rules.executor import RuleExecutor
from adaptive_mini_gpt.rules.grammar_planner import PlanGrammar, TargetGoal
from adaptive_mini_gpt.rules.memory import GroundedSlot, WorkingMemory
from adaptive_mini_gpt.rules.multistep_dataset import MultiStepProblem, format_available_slots
from adaptive_mini_gpt.rules.plan import ExecutionPlan, PlanStep

logger = logging.getLogger(__name__)

# Known operations and units vocabulary for structural embedding
OPERATION_VOCAB = ["add", "average_speed", "compare", "reverse", "unknown"]
UNIT_VOCAB = ["km", "hours", "km/h", "none", "other"]


def format_context_prompt(
    question: str,
    slots: dict[str, GroundedSlot],
    executed_steps: list[PlanStep],
) -> str:
    """Format state prefix (q, s_t) for causal Transformer ingestion."""
    slots_text = format_available_slots(slots).strip()
    user_body = f"{question}\n{slots_text}" if slots_text else question
    lines = [f"<|user|>\n{user_body}\n<|assistant|>\nPlan:"]
    for idx, s in enumerate(executed_steps, start=1):
        u_str = f" [{s.units}]" if s.units else ""
        args_formatted = ", ".join(
            f"{k}={repr(v) if isinstance(v, str) else v}" for k, v in s.args.items()
        )
        lines.append(f"{idx}. {s.output_var} = {s.operation}({args_formatted}){u_str}")

    next_idx = len(executed_steps) + 1
    return "\n".join(lines) + f"\n{next_idx}. "


class ResidualAdapter(nn.Module):
    """Lightweight bottleneck residual adapter for Transformer blocks (M19.3 Arm B)."""

    def __init__(self, d_model: int = 128, d_bottleneck: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, d_bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(d_bottleneck, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Zero-initialize up-projection so adapter is identity at initialization
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.dropout(self.up(self.act(self.down(self.ln(x)))))


def extract_context_representation(
    model: MiniGPT,
    input_ids: Tensor,
    adapters: nn.ModuleList | None = None,
    bypass_adapters: bool = False,
) -> Tensor:
    """Extract causal context representation h_ctx (B, d_model) from Mini-GPT.
    
    Uses the final hidden state of the prompt sequence (at the decision position).
    Safely bounds input length to model context length.
    """
    ctx_len = model.config.context_length
    if input_ids.size(1) > ctx_len:
        input_ids = input_ids[:, -ctx_len:]

    # Clamp token ids to valid embedding table range
    input_ids = torch.clamp(input_ids, 0, model.config.vocab_size - 1)

    b, t = input_ids.size()
    device = input_ids.device
    pos = torch.arange(0, t, dtype=torch.long, device=device)
    tok_emb = model.wte(input_ids)
    pos_emb = model.wpe(pos)
    x = model.drop(tok_emb + pos_emb)

    for i, block in enumerate(model.blocks):
        x, _ = block(x, bypass_moe=True)
        if adapters is not None and not bypass_adapters:
            x = adapters[i](x)

    h = model.ln_f(x)
    return h[:, -1, :]  # (B, d_model)


class CandidateActionFeaturizer(nn.Module):
    """Featurizes complete valid candidate actions using token semantics and structural types."""

    def __init__(self, d_model: int = 128, d_struct: int = 32) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_struct = d_struct

        self.op_emb = nn.Embedding(len(OPERATION_VOCAB), d_struct)
        self.unit_emb = nn.Embedding(len(UNIT_VOCAB), d_struct)
        self.meta_linear = nn.Linear(6, d_struct)

        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model + d_struct, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def _get_op_idx(self, op: str) -> int:
        return OPERATION_VOCAB.index(op) if op in OPERATION_VOCAB else len(OPERATION_VOCAB) - 1

    def _get_unit_idx(self, unit: str | None) -> int:
        if unit is None:
            return UNIT_VOCAB.index("none")
        if unit in ("km", "kilometer", "kilometers"):
            return UNIT_VOCAB.index("km")
        if unit in ("h", "hour", "hours", "hrs"):
            return UNIT_VOCAB.index("hours")
        if unit == "km/h":
            return UNIT_VOCAB.index("km/h")
        return UNIT_VOCAB.index("other")

    def forward(
        self,
        candidate_steps: list[PlanStep],
        model: MiniGPT,
        tokenizer: BPETokenizer,
        device: torch.device,
    ) -> Tensor:
        """Encode K candidate steps into action representations (K, d_model)."""
        k = len(candidate_steps)
        if k == 0:
            return torch.empty((0, self.d_model), device=device)

        op_ids: list[int] = []
        unit_ids: list[int] = []
        meta_features: list[list[float]] = []
        text_embeddings: list[Tensor] = []

        wte_weight = model.wte.weight  # (vocab_size, d_model)

        for step in candidate_steps:
            op_ids.append(self._get_op_idx(step.operation))
            unit_ids.append(self._get_unit_idx(step.units))

            # Structural metadata flags
            n_args = float(len(step.args))
            arg_vals = list(step.args.values())
            is_slot_0 = 1.0 if (len(arg_vals) > 0 and str(arg_vals[0]).startswith("$") and not str(arg_vals[0]).startswith("$step") and not str(arg_vals[0]).startswith("$sum")) else 0.0
            is_var_0 = 1.0 if (len(arg_vals) > 0 and (str(arg_vals[0]).startswith("$step") or str(arg_vals[0]).startswith("$sum"))) else 0.0
            is_slot_1 = 1.0 if (len(arg_vals) > 1 and str(arg_vals[1]).startswith("$") and not str(arg_vals[1]).startswith("$step") and not str(arg_vals[1]).startswith("$sum")) else 0.0
            is_var_1 = 1.0 if (len(arg_vals) > 1 and (str(arg_vals[1]).startswith("$step") or str(arg_vals[1]).startswith("$sum"))) else 0.0
            same_type = 1.0 if (is_slot_0 == is_slot_1 and is_var_0 == is_var_1) else 0.0
            meta_features.append([n_args, is_slot_0, is_var_0, is_slot_1, is_var_1, same_type])

            # Textual token embedding
            cand_text = f"{step.operation}({', '.join(f'{k}={v}' for k, v in sorted(step.args.items()))})"
            cand_tokens = tokenizer.encode(cand_text)
            if cand_tokens:
                cand_tokens = [min(tid, model.config.vocab_size - 1) for tid in cand_tokens]
                t_tensor = torch.tensor(cand_tokens, dtype=torch.long, device=device)
                tok_embs = F.embedding(t_tensor, wte_weight)
                text_embeddings.append(tok_embs.mean(dim=0))
            else:
                text_embeddings.append(torch.zeros(self.d_model, device=device))

        op_tensor = torch.tensor(op_ids, dtype=torch.long, device=device)
        unit_tensor = torch.tensor(unit_ids, dtype=torch.long, device=device)
        meta_tensor = torch.tensor(meta_features, dtype=torch.float32, device=device)

        struct_emb = self.op_emb(op_tensor) + self.unit_emb(unit_tensor) + self.meta_linear(meta_tensor)
        tok_emb_stack = torch.stack(text_embeddings, dim=0)

        fused = self.fusion_proj(torch.cat([tok_emb_stack, struct_emb], dim=-1))
        return fused  # (K, d_model)


class ActionScoringHead(nn.Module):
    """Lightweight 2-layer candidate scoring head: score_theta(q, s, a)."""

    def __init__(self, d_model: int = 128, d_hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        in_dim = d_model * 4  # [h_ctx, e_a, h_ctx * e_a, |h_ctx - e_a|]

        self.net = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, h_ctx: Tensor, e_actions: Tensor) -> Tensor:
        """Compute scalar compatibility scores for K candidate actions."""
        k = e_actions.size(0)
        if k == 0:
            return torch.empty(0, device=h_ctx.device)

        h_exp = h_ctx.expand(k, -1)  # (K, d_model)
        feat = torch.cat([h_exp, e_actions, h_exp * e_actions, torch.abs(h_exp - e_actions)], dim=-1)
        scores = self.net(feat).squeeze(-1)  # (K,)
        return scores


class ActionPolicyModel(nn.Module):
    """Unified Action Policy supporting Arm A (Frozen), Arm B (Residual Adapter), and Arm C (Full)."""

    def __init__(
        self,
        backbone: MiniGPT,
        tokenizer: BPETokenizer,
        d_hidden: int = 128,
        dropout: float = 0.1,
        adapter_mode: str = "none",  # "none" (Arm A) | "residual" (Arm B) | "full" (Arm C)
        d_bottleneck: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.d_model = backbone.config.d_model
        self.adapter_mode = adapter_mode

        if adapter_mode == "none":
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            self.adapters: nn.ModuleList | None = None
        elif adapter_mode == "residual":
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            self.adapters = nn.ModuleList(
                [ResidualAdapter(self.d_model, d_bottleneck, dropout) for _ in range(backbone.config.n_layers)]
            )
        elif adapter_mode == "full":
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.backbone.train()
            self.adapters = None
        else:
            raise ValueError(f"Unknown adapter_mode '{adapter_mode}'. Expected 'none', 'residual', or 'full'.")

        self.featurizer = CandidateActionFeaturizer(d_model=self.d_model, d_struct=32)
        self.head = ActionScoringHead(d_model=self.d_model, d_hidden=d_hidden, dropout=dropout)

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_frozen_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def score_candidates(
        self,
        prompt: str,
        candidates: list[PlanStep],
        device: torch.device,
        bypass_adapters: bool = False,
    ) -> Tensor:
        """Score a list of valid candidate actions given state context prompt."""
        if not candidates:
            return torch.empty(0, device=device)

        token_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        if self.adapter_mode == "none":
            with torch.no_grad():
                h_ctx = extract_context_representation(self.backbone, input_ids, self.adapters, bypass_adapters)
        else:
            h_ctx = extract_context_representation(self.backbone, input_ids, self.adapters, bypass_adapters)

        e_actions = self.featurizer(candidates, self.backbone, self.tokenizer, device)
        scores = self.head(h_ctx, e_actions)
        return scores

    def evaluate_language_loss(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
        bypass_adapters: bool = False,
    ) -> float:
        """Evaluate cross-entropy loss on pre-training language prose to track retention."""
        b, t = token_ids.size()
        device = token_ids.device
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.backbone.wte(token_ids)
        pos_emb = self.backbone.wpe(pos)
        x = self.backbone.drop(tok_emb + pos_emb)

        for i, block in enumerate(self.backbone.blocks):
            x, _ = block(x, bypass_moe=True)
            if self.adapters is not None and not bypass_adapters:
                x = self.adapters[i](x)

        h = self.backbone.ln_f(x)
        logits = self.backbone.lm_head(h)
        tgt = targets if targets is not None else token_ids
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-1)
        return ce_loss.item()


def compute_action_loss(
    logits: Tensor,
    correct_indices: list[int],
) -> Tensor:
    """Multi-target cross-entropy over verified correct candidate action indices.
    
    L = logsumexp(z) - (1 / |C|) * sum_{i in C} z_i
    """
    if logits.numel() == 0 or not correct_indices:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    logsumexp_z = torch.logsumexp(logits, dim=-1)
    target_logits = logits[correct_indices]
    mean_target_logit = target_logits.mean()
    loss = logsumexp_z - mean_target_logit
    return loss


@dataclass
class EvaluationMetrics:
    """Benchmark metrics comparing candidate ranking strategies."""

    uniform_acc: float
    deterministic_acc: float
    policy_acc: float
    oracle_acc: float
    mean_loss: float
    total_traces: int
    archetype_accuracies: dict[str, dict[str, float]]
    position_accuracies: dict[int, dict[str, float]]
    decision_latency_ms: float
    rollout_policy_acc: float = 0.0
    rollout_deterministic_acc: float = 0.0
    policy_ece: float = 0.0
    policy_brier: float = 0.0


def rollout_problem(
    policy_model: ActionPolicyModel,
    problem: MultiStepProblem,
    device: torch.device,
    max_steps: int = 6,
    use_policy: bool = True,
    executor: RuleExecutor | None = None,
) -> tuple[bool, int, list[PlanStep]]:
    """Simulate free-running plan rollout using learned policy or deterministic heuristic.
    
    Returns:
        (is_successful, steps_taken, executed_steps)
    """
    rule_executor = executor or RuleExecutor()
    slots = problem.slots
    goal = TargetGoal.from_question(problem.question, slots)

    live_mem = WorkingMemory()
    live_mem.slots = copy.copy(slots)
    executed_steps: list[PlanStep] = []

    for _ in range(max_steps):
        candidates = PlanGrammar.get_candidate_steps(
            slots=slots,
            working_memory=live_mem,
            executed_steps=executed_steps,
            max_steps=max_steps,
            question=problem.question,
        )
        if not candidates:
            break

        if use_policy:
            prompt = format_context_prompt(problem.question, slots, executed_steps)
            with torch.no_grad():
                scores = policy_model.score_candidates(prompt, candidates, device)
            if scores.numel() == 0:
                break
            chosen_cand = candidates[int(torch.argmax(scores).item())]
        else:
            chosen_cand = candidates[0]

        resolved: dict[str, Any] = {}
        for k, v in chosen_cand.args.items():
            if isinstance(v, str) and v.startswith("$"):
                r_name = v[1:].strip()
                resolved[k] = live_mem.variables.get(
                    r_name, live_mem.slots[r_name].value if r_name in live_mem.slots else None
                )
            else:
                resolved[k] = v

        exec_res = rule_executor.execute(chosen_cand.operation, resolved, verify=True)
        if not exec_res.success or not exec_res.verified:
            break

        live_mem.bind_step_variable(
            var_name=chosen_cand.output_var,
            value=exec_res.result,
            operation=chosen_cand.operation,
            raw_args=chosen_cand.args,
            resolved_args=resolved,
            units=chosen_cand.units,
            step_index=len(executed_steps) + 1,
        )
        live_mem.bind_result(exec_res.result, verified=True)
        executed_steps.append(chosen_cand)

        plan = ExecutionPlan(steps=executed_steps, final_var=chosen_cand.output_var)
        sem_ok, _ = goal.evaluate_semantics(plan, live_mem)
        if sem_ok:
            return True, len(executed_steps), executed_steps

    plan = ExecutionPlan(steps=executed_steps, final_var=executed_steps[-1].output_var if executed_steps else None)
    sem_ok, _ = goal.evaluate_semantics(plan, live_mem)
    return sem_ok, len(executed_steps), executed_steps


def evaluate_action_ranking(
    policy_model: ActionPolicyModel,
    dataset: ActionTraceDataset,
    device: torch.device,
    problems_for_rollout: list[MultiStepProblem] | None = None,
) -> EvaluationMetrics:
    """Evaluate candidate action ranking across baselines on a trace dataset and problems."""
    policy_model.eval()

    n_total = len(dataset)
    if n_total == 0:
        return EvaluationMetrics(0.0, 0.0, 0.0, 1.0, 0.0, 0, {}, {}, 0.0, 0.0, 0.0)

    uniform_correct = 0.0
    det_correct = 0
    policy_correct = 0
    total_loss = 0.0
    calibration_confidences: list[float] = []
    calibration_correct: list[float] = []

    arch_stats: dict[str, dict[str, Any]] = {}
    pos_stats: dict[int, dict[str, Any]] = {}
    latencies: list[float] = []

    for trace in dataset:
        prompt = trace.format_context_prompt()
        candidates = trace.valid_candidates
        correct_set = set(trace.correct_action_indices)
        k = len(candidates)
        arch = trace.archetype
        pos = trace.step_index

        if arch not in arch_stats:
            arch_stats[arch] = {"total": 0, "uniform": 0.0, "det": 0, "policy": 0}
        arch_stats[arch]["total"] += 1

        if pos not in pos_stats:
            pos_stats[pos] = {"total": 0, "uniform": 0.0, "det": 0, "policy": 0}
        pos_stats[pos]["total"] += 1

        # 1. Uniform baseline
        p_uniform = len(correct_set) / max(k, 1)
        uniform_correct += p_uniform
        arch_stats[arch]["uniform"] += p_uniform
        pos_stats[pos]["uniform"] += p_uniform

        # 2. Deterministic baseline
        is_det = 1 if 0 in correct_set else 0
        det_correct += is_det
        arch_stats[arch]["det"] += is_det
        pos_stats[pos]["det"] += is_det

        # 3. Policy head scoring
        t0 = time.perf_counter()
        with torch.no_grad():
            scores = policy_model.score_candidates(prompt, candidates, device)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        if scores.numel() > 0:
            loss = compute_action_loss(scores, trace.correct_action_indices)
            total_loss += loss.item()

            probabilities = torch.softmax(scores, dim=-1)
            confidence, pred_tensor = torch.max(probabilities, dim=-1)
            pred_idx = int(pred_tensor.item())
            is_policy = 1 if pred_idx in correct_set else 0
            calibration_confidences.append(float(confidence.item()))
            calibration_correct.append(float(is_policy))
            policy_correct += is_policy
            arch_stats[arch]["policy"] += is_policy
            pos_stats[pos]["policy"] += is_policy

    # Aggregate archetype metrics
    archetype_accuracies: dict[str, dict[str, float]] = {}
    for arch, s in arch_stats.items():
        tot = max(s["total"], 1)
        archetype_accuracies[arch] = {
            "uniform": round(s["uniform"] / tot * 100.0, 2),
            "deterministic": round(s["det"] / tot * 100.0, 2),
            "policy": round(s["policy"] / tot * 100.0, 2),
            "traces": s["total"],
        }

    # Aggregate position metrics
    position_accuracies: dict[int, dict[str, float]] = {}
    for pos, s in sorted(pos_stats.items()):
        tot = max(s["total"], 1)
        position_accuracies[pos] = {
            "uniform": round(s["uniform"] / tot * 100.0, 2),
            "deterministic": round(s["det"] / tot * 100.0, 2),
            "policy": round(s["policy"] / tot * 100.0, 2),
            "traces": s["total"],
        }

    # Free-running rollout evaluation
    rollout_policy_acc = 0.0
    rollout_det_acc = 0.0
    if problems_for_rollout:
        executor = RuleExecutor()
        n_probs = len(problems_for_rollout)
        p_ok_count = 0
        d_ok_count = 0
        for p in problems_for_rollout:
            p_ok, _, _ = rollout_problem(policy_model, p, device, use_policy=True, executor=executor)
            d_ok, _, _ = rollout_problem(policy_model, p, device, use_policy=False, executor=executor)
            if p_ok:
                p_ok_count += 1
            if d_ok:
                d_ok_count += 1
        rollout_policy_acc = round(p_ok_count / n_probs * 100.0, 2)
        rollout_det_acc = round(d_ok_count / n_probs * 100.0, 2)

    # Calibrate confidence in the selected action's validity, using fixed-width bins.
    policy_ece = 0.0
    n_calibration = len(calibration_confidences)
    if n_calibration:
        n_bins = 10
        for bin_index in range(n_bins):
            lower = bin_index / n_bins
            upper = (bin_index + 1) / n_bins
            indices = [
                i
                for i, confidence in enumerate(calibration_confidences)
                if lower <= confidence < upper or (bin_index == n_bins - 1 and confidence == 1.0)
            ]
            if indices:
                mean_confidence = sum(calibration_confidences[i] for i in indices) / len(indices)
                mean_accuracy = sum(calibration_correct[i] for i in indices) / len(indices)
                policy_ece += len(indices) / n_calibration * abs(mean_accuracy - mean_confidence)
        policy_brier = sum(
            (confidence - correct) ** 2
            for confidence, correct in zip(calibration_confidences, calibration_correct, strict=True)
        ) / n_calibration
    else:
        policy_brier = 0.0

    return EvaluationMetrics(
        uniform_acc=round(uniform_correct / n_total * 100.0, 2),
        deterministic_acc=round(det_correct / n_total * 100.0, 2),
        policy_acc=round(policy_correct / n_total * 100.0, 2),
        oracle_acc=100.0,
        mean_loss=round(total_loss / n_total, 4),
        total_traces=n_total,
        archetype_accuracies=archetype_accuracies,
        position_accuracies=position_accuracies,
        decision_latency_ms=round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        rollout_policy_acc=rollout_policy_acc,
        rollout_deterministic_acc=rollout_det_acc,
        policy_ece=round(policy_ece, 6),
        policy_brier=round(policy_brier, 6),
    )


def train_action_policy(
    policy_model: ActionPolicyModel,
    train_dataset: ActionTraceDataset,
    val_dataset: ActionTraceDataset,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_problems: list[MultiStepProblem] | None = None,
) -> dict[str, Any]:
    """Train the action policy model (Arm A, B, or C) using AdamW."""
    optimizer = AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_acc = -1.0
    best_weights: dict[str, Tensor] = {}
    training_history: list[dict[str, Any]] = []

    logger.info(
        "Beginning policy training [Mode: %s]: %d train traces, %d val traces, %d trainable parameters",
        policy_model.adapter_mode,
        len(train_dataset),
        len(val_dataset),
        policy_model.num_trainable_params,
    )

    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        policy_model.train()
        total_train_loss = 0.0
        train_correct = 0

        trace_indices = torch.randperm(len(train_dataset)).tolist()

        for idx in trace_indices:
            trace = train_dataset[idx]
            prompt = trace.format_context_prompt()
            candidates = trace.valid_candidates
            correct_set = set(trace.correct_action_indices)

            optimizer.zero_grad()
            scores = policy_model.score_candidates(prompt, candidates, device)
            loss = compute_action_loss(scores, trace.correct_action_indices)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pred_idx = int(torch.argmax(scores).item())
            if pred_idx in correct_set:
                train_correct += 1

        train_acc = round(train_correct / len(train_dataset) * 100.0, 2)
        train_loss = round(total_train_loss / len(train_dataset), 4)

        # Validation evaluation
        val_metrics = evaluate_action_ranking(
            policy_model, val_dataset, device, problems_for_rollout=val_problems if epoch == epochs else None
        )

        if val_metrics.policy_acc > best_val_acc:
            best_val_acc = val_metrics.policy_acc
            best_weights = {k: v.clone() for k, v in policy_model.state_dict().items()}

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics.mean_loss,
            "val_policy_acc": val_metrics.policy_acc,
            "val_policy_ece": val_metrics.policy_ece,
            "val_policy_brier": val_metrics.policy_brier,
            "val_det_acc": val_metrics.deterministic_acc,
            "val_uniform_acc": val_metrics.uniform_acc,
        }
        training_history.append(epoch_record)

    if best_weights:
        policy_model.load_state_dict(best_weights, strict=True)

    training_time_sec = round(time.perf_counter() - t_start, 2)

    return {
        "best_val_acc": best_val_acc,
        "training_time_sec": training_time_sec,
        "num_trainable_params": policy_model.num_trainable_params,
        "num_frozen_params": policy_model.num_frozen_params,
        "history": training_history,
    }
