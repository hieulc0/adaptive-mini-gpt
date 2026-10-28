"""Optimization instability forensics and 4-arm contrast for Stage A.8 (Task 2).

Preregistered 4-arm contrast:
- OPT_0_BASELINE: Unchanged Stage A.7 (lr=0.002, clip=1.0)
- OPT_1_COSINE_FLOOR: Cosine LR 0.002 -> 0.0001, beta2=0.98, eps=1e-6, clip=1.0
- OPT_2_DECOUPLED_TIMESCALE: Backbone lr=0.0005 clip=0.5; Readout lr=0.0015 -> 0.00015 clip=1.0
- OPT_3_LOGIT_BOUNDED_BCE: OPT_2 + logit L2 penalty (1e-4) + balanced state weights

Instruments LearningState.step via an opt-in callback logging trajectory forensics.
Saves coverage_5800.pt during OPT_0_BASELINE for Stage 1 evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from frame_a2_common import fork_owner
from run_attribute_learning import evaluate, file_hash
from run_stable_attributes import compact, enrich

from adaptive_mini_gpt.data.attribute_learning import TAG_IDS
from adaptive_mini_gpt.training.learning_state import LearningState, TwoTimescaleScheduler
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda

SPEC_PATH = Path("data/specs/optimization_a8_v1.json")


class OptimizationTelemetryHook:
    """Non-intrusive trajectory forensics hook for LearningState.step."""

    def __init__(self, arm_name: str, total_steps: int = 7200) -> None:
        self.arm_name = arm_name
        self.total_steps = total_steps
        self.logs: list[dict[str, Any]] = []
        self.last_theta: dict[str, torch.Tensor] = {}
        self.step_counter = 0

    def should_log(self, step: int) -> bool:
        if step % 100 == 0:
            return True
        if 5400 <= step <= 7200 and step % 20 == 0:
            return True
        return False

    def pre_clip(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: float,
        state_logits: torch.Tensor,
        tag_logits: torch.Tensor,
        tags: torch.Tensor,
        attribute_ids: Any,
    ) -> None:
        self.step_counter += 1
        if not self.should_log(self.step_counter):
            return

        # 1. Per-module pre-clip gradient norms
        base = getattr(model, "student", None)
        base_model = getattr(base, "base_model", None) if base else None
        bottleneck = getattr(model, "bottleneck", None)

        def param_norm(params: list[torch.Tensor]) -> float:
            grads = [p.grad for p in params if p.grad is not None]
            if not grads:
                return 0.0
            return float(torch.norm(torch.stack([torch.norm(g) for g in grads])).item())

        g_wte = (
            param_norm(list(base_model.wte.parameters()))
            if base_model and hasattr(base_model, "wte")
            else 0.0
        )
        g_block4_attn = (
            param_norm(list(base_model.blocks[-1].attn.parameters()))
            if base_model and hasattr(base_model, "blocks") and len(base_model.blocks) > 0
            else 0.0
        )
        g_block4_mlp = (
            param_norm(list(base_model.blocks[-1].mlp.parameters()))
            if base_model and hasattr(base_model, "blocks") and len(base_model.blocks) > 0
            else 0.0
        )
        g_proj = (
            param_norm(list(bottleneck.concept_proj.parameters()))
            if bottleneck and hasattr(bottleneck, "concept_proj")
            else 0.0
        )
        g_head = (
            param_norm(list(bottleneck.attribute_head.parameters()))
            if bottleneck and hasattr(bottleneck, "attribute_head")
            else 0.0
        )

        all_grads = [p.grad for p in model.parameters() if p.grad is not None]
        total_grad_norm = (
            float(torch.norm(torch.stack([torch.norm(g) for g in all_grads])).item())
            if all_grads
            else 0.0
        )
        clipped = bool(total_grad_norm > 1.0)

        # 2. Adam second-moment (exp_avg_sq) diagnostics
        v_values = []
        step_ages = []
        for p in model.parameters():
            if p in optimizer.state:
                st = optimizer.state[p]
                if "exp_avg_sq" in st:
                    v_values.append(st["exp_avg_sq"].detach().flatten())
                if "step" in st:
                    step_val = st["step"]
                    step_ages.append(
                        int(step_val.item() if isinstance(step_val, torch.Tensor) else step_val)
                    )

        if v_values:
            v_all = torch.cat(v_values)
            v_min = float(v_all.min().item())
            v_median = float(v_all.median().item())
            v_max = float(v_all.max().item())
        else:
            v_min, v_median, v_max = 0.0, 0.0, 0.0

        entry: dict[str, Any] = {
            "step": self.step_counter,
            "loss": loss,
            "total_grad_norm": total_grad_norm,
            "clipped": clipped,
            "grad_norms": {
                "wte": g_wte,
                "block4_attn": g_block4_attn,
                "block4_mlp": g_block4_mlp,
                "concept_proj": g_proj,
                "attribute_head": g_head,
            },
            "adam_second_moment": {
                "min": v_min,
                "median": v_median,
                "max": v_max,
            },
            "step_ages": {
                "min": min(step_ages) if step_ages else 0,
                "max": max(step_ages) if step_ages else 0,
            },
        }
        self.logs.append(entry)

    def pre_step(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        if not self.should_log(self.step_counter):
            return
        self.last_theta = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }

    def post_step(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss: float) -> None:
        if not self.should_log(self.step_counter) or not self.last_theta:
            return

        # 3. Per-module effective Adam update ratio ||theta_t - theta_{t-1}||_2 / (||theta_{t-1}||_2 + 1e-12)
        ratios: dict[str, float] = {}
        for name, p in model.named_parameters():
            if name in self.last_theta:
                delta = torch.norm(p.detach() - self.last_theta[name]).item()
                orig = torch.norm(self.last_theta[name]).item()
                ratios[name] = float(delta / (orig + 1e-12))

        # Aggregate by key modules
        wte_ratios = [v for k, v in ratios.items() if "wte" in k]
        b4_ratios = [v for k, v in ratios.items() if "blocks.4" in k or "blocks.3" in k]
        readout_ratios = [
            v for k, v in ratios.items() if "concept_proj" in k or "attribute_head" in k
        ]

        if self.logs and self.logs[-1]["step"] == self.step_counter:
            self.logs[-1]["update_ratios"] = {
                "wte": sum(wte_ratios) / len(wte_ratios) if wte_ratios else 0.0,
                "block4": sum(b4_ratios) / len(b4_ratios) if b4_ratios else 0.0,
                "readout": sum(readout_ratios) / len(readout_ratios) if readout_ratios else 0.0,
            }
        self.last_theta.clear()


def build_optimizer_and_clipper(
    model: nn.Module,
    arm_name: str,
    spec: dict[str, Any],
) -> tuple[torch.optim.Optimizer, Any, Any]:
    """Builds the configured optimizer, scheduler, and clipping function for the specified arm."""
    arm_cfg = spec["arms"][arm_name]

    if arm_name == "OPT_0_BASELINE":
        lr = arm_cfg["lr"]
        betas = tuple(arm_cfg["betas"])
        eps = arm_cfg["eps"]
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=0.0,
        )
        scheduler = None

        def clipper_0(m: nn.Module, opt: torch.optim.Optimizer) -> None:
            torch.nn.utils.clip_grad_norm_(m.parameters(), arm_cfg["clip"])

        return optimizer, scheduler, clipper_0

    elif arm_name == "OPT_1_COSINE_FLOOR":
        lr_start = arm_cfg["lr_start"]
        lr_min = arm_cfg["lr_min"]
        betas = tuple(arm_cfg["betas"])
        eps = arm_cfg["eps"]
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr_start,
            betas=betas,
            eps=eps,
            weight_decay=0.0,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=spec["steps"], eta_min=lr_min
        )

        def clipper_1(m: nn.Module, opt: torch.optim.Optimizer) -> None:
            torch.nn.utils.clip_grad_norm_(m.parameters(), arm_cfg["clip"])

        return optimizer, scheduler, clipper_1

    elif arm_name in ("OPT_2_DECOUPLED_TIMESCALE", "OPT_3_LOGIT_BOUNDED_BCE"):
        backbone_params = []
        readout_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(part in name for part in ("wte", "blocks.4", "blocks.3", "ln_f")):
                backbone_params.append(p)
            else:
                readout_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": arm_cfg["backbone_lr"], "name": "backbone"},
                {"params": readout_params, "lr": arm_cfg["readout_lr_start"], "name": "readout"},
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )

        scheduler = TwoTimescaleScheduler(
            optimizer,
            total_steps=spec["steps"],
            backbone_lr=arm_cfg["backbone_lr"],
            readout_lr_start=arm_cfg["readout_lr_start"],
            readout_lr_min=arm_cfg["readout_lr_min"],
        )

        def decoupled_clipper(m: nn.Module, opt: torch.optim.Optimizer) -> None:
            torch.nn.utils.clip_grad_norm_(backbone_params, arm_cfg["backbone_clip"])
            torch.nn.utils.clip_grad_norm_(readout_params, arm_cfg["readout_clip"])

        return optimizer, scheduler, decoupled_clipper

    raise ValueError(f"Unknown optimization arm: {arm_name}")


def run_optimization_diagnostics(
    parent_path: Path,
    training_data: dict[str, list[Any]],
    shared_data: dict[str, list[Any]],
    plans: dict[str, list[list[int]]],
    batches: dict[str, Any],
    shared_batches: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Runs the preregistered 4-arm optimization contrast across updates 1..7200."""
    require_remote_cuda()
    spec = json.loads(SPEC_PATH.read_text())
    parent = LearningState.load(parent_path)

    coverage_rows = training_data["coverage"]
    plan = plans["coverage"]
    batch = batches["coverage"]

    results: dict[str, Any] = {
        "spec": spec,
        "arms": {},
    }

    eval_sets = {
        "training": coverage_rows[:756],
        "synthetic": coverage_rows[756:],
        **shared_data,
    }
    eval_batches = {
        "training": tuple(v[:756] for v in batch),
        "synthetic": tuple(v[756:] for v in batch),
        **shared_batches,
    }

    for arm_name in (
        "OPT_0_BASELINE",
        "OPT_1_COSINE_FLOOR",
        "OPT_2_DECOUPLED_TIMESCALE",
        "OPT_3_LOGIT_BOUNDED_BCE",
    ):
        owner = fork_owner(parent, 42)
        optimizer, scheduler, clipper = build_optimizer_and_clipper(owner.model, arm_name, spec)
        owner.optimizer = optimizer
        owner.scheduler = scheduler

        telemetry = OptimizationTelemetryHook(arm_name, total_steps=spec["steps"])
        arm_cfg = spec["arms"][arm_name]
        logit_penalty = arm_cfg.get("logit_penalty", 0.0)
        state_weights = arm_cfg.get("balanced_weights", None)

        curves: list[dict[str, Any]] = []
        for step in range(1, spec["steps"] + 1):
            cursor = (step - 1) % len(plan)
            ix = plan[cursor]

            loss = owner.step(
                *[v[ix] for v in batch],
                attribute_ids=TAG_IDS,
                diagnostic_callback=telemetry,
                clipper=clipper,
                logit_penalty=logit_penalty,
                state_weights=state_weights,
            )

            # Checkpoint at 5800 during OPT_0_BASELINE
            if arm_name == "OPT_0_BASELINE" and step == 5800:
                ckpt_5800 = out_dir / "coverage_5800.pt"
                owner.save(ckpt_5800)

            # Periodic full evaluation at 1200, 2400, 5800, 6800, 7000, 7100, 7200
            if step in (1200, 2400, 5800, 6800, 7000, 7100, 7200):
                metrics = {
                    k: enrich(evaluate(owner.model, eval_batches[k], rs))
                    for k, rs in eval_sets.items()
                }
                curves.append(
                    {
                        "step": step,
                        "loss": loss,
                        "training_exact": metrics["training"]["exact"],
                        "old_known_exact": metrics["old_known"]["exact"],
                        "old_renamed_exact": metrics["old_renamed"]["exact"],
                        "development_familiar_exact": metrics["development_familiar"]["exact"],
                        "development_novel_exact": metrics["development_novel"]["exact"],
                    }
                )

        # Save arm endpoint checkpoint
        endpoint_path = out_dir / f"{arm_name.lower()}_7200.pt"
        owner.save(endpoint_path)

        final_metrics = {
            k: enrich(evaluate(owner.model, eval_batches[k], rs)) for k, rs in eval_sets.items()
        }
        results["arms"][arm_name] = {
            "endpoint_checkpoint": str(endpoint_path),
            "endpoint_sha256": file_hash(endpoint_path),
            "curves": curves,
            "final_metrics": {k: compact(v) for k, v in final_metrics.items()},
            "telemetry_log_count": len(telemetry.logs),
            "telemetry": telemetry.logs,
        }
        del owner

    (out_dir / "optimization_diagnostic_results_a8.json").write_text(json.dumps(results, indent=2))
    return results
