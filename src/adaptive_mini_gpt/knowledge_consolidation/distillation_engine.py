"""Milestone M83 Phase 2: Neural Self-Distillation Engine and Compression Baselines.

Implements multi-teacher self-distillation into a consolidated shared adapter:
1. Soft-Target Knowledge Distillation: Teacher soft probability distributions (temperature T=2.0)
   combined with ground-truth cross-entropy to preserve functional nuance.
2. Comparative Compression Baselines:
   - INDEPENDENT_ORIGINALS: Uncompressed separate adapters.
   - CONSOLIDATED_DISTILLED: Shared bottleneck trunk + skill-specific residual heads.
   - LOW_RANK_SVD: Truncated SVD low-rank compression of individual adapters.
   - WEIGHT_AVERAGING: Direct arithmetic mean of teacher parameters.
   - RANDOM_MERGING: Convex combination of teacher parameters with random weights.
3. Per-Skill Retention Auditing:
   Enforces >= 95% retention of baseline performance, with an absolute floor of >= 85% EM on every skill.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.knowledge_consolidation.consolidated_adapter import (
    ConsolidatedSharedAdapter,
)
from adaptive_mini_gpt.learning_to_learn.capacity_adaptation import (
    BottleneckSequenceAdapter,
)

logger = logging.getLogger(__name__)

RETENTION_RATIO_GATE: float = 0.95
INDIVIDUAL_SKILL_FLOOR_EM: float = 0.85


class CompressionMethod(StrEnum):
    """Categorical consolidation and compression baselines."""

    INDEPENDENT_ORIGINALS = "INDEPENDENT_ORIGINALS"
    CONSOLIDATED_DISTILLED = "CONSOLIDATED_DISTILLED"
    LOW_RANK_SVD = "LOW_RANK_SVD"
    WEIGHT_AVERAGING = "WEIGHT_AVERAGING"
    RANDOM_MERGING = "RANDOM_MERGING"


@dataclass
class DistillationResult:
    """Outcome of knowledge consolidation trial across skills."""

    method: CompressionMethod
    baseline_parameters: int
    consolidated_parameters: int
    compression_pct: float
    per_skill_em: dict[str, float]
    mean_em: float
    relative_retention_ratio: float
    passed_retention_gate: bool
    parameter_hash: str
    elapsed_time: float
    consolidated_model: nn.Module | None


class NeuralSelfDistillationEngine:
    """Orchestrates self-distillation, baseline compressions, and retention qualification."""

    @classmethod
    def distill_skills(
        cls,
        teachers: dict[str, BottleneckSequenceAdapter],
        skill_data: dict[str, dict[str, torch.Tensor]],  # {skill_id: {"train_h": ..., "train_y": ..., "qual_h": ..., "qual_y": ...}}
        device: torch.device,
        d_model: int = 128,
        shared_rank: int = 8,
        residual_rank: int = 2,
        distill_steps: int = 40,
        lr: float = 6e-3,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ) -> DistillationResult:
        """Distill multiple teacher adapters into ConsolidatedSharedAdapter."""
        start_time = time.perf_counter()

        # Step 1: Measure baseline performance of uncompressed teachers
        baseline_params = sum(t.count_parameters() for t in teachers.values())
        baseline_ems: dict[str, float] = {}
        for skill_id, teacher in teachers.items():
            teacher.eval()
            with torch.no_grad():
                qh = skill_data[skill_id]["qual_h"]
                qy = skill_data[skill_id]["qual_y"]
                logits = teacher(qh)
                em = float((torch.argmax(logits, dim=-1) == qy).all(dim=-1).float().mean().item())
                baseline_ems[skill_id] = em
        baseline_mean_em = sum(baseline_ems.values()) / max(1, len(baseline_ems))

        # Step 2: Initialize student consolidated adapter
        student = ConsolidatedSharedAdapter(
            d_model=d_model,
            shared_rank=shared_rank,
            residual_rank=residual_rank,
            num_symbols=16,
        ).to(device)

        for skill_id in teachers.keys():
            student.register_skill_residual(skill_id)

        # Initialize student shared trunk with average teacher weights
        with torch.no_grad():
            for name, p in student.named_parameters():
                if "residuals" not in name:
                    matching = [t.state_dict()[name] for t in teachers.values() if name in t.state_dict()]
                    if matching:
                        avg_w = torch.stack(matching).mean(dim=0)
                        p.copy_(avg_w)

        optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)

        # Step 3: Self-Distillation Training Loop
        student.train()
        for step in range(1, distill_steps + 1):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)

            for skill_id, teacher in teachers.items():
                th = skill_data[skill_id]["train_h"]
                ty = skill_data[skill_id]["train_y"]

                # Teacher soft targets
                teacher.eval()
                with torch.no_grad():
                    teacher_logits = teacher(th)
                    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

                # Student outputs
                student_logits = student(th, skill_id=skill_id)
                student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

                # Combined loss: Task CE + KL Divergence
                ce_loss = F.cross_entropy(student_logits.view(-1, 16), ty.view(-1))
                kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

                skill_loss = (1.0 - alpha) * ce_loss + alpha * kl_loss
                total_loss = total_loss + skill_loss

            total_loss.backward()
            optimizer.step()

        # Step 4: Evaluate consolidated student across all skills
        student.eval()
        student_ems: dict[str, float] = {}
        with torch.no_grad():
            for skill_id in teachers.keys():
                qh = skill_data[skill_id]["qual_h"]
                qy = skill_data[skill_id]["qual_y"]
                logits = student(qh, skill_id=skill_id)
                em = float((torch.argmax(logits, dim=-1) == qy).all(dim=-1).float().mean().item())
                student_ems[skill_id] = round(em, 4)

        student_mean_em = sum(student_ems.values()) / max(1, len(student_ems))
        retention_ratio = (student_mean_em / baseline_mean_em) if baseline_mean_em > 0 else 0.0

        # Retention gates: relative retention >= 95% AND every individual skill >= 85%
        passed_retention = (retention_ratio >= RETENTION_RATIO_GATE) and all(
            em >= INDIVIDUAL_SKILL_FLOOR_EM for em in student_ems.values()
        )

        consolidated_params = student.count_parameters()
        compression_pct = (baseline_params - consolidated_params) / baseline_params * 100.0
        elapsed = time.perf_counter() - start_time

        return DistillationResult(
            method=CompressionMethod.CONSOLIDATED_DISTILLED,
            baseline_parameters=baseline_params,
            consolidated_parameters=consolidated_params,
            compression_pct=round(compression_pct, 2),
            per_skill_em=student_ems,
            mean_em=round(student_mean_em, 4),
            relative_retention_ratio=round(retention_ratio, 4),
            passed_retention_gate=passed_retention,
            parameter_hash=student.compute_parameter_hash(),
            elapsed_time=round(elapsed, 2),
            consolidated_model=student if passed_retention else None,
        )

    @classmethod
    def evaluate_weight_averaging_baseline(
        cls,
        teachers: dict[str, BottleneckSequenceAdapter],
        skill_data: dict[str, dict[str, torch.Tensor]],
        device: torch.device,
    ) -> DistillationResult:
        """Direct arithmetic parameter averaging control."""
        start_time = time.perf_counter()
        sample_teacher = next(iter(teachers.values()))
        avg_adapter = BottleneckSequenceAdapter(
            sample_teacher.d_model, sample_teacher.bottleneck_dim, sample_teacher.num_symbols
        ).to(device)

        with torch.no_grad():
            for name, p in avg_adapter.named_parameters():
                stack_w = torch.stack([t.state_dict()[name] for t in teachers.values()])
                p.copy_(stack_w.mean(dim=0))

        # Evaluate on each skill
        avg_adapter.eval()
        ems: dict[str, float] = {}
        with torch.no_grad():
            for skill_id in teachers.keys():
                qh = skill_data[skill_id]["qual_h"]
                qy = skill_data[skill_id]["qual_y"]
                logits = avg_adapter(qh)
                em = float((torch.argmax(logits, dim=-1) == qy).all(dim=-1).float().mean().item())
                ems[skill_id] = round(em, 4)

        mean_em = sum(ems.values()) / max(1, len(ems))
        baseline_params = sum(t.count_parameters() for t in teachers.values())
        consolidated_params = avg_adapter.count_parameters()
        comp_pct = (baseline_params - consolidated_params) / baseline_params * 100.0

        return DistillationResult(
            method=CompressionMethod.WEIGHT_AVERAGING,
            baseline_parameters=baseline_params,
            consolidated_parameters=consolidated_params,
            compression_pct=round(comp_pct, 2),
            per_skill_em=ems,
            mean_em=round(mean_em, 4),
            relative_retention_ratio=round(mean_em / 1.0, 4),
            passed_retention_gate=(mean_em >= 0.85),
            parameter_hash=avg_adapter.compute_parameter_hash(),
            elapsed_time=round(time.perf_counter() - start_time, 2),
            consolidated_model=avg_adapter,
        )

    @classmethod
    def evaluate_random_merging_baseline(
        cls,
        teachers: dict[str, BottleneckSequenceAdapter],
        skill_data: dict[str, dict[str, torch.Tensor]],
        device: torch.device,
        seed: int = 42,
    ) -> DistillationResult:
        """Random convex combination parameter merging control."""
        start_time = time.perf_counter()
        rng = torch.Generator(device=device).manual_seed(seed)
        weights = F.softmax(torch.randn(len(teachers), generator=rng, device=device), dim=0)

        sample_teacher = next(iter(teachers.values()))
        merged = BottleneckSequenceAdapter(
            sample_teacher.d_model, sample_teacher.bottleneck_dim, sample_teacher.num_symbols
        ).to(device)

        with torch.no_grad():
            for name, p in merged.named_parameters():
                stack_w = torch.stack([t.state_dict()[name] for t in teachers.values()])
                # Weighted sum across teachers
                w_comb = sum(w * stack_w[idx] for idx, w in enumerate(weights))
                p.copy_(w_comb)

        merged.eval()
        ems: dict[str, float] = {}
        with torch.no_grad():
            for skill_id in teachers.keys():
                qh = skill_data[skill_id]["qual_h"]
                qy = skill_data[skill_id]["qual_y"]
                logits = merged(qh)
                em = float((torch.argmax(logits, dim=-1) == qy).all(dim=-1).float().mean().item())
                ems[skill_id] = round(em, 4)

        mean_em = sum(ems.values()) / max(1, len(ems))
        baseline_params = sum(t.count_parameters() for t in teachers.values())
        consolidated_params = merged.count_parameters()
        comp_pct = (baseline_params - consolidated_params) / baseline_params * 100.0

        return DistillationResult(
            method=CompressionMethod.RANDOM_MERGING,
            baseline_parameters=baseline_params,
            consolidated_parameters=consolidated_params,
            compression_pct=round(comp_pct, 2),
            per_skill_em=ems,
            mean_em=round(mean_em, 4),
            relative_retention_ratio=round(mean_em / 1.0, 4),
            passed_retention_gate=(mean_em >= 0.85),
            parameter_hash=merged.compute_parameter_hash(),
            elapsed_time=round(time.perf_counter() - start_time, 2),
            consolidated_model=merged,
        )

    @classmethod
    def evaluate_svd_compression_baseline(
        cls,
        teachers: dict[str, BottleneckSequenceAdapter],
        skill_data: dict[str, dict[str, torch.Tensor]],
        device: torch.device,
        target_rank: int = 4,
    ) -> DistillationResult:
        """Conventional truncated SVD low-rank compression on individual adapters."""
        start_time = time.perf_counter()
        compressed_teachers: dict[str, BottleneckSequenceAdapter] = {}

        total_compressed_params = 0
        ems: dict[str, float] = {}

        for skill_id, teacher in teachers.items():
            comp_adapter = BottleneckSequenceAdapter(
                d_model=teacher.d_model, bottleneck_dim=target_rank, num_symbols=teacher.num_symbols
            ).to(device)

            # Apply SVD truncation on down_proj and up_proj
            with torch.no_grad():
                # Down proj: [rank_orig, d_model]
                u, s, vh = torch.linalg.svd(teacher.down_proj.weight, full_matrices=False)
                comp_adapter.down_proj.weight.copy_(torch.diag(s[:target_rank]) @ vh[:target_rank, :])
                if teacher.down_proj.bias is not None:
                    comp_adapter.down_proj.bias.copy_(teacher.down_proj.bias[:target_rank])

                # Up proj: [num_symbols, rank_orig]
                u_up, s_up, vh_up = torch.linalg.svd(teacher.up_proj.weight, full_matrices=False)
                comp_adapter.up_proj.weight.copy_(u_up[:, :target_rank] @ torch.diag(s_up[:target_rank]))
                if teacher.up_proj.bias is not None:
                    comp_adapter.up_proj.bias.copy_(teacher.up_proj.bias)

                # LayerNorm weights
                comp_adapter.ln.weight.copy_(teacher.ln.weight)
                comp_adapter.ln.bias.copy_(teacher.ln.bias)

            total_compressed_params += comp_adapter.count_parameters()
            compressed_teachers[skill_id] = comp_adapter

            # Evaluate
            comp_adapter.eval()
            with torch.no_grad():
                qh = skill_data[skill_id]["qual_h"]
                qy = skill_data[skill_id]["qual_y"]
                logits = comp_adapter(qh)
                em = float((torch.argmax(logits, dim=-1) == qy).all(dim=-1).float().mean().item())
                ems[skill_id] = round(em, 4)

        mean_em = sum(ems.values()) / max(1, len(ems))
        baseline_params = sum(t.count_parameters() for t in teachers.values())
        comp_pct = (baseline_params - total_compressed_params) / baseline_params * 100.0

        return DistillationResult(
            method=CompressionMethod.LOW_RANK_SVD,
            baseline_parameters=baseline_params,
            consolidated_parameters=total_compressed_params,
            compression_pct=round(comp_pct, 2),
            per_skill_em=ems,
            mean_em=round(mean_em, 4),
            relative_retention_ratio=round(mean_em / 1.0, 4),
            passed_retention_gate=(mean_em >= 0.85),
            parameter_hash="svd_compressed_set",
            elapsed_time=round(time.perf_counter() - start_time, 2),
            consolidated_model=None,
        )
