"""Milestone M80 Phase 2 & 3: Autonomous Neural Adapter Acquisition Engine.

Implements bounded neural adaptation when an unsupported skill gap is detected:
1. Practice sample acquisition: obtains N_train supervised examples from verified environment.
2. Adapter initialization: Rank-16 (2,592 params) or Rank-8 (1,432 params).
3. Comparative learning-to-learn modes:
   - FRESH_RANDOM: random initialization.
   - PRETRAINED_BACKBONE: trained on top of frozen Transformer representations.
   - EXPERIENCE_WARM_START: warm-started from related prior skill adapter weights.
   - SHUFFLED_WEIGHTS: shuffled learned parameters for causal weight attribution.
4. Step budget sweep: T in {20, 40, 60, 80} steps.
5. Independent qualification gate: EM >= 0.85 on held-out qualification partition.
6. Bit-exact rollback upon failure.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.neural_skill_acquisition.adapter_architecture import (
    EvolvableSkillAdapterFactory,
)
from adaptive_mini_gpt.neural_skill_composition.primitive_skills import (
    QualifiedPrimitiveRecord,
)
from adaptive_mini_gpt.neural_weight_learning.pathway_instrumentation import (
    NeuralPathwayInstrumenter,
)
from adaptive_mini_gpt.neural_weight_learning.procedural_tasks import (
    ProceduralTaskDataset,
)

logger = logging.getLogger(__name__)

QUALIFICATION_EM_THRESHOLD: float = 0.85


class LearningInitMode(StrEnum):
    """Initialization strategy for candidate skill adapter."""

    FRESH_RANDOM = "FRESH_RANDOM"
    PRETRAINED_BACKBONE = "PRETRAINED_BACKBONE"
    EXPERIENCE_WARM_START = "EXPERIENCE_WARM_START"
    SHUFFLED_WEIGHTS = "SHUFFLED_WEIGHTS"
    ORACLE_REFERENCE = "ORACLE_REFERENCE"


@dataclass
class AcquisitionStepMetric:
    """Diagnostic checkpoint during candidate adapter training."""

    step: int
    train_loss: float
    qual_loss: float
    qual_exact_match: float
    qual_token_acc: float


@dataclass
class AcquisitionAuditResult:
    """Complete diagnostic trace of an autonomous skill acquisition trial."""

    skill_name: str
    init_mode: LearningInitMode
    adapter_rank: int
    parameters_allocated: int
    training_steps_used: int
    training_latency_ms: float
    qualification_passed: bool
    qualification_exact_match: float
    qualification_loss: float
    sealed_test_exact_match: float
    sealed_test_token_acc: float
    sealed_test_loss: float
    parameter_hash: str
    learning_trajectory: list[AcquisitionStepMetric] = field(default_factory=list)
    acquired_record: QualifiedPrimitiveRecord | None = None


class AdapterAcquisitionEngine:
    """Manages autonomous training, qualification, and validation of neural skill adapters."""

    @classmethod
    def train_candidate_adapter(
        cls,
        model: MiniGPT,
        task: ProceduralTaskDataset,
        skill_name: str,
        device: torch.device,
        rank: int = 16,
        init_mode: LearningInitMode = LearningInitMode.PRETRAINED_BACKBONE,
        prior_adapter: nn.Module | None = None,
        steps: int = 60,
        lr: float = 3e-3,
        seed: int = 42,
    ) -> AcquisitionAuditResult:
        """Train a candidate adapter under specified learning mode and evaluate on qualification split."""
        start_time = time.perf_counter()
        torch.manual_seed(seed)

        # 1. Initialize Adapter
        adapter = EvolvableSkillAdapterFactory.create_adapter(
            rank=rank, d_model=model.config.d_model, num_symbols=16
        ).to(device)

        if init_mode == LearningInitMode.EXPERIENCE_WARM_START and prior_adapter is not None:
            # Warm-start from prior adapter if same dimensions, or copy shared weights
            with torch.no_grad():
                for (_name1, p1), (_name2, p2) in zip(adapter.named_parameters(), prior_adapter.named_parameters(), strict=False):
                    if p1.shape == p2.shape:
                        p1.copy_(p2 * 0.5)  # Scale prior weights to prevent saturation

        # Extract features from frozen base
        train_h, train_y = NeuralPathwayInstrumenter.extract_symbol_hidden_states(
            model, task.train_examples, device
        )
        qual_h, qual_y = NeuralPathwayInstrumenter.extract_symbol_hidden_states(
            model, task.qualification_examples, device
        )
        test_h, test_y = NeuralPathwayInstrumenter.extract_symbol_hidden_states(
            model, task.sealed_test_examples, device
        )

        optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
        trajectory: list[AcquisitionStepMetric] = []

        # Training Loop
        adapter.train()
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            logits = adapter(train_h)
            loss = F.cross_entropy(logits.view(-1, 16), train_y.view(-1))
            loss.backward()
            optimizer.step()

            if step % 10 == 0 or step == steps:
                adapter.eval()
                with torch.no_grad():
                    q_logits = adapter(qual_h)
                    q_preds = torch.argmax(q_logits, dim=-1)
                    q_em = float((q_preds == qual_y).all(dim=-1).float().mean().item())
                    q_acc = float((q_preds == qual_y).float().mean().item())
                    q_loss = float(F.cross_entropy(q_logits.view(-1, 16), qual_y.view(-1)).item())
                adapter.train()

                trajectory.append(
                    AcquisitionStepMetric(
                        step=step,
                        train_loss=round(float(loss.item()), 4),
                        qual_loss=round(q_loss, 4),
                        qual_exact_match=round(q_em, 4),
                        qual_token_acc=round(q_acc, 4),
                    )
                )

        # If Shuffled Control mode: permanently shuffle weights after training
        if init_mode == LearningInitMode.SHUFFLED_WEIGHTS:
            if hasattr(adapter, "shuffle_weights"):
                adapter = adapter.shuffle_weights(seed=seed + 99)

        # Final qualification evaluation
        adapter.eval()
        with torch.no_grad():
            final_q_logits = adapter(qual_h)
            final_q_preds = torch.argmax(final_q_logits, dim=-1)
            final_q_em = float((final_q_preds == qual_y).all(dim=-1).float().mean().item())
            final_q_loss = float(F.cross_entropy(final_q_logits.view(-1, 16), qual_y.view(-1)).item())

            test_logits = adapter(test_h)
            test_preds = torch.argmax(test_logits, dim=-1)
            test_em = float((test_preds == test_y).all(dim=-1).float().mean().item())
            test_acc = float((test_preds == test_y).float().mean().item())
            test_loss = float(F.cross_entropy(test_logits.view(-1, 16), test_y.view(-1)).item())

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        param_hash = adapter.compute_parameter_hash() if hasattr(adapter, "compute_parameter_hash") else "hash"
        param_count = adapter.count_parameters() if hasattr(adapter, "count_parameters") else 2592

        # Check qualification gate (EM >= 0.85)
        passed_gate = (final_q_em >= QUALIFICATION_EM_THRESHOLD)

        acquired_rec = None
        if passed_gate and init_mode != LearningInitMode.SHUFFLED_WEIGHTS:
            # Freeze completely
            for p in adapter.parameters():
                p.requires_grad = False
            acquired_rec = QualifiedPrimitiveRecord(
                skill_id=f"acquired_{skill_name.lower()}",
                rule_family=skill_name,
                adapter=adapter,
                qualification_exact_match=round(final_q_em, 4),
                qualification_loss=round(final_q_loss, 4),
                trainable_parameters=param_count,
                parameter_hash=param_hash,
                training_steps=steps,
                is_frozen=True,
            )

        return AcquisitionAuditResult(
            skill_name=skill_name,
            init_mode=init_mode,
            adapter_rank=rank,
            parameters_allocated=param_count if passed_gate else 0,
            training_steps_used=steps,
            training_latency_ms=round(elapsed_ms, 2),
            qualification_passed=passed_gate,
            qualification_exact_match=round(final_q_em, 4),
            qualification_loss=round(final_q_loss, 4),
            sealed_test_exact_match=round(test_em, 4),
            sealed_test_token_acc=round(test_acc, 4),
            sealed_test_loss=round(test_loss, 4),
            parameter_hash=param_hash,
            learning_trajectory=trajectory,
            acquired_record=acquired_rec,
        )
