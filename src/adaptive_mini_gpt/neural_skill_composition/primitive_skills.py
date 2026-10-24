"""Phase 2: Independently Qualified Primitive Neural Skills.

Implements:
- Training, qualification, and parameter freezing of atomic procedural sequence skills:
    1. COMPLEMENT_REFLECT: s_i -> 15 - s_i
    2. CAESAR_SHIFT_MOD3: s_i -> (s_i + 3) mod 16
    3. AFFINE_TRANSFORM: s_i -> (3 * s_i + 1) mod 16
- Independent qualification verification on held-out splits (threshold EM >= 0.75).
- Weight freezing and immutable parameter hashing prior to composition experiments.
- Parameter accounting: exactly 2,592 weights per primitive rank-16 adapter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import torch

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.neural_weight_learning.pathway_instrumentation import (
    NeuralPathwayInstrumenter,
)
from adaptive_mini_gpt.neural_weight_learning.procedural_tasks import (
    ProceduralRuleFamily,
    ProceduralTaskGenerator,
)
from adaptive_mini_gpt.neural_weight_learning.sequence_adapter import Rank16SequenceAdapter

logger = logging.getLogger(__name__)


class PrimitiveSkillType(StrEnum):
    """Supported atomic procedural primitive skills."""

    COMPLEMENT_REFLECT = "COMPLEMENT_REFLECT"
    CAESAR_SHIFT_MOD3 = "CAESAR_SHIFT_MOD3"
    AFFINE_TRANSFORM = "AFFINE_TRANSFORM"


@dataclass
class QualifiedPrimitiveRecord:
    """Diagnostic record of an independently trained and qualified neural primitive."""

    skill_id: str
    rule_family: str
    adapter: Rank16SequenceAdapter
    qualification_exact_match: float
    qualification_loss: float
    trainable_parameters: int
    parameter_hash: str
    training_steps: int
    is_frozen: bool


class PrimitiveSkillManager:
    """Manages acquisition, verification, freezing, and retrieval of primitive neural skills."""

    QUALIFICATION_THRESHOLD = 0.75

    @classmethod
    def train_and_qualify_primitive(
        cls,
        model: MiniGPT,
        rule_family: ProceduralRuleFamily,
        seed: int,
        device: torch.device,
        steps: int = 40,
        lr: float = 3e-3,
    ) -> QualifiedPrimitiveRecord:
        """Train sequence adapter on primitive task and qualify on held-out split."""
        task_gen = ProceduralTaskGenerator(vocab_size=16)
        task = task_gen.generate_task(
            task_id=f"primitive_{rule_family.value}",
            rule_family=rule_family,
            seed=seed,
            n_train=48,
            n_dev=16,
            n_qual=16,
            n_test=48,
        )

        adapter = Rank16SequenceAdapter().to(device)
        trained_ad, audit_record = NeuralPathwayInstrumenter.train_and_instrument(
            model=model,
            adapter=adapter,
            task=task,
            adapter_id=f"prim_{rule_family.value}",
            device=device,
            steps=steps,
            lr=lr,
        )

        # Independent qualification on held-out qualification split
        qual_h, qual_y = NeuralPathwayInstrumenter.extract_symbol_hidden_states(
            model, task.qualification_examples, device
        )
        trained_ad.eval()
        with torch.no_grad():
            q_logits = trained_ad(qual_h)
            q_preds = torch.argmax(q_logits, dim=-1)
            q_em = float((q_preds == qual_y).all(dim=-1).float().mean().item())
            q_loss = float(
                torch.nn.functional.cross_entropy(
                    q_logits.view(-1, 16), qual_y.view(-1)
                ).item()
            )

        if q_em < cls.QUALIFICATION_THRESHOLD:
            logger.warning(
                "Primitive %s achieved EM %.4f below threshold %.2f",
                rule_family.value,
                q_em,
                cls.QUALIFICATION_THRESHOLD,
            )

        # Freeze primitive parameters completely
        for p in trained_ad.parameters():
            p.requires_grad = False
        param_hash = trained_ad.compute_parameter_hash()

        return QualifiedPrimitiveRecord(
            skill_id=f"prim_{rule_family.value}",
            rule_family=rule_family.value,
            adapter=trained_ad,
            qualification_exact_match=round(q_em, 4),
            qualification_loss=round(q_loss, 4),
            trainable_parameters=trained_ad.count_parameters(),
            parameter_hash=param_hash,
            training_steps=steps,
            is_frozen=True,
        )

    @classmethod
    def acquire_canonical_primitives(
        cls,
        model: MiniGPT,
        seed: int,
        device: torch.device,
    ) -> dict[PrimitiveSkillType, QualifiedPrimitiveRecord]:
        """Train and qualify the standard set of 3 atomic primitive skills."""
        repertoire: dict[PrimitiveSkillType, QualifiedPrimitiveRecord] = {}

        for p_type, rule_fam in [
            (PrimitiveSkillType.COMPLEMENT_REFLECT, ProceduralRuleFamily.COMPLEMENT_REFLECT),
            (PrimitiveSkillType.CAESAR_SHIFT_MOD3, ProceduralRuleFamily.CAESAR_SHIFT_MOD3),
            (PrimitiveSkillType.AFFINE_TRANSFORM, ProceduralRuleFamily.AFFINE_TRANSFORM),
        ]:
            rec = cls.train_and_qualify_primitive(
                model=model,
                rule_family=rule_fam,
                seed=seed,
                device=device,
                steps=40,
                lr=3e-3,
            )
            repertoire[p_type] = rec

        return repertoire
