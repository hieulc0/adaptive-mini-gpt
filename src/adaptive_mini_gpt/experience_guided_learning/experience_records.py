"""Phase 2: Verified Learning-Experience Records and Manifest Infrastructure.

Implements compact, verifiable records of empirical learning attempts:
- Observable pre-training task descriptors (variance, entropy, dev zero-shot probe accuracy).
- Prior qualified skill performance metrics.
- Selected learning action, step budget (20/40/60), and learning rate.
- Execution metrics: actual optimizer steps, GPU time (ms), and loss reduction.
- Development qualification outcome (qualification PASS/FAIL, retention/rollback).
- Strictly segregated sealed-test outcome (recorded post-hoc, isolated from policy decisions).
- Zero external databases or heavy vector services: compact JSON-serializable in-memory records.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class LearningActionType(StrEnum):
    """Permitted bounded learning decisions."""

    REUSE_EXISTING = "REUSE_EXISTING"
    FRESH_ADAPT = "FRESH_ADAPT"
    TRAIN_CANDIDATE = "TRAIN_CANDIDATE"


class RetentionDecision(StrEnum):
    """Disposition of trained skill candidate."""

    RETAINED = "RETAINED"
    DISCARDED = "DISCARDED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass(frozen=True)
class ObservableTaskDescriptors:
    """Task features observable prior to training without inspecting ground-truth rules."""

    sample_count: int
    seq_length: int
    unique_symbols_count: int
    input_variance: float
    input_entropy: float
    zero_shot_dev_em: float
    zero_shot_dev_loss: float
    best_repertoire_em: float
    best_repertoire_loss: float

    def feature_vector(self) -> list[float]:
        """Compact numerical vector for similarity calculation."""
        return [
            float(self.sample_count) / 100.0,
            float(self.seq_length) / 10.0,
            float(self.unique_symbols_count) / 16.0,
            self.input_variance / 25.0,
            self.input_entropy / 4.0,
            self.zero_shot_dev_em,
            min(self.zero_shot_dev_loss / 5.0, 1.0),
            self.best_repertoire_em,
            min(self.best_repertoire_loss / 5.0, 1.0),
        ]


@dataclass(frozen=True)
class LearningAction:
    """Selected bounded learning action and hyperparameters."""

    action_type: LearningActionType
    step_budget: int  # 20, 40, or 60
    learning_rate: float  # 1e-3, 3e-3, 5e-3
    reused_skill_id: str | None = None


@dataclass
class LearningExperienceRecord:
    """Complete, verified record of an executed learning episode."""

    record_id: str
    task_id: str
    episode_index: int
    descriptors: ObservableTaskDescriptors
    action: LearningAction
    actual_steps: int
    gpu_time_ms: float
    initial_loss: float
    final_loss: float
    loss_reduction_ratio: float
    dev_em: float
    qualification_passed: bool
    retention_decision: RetentionDecision
    repertoire_size_after: int
    # Strictly isolated post-hoc evaluation fields:
    # NEVER accessed by policy decision logic!
    sealed_test_em: float | None = None
    sealed_test_loss: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to JSON-compatible dictionary."""
        d = asdict(self)
        d["action"]["action_type"] = self.action.action_type.value
        d["retention_decision"] = self.retention_decision.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearningExperienceRecord:
        """Construct record from JSON dictionary."""
        desc_data = data["descriptors"]
        desc = ObservableTaskDescriptors(**desc_data)
        act_data = data["action"]
        act = LearningAction(
            action_type=LearningActionType(act_data["action_type"]),
            step_budget=act_data["step_budget"],
            learning_rate=act_data["learning_rate"],
            reused_skill_id=act_data.get("reused_skill_id"),
        )
        return cls(
            record_id=data["record_id"],
            task_id=data["task_id"],
            episode_index=data["episode_index"],
            descriptors=desc,
            action=act,
            actual_steps=data["actual_steps"],
            gpu_time_ms=data["gpu_time_ms"],
            initial_loss=data["initial_loss"],
            final_loss=data["final_loss"],
            loss_reduction_ratio=data["loss_reduction_ratio"],
            dev_em=data["dev_em"],
            qualification_passed=data["qualification_passed"],
            retention_decision=RetentionDecision(data["retention_decision"]),
            repertoire_size_after=data["repertoire_size_after"],
            sealed_test_em=data.get("sealed_test_em"),
            sealed_test_loss=data.get("sealed_test_loss"),
        )


@dataclass
class LearningExperienceManifest:
    """Compact repository of verified learning experiences with validation & perturbation."""

    records: list[LearningExperienceRecord] = field(default_factory=list)

    def add_record(self, record: LearningExperienceRecord) -> None:
        """Add a newly completed and verified learning record."""
        self.records.append(record)

    def count(self) -> int:
        return len(self.records)

    def to_json(self) -> str:
        """Serialize records to JSON string."""
        return json.dumps([r.to_dict() for r in self.records], indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> LearningExperienceManifest:
        """Deserialize from JSON string."""
        raw_list = json.loads(json_str)
        records = [LearningExperienceRecord.from_dict(r) for r in raw_list]
        return cls(records=records)

    def generate_shuffled_control(self, seed: int) -> LearningExperienceManifest:
        """Generate identical-volume history with decoupled task-outcome associations.

        Preserves exact feature distributions and outcome distributions, but breaks
        the causal relationship between pre-training task features and learning outcomes.
        """
        if not self.records:
            return LearningExperienceManifest()

        rng = np.random.default_rng(seed)
        shuffled_records = []
        perm_outcomes = rng.permutation(len(self.records))

        for idx, r in enumerate(self.records):
            donor = self.records[perm_outcomes[idx]]
            shuffled_rec = LearningExperienceRecord(
                record_id=f"shuffled_{r.record_id}",
                task_id=r.task_id,
                episode_index=r.episode_index,
                descriptors=r.descriptors,  # original task descriptor
                action=donor.action,  # donor action
                actual_steps=donor.actual_steps,
                gpu_time_ms=donor.gpu_time_ms,
                initial_loss=donor.initial_loss,
                final_loss=donor.final_loss,
                loss_reduction_ratio=donor.loss_reduction_ratio,
                dev_em=donor.dev_em,
                qualification_passed=donor.qualification_passed,
                retention_decision=donor.retention_decision,
                repertoire_size_after=donor.repertoire_size_after,
            )
            shuffled_records.append(shuffled_rec)

        return LearningExperienceManifest(records=shuffled_records)

    def generate_irrelevant_control(self, seed: int) -> LearningExperienceManifest:
        """Generate history from unrelated synthetic task descriptors and outcomes."""
        if not self.records:
            return LearningExperienceManifest()

        rng = np.random.default_rng(seed)
        irrelevant_records = []

        for idx, _r in enumerate(self.records):
            # Unrelated noisy descriptors
            unrelated_desc = ObservableTaskDescriptors(
                sample_count=int(rng.integers(10, 100)),
                seq_length=int(rng.integers(2, 8)),
                unique_symbols_count=int(rng.integers(2, 16)),
                input_variance=float(rng.uniform(0.1, 20.0)),
                input_entropy=float(rng.uniform(0.5, 3.5)),
                zero_shot_dev_em=float(rng.uniform(0.0, 0.4)),
                zero_shot_dev_loss=float(rng.uniform(1.5, 4.0)),
                best_repertoire_em=float(rng.uniform(0.0, 0.5)),
                best_repertoire_loss=float(rng.uniform(1.2, 3.8)),
            )
            unrelated_action = LearningAction(
                action_type=rng.choice(
                    [
                        LearningActionType.REUSE_EXISTING,
                        LearningActionType.FRESH_ADAPT,
                        LearningActionType.TRAIN_CANDIDATE,
                    ]
                ),
                step_budget=int(rng.choice([20, 40, 60])),
                learning_rate=float(rng.choice([0.001, 0.003, 0.005])),
            )
            dev_em = float(rng.uniform(0.2, 0.95))
            qual = dev_em >= 0.80
            irr_rec = LearningExperienceRecord(
                record_id=f"irrelevant_{idx}",
                task_id=f"irr_task_{idx}",
                episode_index=idx,
                descriptors=unrelated_desc,
                action=unrelated_action,
                actual_steps=unrelated_action.step_budget,
                gpu_time_ms=float(unrelated_action.step_budget * 12.5),
                initial_loss=float(rng.uniform(2.5, 3.2)),
                final_loss=float(rng.uniform(0.1, 1.2)),
                loss_reduction_ratio=float(rng.uniform(0.4, 0.95)),
                dev_em=dev_em,
                qualification_passed=qual,
                retention_decision=RetentionDecision.RETAINED if qual else RetentionDecision.DISCARDED,
                repertoire_size_after=int(rng.integers(0, 5)),
            )
            irrelevant_records.append(irr_rec)

        return LearningExperienceManifest(records=irrelevant_records)
