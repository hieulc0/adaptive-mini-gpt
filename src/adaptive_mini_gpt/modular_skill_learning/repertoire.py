"""Phase 2: Minimal Qualified Neural Skill Repertoire & Repertoire Router."""
from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.modular_skill_learning.hidden_extraction import extract_model_hidden
from adaptive_mini_gpt.three_valued_reasoning.architectures import Rank16ResidualAdapter
from adaptive_mini_gpt.three_valued_reasoning.semantics import GroundedThreeValuedEnvironment

logger = logging.getLogger(__name__)


class RepertoireRoutingArm(str, Enum):
    """Benchmark arms comparing modular skill reuse and acquisition strategies."""
    ARM_A_FRESH_ADAPTATION = "Arm_A_Fresh_Adaptation"
    ARM_B_STRONGEST_STATIC_SOURCE = "Arm_B_Strongest_Static_Source"
    ARM_C_M61_EVIDENCE_ROUTING = "Arm_C_M61_Evidence_Routing"
    ARM_D_REPERTOIRE_UTILITY_ROUTING = "Arm_D_Repertoire_Utility_Routing"
    ARM_E_NOVELTY_TRIGGERED_ACQUISITION = "Arm_E_Novelty_Triggered_Acquisition"
    ARM_F_ALWAYS_SPAWN_NEGATIVE_CONTROL = "Arm_F_Always_Spawn_Negative_Control"
    ARM_G_OFFLINE_EMPIRICAL_BEST_BOUND = "Arm_G_Offline_Empirical_Best_Bound"


@dataclass
class QualifiedSkillRecord:
    """Metadata and parameter snapshot of a qualified research skill adapter."""
    skill_id: str
    rule_family: str
    source_env_id: str
    param_hash: str
    macro_f1: float
    held_out_accuracy: float
    classwise_recall: dict[str, float]
    state_dict: dict[str, torch.Tensor] = field(repr=False)


@dataclass
class RepertoireBenchmarkResult:
    """Performance evaluation of a routing or acquisition arm."""
    arm_name: str
    selected_skill_id: str | None
    zero_shot_macro_f1: float
    adapted_macro_f1: float
    paired_correctness: float
    classwise_recall: dict[str, float]
    practice_budget_used: int
    parameter_footprint: int
    decision_reason: str


class ModularSkillRepertoire:
    """Manages an immutable registry of qualified neural skills with zero cross-skill parameter sharing."""

    def __init__(self, d_model: int = 128, bottleneck_dim: int = 16, num_classes: int = 3) -> None:
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes
        self._skills: dict[str, QualifiedSkillRecord] = {}

    def register_skill(
        self,
        skill_id: str,
        rule_family: str,
        source_env_id: str,
        adapter: nn.Module,
        macro_f1: float,
        accuracy: float,
        classwise_recall: dict[str, float],
    ) -> QualifiedSkillRecord:
        """Register a qualified adapter into the immutable repertoire."""
        hasher = hashlib.sha256()
        sd = copy.deepcopy(adapter.state_dict())
        for k in sorted(sd.keys()):
            hasher.update(sd[k].detach().cpu().numpy().tobytes())
        digest = hasher.hexdigest()[:16]

        record = QualifiedSkillRecord(
            skill_id=skill_id,
            rule_family=rule_family,
            source_env_id=source_env_id,
            param_hash=digest,
            macro_f1=float(macro_f1),
            held_out_accuracy=float(accuracy),
            classwise_recall=copy.deepcopy(classwise_recall),
            state_dict=sd,
        )
        self._skills[skill_id] = record
        logger.info(f"Registered qualified skill '{skill_id}' (family: {rule_family}, hash: {digest}, F1: {macro_f1:.4f})")
        return record

    def get_skill(self, skill_id: str) -> QualifiedSkillRecord:
        if skill_id not in self._skills:
            raise KeyError(f"Skill '{skill_id}' not found in repertoire.")
        return self._skills[skill_id]

    def list_skills(self) -> list[str]:
        return list(self._skills.keys())

    def get_skill_ids(self) -> list[str]:
        """Convenience alias for list_skills()."""
        return list(self._skills.keys())

    def build_adapter_instance(self, skill_id: str, device: torch.device) -> Rank16ResidualAdapter:
        record = self.get_skill(skill_id)
        adapter = Rank16ResidualAdapter(
            d_model=self.d_model, bottleneck_dim=self.bottleneck_dim, num_classes=self.num_classes
        ).to(device)
        adapter.load_state_dict(record.state_dict)
        return adapter

    def evaluate_skill_on_environment(
        self,
        skill_id: str,
        model: MiniGPT,
        env: GroundedThreeValuedEnvironment,
        device: torch.device,
    ) -> tuple[float, float, dict[str, float], float]:
        """Evaluate a registered skill zero-shot on an environment."""
        adapter = self.build_adapter_instance(skill_id, device)
        adapter.eval()
        label_to_idx = {"NO": 0, "YES": 1, "UNKNOWN": 2}
        idx_to_label = {0: "NO", 1: "YES", 2: "UNKNOWN"}

        conf_mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        correct_count = 0
        total = len(env.held_out_test_queries)

        with torch.no_grad():
            for q in env.held_out_test_queries:
                tokens = model.tokenizer.encode(q.question_text)
                input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
                hidden = extract_model_hidden(model, input_ids)
                pred_idx = adapter(hidden).argmax(dim=-1).item()
                true_idx = label_to_idx.get(q.expected_answer, 2)
                conf_mat[true_idx][pred_idx] += 1
                if pred_idx == true_idx:
                    correct_count += 1

        recalls = {}
        f1_list = []
        for c, name in idx_to_label.items():
            tp = conf_mat[c][c]
            fp = sum(conf_mat[r][c] for r in range(3) if r != c)
            fn = sum(conf_mat[c][p] for p in range(3) if p != c)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            recalls[name] = float(rec)
            f1_list.append(f1)

        macro_f1 = float(np.mean(f1_list))
        acc = float(correct_count / total) if total > 0 else 0.0

        # Paired correctness
        paired_correct = 0
        for mp in env.minimal_pairs:
            t1 = model.tokenizer.encode(mp.base_query.question_text)
            t2 = model.tokenizer.encode(mp.perturbed_query.question_text)
            with torch.no_grad():
                h1 = extract_model_hidden(model, torch.tensor([t1], dtype=torch.long, device=device))
                h2 = extract_model_hidden(model, torch.tensor([t2], dtype=torch.long, device=device))
                p1 = idx_to_label.get(adapter(h1).argmax(dim=-1).item())
                p2 = idx_to_label.get(adapter(h2).argmax(dim=-1).item())
                if p1 == mp.base_query.expected_answer and p2 == mp.perturbed_query.expected_answer:
                    paired_correct += 1
        paired_rate = float(paired_correct / len(env.minimal_pairs)) if env.minimal_pairs else 0.0

        return macro_f1, acc, recalls, paired_rate

    def evaluate_skill_on_env(
        self,
        skill_id: str,
        model: MiniGPT,
        env: GroundedThreeValuedEnvironment,
        device: torch.device,
    ) -> tuple[float, float, dict[str, float], float]:
        """Convenience alias for evaluate_skill_on_environment."""
        return self.evaluate_skill_on_environment(skill_id, model, env, device)

    def select_best_skill_by_evidence_probe(
        self,
        model: MiniGPT,
        probe_queries: list[Any],
        device: torch.device,
    ) -> tuple[str, float]:
        """Evidence-grounded routing across the small repertoire using limited probe evidence."""
        if not self._skills:
            raise RuntimeError("Cannot route across an empty skill repertoire.")

        label_to_idx = {"NO": 0, "YES": 1, "UNKNOWN": 2}
        best_skill = list(self._skills.keys())[0]
        best_score = -1.0

        for skill_id in self._skills:
            adapter = self.build_adapter_instance(skill_id, device)
            adapter.eval()
            correct = 0
            with torch.no_grad():
                for q in probe_queries:
                    tokens = model.tokenizer.encode(q.question_text)
                    inp = torch.tensor([tokens], dtype=torch.long, device=device)
                    hidden = extract_model_hidden(model, inp)
                    pred = adapter(hidden).argmax(dim=-1).item()
                    expected = label_to_idx.get(q.expected_answer, 2)
                    if pred == expected:
                        correct += 1
            score = float(correct / max(1, len(probe_queries)))
            if score > best_score:
                best_score = score
                best_skill = skill_id

        return best_skill, best_score
