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
    state_dict: dict[str, torch.Tensor]


@dataclass
class RepertoireBenchmarkResult:
    """Quantitative performance comparison across all 7 repertoire arms on a target task."""
    arm_name: str
    target_env_id: str
    macro_f1: float
    accuracy: float
    classwise_recall: dict[str, float]
    paired_correctness: float
    selected_skill_id: str | None
    allocation_occurred: bool
    compute_steps: int
    diagnostics: str


class ModularSkillRepertoire:
    """In-memory registry and routing manager for a minimal repertoire of qualified skills."""

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
        """Register a qualified neural skill adapter into the repertoire."""
        hasher = hashlib.sha256()
        for p in adapter.parameters():
            hasher.update(p.detach().cpu().numpy().tobytes())
        param_hash = hasher.hexdigest()[:16]

        record = QualifiedSkillRecord(
            skill_id=skill_id,
            rule_family=rule_family,
            source_env_id=source_env_id,
            param_hash=param_hash,
            macro_f1=float(macro_f1),
            held_out_accuracy=float(accuracy),
            classwise_recall=copy.deepcopy(classwise_recall),
            state_dict={k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()},
        )
        self._skills[skill_id] = record
        logger.info(f"Registered modular skill '{skill_id}' ({rule_family}, F1={macro_f1:.4f})")
        return record

    def get_skill_ids(self) -> list[str]:
        return sorted(list(self._skills.keys()))

    def build_adapter_instance(self, skill_id: str, device: torch.device) -> Rank16ResidualAdapter:
        """Instantiate a clean adapter loaded with the skill's parameters."""
        if skill_id not in self._skills:
            raise KeyError(f"Skill '{skill_id}' not found in repertoire.")
        adapter = Rank16ResidualAdapter(
            d_model=self.d_model, bottleneck_dim=self.bottleneck_dim, num_classes=self.num_classes
        ).to(device)
        adapter.load_state_dict(self._skills[skill_id].state_dict)
        return adapter

    def evaluate_skill_on_env(
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
                out = model(input_ids)
                hidden = out.hidden_states if hasattr(out, "hidden_states") else out.last_hidden_state
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
                h1 = model(torch.tensor([t1], dtype=torch.long, device=device))
                h2 = model(torch.tensor([t2], dtype=torch.long, device=device))
                hd1 = h1.hidden_states if hasattr(h1, "hidden_states") else h1.last_hidden_state
                hd2 = h2.hidden_states if hasattr(h2, "hidden_states") else h2.last_hidden_state
                p1 = idx_to_label.get(adapter(hd1).argmax(dim=-1).item())
                p2 = idx_to_label.get(adapter(hd2).argmax(dim=-1).item())
                if p1 == mp.base_query.expected_answer and p2 == mp.perturbed_query.expected_answer:
                    paired_correct += 1
        paired_rate = float(paired_correct / len(env.minimal_pairs)) if env.minimal_pairs else 0.0

        return macro_f1, acc, recalls, paired_rate
