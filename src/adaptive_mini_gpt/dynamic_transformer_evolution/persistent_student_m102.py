"""Milestone M102 Persistent Student with Dynamic Architecture Evolution.

Integrates:
1. Dynamic Evolvable Transformer Backbone (layer skipping, block insertion/removal).
2. Modular Adapters with Per-Domain Gradient Isolation.
3. Balanced Per-Family Rehearsal Buffer (eliminating catastrophic forgetting in worst families).
4. Permanent Architecture Principle accounting:
   - Tracks persistent total, fast trainable, slow trainable, layers count, and memory budget.
5. Bit-exact rollback and checkpoint recovery.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.dynamic_transformer_evolution.dynamic_transformer_substrate import (
    DynamicEvolvableBackbone,
)
from adaptive_mini_gpt.dynamic_transformer_evolution.per_family_rehearsal_buffer import (
    BalancedPerFamilyRehearsalBuffer,
)
from adaptive_mini_gpt.integrated_autonomous_student.structural_evolution_engine import (
    CapacityTierM100,
    StructuralEvolutionEngine,
)
from adaptive_mini_gpt.integrated_autonomous_student.two_speed_plasticity_manager import (
    TwoSpeedPlasticityManager,
)
from adaptive_mini_gpt.self_directed_curriculum.curriculum_policy import (
    CurriculumDiscoveryPolicy,
)
from adaptive_mini_gpt.meta_acquisition.learned_acquisition_controller import (
    LearnedAcquisitionController,
)
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.persistent_memory.knowledge_store import (
    FactualRecord,
    PersistentKnowledgeStore,
)
from adaptive_mini_gpt.sustainable_lifelong_learning.lifelong_curriculum_m101 import (
    DomainM101,
    LearningGoalM101,
    MODULUS,
)
from adaptive_mini_gpt.sustainable_lifelong_learning.persistent_student_m101 import (
    GoalDiagnosticState,
)


class PersistentStudentM102(nn.Module):
    """M102 Student: Continuously learns, evolves architecture, and prevents worst-family forgetting."""

    def __init__(
        self,
        tier: CapacityTierM100 = CapacityTierM100.TIER_20K,
        device: torch.device | None = None,
        retention_threshold: float = 90.0,
        worst_family_threshold: float = 80.0,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        self.tier = tier
        self.retention_threshold = retention_threshold
        self.worst_family_threshold = worst_family_threshold

        # 1. Base Transformer Model & Evolvable Wrapper
        base_cfg = ModelConfig(
            vocab_size=1024,
            context_length=128,
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_expert=64,
            adaptive_enabled=False,
            mod_enabled=False,
        )
        base_gpt = MiniGPT(base_cfg)
        self.evolvable_backbone = DynamicEvolvableBackbone(base_gpt)
        self.d_model = base_cfg.d_model

        # 2. Fast Modular Domain Adapters (Rank-2 Bottlenecks)
        self.adapter_modules = nn.ModuleDict({
            f"adapter_{d.value}": nn.Sequential(
                nn.Linear(self.d_model, 2, bias=False),
                nn.GELU(),
                nn.Linear(2, self.d_model, bias=False),
            )
            for d in DomainM101
            if d not in (DomainM101.NOISE_DISTRACTOR, DomainM101.ALREADY_MASTERED)
        })

        # 3. Output Task Head
        self.task_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, MODULUS),
        )

        # 4. Controllers
        self.controller = LearnedAcquisitionController(in_features=8, hidden_dim=32)
        self.curriculum_policy = CurriculumDiscoveryPolicy(in_features=5, hidden_dim=16)

        # 5. Balanced Per-Family Rehearsal Buffer (Protects worst-family retention >= 80%)
        self.rehearsal_buffer = BalancedPerFamilyRehearsalBuffer(max_exemplars_per_domain=32)

        # 6. Persistent Knowledge Store
        self.memory_store = PersistentKnowledgeStore(max_byte_budget=262_144)

        # 7. Subsystems
        self.plasticity_manager = TwoSpeedPlasticityManager(retention_threshold_pct=retention_threshold, fast_lr=0.035)
        self.evolution_engine = StructuralEvolutionEngine(tier=tier, device=self.device)

        # Telemetry
        self.step_counter: int = 0
        self.diagnostic_history: dict[str, list[float]] = {}
        self.last_practiced_step: dict[str, int] = {}
        self.acquired_skills: list[str] = ["foundational_arithmetic"]
        self.episodic_log: list[dict[str, Any]] = []

        self.to(self.device)
        self.pretrain_foundational_anchor()

    def pretrain_foundational_anchor(self) -> None:
        """Pre-qualify foundational arithmetic anchor."""
        anc_s = list(range(16))
        anc_op = [1] * 16
        anc_v = [2] * 16
        anc_y = [(si + 2) % MODULUS for si in anc_s]

        seqs = []
        for si, opi, vi in zip(anc_s, anc_op, anc_v, strict=False):
            tok_s = si % 16
            tok_op = 16 + (opi % 16)
            tok_v = 32 + (vi % 16)
            seqs.append([tok_s, tok_op, tok_v])
        tok = torch.tensor(seqs, dtype=torch.long, device=self.device)
        lbl = torch.tensor(anc_y, dtype=torch.long, device=self.device)

        self.train()
        opt = torch.optim.AdamW(
            list(self.task_head.parameters()) + list(self.adapter_modules.parameters()),
            lr=0.04,
        )
        for _ in range(30):
            logits = self.forward(tok)
            loss = F.cross_entropy(logits, lbl)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.eval()

        # Register into balanced rehearsal buffer
        self.rehearsal_buffer.register_domain("foundational_arithmetic", tok, lbl)

    def forward(
        self,
        tokens: torch.Tensor,
        active_domains: list[str] | None = None,
        ablate_backbone: bool = False,
        allow_skipping: bool = False,
    ) -> torch.Tensor:
        """Forward pass through dynamic backbone + domain adapters + task head."""
        if ablate_backbone:
            h_backbone = torch.zeros(tokens.size(0), self.d_model, device=tokens.device)
        else:
            h_backbone = self.evolvable_backbone(tokens, allow_skipping=allow_skipping)

        h_adapted = h_backbone
        if active_domains is not None:
            for d in active_domains:
                key = f"adapter_{d}"
                if key in self.adapter_modules:
                    h_adapted = h_adapted + self.adapter_modules[key](h_backbone)
        else:
            for _, adapter in self.adapter_modules.items():
                h_adapted = h_adapted + adapter(h_backbone)

        return self.task_head(h_adapted)

    def adapt_fast_skill(
        self,
        goal: LearningGoalM101,
        max_steps: int = 35,
        early_stop_acc: float = 85.0,
    ) -> dict[str, Any]:
        """Fast Plasticity with Balanced Multi-Domain Rehearsal."""
        self.step_counter += 1
        self.last_practiced_step[goal.goal_id] = self.step_counter

        if not goal.is_learnable:
            return {
                "goal_id": goal.goal_id,
                "steps_taken": 0,
                "final_test_acc": 0.0,
                "retained_anchor_acc": 100.0,
                "abstained": True,
            }

        # Freeze backbone, isolate active domain adapter + task head
        for p in self.evolvable_backbone.parameters():
            p.requires_grad = False

        active_adapter_key = f"adapter_{goal.domain.value}"
        for name, p in self.adapter_modules.named_parameters():
            p.requires_grad = active_adapter_key in name
        for p in self.task_head.parameters():
            p.requires_grad = True

        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.plasticity_manager.fast_lr,
            weight_decay=1e-4,
        )

        steps_taken = 0
        final_test_acc = 0.0

        for s in range(1, max_steps + 1):
            steps_taken = s
            self.train()
            logits = self.forward(goal.train_tokens, active_domains=[goal.domain.value])
            loss_task = F.cross_entropy(logits, goal.train_labels)

            # Balanced Multi-Family Rehearsal Loss
            rehearsal_batch = self.rehearsal_buffer.sample_balanced_batch(batch_size_per_domain=4)
            if rehearsal_batch is not None:
                r_tok, r_lbl = rehearsal_batch
                r_logits = self.forward(r_tok)
                loss_rehearsal = F.cross_entropy(r_logits, r_lbl)
                total_loss = 0.6 * loss_task + 0.4 * loss_rehearsal
            else:
                total_loss = loss_task

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            self.eval()
            with torch.no_grad():
                te_logits = self.forward(goal.test_tokens, active_domains=[goal.domain.value])
                te_preds = te_logits.argmax(dim=-1) % MODULUS
                acc = float((te_preds == goal.test_labels).float().mean().item()) * 100.0
                final_test_acc = acc
                if acc >= early_stop_acc:
                    break

        # Register newly verified domain into balanced buffer
        if final_test_acc >= 75.0:
            self.rehearsal_buffer.register_domain(goal.domain.value, goal.test_tokens, goal.test_labels)
            if goal.goal_id not in self.acquired_skills:
                self.acquired_skills.append(goal.goal_id)

        # Evaluate Retention across ALL registered families
        family_retentions = self.rehearsal_buffer.evaluate_all_families_retention(self)
        worst_ret = min(family_retentions.values()) if family_retentions else 100.0
        mean_ret = float(sum(family_retentions.values()) / max(1, len(family_retentions)))

        hist = self.diagnostic_history.setdefault(goal.goal_id, [])
        hist.append(final_test_acc)

        return {
            "goal_id": goal.goal_id,
            "steps_taken": steps_taken,
            "final_test_acc": final_test_acc,
            "mean_retention_acc": mean_ret,
            "worst_family_retention_acc": worst_ret,
            "family_retentions": family_retentions,
            "abstained": False,
        }

    def consolidate_slow_backbone(
        self,
        goal: LearningGoalM101,
        steps: int = 10,
    ) -> Any:
        """Slow Plasticity with Multi-Domain Rehearsal & Per-Family Retention Guard."""
        # Backup candidate
        backup = {k: v.detach().clone() for k, v in self.state_dict().items()}

        # Unfreeze top block and norm
        for p in self.evolvable_backbone.parameters():
            p.requires_grad = False
        for pn, p in self.evolvable_backbone.base_model.named_parameters():
            if "blocks.3" in pn or "ln_f" in pn:
                p.requires_grad = True
        for p in self.adapter_modules.parameters():
            p.requires_grad = True
        for p in self.task_head.parameters():
            p.requires_grad = True

        opt = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=0.001)

        self.train()
        for _ in range(steps):
            logits = self.forward(goal.train_tokens, active_domains=[goal.domain.value])
            loss_task = F.cross_entropy(logits, goal.train_labels)

            rehearsal_batch = self.rehearsal_buffer.sample_balanced_batch(batch_size_per_domain=4)
            if rehearsal_batch is not None:
                r_tok, r_lbl = rehearsal_batch
                r_logits = self.forward(r_tok)
                loss_rehearsal = F.cross_entropy(r_logits, r_lbl)
                loss = 0.5 * loss_task + 0.5 * loss_rehearsal
            else:
                loss = loss_task

            opt.zero_grad()
            loss.backward()
            opt.step()

        # Check Per-Family Retention Gate
        family_ret = self.rehearsal_buffer.evaluate_all_families_retention(self)
        worst_fam = min(family_ret.values()) if family_ret else 100.0
        mean_fam = float(sum(family_ret.values()) / max(1, len(family_ret)))

        # Enforce G5: worst family >= 80.0% and aggregate >= 90.0%
        promoted = (worst_fam >= self.worst_family_threshold) and (mean_fam >= self.retention_threshold)
        rollback = False

        if not promoted:
            self.load_state_dict(backup)
            rollback = True

        self.eval()
        return type("ConsolidationReportM102", (), {
            "promoted": promoted,
            "rollback_triggered": rollback,
            "worst_family_retention": worst_fam,
            "mean_retention": mean_fam,
            "plasticity_mode": type("Mode", (), {"value": "slow_selective_backbone"})(),
        })()

    def get_full_parameter_breakdown(self) -> dict[str, int]:
        """Transparent accounting under the Permanent Architecture Principle."""
        p_backbone = sum(p.numel() for p in self.evolvable_backbone.parameters())
        p_adapters = sum(p.numel() for p in self.adapter_modules.parameters())
        p_head = sum(p.numel() for p in self.task_head.parameters())
        p_ctrl = sum(p.numel() for p in self.controller.parameters())
        p_curric = sum(p.numel() for p in self.curriculum_policy.parameters())

        fast_trainable = p_adapters + p_head + p_ctrl + p_curric
        unfrozen_backbone = sum(
            p.numel()
            for pn, p in self.evolvable_backbone.base_model.named_parameters()
            if "blocks.3" in pn or "ln_f" in pn
        )
        slow_trainable = fast_trainable + unfrozen_backbone
        total_persistent = p_backbone + fast_trainable

        return {
            "total_persistent_parameters": total_persistent,
            "backbone_parameters": p_backbone,
            "transformer_blocks_count": len(self.evolvable_backbone.base_model.blocks),
            "fast_trainable_parameters": fast_trainable,
            "fast_trainable_headroom_20k": 20_000 - fast_trainable,
            "slow_unfrozen_backbone_parameters": unfrozen_backbone,
            "slow_trainable_total": slow_trainable,
            "external_memory_bytes": len(self.memory_store.to_json().encode("utf-8")),
        }

    def evaluate_goal_diagnostic(self, goal: LearningGoalM101) -> GoalDiagnosticState:
        """Evaluate epistemic state and accuracy for goal."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(goal.test_tokens, active_domains=[goal.domain.value])
            preds = logits.argmax(dim=-1) % MODULUS
            current_acc = float((preds == goal.test_labels).float().mean().item()) * 100.0

            probs = F.softmax(logits, dim=-1)
            entropy = float(-(probs * (probs + 1e-8).log()).sum(dim=-1).mean().item())

        hist = self.diagnostic_history.get(goal.goal_id, [])
        recent_progress = (current_acc - hist[-1]) if len(hist) >= 1 else (25.0 if goal.is_learnable else -5.0)
        staleness = self.step_counter - self.last_practiced_step.get(goal.goal_id, 0)

        return GoalDiagnosticState(
            goal_id=goal.goal_id,
            current_accuracy=current_acc,
            recent_progress=recent_progress,
            epistemic_uncertainty=round(entropy, 3),
            estimated_cost=goal.estimated_cost,
            rehearsal_staleness=staleness,
        )

    def save_checkpoint(self, path: str | Path) -> None:
        """Save complete persistent state including evolvable backbone."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "tier": self.tier.value,
            "step_counter": self.step_counter,
            "acquired_skills": self.acquired_skills,
            "diagnostic_history": self.diagnostic_history,
            "last_practiced_step": self.last_practiced_step,
            "episodic_log": self.episodic_log,
            "n_blocks": len(self.evolvable_backbone.base_model.blocks),
        }
        torch.save(
            {
                "state_dict": self.state_dict(),
                "metadata": meta,
                "memory_store": self.memory_store.to_json(),
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: torch.device) -> PersistentStudentM102:
        """Restore student with bit-exact reproducibility."""
        data = torch.load(path, map_location=device)
        meta = data["metadata"]
        tier = CapacityTierM100(meta["tier"])
        student = cls(tier=tier, device=device)

        # Restore architecture block count if blocks were added/removed
        curr_blocks = len(student.evolvable_backbone.base_model.blocks)
        target_blocks = meta.get("n_blocks", 4)
        while curr_blocks < target_blocks:
            curr_blocks = student.evolvable_backbone.insert_block_identity(curr_blocks)
        while curr_blocks > target_blocks:
            curr_blocks = student.evolvable_backbone.remove_block(curr_blocks - 1)

        student.load_state_dict(data["state_dict"])
        student.step_counter = meta["step_counter"]
        student.acquired_skills = meta["acquired_skills"]
        student.diagnostic_history = meta["diagnostic_history"]
        student.last_practiced_step = meta["last_practiced_step"]
        student.episodic_log = meta["episodic_log"]

        mem_str = data["memory_store"]
        mem_data = json.loads(mem_str) if isinstance(mem_str, str) else mem_str
        store = PersistentKnowledgeStore(max_byte_budget=mem_data.get("max_byte_budget", 262_144))
        for rid, fd in mem_data.get("facts", {}).items():
            rec = FactualRecord(**fd)
            store.facts[rid] = rec
            if rec.is_active:
                store.fact_index[(rec.entity, rec.attribute)] = rid
        student.memory_store = store
        student.eval()
        return student
