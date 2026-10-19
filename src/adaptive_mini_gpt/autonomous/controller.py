"""Autonomous learning controller implementing the unified M8/M10 state machine:

EVALUATE -> SELECT_TASK -> PLAN -> TRAIN -> VERIFY -> PROMOTE / ROLLBACK -> STOP
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

from adaptive_mini_gpt.adaptive.growth import grow_model_experts
from adaptive_mini_gpt.autonomous.tasks import (
    TASK_REGISTRY,
    evaluate_task_holdout,
)
from adaptive_mini_gpt.data.dataset import TokenDataset, split_data
from adaptive_mini_gpt.evaluation.evaluator import evaluate_text_retention_suite
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import LowRankOutputAdapter, MiniGPT
from adaptive_mini_gpt.training.checkpoint import CheckpointManager
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.trainer import Trainer

logger = logging.getLogger(__name__)


class ControllerState(StrEnum):
    IDLE = "IDLE"
    EVALUATE = "EVALUATE"
    SELECT_TASK = "SELECT_TASK"
    PLAN = "PLAN"
    TRAIN = "TRAIN"
    VERIFY = "VERIFY"
    PROMOTE = "PROMOTE"
    ROLLBACK = "ROLLBACK"
    STOP = "STOP"


class GrowthAction(StrEnum):
    AUTO = "auto"
    CONTINUE = "continue"
    GROW_EXPERT = "grow_expert"


@dataclass
class LearningPlan:
    task_id: str
    growth_action: str
    budget_steps: int
    learning_rate: float
    batch_size: int
    data_samples: int
    seed: int


@dataclass
class ControllerJournal:
    experiment_id: str
    state: str
    cycle_index: int = 1
    current_step: int = 0
    cumulative_steps: int = 0
    cumulative_tokens: int = 0
    cumulative_gpu_seconds: float = 0.0
    consecutive_rollbacks: int = 0
    selected_task: str | None = None
    plan: dict[str, Any] | None = None
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    candidate_metrics: dict[str, Any] = field(default_factory=dict)
    decision: str | None = None
    decision_reason: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


class LearningController:
    """Unified Autonomous Learning and Adaptive Growth Controller (M8/M10)."""

    def __init__(
        self,
        base_checkpoint_dir: str | Path,
        output_dir: str | Path,
        budget_steps: int = 200,
        max_budget_steps: int = 2000,
        min_improvement: float = 0.05,
        max_regression_tolerance: float = 0.05,
        target_accuracy_threshold: float = 0.95,
        growth_action: GrowthAction = GrowthAction.AUTO,
        force_task: str | None = None,
        device: str = "auto",
        precision: str = "fp16",
        seed: int = 42,
    ) -> None:
        self.base_checkpoint_dir = Path(base_checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.budget_steps = budget_steps
        self.max_budget_steps = max_budget_steps
        self.min_improvement = min_improvement
        self.max_regression_tolerance = max_regression_tolerance
        self.target_accuracy_threshold = target_accuracy_threshold
        self.growth_action = growth_action
        self.force_task = force_task
        self.seed = seed

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.precision = precision

        self.journal_file = self.output_dir / "controller_state.json"
        self.history_file = self.output_dir / "auto_learn_history.json"
        self.journal = self._load_or_create_journal()

    def _load_or_create_journal(self) -> ControllerJournal:
        if self.journal_file.is_file():
            try:
                with open(self.journal_file, encoding="utf-8") as f:
                    data = json.load(f)
                return ControllerJournal(**data)
            except Exception as e:
                logger.warning("Could not read journal file, starting fresh: %s", e)

        exp_id = f"exp_{int(time.time())}_c1"
        return ControllerJournal(
            experiment_id=exp_id,
            state=ControllerState.IDLE.value,
            cycle_index=1,
        )

    def _save_journal(self) -> None:
        temp_file = self.journal_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(asdict(self.journal), f, indent=2)
        os.replace(temp_file, self.journal_file)

    def _append_history(self, record: dict[str, Any]) -> None:
        history: list[dict[str, Any]] = []
        if self.history_file.is_file():
            try:
                with open(self.history_file, encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        history.append(record)
        temp_file = self.history_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        os.replace(temp_file, self.history_file)

    def load_active_model(self) -> tuple[MiniGPT, Any, ModelConfig]:
        """Load the verified active checkpoint and tokenizer using strict lineage validation."""
        active_ckpt = CheckpointManager.find_latest_valid_checkpoint(self.base_checkpoint_dir)
        raw = CheckpointManager.load_raw_checkpoint(active_ckpt)

        cfg_dict = raw.get("model_config", {})
        config = ModelConfig(**cfg_dict)
        model = MiniGPT(config)
        model.load_state_dict(raw["model_state_dict"])
        model.to(self.device)

        # Tokenizer-aware loading with fingerprint and vocab size validation
        tokenizer = CheckpointManager.load_tokenizer(raw, config)

        return model, tokenizer, config

    def step_evaluate(self, model: MiniGPT, tokenizer: Any) -> dict[str, Any]:
        """Evaluate baseline model on task battery and text retention."""
        logger.info("Evaluating baseline model across procedural task battery...")
        task_metrics: dict[str, Any] = {}
        for tid, task in TASK_REGISTRY.items():
            metrics = evaluate_task_holdout(
                model=model,
                tokenizer=tokenizer,
                task=task,
                device=self.device,
                num_samples=100,
                seed=self.seed + 100,
            )
            task_metrics[tid] = metrics

        # Evaluate prose retention suite (clean, numerical, mixed)
        retention = evaluate_text_retention_suite(
            model=model,
            tokenizer=tokenizer,
            device=self.device,
        )

        # For BPE models, evaluate instruction following
        if hasattr(tokenizer, "tokenizer_type") and tokenizer.tokenizer_type == "bpe":
            from adaptive_mini_gpt.data.instruction_data import SEED_INSTRUCTION_EXAMPLES
            from adaptive_mini_gpt.evaluation.language_evaluator import (
                evaluate_instruction_following,
            )

            inst_eval = evaluate_instruction_following(
                model=model,
                tokenizer=tokenizer,
                test_queries=SEED_INSTRUCTION_EXAMPLES,
                device=self.device,
            )
            retention["instructions"] = inst_eval

        baseline_summary = {
            "tasks": task_metrics,
            "retention": retention,
            "checkpoint_evaluated": str(self.base_checkpoint_dir),
        }
        self.journal.baseline_metrics = baseline_summary
        self.journal.state = ControllerState.SELECT_TASK.value
        self._save_journal()
        return baseline_summary

    def step_select_task(self) -> str | None:
        """Select weakest task by improvement potential."""
        if self.force_task:
            if self.force_task not in TASK_REGISTRY:
                raise ValueError(
                    f"Forced task '{self.force_task}' not in {list(TASK_REGISTRY.keys())}"
                )
            selected = self.force_task
            logger.info("Task selection forced to: %s", selected)
        else:
            task_metrics = self.journal.baseline_metrics.get("tasks", {})
            unmastered = [
                (tid, task_metrics.get(tid, {}).get("accuracy", 0.0))
                for tid in TASK_REGISTRY
                if task_metrics.get(tid, {}).get("accuracy", 0.0) < self.target_accuracy_threshold
            ]
            if not unmastered:
                logger.info(
                    "All tasks meet target accuracy threshold (>= %.2f).",
                    self.target_accuracy_threshold,
                )
                self.journal.decision = "NO_ACTION"
                self.journal.decision_reason = "All skills meet mastery threshold."
                self.journal.state = ControllerState.STOP.value
                self._save_journal()
                return None

            unmastered.sort(key=lambda x: x[1])
            selected = unmastered[0][0]
            logger.info(
                "Selected weakest task: '%s' (accuracy: %.4f)", selected, unmastered[0][1]
            )

        self.journal.selected_task = selected
        self.journal.state = ControllerState.PLAN.value
        self._save_journal()
        return selected

    def step_plan(self, model: MiniGPT, task_id: str) -> LearningPlan:
        """Formulate bounded learning and growth plan."""
        remaining_budget = self.max_budget_steps - self.journal.cumulative_steps
        if remaining_budget <= 0:
            logger.warning(
                "Compute budget exhausted (%d steps used).", self.journal.cumulative_steps
            )
            self.journal.decision = "NO_ACTION"
            self.journal.decision_reason = (
                f"Compute budget limit reached ({self.journal.cumulative_steps} steps)."
            )
            self.journal.state = ControllerState.STOP.value
            self._save_journal()
            raise RuntimeError("Compute budget exhausted.")

        effective_budget = min(self.budget_steps, remaining_budget)

        if self.growth_action == GrowthAction.GROW_EXPERT:
            action = GrowthAction.GROW_EXPERT.value
        elif self.growth_action == GrowthAction.CONTINUE:
            action = GrowthAction.CONTINUE.value
        else:
            # AUTO: First attempt continued dense training under matched compute budget.
            # Only grow experts if previous dense cycle was rejected on this skill.
            if self.journal.consecutive_rollbacks > 0 and (
                not model.config.adaptive_enabled or model.config.n_experts == 0
            ):
                action = GrowthAction.GROW_EXPERT.value
            else:
                action = GrowthAction.CONTINUE.value

        plan = LearningPlan(
            task_id=task_id,
            growth_action=action,
            budget_steps=effective_budget,
            learning_rate=1e-3,
            batch_size=16,
            data_samples=5000,
            seed=self.seed + self.journal.cycle_index * 10,
        )
        self.journal.plan = asdict(plan)
        self.journal.state = ControllerState.TRAIN.value
        self._save_journal()
        return plan

    def step_train(
        self,
        base_model: MiniGPT,
        tokenizer: Any,
        plan: LearningPlan,
    ) -> Path:
        """Train candidate model on an isolated path without touching base model."""
        task = TASK_REGISTRY[plan.task_id]
        candidate_dir = self.output_dir / f"candidate_{plan.task_id}_{self.journal.experiment_id}"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Preparing candidate model in isolated directory: %s", candidate_dir)
        candidate_model = copy.deepcopy(base_model)
        is_bpe = hasattr(tokenizer, "tokenizer_type") and tokenizer.tokenizer_type == "bpe"

        # Generate verified task data
        task_text, task_fp = task.generator_fn(plan.data_samples, plan.seed)

        if not is_bpe:
            _ = tokenizer.fit(task_text)
            new_token_ids = sorted(
                list({tokenizer.stoi[ch] for ch in task_text if ch in tokenizer.stoi})
            )
        else:
            new_token_ids = []

        # If growing experts
        if plan.growth_action == GrowthAction.GROW_EXPERT.value and (
            not candidate_model.config.adaptive_enabled or candidate_model.config.n_experts == 0
        ):
            logger.info("Applying function-preserving expert growth (2 residual experts)...")
            grow_model_experts(candidate_model, add_experts=2)
            trigger_id = tokenizer.get_token_id(task.domain_trigger)
            if trigger_id is not None:
                candidate_model.config.domain_trigger_token_ids = [trigger_id]
            if candidate_model.moe_head is None:
                candidate_model.moe_head = LowRankOutputAdapter(
                    candidate_model.config.d_model,
                    candidate_model.config.d_expert,
                    candidate_model.config.vocab_size,
                ).to(self.device)
            candidate_model.freeze_backbone(trainable_token_ids=new_token_ids)
            candidate_model.to(self.device)

        # Build datasets
        if plan.task_id == "instruction_reasoning" and is_bpe:
            from adaptive_mini_gpt.data.instruction_data import InstructionDataset

            tokens = tokenizer.encode(task_text)
            split_idx = int(len(tokens) * 0.9)
            ds_train = InstructionDataset(
                token_ids=tokens[:split_idx],
                context_length=candidate_model.config.context_length,
                assistant_token_id=tokenizer.assistant_id,
                eos_token_id=tokenizer.eos_id,
                user_token_id=tokenizer.user_id,
            )
            ds_val = InstructionDataset(
                token_ids=tokens[split_idx:],
                context_length=candidate_model.config.context_length,
                assistant_token_id=tokenizer.assistant_id,
                eos_token_id=tokenizer.eos_id,
                user_token_id=tokenizer.user_id,
            )
        else:
            tokens = torch.tensor(tokenizer.encode(task_text), dtype=torch.long)
            train_tok, val_tok, _ = split_data(tokens)
            equals_id = tokenizer.get_token_id("=")
            newline_id = tokenizer.get_token_id("\n")
            ds_train = TokenDataset(
                train_tok,
                context_length=candidate_model.config.context_length,
                mask_prompt_before_equals=True,
                equals_token_id=equals_id,
                newline_token_id=newline_id,
            )
            ds_val = TokenDataset(
                val_tok,
                context_length=candidate_model.config.context_length,
                mask_prompt_before_equals=True,
                equals_token_id=equals_id,
                newline_token_id=newline_id,
            )

        train_cfg = TrainingConfig(
            seed=plan.seed,
            batch_size=plan.batch_size,
            gradient_accumulation=2,
            learning_rate=plan.learning_rate,
            max_steps=plan.budget_steps,
            eval_every_steps=max(1, plan.budget_steps // 2),
            device=str(self.device),
            precision=self.precision,
            save_dir=str(candidate_dir),
        )

        trainer = Trainer(
            model=candidate_model,
            model_config=candidate_model.config,
            training_config=train_cfg,
            train_dataset=ds_train,
            val_dataset=ds_val,
            tokenizer=tokenizer,
            dataset_fingerprint=task_fp,
        )
        t0 = time.time()
        trainer.train()
        train_sec = time.time() - t0

        # Update budget tracking in journal
        self.journal.cumulative_steps += trainer.step
        self.journal.cumulative_tokens += trainer.tokens_seen
        self.journal.cumulative_gpu_seconds += train_sec

        self.journal.state = ControllerState.VERIFY.value
        self._save_journal()
        return candidate_dir

    def step_verify(
        self,
        candidate_dir: Path,
        tokenizer: Any,
        task_id: str,
    ) -> bool:
        """Verify candidate on target skill and evaluate retention on prior tasks."""
        cand_ckpt = CheckpointManager.find_latest_valid_checkpoint(candidate_dir)
        raw = CheckpointManager.load_raw_checkpoint(cand_ckpt)

        cfg_dict = raw.get("model_config", {})
        cand_config = ModelConfig(**cfg_dict)
        candidate_model = MiniGPT(cand_config)
        candidate_model.load_state_dict(raw["model_state_dict"])
        candidate_model.to(self.device)

        task = TASK_REGISTRY[task_id]
        cand_task_eval = evaluate_task_holdout(
            model=candidate_model,
            tokenizer=tokenizer,
            task=task,
            device=self.device,
            num_samples=100,
            seed=self.seed + 100,
        )
        cand_retention = evaluate_text_retention_suite(
            model=candidate_model,
            tokenizer=tokenizer,
            device=self.device,
        )

        # For BPE models, evaluate instruction following
        if hasattr(tokenizer, "tokenizer_type") and tokenizer.tokenizer_type == "bpe":
            from adaptive_mini_gpt.data.instruction_data import SEED_INSTRUCTION_EXAMPLES
            from adaptive_mini_gpt.evaluation.language_evaluator import (
                evaluate_instruction_following,
            )

            inst_eval = evaluate_instruction_following(
                model=candidate_model,
                tokenizer=tokenizer,
                test_queries=SEED_INSTRUCTION_EXAMPLES,
                device=self.device,
            )
            cand_retention["instructions"] = inst_eval

        self.journal.candidate_metrics = {
            "task": cand_task_eval,
            "retention": cand_retention,
            "candidate_dir": str(candidate_dir),
        }

        # Gating logic
        base_acc = (
            self.journal.baseline_metrics.get("tasks", {})
            .get(task_id, {})
            .get("accuracy", 0.0)
        )
        cand_acc = cand_task_eval.get("accuracy", 0.0)
        acc_gain = cand_acc - base_acc

        clean_delta = cand_retention.get("clean_prose", {}).get("delta_loss", 0.0)
        num_delta = cand_retention.get("numerical_prose", {}).get("delta_loss", 0.0)
        mix_delta = cand_retention.get("mixed_prose_math", {}).get("delta_loss", 0.0)
        max_retention_regression = max(clean_delta, num_delta, mix_delta)

        logger.info(
            "Gate: Task '%s' Base=%.4f, Cand=%.4f (Gain=%+.4f | Min=%.4f)",
            task_id,
            base_acc,
            cand_acc,
            acc_gain,
            self.min_improvement,
        )
        logger.info(
            "Retention: Clean=%.6f, Num=%.6f, Mix=%.6f (Max=%.4f | Allowed=%.4f)",
            clean_delta,
            num_delta,
            mix_delta,
            max_retention_regression,
            self.max_regression_tolerance,
        )

        # Strict Non-Regression + Improvement / Mastery condition
        target_non_regressed = cand_acc >= (base_acc - 1e-6)
        target_improved = acc_gain >= self.min_improvement
        target_mastered = (cand_acc >= self.target_accuracy_threshold) and target_non_regressed
        target_passed = target_non_regressed and (target_improved or target_mastered)

        retention_passed = max_retention_regression <= self.max_regression_tolerance

        # Check instruction regression for BPE models if present in baseline
        if "instructions" in cand_retention and "instructions" in self.journal.baseline_metrics.get(
            "retention", {}
        ):
            base_inst = self.journal.baseline_metrics["retention"]["instructions"]
            base_inst_acc = base_inst.get("exact_answer_accuracy", 0.0)
            cand_inst_acc = cand_retention["instructions"].get("exact_answer_accuracy", 0.0)
            if cand_inst_acc < (base_inst_acc - self.max_regression_tolerance):
                retention_passed = False

        if target_passed and retention_passed:
            decision = ControllerState.PROMOTE.value
            reason = (
                f"Accuracy improved by {acc_gain:+.4f} (>= {self.min_improvement}) "
                f"with acceptable retention drift ({max_retention_regression:.6f})."
            )
        else:
            decision = ControllerState.ROLLBACK.value
            reasons: list[str] = []
            if not target_non_regressed:
                reasons.append(
                    f"Target regressed ({base_acc:.4f} -> {cand_acc:.4f}, drop {acc_gain:.4f})"
                )
            elif not (target_improved or target_mastered):
                reasons.append(
                    f"Target accuracy gain {acc_gain:+.4f} below min {self.min_improvement}"
                )
            if not retention_passed:
                drift_str = f"Retention drift {max_retention_regression:.6f}"
                reasons.append(f"{drift_str} exceeds tolerance {self.max_regression_tolerance}")
            reason = "; ".join(reasons)

        self.journal.decision = decision
        self.journal.decision_reason = reason
        self.journal.state = decision
        self._save_journal()
        return decision == ControllerState.PROMOTE.value

    def step_promote(self, candidate_dir: Path) -> Path:
        """Promote candidate model using verified staging and atomic version pointer."""
        cand_ckpt = CheckpointManager.find_latest_valid_checkpoint(candidate_dir)
        raw = CheckpointManager.load_raw_checkpoint(cand_ckpt)

        # 1. Immutable versioned publication directory
        version_idx = len(self.journal.history) + 1
        published_dir = (
            self.output_dir / "published" / f"v_{version_idx:04d}_{self.journal.experiment_id}"
        )
        published_dir.mkdir(parents=True, exist_ok=True)
        pub_target = published_dir / "checkpoint.pt"
        pub_temp = published_dir / "checkpoint.pt.tmp"

        torch.save(raw, pub_temp)
        CheckpointManager.load_raw_checkpoint(pub_temp)  # Smoke test verification
        os.replace(pub_temp, pub_target)

        manifest_data = {
            "version_index": version_idx,
            "promoted_at": time.time(),
            "experiment_id": self.journal.experiment_id,
            "selected_task": self.journal.selected_task,
            "decision": "PROMOTE",
            "candidate_dir": str(candidate_dir),
            "published_checkpoint": str(pub_target),
            "step": raw.get("step", 0),
            "tokens_seen": raw.get("tokens_seen", 0),
            "tokenizer_fingerprint": raw.get("tokenizer_fingerprint", ""),
            "model_config": raw.get("model_config", {}),
        }
        with open(published_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # 2. Atomic update to active_model pointer directory
        active_dir = self.output_dir / "active_model"
        active_dir.mkdir(parents=True, exist_ok=True)
        active_target = active_dir / "latest.pt"
        active_temp = active_dir / "latest.pt.tmp"

        torch.save(raw, active_temp)
        CheckpointManager.load_raw_checkpoint(active_temp)  # Smoke test
        os.replace(active_temp, active_target)  # Atomic POSIX replacement

        manifest_temp = active_dir / "manifest.json.tmp"
        with open(manifest_temp, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        os.replace(manifest_temp, active_dir / "manifest.json")

        version_ptr_temp = active_dir / "active_version.json.tmp"
        with open(version_ptr_temp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version_index": version_idx,
                    "published_path": str(pub_target),
                    "timestamp": time.time(),
                },
                f,
                indent=2,
            )
        os.replace(version_ptr_temp, active_dir / "active_version.json")

        # 3. Update controller state & cumulative history
        record = {
            "cycle_index": self.journal.cycle_index,
            "experiment_id": self.journal.experiment_id,
            "decision": "PROMOTE",
            "reason": self.journal.decision_reason,
            "selected_task": self.journal.selected_task,
            "plan": self.journal.plan,
            "baseline_metrics": self.journal.baseline_metrics,
            "candidate_metrics": self.journal.candidate_metrics,
            "promoted_path": str(active_target),
            "published_path": str(pub_target),
            "cumulative_steps": self.journal.cumulative_steps,
            "cumulative_tokens": self.journal.cumulative_tokens,
            "cumulative_gpu_seconds": round(self.journal.cumulative_gpu_seconds, 2),
            "timestamp": time.time(),
        }
        self._append_history(record)
        self.base_checkpoint_dir = active_dir
        self.journal.consecutive_rollbacks = 0
        self.journal.state = ControllerState.STOP.value
        self._save_journal()
        logger.info(
            "Successfully PROMOTED candidate to: %s (Published: %s)", active_target, pub_target
        )
        return active_target

    def step_rollback(self, candidate_dir: Path) -> None:
        """Roll back candidate model and preserve base checkpoint."""
        record = {
            "cycle_index": self.journal.cycle_index,
            "experiment_id": self.journal.experiment_id,
            "decision": "ROLLBACK",
            "reason": self.journal.decision_reason,
            "selected_task": self.journal.selected_task,
            "plan": self.journal.plan,
            "baseline_metrics": self.journal.baseline_metrics,
            "candidate_metrics": self.journal.candidate_metrics,
            "base_checkpoint_preserved": str(self.base_checkpoint_dir),
            "cumulative_steps": self.journal.cumulative_steps,
            "cumulative_tokens": self.journal.cumulative_tokens,
            "cumulative_gpu_seconds": round(self.journal.cumulative_gpu_seconds, 2),
            "timestamp": time.time(),
        }
        self._append_history(record)
        self.journal.consecutive_rollbacks += 1
        self.journal.state = ControllerState.STOP.value
        self._save_journal()
        logger.info(
            "Candidate REJECTED: %s. Base checkpoint preserved at: %s",
            self.journal.decision_reason,
            self.base_checkpoint_dir,
        )

    def prepare_next_cycle(self) -> None:
        """Reset state machine for the next consecutive learning cycle."""
        active_ckpt = self.output_dir / "active_model" / "latest.pt"
        if active_ckpt.is_file():
            self.base_checkpoint_dir = self.output_dir / "active_model"

        self.journal.cycle_index += 1
        self.journal.experiment_id = f"exp_{int(time.time())}_c{self.journal.cycle_index}"
        self.journal.state = ControllerState.IDLE.value
        self.journal.selected_task = None
        self.journal.plan = None
        self.journal.candidate_metrics = {}
        self.journal.decision = None
        self.journal.decision_reason = None
        self._save_journal()
        logger.info("Prepared consecutive learning cycle [%s]", self.journal.experiment_id)

    def run_cycles(self, max_cycles: int = 1) -> list[dict[str, Any]]:
        """Run multiple consecutive autonomous learning cycles."""
        results: list[dict[str, Any]] = []
        for _ in range(max_cycles):
            if self.journal.state == ControllerState.STOP.value:
                if self.journal.decision == "NO_ACTION":
                    logger.info("Termination condition reached: no further weak skills to learn.")
                    break
                if self.journal.consecutive_rollbacks >= 2:
                    logger.info("Termination condition reached: 2 consecutive rollbacks.")
                    break
                if self.journal.cumulative_steps >= self.max_budget_steps:
                    logger.info(
                        "Compute budget limit reached (%d steps).", self.journal.cumulative_steps
                    )
                    break
                self.prepare_next_cycle()

            cycle_result = self.run_cycle()
            results.append(cycle_result)
            if self.journal.decision == "NO_ACTION":
                break
        return results

    def run_cycle(self) -> dict[str, Any]:
        """Execute one complete autonomous learning cycle."""
        logger.info(
            "Starting Autonomous Learning Cycle [%s] (Cycle %d)",
            self.journal.experiment_id,
            self.journal.cycle_index,
        )

        model, tokenizer, _ = self.load_active_model()
        if self.journal.state in (ControllerState.IDLE.value, ControllerState.EVALUATE.value):
            self.step_evaluate(model, tokenizer)

        if self.journal.state == ControllerState.SELECT_TASK.value:
            task_id = self.step_select_task()
            if not task_id:
                return asdict(self.journal)
        else:
            task_id = self.journal.selected_task
            if not task_id:
                return asdict(self.journal)

        if self.journal.state == ControllerState.PLAN.value:
            plan = self.step_plan(model, task_id)
        else:
            plan_dict = self.journal.plan or {}
            plan = LearningPlan(**plan_dict)

        if self.journal.state == ControllerState.TRAIN.value:
            candidate_dir = self.step_train(model, tokenizer, plan)
        else:
            cand_dir_str = self.journal.candidate_metrics.get("candidate_dir")
            candidate_dir = (
                Path(cand_dir_str)
                if cand_dir_str
                else self.output_dir / f"candidate_{plan.task_id}_{self.journal.experiment_id}"
            )

        if self.journal.state == ControllerState.VERIFY.value:
            _ = self.step_verify(candidate_dir, tokenizer, task_id)

        if self.journal.state == ControllerState.PROMOTE.value:
            self.step_promote(candidate_dir)
        elif self.journal.state == ControllerState.ROLLBACK.value:
            self.step_rollback(candidate_dir)

        logger.info(
            "Autonomous Learning Cycle [%s] Finished with Decision: %s",
            self.journal.experiment_id,
            self.journal.decision,
        )
        return asdict(self.journal)
