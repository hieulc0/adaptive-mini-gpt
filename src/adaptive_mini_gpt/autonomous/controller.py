"""Autonomous learning controller implementing the M8 state machine:

EVALUATE -> SELECT_TASK -> PLAN -> TRAIN -> VERIFY -> PROMOTE / ROLLBACK -> STOP
"""

from __future__ import annotations

import copy
import json
import logging
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
from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer
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
    current_step: int = 0
    selected_task: str | None = None
    plan: dict[str, Any] | None = None
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    candidate_metrics: dict[str, Any] = field(default_factory=dict)
    decision: str | None = None
    decision_reason: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


class LearningController:
    """Autonomous Learning and Growth Controller (M8)."""

    def __init__(
        self,
        base_checkpoint_dir: str | Path,
        output_dir: str | Path,
        budget_steps: int = 200,
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

        exp_id = f"exp_{int(time.time())}"
        return ControllerJournal(
            experiment_id=exp_id,
            state=ControllerState.IDLE.value,
        )

    def _save_journal(self) -> None:
        temp_file = self.journal_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(asdict(self.journal), f, indent=2)
        temp_file.replace(self.journal_file)

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
        temp_file.replace(self.history_file)

    def load_active_model(self) -> tuple[MiniGPT, CharacterTokenizer, ModelConfig]:
        """Load the verified active checkpoint and tokenizer."""
        active_ckpt = CheckpointManager.find_latest_valid_checkpoint(self.base_checkpoint_dir)
        raw = CheckpointManager.load_raw_checkpoint(active_ckpt)

        cfg_dict = raw.get("model_config", {})
        config = ModelConfig(**cfg_dict)
        model = MiniGPT(config)
        model.load_state_dict(raw["model_state_dict"])
        model.to(self.device)

        tok_dict = raw.get("tokenizer")
        if tok_dict:
            tokenizer = CharacterTokenizer.from_dict(tok_dict)
        elif raw.get("tokenizer_vocab"):
            tokenizer = CharacterTokenizer.from_dict({"vocab": raw["tokenizer_vocab"]})
        else:
            tokenizer = CharacterTokenizer(use_base_ascii=True).fit("")

        return model, tokenizer, config

    def step_evaluate(
        self, model: MiniGPT, tokenizer: CharacterTokenizer
    ) -> dict[str, Any]:
        """Evaluate baseline model on task battery and text retention."""
        logger.info("Evaluating baseline model across procedural task battery...")
        task_metrics: dict[str, Any] = {}
        for tid, task in TASK_REGISTRY.items():
            metrics = evaluate_task_holdout(
                model=model,
                tokenizer=tokenizer,
                task=task,
                device=self.device,
                num_samples=50,
                seed=self.seed + 100,
            )
            task_metrics[tid] = metrics

        # Evaluate prose retention suite
        retention = evaluate_text_retention_suite(
            model=model,
            tokenizer=tokenizer,
            device=self.device,
        )

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
        if self.growth_action == GrowthAction.GROW_EXPERT:
            action = GrowthAction.GROW_EXPERT.value
        elif self.growth_action == GrowthAction.CONTINUE:
            action = GrowthAction.CONTINUE.value
        else:
            # AUTO: Grow if model has 0 experts
            if not model.config.adaptive_enabled or model.config.n_experts == 0:
                action = GrowthAction.GROW_EXPERT.value
            else:
                action = GrowthAction.CONTINUE.value

        plan = LearningPlan(
            task_id=task_id,
            growth_action=action,
            budget_steps=self.budget_steps,
            learning_rate=1e-3,
            batch_size=16,
            data_samples=5000,
            seed=self.seed,
        )
        self.journal.plan = asdict(plan)
        self.journal.state = ControllerState.TRAIN.value
        self._save_journal()
        return plan

    def step_train(
        self,
        base_model: MiniGPT,
        tokenizer: CharacterTokenizer,
        plan: LearningPlan,
    ) -> Path:
        """Train candidate model on an isolated path without touching base model."""
        task = TASK_REGISTRY[plan.task_id]
        candidate_dir = self.output_dir / f"candidate_{plan.task_id}_{self.journal.experiment_id}"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Preparing candidate model in isolated directory: %s", candidate_dir)
        candidate_model = copy.deepcopy(base_model)

        # Generate verified task data
        task_text, task_fp = task.generator_fn(plan.data_samples, plan.seed)
        _ = tokenizer.fit(task_text)
        new_token_ids = sorted(
            list({tokenizer.stoi[ch] for ch in task_text if ch in tokenizer.stoi})
        )

        # If growing experts
        if plan.growth_action == GrowthAction.GROW_EXPERT.value and (
            not candidate_model.config.adaptive_enabled or candidate_model.config.n_experts == 0
        ):
            logger.info("Applying function-preserving expert growth (2 residual experts)...")
            grow_model_experts(candidate_model, add_experts=2)
            candidate_model.config.domain_trigger_token_ids = [tokenizer.stoi[task.domain_trigger]]
            if candidate_model.moe_head is None:
                candidate_model.moe_head = LowRankOutputAdapter(
                    candidate_model.config.d_model,
                    candidate_model.config.d_expert,
                    candidate_model.config.vocab_size,
                ).to(self.device)
            candidate_model.freeze_backbone(trainable_token_ids=new_token_ids)
            candidate_model.to(self.device)

        # Build datasets
        tokens = torch.tensor(tokenizer.encode(task_text), dtype=torch.long)
        train_tok, val_tok, _ = split_data(tokens)
        ds_train = TokenDataset(
            train_tok,
            context_length=candidate_model.config.context_length,
            mask_prompt_before_equals=True,
            equals_token_id=tokenizer.stoi.get("="),
            newline_token_id=tokenizer.stoi.get("\n"),
        )
        ds_val = TokenDataset(
            val_tok,
            context_length=candidate_model.config.context_length,
            mask_prompt_before_equals=True,
            equals_token_id=tokenizer.stoi.get("="),
            newline_token_id=tokenizer.stoi.get("\n"),
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
        trainer.train()

        self.journal.state = ControllerState.VERIFY.value
        self._save_journal()
        return candidate_dir

    def step_verify(
        self,
        candidate_dir: Path,
        tokenizer: CharacterTokenizer,
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
            num_samples=50,
            seed=self.seed + 100,
        )
        cand_retention = evaluate_text_retention_suite(
            model=candidate_model,
            tokenizer=tokenizer,
            device=self.device,
        )

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
        max_retention_regression = max(clean_delta, num_delta)

        logger.info(
            "Gate: Task '%s' Base=%.4f, Cand=%.4f (Gain=%+.4f | Min=%.4f)",
            task_id,
            base_acc,
            cand_acc,
            acc_gain,
            self.min_improvement,
        )
        logger.info(
            "Retention: Clean=%.6f, Num=%.6f (Max=%.4f | Allowed=%.4f)",
            clean_delta,
            num_delta,
            max_retention_regression,
            self.max_regression_tolerance,
        )

        target_passed = (
            acc_gain >= self.min_improvement
            or cand_acc >= self.target_accuracy_threshold
        )
        retention_passed = max_retention_regression <= self.max_regression_tolerance

        if target_passed and retention_passed:
            decision = ControllerState.PROMOTE.value
            reason = (
                f"Accuracy improved by {acc_gain:+.4f} (>= {self.min_improvement}) "
                f"with acceptable retention drift ({max_retention_regression:.6f})."
            )
        else:
            decision = ControllerState.ROLLBACK.value
            reasons: list[str] = []
            if not target_passed:
                reasons.append(
                    f"Target accuracy gain {acc_gain:+.4f} below min {self.min_improvement}"
                )
            if not retention_passed:
                drift_str = f"Retention drift {max_retention_regression:.6f}"
                reasons.append(f"{drift_str} > {self.max_regression_tolerance}")
            reason = "; ".join(reasons)

        self.journal.decision = decision
        self.journal.decision_reason = reason
        self.journal.state = decision
        self._save_journal()
        return decision == ControllerState.PROMOTE.value

    def step_promote(self, candidate_dir: Path) -> Path:
        """Promote candidate model as the new active baseline."""
        promoted_dir = self.output_dir / "active_model"
        promoted_dir.mkdir(parents=True, exist_ok=True)

        cand_ckpt = CheckpointManager.find_latest_valid_checkpoint(candidate_dir)
        raw = CheckpointManager.load_raw_checkpoint(cand_ckpt)

        target_file = promoted_dir / "latest.pt"
        torch.save(raw, target_file)

        manifest_data = {
            "promoted_at": time.time(),
            "experiment_id": self.journal.experiment_id,
            "selected_task": self.journal.selected_task,
            "decision": "PROMOTE",
            "candidate_dir": str(candidate_dir),
            "latest_checkpoint": target_file.name,
        }
        with open(promoted_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        record = {
            "experiment_id": self.journal.experiment_id,
            "decision": "PROMOTE",
            "reason": self.journal.decision_reason,
            "selected_task": self.journal.selected_task,
            "plan": self.journal.plan,
            "baseline_metrics": self.journal.baseline_metrics,
            "candidate_metrics": self.journal.candidate_metrics,
            "promoted_path": str(target_file),
            "timestamp": time.time(),
        }
        self._append_history(record)
        self.journal.state = ControllerState.STOP.value
        self._save_journal()
        logger.info("Successfully PROMOTED candidate to: %s", target_file)
        return target_file

    def step_rollback(self, candidate_dir: Path) -> None:
        """Roll back candidate model and preserve base checkpoint."""
        record = {
            "experiment_id": self.journal.experiment_id,
            "decision": "ROLLBACK",
            "reason": self.journal.decision_reason,
            "selected_task": self.journal.selected_task,
            "plan": self.journal.plan,
            "baseline_metrics": self.journal.baseline_metrics,
            "candidate_metrics": self.journal.candidate_metrics,
            "base_checkpoint_preserved": str(self.base_checkpoint_dir),
            "timestamp": time.time(),
        }
        self._append_history(record)
        self.journal.state = ControllerState.STOP.value
        self._save_journal()
        logger.info(
            "Candidate REJECTED: %s. Base checkpoint preserved.", self.journal.decision_reason
        )

    def run_cycle(self) -> dict[str, Any]:
        """Execute one complete autonomous learning cycle."""
        logger.info("Starting Autonomous Learning Cycle [%s]", self.journal.experiment_id)

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
