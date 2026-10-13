"""Trainer for Adaptive Mini-GPT with metrics logging, mixed precision, and resumption."""

from __future__ import annotations

import json
import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from adaptive_mini_gpt.data.dataset import TokenDataset
from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.training.checkpoint import CheckpointManager
from adaptive_mini_gpt.training.config import TrainingConfig


def build_cosine_lr_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Build a cosine annealing learning rate scheduler with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * coeff

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Trainer orchestrating optimization, evaluation, checkpointing, and metrics."""

    def __init__(
        self,
        model: MiniGPT,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        train_dataset: TokenDataset,
        val_dataset: TokenDataset,
        tokenizer: CharacterTokenizer,
        dataset_fingerprint: str,
        run_id: str | None = None,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.config = training_config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.dataset_fingerprint = dataset_fingerprint

        self.run_id = run_id or f"run_{int(time.time())}"
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.save_dir / "metrics.jsonl"

        # Resolve device and identify GPU type
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        if self.device.type == "cuda" and torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(self.device)
        else:
            self.gpu_name = "CPU"

        self.model.to(self.device)

        # Set seeds
        self._set_seed(self.config.seed)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        self.scheduler: LRScheduler | None = None
        if self.config.lr_schedule == "cosine":
            min_ratio = self.config.min_lr / self.config.learning_rate
            self.scheduler = build_cosine_lr_schedule(
                self.optimizer,
                warmup_steps=self.config.warmup_steps,
                max_steps=self.config.max_steps,
                min_lr_ratio=min_ratio,
            )

        # Mixed precision & GradScaler
        self.use_amp = (self.config.precision in ("fp16", "bf16")) and (self.device.type == "cuda")
        self.amp_dtype = torch.float16 if self.config.precision == "fp16" else torch.bfloat16
        scaler_enabled = (self.config.precision == "fp16") and (self.device.type == "cuda")
        device_for_scaler = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler(device_for_scaler, enabled=scaler_enabled)

        # Checkpoint manager with optional backup dir (e.g. Google Drive)
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.save_dir,
            backup_dir=self.config.backup_dir,
            keep_last_n=self.config.keep_last_n_checkpoints,
        )

        # State tracking
        self.step = 0
        self.tokens_seen = 0
        self.data_position = 0
        self.best_val_loss = float("inf")
        self.last_checkpoint_time = time.time()

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _log_metric(self, record: dict[str, Any]) -> None:
        record["timestamp"] = datetime.now(UTC).isoformat()
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def resume_from_checkpoint(
        self, checkpoint_path: str | Path, strict_dataset_match: bool = True
    ) -> None:
        """Resume training state from a saved checkpoint with auto-recovery."""
        meta = self.checkpoint_manager.restore_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            tokenizer=self.tokenizer,
            dataset_fingerprint=self.dataset_fingerprint,
            scaler=self.scaler,
            scheduler=self.scheduler,
            strict_dataset_match=strict_dataset_match,
            auto_recover=True,
        )
        self.step = meta["step"]
        self.tokens_seen = meta["tokens_seen"]
        self.data_position = meta["data_position"]
        self.best_val_loss = meta["best_val_loss"]
        self.run_id = meta["run_id"]
        self.last_checkpoint_time = time.time()

    @torch.no_grad()
    def evaluate(self, eval_steps: int | None = None) -> tuple[float, float]:
        """Compute validation loss and perplexity."""
        self.model.eval()
        steps = eval_steps or self.config.eval_steps
        total_loss = 0.0

        for i in range(steps):
            x, y, _ = self.val_dataset.get_sequential_batch(
                start_idx=i * self.config.batch_size * self.model_config.context_length,
                batch_size=self.config.batch_size,
                device=self.device,
            )
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                _, loss = self.model(x, y)
            if loss is not None:
                total_loss += loss.item()

        mean_loss = total_loss / steps if steps > 0 else 0.0
        perplexity = math.exp(min(mean_loss, 20.0))
        self.model.train()
        return mean_loss, perplexity

    def _save_checkpoint_now(self, checkpoint_name: str, is_best: bool = False) -> float:
        """Helper to invoke CheckpointManager and record timing."""
        t0 = time.time()
        self.checkpoint_manager.save_checkpoint(
            checkpoint_name=checkpoint_name,
            model=self.model,
            optimizer=self.optimizer,
            model_config=self.model_config,
            training_config=self.config,
            tokenizer=self.tokenizer,
            dataset_fingerprint=self.dataset_fingerprint,
            step=self.step,
            tokens_seen=self.tokens_seen,
            data_position=self.data_position,
            best_val_loss=self.best_val_loss,
            run_id=self.run_id,
            scaler=self.scaler,
            scheduler=self.scheduler,
            is_best=is_best,
        )
        duration = time.time() - t0
        self._log_metric({
            "type": "checkpoint",
            "step": self.step,
            "checkpoint_name": checkpoint_name,
            "checkpoint_duration_sec": round(duration, 4),
            "backup_copied": bool(self.config.backup_dir),
        })
        return duration

    def train(self) -> dict[str, Any]:
        """Execute the training loop with mixed precision and progress tracking."""
        self.model.train()
        start_time = time.time()
        step_tokens = (
            self.config.batch_size
            * self.model_config.context_length
            * self.config.gradient_accumulation
        )

        print(
            f"Starting training on device={self.device} ({self.gpu_name}) "
            f"precision={self.config.precision} for {self.config.max_steps} steps "
            f"(resumed at step {self.step})."
        )

        while self.step < self.config.max_steps:
            step_start = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            # Gradient accumulation loop
            for _ in range(self.config.gradient_accumulation):
                x, y, self.data_position = self.train_dataset.get_sequential_batch(
                    start_idx=self.data_position,
                    batch_size=self.config.batch_size,
                    device=self.device,
                )
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    _, loss = self.model(x, y)

                if loss is None:
                    continue

                scaled_loss = loss / self.config.gradient_accumulation
                if self.scaler.is_enabled():
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                accum_loss += loss.item() / self.config.gradient_accumulation

            # Gradient clipping & Optimizer Step
            if self.scaler.is_enabled():
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.step += 1
            self.tokens_seen += step_tokens

            step_duration = max(time.time() - step_start, 1e-5)
            throughput = step_tokens / step_duration
            elapsed = time.time() - start_time

            # Log training metrics
            train_record = {
                "type": "train",
                "step": self.step,
                "loss": accum_loss,
                "tokens_seen": self.tokens_seen,
                "lr": self.optimizer.param_groups[0]["lr"],
                "gpu_type": self.gpu_name,
                "step_time_sec": round(step_duration, 4),
                "throughput_tokens_per_sec": round(throughput, 1),
                "wall_clock_sec": round(elapsed, 2),
            }
            if self.device.type == "cuda" and torch.cuda.is_available():
                train_record["peak_vram_mb"] = round(
                    torch.cuda.max_memory_allocated(self.device) / (1024 * 1024), 2
                )
            else:
                train_record["peak_vram_mb"] = 0.0

            self._log_metric(train_record)

            # Periodic validation
            if self.step % self.config.eval_every_steps == 0 or self.step == self.config.max_steps:
                val_loss, val_ppl = self.evaluate()
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                val_record = {
                    "type": "val",
                    "step": self.step,
                    "val_loss": val_loss,
                    "val_perplexity": val_ppl,
                    "best_val_loss": self.best_val_loss,
                    "tokens_seen": self.tokens_seen,
                    "gpu_type": self.gpu_name,
                    "wall_clock_sec": round(elapsed, 2),
                }
                if self.device.type == "cuda" and torch.cuda.is_available():
                    val_record["peak_vram_mb"] = round(
                        torch.cuda.max_memory_allocated(self.device) / (1024 * 1024), 2
                    )
                else:
                    val_record["peak_vram_mb"] = 0.0

                self._log_metric(val_record)

                print(
                    f"Step {self.step:5d} | Train Loss: {accum_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | "
                    f"Throughput: {throughput:.0f} tok/s"
                )

                if is_best:
                    self._save_checkpoint_now("best.pt", is_best=True)

            # Periodic checkpointing
            time_since_ckpt = time.time() - self.last_checkpoint_time
            if (
                self.step % self.config.checkpoint_every_steps == 0
                or time_since_ckpt >= self.config.checkpoint_every_seconds
                or self.step == self.config.max_steps
            ):
                self._save_checkpoint_now("latest.pt", is_best=False)
                self.last_checkpoint_time = time.time()

        print(
            f"Training completed at step {self.step}. Final best val loss: {self.best_val_loss:.4f}"
        )
        return {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "best_val_loss": self.best_val_loss,
            "save_dir": str(self.save_dir),
            "gpu_type": self.gpu_name,
        }
