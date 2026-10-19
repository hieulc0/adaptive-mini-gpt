"""Training configuration schema for Adaptive Mini-GPT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 16
    gradient_accumulation: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 1000
    eval_every_steps: int = 100
    eval_steps: int = 20
    checkpoint_every_seconds: float = 600.0
    checkpoint_every_steps: int = 500
    device: str = "auto"  # "auto", "cpu", "cuda"
    precision: str = "fp32"  # "fp32", "fp16", "bf16"
    grad_clip: float = 1.0
    save_dir: str = "runs/exp-001"
    backup_dir: str | None = None  # Google Drive or persistent external backup path
    lr_schedule: str = "cosine"  # "constant", "cosine"
    warmup_steps: int = 50
    min_lr: float = 1e-5
    keep_last_n_checkpoints: int = 5
    freeze_backbone: bool = False
    task_id: int | None = None
    json_output: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.gradient_accumulation <= 0:
            raise ValueError(
                f"gradient_accumulation must be positive, got {self.gradient_accumulation}"
            )
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.eval_every_steps <= 0:
            raise ValueError(f"eval_every_steps must be positive, got {self.eval_every_steps}")
        if self.eval_steps <= 0:
            raise ValueError(f"eval_steps must be positive, got {self.eval_steps}")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"precision must be fp32, fp16, or bf16, got {self.precision}")
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError(f"lr_schedule must be constant or cosine, got {self.lr_schedule}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)
