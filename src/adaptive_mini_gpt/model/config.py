"""Model configuration schema for Adaptive Mini-GPT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ModelConfig:
    architecture: str = "decoder_only_transformer"
    vocab_size: int = 256
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 512
    context_length: int = 256
    dropout: float = 0.1
    bias: bool = True
    tie_word_embeddings: bool = False
    layer_norm_eps: float = 1e-5
    # Positional encoding: "absolute", "rope", "alibi" (M22)
    pos_encoding: str = "absolute"
    # Adaptive / Modular MoE fields (M3/M4)
    adaptive_enabled: bool = False
    n_experts: int = 4
    top_k: int = 2
    d_expert: int = 128
    routing_mode: str = "learned_topk"  # "learned_topk" or "task"
    aux_loss_weight: float = 0.01
    z_loss_weight: float = 0.001
    # Mixture-of-Depths / Dynamic Layer Skipping (M5)
    mod_enabled: bool = False
    mod_capacity_factor: float = 0.5
    # Selective unseen-token adaptation mask (persisted across checkpoints)
    trainable_token_ids: list[int] | None = None
    # Autonomous domain gating trigger tokens (if None, defaults to trainable_token_ids)
    domain_trigger_token_ids: list[int] | None = None

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.context_length <= 0:
            raise ValueError(f"context_length must be positive, got {self.context_length}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
        if self.pos_encoding not in ("absolute", "rope", "alibi"):
            raise ValueError(
                f"pos_encoding must be one of ('absolute', 'rope', 'alibi'), got {self.pos_encoding}"
            )
        if self.adaptive_enabled:
            if self.n_experts <= 0:
                raise ValueError(f"n_experts must be positive, got {self.n_experts}")
            if not (1 <= self.top_k <= self.n_experts):
                raise ValueError(
                    f"top_k ({self.top_k}) must be in [1, n_experts ({self.n_experts})]"
                )
            if self.d_expert <= 0:
                raise ValueError(f"d_expert must be positive, got {self.d_expert}")
            if self.routing_mode not in ("learned_topk", "task"):
                raise ValueError(f"Unknown routing_mode: {self.routing_mode}")
        if self.mod_enabled:
            if not (0.0 < self.mod_capacity_factor <= 1.0):
                raise ValueError(
                    f"mod_capacity_factor must be in (0.0, 1.0], got {self.mod_capacity_factor}"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        return cls(**data)
