"""Decoder-only Transformer implementation for Adaptive Mini-GPT."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW

from adaptive_mini_gpt.adaptive.experts import AdaptiveMoELayer
from adaptive_mini_gpt.model.attention import CausalSelfAttention
from adaptive_mini_gpt.model.config import ModelConfig


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm Transformer decoder block with optional Modular Residual Experts & MoD."""

    def __init__(self, config: ModelConfig, zero_init_moe: bool = False) -> None:
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = FeedForward(config)

        self.moe: AdaptiveMoELayer | None = None
        if config.adaptive_enabled:
            self.moe = AdaptiveMoELayer(
                d_model=config.d_model,
                d_expert=config.d_expert,
                n_experts=config.n_experts,
                top_k=config.top_k,
                routing_mode=config.routing_mode,
                dropout=config.dropout,
                bias=config.bias,
                zero_init_experts=zero_init_moe,
            )

        self.mod_router: nn.Linear | None = None
        if config.mod_enabled:
            self.mod_router = nn.Linear(config.d_model, 1, bias=config.bias)

    def enable_zero_init_moe(
        self,
        config: ModelConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Attach a zero-initialized MoE layer for function-preserving M4 growth."""
        self.moe = AdaptiveMoELayer(
            d_model=config.d_model,
            d_expert=config.d_expert,
            n_experts=config.n_experts,
            top_k=config.top_k,
            routing_mode=config.routing_mode,
            dropout=config.dropout,
            bias=config.bias,
            zero_init_experts=True,
        ).to(device=device, dtype=dtype)

    def forward(
        self,
        x: Tensor,
        task_id: int | None = None,
        bypass_moe: bool = False,
    ) -> tuple[Tensor, dict[str, Any] | None]:
        if self.mod_router is not None:
            # Mixture-of-Depths (M5): dynamically select top-k tokens to process through block
            router_logits = self.mod_router(x).squeeze(-1)  # (B, T)
            b_sz, t_sz = router_logits.size()
            k = max(1, min(t_sz, int(t_sz * self.config.mod_capacity_factor)))
            _, topk_indices = torch.topk(router_logits, k, dim=-1)
            mod_mask = torch.zeros((b_sz, t_sz), dtype=torch.bool, device=x.device)
            mod_mask.scatter_(1, topk_indices, True)

            x_cand = x + self.attn(self.ln_1(x))
            normed = self.ln_2(x_cand)
            ffn_out = self.mlp(normed)
            aux_info = None
            if self.moe is not None and not bypass_moe:
                moe_out, aux_info = self.moe(normed, task_id=task_id)
                ffn_out = ffn_out + moe_out
            x_cand = x_cand + ffn_out
            x = torch.where(mod_mask.unsqueeze(-1), x_cand, x)
            return x, aux_info

        x = x + self.attn(self.ln_1(x))
        normed = self.ln_2(x)
        ffn_out = self.mlp(normed)
        aux_info = None
        if self.moe is not None and not bypass_moe:
            moe_out, aux_info = self.moe(normed, task_id=task_id)
            ffn_out = ffn_out + moe_out
        x = x + ffn_out
        return x, aux_info


class MiniGPT(nn.Module):
    """Decoder-only Transformer language model with optional adaptive expert routing."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.context_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layers) ** 0.5)

        # Telemetry from the most recent forward pass
        self.last_aux_info: dict[str, Any] = {}

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def freeze_backbone(self, freeze_existing_experts: bool = False) -> None:
        """Freeze shared dense backbone parameters so only modular experts/routers train."""
        for name, param in self.named_parameters():
            if ".moe." in name:
                if freeze_existing_experts and ".experts." in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = False

    def configure_optimizers(
        self,
        learning_rate: float,
        weight_decay: float,
    ) -> AdamW:
        """Create AdamW with decoupled weight decay applied only to >=2D weight tensors."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return AdamW(optim_groups, lr=learning_rate)

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        task_id: int | None = None,
        bypass_moe: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass through the Transformer.

        Returns:
            Tuple of (logits, total_loss).
            Detailed routing metrics are saved in `self.last_aux_info`.
        """
        b, t = idx.size()
        if t > self.config.context_length:
            raise ValueError(
                f"Cannot forward sequence of length {t}, block size is {self.config.context_length}"
            )

        device = idx.device
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        total_balance_loss = torch.tensor(0.0, device=device)
        total_z_loss = torch.tensor(0.0, device=device)
        entropies: list[float] = []
        utilizations: list[Tensor] = []

        for block in self.blocks:
            x, aux = block(x, task_id=task_id, bypass_moe=bypass_moe)
            if aux is not None:
                total_balance_loss = total_balance_loss + aux["balance_loss"]
                total_z_loss = total_z_loss + aux["z_loss"]
                entropies.append(aux["entropy"])
                utilizations.append(aux["utilization"])

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss: Tensor | None = None
        ce_loss_val = 0.0
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            ce_loss_val = ce_loss.item()
            if self.config.adaptive_enabled and len(entropies) > 0 and not bypass_moe:
                n_moe = len(entropies)
                mean_bal = total_balance_loss / n_moe
                mean_z = total_z_loss / n_moe
                loss = (
                    ce_loss
                    + self.config.aux_loss_weight * mean_bal
                    + self.config.z_loss_weight * mean_z
                )
            else:
                loss = ce_loss

        if entropies and not bypass_moe:
            mean_util = torch.stack(utilizations).mean(dim=0).tolist()
            self.last_aux_info = {
                "ce_loss": ce_loss_val,
                "balance_loss": (total_balance_loss / len(entropies)).item(),
                "z_loss": (total_z_loss / len(entropies)).item(),
                "router_entropy": sum(entropies) / len(entropies),
                "expert_utilization": [round(u, 4) for u in mean_util],
            }
        else:
            self.last_aux_info = {"ce_loss": ce_loss_val}

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        eos_id: int | None = None,
        task_id: int | None = None,
        bypass_moe: bool = False,
    ) -> Tensor:
        """Autoregressively generate new tokens with top-k, top-p, and repetition penalty."""
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.context_length
                else idx[:, -self.config.context_length :]
            )
            logits, _ = self(idx_cond, task_id=task_id, bypass_moe=bypass_moe)
            logits = logits[:, -1, :]  # (B, V)

            # Apply repetition penalty to tokens already in context
            if repetition_penalty != 1.0 and repetition_penalty > 0.0:
                for b_idx in range(idx.size(0)):
                    seen_tokens = torch.unique(idx_cond[b_idx])
                    seen_logits = logits[b_idx, seen_tokens]
                    logits[b_idx, seen_tokens] = torch.where(
                        seen_logits < 0,
                        seen_logits * repetition_penalty,
                        seen_logits / repetition_penalty,
                    )

            if temperature <= 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)

                # Top-K filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                # Top-P (Nucleus) filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift right so the first token above threshold is always kept
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits = logits.masked_fill(indices_to_remove, float("-inf"))

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
            if eos_id is not None and (idx_next == eos_id).all():
                break

        return idx

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return total parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
            if not self.config.tie_word_embeddings:
                n_params -= self.wte.weight.numel()
        return n_params

    def get_active_params_per_token(self) -> int:
        """Return active parameters per token (accounts for sparse top-k expert selection)."""
        total = self.get_num_params(non_embedding=False)
        if not self.config.adaptive_enabled:
            return total
        inactive = 0
        for block in self.blocks:
            if isinstance(block, TransformerBlock) and block.moe is not None:
                n_inactive = max(0, block.moe.n_experts - block.moe.top_k)
                if n_inactive > 0 and len(block.moe.experts) > 0:
                    exp_params = sum(p.numel() for p in block.moe.experts[0].parameters())
                    inactive += n_inactive * exp_params
        return total - inactive

    def inspect_architecture(self) -> dict[str, Any]:
        """Return architectural details, total/trainable/active parameter counts."""
        layer_details: list[dict[str, Any]] = []
        for i, block in enumerate(self.blocks):
            layer_params = sum(p.numel() for p in block.parameters())
            layer_details.append({"layer": i, "params": layer_params})

        return {
            "architecture": self.config.architecture,
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "head_dim": self.config.head_dim,
            "d_ff": self.config.d_ff,
            "context_length": self.config.context_length,
            "vocab_size": self.config.vocab_size,
            "tie_word_embeddings": self.config.tie_word_embeddings,
            "adaptive_enabled": self.config.adaptive_enabled,
            "n_experts": self.config.n_experts if self.config.adaptive_enabled else 0,
            "top_k": self.config.top_k if self.config.adaptive_enabled else 0,
            "mod_enabled": self.config.mod_enabled,
            "mod_capacity_factor": (
                self.config.mod_capacity_factor if self.config.mod_enabled else 1.0
            ),
            "total_params": self.get_num_params(non_embedding=False),
            "active_params_per_token": self.get_active_params_per_token(),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "non_embedding_params": self.get_num_params(non_embedding=True),
            "layers": layer_details,
        }
