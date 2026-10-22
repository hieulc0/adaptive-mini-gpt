"""Decoder-only Transformer implementation for Adaptive Mini-GPT."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW

from adaptive_mini_gpt.adaptive.experts import AdaptiveMoELayer
from adaptive_mini_gpt.adaptive.mod import MixtureOfDepthsRouter
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


class LowRankOutputAdapter(nn.Module):
    """Zero-initialized residual output head adapter for continual skill acquisition."""

    def __init__(self, d_model: int, d_bottleneck: int, vocab_size: int) -> None:
        super().__init__()
        self.w_up = nn.Linear(d_model, d_bottleneck, bias=False)
        self.act = nn.GELU()
        self.w_down = nn.Linear(d_bottleneck, vocab_size, bias=False)
        nn.init.normal_(self.w_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.w_down.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(self.act(self.w_up(x)))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer decoder block with Modular Residual Experts & Mixture-of-Depths."""

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

        self.mod_router: MixtureOfDepthsRouter | None = None
        if config.mod_enabled:
            self.mod_router = MixtureOfDepthsRouter(
                d_model=config.d_model,
                capacity_factor=config.mod_capacity_factor,
                bias=config.bias,
            )

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
        domain_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, Any] | None]:
        x = x + self.attn(self.ln_1(x))\n        normed = self.ln_2(x)

        if self.mod_router is not None and not bypass_moe:

            def _ffn_moe_sublayer(sub_x: Tensor) -> tuple[Tensor, dict[str, Any] | None]:
                out_sub = self.mlp(sub_x)
                aux_sub = None
                if self.moe is not None:
                    moe_sub, aux_sub = self.moe(sub_x, task_id=task_id)
                    out_sub = out_sub + moe_sub
                return out_sub, aux_sub

            ffn_out, aux_info = self.mod_router.forward_sublayer(normed, _ffn_moe_sublayer)
            x = x + ffn_out
            return x, aux_info

        ffn_out = self.mlp(normed)
        aux_info = None
        if self.moe is not None and not bypass_moe:
            moe_out, aux_info = self.moe(normed, task_id=task_id)
            if domain_mask is not None:
                moe_out = torch.where(domain_mask.unsqueeze(-1), moe_out, torch.zeros_like(moe_out))
            ffn_out = ffn_out + moe_out
        x = x + ffn_out
        return x, aux_info


class MiniGPT(nn.Module):
    """Decoder-only Transformer language model with adaptive expert, depth routing, and flexible position encodings."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        if config.pos_encoding == "absolute":
            self.wpe: nn.Embedding | None = nn.Embedding(config.context_length, config.d_model)
        else:
            self.wpe = None
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.wte.weight

        self.moe_head: LowRankOutputAdapter | None = None
        if config.adaptive_enabled:
            self.moe_head = LowRankOutputAdapter(config.d_model, config.d_expert, config.vocab_size)

        # M38 Structural Evolution: optional reversible inter-block skip connections
        self.skip_connections: nn.ModuleDict = nn.ModuleDict()

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layers) ** 0.5)
        # Ensure moe_head output projection stays zero-initialized after self.apply
        if self.moe_head is not None:
            nn.init.zeros_(self.moe_head.w_down.weight)

        self.last_aux_info: dict[str, Any] = {}
        self._embedding_grad_mask: Tensor | None = None
        self._wte_hook_registered: bool = False

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

    def freeze_backbone(
        self,
        freeze_existing_experts: bool = False,
        trainable_token_ids: list[int] | None = None,
    ) -> None:
        """Freeze shared dense backbone parameters."""
        if trainable_token_ids is not None:
            self.config.trainable_token_ids = list(trainable_token_ids)
        active_token_ids = self.config.trainable_token_ids

        for pn, p in self.named_parameters():
            if "blocks" in pn:
                if "moe" in pn or "mod_router" in pn:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            elif "moe_head" in pn:
                p.requires_grad = True
            else:
                p.requires_grad = False

        if freeze_existing_experts:
            for block in self.blocks:
                if block.moe is not None:
                    block.moe.freeze_expert_subset()

        if active_token_ids:
            mask = torch.zeros(self.config.vocab_size, 1, dtype=torch.float32)
            for tid in active_token_ids:
                if 0 <= tid < self.config.vocab_size:
                    mask[tid, 0] = 1.0
            self._embedding_grad_mask = mask

            def _mask_grad(grad: Tensor) -> Tensor:
                if self._embedding_grad_mask is None:
                    return grad
                return grad * self._embedding_grad_mask.to(grad.device, dtype=grad.dtype)

            self.wte.weight.requires_grad = True
            if not self._wte_hook_registered:
                self.wte.weight.register_hook(_mask_grad)
                self._wte_hook_registered = True

    def configure_optimizers(
        self,
        learning_rate: float,
        weight_decay: float,
    ) -> AdamW:
        """Create AdamW with decoupled weight decay applied only to >=2D weight tensors."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [
            p
            for n, p in param_dict.items()
            if p.dim() >= 2
            and not (self._embedding_grad_mask is not None and ("wte" in n or "lm_head" in n))
        ]
        nodecay_params = [
            p
            for n, p in param_dict.items()
            if p.dim() < 2
            or (self._embedding_grad_mask is not None and ("wte" in n or "lm_head" in n))
        ]
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
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass through the Transformer with M6 autonomous domain gating and M38 structural connections."""
        b, t = idx.size()
        if self.config.pos_encoding == "absolute" and t > self.config.context_length:
            raise ValueError(
                f"Cannot forward sequence of length {t}, block size is {self.config.context_length}"
            )

        device = idx.device
        tok_emb = self.wte(idx)

        if self.wpe is not None:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.wpe(pos)
            x = self.drop(tok_emb + pos_emb)
        else:
            x = self.drop(tok_emb)

        # M6/M7 Autonomous Causal Domain Gating
        domain_token_mask: Tensor | None = None
        effective_bypass_moe = bypass_moe
        trigger_ids = (
            self.config.domain_trigger_token_ids
            if self.config.domain_trigger_token_ids is not None
            else self.config.trainable_token_ids
        )
        if not bypass_moe and self.config.adaptive_enabled and trigger_ids:
            specialist_ids = torch.tensor(trigger_ids, dtype=idx.dtype, device=device)
            is_spec = torch.isin(idx, specialist_ids)
            domain_token_mask = is_spec.cumsum(dim=-1) > 0
            if not domain_token_mask.any():
                effective_bypass_moe = True

        total_balance_loss = torch.tensor(0.0, device=device)
        total_z_loss = torch.tensor(0.0, device=device)
        total_mod_loss = torch.tensor(0.0, device=device)
        entropies: list[float] = []
        utilizations: list[Tensor] = []
        mod_fractions: list[float] = []

        # M38 Structural Evolution: inter-block skip connection intercepts
        saved_block_outputs: dict[int, Tensor] = {}
        has_skips = hasattr(self, "skip_connections") and len(self.skip_connections) > 0

        for idx_layer, block in enumerate(self.blocks):
            x, aux = block(
                x,
                task_id=task_id,
                bypass_moe=effective_bypass_moe,
                domain_mask=domain_token_mask,
                **kwargs,
            )
            if has_skips:
                saved_block_outputs[idx_layer] = x
                for skip_name, skip_mod in self.skip_connections.items():
                    if (
                        getattr(skip_mod, "target_layer", None) == idx_layer
                        and getattr(skip_mod, "source_layer", None) in saved_block_outputs
                    ):
                        x = skip_mod(saved_block_outputs[skip_mod.source_layer], x)

            if aux is not None:
                if "balance_loss" in aux:
                    total_balance_loss = total_balance_loss + aux["balance_loss"]
                if "z_loss" in aux:
                    total_z_loss = total_z_loss + aux["z_loss"]
                if "router_entropy" in aux:
                    entropies.append(aux["router_entropy"])
                if "expert_utilization" in aux:
                    utilizations.append(aux["expert_utilization"])
                if "mod_loss" in aux:
                    total_mod_loss = total_mod_loss + aux["mod_loss"]
                if "mod_routed_fraction" in aux:
                    mod_fractions.append(aux["mod_routed_fraction"])

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if self.moe_head is not None and not effective_bypass_moe:
            moe_logits = self.moe_head(x)
            if domain_token_mask is not None:
                moe_logits = torch.where(
                    domain_token_mask.unsqueeze(-1), moe_logits, torch.zeros_like(moe_logits)
                )
            logits = logits + moe_logits

        self.last_aux_info = {
            "balance_loss": total_balance_loss.item(),
            "z_loss": total_z_loss.item(),
            "mean_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
            "expert_utilization": utilizations[-1].tolist() if utilizations else [],
            "mod_loss": total_mod_loss.item(),
            "mean_mod_fraction": sum(mod_fractions) / len(mod_fractions) if mod_fractions else 1.0,
        }

        loss = None
        if targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
                ignore_index=-100,
            )
            if self.config.adaptive_enabled:
                loss = (
                    loss
                    + self.config.aux_loss_weight * total_balance_loss
                    + self.config.z_loss_weight * total_z_loss
                )
            if self.config.mod_enabled:
                loss = loss + total_mod_loss

        return logits, loss
