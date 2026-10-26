"""Milestone M103 Text-Enabled Evolvable Student.

Integrates:
1. Dense MiniGPT Transformer decoder backbone with causal self-attention.
2. Dynamic Evolvable Backbone with function-preserving layer insertion.
3. Modular residual adapters per rule domain.
4. ByteTextTokenizerM103 for open-ended natural-language input/output.
5. Causal autoregressive next-token loss with prompt masking.
6. Bit-exact checkpointing and state restoration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.dynamic_transformer_evolution.dynamic_transformer_substrate import (
    DynamicEvolvableBackbone,
)
from adaptive_mini_gpt.integrated_autonomous_student.structural_evolution_engine import (
    CapacityTierM100,
)
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.natural_language_learning_m103.curriculum_m103 import (
    QAItemM103,
    RuleGoalM103,
    TextRuleDomainM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)


class TextStudentM103(nn.Module):
    """Text-input and text-output evolvable persistent student."""

    def __init__(
        self,
        tier: CapacityTierM100 = CapacityTierM100.TIER_20K,
        device: torch.device | None = None,
        enable_persistent_growth: bool = True,
        prune_orphaned_lm_head: bool = False,
    ) -> None:
        super().__init__()
        self.tier = tier
        self._device = device or torch.device("cpu")
        self.enable_persistent_growth = enable_persistent_growth
        self.step_counter = 0

        # 1. Base 4-layer MiniGPT Transformer decoder
        cfg = ModelConfig(
            vocab_size=1024,
            context_length=256,
            d_model=128,
            n_heads=4,
            n_layers=4,
            d_ff=512,
            pos_encoding="absolute",
            dropout=0.0,
            bias=True,
        )
        self.base_model = MiniGPT(cfg).to(self.device)

        # 2. Evolvable backbone substrate with dynamic layer skipping
        self.evolvable_backbone = DynamicEvolvableBackbone(base_model=self.base_model).to(self.device)

        # 3. Modular residual adapters for text rule domains (512 params each, rank-2)
        self.domains = [d.value for d in TextRuleDomainM103]
        self.adapter_modules = nn.ModuleDict({
            f"adapter_{d}": nn.Sequential(
                nn.Linear(128, 2, bias=False),
                nn.GELU(),
                nn.Linear(2, 128, bias=False),
            ).to(self.device)
            for d in self.domains
        })
        for adapter in self.adapter_modules.values():
            nn.init.zeros_(adapter[2].weight)

        # 4. Autoregressive Language Model Head (Linear 128 -> 1024)
        if prune_orphaned_lm_head and hasattr(self.base_model, "lm_head") and self.base_model.lm_head is not None:
            self.base_model.lm_head.weight = self.base_model.wte.weight
            self.lm_head = self.base_model.lm_head
        else:
            self.lm_head = nn.Linear(128, 1024, bias=False).to(self.device)
            self.lm_head.weight = self.base_model.wte.weight  # Tied word embeddings

        # 5. Text Tokenizer
        self.tokenizer = ByteTextTokenizerM103(vocab_size=1024)

        # 6. Function-preserving persistent growth block (M102/M102.1/M102.2)
        self.growth_block_active = False
        if self.enable_persistent_growth:
            self.evolvable_backbone.insert_block_identity(insert_idx=len(self.base_model.blocks))
            self.growth_block_active = True

        # Rehearsal buffer for continuous learning
        self.replay_buffer: dict[str, list[QAItemM103]] = {}

    @property
    def layer_routers(self) -> nn.ModuleList:
        """Dynamic layer skipping routers hosted directly on the evolvable backbone."""
        return self.evolvable_backbone.layer_routers

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return getattr(self, "_device", torch.device("cpu"))

    @device.setter
    def device(self, val: torch.device) -> None:
        self._device = val

    def forward_lm(
        self,
        tokens: torch.Tensor,
        active_domain: str | None = None,
        allow_skipping: bool = False,
    ) -> torch.Tensor:
        """Autoregressive forward pass returning next-token logits (B, T, V)."""
        b, t = tokens.size()
        tok_emb = self.base_model.wte(tokens)
        pos = torch.arange(0, t, dtype=torch.long, device=tokens.device)
        x = self.base_model.drop(tok_emb + self.base_model.wpe(pos))

        for i, block in enumerate(self.base_model.blocks):
            if allow_skipping and i < len(self.layer_routers):
                exec_mask, _ = self.layer_routers[i](x)
                if not exec_mask.any():
                    continue
            x, _ = block(x)

        x = self.base_model.ln_f(x)

        # Apply domain adapter if specified
        if active_domain and f"adapter_{active_domain}" in self.adapter_modules:
            x = x + self.adapter_modules[f"adapter_{active_domain}"](x)

        # Output logits
        return self.lm_head(x)

    # Alias forward to forward_lm for standard nn.Module call syntax
    forward = forward_lm

    def generate(
        self,
        prompt_tokens: list[int] | torch.Tensor,
        max_new_tokens: int = 24,
        active_domain: str | None = None,
        stop_token: int | None = ByteTextTokenizerM103.EOS_ID,
    ) -> list[int]:
        """Greedy autoregressive token generation."""
        self.eval()
        if isinstance(prompt_tokens, list):
            cur_tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        else:
            cur_tokens = prompt_tokens.to(self.device)
            if cur_tokens.dim() == 1:
                cur_tokens = cur_tokens.unsqueeze(0)

        ctx_len = self.base_model.config.context_length
        if cur_tokens.size(1) > ctx_len:
            cur_tokens = cur_tokens[:, -ctx_len:]

        generated: list[int] = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if cur_tokens.size(1) >= ctx_len:
                    break
                logits = self.forward_lm(cur_tokens, active_domain=active_domain)
                next_token_logits = logits[0, -1, :]
                next_token = int(next_token_logits.argmax().item())

                if stop_token is not None and next_token == stop_token:
                    break
                generated.append(next_token)
                next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                cur_tokens = torch.cat([cur_tokens, next_tensor], dim=1)

        return generated

    def compute_qa_loss(
        self,
        items: list[QAItemM103],
        active_domain: str | None = None,
    ) -> torch.Tensor:
        """Compute causal autoregressive cross-entropy loss with prompt masking."""
        if not items:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        token_lists = [it.full_tokens for it in items]
        batch_tokens, mask = self.tokenizer.pad_batch(token_lists, device=self.device)

        logits = self.forward_lm(batch_tokens, active_domain=active_domain)
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch_tokens[:, 1:].contiguous().clone()

        # Mask prompt tokens with -100 so loss is computed solely on answer tokens
        for i, it in enumerate(items):
            p_len = it.prompt_len - 1  # Accounting for shift
            if p_len > 0:
                shift_labels[i, :p_len] = -100

        # Mask padding
        shift_labels[~mask[:, 1:]] = -100

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    def evaluate_exact_match(
        self,
        items: list[QAItemM103],
        active_domain: str | None = None,
        with_explanation: bool = True,
    ) -> float:
        """Evaluate exact string match on answer completions."""
        if not items:
            return 0.0

        matches = 0
        for it in items:
            if with_explanation and it.prompt_len > 0:
                prompt_toks = it.full_tokens[:it.prompt_len]
            else:
                expl = it.explanation if with_explanation else None
                prompt_toks, _ = self.tokenizer.format_qa_sequence(
                    question=it.question,
                    answer="",
                    explanation=expl,
                )
                if prompt_toks and prompt_toks[-1] == self.tokenizer.EOS_ID:
                    prompt_toks = prompt_toks[:-1]

            gen_ids = self.generate(
                prompt_tokens=prompt_toks,
                max_new_tokens=24,
                active_domain=active_domain,
            )
            gen_text = self.tokenizer.decode(gen_ids).strip().lower()
            target_text = it.answer.strip().lower()

            if gen_text == target_text:
                matches += 1

        return round((matches / len(items)) * 100.0, 2)

    def adapt_rule_neural(
        self,
        goal: RuleGoalM103,
        max_steps: int = 25,
        early_stop_acc: float = 90.0,
        lr: float = 0.01,
    ) -> dict[str, Any]:
        """Adapt neural weights to internalize the rule."""
        self.step_counter += 1
        dom = goal.domain.value

        # Freeze backbone, unfreeze target domain adapter
        for p in self.parameters():
            p.requires_grad = False

        adapter_key = f"adapter_{dom}"
        if adapter_key in self.adapter_modules:
            for p in self.adapter_modules[adapter_key].parameters():
                p.requires_grad = True

        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=1e-4,
        )

        steps_taken = 0
        final_val_acc = 0.0

        for s in range(1, max_steps + 1):
            steps_taken = s
            self.train()
            opt.zero_grad()

            loss_task = self.compute_qa_loss(goal.train_items, active_domain=dom)

            # Rehearsal loss across replay buffer
            loss_reh = torch.tensor(0.0, device=self.device)
            has_reh = False
            for past_d, past_items in self.replay_buffer.items():
                if past_d != dom:
                    l_r = self.compute_qa_loss(past_items[:4], active_domain=past_d)
                    loss_reh = loss_reh + l_r
                    has_reh = True

            total_loss = loss_task
            if has_reh:
                total_loss = 0.7 * loss_task + 0.3 * (loss_reh / max(1, len(self.replay_buffer)))

            total_loss.backward()
            opt.step()

            # Validation check
            self.eval()
            with torch.no_grad():
                val_em = self.evaluate_exact_match(goal.val_items, active_domain=dom, with_explanation=True)
                final_val_acc = val_em
                if val_em >= early_stop_acc:
                    break

        # Register exemplars into replay buffer
        self.replay_buffer[dom] = goal.train_items[:4]

        # Evaluate on test set
        test_em = self.evaluate_exact_match(goal.test_items, active_domain=dom, with_explanation=True)
        novel_em = self.evaluate_exact_match(goal.novel_items, active_domain=dom, with_explanation=True)

        return {
            "steps_taken": steps_taken,
            "validation_accuracy": final_val_acc,
            "held_out_accuracy": test_em,
            "novel_generalization_accuracy": novel_em,
        }

    def save_checkpoint(self, path: Path | str) -> None:
        """Save atomic state dictionary."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step_counter": self.step_counter,
            "state_dict": self.state_dict(),
            "growth_block_active": self.growth_block_active,
            "tier": self.tier.value,
        }, p)

    @classmethod
    def load_checkpoint(cls, path: Path | str, device: torch.device | None = None) -> TextStudentM103:
        """Load checkpoint deterministically."""
        p = Path(path)
        ckpt = torch.save if False else torch.load(p, map_location=device or torch.device("cpu"), weights_only=False)
        growth = ckpt.get("growth_block_active", True)
        tier_str = ckpt.get("tier", CapacityTierM100.TIER_20K.value)
        tier = CapacityTierM100(tier_str)

        student = cls(tier=tier, device=device, enable_persistent_growth=growth)
        student.load_state_dict(ckpt["state_dict"])
        student.step_counter = ckpt.get("step_counter", 0)
        return student

    def verify_bit_exact_recovery(
        self,
        reloaded_student: TextStudentM103,
        sample_tokens: list[int] | torch.Tensor,
    ) -> dict[str, Any]:
        """Verify 100% bit-exact parameter and forward pass match."""
        # 1. State dict exact match
        param_match = True
        max_param_diff = 0.0
        s1 = self.state_dict()
        s2 = reloaded_student.state_dict()
        for k in s1:
            diff = float((s1[k] - s2[k]).abs().max().item())
            if diff > max_param_diff:
                max_param_diff = diff
            if not torch.equal(s1[k], s2[k]):
                param_match = False

        # 2. Forward pass continuous logit match
        if isinstance(sample_tokens, list):
            t_tok = torch.tensor([sample_tokens], dtype=torch.long, device=self.device)
        else:
            t_tok = sample_tokens.to(self.device)
            if t_tok.dim() == 1:
                t_tok = t_tok.unsqueeze(0)

        self.eval()
        reloaded_student.eval()
        with torch.no_grad():
            l1 = self.forward_lm(t_tok)
            l2 = reloaded_student.forward_lm(t_tok)
            logit_diff = float((l1 - l2).abs().max().item())
            argmax_match = bool(torch.equal(l1.argmax(dim=-1), l2.argmax(dim=-1)))

        certified = param_match and (logit_diff < 1e-6) and argmax_match
        return {
            "bit_exact_certified": certified,
            "parameter_match": param_match,
            "max_parameter_difference": max_param_diff,
            "max_logit_difference": logit_diff,
            "argmax_predictions_match": argmax_match,
        }
