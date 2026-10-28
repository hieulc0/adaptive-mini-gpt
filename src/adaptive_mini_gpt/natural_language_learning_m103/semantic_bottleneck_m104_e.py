"""Milestone M104-E Grounded Semantic Bottleneck and Integrated Student.

Key Architectural Features:
1. Entity-Conditioned Cross-Attention (ECA):
   - Uses entity/question representation as query to cross-attend over full sequence hidden states.
   - Attends over the supplied prompt; rule/entity separation is not guaranteed.
2. Low-Rank Semantic Bottleneck (~56k parameters):
   - Exactly 56,074 parameters (certified 56.0% reduction vs M104-B's 127,418 params).
3. Selective Plasticity with Grounded Embeddings:
   - Selectively trains Word Token Embeddings (wte), top Transformer block (Block 4), final LayerNorm (ln_f),
     and the semantic bottleneck.
   - Freezes lower Transformer blocks (Blocks 0-3) to prevent representation drift while allowing
     lexical token grounding.
4. Flexible Extraction Modes:
   - Supports "question_only", "full_context", and "entity_conditioned_attention" at inference.
   - Strictly text-only inference: zero privileged oracle attributes or simulator state.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from adaptive_mini_gpt.natural_language_learning_m103.grounded_extractor_m104_e import (
    EntityConditionedAttentionExtractorM104_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)


class SemanticBottleneckM104_E(nn.Module):
    """Calibrated low-rank semantic concept bottleneck mapping Transformer states into grounded causal states."""

    def __init__(
        self,
        d_model: int = 128,
        bottleneck_dim: int = 32,
        action_dim: int = 64,
        num_actions: int = 8,
        num_states: int = 16,
        num_tags: int = 26,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_tags = num_tags

        # 1. Low-rank concept projection from entity representation
        self.concept_proj = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # 2. Auxiliary Attribute Head (supervises concept grounding during training)
        self.attribute_head = nn.Linear(d_model, num_tags)

        # 3. Action embedding
        self.action_emb = nn.Embedding(num_actions, action_dim)

        # 4. Entity/action Transition Dynamics Network (global_repr is currently unused):
        # Exactly 56,074 params (Linear 192 -> 128)
        self.transition_net = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.post_norm = nn.LayerNorm(d_model)

        # 5. State Classification Head
        self.state_head = nn.Linear(d_model, num_states)

    def forward(
        self,
        entity_repr: torch.Tensor,
        global_repr: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            entity_repr: (B, d_model) pooled or attended entity representation.
            global_repr: (B, d_model) pooled whole-sequence causal context.
            action_ids: (B,) action indices.

        Returns:
            state_logits: (B, num_states)
            aux_tag_logits: (B, num_tags)
            z_concept: (B, d_model)
            z_next: (B, d_model)
        """
        z_concept = self.concept_proj(entity_repr)
        aux_tag_logits = self.attribute_head(z_concept)

        a_emb = self.action_emb(action_ids)
        comb = torch.cat([z_concept, a_emb], dim=-1)
        delta_z = self.transition_net(comb)
        z_next = self.post_norm(z_concept + delta_z)

        state_logits = self.state_head(z_next)
        return state_logits, aux_tag_logits, z_concept, z_next


class IntegratedStudentM104_E(nn.Module):
    """Integrates TextStudentM103 with SemanticBottleneckM104_E and Entity-Conditioned Cross-Attention."""

    def __init__(
        self,
        student: TextStudentM103,
        num_actions: int = 8,
        num_states: int = 16,
        num_tags: int = 26,
        mode: str = "selective_plasticity",  # "selective_plasticity", "selective_top2_blocks", "frozen_backbone", "trainable_backbone", "no_adapter"
        default_strategy: str = "entity_conditioned_attention",  # "question_only", "full_context", "entity_conditioned_attention"
        pooling_mode: str = "mean",  # "mean", "span_relative_max_mean", "max_mean"
    ) -> None:
        super().__init__()
        self.student = student
        self.mode = mode
        self.default_strategy = default_strategy
        self.pooling_mode = pooling_mode
        d_model = student.base_model.config.d_model

        if mode in (
            "selective_plasticity",
            "selective_top2_blocks",
            "frozen_backbone",
            "trainable_backbone",
        ):
            self.bottleneck = SemanticBottleneckM104_E(
                d_model=d_model,
                bottleneck_dim=32,
                action_dim=64,
                num_actions=num_actions,
                num_states=num_states,
                num_tags=num_tags,
            ).to(student.device)

            self.eca_extractor = EntityConditionedAttentionExtractorM104_E(
                d_model=d_model,
                d_attn=32,
                num_heads=2,
                dropout=0.1,
            ).to(student.device)

            self.linear_head = None
            self.action_emb = None
        else:
            self.bottleneck = None
            self.eca_extractor = None
            self.action_emb = nn.Embedding(num_actions, 64).to(student.device)
            self.linear_head = nn.Sequential(
                nn.Linear(d_model + 64, d_model),
                nn.GELU(),
                nn.Linear(d_model, num_states),
            ).to(student.device)

        self._configure_plasticity()

    def _configure_plasticity(self) -> None:
        """Configures parameter plasticity across backbone and bottleneck."""
        if self.mode == "frozen_backbone":
            for p in self.student.parameters():
                p.requires_grad = False
            if self.bottleneck is not None:
                for p in self.bottleneck.parameters():
                    p.requires_grad = True
            if self.eca_extractor is not None:
                for p in self.eca_extractor.parameters():
                    p.requires_grad = True

        elif self.mode == "selective_plasticity":
            # Freeze lower blocks (Blocks 0-3) and wpe
            for p in self.student.parameters():
                p.requires_grad = False

            # Selectively unfreeze Word Token Embeddings (wte) to allow lexical grounding
            if hasattr(self.student.base_model, "wte") and self.student.base_model.wte is not None:
                for p in self.student.base_model.wte.parameters():
                    p.requires_grad = True

            # Unfreeze top Transformer block (Block 4) and final LayerNorm
            if (
                hasattr(self.student.base_model, "blocks")
                and len(self.student.base_model.blocks) > 0
            ):
                top_block = self.student.base_model.blocks[-1]
                for p in top_block.parameters():
                    p.requires_grad = True

            if (
                hasattr(self.student.base_model, "ln_f")
                and self.student.base_model.ln_f is not None
            ):
                for p in self.student.base_model.ln_f.parameters():
                    p.requires_grad = True

            if self.bottleneck is not None:
                for p in self.bottleneck.parameters():
                    p.requires_grad = True
            if self.eca_extractor is not None:
                for p in self.eca_extractor.parameters():
                    p.requires_grad = True

        elif self.mode == "selective_top2_blocks":
            # Freeze lower blocks (Blocks 0-2) and wpe
            for p in self.student.parameters():
                p.requires_grad = False

            # Selectively unfreeze Word Token Embeddings (wte) to allow lexical grounding
            if hasattr(self.student.base_model, "wte") and self.student.base_model.wte is not None:
                for p in self.student.base_model.wte.parameters():
                    p.requires_grad = True

            # Unfreeze top 2 Transformer blocks (Block 3 and Block 4) and final LayerNorm
            if (
                hasattr(self.student.base_model, "blocks")
                and len(self.student.base_model.blocks) > 1
            ):
                for top_block in self.student.base_model.blocks[-2:]:
                    for p in top_block.parameters():
                        p.requires_grad = True

            if (
                hasattr(self.student.base_model, "ln_f")
                and self.student.base_model.ln_f is not None
            ):
                for p in self.student.base_model.ln_f.parameters():
                    p.requires_grad = True

            if self.bottleneck is not None:
                for p in self.bottleneck.parameters():
                    p.requires_grad = True
            if self.eca_extractor is not None:
                for p in self.eca_extractor.parameters():
                    p.requires_grad = True

        elif self.mode == "trainable_backbone":
            for p in self.student.parameters():
                p.requires_grad = True
            if self.bottleneck is not None:
                for p in self.bottleneck.parameters():
                    p.requires_grad = True
            if self.eca_extractor is not None:
                for p in self.eca_extractor.parameters():
                    p.requires_grad = True

        elif self.mode == "no_adapter":
            # Trains backbone with linear head
            for p in self.student.parameters():
                p.requires_grad = True
            if self.action_emb is not None:
                for p in self.action_emb.parameters():
                    p.requires_grad = True
            if self.linear_head is not None:
                for p in self.linear_head.parameters():
                    p.requires_grad = True

    @property
    def device(self) -> torch.device:
        return self.student.device

    def extract_sequence_hidden_states(self, tokens: torch.Tensor) -> torch.Tensor:
        """Runs Transformer and returns hidden sequence H: (B, L, d_model)."""
        return self.student.evolvable_backbone.hidden_states(tokens)

    def pool_representations(
        self,
        h_seq: torch.Tensor,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts entity-targeted representation and whole-sequence global causal representation."""
        B, L, D = h_seq.size()
        global_mask = tokens != ByteTextTokenizerM103.PAD_ID
        global_mask_f = global_mask.float().unsqueeze(-1)
        global_repr = (h_seq * global_mask_f).sum(dim=1) / global_mask_f.sum(dim=1).clamp(min=1.0)

        entity_mask = torch.zeros((B, L), dtype=torch.bool, device=tokens.device)
        q_id = ByteTextTokenizerM103.QUESTION_ID
        a_id = ByteTextTokenizerM103.ANSWER_ID

        for b in range(B):
            row = tokens[b]
            q_locs = (row == q_id).nonzero(as_tuple=True)[0]
            a_locs = (row == a_id).nonzero(as_tuple=True)[0]

            if len(q_locs) > 0 and len(a_locs) > 0 and a_locs[0] > q_locs[0]:
                entity_mask[b, q_locs[0] : a_locs[0]] = True
            elif len(q_locs) > 0:
                entity_mask[b, q_locs[0] :] = True
            else:
                half = L // 2
                entity_mask[b, half:] = True

        entity_mask_f = (entity_mask & global_mask).float().unsqueeze(-1)
        mean_repr = (h_seq * entity_mask_f).sum(dim=1) / entity_mask_f.sum(dim=1).clamp(min=1.0)

        if getattr(self, "pooling_mode", "mean") in ("max_mean", "span_relative_max_mean"):
            h_masked = h_seq.masked_fill(~(entity_mask & global_mask).unsqueeze(-1), -1e4)
            max_repr = h_masked.max(dim=1)[0]
            entity_repr = 0.5 * (mean_repr + max_repr)
        else:
            entity_repr = mean_repr

        return entity_repr, global_repr

    def forward_with_strategy(
        self,
        prompt_tokens: torch.Tensor,
        action_ids: torch.Tensor,
        strategy: str = "entity_conditioned_attention",
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Forward pass supporting configurable extraction strategies."""
        h_seq = self.extract_sequence_hidden_states(prompt_tokens)  # (B, L, d_model)
        q_entity_repr, global_repr = self.pool_representations(h_seq, prompt_tokens)

        if self.bottleneck is None:
            assert self.action_emb is not None and self.linear_head is not None
            a_emb = self.action_emb(action_ids)
            comb = torch.cat([global_repr, a_emb], dim=-1)
            state_logits = self.linear_head(comb)
            return state_logits, None, global_repr, global_repr

        # Determine entity representation based on requested strategy
        if strategy == "full_context":
            entity_input = global_repr
        elif strategy == "question_only":
            entity_input = q_entity_repr
        else:  # entity_conditioned_attention (ECA)
            assert self.eca_extractor is not None
            pad_mask = prompt_tokens == ByteTextTokenizerM103.PAD_ID
            q_query = q_entity_repr.unsqueeze(1)  # (B, 1, d_model)
            attended_repr = self.eca_extractor(
                query_repr=q_query,
                seq_hidden_states=h_seq,
                key_padding_mask=pad_mask,
            )
            entity_input = attended_repr

        # Transition dynamics
        state_logits, tag_logits, z_concept, z_next = self.bottleneck(
            entity_repr=entity_input,
            global_repr=global_repr,
            action_ids=action_ids,
        )

        return state_logits, tag_logits, z_concept, z_next

    def forward(
        self,
        prompt_tokens: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Default forward pass using configured default_strategy."""
        return self.forward_with_strategy(prompt_tokens, action_ids, strategy=self.default_strategy)
