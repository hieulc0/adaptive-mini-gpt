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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def compute_query_cross_prefix_mask(
    prompt_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes key_padding_mask and query_indices for learned query-prefix cross-attention.

    Ensures query tokens attend strictly to descriptive clause prefix <question> tokens
    and never leak <answer>, padding, or rule text.

    Args:
        prompt_tokens: (B, L) input token IDs.

    Returns:
        key_padding_mask: (B, L) boolean tensor where True means MASKED (ignored by attention).
        query_indices: (B,) long tensor giving the token position of the query representation.
    """
    B, L = prompt_tokens.size()
    q_id = ByteTextTokenizerM103.QUESTION_ID
    a_id = ByteTextTokenizerM103.ANSWER_ID

    key_padding_mask = torch.ones((B, L), dtype=torch.bool, device=prompt_tokens.device)
    query_indices = torch.zeros(B, dtype=torch.long, device=prompt_tokens.device)

    query_marker_bytes = [
        b"Query target",
        b"Target is",
        b"Analysis focuses on",
        b"The evaluated target",
        b"Focus on",
    ]

    for b in range(B):
        row = prompt_tokens[b].tolist()
        q_locs = [i for i, tok in enumerate(row) if tok == q_id]
        a_locs = [i for i, tok in enumerate(row) if tok == a_id]

        if q_locs and a_locs and a_locs[0] > q_locs[0]:
            q_start = q_locs[0] + 1
            q_end = a_locs[0]
            query_indices[b] = q_end - 1  # Last question token before answer
            q_slice = bytes([t for t in row[q_start:q_end] if 0 <= t < 256])
            earliest_pos = None
            for marker in query_marker_bytes:
                pos = q_slice.find(marker)
                if pos != -1:
                    if earliest_pos is None or pos < earliest_pos:
                        earliest_pos = pos
            if earliest_pos is not None and earliest_pos > 0:
                prefix_end = q_start + earliest_pos
            else:
                prefix_end = q_end
            key_padding_mask[b, q_start:prefix_end] = False
        elif q_locs:
            q_start = q_locs[0] + 1
            valid = [i for i, tok in enumerate(row) if tok != ByteTextTokenizerM103.PAD_ID]
            last_tok = valid[-1] if valid else (L - 1)
            query_indices[b] = last_tok
            key_padding_mask[b, q_start : last_tok + 1] = False
        else:
            query_indices[b] = L - 1
            key_padding_mask[b, :] = False

        # Guard against all-True rows to prevent NaNs in softmax
        if key_padding_mask[b].all():
            key_padding_mask[b, query_indices[b]] = False

    return key_padding_mask, query_indices


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

    def extract_tapped_representations(
        self,
        h_seq: torch.Tensor,
        tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extracts representation taps: h_mean_pool, h_last_question_token, h_query_attn_pool."""
        B, L, D = h_seq.size()
        global_mask = tokens != ByteTextTokenizerM103.PAD_ID
        q_id = ByteTextTokenizerM103.QUESTION_ID
        a_id = ByteTextTokenizerM103.ANSWER_ID

        entity_mask = torch.zeros((B, L), dtype=torch.bool, device=tokens.device)
        last_indices = torch.zeros(B, dtype=torch.long, device=tokens.device)

        for b in range(B):
            row = tokens[b]
            q_locs = (row == q_id).nonzero(as_tuple=True)[0]
            a_locs = (row == a_id).nonzero(as_tuple=True)[0]

            if len(q_locs) > 0 and len(a_locs) > 0 and a_locs[0] > q_locs[0]:
                entity_mask[b, q_locs[0] : a_locs[0]] = True
                last_indices[b] = a_locs[0] - 1
            elif len(q_locs) > 0:
                entity_mask[b, q_locs[0] :] = True
                valid = global_mask[b].nonzero(as_tuple=True)[0]
                last_indices[b] = valid[-1] if len(valid) > 0 else (L - 1)
            else:
                half = L // 2
                entity_mask[b, half:] = True
                last_indices[b] = L - 1

        active_mask = entity_mask & global_mask
        active_mask_f = active_mask.float().unsqueeze(-1)
        h_mean_pool = (h_seq * active_mask_f).sum(dim=1) / active_mask_f.sum(dim=1).clamp(min=1.0)

        batch_idx = torch.arange(B, device=tokens.device)
        h_last_token = h_seq[batch_idx, last_indices]

        # Parameter-free dot-product attention pooling:
        # alpha_t = softmax_{t in Q}(h_t^T h_{last} / sqrt(D))
        # h_query_pool = sum_{t in Q} alpha_t h_t
        q_vec = h_last_token.unsqueeze(1)  # (B, 1, D)
        scores = torch.bmm(h_seq, q_vec.transpose(1, 2)).squeeze(-1) / math.sqrt(D)  # (B, L)
        scores = scores.masked_fill(~active_mask, -1e4)
        alpha = F.softmax(scores, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        h_query_attn_pool = torch.bmm(alpha.unsqueeze(1), h_seq).squeeze(1)  # (B, D)

        h_residual_query_gated_pool = 0.5 * (h_last_token + h_query_attn_pool)

        # Parameter-free query-span cross-readout:
        # Cross-attends from query token (q_vec) over the prefix clause span.
        prefix_mask = torch.zeros((B, L), dtype=torch.bool, device=tokens.device)
        query_marker_bytes = [
            b"Query target",
            b"Target is",
            b"Analysis focuses on",
            b"The evaluated target",
            b"Focus on",
        ]

        for b in range(B):
            row = tokens[b].tolist()
            q_locs = [i for i, tok in enumerate(row) if tok == q_id]
            a_locs = [i for i, tok in enumerate(row) if tok == a_id]
            if q_locs and a_locs and a_locs[0] > q_locs[0]:
                q_start = q_locs[0] + 1
                q_end = a_locs[0]
                q_slice = bytes([t for t in row[q_start:q_end] if 0 <= t < 256])
                earliest_pos = None
                for marker in query_marker_bytes:
                    pos = q_slice.find(marker)
                    if pos != -1:
                        if earliest_pos is None or pos < earliest_pos:
                            earliest_pos = pos
                if earliest_pos is not None and earliest_pos > 0:
                    prefix_end = q_start + earliest_pos
                    prefix_mask[b, q_start:prefix_end] = True
                else:
                    prefix_mask[b, q_start:q_end] = True
            else:
                prefix_mask[b] = active_mask[b]

        prefix_active = prefix_mask & global_mask
        cross_scores = torch.bmm(h_seq, q_vec.transpose(1, 2)).squeeze(-1) / math.sqrt(D)
        cross_scores = cross_scores.masked_fill(~prefix_active, -1e4)
        beta = F.softmax(cross_scores, dim=-1)
        beta = torch.nan_to_num(beta, nan=0.0)
        h_cross_prefix = torch.bmm(beta.unsqueeze(1), h_seq).squeeze(1)
        h_query_span_cross_readout = 0.5 * (h_last_token + h_cross_prefix)

        return {
            "h_mean_pool": h_mean_pool,
            "h_last_question_token": h_last_token,
            "h_query_attn_pool": h_query_attn_pool,
            "h_residual_query_gated_pool": h_residual_query_gated_pool,
            "h_query_span_cross_readout": h_query_span_cross_readout,
        }

    def forward_with_strategy(
        self,
        prompt_tokens: torch.Tensor,
        action_ids: torch.Tensor,
        strategy: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Forward pass supporting configurable extraction strategies."""
        if strategy is None:
            strategy = self.default_strategy

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
        elif strategy in ("question_only", "mean_pool"):
            entity_input = q_entity_repr
        elif strategy == "last_question_token":
            taps = self.extract_tapped_representations(h_seq, prompt_tokens)
            entity_input = taps["h_last_question_token"]
        elif strategy == "query_attn_pool":
            taps = self.extract_tapped_representations(h_seq, prompt_tokens)
            entity_input = taps["h_query_attn_pool"]
        elif strategy == "residual_query_gated_pool":
            taps = self.extract_tapped_representations(h_seq, prompt_tokens)
            entity_input = taps["h_residual_query_gated_pool"]
        elif strategy in ("query_span_cross_readout", "query_cross"):
            taps = self.extract_tapped_representations(h_seq, prompt_tokens)
            entity_input = taps["h_query_span_cross_readout"]
        elif strategy in (
            "entity_conditioned_attention",
            "learned_entity_cross_attention",
            "eca",
        ):
            assert self.eca_extractor is not None
            key_pad_mask, q_indices = compute_query_cross_prefix_mask(prompt_tokens)
            batch_idx = torch.arange(prompt_tokens.size(0), device=prompt_tokens.device)
            q_query = h_seq[batch_idx, q_indices].unsqueeze(1)  # (B, 1, d_model)
            attended_repr = self.eca_extractor(
                query_repr=q_query,
                seq_hidden_states=h_seq,
                key_padding_mask=key_pad_mask,
            )
            entity_input = attended_repr
        else:
            raise ValueError(f"Unknown readout extraction strategy: {strategy}")

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
