"""Milestone M104-D Calibrated Semantic Bottleneck and Integrated Student.

Key Architectural Features:
1. Entity-Targeted Sequence Pooling: Locates entity/question tokens vs general rule tokens,
   pooling distinct entity representations for attribute extraction and preventing rule leakage.
2. Compact Low-Rank Semantic Bottleneck (~50k parameters, maintaining the ~56% adapter reduction).
3. Selective Plasticity: Freezes lower linguistic layers (Blocks 0-3), selectively training only
   the top block (Block 4), final LayerNorm, and semantic bottleneck.
4. Strictly Text-Only Inference: Never accesses oracle attributes or ground-truth simulator state.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)


class SemanticBottleneckM104_D(nn.Module):
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

        # 4. Context-conditioned Transition Dynamics Network:
        # Combines entity concept + global context + action embedding
        self.transition_net = nn.Sequential(
            nn.Linear(d_model * 2 + action_dim, d_model),
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
            entity_repr: (B, d_model) pooled entity description representation.
            global_repr: (B, d_model) pooled whole-sequence causal context.
            action_ids: (B,) action indices.
        Returns:
            state_logits: (B, num_states)
            aux_tag_logits: (B, num_tags)
            z_concept: (B, d_model)
            z_next: (B, d_model)
        """
        # Concept Bottleneck
        z_concept = self.concept_proj(entity_repr)

        # Auxiliary Attribute Logits for grounding supervision
        aux_tag_logits = self.attribute_head(z_concept)

        # Transition Dynamics: z_next = LayerNorm(z_concept + f(z_concept, global_repr, action))
        a_emb = self.action_emb(action_ids)
        comb = torch.cat([z_concept, global_repr, a_emb], dim=-1)
        delta_z = self.transition_net(comb)
        z_next = self.post_norm(z_concept + delta_z)

        # Final Next-State Classification Logits
        state_logits = self.state_head(z_next)

        return state_logits, aux_tag_logits, z_concept, z_next


class IntegratedStudentM104_D(nn.Module):
    """Integrates TextStudentM103 with SemanticBottleneckM104_D with Entity-Targeted Sequence Pooling."""

    def __init__(
        self,
        student: TextStudentM103,
        num_actions: int = 8,
        num_states: int = 16,
        num_tags: int = 26,
        mode: str = "selective_plasticity",  # "selective_plasticity", "frozen_backbone", "trainable_backbone", "no_adapter"
    ) -> None:
        super().__init__()
        self.student = student
        self.mode = mode
        d_model = student.base_model.config.d_model

        if mode in ("selective_plasticity", "frozen_backbone", "trainable_backbone"):
            self.bottleneck = SemanticBottleneckM104_D(
                d_model=d_model,
                bottleneck_dim=32,
                action_dim=64,
                num_actions=num_actions,
                num_states=num_states,
                num_tags=num_tags,
            ).to(student.device)
            self.linear_head = None
        else:
            self.bottleneck = None
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

        elif self.mode == "selective_plasticity":
            # Freeze lower blocks (Blocks 0-3), wte, wpe
            for p in self.student.parameters():
                p.requires_grad = False

            # Selectively unfreeze only the top Transformer block (Block 4) and final LayerNorm
            if hasattr(self.student.base_model, "blocks") and len(self.student.base_model.blocks) > 0:
                top_block = self.student.base_model.blocks[-1]
                for p in top_block.parameters():
                    p.requires_grad = True

            if hasattr(self.student.base_model, "ln_f") and self.student.base_model.ln_f is not None:
                for p in self.student.base_model.ln_f.parameters():
                    p.requires_grad = True

            if self.bottleneck is not None:
                for p in self.bottleneck.parameters():
                    p.requires_grad = True

        elif self.mode == "trainable_backbone":
            for p in self.student.parameters():
                p.requires_grad = True
            if self.bottleneck is not None:
                for p in self.bottleneck.parameters():
                    p.requires_grad = True

    @property
    def device(self) -> torch.device:
        return self.student.device

    def extract_sequence_hidden_states(self, tokens: torch.Tensor) -> torch.Tensor:
        """Runs Transformer and returns hidden sequence H: (B, L, d_model)."""
        b, t = tokens.size()
        tok_emb = self.student.base_model.wte(tokens)
        pos = torch.arange(0, t, dtype=torch.long, device=tokens.device)
        x = self.student.base_model.drop(tok_emb + self.student.base_model.wpe(pos))

        for block in self.student.base_model.blocks:
            x, _ = block(x)

        return self.student.base_model.ln_f(x)

    def pool_representations(
        self,
        h_seq: torch.Tensor,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts entity-targeted representation and whole-sequence global causal representation.

        Args:
            h_seq: (B, L, d_model) Transformer hidden states.
            tokens: (B, L) token IDs.

        Returns:
            entity_repr: (B, d_model) representation pooled specifically over entity question tokens.
            global_repr: (B, d_model) representation pooled over all non-padding prompt tokens.
        """
        B, L, D = h_seq.size()
        # 1. Global prompt mask (non-padding tokens)
        global_mask = (tokens != ByteTextTokenizerM103.PAD_ID) & (tokens != 0)
        global_mask_f = global_mask.float().unsqueeze(-1)  # (B, L, 1)
        global_repr = (h_seq * global_mask_f).sum(dim=1) / global_mask_f.sum(dim=1).clamp(min=1.0)

        # 2. Entity-targeted mask: tokens between QUESTION_ID and ANSWER_ID
        # If question token is not found, fallback to global representation
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
                # Fallback to second half of prompt if delimiters omitted
                half = L // 2
                entity_mask[b, half:] = True

        entity_mask_f = (entity_mask & global_mask).float().unsqueeze(-1)
        entity_repr = (h_seq * entity_mask_f).sum(dim=1) / entity_mask_f.sum(dim=1).clamp(min=1.0)

        return entity_repr, global_repr

    def forward(
        self,
        prompt_tokens: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Full end-to-end forward pass from natural language text prompts."""
        h_seq = self.extract_sequence_hidden_states(prompt_tokens)  # (B, L, d_model)
        entity_repr, global_repr = self.pool_representations(h_seq, prompt_tokens)

        if self.bottleneck is not None:
            return self.bottleneck(entity_repr, global_repr, action_ids)
        else:
            a_emb = self.action_emb(action_ids)
            comb = torch.cat([global_repr, a_emb], dim=-1)
            state_logits = self.linear_head(comb)
            return state_logits, None, global_repr, global_repr
