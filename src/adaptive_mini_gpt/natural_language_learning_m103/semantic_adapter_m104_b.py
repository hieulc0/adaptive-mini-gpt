"""Milestone M104-B Semantic Concept Adapter & Grounding Substrate.

Implements the integrated semantic concept adapter for TextStudentM103:
1. Learnable concept probe tokens Q_concept (K=4) that cross-attend to Transformer hidden states.
2. Bottleneck residual projection (rank-16) to form grounded concept latent representation z_concept.
3. Multi-task grounding during training (Causal Transition Loss + Auxiliary Attribute Grounding Loss).
4. Strictly language-only inference: zero attribute supervision provided at evaluation.
5. Ablation modes:
   - IntegratedSemanticStudent (trainable backbone + semantic adapter)
   - FrozenBackbone_Adapter (frozen backbone + semantic adapter)
   - TrainableBackbone_NoAdapter (trainable backbone without concept adapter)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)


class SemanticConceptAdapterM104_B(nn.Module):
    """Semantic concept adapter mapping transformer hidden states into structured concepts."""

    def __init__(
        self,
        d_model: int = 128,
        num_probes: int = 4,
        bottleneck_dim: int = 16,
        num_actions: int = 8,
        num_states: int = 16,
        num_tags: int = 26,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_probes = num_probes

        # Learnable concept probe tokens (Material, Physical, Arcane, Action relation)
        self.concept_probes = nn.Parameter(torch.randn(1, num_probes, d_model) * 0.02)

        # Multi-head cross-attention: probes attend across sequence hidden states
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Bottleneck feed-forward adapter
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Action embedding and transition dynamics
        self.action_emb = nn.Embedding(num_actions, d_model)
        self.transition_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.post_transition_norm = nn.LayerNorm(d_model)

        # Prediction Heads
        self.state_head = nn.Linear(d_model, num_states)
        self.aux_attribute_head = nn.Linear(d_model, num_tags)

    def forward(
        self,
        hidden_states: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            hidden_states: (B, L, d_model) from Transformer.
            action_ids: (B,) action indices.

        Returns:
            state_logits: (B, num_states)
            aux_tag_logits: (B, num_tags)
            z_concept: (B, d_model)
            z_next: (B, d_model)
        """
        b = hidden_states.size(0)
        probes = self.concept_probes.expand(b, -1, -1)  # (B, K, d_model)

        # Cross attention: probes query transformer hidden states
        attn_out, _ = self.cross_attn(query=probes, key=hidden_states, value=hidden_states)
        x = self.norm1(probes + attn_out)

        # Bottleneck residual
        x = self.norm2(x + self.bottleneck(x))

        # Mean pooling across probe tokens to form unified concept vector z_concept
        z_concept = x.mean(dim=1)  # (B, d_model)

        # Auxiliary attribute prediction (grounding)
        aux_tag_logits = self.aux_attribute_head(z_concept)

        # Causal transition dynamics: z_next = z_concept + f_theta(z_concept, action)
        a_rep = self.action_emb(action_ids)
        comb = torch.cat([z_concept, a_rep], dim=-1)
        delta_z = self.transition_net(comb)
        z_next = self.post_transition_norm(z_concept + delta_z)

        # Next-state classification logits
        state_logits = self.state_head(z_next)

        return state_logits, aux_tag_logits, z_concept, z_next


class IntegratedSemanticStudentM104_B(nn.Module):
    """Integrates TextStudentM103 with the SemanticConceptAdapterM104_B."""

    def __init__(
        self,
        student: TextStudentM103,
        num_actions: int = 8,
        num_states: int = 16,
        num_tags: int = 26,
        mode: str = "integrated_adapter",  # "integrated_adapter", "frozen_backbone", "no_adapter"
    ) -> None:
        super().__init__()
        self.student = student
        self.mode = mode
        d_model = student.base_model.config.d_model

        if mode in ("integrated_adapter", "frozen_backbone"):
            self.adapter = SemanticConceptAdapterM104_B(
                d_model=d_model,
                num_probes=4,
                bottleneck_dim=16,
                num_actions=num_actions,
                num_states=num_states,
                num_tags=num_tags,
            ).to(student.device)
            self.linear_head = None
        else:
            # Baseline ablation: direct linear head without concept adapter
            self.adapter = None
            self.linear_head = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, num_states),
            ).to(student.device)
            self.action_emb = nn.Embedding(num_actions, d_model).to(student.device)

        if mode == "frozen_backbone":
            for p in self.student.parameters():
                p.requires_grad = False

    @property
    def device(self) -> torch.device:
        return self.student.device

    def extract_sequence_hidden_states(self, tokens: torch.Tensor) -> torch.Tensor:
        """Runs Transformer and returns full hidden sequence H: (B, L, d_model)."""
        b, t = tokens.size()
        tok_emb = self.student.base_model.wte(tokens)
        pos = torch.arange(0, t, dtype=torch.long, device=tokens.device)
        x = self.student.base_model.drop(tok_emb + self.student.base_model.wpe(pos))

        for block in self.student.base_model.blocks:
            x, _ = block(x)

        return self.student.base_model.ln_f(x)

    def forward(
        self,
        tokens: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            state_logits, aux_tag_logits, z_concept, z_next
        """
        hidden = self.extract_sequence_hidden_states(tokens)

        if self.adapter is not None:
            return self.adapter(hidden, action_ids)

        # Baseline NoAdapter path
        z_pool = hidden.mean(dim=1)
        a_rep = self.action_emb(action_ids)
        state_logits = self.linear_head(torch.cat([z_pool, a_rep], dim=-1))
        return state_logits, None, z_pool, z_pool
