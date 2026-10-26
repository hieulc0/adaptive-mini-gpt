"""Milestone M103-B Phase 2: Language Substrate Learning.

Implements controlled causal next-token pretraining on structured synthetic English
sentences describing rules, entities, quantities, and transitions.
Grounds byte embeddings, positional encodings, and Transformer representations
in valid language syntax before rule-based QA adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)


@dataclass
class PretrainingReportM103_B:
    num_pretraining_sentences: int
    pretraining_steps: int
    initial_loss: float
    final_loss: float
    loss_reduction_pct: float
    pretraining_certified: bool
    sample_sentence: str


class LanguagePretrainerM103_B:
    """Pretrains MiniGPT decoder on domain-relevant text corpora."""

    CORPUS_TEMPLATES = [
        "In System-Alpha, a blue orb gains {gain} energy units.",
        "A red orb loses {loss} energy unit when processed.",
        "When both rules apply to an orb, blue acts first then red.",
        "An item starts with {s} energy units and ends with {s2} energy units.",
        "In Grid-Beta, moving North adds {step_n} units to position.",
        "Moving South subtracts {step_s} units from coordinate position.",
        "A traveler starts at position {s} and reaches final position {s2}.",
        "Under cipher Gamma, symbol Alpha transforms into symbol Omega.",
        "Logic circuit Delta outputs High when control signal is active.",
        "The energy level of an object increases with positive charges.",
        "A coordinate grid tracks horizontal and vertical spatial movement.",
        "Final values are evaluated sequentially according to assigned priority.",
    ]

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        self.tokenizer = ByteTextTokenizerM103()

    def generate_pretraining_corpus(self) -> list[str]:
        """Generate structured text corpus for pretraining."""
        sentences: list[str] = []
        for s in range(1, 16):
            gain = (s % 3) + 1
            loss = (s % 2) + 1
            step_n = (s % 4) + 1
            step_s = (s % 2) + 1
            s2 = s + gain - loss
            for tmpl in self.CORPUS_TEMPLATES:
                formatted = tmpl.format(
                    s=s, s2=s2, gain=gain, loss=loss, step_n=step_n, step_s=step_s
                )
                sentences.append(formatted)
        return sentences

    def pretrain_student(
        self,
        student: TextStudentM103,
        steps: int = 60,
        lr: float = 0.003,
    ) -> PretrainingReportM103_B:
        """Pretrain base Transformer parameters with next-token causal loss."""
        corpus = self.generate_pretraining_corpus()
        token_lists = [self.tokenizer.encode(sent, add_bos=True, add_eos=True) for sent in corpus]
        batch_tokens, mask = self.tokenizer.pad_batch(token_lists, device=self.device)

        # Unfreeze backbone and embeddings for pretraining
        for p in student.parameters():
            p.requires_grad = True

        opt = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=1e-4,
        )

        initial_loss = 0.0
        final_loss = 0.0

        student.train()
        for s in range(1, steps + 1):
            opt.zero_grad()
            logits = student.forward_lm(batch_tokens, active_domain=None)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch_tokens[:, 1:].contiguous().clone()
            shift_labels[~mask[:, 1:]] = -100

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            if s == 1:
                initial_loss = float(loss.item())

            loss.backward()
            opt.step()
            final_loss = float(loss.item())

        reduction = round(((initial_loss - final_loss) / max(1e-5, initial_loss)) * 100.0, 2)
        certified = (final_loss < 1.5 and reduction >= 60.0)

        return PretrainingReportM103_B(
            num_pretraining_sentences=len(corpus),
            pretraining_steps=steps,
            initial_loss=round(initial_loss, 4),
            final_loss=round(final_loss, 4),
            loss_reduction_pct=reduction,
            pretraining_certified=certified,
            sample_sentence=corpus[0],
        )
