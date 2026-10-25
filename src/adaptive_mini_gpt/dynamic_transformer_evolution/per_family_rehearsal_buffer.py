from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch


@dataclass
class DomainExemplarBatch:
    domain: str
    tokens: torch.Tensor
    labels: torch.Tensor


class BalancedPerFamilyRehearsalBuffer:
    """Stores and samples exemplar tokens evenly across all registered domains."""

    def __init__(self, max_exemplars_per_domain: int = 32) -> None:
        self.max_exemplars = max_exemplars_per_domain
        self.domain_exemplars: dict[str, DomainExemplarBatch] = {}

    def register_domain(self, domain: str, tokens: torch.Tensor, labels: torch.Tensor) -> None:
        """Register or update verified exemplars for a domain."""
        n = min(self.max_exemplars, tokens.size(0))
        self.domain_exemplars[domain] = DomainExemplarBatch(
            domain=domain,
            tokens=tokens[:n].detach().clone(),
            labels=labels[:n].detach().clone(),
        )

    def sample_domain_batches(self, batch_size_per_domain: int = 16) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        """Sample distinct per-domain batches with their matching domain key."""
        batches = []
        for domain, batch in self.domain_exemplars.items():
            k = min(batch_size_per_domain, batch.tokens.size(0))
            if k == batch.tokens.size(0):
                batches.append((domain, batch.tokens, batch.labels))
            else:
                idx = torch.randperm(batch.tokens.size(0), device=batch.tokens.device)[:k]
                batches.append((domain, batch.tokens[idx], batch.labels[idx]))
        return batches

    def sample_balanced_batch(self, batch_size_per_domain: int = 16) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample an equal number of exemplars from every registered domain."""
        if not self.domain_exemplars:
            return None

        all_tok = []
        all_lbl = []
        for _, batch in self.domain_exemplars.items():
            k = min(batch_size_per_domain, batch.tokens.size(0))
            if k == batch.tokens.size(0):
                all_tok.append(batch.tokens)
                all_lbl.append(batch.labels)
            else:
                idx = torch.randperm(batch.tokens.size(0), device=batch.tokens.device)[:k]
                all_tok.append(batch.tokens[idx])
                all_lbl.append(batch.labels[idx])

        return torch.cat(all_tok, dim=0), torch.cat(all_lbl, dim=0)

    def clone(self) -> BalancedPerFamilyRehearsalBuffer:
        """Create an independent copy of the buffer."""
        buf = BalancedPerFamilyRehearsalBuffer(max_exemplars_per_domain=self.max_exemplars)
        for dom, batch in self.domain_exemplars.items():
            buf.domain_exemplars[dom] = DomainExemplarBatch(
                domain=batch.domain,
                tokens=batch.tokens.detach().clone(),
                labels=batch.labels.detach().clone(),
            )
        return buf

    def evaluate_all_families_retention(
        self,
        student: Any,
    ) -> dict[str, float]:
        """Evaluate exact match retention across every individually qualified family."""
        student.eval()
        results: dict[str, float] = {}
        with torch.no_grad():
            for domain, batch in self.domain_exemplars.items():
                logits = student.forward(batch.tokens, active_domains=[domain])
                preds = logits.argmax(dim=-1) % 16
                acc = float((preds == batch.labels).float().mean().item()) * 100.0
                results[domain] = round(acc, 2)
        return results
