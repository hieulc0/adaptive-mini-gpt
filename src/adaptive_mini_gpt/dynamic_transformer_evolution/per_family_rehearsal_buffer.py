"""Milestone M102 Balanced Per-Family Rehearsal Buffer & Safeguards.

Repairs catastrophic forgetting across individual skill families:
1. Multi-Domain Repertoire Buffer: stores verified exemplars for ALL qualified domains.
2. Balanced Batch Sampler: interleaves current goal training tokens with equal-fraction
   rehearsal samples across all qualified prior families.
3. Strict Per-Family Retention Gate: consolidation and adaptation candidates are promoted
   ONLY IF worst-family retention >= 80.0% and aggregate retention >= 90.0%.
4. Atomic Rollback on Any Family Violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import torch


@dataclass
class DomainExemplarBatch:
    """Packaged rehearsal tokens and labels for a specific domain."""

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

    def sample_balanced_batch(self, batch_size_per_domain: int = 4) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample an equal number of exemplars from every registered domain."""
        if not self.domain_exemplars:
            return None

        all_tok = []
        all_lbl = []
        for domain, batch in self.domain_exemplars.items():
            k = min(batch_size_per_domain, batch.tokens.size(0))
            idx = torch.randperm(batch.tokens.size(0))[:k]
            all_tok.append(batch.tokens[idx])
            all_lbl.append(batch.labels[idx])

        return torch.cat(all_tok, dim=0), torch.cat(all_lbl, dim=0)

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
