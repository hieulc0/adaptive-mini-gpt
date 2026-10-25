"""Milestone M102.1 Rehearsal Buffer with Per-Family Uncertainty Tracking.

Maintains verified exemplar sets for every qualified domain, ensuring:
1. Full representation of all input-output mappings (no under-sampled token amnesia).
2. Per-family sample counts, accuracy, and standard error uncertainty estimation.
3. Strict domain-isolated evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FamilyRetentionReportM102_1:
    domain: str
    sample_count: int
    retention_accuracy: float
    standard_error: float


@dataclass
class RehearsalBufferAuditSummaryM102_1:
    registered_domains_count: int
    mean_retention_accuracy: float
    worst_family_retention_accuracy: float
    worst_family_domain: str
    family_reports: dict[str, FamilyRetentionReportM102_1]


class BalancedRehearsalBufferM102_1:
    """Stores and audits exemplar tokens evenly across all registered domains."""

    def __init__(self, max_exemplars_per_domain: int = 32) -> None:
        self.max_exemplars = max_exemplars_per_domain
        self.domain_exemplars: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def register_domain(self, domain: str, tokens: torch.Tensor, labels: torch.Tensor) -> None:
        """Register or update verified exemplars for a domain."""
        n = min(self.max_exemplars, tokens.size(0))
        self.domain_exemplars[domain] = (
            tokens[:n].detach().clone(),
            labels[:n].detach().clone(),
        )

    def sample_domain_batches(self, batch_size_per_domain: int = 16) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        """Sample distinct per-domain batches with their matching domain key."""
        batches = []
        for domain, (toks, lbls) in self.domain_exemplars.items():
            k = min(batch_size_per_domain, toks.size(0))
            if k == toks.size(0):
                batches.append((domain, toks, lbls))
            else:
                idx = torch.randperm(toks.size(0), device=toks.device)[:k]
                batches.append((domain, toks[idx], lbls[idx]))
        return batches

    def clone(self) -> BalancedRehearsalBufferM102_1:
        """Create an independent copy of the buffer."""
        buf = BalancedRehearsalBufferM102_1(max_exemplars_per_domain=self.max_exemplars)
        for dom, (t, l) in self.domain_exemplars.items():
            buf.domain_exemplars[dom] = (t.detach().clone(), l.detach().clone())
        return buf

    def evaluate_all_families_retention(
        self,
        student: Any,
    ) -> RehearsalBufferAuditSummaryM102_1:
        """Evaluate retention and uncertainty across every individually qualified family."""
        student.eval()
        reports: dict[str, FamilyRetentionReportM102_1] = {}

        with torch.no_grad():
            for domain, (tokens, labels) in self.domain_exemplars.items():
                logits = student.forward(tokens, active_domains=[domain])
                preds = logits.argmax(dim=-1) % 16
                correct = (preds == labels).float()
                acc = float(correct.mean().item()) * 100.0
                n = tokens.size(0)
                p = max(0.0, min(1.0, acc / 100.0))
                se = (math.sqrt(p * (1.0 - p) / max(1, n)) * 100.0) if n > 1 else 0.0

                reports[domain] = FamilyRetentionReportM102_1(
                    domain=domain,
                    sample_count=n,
                    retention_accuracy=round(acc, 2),
                    standard_error=round(se, 2),
                )

        if not reports:
            return RehearsalBufferAuditSummaryM102_1(
                registered_domains_count=0,
                mean_retention_accuracy=100.0,
                worst_family_retention_accuracy=100.0,
                worst_family_domain="none",
                family_reports={},
            )

        worst_dom = min(reports.keys(), key=lambda d: reports[d].retention_accuracy)
        worst_acc = reports[worst_dom].retention_accuracy
        mean_acc = float(sum(r.retention_accuracy for r in reports.values()) / len(reports))

        return RehearsalBufferAuditSummaryM102_1(
            registered_domains_count=len(reports),
            mean_retention_accuracy=round(mean_acc, 2),
            worst_family_retention_accuracy=round(worst_acc, 2),
            worst_family_domain=worst_dom,
            family_reports=reports,
        )
