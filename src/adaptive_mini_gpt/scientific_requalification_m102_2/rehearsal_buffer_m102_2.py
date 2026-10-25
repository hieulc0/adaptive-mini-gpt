"""Milestone M102.2 Scientific Rehearsal and Retention Verification.

Distinctly tracks:
1. Stored Replay Exemplars (Used during fast adaptation and consolidation).
2. Independent Retention Test Sets (Never used for training or replay; 0% leakage).
3. Evaluates both:
   - Oracle-Routed Retention (with explicit domain label).
   - Autonomous Unlabeled Retention (dispatched purely via internal DomainRouter).
4. Full Uncertainty Estimation (standard error per family).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DomainRetentionAuditM102_2:
    domain: str
    sample_count: int
    oracle_retention_acc: float
    unlabeled_retention_acc: float
    domain_routing_acc: float
    standard_error: float


@dataclass
class LifelongRetentionSummaryM102_2:
    num_domains: int
    mean_independent_retention: float
    worst_family_independent_retention: float
    worst_family_domain: str
    mean_unlabeled_retention: float
    mean_domain_routing_accuracy: float
    exemplar_replay_retention: float
    family_audits: dict[str, DomainRetentionAuditM102_2]


class ScientificRetentionManagerM102_2:
    """Manages replay exemplars and independent unseen evaluation sets."""

    def __init__(self, max_exemplars_per_domain: int = 32) -> None:
        self.max_exemplars = max_exemplars_per_domain
        self.domain_exemplars: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.independent_test_sets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def register_domain(
        self,
        domain: str,
        train_tokens: torch.Tensor,
        train_labels: torch.Tensor,
        independent_test_tokens: torch.Tensor,
        independent_test_labels: torch.Tensor,
    ) -> None:
        """Register replay exemplars from training data, and store independent unseen test set."""
        n_ex = min(self.max_exemplars, train_tokens.size(0))
        self.domain_exemplars[domain] = (
            train_tokens[:n_ex].detach().clone(),
            train_labels[:n_ex].detach().clone(),
        )
        self.independent_test_sets[domain] = (
            independent_test_tokens.detach().clone(),
            independent_test_labels.detach().clone(),
        )

    def sample_replay_batches(self, batch_size_per_domain: int = 16) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        """Sample balanced replay batches from stored exemplars."""
        batches = []
        for domain, (toks, lbls) in self.domain_exemplars.items():
            k = min(batch_size_per_domain, toks.size(0))
            if k == toks.size(0):
                batches.append((domain, toks, lbls))
            else:
                idx = torch.randperm(toks.size(0), device=toks.device)[:k]
                batches.append((domain, toks[idx], lbls[idx]))
        return batches

    def clone(self) -> ScientificRetentionManagerM102_2:
        """Deep copy for rollback and causal arm evaluation."""
        mgr = ScientificRetentionManagerM102_2(max_exemplars_per_domain=self.max_exemplars)
        for d, (t, l) in self.domain_exemplars.items():
            mgr.domain_exemplars[d] = (t.detach().clone(), l.detach().clone())
        for d, (t, l) in self.independent_test_sets.items():
            mgr.independent_test_sets[d] = (t.detach().clone(), l.detach().clone())
        return mgr

    def evaluate_retention(self, student: Any) -> LifelongRetentionSummaryM102_2:
        """Evaluate both stored exemplar accuracy and independent held-out retention."""
        student.eval()
        audits: dict[str, DomainRetentionAuditM102_2] = {}
        exemplar_scores = []

        with torch.no_grad():
            # 1. Performance on stored exemplars
            for dom, (e_tok, e_lbl) in self.domain_exemplars.items():
                logits = student.forward(e_tok, active_domains=[dom])
                preds = logits.argmax(dim=-1) % 16
                acc = float((preds == e_lbl).float().mean().item()) * 100.0
                exemplar_scores.append(acc)

            # 2. Performance on independent never-trained test sets
            for dom, (t_tok, t_lbl) in self.independent_test_sets.items():
                n = t_tok.size(0)

                # A. Oracle-routed retention (with domain label)
                oracle_logits = student.forward(t_tok, active_domains=[dom])
                oracle_preds = oracle_logits.argmax(dim=-1) % 16
                oracle_acc = float((oracle_preds == t_lbl).float().mean().item()) * 100.0

                # B. Autonomous unlabeled retention (without domain label)
                unlabeled_logits = student.forward(t_tok, active_domains=None)
                unlabeled_preds = unlabeled_logits.argmax(dim=-1) % 16
                unlabeled_acc = float((unlabeled_preds == t_lbl).float().mean().item()) * 100.0

                # C. Domain router classification accuracy
                h_emb = student.evolvable_backbone(t_tok)
                router_logits = student.head_manager.domain_router(h_emb)
                router_preds = router_logits.argmax(dim=-1)
                dom_idx = student.domains.index(dom) if dom in student.domains else 0
                routing_acc = float((router_preds == dom_idx).float().mean().item()) * 100.0

                p = max(0.0, min(1.0, oracle_acc / 100.0))
                se = (math.sqrt(p * (1.0 - p) / max(1, n)) * 100.0) if n > 1 else 0.0

                audits[dom] = DomainRetentionAuditM102_2(
                    domain=dom,
                    sample_count=n,
                    oracle_retention_acc=round(oracle_acc, 2),
                    unlabeled_retention_acc=round(unlabeled_acc, 2),
                    domain_routing_acc=round(routing_acc, 2),
                    standard_error=round(se, 2),
                )

        if not audits:
            return LifelongRetentionSummaryM102_2(
                num_domains=0,
                mean_independent_retention=100.0,
                worst_family_independent_retention=100.0,
                worst_family_domain="none",
                mean_unlabeled_retention=100.0,
                mean_domain_routing_accuracy=100.0,
                exemplar_replay_retention=100.0,
                family_audits={},
            )

        mean_indep = sum(a.oracle_retention_acc for a in audits.values()) / len(audits)
        worst_dom = min(audits.keys(), key=lambda d: audits[d].oracle_retention_acc)
        worst_indep = audits[worst_dom].oracle_retention_acc
        mean_unlbl = sum(a.unlabeled_retention_acc for a in audits.values()) / len(audits)
        mean_routing = sum(a.domain_routing_acc for a in audits.values()) / len(audits)
        mean_ex = (sum(exemplar_scores) / len(exemplar_scores)) if exemplar_scores else 100.0

        return LifelongRetentionSummaryM102_2(
            num_domains=len(audits),
            mean_independent_retention=round(mean_indep, 2),
            worst_family_independent_retention=round(worst_indep, 2),
            worst_family_domain=worst_dom,
            mean_unlabeled_retention=round(mean_unlbl, 2),
            mean_domain_routing_accuracy=round(mean_routing, 2),
            exemplar_replay_retention=round(mean_ex, 2),
            family_audits=audits,
        )
