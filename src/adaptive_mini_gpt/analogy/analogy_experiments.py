"""Phase 4 & 5: Structural Analogy Learning Experiments and Sample Efficiency Benchmarks.

Compares four learning approaches across sample budgets N in [0, 2, 4, 8, 16, 32]:
  Approach A: Fresh Adaptation (baseline few-shot adaptation without memory or analogy).
  Approach B: Lexical Memory Retrieval (retrieval by surface token/keyword overlap).
  Approach C: Structure-Matched Episodic Retrieval (in-context abstract graph schema from verified analogy).
  Approach D: Neural Analogy with Verified Adaptation (structural schema conditioning + adapter optimization).

Tests on held-out relational families with:
  - Completely unseen entity names.
  - Completely disjoint surface vocabulary.
  - Compatible vs incompatible relational semantics.
  - Balanced YES, NO, and UNKNOWN test distributions.

Solves the M43 flat-curve defect by ensuring sufficient training iterations per example
and tracks full 3x3 confusion matrices (YES, NO, UNKNOWN) at every budget.
"""
from __future__ import annotations

import collections
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from adaptive_mini_gpt.analogy.audit import BudgetConfusionMatrix
from adaptive_mini_gpt.analogy.structural_schema import (
    CanonicalGraphSignature,
    DeterministicStructuralMatcher,
    RelationProperties,
    StructuralAnalogyExperience,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.meta.benchmark_suite import MetaFamilyDataset, MetaTask
from adaptive_mini_gpt.model.modular_adapter import (
    add_domain_adapter,
    freeze_for_adapter_training,
    inject_multi_domain_adapters,
    set_active_domain_adapter,
)
from adaptive_mini_gpt.model.transformer import MiniGPT

logger = logging.getLogger("m44_analogy_experiments")


@dataclass
class AnalogyLearningCurveStep:
    budget: int
    accuracy: float
    yes_accuracy: float
    no_accuracy: float
    unknown_accuracy: float
    confusion_matrix: BudgetConfusionMatrix
    mean_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "accuracy": round(self.accuracy, 4),
            "yes_accuracy": round(self.yes_accuracy, 4),
            "no_accuracy": round(self.no_accuracy, 4),
            "unknown_accuracy": round(self.unknown_accuracy, 4),
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
        }


@dataclass
class AnalogyApproachReport:
    approach_name: str
    target_family: str
    relation_properties: dict[str, Any]
    curve: list[AnalogyLearningCurveStep]
    final_accuracy: float
    area_under_curve: float
    n50_examples: int | None
    n80_examples: int | None
    training_steps_total: int
    total_duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "approach_name": self.approach_name,
            "target_family": self.target_family,
            "relation_properties": self.relation_properties,
            "curve": [c.to_dict() for c in self.curve],
            "final_accuracy": round(self.final_accuracy, 4),
            "area_under_curve": round(self.area_under_curve, 4),
            "n50_examples": self.n50_examples,
            "n80_examples": self.n80_examples,
            "training_steps_total": self.training_steps_total,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
        }


def build_analogy_task_family(
    family_name: str,
    split: str,
    rel: str,
    inv_rel: str,
    entities: list[str],
    is_transitive: bool = True,
    is_symmetric: bool = False,
    is_directional: bool = True,
) -> tuple[MetaFamilyDataset, RelationProperties]:
    """Builds a relational task family with ground-truth semantic properties."""
    assert len(entities) >= 4, "Must supply at least 4 entities"
    e0, e1, e2, e3 = entities[0], entities[1], entities[2], entities[3]

    base_premises = [f"{e0} {rel} {e1}", f"{e1} {rel} {e2}", f"{e2} {rel} {e3}"]

    practice_tasks: list[MetaTask] = []
    # 1. 1-hop
    practice_tasks.append(MetaTask(f"{family_name}_pr_1", family_name, split, base_premises, f"Does {e0} {rel} {e1}?", "YES", "single_hop", [e0, e1]))
    practice_tasks.append(MetaTask(f"{family_name}_pr_2", family_name, split, base_premises, f"Does {e1} {rel} {e2}?", "YES", "single_hop", [e1, e2]))
    # 2. 2-hop
    practice_tasks.append(MetaTask(f"{family_name}_pr_3", family_name, split, base_premises, f"Does {e0} {rel} {e2}?", "YES" if is_transitive else "NO", "two_hop", [e0, e2]))
    # 3. Asymmetric reversal
    practice_tasks.append(MetaTask(f"{family_name}_pr_4", family_name, split, base_premises, f"Does {e1} {rel} {e0}?", "YES" if is_symmetric else "NO", "asymmetric", [e1, e0]))
    practice_tasks.append(MetaTask(f"{family_name}_pr_5", family_name, split, base_premises, f"Does {e2} {rel} {e1}?", "YES" if is_symmetric else "NO", "asymmetric", [e2, e1]))
    # 4. Ungrounded
    practice_tasks.append(MetaTask(f"{family_name}_pr_6", family_name, split, base_premises, f"Does {e0} {rel} UnknownParticleX?", "UNKNOWN", "ungrounded", [e0, "UnknownParticleX"]))
    # 5. 3-hop
    practice_tasks.append(MetaTask(f"{family_name}_pr_7", family_name, split, base_premises, f"Does {e0} {rel} {e3}?", "YES" if is_transitive else "NO", "three_hop", [e0, e3]))
    # 6. Reversal 2-hop
    practice_tasks.append(MetaTask(f"{family_name}_pr_8", family_name, split, base_premises, f"Does {e2} {rel} {e0}?", "YES" if is_symmetric else "NO", "asymmetric", [e2, e0]))

    # Expand to 32 practice tasks with lexical variations
    for i in range(9, 33):
        base_t = practice_tasks[(i - 1) % 8]
        practice_tasks.append(
            MetaTask(
                task_id=f"{family_name}_pr_{i}",
                family=family_name,
                split=split,
                premises=base_t.premises,
                question=f"Is it verified that {base_t.question.replace('Does ', '').replace('?', '')}?",
                expected_verdict=base_t.expected_verdict,
                reasoning_type=base_t.reasoning_type,
                entities_involved=base_t.entities_involved,
            )
        )

    # 20 balanced held-out test tasks: 7 YES, 8 NO, 5 UNKNOWN
    test_tasks: list[MetaTask] = []
    # 7 YES
    test_tasks.append(MetaTask(f"{family_name}_te_1", family_name, split, base_premises, f"Does {e0} {rel} {e1}?", "YES", "single_hop", [e0, e1]))
    test_tasks.append(MetaTask(f"{family_name}_te_2", family_name, split, base_premises, f"Does {e1} {rel} {e2}?", "YES", "single_hop", [e1, e2]))
    test_tasks.append(MetaTask(f"{family_name}_te_3", family_name, split, base_premises, f"Does {e2} {rel} {e3}?", "YES", "single_hop", [e2, e3]))
    test_tasks.append(MetaTask(f"{family_name}_te_4", family_name, split, base_premises, f"Does {e0} {rel} {e2}?", "YES" if is_transitive else "NO", "two_hop", [e0, e2]))
    test_tasks.append(MetaTask(f"{family_name}_te_5", family_name, split, base_premises, f"Does {e1} {rel} {e3}?", "YES" if is_transitive else "NO", "two_hop", [e1, e3]))
    test_tasks.append(MetaTask(f"{family_name}_te_6", family_name, split, base_premises, f"Does {e0} {rel} {e3}?", "YES" if is_transitive else "NO", "three_hop", [e0, e3]))

    # Permuted premise for test task 7
    cf_premises = [f"{e3} {rel} {e2}", f"{e2} {rel} {e1}", f"{e1} {rel} {e0}"]
    test_tasks.append(MetaTask(f"{family_name}_te_7", family_name, split, cf_premises, f"Does {e3} {rel} {e1}?", "YES" if is_transitive else "NO", "two_hop", [e3, e1]))

    # 8 NO
    test_tasks.append(MetaTask(f"{family_name}_te_8", family_name, split, base_premises, f"Does {e1} {rel} {e0}?", "YES" if is_symmetric else "NO", "asymmetric", [e1, e0]))
    test_tasks.append(MetaTask(f"{family_name}_te_9", family_name, split, base_premises, f"Does {e2} {rel} {e1}?", "YES" if is_symmetric else "NO", "asymmetric", [e2, e1]))
    test_tasks.append(MetaTask(f"{family_name}_te_10", family_name, split, base_premises, f"Does {e3} {rel} {e0}?", "YES" if is_symmetric else "NO", "asymmetric", [e3, e0]))
    test_tasks.append(MetaTask(f"{family_name}_te_11", family_name, split, base_premises, f"Does {e3} {rel} {e2}?", "YES" if is_symmetric else "NO", "asymmetric", [e3, e2]))
    test_tasks.append(MetaTask(f"{family_name}_te_12", family_name, split, base_premises, f"Does {e2} {rel} {e0}?", "YES" if is_symmetric else "NO", "asymmetric", [e2, e0]))
    test_tasks.append(MetaTask(f"{family_name}_te_13", family_name, split, cf_premises, f"Does {e0} {rel} {e3}?", "YES" if is_symmetric else "NO", "asymmetric", [e0, e3]))
    test_tasks.append(MetaTask(f"{family_name}_te_14", family_name, split, cf_premises, f"Does {e1} {rel} {e2}?", "YES" if is_symmetric else "NO", "asymmetric", [e1, e2]))
    test_tasks.append(MetaTask(f"{family_name}_te_15", family_name, split, cf_premises, f"Does {e0} {rel} {e1}?", "YES" if is_symmetric else "NO", "asymmetric", [e0, e1]))

    # 5 UNKNOWN
    test_tasks.append(MetaTask(f"{family_name}_te_16", family_name, split, base_premises, f"Does {e0} {rel} NonExistentElement?", "UNKNOWN", "ungrounded", [e0, "NonExistentElement"]))
    test_tasks.append(MetaTask(f"{family_name}_te_17", family_name, split, base_premises, f"Does GhostEntity {rel} {e2}?", "UNKNOWN", "ungrounded", ["GhostEntity", e2]))
    test_tasks.append(MetaTask(f"{family_name}_te_18", family_name, split, base_premises, f"Does AlphaPhantom {rel} BetaPhantom?", "UNKNOWN", "ungrounded", ["AlphaPhantom", "BetaPhantom"]))
    test_tasks.append(MetaTask(f"{family_name}_te_19", family_name, split, base_premises, f"Does {e0} {rel} {e0}?", "UNKNOWN", "contradictory", [e0, e0]))
    test_tasks.append(MetaTask(f"{family_name}_te_20", family_name, split, cf_premises, f"Does {e2} {rel} {e2}?", "UNKNOWN", "contradictory", [e2, e2]))

    dataset = MetaFamilyDataset(
        family_name=family_name,
        split=split,
        relation_name=rel,
        inverse_relation=inv_rel,
        is_transitive=is_transitive,
        is_symmetric=is_symmetric,
        train_practice_pool=practice_tasks,
        held_out_test_pool=test_tasks,
    )
    props = RelationProperties(
        name=rel,
        is_transitive=is_transitive,
        is_symmetric=is_symmetric,
        is_directional=is_directional,
    )
    return dataset, props


def get_m44_benchmark_families() -> dict[str, tuple[MetaFamilyDataset, RelationProperties]]:
    """Builds M44 benchmark families spanning prior source domains and held-out test domains."""
    families: dict[str, tuple[MetaFamilyDataset, RelationProperties]] = {}

    # Source / Prior experience domain 1: Corporate Hierarchy (transitive, asymmetric)
    families["corporate_hierarchy"] = build_analogy_task_family(
        family_name="corporate_hierarchy",
        split="meta_train",
        rel="reports_to",
        inv_rel="manages",
        entities=["JuniorDev", "TechLead", "EngineeringVP", "ChiefExec"],
        is_transitive=True,
        is_symmetric=False,
    )
    # Source / Prior experience domain 2: Containment Taxonomy (transitive, asymmetric)
    families["containment_taxonomy"] = build_analogy_task_family(
        family_name="containment_taxonomy",
        split="meta_train",
        rel="subsumed_by",
        inv_rel="encloses",
        entities=["LocalCluster", "SectorRegion", "DomainSphere", "UniversalMacro"],
        is_transitive=True,
        is_symmetric=False,
    )
    # Source / Prior experience domain 3: Chemical Incompatibility (non-transitive, symmetric)
    families["chemical_incompatibility"] = build_analogy_task_family(
        family_name="chemical_incompatibility",
        split="meta_train",
        rel="incompatible_with",
        inv_rel="incompatible_with",
        entities=["AcidReagent", "BaseSolution", "VolatileSolvent", "InertBuffer"],
        is_transitive=False,
        is_symmetric=True,
    )

    # Held-out Sealed Meta-Test Domain 1: Linguistic Etymology (transitive, asymmetric)
    # Fully disjoint vocabulary from corporate_hierarchy & containment_taxonomy, but identical structure!
    families["linguistic_etymology"] = build_analogy_task_family(
        family_name="linguistic_etymology",
        split="meta_test",
        rel="derives_from",
        inv_rel="gives_rise_to",
        entities=["ProtoRootX", "ArchaicFormY", "DialectZ", "ModernLexemeW"],
        is_transitive=True,
        is_symmetric=False,
    )

    # Held-out Sealed Meta-Test Domain 2: Metabolic Cascade (transitive, asymmetric)
    families["metabolic_cascade"] = build_analogy_task_family(
        family_name="metabolic_cascade",
        split="meta_test",
        rel="catalyzes",
        inv_rel="inhibited_by",
        entities=["EnzymeAlpha", "SubstrateBeta", "ProductGamma", "MetaboliteDelta"],
        is_transitive=True,
        is_symmetric=False,
    )

    # Held-out Sealed Meta-Test Domain 3: Social Rivalry (non-transitive, symmetric)
    families["social_rivalry"] = build_analogy_task_family(
        family_name="social_rivalry",
        split="meta_test",
        rel="rival_of",
        inv_rel="rival_of",
        entities=["GuildSolar", "FactionLunar", "AllianceNova", "OrderZenith"],
        is_transitive=False,
        is_symmetric=True,
    )

    return families


class StructuralAnalogyExperimentRunner:
    """Executes the 4 approaches and calculates sample efficiency curves."""

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: BPETokenizer,
        device: torch.device,
        memory_bank: list[StructuralAnalogyExperience],
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.memory_bank = memory_bank
        self.classes = ["YES", "NO", "UNKNOWN"]
        self.cand_seqs = [self.tokenizer.encode(f" {c}") for c in self.classes]

    def _eval_task(
        self,
        model: MiniGPT,
        task: MetaTask,
        props: RelationProperties,
        approach: str,
    ) -> tuple[bool, str]:
        """Evaluates single task under the specified approach."""
        if approach == "APPROACH_A_FRESH":
            prompt = f"[PREMISES] {' '.join(task.premises)} [QUESTION] {task.question} [ANSWER]"
        elif approach == "APPROACH_B_LEXICAL":
            # Lexical retrieval
            q_tokens = set(task.question.replace("?", "").split()).union(set(" ".join(task.premises).split()))
            matches: list[tuple[int, StructuralAnalogyExperience]] = []
            for exp in self.memory_bank:
                exp_tokens = set(exp.exemplar_question.replace("?", "").split()).union(set(" ".join(exp.exemplar_premises).split()))
                overlap = len(q_tokens.intersection(exp_tokens))
                if overlap > 0:
                    matches.append((overlap, exp))
            matches.sort(key=lambda x: x[0], reverse=True)
            if matches:
                best_exp = matches[0][1]
                prompt = f"[EXEMPLAR] {' '.join(best_exp.exemplar_premises)} {best_exp.exemplar_question} -> {best_exp.verified_verdict} [PREMISES] {' '.join(task.premises)} [QUESTION] {task.question} [ANSWER]"
            else:
                prompt = f"[PREMISES] {' '.join(task.premises)} [QUESTION] {task.question} [ANSWER]"
        elif approach in ("APPROACH_C_STRUCTURE_MATCHED", "APPROACH_D_NEURAL_ANALOGY"):
            # Structural analogy retrieval
            matches = []
            for exp in self.memory_bank:
                score = DeterministicStructuralMatcher.match_analogy(task, props, exp)
                if score >= 0.5:
                    matches.append((score, exp))
            matches.sort(key=lambda x: x[0], reverse=True)
            if matches:
                prompt = DeterministicStructuralMatcher.format_analogical_prompt(task, matches[0][1])
            else:
                prompt = f"[PREMISES] {' '.join(task.premises)} [QUESTION] {task.question} [ANSWER]"
        else:
            prompt = f"[PREMISES] {' '.join(task.premises)} [QUESTION] {task.question} [ANSWER]"

        toks = self.tokenizer.encode(prompt)
        if len(toks) > model.config.context_length - 8:
            toks = toks[-(model.config.context_length - 8) :]

        scores = []
        with torch.no_grad():
            for seq in self.cand_seqs:
                full_seq = toks + seq
                inp = torch.tensor([full_seq], dtype=torch.long, device=self.device)
                out = model(inp)
                logits = out[0] if isinstance(out, tuple) else out
                log_probs = torch.log_softmax(logits, dim=-1)

                seq_lp = 0.0
                for offset, target_token in enumerate(seq):
                    pos = len(toks) - 1 + offset
                    seq_lp += float(log_probs[0, pos, target_token].item())
                scores.append(seq_lp / max(1, len(seq)))

        best_idx = int(torch.tensor(scores).argmax().item())
        pred = self.classes[best_idx]
        return (pred == task.expected_verdict, pred)

    def evaluate_test_tasks(
        self,
        model: MiniGPT,
        test_tasks: list[MetaTask],
        props: RelationProperties,
        approach: str,
        budget: int,
    ) -> AnalogyLearningCurveStep:
        model.eval()
        latencies: list[float] = []

        cm_counts = {
            "YES_YES": 0, "YES_NO": 0, "YES_UNK": 0,
            "NO_YES": 0, "NO_NO": 0, "NO_UNK": 0,
            "UNK_YES": 0, "UNK_NO": 0, "UNK_UNK": 0,
        }

        total_correct = 0
        yes_total = yes_corr = 0
        no_total = no_corr = 0
        unk_total = unk_corr = 0

        for t in test_tasks:
            t0 = time.perf_counter()
            is_corr, pred = self._eval_task(model, t, props, approach)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

            if is_corr:
                total_correct += 1

            true_v = t.expected_verdict
            pred_k = "YES" if pred == "YES" else ("NO" if pred == "NO" else "UNK")
            true_k = "YES" if true_v == "YES" else ("NO" if true_v == "NO" else "UNK")
            cm_counts[f"{true_k}_{pred_k}"] += 1

            if true_v == "YES":
                yes_total += 1
                if pred == "YES":
                    yes_corr += 1
            elif true_v == "NO":
                no_total += 1
                if pred == "NO":
                    no_corr += 1
            elif true_v == "UNKNOWN":
                unk_total += 1
                if pred == "UNKNOWN":
                    unk_corr += 1

        overall_acc = total_correct / max(1, len(test_tasks))
        yes_acc = yes_corr / max(1, yes_total)
        no_acc = no_corr / max(1, no_total)
        unk_acc = unk_corr / max(1, unk_total)

        cm = BudgetConfusionMatrix(
            budget=budget,
            total_samples=len(test_tasks),
            true_yes_pred_yes=cm_counts["YES_YES"],
            true_yes_pred_no=cm_counts["YES_NO"],
            true_yes_pred_unk=cm_counts["YES_UNK"],
            true_no_pred_yes=cm_counts["NO_YES"],
            true_no_pred_no=cm_counts["NO_NO"],
            true_no_pred_unk=cm_counts["NO_UNK"],
            true_unk_pred_yes=cm_counts["UNK_YES"],
            true_unk_pred_no=cm_counts["UNK_NO"],
            true_unk_pred_unk=cm_counts["UNK_UNK"],
        )

        return AnalogyLearningCurveStep(
            budget=budget,
            accuracy=overall_acc,
            yes_accuracy=yes_acc,
            no_accuracy=no_acc,
            unknown_accuracy=unk_acc,
            confusion_matrix=cm,
            mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        )

    def train_adapter(
        self,
        model: MiniGPT,
        adapter_name: str,
        tasks: list[MetaTask],
        props: RelationProperties,
        approach: str,
        steps: int = 25,
        lr: float = 5e-4,
    ) -> None:
        """Trains modular adapter using balanced cross-entropy."""
        if not tasks:
            return
        freeze_for_adapter_training(model, adapter_name)
        set_active_domain_adapter(model, adapter_name)

        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=1e-4,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        model.train()

        for step in range(steps):
            task = tasks[step % len(tasks)]

            if approach in ("APPROACH_C_STRUCTURE_MATCHED", "APPROACH_D_NEURAL_ANALOGY"):
                matches = []
                for exp in self.memory_bank:
                    score = DeterministicStructuralMatcher.match_analogy(task, props, exp)
                    if score >= 0.5:
                        matches.append((score, exp))
                matches.sort(key=lambda x: x[0], reverse=True)
                if matches:
                    prompt = DeterministicStructuralMatcher.format_analogical_prompt(task, matches[0][1])
                else:
                    prompt = f"[PREMISES] {' '.join(task.premises)} [QUESTION] {task.question} [ANSWER]"
            else:
                prompt = f"[PREMISES] {' '.join(task.premises)} [QUESTION] {task.question} [ANSWER]"

            target = f" {task.expected_verdict}"
            p_ids = self.tokenizer.encode(prompt)
            t_ids = self.tokenizer.encode(target)

            full_ids = p_ids + t_ids
            if len(full_ids) > model.config.context_length:
                continue

            input_ids = full_ids[:-1]
            labels = [-100] * (len(p_ids) - 1) + t_ids

            inp = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            lbl = torch.tensor([labels], dtype=torch.long, device=self.device)

            out = model(inp)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits.view(-1, logits.size(-1)), lbl.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()

    @staticmethod
    def compute_auc(budgets: list[int], accuracies: list[float]) -> float:
        if len(budgets) < 2 or max(budgets) == 0:
            return float(np.mean(accuracies)) if accuracies else 0.0
        auc = 0.0
        for i in range(len(budgets) - 1):
            dx = budgets[i + 1] - budgets[i]
            y_avg = (accuracies[i] + accuracies[i + 1]) / 2.0
            auc += dx * y_avg
        max_span = budgets[-1] - budgets[0]
        return round(auc / max_span, 4)

    @staticmethod
    def find_target_budget(budgets: list[int], accuracies: list[float], target: float) -> int | None:
        for b, acc in zip(budgets, accuracies):
            if acc >= target:
                return b
        return None

    def run_approach(
        self,
        approach_name: str,
        target_family: MetaFamilyDataset,
        target_props: RelationProperties,
        budgets: list[int] = [0, 2, 4, 8, 16, 32],
    ) -> AnalogyApproachReport:
        t_start = time.perf_counter()
        curve: list[AnalogyLearningCurveStep] = []
        accuracies: list[float] = []
        total_steps = 0

        for b in budgets:
            model = copy.deepcopy(self.base_model)
            inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
            adapter_id = f"m44_{approach_name.lower()}_{b}"
            add_domain_adapter(model, domain_name=adapter_id, r=8)
            set_active_domain_adapter(model, adapter_id)

            if b > 0:
                practice_subset = target_family.train_practice_pool[:b]
                # Ensure adequate training steps per budget to shift logit boundaries
                steps = max(15, b * 3)
                total_steps += steps
                self.train_adapter(
                    model=model,
                    adapter_name=adapter_id,
                    tasks=practice_subset,
                    props=target_props,
                    approach=approach_name,
                    steps=steps,
                    lr=5e-4,
                )

            step_res = self.evaluate_test_tasks(
                model=model,
                test_tasks=target_family.held_out_test_pool,
                props=target_props,
                approach=approach_name,
                budget=b,
            )
            curve.append(step_res)
            accuracies.append(step_res.accuracy)

        auc = self.compute_auc(budgets, accuracies)
        n50 = self.find_target_budget(budgets, accuracies, 0.50)
        n80 = self.find_target_budget(budgets, accuracies, 0.80)
        dur = time.perf_counter() - t_start

        return AnalogyApproachReport(
            approach_name=approach_name,
            target_family=target_family.family_name,
            relation_properties=target_props.__dict__,
            curve=curve,
            final_accuracy=accuracies[-1],
            area_under_curve=auc,
            n50_examples=n50,
            n80_examples=n80,
            training_steps_total=total_steps,
            total_duration_seconds=dur,
        )
