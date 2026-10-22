"""Phase 4 & 5: Controlled Learning Experiments and Sample Efficiency Benchmarks.

Compares four primary approaches across sample budgets N in [0, 2, 4, 8, 16, 32]:
  1. Fresh adaptation with no memory (baseline neural few-shot learning).
  2. Memory-assisted inference with frozen weights (external retrieval benefit without weight updates).
  3. Memory-assisted training followed by neural-only evaluation (tests if memory improves neural weights).
  4. Selective memory plus memory-assisted training (selective prioritization, tested both neural-only and memory-assisted).
Plus baselines:
  - Full-memory baseline (uncurated candidate retrieval).
  - Recency-only baseline.

Enforces strict evaluation isolation: retrieved exemplars and test labels NEVER leak into neural-only test prompts.
Computes Learning Curve Area Under Curve (AUC), N50, N80 budgets, tri-state accuracy, and improvement per compute step.
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

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.memory.episodic_memory import EpisodicMemoryBank, LearningExperience
from adaptive_mini_gpt.memory.retrieval import MemoryRetriever, RetrievalScoringWeights
from adaptive_mini_gpt.meta.benchmark_suite import MetaFamilyDataset, MetaTask
from adaptive_mini_gpt.model.modular_adapter import (
    add_domain_adapter,
    freeze_for_adapter_training,
    inject_multi_domain_adapters,
    set_active_domain_adapter,
)
from adaptive_mini_gpt.model.transformer import MiniGPT

logger = logging.getLogger("m43_learning_experiments")


@dataclass
class LearningCurveStep:
    budget: int
    accuracy: float
    yes_accuracy: float
    no_accuracy: float
    unknown_accuracy: float
    mean_latency_ms: float


@dataclass
class ApproachEvaluationReport:
    approach_name: str
    target_family: str
    evaluation_mode: str  # "neural_only" or "memory_assisted"
    curve: list[LearningCurveStep]
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
            "evaluation_mode": self.evaluation_mode,
            "curve": [c.__dict__ for c in self.curve],
            "final_accuracy": self.final_accuracy,
            "area_under_curve": self.area_under_curve,
            "n50_examples": self.n50_examples,
            "n80_examples": self.n80_examples,
            "training_steps_total": self.training_steps_total,
            "total_duration_seconds": self.total_duration_seconds,
        }


def build_m43_task_family(
    family_name: str,
    split: str,
    rel: str,
    inv_rel: str,
    entities: list[str],
    is_transitive: bool = True,
    is_symmetric: bool = False,
) -> MetaFamilyDataset:
    """Constructs unfamiliar relational task family with disjoint entity and relation structures."""
    e0, e1, e2, e3 = entities[0], entities[1], entities[2], entities[3]

    base_premises = [
        f"{e0} {rel} {e1}",
        f"{e1} {rel} {e2}",
        f"{e2} {rel} {e3}",
    ]

    practice_tasks: list[MetaTask] = []
    # 1. 1-hop
    practice_tasks.append(MetaTask(f"{family_name}_pr_1", family_name, split, base_premises, f"Does {e0} {rel} {e1}?", "YES", "single_hop", [e0, e1]))
    practice_tasks.append(MetaTask(f"{family_name}_pr_2", family_name, split, base_premises, f"Does {e1} {rel} {e2}?", "YES", "single_hop", [e1, e2]))
    # 2. 2-hop
    practice_tasks.append(MetaTask(f"{family_name}_pr_3", family_name, split, base_premises, f"Does {e0} {rel} {e2}?", "YES" if is_transitive else "NO", "two_hop", [e0, e2]))
    # 3. Asymmetric
    practice_tasks.append(MetaTask(f"{family_name}_pr_4", family_name, split, base_premises, f"Does {e1} {rel} {e0}?", "YES" if is_symmetric else "NO", "asymmetric", [e1, e0]))
    practice_tasks.append(MetaTask(f"{family_name}_pr_5", family_name, split, base_premises, f"Does {e2} {rel} {e1}?", "YES" if is_symmetric else "NO", "asymmetric", [e2, e1]))
    # 4. Ungrounded
    practice_tasks.append(MetaTask(f"{family_name}_pr_6", family_name, split, base_premises, f"Does {e0} {rel} NovelVoidX?", "UNKNOWN", "ungrounded", [e0, "NovelVoidX"]))
    # 5. 3-hop
    practice_tasks.append(MetaTask(f"{family_name}_pr_7", family_name, split, base_premises, f"Does {e0} {rel} {e3}?", "YES" if is_transitive else "NO", "three_hop", [e0, e3]))
    # 6. Reversal 2-hop
    practice_tasks.append(MetaTask(f"{family_name}_pr_8", family_name, split, base_premises, f"Does {e2} {rel} {e0}?", "YES" if is_symmetric else "NO", "asymmetric", [e2, e0]))

    for i in range(9, 33):
        idx = i % 8
        base_t = practice_tasks[idx]
        practice_tasks.append(MetaTask(
            task_id=f"{family_name}_pr_{i}",
            family=family_name,
            split=split,
            premises=base_t.premises,
            question=f"Is it verified that {base_t.question.replace('Does ', '').replace('?', '')}?",
            expected_verdict=base_t.expected_verdict,
            reasoning_type=base_t.reasoning_type,
            entities_involved=base_t.entities_involved,
        ))

    # Held-out test pool (20 balanced tasks across all reasoning modalities)
    test_tasks: list[MetaTask] = []
    test_tasks.append(MetaTask(f"{family_name}_te_1", family_name, split, base_premises, f"Does {e0} {rel} {e1}?", "YES", "single_hop", [e0, e1]))
    test_tasks.append(MetaTask(f"{family_name}_te_2", family_name, split, base_premises, f"Does {e2} {rel} {e3}?", "YES", "single_hop", [e2, e3]))
    test_tasks.append(MetaTask(f"{family_name}_te_3", family_name, split, base_premises, f"Does {e0} {rel} {e2}?", "YES" if is_transitive else "NO", "two_hop", [e0, e2]))
    test_tasks.append(MetaTask(f"{family_name}_te_4", family_name, split, base_premises, f"Does {e1} {rel} {e3}?", "YES" if is_transitive else "NO", "two_hop", [e1, e3]))
    test_tasks.append(MetaTask(f"{family_name}_te_5", family_name, split, base_premises, f"Does {e0} {rel} {e3}?", "YES" if is_transitive else "NO", "three_hop", [e0, e3]))
    test_tasks.append(MetaTask(f"{family_name}_te_6", family_name, split, base_premises, f"Does {e1} {rel} {e0}?", "YES" if is_symmetric else "NO", "asymmetric", [e1, e0]))
    test_tasks.append(MetaTask(f"{family_name}_te_7", family_name, split, base_premises, f"Does {e2} {rel} {e1}?", "YES" if is_symmetric else "NO", "asymmetric", [e2, e1]))
    test_tasks.append(MetaTask(f"{family_name}_te_8", family_name, split, base_premises, f"Does {e3} {rel} {e0}?", "YES" if is_symmetric else "NO", "asymmetric", [e3, e0]))
    test_tasks.append(MetaTask(f"{family_name}_te_9", family_name, split, base_premises, f"Does {e3} {rel} {e2}?", "YES" if is_symmetric else "NO", "nontransitive", [e3, e2]))
    test_tasks.append(MetaTask(f"{family_name}_te_10", family_name, split, base_premises, f"Does {e2} {rel} {e0}?", "YES" if is_symmetric else "NO", "nontransitive", [e2, e0]))
    test_tasks.append(MetaTask(f"{family_name}_te_11", family_name, split, base_premises, f"Does {e0} {rel} UnseenParticle?", "UNKNOWN", "ungrounded", [e0, "UnseenParticle"]))
    test_tasks.append(MetaTask(f"{family_name}_te_12", family_name, split, base_premises, f"Does AlienSpecimen {rel} {e1}?", "UNKNOWN", "ungrounded", ["AlienSpecimen", e1]))
    test_tasks.append(MetaTask(f"{family_name}_te_13", family_name, split, base_premises, f"Does UnknownP {rel} UnknownQ?", "UNKNOWN", "ungrounded", ["UnknownP", "UnknownQ"]))
    test_tasks.append(MetaTask(f"{family_name}_te_14", family_name, split, base_premises, f"Does {e0} {rel} {e0}?", "UNKNOWN", "contradictory", [e0, e0]))
    test_tasks.append(MetaTask(f"{family_name}_te_15", family_name, split, base_premises, f"Does {e2} {rel} {e2}?", "UNKNOWN", "contradictory", [e2, e2]))

    # Counterfactual permuted test tasks
    cf_premises = [f"{e3} {rel} {e2}", f"{e2} {rel} {e1}", f"{e1} {rel} {e0}"]
    test_tasks.append(MetaTask(f"{family_name}_te_16", family_name, split, cf_premises, f"Does {e3} {rel} {e1}?", "YES" if is_transitive else "NO", "two_hop", [e3, e1]))
    test_tasks.append(MetaTask(f"{family_name}_te_17", family_name, split, cf_premises, f"Does {e0} {rel} {e3}?", "YES" if is_symmetric else "NO", "asymmetric", [e0, e3]))
    test_tasks.append(MetaTask(f"{family_name}_te_18", family_name, split, cf_premises, f"Does {e3} {rel} {e0}?", "YES" if is_transitive else "NO", "three_hop", [e3, e0]))
    test_tasks.append(MetaTask(f"{family_name}_te_19", family_name, split, cf_premises, f"Does {e1} {rel} GhostElement?", "UNKNOWN", "ungrounded", [e1, "GhostElement"]))
    test_tasks.append(MetaTask(f"{family_name}_te_20", family_name, split, cf_premises, f"Does {e1} {rel} {e2}?", "YES" if is_symmetric else "NO", "asymmetric", [e1, e2]))

    return MetaFamilyDataset(
        family_name=family_name,
        split=split,
        relation_name=rel,
        inverse_relation=inv_rel,
        is_transitive=is_transitive,
        is_symmetric=is_symmetric,
        train_practice_pool=practice_tasks,
        held_out_test_pool=test_tasks,
    )


def get_m43_benchmark_families() -> dict[str, MetaFamilyDataset]:
    """Instantiate novel, unfamiliar relational task families for M43."""
    families: dict[str, MetaFamilyDataset] = {}

    # 1. Experience/Prior Families (for accumulating memories)
    families["metabolic_cascade"] = build_m43_task_family(
        family_name="metabolic_cascade",
        split="meta_train",
        rel="catalyzes",
        inv_rel="inhibited_by",
        entities=["EnzymeAlpha", "SubstrateBeta", "ProductGamma", "MetaboliteDelta"],
        is_transitive=True,
        is_symmetric=False,
    )
    families["quantum_entanglement"] = build_m43_task_family(
        family_name="quantum_entanglement",
        split="meta_train",
        rel="entangled_with",
        inv_rel="entangled_with",
        entities=["Qubit0", "Qubit1", "Qubit2", "Qubit3"],
        is_transitive=True,
        is_symmetric=True,
    )
    families["cryptographic_trust"] = build_m43_task_family(
        family_name="cryptographic_trust",
        split="meta_train",
        rel="endorses",
        inv_rel="endorsed_by",
        entities=["RootCert", "IntermediateCA", "SignerNode", "LeafDevice"],
        is_transitive=True,
        is_symmetric=False,
    )

    # 2. Development Family
    families["orbital_mechanics"] = build_m43_task_family(
        family_name="orbital_mechanics",
        split="dev",
        rel="orbits_around",
        inv_rel="orbited_by",
        entities=["MoonLuna", "PlanetTerra", "StarSol", "CoreGalaxy"],
        is_transitive=True,
        is_symmetric=False,
    )

    # 3. Sealed Meta-Test Families (completely unseen entities and rules)
    families["linguistic_etymology"] = build_m43_task_family(
        family_name="linguistic_etymology",
        split="meta_test",
        rel="derives_from",
        inv_rel="gives_rise_to",
        entities=["ProtoRootX", "ArchaicFormY", "DialectZ", "ModernLexemeW"],
        is_transitive=True,
        is_symmetric=False,
    )
    families["stellar_nucleosynthesis"] = build_m43_task_family(
        family_name="stellar_nucleosynthesis",
        split="meta_test",
        rel="fuses_into",
        inv_rel="fission_product_of",
        entities=["HydrogenIsotope", "HeliumCore", "CarbonPlasma", "IronPeak"],
        is_transitive=True,
        is_symmetric=False,
    )

    return families


class ControlledLearningExperimentRunner:
    """Orchestrates sample efficiency benchmarks across the 4 learning approaches."""

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: BPETokenizer,
        device: torch.device,
        memory_bank: EpisodicMemoryBank,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.memory_bank = memory_bank
        self.retriever = MemoryRetriever(memory_bank)
        self.classes = ["YES", "NO", "UNKNOWN"]
        self.cand_seqs = [self.tokenizer.encode(f" {c}") for c in self.classes]

    def _eval_single_task(
        self,
        model: MiniGPT,
        task: MetaTask,
        use_memory: bool = False,
        retrieval_method: str = "selective",
    ) -> tuple[bool, str]:
        """Evaluates a task either neural-only or memory-assisted."""
        premises = task.premises
        query = task.question

        if use_memory:
            if retrieval_method == "selective":
                exemplars = self.retriever.retrieve_selective(query, premises)
            elif retrieval_method == "keyword":
                exemplars = self.retriever.retrieve_keyword(query, premises)
            elif retrieval_method == "recency":
                exemplars = self.retriever.retrieve_recency_baseline(query, premises)
            elif retrieval_method == "full":
                exemplars = self.retriever.retrieve_full_baseline(query, premises)
            else:
                exemplars = []
            prompt = self.retriever.format_in_context_prompt(query, premises, exemplars)
        else:
            prompt = f"[PREMISES] {' '.join(premises)} [QUESTION] {query} [ANSWER]"

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

    def evaluate_test_set(
        self,
        model: MiniGPT,
        test_tasks: list[MetaTask],
        use_memory: bool = False,
        retrieval_method: str = "selective",
    ) -> tuple[float, float, float, float, float]:
        """Returns (overall_acc, yes_acc, no_acc, unk_acc, mean_latency_ms)."""
        model.eval()
        yes_c = yes_t = 0
        no_c = no_t = 0
        unk_c = unk_t = 0
        latencies: list[float] = []

        for t in test_tasks:
            t0 = time.perf_counter()
            corr, pred = self._eval_single_task(model, t, use_memory=use_memory, retrieval_method=retrieval_method)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

            if t.expected_verdict == "YES":
                yes_t += 1
                if corr:
                    yes_c += 1
            elif t.expected_verdict == "NO":
                no_t += 1
                if corr:
                    no_c += 1
            else:
                unk_t += 1
                if corr:
                    unk_c += 1

        tot = max(1, len(test_tasks))
        overall = (yes_c + no_c + unk_c) / tot
        yes_acc = yes_c / max(1, yes_t)
        no_acc = no_c / max(1, no_t)
        unk_acc = unk_c / max(1, unk_t)
        mean_lat = float(np.mean(latencies)) if latencies else 0.0

        return (
            round(overall, 4),
            round(yes_acc, 4),
            round(no_acc, 4),
            round(unk_acc, 4),
            round(mean_lat, 2),
        )

    def train_adapter_on_tasks(
        self,
        model: MiniGPT,
        adapter_name: str,
        tasks: list[MetaTask],
        steps: int = 15,
        lr: float = 5e-4,
    ) -> None:
        """Fine-tunes the modular adapter using standard causal cross-entropy."""
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
            prompt = f"[PREMISES] {' '.join(task.premises)} [QUESTION] {task.question} [ANSWER]"
            target = f" {task.expected_verdict}"

            prompt_ids = self.tokenizer.encode(prompt)
            target_ids = self.tokenizer.encode(target)

            full_ids = prompt_ids + target_ids
            if len(full_ids) > model.config.context_length:
                continue

            input_ids = full_ids[:-1]
            labels = [-100] * (len(prompt_ids) - 1) + target_ids

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
        """Computes Area Under Learning Curve normalized by max budget span."""
        if len(budgets) < 2 or max(budgets) == 0:
            return float(np.mean(accuracies)) if accuracies else 0.0
        # Trapezoidal rule over [0, 32]
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

    # -------------------------------------------------------------------------
    # Core Benchmark Runner
    # -------------------------------------------------------------------------

    def run_approach_1_fresh(
        self,
        target_family: MetaFamilyDataset,
        budgets: list[int] = [0, 2, 4, 8, 16, 32],
    ) -> ApproachEvaluationReport:
        """Approach 1: Fresh adaptation with no memory."""
        t_start = time.perf_counter()
        curve: list[LearningCurveStep] = []
        accuracies: list[float] = []

        for b in budgets:
            model = copy.deepcopy(self.base_model)
            inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
            add_domain_adapter(model, domain_name="fresh_adapter", r=8)
            set_active_domain_adapter(model, "fresh_adapter")

            if b > 0:
                practice_subset = target_family.train_practice_pool[:b]
                self.train_adapter_on_tasks(model, "fresh_adapter", practice_subset, steps=b * 2)

            acc, y_acc, n_acc, u_acc, lat = self.evaluate_test_set(
                model, target_family.held_out_test_pool, use_memory=False
            )
            curve.append(LearningCurveStep(b, acc, y_acc, n_acc, u_acc, lat))
            accuracies.append(acc)

        auc = self.compute_auc(budgets, accuracies)
        n50 = self.find_target_budget(budgets, accuracies, 0.50)
        n80 = self.find_target_budget(budgets, accuracies, 0.80)
        duration = round(time.perf_counter() - t_start, 2)

        return ApproachEvaluationReport(
            approach_name="APPROACH_1_FRESH_NO_MEMORY",
            target_family=target_family.family_name,
            evaluation_mode="neural_only",
            curve=curve,
            final_accuracy=accuracies[-1],
            area_under_curve=auc,
            n50_examples=n50,
            n80_examples=n80,
            training_steps_total=sum(b * 2 for b in budgets),
            total_duration_seconds=duration,
        )

    def run_approach_2_memory_inference(
        self,
        target_family: MetaFamilyDataset,
        retrieval_method: str = "selective",
        budgets: list[int] = [0, 2, 4, 8, 16, 32],
    ) -> ApproachEvaluationReport:
        """Approach 2: Memory-assisted inference with frozen weights (zero weight updates)."""
        t_start = time.perf_counter()
        curve: list[LearningCurveStep] = []
        accuracies: list[float] = []

        model = copy.deepcopy(self.base_model)
        inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
        add_domain_adapter(model, domain_name="frozen_adapter", r=8)
        set_active_domain_adapter(model, "frozen_adapter")

        # In this approach, weights remain frozen, but memory is enabled
        acc, y_acc, n_acc, u_acc, lat = self.evaluate_test_set(
            model,
            target_family.held_out_test_pool,
            use_memory=True,
            retrieval_method=retrieval_method,
        )

        for b in budgets:
            curve.append(LearningCurveStep(b, acc, y_acc, n_acc, u_acc, lat))
            accuracies.append(acc)

        auc = self.compute_auc(budgets, accuracies)
        n50 = self.find_target_budget(budgets, accuracies, 0.50)
        n80 = self.find_target_budget(budgets, accuracies, 0.80)
        duration = round(time.perf_counter() - t_start, 2)

        return ApproachEvaluationReport(
            approach_name=f"APPROACH_2_MEMORY_INFERENCE_FROZEN_{retrieval_method.upper()}",
            target_family=target_family.family_name,
            evaluation_mode="memory_assisted",
            curve=curve,
            final_accuracy=accuracies[-1],
            area_under_curve=auc,
            n50_examples=n50,
            n80_examples=n80,
            training_steps_total=0,
            total_duration_seconds=duration,
        )

    def run_approach_3_memory_assisted_training(
        self,
        target_family: MetaFamilyDataset,
        budgets: list[int] = [0, 2, 4, 8, 16, 32],
    ) -> ApproachEvaluationReport:
        """Approach 3: Memory-assisted training followed by NEURAL-ONLY evaluation (memory disabled)."""
        t_start = time.perf_counter()
        curve: list[LearningCurveStep] = []
        accuracies: list[float] = []

        for b in budgets:
            model = copy.deepcopy(self.base_model)
            inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
            add_domain_adapter(model, domain_name="memory_trained_adapter", r=8)
            set_active_domain_adapter(model, "memory_trained_adapter")

            if b > 0:
                # Augmented practice pool: Combine fresh practice with verified memory exemplars
                raw_practice = target_family.train_practice_pool[:b]
                memory_practice: list[MetaTask] = []
                for p in raw_practice:
                    retrieved = self.retriever.retrieve_selective(p.question, p.premises)
                    for r in retrieved:
                        memory_practice.append(MetaTask(
                            task_id=f"mem_{r.task_id}",
                            family=r.family,
                            split="memory_prior",
                            premises=r.premises,
                            question=r.question,
                            expected_verdict=r.verified_answer,
                            reasoning_type=r.error_category,
                            entities_involved=[],
                        ))
                combined_practice = raw_practice + memory_practice[:b]
                self.train_adapter_on_tasks(model, "memory_trained_adapter", combined_practice, steps=b * 2)

            # CRITICAL EVALUATION REQUIREMENT:
            # Memory is STRICTLY DISABLED during testing to evaluate actual neural weight acquisition!
            acc, y_acc, n_acc, u_acc, lat = self.evaluate_test_set(
                model, target_family.held_out_test_pool, use_memory=False
            )
            curve.append(LearningCurveStep(b, acc, y_acc, n_acc, u_acc, lat))
            accuracies.append(acc)

        auc = self.compute_auc(budgets, accuracies)
        n50 = self.find_target_budget(budgets, accuracies, 0.50)
        n80 = self.find_target_budget(budgets, accuracies, 0.80)
        duration = round(time.perf_counter() - t_start, 2)

        return ApproachEvaluationReport(
            approach_name="APPROACH_3_MEMORY_TRAINED_NEURAL_ONLY",
            target_family=target_family.family_name,
            evaluation_mode="neural_only",
            curve=curve,
            final_accuracy=accuracies[-1],
            area_under_curve=auc,
            n50_examples=n50,
            n80_examples=n80,
            training_steps_total=sum(b * 2 for b in budgets),
            total_duration_seconds=duration,
        )

    def run_approach_4_selective_memory_combined(
        self,
        target_family: MetaFamilyDataset,
        budgets: list[int] = [0, 2, 4, 8, 16, 32],
    ) -> tuple[ApproachEvaluationReport, ApproachEvaluationReport]:
        """Approach 4: Selective memory training evaluated both Neural-Only and Memory-Assisted."""
        t_start = time.perf_counter()
        curve_neural_only: list[LearningCurveStep] = []
        curve_memory_assisted: list[LearningCurveStep] = []
        accs_neural: list[float] = []
        accs_assisted: list[float] = []

        for b in budgets:
            model = copy.deepcopy(self.base_model)
            inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
            add_domain_adapter(model, domain_name="selective_adapter", r=8)
            set_active_domain_adapter(model, "selective_adapter")

            if b > 0:
                raw_practice = target_family.train_practice_pool[:b]
                selective_exemplars: list[MetaTask] = []
                for p in raw_practice:
                    # Selective multi-factor filtering: high relevance, high usefulness, active status
                    retrieved = self.retriever.retrieve_selective(p.question, p.premises)
                    for r in retrieved:
                        if r.usefulness_score >= 0.5 and not r.is_superseded:
                            selective_exemplars.append(MetaTask(
                                task_id=f"sel_{r.task_id}",
                                family=r.family,
                                split="selective_prior",
                                premises=r.premises,
                                question=r.question,
                                expected_verdict=r.verified_answer,
                                reasoning_type=r.error_category,
                                entities_involved=[],
                            ))
                combined = raw_practice + selective_exemplars[:b]
                self.train_adapter_on_tasks(model, "selective_adapter", combined, steps=b * 2)

            # Evaluate neural-only
            acc_n, y_n, no_n, u_n, lat_n = self.evaluate_test_set(
                model, target_family.held_out_test_pool, use_memory=False
            )
            curve_neural_only.append(LearningCurveStep(b, acc_n, y_n, no_n, u_n, lat_n))
            accs_neural.append(acc_n)

            # Evaluate memory-assisted
            acc_a, y_a, no_a, u_a, lat_a = self.evaluate_test_set(
                model, target_family.held_out_test_pool, use_memory=True, retrieval_method="selective"
            )
            curve_memory_assisted.append(LearningCurveStep(b, acc_a, y_a, no_a, u_a, lat_a))
            accs_assisted.append(acc_a)

        duration = round(time.perf_counter() - t_start, 2)
        rep_neural = ApproachEvaluationReport(
            approach_name="APPROACH_4_SELECTIVE_MEMORY_NEURAL_ONLY",
            target_family=target_family.family_name,
            evaluation_mode="neural_only",
            curve=curve_neural_only,
            final_accuracy=accs_neural[-1],
            area_under_curve=self.compute_auc(budgets, accs_neural),
            n50_examples=self.find_target_budget(budgets, accs_neural, 0.50),
            n80_examples=self.find_target_budget(budgets, accs_neural, 0.80),
            training_steps_total=sum(b * 2 for b in budgets),
            total_duration_seconds=duration,
        )
        rep_assisted = ApproachEvaluationReport(
            approach_name="APPROACH_4_SELECTIVE_MEMORY_ASSISTED",
            target_family=target_family.family_name,
            evaluation_mode="memory_assisted",
            curve=curve_memory_assisted,
            final_accuracy=accs_assisted[-1],
            area_under_curve=self.compute_auc(budgets, accs_assisted),
            n50_examples=self.find_target_budget(budgets, accs_assisted, 0.50),
            n80_examples=self.find_target_budget(budgets, accs_assisted, 0.80),
            training_steps_total=sum(b * 2 for b in budgets),
            total_duration_seconds=duration,
        )
        return rep_neural, rep_assisted
