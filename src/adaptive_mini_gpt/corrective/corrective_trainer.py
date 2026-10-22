"""M41 Phase 3: Controlled Learning Representation Arms.

Compares 4 experimental arms under strictly matched parameter and token budgets:
  Arm A: M40 Baseline (positive multi-hop Q->A pairs only).
  Arm B: Directional Supervision (adds reversed-negative pairs to enforce asymmetry).
  Arm C: Directional + Unknown Calibration (adds ungrounded pairs to enforce epistemic boundary).
  Arm D: Corrective Experience Learning (interactive error discovery -> targeted practice -> verification).
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW

from adaptive_mini_gpt.corrective.benchmark_suite import (
    M41_DIRECTIONAL_BENCHMARK,
    DirectionalTask,
)
from adaptive_mini_gpt.corrective.directional_world import (
    DirectionalFact,
    RelationCategory,
    TriStateGraphReasoner,
    TriStateVerdict,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.modular_adapter import (
    add_domain_adapter,
    compute_base_model_hash,
    freeze_for_adapter_training,
    inject_multi_domain_adapters,
    set_active_domain_adapter,
)
from adaptive_mini_gpt.model.transformer import MiniGPT


@dataclass
class CorrectiveArmResult:
    """Benchmark outcome for one learning representation arm."""

    arm_name: str
    description: str
    trainable_parameters: int
    parameter_growth_ratio: float
    single_hop_accuracy: float
    two_hop_accuracy: float
    three_hop_accuracy: float
    asymmetric_accuracy: float
    nontransitive_accuracy: float
    ungrounded_accuracy: float
    contradictory_accuracy: float
    counterfactual_transfer_accuracy: float
    overall_accuracy: float
    mean_inference_latency_ms: float
    base_hash_invariant: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CorrectiveTrainer:
    """Orchestrates 4-arm training and evaluation across directional tasks."""

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: BPETokenizer,
        device: torch.device,
        initial_base_hash: str | None = None,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.initial_base_hash = initial_base_hash or compute_base_model_hash(base_model)
        self.total_base_params = sum(p.numel() for p in base_model.parameters())

    def train_adapter_on_pairs(
        self,
        model: MiniGPT,
        domain_name: str,
        text_pairs: list[tuple[str, str]],
        epochs: int = 15,
        lr: float = 5e-4,
    ) -> float:
        """Fine-tune the designated domain adapter on input-target token pairs."""
        freeze_for_adapter_training(model, domain_name)
        set_active_domain_adapter(model, domain_name)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            return 0.0

        optimizer = AdamW(trainable_params, lr=lr, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        model.train()

        last_loss = 0.0
        for _ in range(epochs):
            for prompt_str, target_str in text_pairs:
                full_text = f"{prompt_str} {target_str}"
                tokens = self.tokenizer.encode(full_text)
                if len(tokens) > model.config.context_length:
                    tokens = tokens[: model.config.context_length]
                if len(tokens) < 3:
                    continue

                prompt_len = len(self.tokenizer.encode(prompt_str))
                input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=self.device)
                target_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=self.device)

                mask_limit = min(prompt_len - 1, target_ids.size(1))
                if mask_limit > 0:
                    target_ids[0, :mask_limit] = -100

                optimizer.zero_grad()
                out = model(input_ids)
                logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())

        model.eval()
        return round(last_loss, 4)

    def evaluate_task(self, model: MiniGPT, task: DirectionalTask) -> bool:
        """Evaluate whether model's greedy generation matches expected tri-state verdict."""
        prompt = f"[PREMISES] {' '.join(task.premises)} [QUESTION] {task.question} [ANSWER]"
        input_tokens = self.tokenizer.encode(prompt)
        if len(input_tokens) > model.config.context_length - 8:
            input_tokens = input_tokens[-(model.config.context_length - 8) :]

        input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=self.device)

        with torch.no_grad():
            curr_tensor = input_tensor
            generated: list[int] = []
            for _ in range(5):
                out = model(curr_tensor)
                logits = out[0] if isinstance(out, tuple) else out
                next_tok = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                generated.append(next_tok)
                if next_tok == getattr(self.tokenizer, "eos_id", 0):
                    break
                curr_tensor = torch.cat([curr_tensor, torch.tensor([[next_tok]], device=self.device)], dim=1)

        gen_text = self.tokenizer.decode(generated).strip().upper()
        exp_verdict = task.expected_verdict.strip().upper()

        if exp_verdict == "YES":
            return "YES" in gen_text and "NO" not in gen_text and "UNKNOWN" not in gen_text
        if exp_verdict == "NO":
            return "NO" in gen_text and "YES" not in gen_text and "UNKNOWN" not in gen_text
        if exp_verdict == "UNKNOWN":
            return any(w in gen_text for w in ["UNKNOWN", "ABSTAIN", "CANNOT", "UNDERDETERMINED"])
        return False

    def measure_category_accuracies(
        self,
        model: MiniGPT,
        tasks: list[DirectionalTask],
    ) -> tuple[dict[str, float], float]:
        """Compute per-category accuracies and mean latency across tasks."""
        cat_counts: dict[str, int] = {}
        cat_correct: dict[str, int] = {}

        t0 = time.perf_counter()
        for task in tasks:
            cat_counts[task.category] = cat_counts.get(task.category, 0) + 1
            if self.evaluate_task(model, task):
                cat_correct[task.category] = cat_correct.get(task.category, 0) + 1
        t1 = time.perf_counter()

        mean_latency = ((t1 - t0) / max(1, len(tasks))) * 1000.0
        accs = {
            cat: round(cat_correct.get(cat, 0) / cat_counts[cat], 4)
            for cat in cat_counts
        }
        return accs, round(mean_latency, 3)

    # -------------------------------------------------------------------------
    # Arm A: M40 Baseline Adapter (Positive Q->A pairs only)
    # -------------------------------------------------------------------------
    def evaluate_arm_a_baseline(self) -> CorrectiveArmResult:
        """Arm A: Standard adapter trained only on positive deduction pairs."""
        model = copy.deepcopy(self.base_model).to(self.device)
        inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
        add_domain_adapter(model, domain_name="arm_a_m40_baseline", r=8)

        # Standard positive training pairs
        train_pairs = [
            (f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question}", f"[ANSWER] {t.expected_verdict}")
            for t in M41_DIRECTIONAL_BENCHMARK
            if t.category in ["single_hop", "two_hop"] and t.expected_verdict == "YES" and not t.is_counterfactual
        ]

        self.train_adapter_on_pairs(model, "arm_a_m40_baseline", train_pairs, epochs=15, lr=5e-4)
        set_active_domain_adapter(model, "arm_a_m40_baseline")

        accs, latency = self.measure_category_accuracies(model, M41_DIRECTIONAL_BENCHMARK)
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        overall = round(sum(accs.values()) / max(1, len(accs)), 4)

        return CorrectiveArmResult(
            arm_name="ARM_A_M40_BASELINE",
            description="M40 Baseline Adapter (positive multi-hop Q->A pairs only)",
            trainable_parameters=trainable_p,
            parameter_growth_ratio=round(trainable_p / self.total_base_params, 6),
            single_hop_accuracy=accs.get("single_hop", 0.0),
            two_hop_accuracy=accs.get("two_hop", 0.0),
            three_hop_accuracy=accs.get("three_hop", 0.0),
            asymmetric_accuracy=accs.get("asymmetric", 0.0),
            nontransitive_accuracy=accs.get("nontransitive", 0.0),
            ungrounded_accuracy=accs.get("ungrounded", 0.0),
            contradictory_accuracy=accs.get("contradictory", 0.0),
            counterfactual_transfer_accuracy=accs.get("counterfactual", 0.0),
            overall_accuracy=overall,
            mean_inference_latency_ms=latency,
            base_hash_invariant=False,
        )

    # -------------------------------------------------------------------------
    # Arm B: Directional Supervision (adds reversed-negative pairs)
    # -------------------------------------------------------------------------
    def evaluate_arm_b_directional(self) -> CorrectiveArmResult:
        """Arm B: Adds explicit positive and reversed-negative pairs to enforce asymmetry."""
        model = copy.deepcopy(self.base_model).to(self.device)
        inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
        add_domain_adapter(model, domain_name="arm_b_directional", r=8)

        # Positive pairs + reversed-negative pairs (e.g. A in B -> B in A is NO)
        pos_pairs = [
            (f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question}", f"[ANSWER] {t.expected_verdict}")
            for t in M41_DIRECTIONAL_BENCHMARK
            if t.category in ["single_hop", "two_hop"] and t.expected_verdict == "YES" and not t.is_counterfactual
        ]
        asym_pairs = [
            (f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question}", f"[ANSWER] {t.expected_verdict}")
            for t in M41_DIRECTIONAL_BENCHMARK
            if t.category in ["asymmetric", "nontransitive"] and t.expected_verdict == "NO" and not t.is_counterfactual
        ]
        train_pairs = pos_pairs + asym_pairs

        self.train_adapter_on_pairs(model, "arm_b_directional", train_pairs, epochs=15, lr=5e-4)
        set_active_domain_adapter(model, "arm_b_directional")

        accs, latency = self.measure_category_accuracies(model, M41_DIRECTIONAL_BENCHMARK)
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        overall = round(sum(accs.values()) / max(1, len(accs)), 4)

        return CorrectiveArmResult(
            arm_name="ARM_B_DIRECTIONAL_SUPERVISION",
            description="Directional Supervision (positive + reversed-negative pairs for asymmetry)",
            trainable_parameters=trainable_p,
            parameter_growth_ratio=round(trainable_p / self.total_base_params, 6),
            single_hop_accuracy=accs.get("single_hop", 0.0),
            two_hop_accuracy=accs.get("two_hop", 0.0),
            three_hop_accuracy=accs.get("three_hop", 0.0),
            asymmetric_accuracy=accs.get("asymmetric", 0.0),
            nontransitive_accuracy=accs.get("nontransitive", 0.0),
            ungrounded_accuracy=accs.get("ungrounded", 0.0),
            contradictory_accuracy=accs.get("contradictory", 0.0),
            counterfactual_transfer_accuracy=accs.get("counterfactual", 0.0),
            overall_accuracy=overall,
            mean_inference_latency_ms=latency,
            base_hash_invariant=False,
        )

    # -------------------------------------------------------------------------
    # Arm C: Directional + Unknown Calibration
    # -------------------------------------------------------------------------
    def evaluate_arm_c_unknown_calibrated(self) -> CorrectiveArmResult:
        """Arm C: Directional training plus explicit ungrounded unknown targets."""
        model = copy.deepcopy(self.base_model).to(self.device)
        inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
        add_domain_adapter(model, domain_name="arm_c_unknown_calib", r=8)

        pos_pairs = [
            (f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question}", f"[ANSWER] {t.expected_verdict}")
            for t in M41_DIRECTIONAL_BENCHMARK
            if t.category in ["single_hop", "two_hop"] and t.expected_verdict == "YES" and not t.is_counterfactual
        ]
        asym_pairs = [
            (f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question}", f"[ANSWER] {t.expected_verdict}")
            for t in M41_DIRECTIONAL_BENCHMARK
            if t.category in ["asymmetric", "nontransitive"] and t.expected_verdict == "NO" and not t.is_counterfactual
        ]
        unknown_pairs = [
            (f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question}", f"[ANSWER] {t.expected_verdict}")
            for t in M41_DIRECTIONAL_BENCHMARK
            if t.category in ["ungrounded", "contradictory"] and t.expected_verdict == "UNKNOWN" and not t.is_counterfactual
        ]
        train_pairs = pos_pairs + asym_pairs + unknown_pairs

        self.train_adapter_on_pairs(model, "arm_c_unknown_calib", train_pairs, epochs=15, lr=5e-4)
        set_active_domain_adapter(model, "arm_c_unknown_calib")

        accs, latency = self.measure_category_accuracies(model, M41_DIRECTIONAL_BENCHMARK)
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        overall = round(sum(accs.values()) / max(1, len(accs)), 4)

        return CorrectiveArmResult(
            arm_name="ARM_C_DIRECTIONAL_UNKNOWN_CALIBRATED",
            description="Directional + Unknown Calibrated (YES, NO, and UNKNOWN supervision)",
            trainable_parameters=trainable_p,
            parameter_growth_ratio=round(trainable_p / self.total_base_params, 6),
            single_hop_accuracy=accs.get("single_hop", 0.0),
            two_hop_accuracy=accs.get("two_hop", 0.0),
            three_hop_accuracy=accs.get("three_hop", 0.0),
            asymmetric_accuracy=accs.get("asymmetric", 0.0),
            nontransitive_accuracy=accs.get("nontransitive", 0.0),
            ungrounded_accuracy=accs.get("ungrounded", 0.0),
            contradictory_accuracy=accs.get("contradictory", 0.0),
            counterfactual_transfer_accuracy=accs.get("counterfactual", 0.0),
            overall_accuracy=overall,
            mean_inference_latency_ms=latency,
            base_hash_invariant=False,
        )

    # -------------------------------------------------------------------------
    # Arm D: Corrective Feedback Sequence & Practice
    # -------------------------------------------------------------------------
    def evaluate_arm_d_corrective_learning(self) -> tuple[CorrectiveArmResult, dict[str, Any]]:
        """Arm D: Interactive corrective experience learning (discover errors -> practice -> evaluate)."""
        model = copy.deepcopy(self.base_model).to(self.device)
        inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
        add_domain_adapter(model, domain_name="arm_d_corrective", r=8)

        # 1. Initial base training on standard positive pairs
        initial_pairs = [
            (f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question}", f"[ANSWER] {t.expected_verdict}")
            for t in M41_DIRECTIONAL_BENCHMARK
            if t.category in ["single_hop", "two_hop"] and t.expected_verdict == "YES" and not t.is_counterfactual
        ]
        self.train_adapter_on_pairs(model, "arm_d_corrective", initial_pairs, epochs=10, lr=5e-4)
        set_active_domain_adapter(model, "arm_d_corrective")

        # 2. Interactive Error Discovery: Probe model on diagnostic probe tasks
        probe_tasks = [
            t for t in M41_DIRECTIONAL_BENCHMARK
            if t.category in ["asymmetric", "ungrounded", "nontransitive"] and not t.is_counterfactual
        ]
        discovered_errors: list[tuple[DirectionalTask, str]] = []
        for pt in probe_tasks:
            if not self.evaluate_task(model, pt):
                discovered_errors.append((pt, pt.expected_verdict))

        # 3. Categorize errors into targeted practice curriculum
        targeted_practice_pairs: list[tuple[str, str]] = []
        error_categories: dict[str, int] = {}
        for err_task, exp in discovered_errors:
            cat = err_task.category
            error_categories[cat] = error_categories.get(cat, 0) + 1
            prompt_str = f"[PREMISES] {' '.join(err_task.premises)} [QUESTION] {err_task.question}"
            targeted_practice_pairs.append((prompt_str, f"[ANSWER] {exp}"))

        # 4. Corrective Practice Training (fine-tuning on discovered mistakes + replay)
        combined_corrective_pairs = targeted_practice_pairs + initial_pairs
        self.train_adapter_on_pairs(model, "arm_d_corrective", combined_corrective_pairs, epochs=10, lr=3e-4)
        set_active_domain_adapter(model, "arm_d_corrective")

        # 5. Full Held-Out Benchmark Evaluation
        accs, latency = self.measure_category_accuracies(model, M41_DIRECTIONAL_BENCHMARK)
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        overall = round(sum(accs.values()) / max(1, len(accs)), 4)

        corrective_summary = {
            "initial_probed_tasks": len(probe_tasks),
            "discovered_mistakes_count": len(discovered_errors),
            "error_categories": error_categories,
            "targeted_practice_examples_applied": len(targeted_practice_pairs),
            "post_correction_accuracy": overall,
        }

        arm_res = CorrectiveArmResult(
            arm_name="ARM_D_CORRECTIVE_LEARNING",
            description="Corrective Experience Learning (Interactive Error Discovery -> Practice -> Retain)",
            trainable_parameters=trainable_p,
            parameter_growth_ratio=round(trainable_p / self.total_base_params, 6),
            single_hop_accuracy=accs.get("single_hop", 0.0),
            two_hop_accuracy=accs.get("two_hop", 0.0),
            three_hop_accuracy=accs.get("three_hop", 0.0),
            asymmetric_accuracy=accs.get("asymmetric", 0.0),
            nontransitive_accuracy=accs.get("nontransitive", 0.0),
            ungrounded_accuracy=accs.get("ungrounded", 0.0),
            contradictory_accuracy=accs.get("contradictory", 0.0),
            counterfactual_transfer_accuracy=accs.get("counterfactual", 0.0),
            overall_accuracy=overall,
            mean_inference_latency_ms=latency,
            base_hash_invariant=False,
        )

        return arm_res, corrective_summary
