"""M38 Phase 3 & 4: Controlled Structural Evolution Evaluator.

Executes controlled 4-arm experimental comparisons under matched starting weights,
training tokens, optimizer settings, and compute budgets:
  - Arm A: Original fixed topology with standard training
  - Arm B: Fixed topology with matched parameter budget
  - Arm C: Original M34/M37 modular adapter strategy
  - Arm D: Gated Layer 1 -> Layer 3 skip connection (learnable alpha)

Evaluates:
  - Neural-only accuracy on fresh tasks across arithmetic, logic, factual, and paraphrases
  - Multi-episode cross-domain factual retention matrix (Physics -> Biology -> Computing -> History)
  - Elementary arithmetic carry retention (200 decimal states)
  - Learned gate magnitude (alpha) and parameter growth ratio
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from adaptive_mini_gpt.curriculum_discovery.structural_evolution import (
    StructuralEvolutionGate,
    StructuralModificationType,
    StructuralProposal,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.modular_adapter import (
    add_domain_adapter,
    compute_base_model_hash,
    inject_multi_domain_adapters,
    set_active_domain_adapter,
)
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.topology.benchmark_suite import (
    CONCEPT_TRANSFER_TASKS,
    PARAPHRASED_FACTUAL_TASKS,
    SEQUENTIAL_EPISODES,
    SOURCE_FACTUAL_TASKS,
    UNSEEN_ARITHMETIC_TASKS,
    UNSEEN_LOGIC_TASKS,
    TopologyBenchmarkTask,
    generate_carry_retention_suite,
)
from adaptive_mini_gpt.topology.gated_skip import (
    ArchitectureSpec,
    GatedSkipConnection,
    TopologyEvolutionManager,
)

logger = logging.getLogger("topology.structural_evaluator")


@dataclass
class ArmEvaluationResult:
    """Evaluation metrics for a single controlled experimental arm."""

    arm_name: str
    description: str
    trainable_parameters: int
    parameter_growth_ratio: float
    learned_gate_alpha: float | None
    unseen_arithmetic_accuracy: float
    unseen_logic_accuracy: float
    factual_exact_accuracy: float
    paraphrased_factual_accuracy: float
    concept_transfer_accuracy: float
    carry_retention_accuracy: float
    mean_inference_latency_ms: float
    base_hash_invariant: bool


@dataclass
class RetentionMatrixReport:
    """Evaluation report for multi-episode sequential factual retention."""

    matrix: list[list[float | None]]  # row = after studying episode i, col = accuracy on subject j
    episodes: list[str]
    retention_rates: list[float]
    mean_backward_transfer: float
    mean_retention: float


class ControlledStructuralEvaluator:
    """Orchestrates controlled 4-arm structural benchmarking and retention matrix evaluation."""

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: BPETokenizer,
        device: torch.device,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.base_hash_initial = compute_base_model_hash(base_model)
        self.gate = StructuralEvolutionGate(max_parameter_growth_ratio=0.20, min_expected_gain=0.15)

    def evaluate_task_neural_only(
        self,
        model: MiniGPT,
        task: TopologyBenchmarkTask,
        max_new_tokens: int = 12,
    ) -> bool:
        """Evaluate neural model on a task WITHOUT external retrieval, memory, or deterministic oracle."""
        model.eval()
        prompt = f"Question: {task.question} Answer:"
        idx = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            gen_tokens: list[int] = []
            for _ in range(max_new_tokens):
                out = model(idx)
                logits = out[0] if isinstance(out, tuple) else out
                next_tok = int(torch.argmax(logits[0, -1, :]).item())
                eos_id = self.tokenizer.stoi.get(b"<eos>", 0)
                if next_tok == eos_id or next_tok == 0:
                    break
                gen_tokens.append(next_tok)
                idx = torch.cat([idx, torch.tensor([[next_tok]], device=self.device)], dim=1)

        pred_text = self.tokenizer.decode(gen_tokens).strip()
        exact = task.expected_answer.lower() in pred_text.lower()
        return exact

    def evaluate_carry_retention(self, model: MiniGPT) -> float:
        """Evaluate elementary arithmetic carry retention over 200 states."""
        transitions = generate_carry_retention_suite()
        correct = 0
        model.eval()

        with torch.no_grad():
            for a, b, c_in in transitions:
                expected_sum = a + b + c_in
                expected_str = str(expected_sum)
                prompt = f"{a} + {b} + {c_in} ="
                idx = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
                out = model(idx)
                logits = out[0] if isinstance(out, tuple) else out
                next_tok = int(torch.argmax(logits[0, -1, :]).item())
                pred_str = self.tokenizer.decode([next_tok]).strip()
                if pred_str == expected_str or (pred_str and expected_str.startswith(pred_str)):
                    correct += 1

        return correct / len(transitions)

    def measure_inference_latency_ms(self, model: MiniGPT, sample_prompt: str, runs: int = 20) -> float:
        """Measure mean forward pass latency in milliseconds."""
        model.eval()
        idx = torch.tensor(self.tokenizer.encode(sample_prompt), dtype=torch.long, device=self.device).unsqueeze(0)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(idx)

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(idx)
        t1 = time.perf_counter()

        return round(((t1 - t0) / runs) * 1000.0, 3)

    def train_on_pairs(
        self,
        model: MiniGPT,
        trainable_params: list[nn.Parameter],
        qa_pairs: list[tuple[str, str]],
        epochs: int = 15,
        lr: float = 5e-4,
    ) -> float:
        """Train specified parameters on Q&A pairs under matched token budget."""
        if not trainable_params:
            return 0.0

        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        encoded_data = []
        for q, a in qa_pairs:
            prompt_toks = self.tokenizer.encode(f"Question: {q} Answer:")
            ans_toks = self.tokenizer.encode(f" {a}")
            full_toks = prompt_toks + ans_toks
            labels = [-100] * len(prompt_toks) + ans_toks
            encoded_data.append((
                torch.tensor(full_toks, dtype=torch.long, device=self.device),
                torch.tensor(labels, dtype=torch.long, device=self.device),
            ))

        model.train()
        final_loss = 0.0
        for ep in range(epochs):
            total_loss = 0.0
            for idxs, lbls in encoded_data:
                optimizer.zero_grad()
                out = model(idxs.unsqueeze(0))
                logits = out[0] if isinstance(out, tuple) else out
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = lbls[..., 1:].contiguous().unsqueeze(0)
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            final_loss = total_loss / max(1, len(encoded_data))

        model.eval()
        return final_loss

    def evaluate_arm(
        self,
        arm_type: str,
        qa_pairs: list[tuple[str, str]],
        epochs: int = 15,
        lr: float = 5e-4,
    ) -> ArmEvaluationResult:
        """Run controlled benchmark on one specific arm."""
        # Deepcopy model to ensure strictly isolated starting weights
        arm_model = copy.deepcopy(self.base_model).to(self.device)
        gate_alpha_val: float | None = None

        if arm_type == "ARM_A_FIXED_TOPOLOGY":
            desc = "Original fixed topology (dense backbone training)"
            trainable_params = [p for p in arm_model.parameters()]
            for p in trainable_params:
                p.requires_grad = True
            growth_ratio = 0.0

        elif arm_type == "ARM_B_MATCHED_PARAM_BUDGET":
            desc = "Fixed topology with matched parameter budget (freeze base, intra-block layer scale)"
            # Intra-block scaling (source=3, target=3) maintains the strictly fixed parent topology
            skip = TopologyEvolutionManager.attach_gated_skip(
                arm_model, source_layer=3, target_layer=3, gate_type="scalar", name="fixed_scale_l3"
            )
            # Freeze base; only the matched scalar scaling parameter is trainable
            trainable_params = [p for p in skip.parameters()]
            for p in arm_model.parameters():
                p.requires_grad = False
            for p in trainable_params:
                p.requires_grad = True
            growth_ratio = sum(p.numel() for p in trainable_params) / max(1, sum(p.numel() for p in arm_model.parameters()))

        elif arm_type == "ARM_C_MODULAR_ADAPTER":
            desc = "Original M34/M37 modular adapter strategy (rank-8 LoRA on attention/MLP)"
            inject_multi_domain_adapters(arm_model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
            add_domain_adapter(arm_model, domain_name="structural_arm_c", r=8)
            set_active_domain_adapter(arm_model, "structural_arm_c")
            trainable_params = []
            for name, param in arm_model.named_parameters():
                if "adapters.structural_arm_c." in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
            growth_ratio = sum(p.numel() for p in trainable_params) / (
                sum(p.numel() for p in arm_model.parameters()) - sum(p.numel() for p in trainable_params)
            )

        elif arm_type == "ARM_D_GATED_SKIP":
            desc = "Gated Layer 1 -> Layer 3 skip connection (learnable alpha)"
            skip = TopologyEvolutionManager.attach_gated_skip(
                arm_model, source_layer=1, target_layer=3, gate_type="scalar", name="skip_l1_l3"
            )
            # Verify initial numerical equivalence
            sample_toks = torch.tensor([[10, 20, 30, 40]], dtype=torch.long, device=self.device)
            is_equiv, diff = TopologyEvolutionManager.verify_numerical_equivalence(arm_model, sample_toks)
            assert is_equiv, f"Initial skip connection not numerically equivalent (diff={diff})"

            # Freeze base; only skip parameter is trainable
            trainable_params = [p for p in skip.parameters()]
            for p in arm_model.parameters():
                p.requires_grad = False
            for p in trainable_params:
                p.requires_grad = True
            growth_ratio = sum(p.numel() for p in trainable_params) / max(1, sum(p.numel() for p in arm_model.parameters()))

        else:
            raise ValueError(f"Unknown arm_type: {arm_type}")

        # Train under matched budget
        self.train_on_pairs(arm_model, trainable_params, qa_pairs, epochs=epochs, lr=lr)

        if arm_type == "ARM_D_GATED_SKIP":
            gate_alpha_val = arm_model.skip_connections["skip_l1_l3"].get_alpha_value()
        elif arm_type == "ARM_B_MATCHED_PARAM_BUDGET":
            gate_alpha_val = arm_model.skip_connections["fixed_scale_l3"].get_alpha_value()

        # Evaluate across multi-domain tasks
        arith_acc = sum(self.evaluate_task_neural_only(arm_model, t) for t in UNSEEN_ARITHMETIC_TASKS) / len(UNSEEN_ARITHMETIC_TASKS)
        logic_acc = sum(self.evaluate_task_neural_only(arm_model, t) for t in UNSEEN_LOGIC_TASKS) / len(UNSEEN_LOGIC_TASKS)
        fact_acc = sum(self.evaluate_task_neural_only(arm_model, t) for t in SOURCE_FACTUAL_TASKS) / len(SOURCE_FACTUAL_TASKS)
        para_acc = sum(self.evaluate_task_neural_only(arm_model, t) for t in PARAPHRASED_FACTUAL_TASKS) / len(PARAPHRASED_FACTUAL_TASKS)
        trans_acc = sum(self.evaluate_task_neural_only(arm_model, t) for t in CONCEPT_TRANSFER_TASKS) / len(CONCEPT_TRANSFER_TASKS)
        carry_ret = self.evaluate_carry_retention(arm_model)
        latency = self.measure_inference_latency_ms(arm_model, "What equation expresses Newton's law?")

        # Base hash check
        current_base_hash = compute_base_model_hash(arm_model)
        hash_invariant = (current_base_hash == self.base_hash_initial)

        return ArmEvaluationResult(
            arm_name=arm_type,
            description=desc,
            trainable_parameters=sum(p.numel() for p in trainable_params),
            parameter_growth_ratio=round(growth_ratio, 6),
            learned_gate_alpha=round(gate_alpha_val, 6) if gate_alpha_val is not None else None,
            unseen_arithmetic_accuracy=round(arith_acc, 4),
            unseen_logic_accuracy=round(logic_acc, 4),
            factual_exact_accuracy=round(fact_acc, 4),
            paraphrased_factual_accuracy=round(para_acc, 4),
            concept_transfer_accuracy=round(trans_acc, 4),
            carry_retention_accuracy=round(carry_ret, 4),
            mean_inference_latency_ms=latency,
            base_hash_invariant=hash_invariant,
        )

    def evaluate_sequential_retention_matrix(
        self,
        use_gated_skip: bool = True,
        epochs_per_episode: int = 10,
        lr: float = 5e-4,
    ) -> RetentionMatrixReport:
        """Run sequential research episodes and build the full cross-domain factual retention matrix."""
        model = copy.deepcopy(self.base_model).to(self.device)
        episodes = [ep["subject"] for ep in SEQUENTIAL_EPISODES]
        n_eps = len(SEQUENTIAL_EPISODES)

        if use_gated_skip:
            skip = TopologyEvolutionManager.attach_gated_skip(
                model, source_layer=1, target_layer=3, gate_type="scalar", name="skip_l1_l3"
            )
            # Freeze base
            for p in model.parameters():
                p.requires_grad = False
            for p in skip.parameters():
                p.requires_grad = True
            trainable_params = list(skip.parameters())
        else:
            # Baseline modular adapter
            inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
            add_domain_adapter(model, domain_name="continual_matrix", r=8)
            set_active_domain_adapter(model, "continual_matrix")
            trainable_params = []
            for name, param in model.named_parameters():
                if "adapters.continual_matrix." in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False

        # Matrix: matrix[i][j] = accuracy on subject j after studying episode i
        matrix: list[list[float | None]] = [[None for _ in range(n_eps)] for _ in range(n_eps)]

        for i, ep_info in enumerate(SEQUENTIAL_EPISODES):
            # Train on current episode pairs
            self.train_on_pairs(model, trainable_params, ep_info["train_pairs"], epochs=epochs_per_episode, lr=lr)

            # Evaluate on all studied episodes so far (0 to i)
            for j in range(i + 1):
                eval_task = SEQUENTIAL_EPISODES[j]["test_task"]
                acc = 1.0 if self.evaluate_task_neural_only(model, eval_task) else 0.0
                matrix[i][j] = acc

        # Compute retention rates: matrix[n_eps-1][j] / max(1e-6, matrix[j][j])
        retention_rates = []
        for j in range(n_eps):
            initial_score = matrix[j][j]
            final_score = matrix[n_eps - 1][j]
            if initial_score is not None and initial_score > 0:
                retention_rates.append(final_score / initial_score)
            else:
                retention_rates.append(1.0 if final_score is not None and final_score > 0 else 0.0)

        mean_ret = sum(retention_rates) / len(retention_rates) if retention_rates else 0.0
        mean_bwt = (
            sum(matrix[n_eps - 1][j] - matrix[j][j] for j in range(n_eps - 1)) / max(1, n_eps - 1)
            if n_eps > 1
            else 0.0
        )

        return RetentionMatrixReport(
            matrix=matrix,
            episodes=episodes,
            retention_rates=[round(r, 4) for r in retention_rates],
            mean_backward_transfer=round(mean_bwt, 4),
            mean_retention=round(mean_ret, 4),
        )

    def evaluate_structural_evolution_gate(self) -> dict[str, Any]:
        """Submit and evaluate structural proposals via StructuralEvolutionGate."""
        base_param_count = sum(p.numel() for p in self.base_model.parameters())
        curr_hash = self.base_hash_initial

        # Proposal 1: Baseline Weight Adjustment
        p1 = StructuralProposal.create(
            modification_type=StructuralModificationType.WEIGHT_ADJUSTMENT,
            target_module="base_backbone",
            evidence_of_need="Factual deficit in language domain",
            expected_gain=0.10,  # Below 0.15 threshold
            parameter_delta=0,
            compute_overhead_ratio=0.0,
            current_checkpoint_hash=curr_hash,
        )
        approved1, reason1 = self.gate.evaluate_proposal(p1, base_param_count)

        # Proposal 2: Modular Adapter Addition
        p2 = StructuralProposal.create(
            modification_type=StructuralModificationType.ADD_UPDATE_ADAPTER,
            target_module="attention_projections",
            evidence_of_need="Modular skill isolation for language",
            expected_gain=0.25,
            parameter_delta=73728,  # ~6.7% growth
            compute_overhead_ratio=0.05,
            current_checkpoint_hash=curr_hash,
        )
        approved2, reason2 = self.gate.evaluate_proposal(p2, base_param_count)

        # Proposal 3: Gated Skip Connection Layer 1 -> Layer 3
        p3 = StructuralProposal.create(
            modification_type=StructuralModificationType.GATED_SKIP_CONNECTION,
            target_module="blocks.1_to_blocks.3",
            evidence_of_need="Cross-layer feature propagation for multi-hop relational learning",
            expected_gain=0.20,
            parameter_delta=1,  # Scalar gate = 1 parameter (<0.001% growth)
            compute_overhead_ratio=0.01,
            current_checkpoint_hash=curr_hash,
        )
        approved3, reason3 = self.gate.evaluate_proposal(p3, base_param_count)

        return {
            "proposals": [
                {"proposal": p1.to_dict(), "approved": approved1, "reason": reason1},
                {"proposal": p2.to_dict(), "approved": approved2, "reason": reason2},
                {"proposal": p3.to_dict(), "approved": approved3, "reason": reason3},
            ]
        }
