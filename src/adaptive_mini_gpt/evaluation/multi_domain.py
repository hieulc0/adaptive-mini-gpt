"""Unified Multi-Domain Benchmark Interface (M25 / M27).

Provides a minimal, reproducible, modular benchmark interface separating:
  1. Neural learning and representation acquisition.
  2. Multi-domain evaluation (Arithmetic, Language, Logic, Retention).
  3. Strict telemetry tracking (accuracy, generalization, abstention, retention drift).
  4. Isolation between validation and sealed-test holdouts.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

try:
    import torch
    import torch.nn as nn
    from adaptive_mini_gpt.model.transformer import MiniGPT
    _no_grad = torch.no_grad
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    MiniGPT = None  # type: ignore
    def _no_grad():
        def decorator(f):
            return f
        return decorator

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer


class DomainType(str, Enum):
    """Supported multi-domain capability tracks."""

    ARITHMETIC = "arithmetic"
    LANGUAGE = "language"
    LOGIC = "logic"
    RETENTION = "retention"


class BenchmarkSplit(str, Enum):
    """Disjoint evaluation split designations."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class BenchmarkSample:
    """Individual benchmark problem across any supported domain."""

    sample_id: str
    domain: DomainType
    split: BenchmarkSplit
    prompt: str
    expected_answer: str
    allowed_answers: list[str] = field(default_factory=list)
    is_unanswerable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainResult:
    """Standardized performance report for a single capability domain."""

    domain: str
    split: str
    total_samples: int
    exact_accuracy: float
    abstention_accuracy: float
    format_compliance_rate: float
    grounded_accuracy: float = 0.0
    retention_drift_nats: float = 0.0
    mean_latency_ms: float = 0.0
    sub_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class MultiDomainReport:
    """Comprehensive evaluation report across all evaluated domains."""

    timestamp: str
    model_lineage: str
    checkpoint_step: int
    domains: dict[str, DomainResult] = field(default_factory=dict)
    summary_score: float = 0.0
    passed_all_gates: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model_lineage": self.model_lineage,
            "checkpoint_step": self.checkpoint_step,
            "summary_score": self.summary_score,
            "passed_all_gates": self.passed_all_gates,
            "domains": {k: asdict(v) for k, v in self.domains.items()},
        }


class MultiDomainBenchmark:
    """Central harness coordinating multi-domain evaluation without leakage."""

    def __init__(self) -> None:
        self.samples: dict[DomainType, dict[BenchmarkSplit, list[BenchmarkSample]]] = {
            d: {s: [] for s in BenchmarkSplit} for d in DomainType
        }
        self.custom_evaluators: dict[DomainType, Callable[..., DomainResult]] = {}

    def register_samples(
        self,
        domain: DomainType,
        split: BenchmarkSplit,
        samples: list[BenchmarkSample],
    ) -> None:
        """Register benchmark samples for a domain and split."""
        self.samples[domain][split].extend(samples)

    def register_evaluator(
        self,
        domain: DomainType,
        evaluator_fn: Callable[..., DomainResult],
    ) -> None:
        """Register domain-specific evaluation logic (e.g. recurrent arithmetic unrolling)."""
        self.custom_evaluators[domain] = evaluator_fn

    def get_samples(self, domain: DomainType, split: BenchmarkSplit) -> list[BenchmarkSample]:
        """Retrieve all samples registered for a specific domain and split."""
        return list(self.samples[domain][split])

    @_no_grad()
    def evaluate_generation_task(
        self,
        model: MiniGPT,
        tokenizer: BPETokenizer,
        domain: DomainType,
        split: BenchmarkSplit,
        parse_fn: Callable[[str], tuple[str, bool, bool]],
        verify_fn: Callable[[str, BenchmarkSample], bool],
        verify_grounded_fn: Callable[[str, BenchmarkSample], bool] | None = None,
        device: Any = "cuda",
        max_new_tokens: int = 32,
    ) -> DomainResult:
        """Standardized neural generation evaluation loop for textual QA/reasoning tasks."""
        if torch is None:
            raise RuntimeError("PyTorch is required for neural model evaluation")

        samples = self.get_samples(domain, split)
        if not samples:
            raise ValueError(f"No samples registered for domain {domain.value} split {split.value}")

        model.eval()
        correct_count = 0
        grounded_count = 0
        abstain_total = 0
        abstain_correct = 0
        compliant_count = 0
        latencies: list[float] = []

        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and "cuda" in str(device) else None
        end_time = torch.cuda.Event(enable_timing=True) if start_time is not None else None

        for s in samples:
            prompt_ids = tokenizer.encode(s.prompt)
            if len(prompt_ids) >= model.config.context_length:
                prompt_ids = prompt_ids[-(model.config.context_length - max_new_tokens - 1) :]

            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            if start_time is not None:
                start_time.record()

            # Autoregressive greedy decoding
            for _ in range(max_new_tokens):
                if input_tensor.size(1) >= model.config.context_length:
                    break
                logits, _ = model(input_tensor)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_tensor = torch.cat([input_tensor, next_tok], dim=1)
                if next_tok.item() == tokenizer.eos_id:
                    break

            if end_time is not None:
                end_time.record()
                torch.cuda.synchronize()
                latencies.append(start_time.elapsed_time(end_time))

            # Decode only newly generated tokens directly to avoid string boundary mismatch
            generated_toks = input_tensor[0, len(prompt_ids) :].tolist()
            completion = tokenizer.decode(generated_toks).strip()

            parsed_answer, is_compliant, is_abstained = parse_fn(completion)
            if is_compliant:
                compliant_count += 1

            if s.is_unanswerable:
                abstain_total += 1
                if is_abstained:
                    abstain_correct += 1

            is_correct = verify_fn(parsed_answer, s)
            if is_correct:
                correct_count += 1

            if verify_grounded_fn is not None:
                if verify_grounded_fn(parsed_answer, s):
                    grounded_count += 1
            else:
                if is_correct:
                    grounded_count += 1

        total = len(samples)
        exact_acc = round(correct_count / total, 4) if total > 0 else 0.0
        grounded_acc = round(grounded_count / total, 4) if total > 0 else 0.0
        abstain_acc = round(abstain_correct / abstain_total, 4) if abstain_total > 0 else 1.0
        compliance = round(compliant_count / total, 4) if total > 0 else 0.0
        mean_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

        sub_m = {
            "exact_accuracy": exact_acc,
            "grounded_accuracy": grounded_acc,
            "abstention_accuracy": abstain_acc,
            "format_compliance_rate": compliance,
        }

        return DomainResult(
            domain=domain.value,
            split=split.value,
            total_samples=total,
            exact_accuracy=exact_acc,
            abstention_accuracy=abstain_acc,
            format_compliance_rate=compliance,
            grounded_accuracy=grounded_acc,
            mean_latency_ms=mean_lat,
            sub_metrics=sub_m,
        )

    def evaluate_domain(
        self,
        domain: DomainType,
        model: MiniGPT,
        tokenizer: BPETokenizer,
        split: BenchmarkSplit = BenchmarkSplit.TEST,
        device: Any = "cuda",
        **kwargs: Any,
    ) -> DomainResult:
        """Evaluate a single capability domain using registered evaluator or defaults."""
        if domain in self.custom_evaluators:
            return self.custom_evaluators[domain](
                model=model,
                tokenizer=tokenizer,
                split=split,
                device=device,
                **kwargs,
            )
        raise NotImplementedError(
            f"No evaluator registered for domain {domain.value}. Use register_evaluator() or evaluate_generation_task()."
        )
