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

import torch
import torch.nn as nn

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.transformer import MiniGPT


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
class SamplePrediction:
    """Prediction result and evaluation telemetry for a single benchmark sample."""

    sample_id: str
    domain: DomainType
    prompt: str
    raw_completion: str
    parsed_answer: str
    is_correct: bool
    is_abstained: bool
    is_format_compliant: bool
    error_type: str | None = None
    latency_ms: float = 0.0


@dataclass
class DomainResult:
    """Domain-level evaluation outcome with fine-grained sub-metrics."""

    domain: str
    split: str
    total_samples: int
    exact_accuracy: float
    abstention_accuracy: float
    format_compliance_rate: float
    mean_latency_ms: float = 0.0
    carry_accuracy: float | None = None
    grounded_accuracy: float | None = None
    loss: float | None = None
    bpc: float | None = None
    sub_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MultiDomainReport:
    """Comprehensive evaluation record covering all active domains and retention drift."""

    benchmark_version: str
    model_id: str
    checkpoint_sha256: str
    tokenizer_fingerprint: str
    param_count: int
    device: str
    timestamp: str
    domain_results: dict[str, DomainResult]
    retention_drift: dict[str, float] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


class MultiDomainBenchmark:
    """Minimal shared multi-domain benchmark coordinator.

    Maintains task registries and executes standardized evaluations without
    cross-domain contamination or data leakage.
    """

    def __init__(self, version: str = "m27-v1") -> None:
        self.version = version
        self.samples: dict[DomainType, dict[BenchmarkSplit, list[BenchmarkSample]]] = {
            domain: {split: [] for split in BenchmarkSplit}
            for domain in DomainType
        }
        self.custom_evaluators: dict[DomainType, Callable[..., DomainResult]] = {}

    def register_sample(self, sample: BenchmarkSample) -> None:
        """Register a single benchmark problem into its designated domain and split."""
        self.samples[sample.domain][sample.split].append(sample)

    def register_samples(self, samples: list[BenchmarkSample]) -> None:
        """Batch-register multiple benchmark problems."""
        for s in samples:
            self.register_sample(s)

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

    @torch.no_grad()
    def evaluate_generation_task(
        self,
        model: MiniGPT,
        tokenizer: BPETokenizer,
        domain: DomainType,
        split: BenchmarkSplit,
        parse_fn: Callable[[str], tuple[str, bool, bool]],
        verify_fn: Callable[[str, BenchmarkSample], bool],
        verify_grounded_fn: Callable[[str, BenchmarkSample], bool] | None = None,
        device: torch.device | str = "cuda",
        max_new_tokens: int = 32,
    ) -> DomainResult:
        """Standardized neural generation evaluation loop for textual QA/reasoning tasks.

        Args:
            model: Neural MiniGPT model in eval mode.
            tokenizer: BPETokenizer matching model lineage.
            domain: Target domain (e.g. LANGUAGE, LOGIC).
            split: Target evaluation split (VAL or TEST).
            parse_fn: Function mapping completion string -> (parsed_answer, is_compliant, is_abstained).
            verify_fn: Function checking whether parsed answer matches expected sample (exact match).
            verify_grounded_fn: Optional function checking grounded semantic match.
            device: Execution device.
            max_new_tokens: Autoregressive decoding limit.
        """
        model.eval()
        samples = self.get_samples(domain, split)
        if not samples:
            return DomainResult(
                domain=domain.value,
                split=split.value,
                total_samples=0,
                exact_accuracy=0.0,
                abstention_accuracy=0.0,
                format_compliance_rate=0.0,
            )

        correct_count = 0
        grounded_count = 0
        abstain_correct = 0
        abstain_total = 0
        compliant_count = 0
        latencies: list[float] = []

        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and "cuda" in str(device) else None
        end_time = torch.cuda.Event(enable_timing=True) if start_time is not None else None

        for s in samples:
            prompt_ids = tokenizer.encode(s.prompt)
            # Guard against exceeding context length
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
        device: torch.device | str = "cuda",
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
            f"No evaluator registered for domain {domain.value}. "
            f"Register an evaluator using register_evaluator()."
        )
