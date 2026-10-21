"""Language Comprehension and Instruction-Following Benchmark Dataset (M25-B / M26 / M27 / M28).

Implements reproducible neural language comprehension with strictly disjoint
train, validation, and sealed-test document partitions.

Features:
  - Reading short factual passages across diverse scientific and historical domains.
  - Extracting evidence-backed information with short, canonical answers.
  - Recognizing unanswerable questions (calibrated safe abstention).
  - Paraphrased query robustness.
  - Prompt formatting without recency copying bias.
  - Token loss masking on prompt tokens (assistant-only supervision with ignore_index=-100).
  - Document-level split isolation (0% fact, topic, or document overlap between splits).
  - Dual evaluation metrics: exact match and grounded extraction.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import TYPE_CHECKING, Any

from adaptive_mini_gpt.data.language_corpus import (
    ABSTENTION_TARGET,
    ABSTENTION_VARIANTS,
    DocumentPassage,
    SEALED_TEST_QA_DOCUMENTS,
    TRAIN_QA_DOCUMENTS,
    VAL_DOCUMENTS as M28_VAL_DOCS,
    format_qa_completion,
    format_qa_prompt,
    normalize_text,
    parse_language_completion,
    verify_format_compliance,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.evaluation.multi_domain import (
    BenchmarkSample,
    BenchmarkSplit,
    DomainType,
)

if TYPE_CHECKING:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset
else:
    try:
        import torch
        from torch import Tensor
        from torch.utils.data import Dataset
    except ImportError:
        torch = None
        Tensor = None
        Dataset = object

# Backwards compatible alias bindings (preserves exact 12/4/3 split counts for M25/M26/M27 test suites)
TRAIN_DOCUMENTS: list[DocumentPassage] = TRAIN_QA_DOCUMENTS[:12]
VAL_DOCUMENTS: list[DocumentPassage] = M28_VAL_DOCS[:4]
TEST_DOCUMENTS: list[DocumentPassage] = SEALED_TEST_QA_DOCUMENTS[:3]


def verify_language_exact(parsed_answer: str, sample: BenchmarkSample) -> bool:
    """Check exact match after standard normalization and article stripping."""
    if not parsed_answer:
        return False

    clean = parsed_answer.replace("<|eos|>", "").strip()
    norm_parsed = normalize_text(clean)

    # Case 1: Unanswerable
    if sample.is_unanswerable:
        return any(term in norm_parsed for term in [
            "insufficient information",
            "unanswerable",
            "not mentioned",
            "unknown",
            "cannot be determined",
        ])

    # If abstained on answerable question -> False
    if any(term in norm_parsed for term in ["insufficient information", "unanswerable", "not mentioned", "unknown"]):
        return False

    norm_expected = normalize_text(sample.expected_answer)
    if norm_parsed == norm_expected:
        return True

    for allowed in sample.allowed_answers:
        if norm_parsed == normalize_text(allowed):
            return True

    return False


def verify_language_grounded(parsed_answer: str, sample: BenchmarkSample) -> bool:
    """Check grounded semantic/substring match, guarding against trivial stopword false positives."""
    if not parsed_answer:
        return False

    clean = parsed_answer.replace("<|eos|>", "").strip()
    norm_parsed = normalize_text(clean)

    # Case 1: Unanswerable
    if sample.is_unanswerable:
        return any(term in norm_parsed for term in [
            "insufficient information",
            "unanswerable",
            "not mentioned",
            "unknown",
            "cannot be determined",
        ])

    # If abstained on answerable question -> False
    if any(term in norm_parsed for term in ["insufficient information", "unanswerable", "not mentioned", "unknown"]):
        return False

    norm_expected = normalize_text(sample.expected_answer)

    # 1. Exact equality
    if norm_parsed == norm_expected:
        return True

    # 2. Meaningful substring match (require length >= 3 to avoid stopword false matches)
    if len(norm_parsed) >= 3 and norm_parsed in norm_expected:
        return True
    if len(norm_expected) >= 3 and norm_expected in norm_parsed:
        return True

    # 3. Check allowed aliases
    for allowed in sample.allowed_answers:
        norm_allowed = normalize_text(allowed)
        if norm_parsed == norm_allowed:
            return True
        if len(norm_parsed) >= 3 and norm_parsed in norm_allowed:
            return True
        if len(norm_allowed) >= 3 and norm_allowed in norm_parsed:
            return True

    return False


def verify_language_answer(parsed_answer: str, sample: BenchmarkSample) -> bool:
    """Default verifier (evaluates grounded extraction match)."""
    return verify_language_grounded(parsed_answer, sample)


def build_language_benchmark_samples(
    split: BenchmarkSplit,
    include_paraphrases: bool = True,
) -> list[BenchmarkSample]:
    """Construct deterministic BenchmarkSample list for the designated split."""
    if split == BenchmarkSplit.TRAIN:
        doc_pool = TRAIN_DOCUMENTS
    elif split == BenchmarkSplit.VAL:
        doc_pool = VAL_DOCUMENTS
    elif split == BenchmarkSplit.TEST:
        doc_pool = TEST_DOCUMENTS
    else:
        raise ValueError(f"Unknown split: {split}")

    samples: list[BenchmarkSample] = []
    for doc in doc_pool:
        # 1. Answerable questions
        for idx, (q, ans, aliases) in enumerate(doc.answerable_pairs):
            sample_id = f"{doc.doc_id}_ans_{idx}"
            prompt = format_qa_prompt(doc.text, q)
            samples.append(
                BenchmarkSample(
                    sample_id=sample_id,
                    domain=DomainType.LANGUAGE,
                    split=split,
                    prompt=prompt,
                    expected_answer=ans,
                    allowed_answers=aliases,
                    is_unanswerable=False,
                    metadata={"doc_id": doc.doc_id, "topic": doc.topic, "type": "factual_extraction"},
                )
            )

        # 2. Unanswerable questions (abstention battery)
        for idx, q_tuple in enumerate(doc.unanswerable_questions):
            q = q_tuple[0] if isinstance(q_tuple, tuple) else q_tuple
            sample_id = f"{doc.doc_id}_unans_{idx}"
            prompt = format_qa_prompt(doc.text, q)
            samples.append(
                BenchmarkSample(
                    sample_id=sample_id,
                    domain=DomainType.LANGUAGE,
                    split=split,
                    prompt=prompt,
                    expected_answer=ABSTENTION_TARGET,
                    allowed_answers=list(ABSTENTION_VARIANTS),
                    is_unanswerable=True,
                    metadata={"doc_id": doc.doc_id, "topic": doc.topic, "type": "safe_abstention"},
                )
            )

        # 3. Paraphrased questions (robustness test)
        if include_paraphrases and doc.paraphrase_pairs:
            for idx, (q, ans, aliases) in enumerate(doc.paraphrase_pairs):
                sample_id = f"{doc.doc_id}_para_{idx}"
                prompt = format_qa_prompt(doc.text, q)
                samples.append(
                    BenchmarkSample(
                        sample_id=sample_id,
                        domain=DomainType.LANGUAGE,
                        split=split,
                        prompt=prompt,
                        expected_answer=ans,
                        allowed_answers=aliases,
                        is_unanswerable=False,
                        metadata={"doc_id": doc.doc_id, "topic": doc.topic, "type": "paraphrase_robustness"},
                    )
                )

    return samples


class LanguageComprehensionDataset(Dataset):
    """PyTorch Dataset for language comprehension with prompt token loss masking."""

    def __init__(
        self,
        samples: list[BenchmarkSample],
        tokenizer: BPETokenizer,
        context_length: int = 256,
        ignore_index: int = -100,
    ) -> None:
        self.context_length = context_length
        self.ignore_index = ignore_index
        self.data: list[tuple[Tensor, Tensor]] = []

        if torch is None:
            return

        for s in samples:
            prompt_ids = tokenizer.encode(s.prompt)
            answer_ids = tokenizer.encode(format_qa_completion(s.expected_answer))
            token_ids = prompt_ids + answer_ids

            if len(token_ids) > context_length:
                # Keep the suffix so the assistant answer is preserved
                token_ids = token_ids[-context_length:]
                prompt_len = max(0, len(token_ids) - len(answer_ids))
            else:
                prompt_len = len(prompt_ids)

            input_ids = torch.tensor(token_ids, dtype=torch.long)
            # Create shifted target labels with ignore_index on prompt tokens
            target_ids = input_ids.clone()
            target_ids[:prompt_len] = ignore_index

            self.data.append((input_ids, target_ids))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.data[idx]


def collate_language_batch(
    batch: list[tuple[Tensor, Tensor]],
    pad_token_id: int,
    ignore_index: int = -100,
) -> tuple[Tensor, Tensor]:
    """Pad variable-length language comprehension samples to max sequence length in batch."""
    max_len = max(x.size(0) for x, _ in batch)
    inputs_padded: list[Tensor] = []
    targets_padded: list[Tensor] = []

    for x, y in batch:
        pad_len = max_len - x.size(0)
        if pad_len > 0:
            x_pad = torch.cat([x, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            y_pad = torch.cat([y, torch.full((pad_len,), ignore_index, dtype=torch.long)])
        else:
            x_pad = x
            y_pad = y
        inputs_padded.append(x_pad)
        targets_padded.append(y_pad)

    return torch.stack(inputs_padded, dim=0), torch.stack(targets_padded, dim=0)
