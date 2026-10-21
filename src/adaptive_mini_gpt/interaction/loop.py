"""Closed-Loop Interactive Learning System (M29 Phase 1 & 3).

Orchestrates the 6-step loop:
  INTERACT -> ATTEMPT -> VERIFY -> IDENTIFY GAP -> SELECT ACTION -> EXECUTE/PRACTICE
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from adaptive_mini_gpt.data.language_corpus import (
    ABSTENTION_TARGET,
    format_qa_prompt,
    normalize_text,
    parse_language_completion,
    verify_format_compliance,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.evaluation.multi_domain import DomainType
from adaptive_mini_gpt.interaction.action_selector import (
    ActionDecision,
    BaseActionPolicy,
    InteractionActionSelector,
    LearningAction,
)
from adaptive_mini_gpt.interaction.knowledge_gap import (
    ErrorCategory,
    KnowledgeGapRecord,
    KnowledgeGapTracker,
)
from adaptive_mini_gpt.interaction.verified_memory import (
    VerifiedMemoryBank,
    VerifiedMemoryEntry,
)
from adaptive_mini_gpt.model.transformer import MiniGPT

logger = logging.getLogger(__name__)


@dataclass
class InteractionResult:
    """Complete traceable outcome of a single interaction cycle."""

    cycle_id: int
    domain: DomainType
    question: str
    predicted_answer: str
    confidence_score: float
    is_verified: bool
    verification_status: str
    gap_record: KnowledgeGapRecord | None
    decision: ActionDecision
    final_output: str
    execution_detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["domain"] = self.domain.value
        if self.gap_record:
            data["gap_record"] = self.gap_record.to_dict()
        data["decision"] = self.decision.to_dict()
        return data


class ClosedLoopInteractiveLearner:
    """Manages neural generation, verification, gap tracking, and action execution."""

    def __init__(
        self,
        model: MiniGPT,
        tokenizer: BPETokenizer,
        memory_bank: VerifiedMemoryBank | None = None,
        gap_tracker: KnowledgeGapTracker | None = None,
        action_selector: BaseActionPolicy | None = None,
        device: torch.device | None = None,
        checkpoint_name: str = "base_1m",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_bank = memory_bank or VerifiedMemoryBank()
        self.gap_tracker = gap_tracker or KnowledgeGapTracker()
        self.action_selector = action_selector or InteractionActionSelector()
        self.checkpoint_name = checkpoint_name
        self.cycle_count = 0

    def generate_with_confidence(
        self,
        prompt: str,
        max_new_tokens: int = 24,
    ) -> tuple[str, float]:
        """Generate tokens autoregressively and compute average token confidence."""
        self.model.eval()
        enc = self.tokenizer.encode(prompt)
        max_ctx = getattr(self.model.config, "context_length", 256)
        if len(enc) > max_ctx - max_new_tokens:
            enc = enc[-(max_ctx - max_new_tokens):]
        input_ids = torch.tensor([enc], dtype=torch.long, device=self.device)

        probs_list: list[float] = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.model(input_ids[:, -max_ctx:])
                logits = out[0] if isinstance(out, tuple) else out
                next_logits = logits[0, -1, :]
                probs = F.softmax(next_logits, dim=-1)
                best_prob, next_id = torch.max(probs, dim=-1)

                probs_list.append(float(best_prob.item()))
                token_item = next_id.item()
                input_ids = torch.cat([input_ids, next_id.unsqueeze(0).unsqueeze(0)], dim=1)

                decoded_partial = self.tokenizer.decode(input_ids[0].tolist())
                eos_id = getattr(self.tokenizer, "eos_id", -1)
                if "\n" in decoded_partial[len(prompt):] or token_item == eos_id:
                    break

        full_decoded = self.tokenizer.decode(input_ids[0].tolist())
        completion = full_decoded[len(prompt):].strip()
        mean_confidence = sum(probs_list) / max(1, len(probs_list))
        return completion, round(mean_confidence, 4)

    def interact(
        self,
        domain: DomainType,
        question: str,
        context: str | None = None,
        ground_truth: str | None = None,
        has_approved_reference: bool = False,
        approved_reference_text: str | None = None,
    ) -> InteractionResult:
        """Execute a full closed-loop interaction step."""
        self.cycle_count += 1

        # 1. ATTEMPT: Format prompt and generate neural response
        if domain == DomainType.LANGUAGE:
            prompt = format_qa_prompt(passage=context or "General knowledge.", question=question)
            raw_completion, confidence = self.generate_with_confidence(prompt, max_new_tokens=24)
            predicted_answer, _, _ = parse_language_completion(raw_completion)
        else:
            prompt = f"{question.strip()}\n#### "
            raw_completion, confidence = self.generate_with_confidence(prompt, max_new_tokens=16)
            predicted_answer = raw_completion.split("\n")[0].strip()

        # 2. VERIFY: Evaluate answer validity
        is_verified = False
        verification_status = "UNKNOWN"

        if ground_truth is not None:
            norm_pred = normalize_text(predicted_answer)
            norm_gold = normalize_text(ground_truth)

            if norm_pred == normalize_text(ABSTENTION_TARGET):
                if norm_gold == normalize_text(ABSTENTION_TARGET):
                    is_verified = True
                    verification_status = "PASS"
                else:
                    is_verified = False
                    verification_status = "ABSTAIN"
            elif norm_pred == norm_gold:
                is_verified = True
                verification_status = "PASS"
            elif context and norm_pred in normalize_text(context):
                # Extracted text exists in passage but does not match target
                is_verified = False
                verification_status = "FAIL"
            else:
                is_verified = False
                verification_status = "UNSUPPORTED"
        else:
            # When ground truth is absent, check format compliance and memory
            compliance = verify_format_compliance(f"#### {predicted_answer}")
            memory_match = self.memory_bank.retrieve(domain, question)
            if memory_match and normalize_text(predicted_answer) == normalize_text(memory_match.verified_fact):
                is_verified = True
                verification_status = "PASS"
            elif compliance:
                verification_status = "PASS" if confidence >= 0.75 else "UNKNOWN"
            else:
                verification_status = "FAIL"

        # 3. IDENTIFY GAP: Record knowledge deficit if verification failed
        gap_record = self.gap_tracker.record_interaction(
            domain=domain,
            question=question,
            predicted_answer=predicted_answer,
            ground_truth=ground_truth,
            confidence_score=confidence,
            is_verified=is_verified,
            verification_status=verification_status,
            context=context,
            supporting_evidence=approved_reference_text,
            model_checkpoint=self.checkpoint_name,
            candidate_learning_objective=f"Master {domain.value} topic: {question[:30]}",
        )

        # 4. SELECT ACTION: Action selector policy determines next step
        decision = self.action_selector.select_action(
            domain=domain,
            question=question,
            predicted_answer=predicted_answer,
            confidence_score=confidence,
            is_verified=is_verified,
            verification_status=verification_status,
            gap_tracker=self.gap_tracker,
            memory_bank=self.memory_bank,
            has_approved_reference=has_approved_reference,
            interaction_step=self.cycle_count,
        )

        # 5. EXECUTE ACTION: Handle the chosen action
        execution_detail: dict[str, Any] = {}
        final_output = predicted_answer

        if decision.action == LearningAction.ANSWER:
            final_output = predicted_answer
            execution_detail["note"] = "Neural answer accepted."
        elif decision.action == LearningAction.ABSTAIN:
            final_output = ABSTENTION_TARGET
            execution_detail["note"] = "Safely abstained due to insufficient verified evidence."
        elif decision.action == LearningAction.RETRIEVE:
            mem_entry = self.memory_bank.retrieve(domain, question)
            if mem_entry:
                final_output = mem_entry.verified_fact
                execution_detail["source"] = f"memory:{mem_entry.entry_id}"
            elif approved_reference_text and ground_truth:
                final_output = ground_truth
                execution_detail["source"] = "approved_local_doc"
                # Store retrieved verified fact in memory bank
                self.memory_bank.store_verified_fact(
                    domain=domain,
                    topic_or_key=question,
                    verified_fact=ground_truth,
                    evidence_source="approved_local_doc:reference",
                    verifier_identity="grounded_oracle",
                    model_version=self.checkpoint_name,
                )
            else:
                final_output = ABSTENTION_TARGET
                execution_detail["source"] = "retrieval_failed_abstained"
        elif decision.action == LearningAction.REMEMBER:
            if ground_truth:
                entry = self.memory_bank.store_verified_fact(
                    domain=domain,
                    topic_or_key=question,
                    verified_fact=ground_truth,
                    evidence_source=f"verified_exercise_generator:{domain.value}",
                    verifier_identity="oracle_verifier",
                    model_version=self.checkpoint_name,
                )
                final_output = ground_truth
                execution_detail["stored_entry_id"] = entry.entry_id
        elif decision.action == LearningAction.PRACTICE:
            execution_detail["practice_scheduled"] = True
            execution_detail["domain"] = domain.value
            final_output = (
                f"[Practice Scheduled] Neural learner identified gap in {domain.value} rules; "
                f"generating practice exercises."
            )
        elif decision.action == LearningAction.TRAIN:
            execution_detail["training_proposed"] = True
            execution_detail["domain"] = domain.value
            execution_detail["rationale"] = decision.rationale
            final_output = (
                f"[Candidate Training Proposed] Repeated procedural failures in {domain.value}. "
                f"Controller formulated bounded GPU experiment."
            )

        return InteractionResult(
            cycle_id=self.cycle_count,
            domain=domain,
            question=question,
            predicted_answer=predicted_answer,
            confidence_score=confidence,
            is_verified=is_verified,
            verification_status=verification_status,
            gap_record=gap_record,
            decision=decision,
            final_output=final_output,
            execution_detail=execution_detail,
        )
