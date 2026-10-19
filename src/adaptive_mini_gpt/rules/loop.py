"""Reasoning and Verification Loop with Bounded Retries (M14).

Coordinates Mini-GPT, Rule Execution, and Verification:
1. Propose operation from natural language question.
2. Validate against Rule Registry schema.
3. Execute deterministically.
4. Verify postconditions.
5. Provide structured error feedback on failure and retry up to bounded limit.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.executor import RuleExecutionResult, RuleExecutor

logger = logging.getLogger(__name__)


def format_rule_call(rule: str, args: dict[str, Any]) -> str:
    """Format a canonical rule call string for prompt completion."""
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
    return f"Action: {rule}({args_str})"


def parse_rule_call(text: str) -> tuple[str, dict[str, Any]]:
    """Robustly extract rule name and keyword arguments from model completion."""
    cleaned = text.strip()
    if cleaned.endswith("<|eos|>"):
        cleaned = cleaned[:-len("<|eos|>")].strip()

    # Try 1: JSON block extraction
    m_json = re.search(r"\{.*\"rule\".*\}", cleaned, re.DOTALL)
    if m_json:
        try:
            data = json.loads(m_json.group(0))
            if "rule" in data and "args" in data and isinstance(data["args"], dict):
                return data["rule"].lower(), data["args"]
        except Exception:
            pass

    # Try 2: Functional call Action: name(...) or <|call:name(...)|> or name(...)
    pat = r"(?:Action:\s*|<\|call:)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)(?:\|>)?"
    m = re.search(pat, cleaned, re.DOTALL)
    if m:
        rule_name = m.group(1).lower()
        args_str = m.group(2).strip()
        args: dict[str, Any] = {}
        try:
            tree = ast.parse(f"dummy({args_str})")
            call_node = tree.body[0].value  # type: ignore[attr-defined]
            for kw in call_node.keywords:
                val = ast.literal_eval(kw.value)
                args[kw.arg] = val
            return rule_name, args
        except Exception:
            pass

    raise ValueError(f"Could not parse valid rule call from completion: '{text[:80]}'")


@dataclass
class AttemptRecord:
    """Record of a single reasoning/execution attempt in the loop."""

    attempt_index: int
    raw_completion: str
    parsed_rule: str | None = None
    parsed_args: dict[str, Any] | None = None
    execution_result: RuleExecutionResult | None = None
    error_feedback: str | None = None


@dataclass
class ReasoningResult:
    """End-to-end outcome of reasoning and verification loop."""

    question: str
    ground_truth_rule: str
    expected_answer: str
    mode: str
    status: str  # "success" | "failed"
    final_answer: str | None = None
    rule_selected: str | None = None
    rule_selection_correct: bool = False
    arg_extraction_correct: bool = False
    answer_correct: bool = False
    attempts: list[AttemptRecord] = field(default_factory=list)
    total_retries: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "ground_truth_rule": self.ground_truth_rule,
            "expected_answer": self.expected_answer,
            "mode": self.mode,
            "status": self.status,
            "final_answer": self.final_answer,
            "rule_selected": self.rule_selected,
            "rule_selection_correct": self.rule_selection_correct,
            "arg_extraction_correct": self.arg_extraction_correct,
            "answer_correct": self.answer_correct,
            "total_retries": self.total_retries,
            "latency_ms": round(self.latency_ms, 2),
            "num_attempts": len(self.attempts),
        }


class ReasoningController:
    """Coordinates Mini-GPT, Rule Execution, and Verification across M14 ablation modes."""

    def __init__(
        self,
        model: MiniGPT,
        tokenizer: Any,
        executor: RuleExecutor | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.executor = executor if executor is not None else RuleExecutor()
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _generate(self, prompt: str, max_tokens: int = 48) -> str:
        """Autoregressively generate completion from prompt."""
        was_training = self.model.training
        self.model.eval()

        prompt_tokens = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        eos_id = getattr(self.tokenizer, "eos_id", -1)

        with torch.no_grad():
            for _ in range(max_tokens):
                if input_tensor.size(1) >= self.model.config.context_length:
                    break
                logits, _ = self.model(input_tensor)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_tensor = torch.cat([input_tensor, next_token], dim=1)
                if next_token.item() == eos_id:
                    break

        generated_full = self.tokenizer.decode(input_tensor[0].tolist())
        completion = generated_full[len(prompt) :]

        if was_training:
            self.model.train()

        return completion.strip()

    def solve(
        self,
        question: str,
        expected_rule: str,
        expected_answer: str,
        ground_truth_args: dict[str, Any] | None = None,
        mode: str = "rule_engine_retry",
        max_retries: int = 2,
    ) -> ReasoningResult:
        """Solve a question under one of four M14 ablation configurations:

        1. 'neural_only': Direct model completion without tool use.
        2. 'oracle': Oracle rule selection and oracle arguments passed to executor.
        3. 'rule_engine': Model selects rule and arguments, runs executor once (no retry).
        4. 'rule_engine_retry': Model + executor + verifier with feedback retry loop.
        """
        t0 = time.perf_counter()
        attempts: list[AttemptRecord] = []

        # =================================================================
        # Configuration 1: Neural Only (Baseline)
        # =================================================================
        if mode == "neural_only":
            prompt = f"<|user|>\n{question}\n<|assistant|>\n"
            raw_ans = self._generate(prompt)
            clean_ans = raw_ans.replace("<|eos|>", "").strip()
            # Standardize numeric or string match
            is_correct = clean_ans.lower() == expected_answer.lower()
            latency = (time.perf_counter() - t0) * 1000.0
            return ReasoningResult(
                question=question,
                ground_truth_rule=expected_rule,
                expected_answer=expected_answer,
                mode=mode,
                status="success" if is_correct else "failed",
                final_answer=clean_ans,
                answer_correct=is_correct,
                attempts=[AttemptRecord(attempt_index=1, raw_completion=raw_ans)],
                latency_ms=latency,
            )

        # =================================================================
        # Configuration 2: Oracle Rule Engine (Upper Bound)
        # =================================================================
        if mode == "oracle":
            if ground_truth_args is None:
                raise ValueError("ground_truth_args required for oracle mode")
            exec_res = self.executor.execute(expected_rule, ground_truth_args, verify=True)
            final_ans = str(exec_res.result) if exec_res.success else None
            is_correct = exec_res.verified and str(exec_res.result).lower() == expected_answer.lower()
            latency = (time.perf_counter() - t0) * 1000.0
            return ReasoningResult(
                question=question,
                ground_truth_rule=expected_rule,
                expected_answer=expected_answer,
                mode=mode,
                status="success" if is_correct else "failed",
                final_answer=final_ans,
                rule_selected=expected_rule,
                rule_selection_correct=True,
                arg_extraction_correct=True,
                answer_correct=is_correct,
                attempts=[
                    AttemptRecord(
                        attempt_index=1,
                        raw_completion=f"Oracle: {expected_rule}({ground_truth_args})",
                        parsed_rule=expected_rule,
                        parsed_args=ground_truth_args,
                        execution_result=exec_res,
                    )
                ],
                latency_ms=latency,
            )

        # =================================================================
        # Configurations 3 & 4: Model + Rule Engine (+ optional Retry)
        # =================================================================
        base_prompt = f"<|user|>\n{question}\n<|assistant|>\n"
        current_prompt = base_prompt
        retries_allowed = max_retries if mode == "rule_engine_retry" else 0
        attempt_idx = 0

        final_answer: str | None = None
        status = "failed"
        rule_selected: str | None = None
        rule_sel_correct = False
        arg_ext_correct = False
        ans_correct = False

        while attempt_idx <= retries_allowed:
            attempt_idx += 1
            raw_comp = self._generate(current_prompt)
            record = AttemptRecord(attempt_index=attempt_idx, raw_completion=raw_comp)

            try:
                rule_name, args = parse_rule_call(raw_comp)
                record.parsed_rule = rule_name
                record.parsed_args = args
                rule_selected = rule_name
                rule_sel_correct = rule_name.lower() == expected_rule.lower()

                # Argument extraction correctness against ground truth if available
                if ground_truth_args is not None:
                    arg_ext_correct = rule_sel_correct and (args == ground_truth_args)

                # Execute rule with verifier
                exec_res = self.executor.execute(rule_name, args, verify=True)
                record.execution_result = exec_res

                if exec_res.success and exec_res.verified:
                    final_answer = str(exec_res.result)
                    ans_correct = final_answer.lower() == expected_answer.lower()
                    if ans_correct:
                        status = "success"
                        attempts.append(record)
                        break
                    else:
                        error_msg = f"Output '{final_answer}' failed expected check"
                        record.error_feedback = error_msg
                else:
                    error_msg = exec_res.error or "Rule verification failed"
                    record.error_feedback = error_msg

            except Exception as e:
                error_msg = f"Call parse error: {str(e)}"
                record.error_feedback = error_msg

            attempts.append(record)

            # If retries remain, append feedback to conversation
            if attempt_idx <= retries_allowed:
                feedback = (
                    f"{raw_comp}\n<|system|>\nError: {record.error_feedback}. "
                    f"Please correct your operation call format: Action: <rule>(args...)\n<|assistant|>\n"
                )
                current_prompt = current_prompt + feedback
                # Bound prompt if approaching context limit
                if len(self.tokenizer.encode(current_prompt)) > self.model.config.context_length - 40:
                    current_prompt = (
                        f"<|user|>\n{question}\n<|system|>\nPrevious error: {record.error_feedback}. "
                        f"Provide corrected call.\n<|assistant|>\n"
                    )

        latency = (time.perf_counter() - t0) * 1000.0
        return ReasoningResult(
            question=question,
            ground_truth_rule=expected_rule,
            expected_answer=expected_answer,
            mode=mode,
            status=status,
            final_answer=final_answer,
            rule_selected=rule_selected,
            rule_selection_correct=rule_sel_correct,
            arg_extraction_correct=arg_ext_correct,
            answer_correct=ans_correct,
            attempts=attempts,
            total_retries=attempt_idx - 1,
            latency_ms=latency,
        )
