"""Reasoning and Verification Loop with Bounded Retries, Working Memory, and Multi-Step Planning (M14/M15/M16).

Coordinates Mini-GPT, Grounded Extraction, Working Memory, and Multi-Step Plans:
1. Select operation (autonomous Mini-GPT, deterministic rule classifier, or oracle).
2. Ground arguments with exact source-span provenance into WorkingMemory (M15).
3. Synthesize and validate multi-step execution plans with dependency checking (M16).
4. Execute deterministically via RuleExecutor and PlanExecutor.
5. Verify postconditions at every step.
6. Support explicit abstention on malformed, contradictory, or unsupported queries.
7. Support bounded retries on autoregressive feedback without leaking benchmark ground-truth answers.
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
from adaptive_mini_gpt.rules.grounding import (
    AbstentionException,
    DeterministicArgumentGrounder,
    DeterministicRuleSelector,
    GroundingError,
    SpanCopyArgumentGrounder,
)
from adaptive_mini_gpt.rules.memory import WorkingMemory
from adaptive_mini_gpt.rules.plan import ExecutionPlan, PlanExecutionResult, PlanExecutor, PlanValidationError
from adaptive_mini_gpt.rules.planner import DeterministicTemplatePlanner, parse_plan

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


def are_arguments_semantically_equivalent(
    rule_name: str,
    args1: dict[str, Any] | None,
    args2: dict[str, Any] | None,
) -> bool:
    """Check semantic equivalence between argument dictionaries, accounting for commutative operations."""
    if args1 is None or args2 is None:
        return args1 == args2
    if args1 == args2:
        return True

    # Addition commutativity: a + b == b + a
    if rule_name == "add":
        if set(args1.keys()) == {"a", "b"} and set(args2.keys()) == {"a", "b"}:
            return args1["a"] == args2["b"] and args1["b"] == args2["a"]

    return False


@dataclass
class AttemptRecord:
    """Record of a single reasoning/execution attempt in the loop."""

    attempt_index: int
    raw_completion: str
    parsed_rule: str | None = None
    parsed_args: dict[str, Any] | None = None
    parsed_plan: ExecutionPlan | None = None
    execution_result: RuleExecutionResult | PlanExecutionResult | None = None
    error_feedback: str | None = None
    working_memory: WorkingMemory | None = None


@dataclass
class ReasoningResult:
    """End-to-end outcome of reasoning, grounding, planning, and verification loop."""

    question: str
    ground_truth_rule: str
    expected_answer: str | None
    mode: str
    status: str  # "success" | "failed" | "abstained"
    final_answer: str | None = None
    rule_selected: str | None = None
    rule_selection_correct: bool = False
    arg_extraction_correct: bool = False
    arg_semantic_correct: bool = False
    answer_correct: bool = False
    abstained: bool = False
    abstention_correct: bool = True
    working_memory: WorkingMemory | None = None
    plan: ExecutionPlan | None = None
    plan_valid: bool = False
    plan_exact_match: bool = False
    num_steps: int = 1
    step_results: dict[str, Any] = field(default_factory=dict)
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
            "arg_semantic_correct": self.arg_semantic_correct,
            "answer_correct": self.answer_correct,
            "abstained": self.abstained,
            "abstention_correct": self.abstention_correct,
            "working_memory": self.working_memory.to_dict() if self.working_memory else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "plan_valid": self.plan_valid,
            "plan_exact_match": self.plan_exact_match,
            "num_steps": self.num_steps,
            "step_results": self.step_results,
            "total_retries": self.total_retries,
            "latency_ms": round(self.latency_ms, 2),
            "num_attempts": len(self.attempts),
        }


class ReasoningController:
    """Coordinates Mini-GPT, Grounding, Multi-Step Planning, Working Memory, and Verification."""

    def __init__(
        self,
        model: MiniGPT | None = None,
        tokenizer: Any = None,
        executor: RuleExecutor | None = None,
        plan_executor: PlanExecutor | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.executor = executor if executor is not None else RuleExecutor()
        self.plan_executor = plan_executor if plan_executor is not None else PlanExecutor(self.executor)
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _generate(self, prompt: str, max_tokens: int = 64) -> str:
        """Autoregressively generate completion from prompt."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer are required for neural generation")

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
        expected_answer: str | None,
        ground_truth_args: dict[str, Any] | None = None,
        mode: str = "deterministic",
        max_retries: int = 2,
        should_abstain: bool = False,
    ) -> ReasoningResult:
        """Solve a single-step question (M14/M15)."""
        t0 = time.perf_counter()

        # =================================================================
        # Configuration 1: Neural Only (Baseline)
        # =================================================================
        if mode == "neural_only":
            prompt = f"<|user|>\n{question}\n<|assistant|>\n"
            raw_ans = self._generate(prompt)
            clean_ans = raw_ans.replace("<|eos|>", "").strip()
            is_correct = (
                expected_answer is not None
                and clean_ans.lower() == expected_answer.lower()
            )
            latency = (time.perf_counter() - t0) * 1000.0
            return ReasoningResult(
                question=question,
                ground_truth_rule=expected_rule,
                expected_answer=expected_answer,
                mode=mode,
                status="success" if is_correct else "failed",
                final_answer=clean_ans,
                answer_correct=is_correct,
                abstained=False,
                abstention_correct=not should_abstain,
                attempts=[AttemptRecord(attempt_index=1, raw_completion=raw_ans)],
                latency_ms=latency,
            )

        # =================================================================
        # Configuration 2: Oracle Rule Engine (Upper Bound)
        # =================================================================
        if mode == "oracle":
            if should_abstain:
                latency = (time.perf_counter() - t0) * 1000.0
                return ReasoningResult(
                    question=question,
                    ground_truth_rule=expected_rule,
                    expected_answer=expected_answer,
                    mode=mode,
                    status="abstained",
                    rule_selected=expected_rule,
                    rule_selection_correct=True,
                    arg_extraction_correct=True,
                    arg_semantic_correct=True,
                    answer_correct=True,
                    abstained=True,
                    abstention_correct=True,
                    latency_ms=latency,
                )
            if ground_truth_args is None:
                raise ValueError("ground_truth_args required for oracle mode")

            mem = WorkingMemory(operation=expected_rule)
            for k, v in ground_truth_args.items():
                mem.bind_argument(k, v, str(v), (0, 0), type(v).__name__, "oracle")

            exec_res = self.executor.execute(expected_rule, ground_truth_args, verify=True)
            if exec_res.success and exec_res.verified:
                mem.bind_result(exec_res.result, verified=True)
            else:
                mem.reject(exec_res.error or "Oracle execution failed")

            final_ans = str(exec_res.result) if exec_res.success else None
            is_correct = (
                exec_res.verified
                and expected_answer is not None
                and str(exec_res.result).lower() == expected_answer.lower()
            )
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
                arg_semantic_correct=True,
                answer_correct=is_correct,
                abstained=False,
                abstention_correct=not should_abstain,
                working_memory=mem,
                attempts=[
                    AttemptRecord(
                        attempt_index=1,
                        raw_completion=f"Oracle: {expected_rule}({ground_truth_args})",
                        parsed_rule=expected_rule,
                        parsed_args=ground_truth_args,
                        execution_result=exec_res,
                        working_memory=mem,
                    )
                ],
                latency_ms=latency,
            )

        # =================================================================
        # Grounded Arms (deterministic, span_copy, and hybrids)
        # =================================================================
        if mode in (
            "deterministic",
            "span_copy",
            "model_rule_deterministic_grounding",
            "model_rule_span_copy",
        ):
            rule_selected = None
            rule_sel_correct = False
            raw_trace = ""

            try:
                if mode in ("model_rule_deterministic_grounding", "model_rule_span_copy"):
                    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
                    raw_trace = self._generate(prompt)
                    try:
                        rule_selected, _ = parse_rule_call(raw_trace)
                    except Exception:
                        for r_candidate in ("reverse", "add", "compare", "average_speed"):
                            if r_candidate in raw_trace.lower():
                                rule_selected = r_candidate
                                break
                else:
                    rule_selected = DeterministicRuleSelector.select_rule(question)
                    raw_trace = f"DeterministicSelector: {rule_selected}"

                if rule_selected:
                    rule_sel_correct = rule_selected.lower() == expected_rule.lower()
            except Exception as e:
                rule_selected = None
                raw_trace = f"Rule selection error: {str(e)}"

            if not rule_selected:
                latency = (time.perf_counter() - t0) * 1000.0
                mem = WorkingMemory()
                mem.reject(f"Could not select rule: {raw_trace}")
                return ReasoningResult(
                    question=question,
                    ground_truth_rule=expected_rule,
                    expected_answer=expected_answer,
                    mode=mode,
                    status="abstained" if should_abstain else "failed",
                    rule_selected=None,
                    rule_selection_correct=False,
                    arg_extraction_correct=False,
                    arg_semantic_correct=False,
                    answer_correct=should_abstain,
                    abstained=True,
                    abstention_correct=should_abstain,
                    working_memory=mem,
                    attempts=[AttemptRecord(attempt_index=1, raw_completion=raw_trace, working_memory=mem)],
                    latency_ms=latency,
                )

            # Grounding
            grounder_type = "span_copy" if "span_copy" in mode else "deterministic"
            try:
                if grounder_type == "span_copy":
                    mem = SpanCopyArgumentGrounder.ground(rule_selected, question)
                else:
                    mem = DeterministicArgumentGrounder.ground(rule_selected, question)
            except Exception as e:
                mem = WorkingMemory(operation=rule_selected)
                mem.reject(f"Grounding error: {str(e)}")

            is_grounded = mem.status == "grounded" and not mem.error
            arg_ext_correct = False
            arg_sem_correct = False
            if ground_truth_args is not None and is_grounded:
                arg_ext_correct = rule_sel_correct and (mem.arguments == ground_truth_args)
                arg_sem_correct = rule_sel_correct and are_arguments_semantically_equivalent(
                    rule_selected, mem.arguments, ground_truth_args
                )

            if not is_grounded:
                latency = (time.perf_counter() - t0) * 1000.0
                return ReasoningResult(
                    question=question,
                    ground_truth_rule=expected_rule,
                    expected_answer=expected_answer,
                    mode=mode,
                    status="abstained" if should_abstain else "failed",
                    rule_selected=rule_selected,
                    rule_selection_correct=rule_sel_correct,
                    arg_extraction_correct=arg_ext_correct,
                    arg_semantic_correct=arg_sem_correct,
                    answer_correct=should_abstain,
                    abstained=True,
                    abstention_correct=should_abstain,
                    working_memory=mem,
                    attempts=[
                        AttemptRecord(
                            attempt_index=1,
                            raw_completion=raw_trace,
                            parsed_rule=rule_selected,
                            working_memory=mem,
                            error_feedback=mem.error,
                        )
                    ],
                    latency_ms=latency,
                )

            exec_res = self.executor.execute(rule_selected, mem.arguments, verify=True)
            if exec_res.success and exec_res.verified:
                mem.bind_result(exec_res.result, verified=True)
                final_answer = str(exec_res.result)
                ans_correct = (
                    expected_answer is not None
                    and final_answer.lower() == expected_answer.lower()
                )
                status = "success" if ans_correct else "failed"
            else:
                mem.reject(exec_res.error or "Verification failed")
                final_answer = None
                ans_correct = False
                status = "abstained" if should_abstain else "failed"

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
                arg_semantic_correct=arg_sem_correct,
                answer_correct=ans_correct if not should_abstain else (status == "abstained"),
                abstained=(status == "abstained"),
                abstention_correct=(status == "abstained") if should_abstain else True,
                working_memory=mem,
                attempts=[
                    AttemptRecord(
                        attempt_index=1,
                        raw_completion=raw_trace,
                        parsed_rule=rule_selected,
                        parsed_args=mem.arguments,
                        execution_result=exec_res,
                        working_memory=mem,
                    )
                ],
                latency_ms=latency,
            )

        # Autoregressive tool call fallback
        latency = (time.perf_counter() - t0) * 1000.0
        return ReasoningResult(
            question=question,
            ground_truth_rule=expected_rule,
            expected_answer=expected_answer,
            mode=mode,
            status="failed",
            latency_ms=latency,
        )

    def solve_multistep(
        self,
        question: str,
        expected_plan: ExecutionPlan | None,
        expected_answer: str | None,
        mode: str = "deterministic_template",
        max_retries: int = 2,
        should_abstain: bool = False,
    ) -> ReasoningResult:
        """Solve a multi-step composition problem across M16 planning configurations:

        1. 'neural_only': Direct Mini-GPT completion without tools.
        2. 'oracle': Oracle-generated plan executed deterministically.
        3. 'deterministic_template': Deterministic Template Planner with grounded execution.
        4. 'model_plan': Mini-GPT generated structured plan with deterministic execution (no retry).
        5. 'model_plan_retry': Mini-GPT plan generation with bounded verification feedback loop.
        """
        t0 = time.perf_counter()

        # =================================================================
        # Configuration 1: Neural Only Direct Completion
        # =================================================================
        if mode == "neural_only":
            prompt = f"<|user|>\n{question}\n<|assistant|>\n"
            raw_ans = self._generate(prompt, max_tokens=32)
            clean_ans = raw_ans.replace("<|eos|>", "").strip()
            is_correct = (
                expected_answer is not None
                and clean_ans.lower() == expected_answer.lower()
            )
            latency = (time.perf_counter() - t0) * 1000.0
            return ReasoningResult(
                question=question,
                ground_truth_rule="multistep_direct",
                expected_answer=expected_answer,
                mode=mode,
                status="success" if is_correct else "failed",
                final_answer=clean_ans,
                answer_correct=is_correct,
                abstained=False,
                abstention_correct=not should_abstain,
                num_steps=1,
                attempts=[AttemptRecord(attempt_index=1, raw_completion=raw_ans)],
                latency_ms=latency,
            )

        # =================================================================
        # Configuration 2: Oracle Plan Execution
        # =================================================================
        if mode == "oracle":
            if should_abstain:
                latency = (time.perf_counter() - t0) * 1000.0
                return ReasoningResult(
                    question=question,
                    ground_truth_rule="oracle_plan",
                    expected_answer=expected_answer,
                    mode=mode,
                    status="abstained",
                    plan=expected_plan,
                    plan_valid=True,
                    plan_exact_match=True,
                    answer_correct=True,
                    abstained=True,
                    abstention_correct=True,
                    num_steps=len(expected_plan.steps) if expected_plan else 0,
                    latency_ms=latency,
                )

            if expected_plan is None:
                raise ValueError("expected_plan required for oracle multi-step mode")

            plan_res = self.plan_executor.execute(expected_plan)
            final_ans = str(plan_res.final_result) if plan_res.success else None
            is_correct = (
                plan_res.success
                and expected_answer is not None
                and str(plan_res.final_result).lower() == expected_answer.lower()
            )
            latency = (time.perf_counter() - t0) * 1000.0
            return ReasoningResult(
                question=question,
                ground_truth_rule="oracle_plan",
                expected_answer=expected_answer,
                mode=mode,
                status="success" if is_correct else "failed",
                final_answer=final_ans,
                plan=expected_plan,
                plan_valid=True,
                plan_exact_match=True,
                answer_correct=is_correct,
                abstained=False,
                abstention_correct=not should_abstain,
                working_memory=plan_res.working_memory,
                step_results=plan_res.step_results,
                num_steps=len(expected_plan.steps),
                attempts=[
                    AttemptRecord(
                        attempt_index=1,
                        raw_completion=expected_plan.to_text_format(),
                        parsed_plan=expected_plan,
                        execution_result=plan_res,
                        working_memory=plan_res.working_memory,
                    )
                ],
                latency_ms=latency,
            )

        # =================================================================
        # Configuration 3: Deterministic Template Planner (Arm C)
        # =================================================================
        if mode == "deterministic_template":
            try:
                plan = DeterministicTemplatePlanner.plan(question)
                plan_valid = True
                plan_error = None
            except AbstentionException as e:
                # Safely abstained
                latency = (time.perf_counter() - t0) * 1000.0
                mem = WorkingMemory()
                mem.reject(str(e))
                return ReasoningResult(
                    question=question,
                    ground_truth_rule="deterministic_plan",
                    expected_answer=expected_answer,
                    mode=mode,
                    status="abstained",
                    plan=None,
                    plan_valid=False,
                    plan_exact_match=False,
                    answer_correct=should_abstain,
                    abstained=True,
                    abstention_correct=should_abstain,
                    working_memory=mem,
                    attempts=[AttemptRecord(attempt_index=1, raw_completion="Abstain", error_feedback=str(e))],
                    latency_ms=latency,
                )
            except Exception as e:
                plan = None
                plan_valid = False
                plan_error = str(e)

            if not plan:
                latency = (time.perf_counter() - t0) * 1000.0
                return ReasoningResult(
                    question=question,
                    ground_truth_rule="deterministic_plan",
                    expected_answer=expected_answer,
                    mode=mode,
                    status="abstained" if should_abstain else "failed",
                    plan=None,
                    plan_valid=False,
                    plan_exact_match=False,
                    answer_correct=should_abstain,
                    abstained=True,
                    abstention_correct=should_abstain,
                    attempts=[AttemptRecord(attempt_index=1, raw_completion="Plan failed", error_feedback=plan_error)],
                    latency_ms=latency,
                )

            plan_res = self.plan_executor.execute(plan)
            final_ans = str(plan_res.final_result) if plan_res.success else None
            is_correct = (
                plan_res.success
                and expected_answer is not None
                and str(plan_res.final_result).lower() == expected_answer.lower()
            )
            is_exact = (
                expected_plan is not None
                and len(plan.steps) == len(expected_plan.steps)
                and [s.operation for s in plan.steps] == [s.operation for s in expected_plan.steps]
            )
            latency = (time.perf_counter() - t0) * 1000.0
            return ReasoningResult(
                question=question,
                ground_truth_rule="deterministic_plan",
                expected_answer=expected_answer,
                mode=mode,
                status="success" if is_correct else ("abstained" if should_abstain else "failed"),
                final_answer=final_ans,
                plan=plan,
                plan_valid=plan_valid,
                plan_exact_match=is_exact,
                answer_correct=is_correct if not should_abstain else (not plan_res.success),
                abstained=not plan_res.success,
                abstention_correct=(not plan_res.success) if should_abstain else True,
                working_memory=plan_res.working_memory,
                step_results=plan_res.step_results,
                num_steps=len(plan.steps),
                attempts=[
                    AttemptRecord(
                        attempt_index=1,
                        raw_completion=plan.to_text_format(),
                        parsed_plan=plan,
                        execution_result=plan_res,
                        working_memory=plan_res.working_memory,
                    )
                ],
                latency_ms=latency,
            )

        # =================================================================
        # Configurations 4 & 5: Model Plan Generation (+ Optional Feedback Retry)
        # =================================================================
        base_prompt = f"<|user|>\n{question}\n<|assistant|>\n"
        current_prompt = base_prompt
        retries_allowed = max_retries if mode == "model_plan_retry" else 0
        attempt_idx = 0

        final_answer = None
        status = "failed"
        plan_parsed: ExecutionPlan | None = None
        plan_valid = False
        plan_exact = False
        ans_correct = False
        step_results: dict[str, Any] = {}
        attempts: list[AttemptRecord] = []
        last_mem: WorkingMemory | None = None

        while attempt_idx <= retries_allowed:
            attempt_idx += 1
            raw_comp = self._generate(current_prompt, max_tokens=96)
            record = AttemptRecord(attempt_index=attempt_idx, raw_completion=raw_comp)

            # Check for model abstention
            if raw_comp.strip().lower().startswith("abstain"):
                record.error_feedback = "Model abstained"
                attempts.append(record)
                status = "abstained"
                break

            try:
                plan_parsed = parse_plan(raw_comp)
                record.parsed_plan = plan_parsed
                plan_valid = True

                if expected_plan is not None:
                    plan_exact = (
                        len(plan_parsed.steps) == len(expected_plan.steps)
                        and [s.operation for s in plan_parsed.steps] == [s.operation for s in expected_plan.steps]
                    )

                # Execute plan
                plan_res = self.plan_executor.execute(plan_parsed)
                record.execution_result = plan_res
                record.working_memory = plan_res.working_memory
                last_mem = plan_res.working_memory

                if plan_res.success:
                    final_answer = str(plan_res.final_result)
                    step_results = plan_res.step_results
                    ans_correct = (
                        expected_answer is not None
                        and final_answer.lower() == expected_answer.lower()
                    )
                    status = "success"
                    attempts.append(record)
                    break
                else:
                    error_feedback = f"Plan execution failed at step {plan_res.failed_step_index}: {plan_res.error}"
                    record.error_feedback = error_feedback

            except Exception as e:
                error_feedback = f"Plan syntax/validation error: {str(e)}"
                record.error_feedback = error_feedback

            attempts.append(record)

            # Bounded retry feedback: strictly never leak expected answer!
            if attempt_idx <= retries_allowed:
                feedback = (
                    f"{raw_comp}\n<|system|>\nError: {record.error_feedback}. "
                    f"Please provide a corrected plan in format: 1. var = op(args...) [units]\\n<|assistant|>\n"
                )
                current_prompt = current_prompt + feedback
                if len(self.tokenizer.encode(current_prompt)) > self.model.config.context_length - 60:
                    current_prompt = (
                        f"<|user|>\n{question}\n<|system|>\nPrevious plan error: {record.error_feedback}. "
                        f"Provide corrected plan.\\n<|assistant|>\n"
                    )

        latency = (time.perf_counter() - t0) * 1000.0
        return ReasoningResult(
            question=question,
            ground_truth_rule="model_plan",
            expected_answer=expected_answer,
            mode=mode,
            status=status,
            final_answer=final_answer,
            plan=plan_parsed,
            plan_valid=plan_valid,
            plan_exact_match=plan_exact,
            answer_correct=ans_correct if not should_abstain else (status == "abstained"),
            abstained=(status == "abstained"),
            abstention_correct=(status == "abstained") if should_abstain else True,
            working_memory=last_mem,
            step_results=step_results,
            num_steps=len(plan_parsed.steps) if plan_parsed else 1,
            attempts=attempts,
            total_retries=attempt_idx - 1,
            latency_ms=latency,
        )
