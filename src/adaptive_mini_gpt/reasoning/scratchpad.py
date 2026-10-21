"""Structured Scratchpads, Direct Targets, and Recurrent Steps (M31 Phase 2).

Defines representations and verifiable formats across three core domains:
  1. Arithmetic: Column-by-column digit and carry traces (lengths 2 to 32 digits).
  2. Symbolic Logic: Step-by-step modus ponens & negation deductions (depths 2 to 5 hops).
  3. Language Comprehension: Passage evidence extraction and structured grounding plans.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Any


class ApproachType(str, enum.Enum):
    DIRECT = "direct"
    SCRATCHPAD = "scratchpad"
    RECURRENT = "recurrent"


class ReasoningDomain(str, enum.Enum):
    ARITHMETIC = "arithmetic"
    LOGIC = "logic"
    LANGUAGE = "language"


@dataclass
class ReasoningSample:
    """Unified benchmark sample containing Direct, Scratchpad, and Recurrent representations."""

    sample_id: str
    domain: ReasoningDomain
    prompt_direct: str
    target_direct: str
    prompt_scratchpad: str
    target_scratchpad: str
    recurrent_steps: list[tuple[str, str]] = field(default_factory=list)
    expected_final_answer: str = ""
    intermediate_ground_truth: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedReasoningCompletion:
    """Traceable parse of an autoregressive completion from any approach."""

    raw_text: str
    intermediate_steps: list[str]
    parsed_final_answer: str
    has_final_answer_tag: bool
    is_valid_syntax: bool
    first_error_step: int | None = None


# =========================================================================
# 1. Arithmetic Reasoning Builders
# =========================================================================

def build_arithmetic_reasoning_sample(a: int, b: int) -> ReasoningSample:
    """Build a multi-digit addition sample with Direct, Scratchpad, and Recurrent traces."""
    total = a + b
    str_a = str(a)
    str_b = str(b)
    expected_ans = str(total)

    # 1. Direct Target
    prompt_direct = f"<|user|>\nAdd: {a} + {b} =\n<|assistant|>\n"
    target_direct = f"{expected_ans}<|eos|>\n"

    # 2. Scratchpad Target (Column-by-column digit & carry trace)
    # Reverse to process from least-significant column (col 0)
    rev_a = [int(d) for d in reversed(str_a)]
    rev_b = [int(d) for d in reversed(str_b)]
    max_len = max(len(rev_a), len(rev_b))

    steps_text: list[str] = []
    recurrent_steps: list[tuple[str, str]] = []
    carry = 0

    for col in range(max_len):
        da = rev_a[col] if col < len(rev_a) else 0
        db = rev_b[col] if col < len(rev_b) else 0
        col_sum = da + db + carry
        write_d = col_sum % 10
        next_carry = col_sum // 10

        step_str = f"Step {col + 1}: col {col}: {da} + {db} + {carry} = {col_sum} -> write {write_d}, carry {next_carry}"
        steps_text.append(step_str)

        # Recurrent single-step pair
        is_last = (col == max_len - 1) and (next_carry == 0)
        halt_tag = " | HALT" if is_last else " | CONT"
        rec_prompt = f"<|user|>\nCol: {da} + {db} + {carry}\n<|assistant|>\n"
        rec_target = f"{col_sum} > {write_d} : {next_carry}{halt_tag}<|eos|>\n"
        recurrent_steps.append((rec_prompt, rec_target))

        carry = next_carry

    # Final carry column if non-zero
    if carry > 0:
        step_str = f"Step {max_len + 1}: col {max_len}: 0 + 0 + {carry} = {carry} -> write {carry}, carry 0"
        steps_text.append(step_str)

        rec_prompt = f"<|user|>\nCol: 0 + 0 + {carry}\n<|assistant|>\n"
        rec_target = f"{carry} > {carry} : 0 | HALT<|eos|>\n"
        recurrent_steps.append((rec_prompt, rec_target))

    prompt_scratchpad = f"<|user|>\nAdd with steps: {a} + {b}\n<|assistant|>\n"
    scratchpad_body = "\n".join(steps_text)
    target_scratchpad = f"{scratchpad_body}\nFinal Answer: {expected_ans}<|eos|>\n"

    return ReasoningSample(
        sample_id=f"arith_{a}_{b}",
        domain=ReasoningDomain.ARITHMETIC,
        prompt_direct=prompt_direct,
        target_direct=target_direct,
        prompt_scratchpad=prompt_scratchpad,
        target_scratchpad=target_scratchpad,
        recurrent_steps=recurrent_steps,
        expected_final_answer=expected_ans,
        intermediate_ground_truth=steps_text,
        metadata={"a": a, "b": b, "digits": max(len(str_a), len(str_b)), "has_carry": (carry > 0 or "carry 1" in target_scratchpad)},
    )


# =========================================================================
# 2. Symbolic Logic Reasoning Builders
# =========================================================================

def build_logic_reasoning_sample(
    var_chain: list[str],
    negations: list[bool],
    query_var: str,
    initial_fact: tuple[str, bool] = ("A", True),
) -> ReasoningSample:
    """Build a multi-hop chained logic deduction sample with intermediate steps."""
    # Deduce truth values step by step
    facts: dict[str, bool] = {initial_fact[0]: initial_fact[1]}
    rules_text: list[str] = []
    steps_text: list[str] = []
    recurrent_steps: list[tuple[str, str]] = []

    curr_val = initial_fact[1]
    for i in range(len(var_chain) - 1):
        src = var_chain[i]
        tgt = var_chain[i + 1]
        is_neg = negations[i]

        if is_neg:
            rule_str = f"If {src} then not {tgt}"
            next_val = not curr_val
        else:
            rule_str = f"If {src} then {tgt}"
            next_val = curr_val

        rules_text.append(rule_str)
        facts[tgt] = next_val

        val_str = "True" if next_val else "False"
        step_str = f"Step {i + 1}: {src} is {curr_val} and ({rule_str}) -> {tgt} is {val_str}"
        steps_text.append(step_str)

        is_last = (i == len(var_chain) - 2)
        halt_tag = " | HALT" if is_last else " | CONT"
        rec_prompt = f"<|user|>\nState: {src}={curr_val} | Apply: {rule_str}\n<|assistant|>\n"
        rec_target = f"{tgt}={next_val}{halt_tag}<|eos|>\n"
        recurrent_steps.append((rec_prompt, rec_target))

        curr_val = next_val

    expected_bool = facts.get(query_var, False)
    expected_ans = "True" if expected_bool else "False"

    facts_prompt_str = f"{initial_fact[0]} is {initial_fact[1]}"
    rules_prompt_str = ". ".join(rules_text) + "."

    prompt_direct = f"<|user|>\nFacts: {facts_prompt_str}. Rules: {rules_prompt_str} Query: Is {query_var} True?\n<|assistant|>\n"
    target_direct = f"{expected_ans}<|eos|>\n"

    prompt_scratchpad = f"<|user|>\nDeduce with steps: Facts: {facts_prompt_str}. Rules: {rules_prompt_str} Query: Is {query_var} True?\n<|assistant|>\n"
    scratchpad_body = "\n".join(steps_text)
    target_scratchpad = f"{scratchpad_body}\nFinal Answer: {expected_ans}<|eos|>\n"

    return ReasoningSample(
        sample_id=f"logic_{len(var_chain)}_hop_{query_var}",
        domain=ReasoningDomain.LOGIC,
        prompt_direct=prompt_direct,
        target_direct=target_direct,
        prompt_scratchpad=prompt_scratchpad,
        target_scratchpad=target_scratchpad,
        recurrent_steps=recurrent_steps,
        expected_final_answer=expected_ans,
        intermediate_ground_truth=steps_text,
        metadata={"hop_depth": len(var_chain) - 1, "query_var": query_var, "expected_bool": expected_bool},
    )


# =========================================================================
# 3. Language Grounded Reasoning Builders
# =========================================================================

def build_language_reasoning_sample(
    sample_id: str,
    context: str,
    question: str,
    evidence_span: str,
    extracted_fact: str,
    expected_answer: str,
) -> ReasoningSample:
    """Build a passage-grounded reading comprehension sample with evidence span extraction."""
    clean_ctx = context.strip()
    clean_q = question.strip()
    clean_ev = evidence_span.strip()
    clean_fact = extracted_fact.strip()
    clean_ans = expected_answer.strip()

    prompt_direct = f"<|user|>\nContext: {clean_ctx}\nQuestion: {clean_q}\n<|assistant|>\nAnswer: "
    target_direct = f"{clean_ans}<|eos|>\n"

    prompt_scratchpad = f"<|user|>\nContext: {clean_ctx}\nQuestion: {clean_q}\nReason with evidence:\n<|assistant|>\n"
    step_1 = f"Evidence: \"{clean_ev}\""
    step_2 = f"Extracted Fact: {clean_fact}"
    scratchpad_body = f"{step_1}\n{step_2}"
    target_scratchpad = f"{scratchpad_body}\nFinal Answer: {clean_ans}<|eos|>\n"

    # For language recurrent steps, Step 1 is Evidence Extraction, Step 2 is Grounded Fact Inference
    rec_step_1 = (
        f"<|user|>\nContext: {clean_ctx}\nExtract evidence for: {clean_q}\n<|assistant|>\n",
        f"Evidence: \"{clean_ev}\" | CONT<|eos|>\n",
    )
    rec_step_2 = (
        f"<|user|>\nEvidence: \"{clean_ev}\"\nConclude answer for: {clean_q}\n<|assistant|>\n",
        f"Answer: {clean_ans} | HALT<|eos|>\n",
    )

    return ReasoningSample(
        sample_id=sample_id,
        domain=ReasoningDomain.LANGUAGE,
        prompt_direct=prompt_direct,
        target_direct=target_direct,
        prompt_scratchpad=prompt_scratchpad,
        target_scratchpad=target_scratchpad,
        recurrent_steps=[rec_step_1, rec_step_2],
        expected_final_answer=clean_ans,
        intermediate_ground_truth=[step_1, step_2],
        metadata={"has_evidence": bool(clean_ev), "is_abstention": (clean_ans.lower() == "insufficient information.")},
    )


# =========================================================================
# 4. Parsing and Error Identification
# =========================================================================

def parse_reasoning_completion(
    raw_text: str,
    approach: ApproachType,
    domain: ReasoningDomain,
) -> ParsedReasoningCompletion:
    """Parse output text from model rollout across Direct, Scratchpad, and Recurrent approaches."""
    clean = raw_text.strip()

    if approach == ApproachType.DIRECT:
        # In direct generation, the entire first line is the final answer
        line0 = clean.split("\n")[0].strip()
        # Strip common label prefixes if model hallucinated them
        for prefix in ("Answer:", "Final Answer:", "Result:"):
            if line0.startswith(prefix):
                line0 = line0[len(prefix):].strip()
        return ParsedReasoningCompletion(
            raw_text=raw_text,
            intermediate_steps=[],
            parsed_final_answer=line0,
            has_final_answer_tag=False,
            is_valid_syntax=bool(line0),
            first_error_step=None,
        )

    if approach == ApproachType.SCRATCHPAD:
        lines = [line.strip() for line in clean.split("\n") if line.strip()]
        steps: list[str] = []
        final_answer = ""
        has_tag = False

        for line in lines:
            if "Final Answer:" in line:
                final_answer = line.split("Final Answer:")[-1].strip()
                has_tag = True
            elif "Answer:" in line and not final_answer:
                final_answer = line.split("Answer:")[-1].strip()
                has_tag = True
            elif line.startswith("Step ") or line.startswith("Evidence:") or line.startswith("Extracted Fact:"):
                steps.append(line)

        # Fallback if no explicit tag: last non-empty line
        if not final_answer and lines:
            final_answer = lines[-1].strip()

        # Check for syntax validity
        is_valid = bool(steps and final_answer and has_tag)

        return ParsedReasoningCompletion(
            raw_text=raw_text,
            intermediate_steps=steps,
            parsed_final_answer=final_answer,
            has_final_answer_tag=has_tag,
            is_valid_syntax=is_valid,
            first_error_step=None,
        )

    if approach == ApproachType.RECURRENT:
        # Single recurrent transition step
        line0 = clean.split("\n")[0].strip()
        return ParsedReasoningCompletion(
            raw_text=raw_text,
            intermediate_steps=[line0],
            parsed_final_answer=line0,
            has_final_answer_tag=("HALT" in clean),
            is_valid_syntax=bool(line0),
            first_error_step=None,
        )

    return ParsedReasoningCompletion(
        raw_text=raw_text,
        intermediate_steps=[],
        parsed_final_answer=clean,
        has_final_answer_tag=False,
        is_valid_syntax=bool(clean),
        first_error_step=None,
    )
