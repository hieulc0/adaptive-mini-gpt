"""Procedural Dataset and Structured Supervision for M20 Neural Scaling.

Converts multi-step reasoning problems into structured procedural traces with:
- Canonical grounded slots
- Variable dependencies ($step1, $step2)
- Stepwise intermediate arithmetic calculations
- Explicit final-answer prediction
- Target-only loss masking (ignoring prompt tokens in causal LM loss)
- Unseen operand distributions, carry-heavy arithmetic, and depth ladders
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.rules.grounding import format_available_slots
from adaptive_mini_gpt.rules.memory import GroundedSlot, WorkingMemory
from adaptive_mini_gpt.rules.multistep_dataset import (
    MultiStepProblem,
    generate_addition_depth_ladder,
    generate_multistep_problems,
)
from adaptive_mini_gpt.rules.plan import ExecutionPlan, PlanStep


@dataclass
class ProceduralStep:
    """A single structured procedural reasoning step."""

    step_index: int
    output_var: str
    operation: str
    args: dict[str, Any]
    intermediate_calc: str
    result: Any
    units: str | None = None


@dataclass
class ProceduralExample:
    """A structured multi-step procedural training or evaluation example."""

    question: str
    slots: dict[str, GroundedSlot]
    steps: list[ProceduralStep]
    final_answer: str
    archetype: str
    depth: int
    unseen_operands: bool = False
    has_complex_carry: bool = False

    def format_prompt(self) -> str:
        """Format input prompt up to the start of procedural generation."""
        slots_text = format_available_slots(self.slots).strip()
        user_body = f"Problem: {self.question}\n{slots_text}" if slots_text else f"Problem: {self.question}"
        return f"<|user|>\n{user_body}\n<|assistant|>\nProcedure:\n"

    def format_target(self) -> str:
        """Format assistant response containing intermediate calculations and final answer."""
        lines = []
        for s in self.steps:
            args_formatted = ", ".join(
                f"{k}={repr(v) if isinstance(v, str) else v}" for k, v in sorted(s.args.items())
            )
            u_str = f" [{s.units}]" if s.units else ""
            lines.append(f"{s.step_index}. {s.output_var} = {s.operation}({args_formatted}) -> {s.intermediate_calc}{u_str}")
        lines.append(f"Final Answer: {self.final_answer}")
        return "\n".join(lines) + "<|eos|>\n"

    def format_full_sequence(self) -> str:
        """Format full prompt + target sequence."""
        return self.format_prompt() + self.format_target()


def _resolve_arg_val(arg: Any, mem: dict[str, Any], slots: dict[str, GroundedSlot]) -> Any:
    """Resolve argument from intermediate variables or grounded slots."""
    if isinstance(arg, str) and arg.startswith("$"):
        var_name = arg[1:].strip()
        if var_name in mem:
            return mem[var_name]
        if var_name in slots:
            return slots[var_name].value
        return arg
    return arg


def problem_to_procedural_example(problem: MultiStepProblem) -> ProceduralExample:
    """Convert a MultiStepProblem into a structured ProceduralExample."""
    plan = problem.expected_slot_plan if problem.expected_slot_plan is not None else problem.expected_plan
    mem_vars: dict[str, Any] = {}
    procedural_steps: list[ProceduralStep] = []

    for idx, step in enumerate(plan.steps, start=1):
        res_args: dict[str, Any] = {}
        for k, v in step.args.items():
            res_args[k] = _resolve_arg_val(v, mem_vars, problem.slots)

        op = step.operation
        calc_str = ""
        result: Any = None

        if op == "add":
            val_a = res_args.get("a", 0)
            val_b = res_args.get("b", 0)
            try:
                num_a = int(val_a)
                num_b = int(val_b)
                result = num_a + num_b
                calc_str = f"{num_a} + {num_b} = {result}"
            except (ValueError, TypeError):
                result = str(val_a) + str(val_b)
                calc_str = f"{val_a} + {val_b} = {result}"

        elif op == "average_speed":
            d_val = res_args.get("distance", 0)
            t_val = res_args.get("duration", 1)
            try:
                num_d = float(d_val)
                num_t = float(t_val)
                res_num = num_d / num_t if num_t != 0 else 0.0
                result = int(res_num) if res_num.is_integer() else round(res_num, 2)
                calc_str = f"{int(num_d) if num_d.is_integer() else num_d} / {int(num_t) if num_t.is_integer() else num_t} = {result}"
            except (ValueError, TypeError):
                result = 0
                calc_str = f"{d_val} / {t_val} = 0"

        elif op == "compare":
            val_a = res_args.get("a", 0)
            val_b = res_args.get("b", 0)
            try:
                num_a = float(val_a)
                num_b = float(val_b)
                if num_a > num_b:
                    result = "heavier"
                    calc_str = f"{val_a} > {val_b} -> heavier"
                elif num_a < num_b:
                    result = "lighter"
                    calc_str = f"{val_a} < {val_b} -> lighter"
                else:
                    result = "equal"
                    calc_str = f"{val_a} == {val_b} -> equal"
            except (ValueError, TypeError):
                result = "unknown"
                calc_str = f"{val_a} vs {val_b} -> unknown"

        elif op == "reverse":
            val_t = str(res_args.get("text", ""))
            result = val_t[::-1]
            calc_str = f"reverse('{val_t}') = {result}"

        else:
            result = "unknown"
            calc_str = f"{op} -> unknown"

        clean_var = step.output_var[1:] if step.output_var.startswith("$") else step.output_var
        mem_vars[clean_var] = result

        procedural_steps.append(
            ProceduralStep(
                step_index=idx,
                output_var=step.output_var,
                operation=op,
                args=step.args,
                intermediate_calc=calc_str,
                result=result,
                units=step.units,
            )
        )

    expected_ans = str(problem.expected_answer) if problem.expected_answer is not None else str(result)

    return ProceduralExample(
        question=problem.question,
        slots=problem.slots,
        steps=procedural_steps,
        final_answer=expected_ans,
        archetype=problem.archetype,
        depth=len(procedural_steps),
    )


class ProceduralDataset(Dataset):
    """PyTorch Dataset for causal language modeling of structured procedures with target-only loss."""

    def __init__(
        self,
        examples: list[ProceduralExample],
        tokenizer: BPETokenizer,
        context_length: int = 256,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.context_length = context_length

        self.input_ids_list: list[Tensor] = []
        self.labels_list: list[Tensor] = []

        self._encode_all()

    def _encode_all(self) -> None:
        for ex in self.examples:
            prompt_str = ex.format_prompt()
            target_str = ex.format_target()

            prompt_toks = self.tokenizer.encode(prompt_str)
            target_toks = self.tokenizer.encode(target_str)

            full_toks = prompt_toks + target_toks
            if len(full_toks) > self.context_length:
                # Truncate keeping prompt start and target end
                full_toks = full_toks[: self.context_length]
                prompt_len = min(len(prompt_toks), self.context_length)
            else:
                prompt_len = len(prompt_toks)

            # Target labels: mask prompt with -100
            labels = [-100] * prompt_len + full_toks[prompt_len:]

            self.input_ids_list.append(torch.tensor(full_toks, dtype=torch.long))
            self.labels_list.append(torch.tensor(labels, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids_list)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.input_ids_list[idx], self.labels_list[idx]


def collate_procedural_batch(
    batch: list[tuple[Tensor, Tensor]],
    pad_token_id: int = 0,
) -> tuple[Tensor, Tensor]:
    """Collate variable-length sequences with left/right padding."""
    max_len = max(len(x[0]) for x in batch)
    b_input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    b_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (inp, lab) in enumerate(batch):
        b_input_ids[i, : len(inp)] = inp
        b_labels[i, : len(lab)] = lab

    return b_input_ids, b_labels


def generate_unseen_carry_dataset(
    num_samples: int = 20,
    seed: int = 5003,
    split: str = "test",
) -> list[ProceduralExample]:
    """Generate chained addition problems with 3-4 digit numbers requiring cascading carries.
    
    Tests whether the model has acquired general multi-step addition or memorized small additions.
    """
    import random
    rng = random.Random(seed)
    examples: list[ProceduralExample] = []

    for idx in range(num_samples):
        # 3 operands with 3-4 digits designed to produce multiple carry transitions (e.g. 889 + 345 + 778)
        n1 = rng.randint(450, 999)
        n2 = rng.randint(450, 999)
        n3 = rng.randint(450, 999)

        slots = {
            "operand_1": GroundedSlot("operand_1", n1, None, str(n1), 0, len(str(n1)), type_name="int"),
            "operand_2": GroundedSlot("operand_2", n2, None, str(n2), 0, len(str(n2)), type_name="int"),
            "operand_3": GroundedSlot("operand_3", n3, None, str(n3), 0, len(str(n3)), type_name="int"),
        }

        s1_res = n1 + n2
        s2_res = s1_res + n3

        steps = [
            ProceduralStep(
                step_index=1,
                output_var="$step1",
                operation="add",
                args={"a": "$operand_1", "b": "$operand_2"},
                intermediate_calc=f"{n1} + {n2} = {s1_res}",
                result=s1_res,
                units=None,
            ),
            ProceduralStep(
                step_index=2,
                output_var="$step2",
                operation="add",
                args={"a": "$step1", "b": "$operand_3"},
                intermediate_calc=f"{s1_res} + {n3} = {s2_res}",
                result=s2_res,
                units=None,
            ),
        ]

        q = f"Calculate the sum of {n1}, {n2}, and {n3}."
        examples.append(
            ProceduralExample(
                question=q,
                slots=slots,
                steps=steps,
                final_answer=str(s2_res),
                archetype="chained_addition_carry_ood",
                depth=2,
                unseen_operands=True,
                has_complex_carry=True,
            )
        )

    return examples


def parse_procedural_completion(raw_text: str) -> dict[str, Any]:
    """Parse raw autoregressive text completion into structured procedure components.
    
    Extracts:
    - final_answer: str or None
    - parsed_steps: list of dicts with step_num, var, op, calc, result
    - is_valid: whether procedure followed expected step syntax
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    final_answer: str | None = None
    parsed_steps: list[dict[str, Any]] = []

    # Final answer regex pattern
    fa_match = re.search(r"Final Answer:\s*([^\n<|]+)", raw_text, re.IGNORECASE)
    if fa_match:
        final_answer = fa_match.group(1).strip()

    # Step regex pattern: e.g. "1. $step1 = add(a=$operand_1, b=$operand_2) -> 120 + 80 = 200"
    step_pattern = re.compile(
        r"^(\d+)\.\s+(\$\w+)\s*=\s*(\w+)\(([^)]*)\)(?:\s*->\s*([^\[\n]+))?(?:\s*\[([^\]]+)\])?"
    )

    for line in lines:
        m = step_pattern.match(line)
        if m:
            s_idx = int(m.group(1))
            s_var = m.group(2)
            s_op = m.group(3)
            s_args_str = m.group(4)
            s_calc = m.group(5).strip() if m.group(5) else ""
            s_units = m.group(6).strip() if m.group(6) else None

            # Extract result from calculation if present
            s_res = None
            if "=" in s_calc:
                s_res = s_calc.split("=")[-1].strip()
            elif "->" in s_calc:
                s_res = s_calc.split("->")[-1].strip()

            parsed_steps.append({
                "step_index": s_idx,
                "output_var": s_var,
                "operation": s_op,
                "raw_args": s_args_str,
                "calc": s_calc,
                "result": s_res,
                "units": s_units,
            })

    return {
        "final_answer": final_answer,
        "parsed_steps": parsed_steps,
        "num_steps": len(parsed_steps),
        "has_final_answer": final_answer is not None,
        "is_valid": len(parsed_steps) > 0 and final_answer is not None,
    }
