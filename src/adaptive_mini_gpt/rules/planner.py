"""Deterministic and Neural Multi-Step Planners with Grounded Slot Binding (M16/M17).

Implements:
1. parse_plan: Robust parser for model-generated line-based and JSON plans with slot references ($slot_id).
2. DeterministicTemplatePlanner: Deconstructs multi-step composite queries into typed ExecutionPlans (with literal arguments or grounded slots).
3. Plan validation and error diagnostics for feedback retry loops.

Safety:
- Bounded to max 5 steps.
- Rejects arbitrary Python execution (no eval, no exec).
- Rejects cycles, self-dependencies, undefined variable references, and unit mismatches.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from adaptive_mini_gpt.rules.grounding import AbstentionException, GroundingError, SlotExtractor
from adaptive_mini_gpt.rules.memory import GroundedSlot
from adaptive_mini_gpt.rules.plan import ExecutionPlan, PlanStep, PlanValidationError
from adaptive_mini_gpt.rules.registry import RULE_REGISTRY


def parse_plan(text: str) -> ExecutionPlan:
    """Robustly parse an ExecutionPlan from model-generated text (line-based or JSON), supporting slots and variables."""
    cleaned = text.strip()
    if cleaned.endswith("<|eos|>"):
        cleaned = cleaned[:-len("<|eos|>")].strip()

    # Try 1: JSON plan format
    m_json = re.search(r"\{.*\"steps\".*\}", cleaned, re.DOTALL)
    if m_json:
        try:
            data = json.loads(m_json.group(0))
            if "steps" in data and isinstance(data["steps"], list):
                steps: list[PlanStep] = []
                for s_data in data["steps"]:
                    steps.append(
                        PlanStep(
                            output_var=s_data["output_var"],
                            operation=s_data["operation"],
                            args=s_data.get("args", {}),
                            units=s_data.get("units"),
                        )
                    )
                plan = ExecutionPlan(steps=steps, final_var=data.get("final_var"))
                plan.validate()
                return plan
        except Exception:
            pass

    # Try 2: Line-based format
    # Example:
    # 1. d_total = add(a=$distance_1, b=$distance_2) [km]
    # OR positional:
    # 1. d_total = add($distance_1, $distance_2) [km]
    # 2. t_total = add($duration_1, $duration_2) [h]
    # 3. overall_speed = average_speed(distance_km=$d_total, time_hours=$t_total) [km/h]
    # Result: $overall_speed
    step_pattern = re.compile(
        r"^(?:\d+[\.:]\s*|Step\s*\d+[\.:]\s*)?([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)(?:\s*\[(.*?)\])?",
        re.IGNORECASE,
    )
    result_pattern = re.compile(r"^Result:\s*\$?([a-zA-Z_][a-zA-Z0-9_\.]*)", re.IGNORECASE)

    steps: list[PlanStep] = []
    final_var: str | None = None

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("plan:"):
            continue

        # Check for final result line
        m_res = result_pattern.search(line)
        if m_res:
            final_var = m_res.group(1)
            continue

        m_step = step_pattern.search(line)
        if m_step:
            var_name = m_step.group(1)
            op_name = m_step.group(2).lower()
            args_str = m_step.group(3).strip()
            units = m_step.group(4).strip() if m_step.group(4) else None

            # Determine schema required argument keys if operation registered
            req_keys: list[str] = []
            if op_name in RULE_REGISTRY:
                req_keys = list(RULE_REGISTRY[op_name].args_schema.keys())

            # Parse arguments
            args: dict[str, Any] = {}
            if args_str:
                # Custom safe parser handling $var, $slot, $turn1.var and literals
                # Wrap $var into '$var' string literal so ast can parse cleanly
                subbed_args = re.sub(r"\$([a-zA-Z_][a-zA-Z0-9_\.]*)", lambda m: f"'{m.group(0)}'", args_str)
                try:
                    tree = ast.parse(f"dummy({subbed_args})")
                    call_node = tree.body[0].value  # type: ignore[attr-defined]

                    # 1. Positional arguments (mapped to schema keys)
                    for pos_idx, arg_node in enumerate(call_node.args):
                        arg_key = req_keys[pos_idx] if pos_idx < len(req_keys) else f"arg_{pos_idx + 1}"
                        val = ast.literal_eval(arg_node)
                        args[arg_key] = val

                    # 2. Keyword arguments
                    for kw in call_node.keywords:
                        val = ast.literal_eval(kw.value)
                        args[kw.arg] = val

                except Exception:
                    # Fallback token split
                    for idx, part in enumerate(args_str.split(",")):
                        part = part.strip()
                        if not part:
                            continue
                        if "=" in part:
                            k, v = part.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
                        else:
                            k = req_keys[idx] if idx < len(req_keys) else f"arg_{idx + 1}"
                            v = part.strip().strip("'\"")

                        if v.startswith("$"):
                            args[k] = v
                        else:
                            try:
                                args[k] = float(v) if "." in v else int(v)
                            except ValueError:
                                args[k] = v

            steps.append(PlanStep(output_var=var_name, operation=op_name, args=args, units=units))

    if not steps:
        raise PlanValidationError(f"Could not parse valid plan steps from: '{text[:120]}'")

    plan = ExecutionPlan(steps=steps, final_var=final_var)
    plan.validate()
    return plan


class DeterministicTemplatePlanner:
    """Synthesizes structured multi-step ExecutionPlans from natural-language queries."""

    @classmethod
    def plan(cls, question: str) -> ExecutionPlan:
        """Analyze question text and generate validated ExecutionPlan with literal arguments (M16 baseline)."""
        q = question.strip()
        q_lower = q.lower()

        # Check for abstention / malformed query triggers
        # Case A: Dimensional incompatibility: adding km and hours directly
        if re.search(r"\badd\b.*?\bkm\b.*?\b(?:hours?|hrs?|h)\b", q_lower) or re.search(
            r"\bwhat is\b.*?\bkm\b\s*plus\s*.*?\b(?:hours?|hrs?|h)\b", q_lower
        ):
            raise AbstentionException("Incompatible physical dimensions: cannot add distance to duration")

        # Case B: Multi-segment average speed
        if "average speed" in q_lower or (
            ("travel" in q_lower or "cover" in q_lower or "drive" in q_lower or "fly" in q_lower)
            and "km" in q_lower
            and ("hour" in q_lower or "hrs" in q_lower or "h" in q_lower)
            and ("then" in q_lower or "another" in q_lower or "next" in q_lower or "whole journey" in q_lower or "overall" in q_lower)
        ):
            return cls._plan_multi_segment_speed(q)

        # Case C: Chained addition (3+ operands)
        if any(kw in q_lower for kw in ("sum of", "plus", "total of", "add")) and (
            "and" in q_lower or "," in q_lower or "together" in q_lower
        ):
            numbers = list(re.finditer(r"\b\d+(?:\\.\d+)?\b", q))
            # Distractor year filtering
            filtered_numbers = [m for m in numbers if not (len(m.group(0)) == 4 and m.group(0).startswith("20"))]
            if len(filtered_numbers) >= 3:
                return cls._plan_chained_addition(filtered_numbers)

        # Case D: Multi-entity comparison chain (4+ entities)
        if "taller" in q_lower or "faster" in q_lower or "heavier" in q_lower or "older" in q_lower:
            return cls._plan_multi_entity_compare(q)

        # Case E: Single operation delegation as a 1-step plan
        return cls._plan_single_operation_as_plan(q)

    @classmethod
    def plan_with_slots(cls, question: str) -> tuple[ExecutionPlan, dict[str, GroundedSlot]]:
        """Synthesize structural plan referencing typed GroundedSlots (M17 Phase 1 & 2)."""
        slots = SlotExtractor.extract(question)
        q_lower = question.lower()

        # Case A: Kinematics / Average Speed
        if "distance_1" in slots and "duration_1" in slots:
            num_segs = sum(1 for k in slots if k.startswith("distance_"))
            steps: list[PlanStep] = []

            if num_segs == 1:
                steps.append(
                    PlanStep(
                        output_var="overall_speed",
                        operation="average_speed",
                        args={"distance_km": "$distance_1", "time_hours": "$duration_1"},
                        units="km/h",
                    )
                )
                plan = ExecutionPlan(steps=steps, final_var="overall_speed")
                plan.validate(allowed_slots=set(slots.keys()))
                return plan, slots

            if num_segs == 2:
                steps.append(PlanStep(output_var="d_total", operation="add", args={"a": "$distance_1", "b": "$distance_2"}, units="km"))
                steps.append(PlanStep(output_var="t_total", operation="add", args={"a": "$duration_1", "b": "$duration_2"}, units="h"))
                steps.append(
                    PlanStep(
                        output_var="overall_speed",
                        operation="average_speed",
                        args={"distance_km": "$d_total", "time_hours": "$t_total"},
                        units="km/h",
                    )
                )
                plan = ExecutionPlan(steps=steps, final_var="overall_speed")
                plan.validate(allowed_slots=set(slots.keys()))
                return plan, slots

            elif num_segs == 3:
                steps.append(PlanStep(output_var="d_12", operation="add", args={"a": "$distance_1", "b": "$distance_2"}, units="km"))
                steps.append(PlanStep(output_var="d_total", operation="add", args={"a": "$d_12", "b": "$distance_3"}, units="km"))
                steps.append(PlanStep(output_var="t_12", operation="add", args={"a": "$duration_1", "b": "$duration_2"}, units="h"))
                steps.append(PlanStep(output_var="t_total", operation="add", args={"a": "$t_12", "b": "$duration_3"}, units="h"))
                steps.append(
                    PlanStep(
                        output_var="overall_speed",
                        operation="average_speed",
                        args={"distance_km": "$d_total", "time_hours": "$t_total"},
                        units="km/h",
                    )
                )
                plan = ExecutionPlan(steps=steps, final_var="overall_speed")
                plan.validate(allowed_slots=set(slots.keys()))
                return plan, slots

        # Case B: Chained Addition
        operand_keys = sorted([k for k in slots if k.startswith("operand_")], key=lambda x: int(x.split("_")[1]))
        if len(operand_keys) >= 2:
            steps = []
            steps.append(PlanStep(output_var="sum_1", operation="add", args={"a": f"${operand_keys[0]}", "b": f"${operand_keys[1]}"}))
            last_var = "sum_1"
            for i in range(2, len(operand_keys)):
                cur_var = f"sum_{i}"
                steps.append(PlanStep(output_var=cur_var, operation="add", args={"a": f"${last_var}", "b": f"${operand_keys[i]}"}))
                last_var = cur_var
            plan = ExecutionPlan(steps=steps, final_var=last_var)
            plan.validate(allowed_slots=set(slots.keys()))
            return plan, slots

        # Case C: Compare or Reverse fallback
        lit_plan = cls.plan(question)
        return lit_plan, slots

    @classmethod
    def _plan_multi_segment_speed(cls, question: str) -> ExecutionPlan:
        """Deconstruct multi-segment journey into distance sum, duration sum, and kinematics rate."""
        seg_pat = re.compile(
            r"(-?\d+(?:\.\d+)?)\s*(?:km|kilometers?)\s*(?:in|for|over)\s*(-?\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b",
            re.IGNORECASE,
        )
        segments = list(seg_pat.finditer(question))

        if not segments:
            seg_pat_inv = re.compile(
                r"(-?\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\s*(?:to cover|traveling)\s*(-?\d+(?:\.\d+)?)\s*(?:km|kilometers?)\b",
                re.IGNORECASE,
            )
            inv_segments = list(seg_pat_inv.finditer(question))
            if inv_segments:
                parsed_segs = [(float(m.group(2)), float(m.group(1))) for m in inv_segments]
            else:
                raise AbstentionException("Could not identify valid journey segments for average speed calculation")
        else:
            parsed_segs = [(float(m.group(1)), float(m.group(2))) for m in segments]

        if len(parsed_segs) < 2:
            raise AbstentionException("Multi-segment average speed requires at least two journey segments")

        for d, t in parsed_segs:
            if t <= 0:
                raise AbstentionException(f"Invalid duration in segment: {t} <= 0 violates physical kinematics")
            if d < 0:
                raise AbstentionException(f"Invalid distance in segment: {d} < 0 violates physical constraints")

        steps: list[PlanStep] = []

        if len(parsed_segs) == 2:
            (d1, t1), (d2, t2) = parsed_segs[0], parsed_segs[1]
            steps.append(PlanStep(output_var="d_total", operation="add", args={"a": d1, "b": d2}, units="km"))
            steps.append(PlanStep(output_var="t_total", operation="add", args={"a": t1, "b": t2}, units="h"))
            steps.append(
                PlanStep(
                    output_var="overall_speed",
                    operation="average_speed",
                    args={"distance_km": "$d_total", "time_hours": "$t_total"},
                    units="km/h",
                )
            )
            return ExecutionPlan(steps=steps, final_var="overall_speed")

        elif len(parsed_segs) == 3:
            (d1, t1), (d2, t2), (d3, t3) = parsed_segs[0], parsed_segs[1], parsed_segs[2]
            steps.append(PlanStep(output_var="d_12", operation="add", args={"a": d1, "b": d2}, units="km"))
            steps.append(PlanStep(output_var="d_total", operation="add", args={"a": "$d_12", "b": d3}, units="km"))
            steps.append(PlanStep(output_var="t_12", operation="add", args={"a": t1, "b": t2}, units="h"))
            steps.append(PlanStep(output_var="t_total", operation="add", args={"a": "$t_12", "b": d3}, units="h"))
            steps.append(
                PlanStep(
                    output_var="overall_speed",
                    operation="average_speed",
                    args={"distance_km": "$d_total", "time_hours": "$t_total"},
                    units="km/h",
                )
            )
            return ExecutionPlan(steps=steps, final_var="overall_speed")

        raise AbstentionException(f"Unsupported number of segments: {len(parsed_segs)} (max 3 segments supported)")

    @classmethod
    def _plan_chained_addition(cls, number_matches: list[Any]) -> ExecutionPlan:
        """Compose chained addition across 3 to 5 operands."""
        nums = [float(m.group(0)) if "." in m.group(0) else int(m.group(0)) for m in number_matches]
        if len(nums) > 5:
            nums = nums[:5]

        steps: list[PlanStep] = []
        steps.append(PlanStep(output_var="sum_1", operation="add", args={"a": nums[0], "b": nums[1]}))

        last_var = "sum_1"
        for i in range(2, len(nums)):
            cur_var = f"sum_{i}"
            steps.append(PlanStep(output_var=cur_var, operation="add", args={"a": f"${last_var}", "b": nums[i]}))
            last_var = cur_var

        return ExecutionPlan(steps=steps, final_var=last_var)

    @classmethod
    def _plan_multi_entity_compare(cls, question: str) -> ExecutionPlan:
        """Chain multi-entity comparisons."""
        adj_pairs = {"taller": "shorter", "faster": "slower", "heavier": "lighter", "older": "younger", "stronger": "weaker"}
        all_adjs = list(adj_pairs.keys()) + list(adj_pairs.values())
        adj_pattern = "|".join(all_adjs)
        ent_pat = r"([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)"
        premise_pat = rf"{ent_pat}\s+is\s+({adj_pattern})\s+than\s+{ent_pat}"

        premise_matches = list(re.finditer(premise_pat, question))
        if len(premise_matches) < 2:
            raise AbstentionException("Multi-entity comparison requires multiple premises")

        relations = []
        for pm in premise_matches:
            ent1 = re.sub(r"^(?:If|Suppose|Given that|Given)\s+", "", pm.group(1).strip(), flags=re.IGNORECASE).strip()
            adj = pm.group(2).strip().lower()
            ent2 = pm.group(3).strip()
            if adj in adj_pairs:
                relations.append(f"{ent1} > {ent2}")
            else:
                relations.append(f"{ent2} > {ent1}")

        target_pat = rf"(?:who|which one)\s+is\s+(?:[a-zA-Z]+),\s*{ent_pat}\s+or\s+{ent_pat}"
        tm = re.search(target_pat, question, re.IGNORECASE)
        if not tm:
            tm = re.search(rf"Between\s+{ent_pat}\s+and\s+{ent_pat}", question, re.IGNORECASE)

        if not tm:
            raise AbstentionException("Could not identify target entities for comparison")

        target_a = tm.group(1).strip()
        target_b = tm.group(2).strip()

        step = PlanStep(
            output_var="comparison_result",
            operation="compare",
            args={"entity_a": target_a, "entity_b": target_b, "relations": relations},
        )
        return ExecutionPlan(steps=[step], final_var="comparison_result")

    @classmethod
    def _plan_single_operation_as_plan(cls, question: str) -> ExecutionPlan:
        """Wrap single-operation query as a 1-step plan."""
        from adaptive_mini_gpt.rules.grounding import DeterministicArgumentGrounder, DeterministicRuleSelector

        rule = DeterministicRuleSelector.select_rule(question)
        mem = DeterministicArgumentGrounder.ground(rule, question)
        if mem.status == "rejected" or mem.error:
            raise AbstentionException(mem.error or "Grounding rejected single operation")

        step = PlanStep(output_var="result", operation=rule, args=mem.arguments)
        return ExecutionPlan(steps=[step], final_var="result")
