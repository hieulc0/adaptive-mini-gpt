"""Deterministic and Neural Multi-Step Planners with Bounded Plan Validation (M16).

Implements:
1. parse_plan: Robust parser for model-generated line-based and JSON plans.
2. DeterministicTemplatePlanner: Deconstructs multi-step composite queries into typed ExecutionPlans.
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

from adaptive_mini_gpt.rules.grounding import AbstentionException, GroundingError
from adaptive_mini_gpt.rules.plan import ExecutionPlan, PlanStep, PlanValidationError
from adaptive_mini_gpt.rules.registry import RULE_REGISTRY


def parse_plan(text: str) -> ExecutionPlan:
    """Robustly parse an ExecutionPlan from model-generated text (line-based or JSON)."""
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
    # 1. d_tot = add(a=120, b=90) [km]
    # 2. t_tot = add(a=2, b=3) [h]
    # 3. speed = average_speed(distance_km=$d_tot, time_hours=$t_tot) [km/h]
    # Result: $speed
    step_pattern = re.compile(
        r"^(?:\d+[\.:]\s*|Step\s*\d+[\.:]\s*)?([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)(?:\s*\[(.*?)\])?",
        re.IGNORECASE,
    )
    result_pattern = re.compile(r"^Result:\s*\$?([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)

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

            # Parse arguments
            args: dict[str, Any] = {}
            if args_str:
                # Custom safe parser handling $var and literals
                # Replace $var with "$var" so ast can parse as string literal
                subbed_args = re.sub(r"(?<=[=,\(\s])\$([a-zA-Z_][a-zA-Z0-9_]*)", r"'$_\1'", args_str)
                try:
                    tree = ast.parse(f"dummy({subbed_args})")
                    call_node = tree.body[0].value  # type: ignore[attr-defined]
                    for kw in call_node.keywords:
                        val = ast.literal_eval(kw.value)
                        # Re-substitute $_ back to $
                        if isinstance(val, str) and val.startswith("$_"):
                            val = "$" + val[2:]
                        elif isinstance(val, list):
                            val = [("$" + item[2:] if isinstance(item, str) and item.startswith("$_") else item) for item in val]
                        args[kw.arg] = val
                except Exception:
                    # Fallback token split
                    for part in args_str.split(","):
                        if "=" in part:
                            k, v = part.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
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
        """Analyze question text and generate validated ExecutionPlan or raise AbstentionException."""
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
            ("travels" in q_lower or "covers" in q_lower or "drives" in q_lower or "flies" in q_lower)
            and "km" in q_lower
            and ("hour" in q_lower or "hrs" in q_lower or "h" in q_lower)
            and ("then" in q_lower or "another" in q_lower or "next" in q_lower or "whole journey" in q_lower or "overall" in q_lower)
        ):
            return cls._plan_multi_segment_speed(q)

        # Case C: Chained addition (3+ operands)
        if any(kw in q_lower for kw in ("sum of", "plus", "total of", "add")) and (
            "and" in q_lower or "," in q_lower or "together" in q_lower
        ):
            numbers = list(re.finditer(r"\b\d+(?:\.\d+)?\b", q))
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
    def _plan_multi_segment_speed(cls, question: str) -> ExecutionPlan:
        """Deconstruct multi-segment journey into distance sum, duration sum, and kinematics rate."""
        # Find all segment pairs: (distance_km, duration_hours)
        # Match patterns like: "120 km in 2 hours", "covers 90 km in 3 hours"
        seg_pat = re.compile(
            r"(-?\d+(?:\.\d+)?)\s*(?:km|kilometers?)\s*(?:in|for|over)\s*(-?\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b",
            re.IGNORECASE,
        )
        segments = list(seg_pat.finditer(question))

        if not segments:
            # Try inverted pattern: "2 hours to cover 120 km"
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

        # Check preconditions
        for d, t in parsed_segs:
            if t <= 0:
                raise AbstentionException(f"Invalid duration in segment: {t} <= 0 violates physical kinematics")
            if d < 0:
                raise AbstentionException(f"Invalid distance in segment: {d} < 0 violates physical constraints")

        steps: list[PlanStep] = []

        if len(parsed_segs) == 2:
            (d1, t1), (d2, t2) = parsed_segs[0], parsed_segs[1]
            # Step 1: d_total
            steps.append(
                PlanStep(
                    output_var="d_total",
                    operation="add",
                    args={"a": d1, "b": d2},
                    units="km",
                )
            )
            # Step 2: t_total
            steps.append(
                PlanStep(
                    output_var="t_total",
                    operation="add",
                    args={"a": t1, "b": t2},
                    units="h",
                )
            )
            # Step 3: overall speed
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
            # 5-step plan
            steps.append(PlanStep(output_var="d_12", operation="add", args={"a": d1, "b": d2}, units="km"))
            steps.append(PlanStep(output_var="d_total", operation="add", args={"a": "$d_12", "b": d3}, units="km"))
            steps.append(PlanStep(output_var="t_12", operation="add", args={"a": t1, "b": t2}, units="h"))
            steps.append(PlanStep(output_var="t_total", operation="add", args={"a": "$t_12", "b": t3}, units="h"))
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
            nums = nums[:5]  # bound to max 5

        steps: list[PlanStep] = []
        # Step 1: add first two
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
        # Check for contradictions
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

        # Target entities
        target_pat = rf"(?:who|which one)\s+is\s+(?:[a-zA-Z]+),\s*{ent_pat}\s+or\s+{ent_pat}"
        tm = re.search(target_pat, question, re.IGNORECASE)
        if not tm:
            tm = re.search(rf"Between\s+{ent_pat}\s+and\s+{ent_pat}", question, re.IGNORECASE)

        if not tm:
            raise AbstentionException("Could not identify target entities for comparison")

        target_a = tm.group(1).strip()
        target_b = tm.group(2).strip()

        # Step 1: compare target entities using all relations
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
