"""Grounded Argument Extraction, Provenance Linking, and Typed Slot Registry (M15/M16/M17).

Extracts rule arguments and typed slots directly from input text by binding source spans:
- Arm 2: Deterministic rule-aware argument parsing with exact span tracking.
- Arm 3: Source-span selection with exact text slice copying.
- Phase 1 (M17): Typed Grounded Slot Extraction with stable slot IDs and unit validation.
- Rejection/abstention on malformed, ambiguous, contradictory, or unsupported inputs.

Safety:
- Disallows arbitrary Python execution (no eval, no exec).
- Bound to strictly typed argument schemas and precondition validation.
"""

from __future__ import annotations

import re
from typing import Any

from adaptive_mini_gpt.rules.memory import ArgumentProvenance, GroundedSlot, WorkingMemory
from adaptive_mini_gpt.rules.registry import RULE_REGISTRY, SchemaValidationError


class GroundingError(Exception):
    """Raised when input text cannot be validly grounded into rule arguments."""


class AbstentionException(GroundingError):
    """Raised when the system must explicitly abstain from execution due to invalid/contradictory input."""


# =====================================================================
# Deterministic Rule Selector
# =====================================================================


class DeterministicRuleSelector:
    """Lightweight rule intent classifier based on syntactic patterns."""

    @classmethod
    def select_rule(cls, text: str) -> str:
        """Deterministically predict rule from question text."""
        t = text.lower()

        # 1. Reverse patterns
        if any(
            kw in t
            for kw in (
                "backward",
                "reverse",
                "invert",
                "in reverse order",
                "spell the word",
                "reverse of",
            )
        ):
            return "reverse"

        # 2. Average speed patterns
        if any(
            kw in t
            for kw in (
                "average speed",
                "speed of a vehicle",
                "find the average speed",
                "calculate average speed",
                "what is its average speed",
                "what is the average speed",
                "km/h",
            )
        ) or (
            ("travels" in t or "covers" in t or "flies" in t or "taking" in t)
            and ("km" in t or "kilometer" in t)
            and ("hour" in t or "hr" in t or "h" in t)
        ):
            return "average_speed"

        # 3. Add patterns
        if any(
            kw in t
            for kw in (
                "plus",
                "sum of",
                "add ",
                "total of",
                "added to",
                "compute ",
                "calculate the sum",
            )
        ) and ("+" in t or "plus" in t or "and" in t or "to" in t):
            # Check it's not a speed or comparison question
            if "than" not in t and "speed" not in t:
                return "add"

        # 4. Compare patterns
        if any(
            kw in t
            for kw in (
                "taller",
                "faster",
                "heavier",
                "older",
                "stronger",
                "shorter",
                "slower",
                "lighter",
                "younger",
                "weaker",
            )
        ) and ("than" in t or "between" in t or "who is" in t or "which one is" in t):
            return "compare"

        # Fallback keyword checks
        if "+" in t:
            return "add"

        raise GroundingError(f"Could not deterministically determine rule intent for text: '{text[:60]}'")


# =====================================================================
# Deterministic Argument Grounder (Arm 2)
# =====================================================================


class DeterministicArgumentGrounder:
    """Parses and grounds arguments deterministically with full provenance tracking."""

    @classmethod
    def ground(cls, rule_name: str, question: str) -> WorkingMemory:
        """Ground rule arguments from question text into WorkingMemory."""
        rule_name = rule_name.strip().lower()
        if rule_name not in RULE_REGISTRY:
            raise GroundingError(f"Cannot ground unknown rule: '{rule_name}'")

        mem = WorkingMemory(operation=rule_name)

        if rule_name == "reverse":
            return cls._ground_reverse(question, mem)
        elif rule_name == "add":
            return cls._ground_add(question, mem)
        elif rule_name == "compare":
            return cls._ground_compare(question, mem)
        elif rule_name == "average_speed":
            return cls._ground_average_speed(question, mem)

        mem.reject(f"No deterministic grounder implemented for rule: '{rule_name}'")
        return mem

    @classmethod
    def _ground_reverse(cls, question: str, mem: WorkingMemory) -> WorkingMemory:
        """Ground string sequence to reverse."""
        # 1. Match quoted string: '...' or "..."
        m_quote = re.search(r"['\"]([a-zA-Z0-9_-]+)['\"]", question)
        if m_quote:
            s_val = m_quote.group(1)
            span = (m_quote.start(1), m_quote.end(1))
            mem.bind_argument("s", s_val, s_val, span, "str", extraction_method="regex_span")
            return mem

        # 2. Match unquoted string: after word/string/sequence (excluding stop words)
        m_word = re.search(r"\b(?:word|string|sequence)\s+([a-zA-Z0-9_-]+)\b", question, re.IGNORECASE)
        if m_word:
            s_val = m_word.group(1)
            if s_val.lower() not in {"in", "the", "a", "an", "of", "to", "backward", "backwards", "reverse", "order"}:
                span = (m_word.start(1), m_word.end(1))
                mem.bind_argument("s", s_val, s_val, span, "str", extraction_method="regex_span")
                return mem

        mem.reject("Missing or unidentifiable string to reverse")
        return mem

    @classmethod
    def _ground_add(cls, question: str, mem: WorkingMemory) -> WorkingMemory:
        """Ground addition operands with span tracking and distracting-number filtering."""
        # Check for explicit arithmetic patterns first
        # Pattern 1: X plus Y
        m_plus = re.search(r"(\b\d+(?:\.\d+)?\b)\s+(?:plus|\+)\s+(\b\d+(?:\.\d+)?\b)", question, re.IGNORECASE)
        if m_plus:
            a_str, b_str = m_plus.group(1), m_plus.group(2)
            a_val = float(a_str) if "." in a_str else int(a_str)
            b_val = float(b_str) if "." in b_str else int(b_str)
            mem.bind_argument("a", a_val, a_str, (m_plus.start(1), m_plus.end(1)), "int" if isinstance(a_val, int) else "float", "regex_span")
            mem.bind_argument("b", b_val, b_str, (m_plus.start(2), m_plus.end(2)), "int" if isinstance(b_val, int) else "float", "regex_span")
            return mem

        # Pattern 2: sum/total of X and Y
        m_sum = re.search(r"(?:sum|total)\s+of\s+(\b\d+(?:\.\d+)?\b)\s+and\s+(\b\d+(?:\.\d+)?\b)", question, re.IGNORECASE)
        if m_sum:
            a_str, b_str = m_sum.group(1), m_sum.group(2)
            a_val = float(a_str) if "." in a_str else int(a_str)
            b_val = float(b_str) if "." in b_str else int(b_str)
            mem.bind_argument("a", a_val, a_str, (m_sum.start(1), m_sum.end(1)), "int" if isinstance(a_val, int) else "float", "regex_span")
            mem.bind_argument("b", b_val, b_str, (m_sum.start(2), m_sum.end(2)), "int" if isinstance(b_val, int) else "float", "regex_span")
            return mem

        # Pattern 3: Add X and Y together / Add X to Y
        m_add = re.search(r"\badd\s+(\b\d+(?:\.\d+)?\b)\s+(?:and|to)\s+(\b\d+(?:\.\d+)?\b)", question, re.IGNORECASE)
        if m_add:
            a_str, b_str = m_add.group(1), m_add.group(2)
            a_val = float(a_str) if "." in a_str else int(a_str)
            b_val = float(b_str) if "." in b_str else int(b_str)
            mem.bind_argument("a", a_val, a_str, (m_add.start(1), m_add.end(1)), "int" if isinstance(a_val, int) else "float", "regex_span")
            mem.bind_argument("b", b_val, b_str, (m_add.start(2), m_add.end(2)), "int" if isinstance(b_val, int) else "float", "regex_span")
            return mem

        # Pattern 4: General numbers if exactly 2 exist
        all_numbers = list(re.finditer(r"\b\d+(?:\.\d+)?\b", question))
        if len(all_numbers) == 2:
            m1, m2 = all_numbers[0], all_numbers[1]
            a_str, b_str = m1.group(0), m2.group(0)
            a_val = float(a_str) if "." in a_str else int(a_str)
            b_val = float(b_str) if "." in b_str else int(b_str)
            mem.bind_argument("a", a_val, a_str, (m1.start(), m1.end()), "int" if isinstance(a_val, int) else "float", "regex_span")
            mem.bind_argument("b", b_val, b_str, (m2.start(), m2.end()), "int" if isinstance(b_val, int) else "float", "regex_span")
            return mem

        if len(all_numbers) < 2:
            mem.reject("Missing operands for addition (fewer than 2 numbers found)")
        else:
            mem.reject(f"Ambiguous addition query with {len(all_numbers)} numbers and no clear operator anchoring")
        return mem

    @classmethod
    def _ground_compare(cls, question: str, mem: WorkingMemory) -> WorkingMemory:
        """Ground transitive comparison premises and target entities."""
        adj_pairs = {
            "taller": "shorter",
            "faster": "slower",
            "heavier": "lighter",
            "older": "younger",
            "stronger": "weaker",
        }
        all_adjs = list(adj_pairs.keys()) + list(adj_pairs.values())
        adj_pattern = "|".join(all_adjs)

        ent_pat = r"([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)"
        premise_pat = rf"{ent_pat}\s+is\s+({adj_pattern})\s+than\s+{ent_pat}"

        relations: list[str] = []
        premise_matches = list(re.finditer(premise_pat, question))

        if not premise_matches:
            mem.reject("No valid comparison premises found in text")
            return mem

        for pm in premise_matches:
            ent1 = pm.group(1).strip()
            adj = pm.group(2).strip().lower()
            ent2 = pm.group(3).strip()

            ent1 = re.sub(r"^(?:If|Suppose|Given that|Given|Assuming)\s+", "", ent1, flags=re.IGNORECASE).strip()

            if adj in adj_pairs:
                relations.append(f"{ent1} > {ent2}")
            else:
                relations.append(f"{ent2} > {ent1}")

        target_pat = rf"(?:who|which one)\s+is\s+(?:[a-zA-Z]+),\s*{ent_pat}\s+or\s+{ent_pat}"
        tm = re.search(target_pat, question, re.IGNORECASE)
        if not tm:
            target_pat_alt = rf"Between\s+{ent_pat}\s+and\s+{ent_pat}"
            tm = re.search(target_pat_alt, question, re.IGNORECASE)

        if tm:
            target_a = tm.group(1).strip()
            target_b = tm.group(2).strip()
            mem.bind_argument("entity_a", target_a, target_a, (tm.start(1), tm.end(1)), "str", "regex_span")
            mem.bind_argument("entity_b", target_b, target_b, (tm.start(2), tm.end(2)), "str", "regex_span")
            mem.bind_argument("relations", relations, str(relations), (premise_matches[0].start(), premise_matches[-1].end()), "list[str]\", \"derived_premises")
            return mem

        mem.reject("Could not identify target entities to compare")
        return mem

    @classmethod
    def _ground_average_speed(cls, question: str, mem: WorkingMemory) -> WorkingMemory:
        """Ground distance and duration values with unit parsing and precondition validation."""
        d_m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:km|kilometers?)\b", question, re.IGNORECASE)
        t_m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", question, re.IGNORECASE)

        if not d_m:
            mem.reject("Missing or invalid distance specification (expected km or kilometers)")
            return mem
        if not t_m:
            mem.reject("Missing or invalid duration specification (expected hours or hrs)")
            return mem

        d_str, t_str = d_m.group(1), t_m.group(1)
        d_val = float(d_str)
        t_val = float(t_str)

        if t_val <= 0:
            mem.reject(f"Invalid duration: time must be strictly positive (>0), got {t_val}")
            return mem
        if d_val < 0:
            mem.reject(f"Invalid distance: distance must be non-negative (>=0), got {d_val}")
            return mem

        mem.bind_argument("distance_km", d_val, d_str, (d_m.start(1), d_m.end(1)), "float", "regex_span")
        mem.bind_argument("time_hours", t_val, t_str, (t_m.start(1), t_m.end(1)), "float", "regex_span")
        return mem


# =====================================================================
# Source-Span Selection with Exact Text Slicing (Arm 3)
# =====================================================================


class SpanCopyArgumentGrounder:
    """Extracts arguments by selecting candidate text offsets and copying exact substrings."""

    @classmethod
    def ground(cls, rule_name: str, question: str) -> WorkingMemory:
        """Ground arguments by locating candidate entity spans and copying exact text slices."""
        rule_name = rule_name.strip().lower()
        mem = WorkingMemory(operation=rule_name)

        if rule_name == "reverse":
            quotes = list(re.finditer(r"['\"]([a-zA-Z0-9_-]+)['\"]", question))
            if quotes:
                span_match = quotes[0]
                slice_text = question[span_match.start(1) : span_match.end(1)]
                mem.bind_argument(
                    "s",
                    slice_text,
                    slice_text,
                    (span_match.start(1), span_match.end(1)),
                    "str",
                    extraction_method="span_copy",
                )
                return mem
            mem.reject("No valid candidate string span found for reverse")
            return mem

        elif rule_name == "add":
            num_spans = list(re.finditer(r"\b\d+(?:\\.\d+)?\b", question))
            plus_match = re.search(r"\b(?:plus|\+|sum of|and|total of)\b", question, re.IGNORECASE)
            if plus_match and len(num_spans) >= 2:
                anchor = plus_match.start()
                sorted_spans = sorted(num_spans, key=lambda s: abs(s.start() - anchor))
                chosen_spans = sorted(sorted_spans[:2], key=lambda s: s.start())
                span1, span2 = chosen_spans[0], chosen_spans[1]

                raw1 = question[span1.start() : span1.end()]
                raw2 = question[span2.start() : span2.end()]
                val1 = float(raw1) if "." in raw1 else int(raw1)
                val2 = float(raw2) if "." in raw2 else int(raw2)

                mem.bind_argument("a", val1, raw1, (span1.start(), span1.end()), "int" if isinstance(val1, int) else "float", "span_copy")
                mem.bind_argument("b", val2, raw2, (span2.start(), span2.end()), "int" if isinstance(val2, int) else "float", "span_copy")
                return mem

            mem.reject("Could not identify valid numeric operand spans for addition")
            return mem

        elif rule_name == "average_speed":
            d_span = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:km|kilometers?)\b", question, re.IGNORECASE)
            t_span = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", question, re.IGNORECASE)

            if d_span and t_span:
                d_raw = question[d_span.start(1) : d_span.end(1)]
                t_raw = question[t_span.start(1) : t_span.end(1)]
                d_val = float(d_raw)
                t_val = float(t_raw)

                if t_val <= 0 or d_val < 0:
                    mem.reject("Precondition violation in span-copied kinematics values")
                    return mem

                mem.bind_argument("distance_km", d_val, d_raw, (d_span.start(1), d_span.end(1)), "float", "span_copy")
                mem.bind_argument("time_hours", t_val, t_raw, (t_span.start(1), t_span.end(1)), "float", "span_copy")
                return mem

            mem.reject("Could not find matching unit spans for distance and duration")
            return mem

        elif rule_name == "compare":
            return DeterministicArgumentGrounder._ground_compare(question, mem)

        mem.reject(f"Span copy not supported for rule '{rule_name}'")
        return mem


# =====================================================================
# Typed Grounded Slot Extractor (M17 Phase 1)
# =====================================================================


def format_available_slots(slots: dict[str, GroundedSlot]) -> str:
    """Format slot list into clean string representation for prompt injection."""
    if not slots:
        return "Available slots: None"
    lines = ["Available slots:"]
    for sid, s in slots.items():
        u_str = f" {s.unit}" if s.unit else ""
        lines.append(f"- ${sid}: {s.value}{u_str}")
    return "\n".join(lines)


class SlotExtractor:
    """Extracts typed, source-grounded input slots from question text (M17 Phase 1).

    Produces stable canonical slot identifiers and typed values:
    - Kinematics: distance_1, duration_1, distance_2, duration_2, ...
    - Arithmetic: operand_1, operand_2, operand_3, ...
    - Comparison: entity_1, metric_1, entity_2, metric_2, ...
    - Exact source-span offsets, physical units, and provenance.
    - Rejects invalid dimensions, negative durations, or ambiguous references.
    """

    @classmethod
    def extract(cls, question: str) -> dict[str, GroundedSlot]:
        """Extract typed GroundedSlot registry from question text."""
        q = question.strip()
        q_lower = q.lower()

        # Step 1: Detect dimensional incompatibilities
        if re.search(r"\badd\b.*?\bkm\b.*?\b(?:hours?|hrs?|h)\b", q_lower) or re.search(
            r"\bwhat is\b.*?\bkm\b\s*plus\s*.*?(?:hours?|hrs?|h)\b", q_lower
        ):
            raise AbstentionException("Incompatible physical dimensions: cannot add distance to duration")

        # Step 2: Multi-segment kinematics / Average Speed
        is_speed_query = "average speed" in q_lower or (
            ("km" in q_lower or "kilometer" in q_lower)
            and ("hour" in q_lower or "hrs" in q_lower or "h" in q_lower)
        )
        if is_speed_query:
            try:
                return cls._extract_speed_slots(q)
            except AbstentionException:
                raise
            except Exception:
                pass

        # Step 3: Arithmetic chained addition
        is_arithmetic = any(kw in q_lower for kw in ("sum of", "plus", "total of", "add")) and (
            "+" in q_lower or "and" in q_lower or "," in q_lower or "together" in q_lower or "plus" in q_lower
        )
        if is_arithmetic and "than" not in q_lower:
            return cls._extract_arithmetic_slots(q)

        # Step 4: Comparison
        if any(kw in q_lower for kw in ("taller", "faster", "heavier", "older", "stronger", "shorter", "slower")):
            return cls._extract_comparison_slots(q)

        # Step 5: Reverse
        if any(kw in q_lower for kw in ("reverse", "backward", "invert")):
            return cls._extract_reverse_slots(q)

        # Fallback numeric extraction
        return cls._extract_fallback_numbers(q)

    @classmethod
    def _extract_speed_slots(cls, question: str) -> dict[str, GroundedSlot]:
        """Extract kinematics slots (distance_i, duration_i) from journey description."""
        slots: dict[str, GroundedSlot] = {}

        # Standard pattern: "X km in Y hours"
        seg_pat = re.compile(
            r"(-?\d+(?:\.\d+)?)\s*(?:km|kilometers?)\s*(?:in|for|over)\s*(-?\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b",
            re.IGNORECASE,
        )
        segments = list(seg_pat.finditer(question))

        if segments:
            for idx, m in enumerate(segments, start=1):
                d_str, t_str = m.group(1), m.group(2)
                d_val, t_val = float(d_str), float(t_str)
                if t_val <= 0:
                    raise AbstentionException(f"Invalid duration in segment {idx}: {t_val} <= 0 violates physical kinematics")
                if d_val < 0:
                    raise AbstentionException(f"Invalid distance in segment {idx}: {d_val} < 0 violates physical constraints")

                slots[f"distance_{idx}"] = GroundedSlot(
                    slot_id=f"distance_{idx}",
                    value=d_val,
                    unit="km",
                    source_text=m.group(1) + " km",
                    start_idx=m.start(1),
                    end_idx=m.end(1),
                    extraction_method="deterministic_kinematics",
                    verified=True,
                    type_name="float",
                )
                slots[f"duration_{idx}"] = GroundedSlot(
                    slot_id=f"duration_{idx}",
                    value=t_val,
                    unit="h",
                    source_text=m.group(2) + " hours",
                    start_idx=m.start(2),
                    end_idx=m.end(2),
                    extraction_method="deterministic_kinematics",
                    verified=True,
                    type_name="float",
                )
            return slots

        # Inverted pattern: "Y hours to cover X km"
        seg_pat_inv = re.compile(
            r"(-?\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\s*(?:to cover|traveling)\s*(-?\d+(?:\.\d+)?)\s*(?:km|kilometers?)\b",
            re.IGNORECASE,
        )
        inv_segments = list(seg_pat_inv.finditer(question))
        if inv_segments:
            for idx, m in enumerate(inv_segments, start=1):
                t_str, d_str = m.group(1), m.group(2)
                t_val, d_val = float(t_str), float(d_str)
                if t_val <= 0:
                    raise AbstentionException(f"Invalid duration in segment {idx}: {t_val} <= 0 violates physical kinematics")
                if d_val < 0:
                    raise AbstentionException(f"Invalid distance in segment {idx}: {d_val} < 0 violates physical constraints")

                slots[f"distance_{idx}"] = GroundedSlot(
                    slot_id=f"distance_{idx}",
                    value=d_val,
                    unit="km",
                    source_text=m.group(2) + " km",
                    start_idx=m.start(2),
                    end_idx=m.end(2),
                    extraction_method="deterministic_kinematics",
                    verified=True,
                    type_name="float",
                )
                slots[f"duration_{idx}"] = GroundedSlot(
                    slot_id=f"duration_{idx}",
                    value=t_val,
                    unit="h",
                    source_text=m.group(1) + " hours",
                    start_idx=m.start(1),
                    end_idx=m.end(1),
                    extraction_method="deterministic_kinematics",
                    verified=True,
                    type_name="float",
                )
            return slots

        # Single segment speed fallback
        d_m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:km|kilometers?)\b", question, re.IGNORECASE)
        t_m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", question, re.IGNORECASE)
        if d_m and t_m:
            d_val, t_val = float(d_m.group(1)), float(t_m.group(1))
            if t_val <= 0:
                raise AbstentionException(f"Invalid duration: {t_val} <= 0 violates physical kinematics")
            if d_val < 0:
                raise AbstentionException(f"Invalid distance: {d_val} < 0 violates physical constraints")

            slots["distance_1"] = GroundedSlot(
                slot_id="distance_1",
                value=d_val,
                unit="km",
                source_text=d_m.group(0),
                start_idx=d_m.start(1),
                end_idx=d_m.end(1),
                extraction_method="deterministic_kinematics",
                verified=True,
                type_name="float",
            )
            slots["duration_1"] = GroundedSlot(
                slot_id="duration_1",
                value=t_val,
                unit="h",
                source_text=t_m.group(0),
                start_idx=t_m.start(1),
                end_idx=t_m.end(1),
                extraction_method="deterministic_kinematics",
                verified=True,
                type_name="float",
            )
            return slots

        raise AbstentionException("Could not extract valid kinematics slots from journey description")

    @classmethod
    def _extract_arithmetic_slots(cls, question: str) -> dict[str, GroundedSlot]:
        """Extract arithmetic operand slots (operand_1, operand_2, ...) with distractor filtering."""
        slots: dict[str, GroundedSlot] = {}

        # Filter out 4-digit years like 'year 2024' or 'flight 204'
        all_numbers = list(re.finditer(r"\b\d+(?:\.\d+)?\b", question))
        filtered: list[re.Match[str]] = []
        for m in all_numbers:
            txt = m.group(0)
            if len(txt) == 4 and (txt.startswith("20") or txt.startswith("19")):
                prefix = question[max(0, m.start() - 15) : m.start()].lower()
                if any(w in prefix for w in ("year", "in the", "since", "during")):
                    continue
            filtered.append(m)

        if not filtered:
            raise AbstentionException("No valid numeric operands found for arithmetic")

        for idx, m in enumerate(filtered, start=1):
            val_str = m.group(0)
            val = float(val_str) if "." in val_str else int(val_str)
            slots[f"operand_{idx}"] = GroundedSlot(
                slot_id=f"operand_{idx}",
                value=val,
                unit=None,
                source_text=val_str,
                start_idx=m.start(),
                end_idx=m.end(),
                extraction_method="deterministic_arithmetic",
                verified=True,
                type_name="float" if isinstance(val, float) else "int",
            )
        return slots

    @classmethod
    def _extract_comparison_slots(cls, question: str) -> dict[str, GroundedSlot]:
        """Extract entities and metrics for comparison chains."""
        slots: dict[str, GroundedSlot] = {}
        ent_metric_pat = re.compile(
            r"([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+is\s+(\d+(?:\.\d+)?)\s*([a-zA-Z/]+)?",
            re.IGNORECASE,
        )
        matches = list(ent_metric_pat.finditer(question))
        if matches:
            for idx, m in enumerate(matches, start=1):
                ent = m.group(1).strip()
                val_str = m.group(2)
                val = float(val_str) if "." in val_str else int(val_str)
                unit = m.group(3).strip() if m.group(3) else None
                slots[f"entity_{idx}"] = GroundedSlot(
                    slot_id=f"entity_{idx}",
                    value=ent,
                    unit=None,
                    source_text=ent,
                    start_idx=m.start(1),
                    end_idx=m.end(1),
                    extraction_method="deterministic_comparison",
                    verified=True,
                    type_name="str",
                )
                slots[f"metric_{idx}"] = GroundedSlot(
                    slot_id=f"metric_{idx}",
                    value=val,
                    unit=unit,
                    source_text=val_str,
                    start_idx=m.start(2),
                    end_idx=m.end(2),
                    extraction_method="deterministic_comparison",
                    verified=True,
                    type_name="float" if isinstance(val, float) else "int",
                )
            return slots

        ents = list(re.finditer(r"\b([A-Z][a-z]+)\b", question))
        if ents:
            seen: set[str] = set()
            idx = 1
            for em in ents:
                w = em.group(1)
                if w not in seen and w not in ("If", "Suppose", "Given", "Between", "Who", "Which"):
                    seen.add(w)
                    slots[f"entity_{idx}"] = GroundedSlot(
                        slot_id=f"entity_{idx}",
                        value=w,
                        unit=None,
                        source_text=w,
                        start_idx=em.start(),
                        end_idx=em.end(),
                        extraction_method="deterministic_comparison",
                        verified=True,
                        type_name="str",
                    )
                    idx += 1
            return slots

        raise AbstentionException("Could not extract comparison entities from text")

    @classmethod
    def _extract_reverse_slots(cls, question: str) -> dict[str, GroundedSlot]:
        """Extract text slot to reverse."""
        quotes = list(re.finditer(r"['\"]([a-zA-Z0-9_-]+)['\"]", question))
        if quotes:
            m = quotes[0]
            val = m.group(1)
            return {
                "text_1": GroundedSlot(
                    slot_id="text_1",
                    value=val,
                    unit=None,
                    source_text=val,
                    start_idx=m.start(1),
                    end_idx=m.end(1),
                    extraction_method="deterministic_string",
                    verified=True,
                    type_name="str",
                )
            }
        m_word = re.search(r"(?:reverse|invert|backward|reverse of)\s+([a-zA-Z0-9_-]+)", question, re.IGNORECASE)
        if m_word:
            val = m_word.group(1)
            return {
                "text_1": GroundedSlot(
                    slot_id="text_1",
                    value=val,
                    unit=None,
                    source_text=val,
                    start_idx=m_word.start(1),
                    end_idx=m_word.end(1),
                    extraction_method="deterministic_string",
                    verified=True,
                    type_name="str",
                )
            }
        raise AbstentionException("No target text found to reverse")

    @classmethod
    def _extract_fallback_numbers(cls, question: str) -> dict[str, GroundedSlot]:
        """Extract all numbers as generic operand slots."""
        matches = list(re.finditer(r"\b\d+(?:\.\d+)?\b", question))
        slots: dict[str, GroundedSlot] = {}
        for idx, m in enumerate(matches, start=1):
            val_str = m.group(0)
            val = float(val_str) if "." in val_str else int(val_str)
            slots[f"operand_{idx}"] = GroundedSlot(
                slot_id=f"operand_{idx}",
                value=val,
                unit=None,
                source_text=val_str,
                start_idx=m.start(),
                end_idx=m.end(),
                extraction_method="fallback_number",
                verified=True,
                type_name="float" if isinstance(val, float) else "int",
            )
        return slots
