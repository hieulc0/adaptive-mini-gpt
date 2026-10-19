"""Grounded Argument Extraction and Provenance Linking (M15).

Extracts rule arguments directly from input text by binding source spans:
- Arm 2: Deterministic rule-aware argument parsing with exact span tracking.
- Arm 3: Source-span selection with exact text slice copying.
- Rejection/abstention on malformed, ambiguous, contradictory, or unsupported inputs.

Safety:
- Disallows arbitrary Python execution (no eval, no exec).
- Bound to strictly typed argument schemas and precondition validation.
"""

from __future__ import annotations

import re
from typing import Any

from adaptive_mini_gpt.rules.memory import ArgumentProvenance, WorkingMemory
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
    """Extracts typed arguments with source spans from natural language prompts."""

    @classmethod
    def ground(cls, rule_name: str, question: str) -> WorkingMemory:
        """Ground question into working memory with full provenance."""
        rule_name = rule_name.strip().lower()
        if rule_name not in RULE_REGISTRY:
            raise SchemaValidationError(f"Unregistered rule '{rule_name}'")

        mem = WorkingMemory(operation=rule_name)

        if rule_name == "reverse":
            return cls._ground_reverse(question, mem)
        elif rule_name == "add":
            return cls._ground_add(question, mem)
        elif rule_name == "compare":
            return cls._ground_compare(question, mem)
        elif rule_name == "average_speed":
            return cls._ground_average_speed(question, mem)
        else:
            raise SchemaValidationError(f"No grounder implemented for rule '{rule_name}'")

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
        # Comparison adjectives and their antonyms
        adj_pairs = {
            "taller": "shorter",
            "faster": "slower",
            "heavier": "lighter",
            "older": "younger",
            "stronger": "weaker",
        }
        all_adjs = list(adj_pairs.keys()) + list(adj_pairs.values())
        adj_pattern = "|".join(all_adjs)

        # Entity can be single word or multiword title: e.g. "Mary Jane", "Ben", "Uncle Bob"
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

            # Strip leading discourse markers
            ent1 = re.sub(r"^(?:If|Suppose|Given that|Given|Assuming)\s+", "", ent1, flags=re.IGNORECASE).strip()

            # Normalize to >: if adj is standard (taller), ent1 > ent2; if inverse (shorter), ent2 > ent1
            if adj in adj_pairs:
                relations.append(f"{ent1} > {ent2}")
            else:
                relations.append(f"{ent2} > {ent1}")

        # Target entities to compare: "who is SUP, X or Y?" or "Between X and Y, who is SUP?"
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
            mem.bind_argument("relations", relations, str(relations), (premise_matches[0].start(), premise_matches[-1].end()), "list[str]", "derived_premises")
            return mem

        mem.reject("Could not identify target entities to compare")
        return mem

    @classmethod
    def _ground_average_speed(cls, question: str, mem: WorkingMemory) -> WorkingMemory:
        """Ground distance and duration values with unit parsing and precondition validation."""
        # Distance: [-]N km / [-]N kilometers
        d_m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:km|kilometers?)\b", question, re.IGNORECASE)
        # Time: [-]N hours / [-]N hrs / [-]N h
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

        # Precondition checks
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
            num_spans = list(re.finditer(r"\b\d+(?:\.\d+)?\b", question))
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
