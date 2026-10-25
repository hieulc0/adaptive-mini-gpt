"""Milestone M95 Phase 1: Cross-Domain Concept Curriculum.

Constructs paired, structurally isomorphic evaluation datasets across three distinct domains:
1. Domain A: Program Execution (Code syntax: statements, loops, conditionals, accumulators)
2. Domain B: Natural-Language Procedural Instructions (Verbal step-by-step descriptions)
3. Domain C: Symbolic Logic & Planning (Formal action preconditions, transitions, goals)

Underlying concepts share identical computational semantics in Z_16:
- State initialization & variable binding
- Arithmetic state updates (+, -)
- Conditional branching based on state comparisons (>, <, ==)
- Iterated / repeated actions (loops)
- Stateful accumulation across steps
- Unsupported / adversarial contradiction detection (epistemic abstention)
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass
from typing import Any

MODULUS = 16


class Domain(str, enum.Enum):
    """Representational domains sharing underlying computational semantics."""

    PROGRAM = "PROGRAM"
    NATURAL_LANGUAGE = "NATURAL_LANGUAGE"
    PLANNING = "PLANNING"


class ComputationalConcept(str, enum.Enum):
    """Abstract computational concepts instantiated across domains."""

    ASSIGN = "ASSIGN"
    ARITHMETIC = "ARITHMETIC"
    CONDITION = "CONDITION"
    LOOP = "LOOP"
    STATE_ACCUM = "STATE_ACCUM"
    NESTED_FILTER = "NESTED_FILTER"
    COMPOSED_ALGO = "COMPOSED_ALGO"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class CrossDomainExample:
    """An executable example in a specific representation domain."""

    example_id: str
    domain: Domain
    concept: ComputationalConcept
    raw_text: str
    token_ids: list[int]
    expected_output: int | None
    intermediate_trace: list[int]
    is_supported: bool = True
    requires_abstention: bool = False
    metadata: dict[str, Any] | None = None


class CrossDomainTokenizer:
    """Unified subword tokenizer across Code, Natural Language, and Planning."""

    PAD_ID = 0
    SOS_ID = 1
    EOS_ID = 2
    ABSTAIN_ID = 3

    # Common shared token vocabulary
    VOCAB: dict[str, int] = {
        "<pad>": 0,
        "<sos>": 1,
        "<eos>": 2,
        "<abstain>": 3,
        # Program keywords & symbols
        "=": 10,
        "+": 11,
        "-": 12,
        ">": 13,
        "<": 14,
        "==": 15,
        "for": 16,
        "in": 17,
        "if": 18,
        "else": 19,
        "print": 20,
        ":": 21,
        "[": 22,
        "]": 23,
        "(": 24,
        ")": 25,
        ",": 26,
        # Natural Language tokens
        "start": 30,
        "with": 31,
        "at": 32,
        "initialize": 33,
        "set": 34,
        "let": 35,
        "to": 36,
        "be": 37,
        "add": 38,
        "subtract": 39,
        "increase": 40,
        "decrease": 41,
        "by": 42,
        "from": 43,
        "is": 44,
        "greater": 45,
        "less": 46,
        "equal": 47,
        "than": 48,
        "exceeds": 49,
        "then": 50,
        "otherwise": 51,
        "repeat": 52,
        "times": 53,
        "what": 54,
        "result": 55,
        "value": 56,
        "total": 57,
        "counter": 58,
        # Planning / Symbolic tokens
        "init": 60,
        "act": 61,
        "cond": 62,
        "loop": 63,
        "goal": 64,
        "val": 65,
        "gt": 66,
        "lt": 67,
        "eq": 68,
        "->": 69,
        # Variables
        "a": 80,
        "b": 81,
        "c": 82,
        "x": 83,
        "y": 84,
        "acc": 85,
        "ctr": 86,
        "reg": 87,
        "state": 88,
    }

    # Assign integer literals 0..15 to token IDs 100..115
    for i in range(16):
        VOCAB[str(i)] = 100 + i

    REV_VOCAB = {v: k for k, v in VOCAB.items()}

    @classmethod
    def tokenize(cls, text: str) -> list[int]:
        """Convert arbitrary domain string into canonical token IDs."""
        clean = (
            text.replace("(", " ( ")
            .replace(")", " ) ")
            .replace("[", " [ ")
            .replace("]", " ] ")
            .replace(":", " : ")
            .replace(",", " , ")
            .replace("->", " -> ")
            .replace(".", " ")
            .lower()
        )
        words = clean.split()
        ids = [cls.SOS_ID]
        for w in words:
            if w in cls.VOCAB:
                ids.append(cls.VOCAB[w])
            elif w.isdigit() and int(w) < 16:
                ids.append(100 + int(w))
            else:
                ids.append(cls.ABSTAIN_ID)
        ids.append(cls.EOS_ID)
        return ids


class CrossDomainCurriculumGenerator:
    """Generator for isomorphic evaluation suites across Program, NL, and Planning."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_domain_suite(
        self, domain: Domain, n_train: int = 40, n_dev: int = 15, n_test: int = 25
    ) -> dict[str, list[CrossDomainExample]]:
        """Generate train, dev, held_out, and unsupported splits for a given domain."""
        train_ex: list[CrossDomainExample] = []
        dev_ex: list[CrossDomainExample] = []
        test_ex: list[CrossDomainExample] = []
        held_out_ex: list[CrossDomainExample] = []
        unsupported_ex: list[CrossDomainExample] = []

        rng = random.Random(self.seed + hash(domain) % 10000)

        # 1. Concept: Simple Variable State (Assign)
        for i in range(n_train):
            val = rng.randint(0, 15)
            train_ex.append(self._make_assign_example(domain, f"{domain}_assign_tr_{i}", val, "a"))
        for i in range(n_dev):
            val = rng.randint(0, 15)
            dev_ex.append(self._make_assign_example(domain, f"{domain}_assign_dev_{i}", val, "b"))
        for i in range(n_test):
            val = rng.randint(0, 15)
            test_ex.append(self._make_assign_example(domain, f"{domain}_assign_te_{i}", val, "x"))
        # Held-out variable names & values
        for i in range(n_test):
            val = rng.randint(0, 15)
            held_out_ex.append(self._make_assign_example(domain, f"{domain}_assign_held_{i}", val, "reg"))

        # 2. Concept: Arithmetic State Updates (Add/Sub)
        for i in range(n_train):
            v0 = rng.randint(0, 15)
            op = rng.choice(["+", "-"])
            k = rng.randint(1, 5)
            train_ex.append(self._make_arithmetic_example(domain, f"{domain}_arith_tr_{i}", v0, op, k))
        for i in range(n_dev):
            v0 = rng.randint(0, 15)
            op = rng.choice(["+", "-"])
            k = rng.randint(1, 5)
            dev_ex.append(self._make_arithmetic_example(domain, f"{domain}_arith_dev_{i}", v0, op, k))
        for i in range(n_test):
            v0 = rng.randint(0, 15)
            op = rng.choice(["+", "-"])
            k = rng.randint(1, 7)
            test_ex.append(self._make_arithmetic_example(domain, f"{domain}_arith_te_{i}", v0, op, k))
        # Held-out operand combinations
        for i in range(n_test):
            v0 = rng.randint(0, 15)
            op = rng.choice(["+", "-"])
            k = rng.choice([7, 9, 11])
            held_out_ex.append(self._make_arithmetic_example(domain, f"{domain}_arith_held_{i}", v0, op, k))

        # 3. Concept: Conditional Branching (If > / < / ==)
        for i in range(n_train):
            v0 = rng.randint(0, 15)
            thresh = rng.choice([2, 4, 6])
            cmp_op = rng.choice([">", "<"])
            train_ex.append(self._make_condition_example(domain, f"{domain}_cond_tr_{i}", v0, cmp_op, thresh))
        for i in range(n_dev):
            v0 = rng.randint(0, 15)
            thresh = rng.choice([3, 5])
            cmp_op = rng.choice([">", "<"])
            dev_ex.append(self._make_condition_example(domain, f"{domain}_cond_dev_{i}", v0, cmp_op, thresh))
        for i in range(n_test):
            v0 = rng.randint(0, 15)
            thresh = rng.choice([4, 8])
            cmp_op = rng.choice([">", "=="])
            test_ex.append(self._make_condition_example(domain, f"{domain}_cond_te_{i}", v0, cmp_op, thresh))
        # Held-out thresholds
        for i in range(n_test):
            v0 = rng.randint(0, 15)
            thresh = rng.choice([7, 10, 12])
            cmp_op = rng.choice([">", "<", "=="])
            held_out_ex.append(self._make_condition_example(domain, f"{domain}_cond_held_{i}", v0, cmp_op, thresh))

        # 4. Concept: Repeated Actions & Loops
        for i in range(n_train):
            v0 = rng.randint(0, 5)
            reps = rng.choice([2, 3])
            step = rng.randint(1, 4)
            train_ex.append(self._make_loop_example(domain, f"{domain}_loop_tr_{i}", v0, reps, step))
        for i in range(n_dev):
            v0 = rng.randint(0, 5)
            reps = rng.choice([2, 3])
            step = rng.randint(1, 4)
            dev_ex.append(self._make_loop_example(domain, f"{domain}_loop_dev_{i}", v0, reps, step))
        for i in range(n_test):
            v0 = rng.randint(0, 5)
            reps = 3
            step = rng.randint(1, 5)
            test_ex.append(self._make_loop_example(domain, f"{domain}_loop_te_{i}", v0, reps, step))
        # Held-out length generalization: 4 and 5 repetitions
        for i in range(n_test):
            v0 = rng.randint(0, 5)
            reps = rng.choice([4, 5])
            step = rng.randint(1, 4)
            held_out_ex.append(self._make_loop_example(domain, f"{domain}_loop_held_{i}", v0, reps, step))

        # 5. Concept: Stateful Accumulators & Nested Filter-Loops
        for i in range(n_train):
            v0 = rng.randint(0, 5)
            reps = 3
            thresh = 2
            train_ex.append(self._make_nested_example(domain, f"{domain}_nested_tr_{i}", v0, reps, thresh))
        for i in range(n_dev):
            v0 = rng.randint(0, 5)
            reps = 3
            thresh = 3
            dev_ex.append(self._make_nested_example(domain, f"{domain}_nested_dev_{i}", v0, reps, thresh))
        for i in range(n_test):
            v0 = rng.randint(0, 5)
            reps = 3
            thresh = 3
            test_ex.append(self._make_nested_example(domain, f"{domain}_nested_te_{i}", v0, reps, thresh))
        # Held-out complex compositions: 4 reps with unseen threshold 4
        for i in range(n_test):
            v0 = rng.randint(0, 5)
            reps = 4
            thresh = 4
            held_out_ex.append(self._make_nested_example(domain, f"{domain}_nested_held_{i}", v0, reps, thresh))

        # 6. Unsupported & Contradictory Tasks for Epistemic Abstention
        for i in range(15):
            unsupported_ex.append(self._make_unsupported_example(domain, f"{domain}_unsup_{i}", rng))

        return {
            "train": train_ex,
            "dev": dev_ex,
            "test": test_ex,
            "held_out": held_out_ex,
            "unsupported": unsupported_ex,
        }

    # -----------------------------------------------------------------------
    # Domain Renderers
    # -----------------------------------------------------------------------
    def _make_assign_example(self, domain: Domain, ex_id: str, val: int, var_name: str) -> CrossDomainExample:
        out = val % MODULUS
        trace = [out]
        if domain == Domain.PROGRAM:
            text = f"{var_name} = {val}\nprint({var_name})"
        elif domain == Domain.NATURAL_LANGUAGE:
            text = f"start with {var_name} at {val} . what is value of {var_name}"
        else:  # PLANNING
            text = f"init({var_name}, {val}) -> goal(val({var_name}))"

        return CrossDomainExample(
            example_id=ex_id,
            domain=domain,
            concept=ComputationalConcept.ASSIGN,
            raw_text=text,
            token_ids=CrossDomainTokenizer.tokenize(text),
            expected_output=out,
            intermediate_trace=trace,
            metadata={"val": out, "var_name": var_name},
        )

    def _make_arithmetic_example(
        self, domain: Domain, ex_id: str, v0: int, op: str, k: int
    ) -> CrossDomainExample:
        v0_mod = v0 % MODULUS
        k_mod = k % MODULUS
        out = (v0_mod + k_mod) % MODULUS if op == "+" else (v0_mod - k_mod) % MODULUS
        trace = [v0_mod, out]

        if domain == Domain.PROGRAM:
            text = f"a = {v0_mod}\na = a {op} {k_mod}\nprint(a)"
        elif domain == Domain.NATURAL_LANGUAGE:
            word = "add" if op == "+" else "subtract"
            prep = "to" if op == "+" else "from"
            text = f"start with total at {v0_mod} . {word} {k_mod} {prep} total . what is total"
        else:  # PLANNING
            act_str = "add" if op == "+" else "sub"
            text = f"init(state, {v0_mod}) -> act({act_str}, {k_mod}) -> goal(val(state))"

        return CrossDomainExample(
            example_id=ex_id,
            domain=domain,
            concept=ComputationalConcept.ARITHMETIC,
            raw_text=text,
            token_ids=CrossDomainTokenizer.tokenize(text),
            expected_output=out,
            intermediate_trace=trace,
            metadata={"v0": v0_mod, "op": op, "k": k_mod, "out": out},
        )

    def _make_condition_example(
        self, domain: Domain, ex_id: str, v0: int, cmp_op: str, thresh: int
    ) -> CrossDomainExample:
        v0_mod = v0 % MODULUS
        th_mod = thresh % MODULUS
        if cmp_op == ">":
            cond = v0_mod > th_mod
        elif cmp_op == "<":
            cond = v0_mod < th_mod
        else:
            cond = v0_mod == th_mod

        delta = 3 if cond else 1
        out = (v0_mod + delta) % MODULUS
        trace = [v0_mod, 1 if cond else 0, out]

        if domain == Domain.PROGRAM:
            text = f"a = {v0_mod}\nif a {cmp_op} {th_mod}:\n    a = a + 3\nelse:\n    a = a + 1\nprint(a)"
        elif domain == Domain.NATURAL_LANGUAGE:
            cmp_word = "greater than" if cmp_op == ">" else ("less than" if cmp_op == "<" else "equal to")
            text = f"start with counter at {v0_mod} . if counter is {cmp_word} {th_mod} then add 3 otherwise add 1 . what is counter"
        else:  # PLANNING
            cmp_token = "gt" if cmp_op == ">" else ("lt" if cmp_op == "<" else "eq")
            text = f"init(x, {v0_mod}) -> cond({cmp_token}, {th_mod}) -> act(add, 3) -> goal(val(x))"

        return CrossDomainExample(
            example_id=ex_id,
            domain=domain,
            concept=ComputationalConcept.CONDITION,
            raw_text=text,
            token_ids=CrossDomainTokenizer.tokenize(text),
            expected_output=out,
            intermediate_trace=trace,
            metadata={
                "v0": v0_mod,
                "cmp_op": cmp_op,
                "thresh": th_mod,
                "delta_true": 3,
                "delta_false": 1,
                "cond": cond,
                "out": out,
            },
        )

    def _make_loop_example(
        self, domain: Domain, ex_id: str, v0: int, reps: int, step: int
    ) -> CrossDomainExample:
        curr = v0 % MODULUS
        trace = [curr]
        for _ in range(reps):
            curr = (curr + step) % MODULUS
            trace.append(curr)

        items_str = ", ".join(str(step) for _ in range(reps))
        if domain == Domain.PROGRAM:
            text = f"a = {v0 % MODULUS}\nfor x in [{items_str}]:\n    a = a + x\nprint(a)"
        elif domain == Domain.NATURAL_LANGUAGE:
            text = f"initialize total to {v0 % MODULUS} . repeat {reps} times add {step} to total . what is total"
        else:  # PLANNING
            text = f"init(total, {v0 % MODULUS}) -> loop({reps}) -> act(add, {step}) -> goal(val(total))"

        return CrossDomainExample(
            example_id=ex_id,
            domain=domain,
            concept=ComputationalConcept.LOOP,
            raw_text=text,
            token_ids=CrossDomainTokenizer.tokenize(text),
            expected_output=curr,
            intermediate_trace=trace,
            metadata={"v0": v0 % MODULUS, "reps": reps, "step": step, "out": curr},
        )

    def _make_nested_example(
        self, domain: Domain, ex_id: str, v0: int, reps: int, thresh: int
    ) -> CrossDomainExample:
        curr = v0 % MODULUS
        trace = [curr]
        # Repetition sequence: items are [2, 5, 1, 6, 3][:reps]
        items = [2, 5, 1, 6, 3][:reps]
        for it in items:
            if it > thresh:
                curr = (curr + it) % MODULUS
            trace.append(curr)

        items_str = ", ".join(str(x) for x in items)
        if domain == Domain.PROGRAM:
            text = f"total = {v0 % MODULUS}\nfor x in [{items_str}]:\n    if x > {thresh}:\n        total = total + x\nprint(total)"
        elif domain == Domain.NATURAL_LANGUAGE:
            text = f"set total to {v0 % MODULUS} . repeat {reps} times if item exceeds {thresh} then add item to total . what is total"
        else:  # PLANNING
            text = f"init(acc, {v0 % MODULUS}) -> loop({reps}) -> cond(gt, {thresh}) -> act(add, item) -> goal(val(acc))"

        return CrossDomainExample(
            example_id=ex_id,
            domain=domain,
            concept=ComputationalConcept.NESTED_FILTER,
            raw_text=text,
            token_ids=CrossDomainTokenizer.tokenize(text),
            expected_output=curr,
            intermediate_trace=trace,
            metadata={"v0": v0 % MODULUS, "items": items, "thresh": thresh, "out": curr},
        )

    def _make_unsupported_example(
        self, domain: Domain, ex_id: str, rng: random.Random
    ) -> CrossDomainExample:
        if domain == Domain.PROGRAM:
            text = "a = undefined_macro(42)\nprint(unknown_var)"
        elif domain == Domain.NATURAL_LANGUAGE:
            text = "multiply quantum state by imaginary tensor . return infinity"
        else:  # PLANNING
            text = "init(contradiction) -> act(teleport, galaxy) -> goal(impossible)"

        return CrossDomainExample(
            example_id=ex_id,
            domain=domain,
            concept=ComputationalConcept.UNSUPPORTED,
            raw_text=text,
            token_ids=CrossDomainTokenizer.tokenize(text),
            expected_output=16,  # ABSTAIN symbol
            intermediate_trace=[16],
            is_supported=False,
            requires_abstention=True,
        )
