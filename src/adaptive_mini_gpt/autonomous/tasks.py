"""Procedural task definitions, partitioned generators, and verifiers (M8-M12.1)."""

from __future__ import annotations

import hashlib
import random
import re
import string
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from adaptive_mini_gpt.model.transformer import MiniGPT


@dataclass
class TaskSpecification:
    """Specification of a machine-verifiable procedural or instruction task."""

    task_id: str
    name: str
    description: str
    domain_trigger: str  # Token that activates specialist domain (e.g. '=')
    generator_fn: Callable[..., tuple[str, str]]
    verifier_fn: Callable[[str, str], bool]
    split_support: bool = False


# =====================================================================
# Strict Independent Ground-Truth Verifiers (M12.1)
# =====================================================================


def verify_addition(prompt: str, completion: str) -> bool:
    """Independently verify whether completion correctly solves prompt 'a+b=' with exact match."""
    match = re.search(r"(\d+)\s*\+\s*(\d+)\s*=", prompt)
    if not match:
        return False
    a, b = int(match.group(1)), int(match.group(2))
    expected = str(a + b)
    comp_clean = (
        completion.strip().replace("<|eos|>", "").split("\n")[0].split(" ")[0].strip().rstrip(".")
    )
    return comp_clean == expected


def verify_subtraction(prompt: str, completion: str) -> bool:
    """Independently verify whether completion correctly solves prompt 'a-b=' with exact match."""
    match = re.search(r"(\d+)\s*-\s*(\d+)\s*=", prompt)
    if not match:
        return False
    a, b = int(match.group(1)), int(match.group(2))
    expected = str(a - b)
    comp_clean = (
        completion.strip().replace("<|eos|>", "").split("\n")[0].split(" ")[0].strip().rstrip(".")
    )
    return comp_clean == expected


def verify_multiplication(prompt: str, completion: str) -> bool:
    """Independently verify whether completion correctly solves prompt 'a*b=' with exact match."""
    match = re.search(r"(\d+)\s*\*\s*(\d+)\s*=", prompt)
    if not match:
        return False
    a, b = int(match.group(1)), int(match.group(2))
    expected = str(a * b)
    comp_clean = (
        completion.strip().replace("<|eos|>", "").split("\n")[0].split(" ")[0].strip().rstrip(".")
    )
    return comp_clean == expected


def verify_parity(prompt: str, completion: str) -> bool:
    """Independently verify whether completion correctly predicts parity with exact match."""
    match = re.search(r"parity\s+(\d+)\s*=", prompt)
    if not match:
        match = re.search(r"(\d+)\s+is\s*=", prompt)
        if not match:
            return False
    n = int(match.group(1))
    expected = "even" if n % 2 == 0 else "odd"
    comp_clean = (
        completion.strip()
        .lower()
        .replace("<|eos|>", "")
        .split("\n")[0]
        .split(" ")[0]
        .strip()
        .rstrip(".")
    )
    return comp_clean == expected


def verify_reverse(prompt: str, completion: str) -> bool:
    """Independently verify whether completion reverses a short token string with exact match."""
    match = re.search(r"rev\(([a-zA-Z0-9]+)\)\s*=", prompt)
    if not match:
        return False
    s = match.group(1)
    expected = s[::-1]
    comp_clean = (
        completion.strip().replace("<|eos|>", "").split("\n")[0].split(" ")[0].strip().rstrip(".")
    )
    return comp_clean == expected


def verify_instruction_reasoning(prompt: str, completion: str) -> bool:
    """Verify that assistant response follows formatting and produces expected answer.

    Strictly rejects prompts without recognizable arithmetic.
    """
    comp_clean = completion.strip()
    if not comp_clean:
        return False
    match = re.search(r"(\d+)\s*([\+\-])\s*(\d+)", prompt)
    if not match:
        # P2: Strictly reject prompts without recognizable arithmetic
        return False
    a = int(match.group(1))
    op = match.group(2)
    b = int(match.group(3))
    expected = (a + b) if op == "+" else (a - b)

    ans_match = re.search(r"(?:result|answer\s+is|=)\s*[:]?\s*([0-9]+)", comp_clean, re.IGNORECASE)
    if ans_match:
        return int(ans_match.group(1)) == expected

    nums = [int(s) for s in re.findall(r"\b\d+\b", comp_clean)]
    return len(nums) > 0 and (nums[-1] == expected or expected in nums)


def verify_instruction_symbolic(prompt: str, completion: str) -> bool:
    """Strictly verify that assistant response correctly reverses the target word.

    Validates the entire response using anchored patterns, permits only the intended
    terminators (<|eos|>, period, space), and rejects contradictory text.
    """
    comp_clean = completion.strip()
    if not comp_clean:
        return False

    match = re.search(r"word\s*['\"]([a-zA-Z0-9]+)['\"]", prompt, re.IGNORECASE)
    if not match:
        match = re.search(r"word\s+([a-zA-Z0-9]+)", prompt, re.IGNORECASE)
        if not match:
            return False

    target_word = match.group(1).lower()
    expected = target_word[::-1]

    cleaned = comp_clean
    if cleaned.endswith("<|eos|>"):
        cleaned = cleaned[: -len("<|eos|>")].strip()

    # Anchored patterns covering the full response string:
    # Pattern 1: Structured answer format: 'The reversed sequence is: {answer}'
    # Pattern 2: Result format: 'Result: {answer}'
    # Pattern 3: Direct answer: '{answer}'
    patterns = [
        r"^(?:the\s+)?reversed\s+(?:sequence|string|word)?\s*(?:is)?\s*:\s*([a-zA-Z0-9]+)[.]?$",
        r"^(?:the\s+)?reverse\s*(?:of\s+['\"][a-zA-Z0-9]+['\"]\s*)?(?:is)?\s*:\s*([a-zA-Z0-9]+)[.]?$",
        r"^result\s*:\s*([a-zA-Z0-9]+)[.]?$",
        r"^([a-zA-Z0-9]+)[.]?$",
    ]

    for pat in patterns:
        m = re.fullmatch(pat, cleaned, re.IGNORECASE)
        if m:
            extracted = m.group(1).lower()
            return extracted == expected

    return False


def verify_instruction_symbolic_char(prompt: str, completion: str) -> bool:
    """Strictly verify that assistant response correctly reverses a character-delimited sequence.

    Handles space-delimited character tokens (e.g. "c a t" -> "t a c").
    Validates entire response using anchored patterns and permits only intended terminators.
    """
    comp_clean = completion.strip()
    if not comp_clean:
        return False

    match = re.search(
        r"characters(?:\s*in\s*['\"]|\s*:\s*)([a-zA-Z0-9\s]+?)(?:['\"]|\n|<\|assistant)",
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return False

    chars = [c for c in match.group(1).split() if c]
    if not chars:
        return False
    expected_spaced = " ".join(chars[::-1]).lower()
    expected_plain = "".join(chars[::-1]).lower()

    cleaned = comp_clean
    if cleaned.endswith("<|eos|>"):
        cleaned = cleaned[:-len("<|eos|>")].strip()

    patterns = [
        r"^(?:the\s+)?reversed\s+(?:sequence|string|word|characters)?\s*(?:is)?\s*:\s*([a-zA-Z0-9\s]+?)[.]?$",
        r"^(?:the\s+)?reverse\s*(?:of\s+[^\n:]+?\s*)?(?:is)?\s*:\s*([a-zA-Z0-9\s]+?)[.]?$",
        r"^result\s*:\s*([a-zA-Z0-9\s]+?)[.]?$",
        r"^([a-zA-Z0-9\s]+?)[.]?$",
    ]

    for pat in patterns:
        m = re.fullmatch(pat, cleaned, re.IGNORECASE)
        if m:
            extracted = m.group(1).strip().lower()
            ext_tokens = [c for c in extracted.split() if c]
            if " ".join(ext_tokens) == expected_spaced or "".join(ext_tokens) == expected_plain:
                return True

    return False


def score_instruction_symbolic_stepwise(prompt: str, completion: str) -> dict[str, Any]:
    """Score explicit procedural stepwise sequence reversal with independent step verification.

    Checks:
    - Step 1: Characters read correctly ("Step 1: Read characters: c, a, t")
    - Step 2: Index-based reversal ("0: t\n1: a\n2: c")
    - Result: Final reversed output ("Result: tac." or "The reversed sequence is: tac.")
    """
    comp_clean = completion.strip()
    if not comp_clean:
        return {"step1_valid": False, "step2_valid": False, "result_valid": False, "all_valid": False}

    match = re.search(r"word\s*['\"]([a-zA-Z0-9]+)['\"]", prompt, re.IGNORECASE)
    if not match:
        match = re.search(r"word\s+([a-zA-Z0-9]+)", prompt, re.IGNORECASE)
        if not match:
            return {"step1_valid": False, "step2_valid": False, "result_valid": False, "all_valid": False}

    target_word = match.group(1).lower()
    expected = target_word[::-1]

    # Verify Step 1: Read characters
    m1 = re.search(
        r"step\s*1\s*:\s*read\s+characters\s*:\s*([a-zA-Z0-9,\s]+?)(?=\n|step\s*2|$)",
        comp_clean,
        re.IGNORECASE,
    )
    step1_valid = False
    if m1:
        step1_chars = [c.strip().lower() for c in m1.group(1).split(",") if c.strip()]
        step1_valid = (step1_chars == list(target_word))

    # Verify Step 2: Reverse index
    m2_block = re.search(
        r"step\s*2\s*:\s*reverse\s+index\s*:\s*(.*?)(?=result\s*:|the\s+reversed|<\|eos\|>|$)",
        comp_clean,
        re.IGNORECASE | re.DOTALL,
    )
    step2_valid = False
    if m2_block:
        index_map: dict[int, str] = {}
        for line in m2_block.group(1).splitlines():
            line_m = re.match(r"^\s*(\d+)\s*:\s*([a-zA-Z0-9])\s*$", line.strip())
            if line_m:
                index_map[int(line_m.group(1))] = line_m.group(2).lower()
        if len(index_map) == len(expected) and all(
            index_map.get(i) == expected[i] for i in range(len(expected))
        ):
            step2_valid = True

    # Verify Result
    cleaned_eos = comp_clean
    if cleaned_eos.endswith("<|eos|>"):
        cleaned_eos = cleaned_eos[:-len("<|eos|>")].strip()

    m_res = re.search(
        r"(?:result\s*:\s*|(?:the\s+)?reversed\s+(?:sequence|string|word)?\s*(?:is)?\s*:\s*)([a-zA-Z0-9]+)[.]?$",
        cleaned_eos,
        re.IGNORECASE,
    )
    result_valid = False
    if m_res:
        result_valid = (m_res.group(1).lower() == expected)

    all_valid = step1_valid and step2_valid and result_valid
    return {
        "step1_valid": step1_valid,
        "step2_valid": step2_valid,
        "result_valid": result_valid,
        "all_valid": all_valid,
    }


def verify_instruction_symbolic_stepwise(prompt: str, completion: str) -> bool:
    """Strictly verify stepwise procedural sequence reversal."""
    return score_instruction_symbolic_stepwise(prompt, completion)["all_valid"]


def verify_instruction_symbolic_curriculum(prompt: str, completion: str) -> bool:
    """Verify progressive curriculum tasks: copying, 2-character swap, short and full reversal."""
    comp_clean = completion.strip()
    if not comp_clean:
        return False
    if comp_clean.endswith("<|eos|>"):
        comp_clean = comp_clean[:-len("<|eos|>")].strip()

    p_clean = prompt.strip().lower()

    # Case 1: Copying
    if "copy" in p_clean:
        m = re.search(
            r"characters(?:\s*in\s*['\"]|\s*:\s*['\"]?)([a-zA-Z0-9]+)",
            prompt,
            re.IGNORECASE,
        )
        if not m:
            return False
        expected = m.group(1).lower()
        pats = [
            r"^(?:the\s+)?copied\s+(?:sequence|string|word)?\s*(?:is)?\s*:\s*([a-zA-Z0-9]+)[.]?$",
            r"^result\s*:\s*([a-zA-Z0-9]+)[.]?$",
            r"^([a-zA-Z0-9]+)[.]?$",
        ]
        for pat in pats:
            rm = re.fullmatch(pat, comp_clean, re.IGNORECASE)
            if rm and rm.group(1).lower() == expected:
                return True
        return False

    # Case 2: 2-character swap
    elif "swap" in p_clean:
        m = re.search(
            r"characters(?:\s*in\s*['\"]|\s*:\s*['\"]?)([a-zA-Z0-9]\s*[a-zA-Z0-9])",
            prompt,
            re.IGNORECASE,
        )
        if not m:
            return False
        target = "".join(m.group(1).split())
        expected_plain = target[::-1].lower()
        expected_spaced = " ".join(list(target[::-1])).lower()
        pats = [
            r"^(?:the\s+)?swapped\s+(?:sequence|string|word|characters)?\s*(?:is)?\s*:\s*([a-zA-Z0-9\s]+?)[.]?$",
            r"^result\s*:\s*([a-zA-Z0-9\s]+?)[.]?$",
            r"^([a-zA-Z0-9\s]+?)[.]?$",
        ]
        for pat in pats:
            rm = re.fullmatch(pat, comp_clean, re.IGNORECASE)
            if rm:
                ans_tokens = [c for c in rm.group(1).split() if c]
                if "".join(ans_tokens).lower() == expected_plain or " ".join(ans_tokens).lower() == expected_spaced:
                    return True
        return False

    # Case 3: Reversal (default)
    else:
        return verify_instruction_symbolic(prompt, completion)


# Canonical Factual QA Knowledge Base (Train, Paraphrase, Unseen)
# (regex_pattern, [acceptable_answers], category)
QA_KNOWLEDGE_BASE: list[tuple[str, list[str], str]] = [
    # --- Train Facts (20 core facts used during training) ---
    (r"capital of france", ["paris"], "train"),
    (r"capital of japan", ["tokyo"], "train"),
    (r"closest (?:planet )?to the sun|planet is closest to the sun", ["mercury"], "train"),
    (r"speed of light in vacuum", ["300,000", "300000", "300,000 km/s"], "train"),
    (r"boiling point of water in celsius", ["100", "100 degrees", "100°c"], "train"),
    (
        r"wrote the play romeo and juliet|author of romeo and juliet",
        ["shakespeare", "william shakespeare"],
        "train",
    ),
    (r"green plants absorb during photosynthesis", ["carbon dioxide", "co2"], "train"),
    (r"largest ocean on earth", ["pacific", "pacific ocean"], "train"),
    (r"chemical symbol for gold", ["au"], "train"),
    (r"freezing point of water in celsius", ["0", "0 degrees", "0°c"], "train"),
    (r"chemical symbol for iron", ["fe"], "train"),
    (
        r"primary gas in earth's atmosphere|most abundant gas in the atmosphere",
        ["nitrogen", "n2"],
        "train",
    ),
    (r"planet with prominent rings|planet is famous for its rings", ["saturn"], "train"),
    (r"capital of italy", ["rome"], "train"),
    (
        r"tallest mountain above sea level|highest mountain on earth",
        ["mount everest", "everest"],
        "train",
    ),
    (r"primary currency of the united kingdom", ["pound", "pound sterling", "gbp"], "train"),
    (
        r"human organ responsible for thinking|central organ of the nervous system",
        ["brain"],
        "train",
    ),
    (r"number of continents on earth", ["7", "seven"], "train"),
    (
        r"formula for force in classical mechanics|force equals",
        ["f=ma", "mass times acceleration", "f = ma"],
        "train",
    ),
    (
        r"father of modern physics|developed theory of relativity",
        ["albert einstein", "einstein"],
        "train",
    ),
    # --- Paraphrase Facts (20 semantic variations of train facts for validation) ---
    (
        r"city serves as france's capital|which city is the capital of france",
        ["paris"],
        "paraphrase",
    ),
    (r"japan's capital city|what is the capital city of japan", ["tokyo"], "paraphrase"),
    (
        r"celestial body is nearest to the sun|which planet orbits closest to the sun",
        ["mercury"],
        "paraphrase",
    ),
    (
        r"how fast does light travel in a vacuum",
        ["300,000", "300000", "300,000 km/s"],
        "paraphrase",
    ),
    (r"temperature in celsius does water boil", ["100", "100 degrees", "100°c"], "paraphrase"),
    (
        r"playwright behind romeo and juliet|who created romeo and juliet",
        ["shakespeare", "william shakespeare"],
        "paraphrase",
    ),
    (
        r"gas is taken in by green plants for photosynthesis",
        ["carbon dioxide", "co2"],
        "paraphrase",
    ),
    (
        r"world's oceans has the largest surface area|largest body of water on earth",
        ["pacific", "pacific ocean"],
        "paraphrase",
    ),
    (r"elemental symbol representing gold", ["au"], "paraphrase"),
    (r"temperature does pure water freeze in celsius", ["0", "0 degrees", "0°c"], "paraphrase"),
    (r"atomic symbol assigned to iron", ["fe"], "paraphrase"),
    (r"gas makes up about 78 percent of air", ["nitrogen", "n2"], "paraphrase"),
    (r"ringed planet in our solar system", ["saturn"], "paraphrase"),
    (r"city serves as italy's capital", ["rome"], "paraphrase"),
    (r"peak rises highest above sea level", ["mount everest", "everest"], "paraphrase"),
    (r"official currency used in britain", ["pound", "pound sterling", "gbp"], "paraphrase"),
    (r"organ processes neural signals and thought", ["brain"], "paraphrase"),
    (r"how many continents are there", ["7", "seven"], "paraphrase"),
    (
        r"newton's second law expresses force as",
        ["f=ma", "mass times acceleration", "f = ma"],
        "paraphrase",
    ),
    (
        r"physicist who introduced the theory of relativity",
        ["albert einstein", "einstein"],
        "paraphrase",
    ),
    # --- Unseen Facts (20 novel facts strictly reserved for holdout test) ---
    (r"capital of germany", ["berlin"], "unseen"),
    (r"capital of canada", ["ottawa"], "unseen"),
    (r"known as the red planet", ["mars"], "unseen"),
    (r"hardest natural mineral on earth", ["diamond"], "unseen"),
    (r"painted the mona lisa", ["leonardo da vinci", "da vinci"], "unseen"),
    (r"chemical formula for water", ["h2o"], "unseen"),
    (r"organ pumps blood through the human body", ["heart"], "unseen"),
    (r"smallest prime number", ["2"], "unseen"),
    (
        r"invented the python programming language|created python",
        ["guido van rossum", "guido"],
        "unseen",
    ),
    (r"earth's only natural satellite", ["moon", "the moon"], "unseen"),
    (r"capital of australia", ["canberra"], "unseen"),
    (r"chemical symbol for silver", ["ag"], "unseen"),
    (r"largest mammal in the world", ["blue whale", "whale"], "unseen"),
    (r"discovered penicillin", ["alexander fleming", "fleming"], "unseen"),
    (r"si unit of electric current", ["ampere", "amp", "a"], "unseen"),
    (r"capital of egypt", ["cairo"], "unseen"),
    (r"atomic number of carbon", ["6", "six"], "unseen"),
    (r"chemical formula for table salt", ["nacl", "sodium chloride"], "unseen"),
    (r"longest river in the world", ["nile", "amazon"], "unseen"),
    (r"author of the epic poem the odyssey", ["homer"], "unseen"),
]


def score_instruction_qa(prompt: str, completion: str) -> tuple[bool, str]:
    """Score factual QA returning (is_correct, status_code).

    Status codes:
    - 'correct': Question recognized and expected answer correctly produced.
    - 'incorrect': Question recognized but completion contains incorrect answer.
    - 'unscorable': Question not recognized in knowledge base (strict failure).
    """
    comp_clean = completion.strip().lower()
    if not comp_clean:
        return False, "incorrect"

    p_clean = prompt.strip().lower()
    for pattern, acceptable_answers, _ in QA_KNOWLEDGE_BASE:
        if re.search(pattern, p_clean, re.IGNORECASE):
            for ans in acceptable_answers:
                if re.search(rf"\b{re.escape(ans.lower())}\b", comp_clean):
                    return True, "correct"
            return False, "incorrect"

    # Strictly fail unknown questions (fixes permissive len > 0 bug from M11)
    return False, "unscorable"


def verify_instruction_qa(prompt: str, completion: str) -> bool:
    """Strictly verify factual QA completion.

    Unknown questions return False; never defaults to True on non-empty completions.
    """
    is_correct, _ = score_instruction_qa(prompt, completion)
    return is_correct


# =====================================================================
# Deterministic Partitioning & Data Generators (M12.1)
# =====================================================================


def get_string_split(s: str) -> str:
    """Deterministically partition any string into train (70%), val (15%), or test (15%)."""
    h = int(hashlib.sha256(f"m12_split_salt:{s}".encode()).hexdigest()[:8], 16)
    bucket = h % 100
    if bucket < 70:
        return "train"
    elif bucket < 85:
        return "val"
    else:
        return "test"


def generate_instruction_symbolic_data(
    num_samples: int = 100,
    seed: int = 42,
    split: str = "train",
) -> tuple[str, str]:
    """Generate partitioned string reversal instruction pairs.

    - 'train': Lengths 3-6 characters strictly partitioned to train split.
    - 'val': Lengths 3-6 characters strictly partitioned to val split.
    - 'test': Unseen strings with lengths 3-6 (test split) AND novel lengths 7-8
      guaranteed absent from train and val.
    """
    rng = random.Random(seed)
    chars = string.ascii_lowercase + string.digits
    pairs: list[str] = []
    generated_strings: set[str] = set()

    max_attempts = num_samples * 200
    attempts = 0

    while len(pairs) < num_samples and attempts < max_attempts:
        attempts += 1
        if split == "test":
            # Length distribution: 50% unseen lengths 7-8, 50% standard lengths 3-6
            if rng.random() < 0.5:
                length = rng.randint(7, 8)  # Distinct length distribution for OOD generalization!
            else:
                length = rng.randint(3, 6)
        else:
            length = rng.randint(3, 6)

        w = "".join(rng.choice(chars) for _ in range(length))
        if w in generated_strings:
            continue

        # Enforce strict hash partitioning
        if length <= 6:
            w_split = get_string_split(w)
            if w_split != split:
                continue
        else:
            # Lengths 7-8 are strictly reserved for test split
            if split != "test":
                continue

        generated_strings.add(w)
        rev = w[::-1]
        pairs.append(
            f"<|user|>\nReverse the sequence of characters in the word '{w}'.\n"
            f"<|assistant|>\nThe reversed sequence is: {rev}.<|eos|>\n"
        )

    text = "".join(pairs)
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


def generate_instruction_symbolic_char_data(
    num_samples: int = 100,
    seed: int = 42,
    split: str = "train",
) -> tuple[str, str]:
    """Generate partitioned character-delimited string reversal instruction pairs (Arm B).

    Input: "c a t" -> Output: "t a c"
    """
    rng = random.Random(seed)
    chars = string.ascii_lowercase + string.digits
    pairs: list[str] = []
    generated_strings: set[str] = set()

    max_attempts = num_samples * 200
    attempts = 0

    while len(pairs) < num_samples and attempts < max_attempts:
        attempts += 1
        if split == "test":
            if rng.random() < 0.5:
                length = rng.randint(7, 8)
            else:
                length = rng.randint(3, 6)
        else:
            length = rng.randint(3, 6)

        w = "".join(rng.choice(chars) for _ in range(length))
        if w in generated_strings:
            continue

        if length <= 6:
            if get_string_split(w) != split:
                continue
        else:
            if split != "test":
                continue

        generated_strings.add(w)
        spaced_w = " ".join(list(w))
        spaced_rev = " ".join(list(w[::-1]))
        pairs.append(
            f"<|user|>\nReverse the sequence of characters: {spaced_w}\n"
            f"<|assistant|>\nThe reversed sequence is: {spaced_rev}.<|eos|>\n"
        )

    text = "".join(pairs)
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


def generate_instruction_symbolic_stepwise_data(
    num_samples: int = 100,
    seed: int = 42,
    split: str = "train",
) -> tuple[str, str]:
    """Generate partitioned stepwise procedural string reversal instruction pairs (Arm C).

    Step 1: Read characters
    Step 2: Reverse index
    Result: Reversed word
    """
    rng = random.Random(seed)
    chars = string.ascii_lowercase + string.digits
    pairs: list[str] = []
    generated_strings: set[str] = set()

    max_attempts = num_samples * 200
    attempts = 0

    while len(pairs) < num_samples and attempts < max_attempts:
        attempts += 1
        if split == "test":
            if rng.random() < 0.5:
                length = rng.randint(7, 8)
            else:
                length = rng.randint(3, 6)
        else:
            length = rng.randint(3, 6)

        w = "".join(rng.choice(chars) for _ in range(length))
        if w in generated_strings:
            continue

        if length <= 6:
            if get_string_split(w) != split:
                continue
        else:
            if split != "test":
                continue

        generated_strings.add(w)
        rev = w[::-1]
        read_step = ", ".join(list(w))
        index_lines = "\n".join(f"{i}: {rev[i]}" for i in range(len(rev)))
        pairs.append(
            f"<|user|>\nReverse the sequence of characters step-by-step for the word '{w}'.\n"
            f"<|assistant|>\n"
            f"Step 1: Read characters: {read_step}\n"
            f"Step 2: Reverse index:\n{index_lines}\n"
            f"Result: {rev}.<|eos|>\n"
        )

    text = "".join(pairs)
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


def generate_instruction_symbolic_curriculum_data(
    num_samples: int = 100,
    seed: int = 42,
    split: str = "train",
) -> tuple[str, str]:
    """Generate progressive curriculum instruction pairs (Arm D).

    Train stages:
    1. Copying sequence (length 2-5)
    2. 2-character swap (length 2)
    3. Short reversal (length 3-4)
    4. Full reversal (length 3-6)

    Val and test splits:
    Target reversal task (lengths 3-6 for val; 3-6 + 7-8 for test) for direct comparability.
    """
    rng = random.Random(seed)
    chars = string.ascii_lowercase + string.digits
    pairs: list[str] = []
    generated_strings: set[str] = set()

    max_attempts = num_samples * 200
    attempts = 0

    if split in ("val", "test"):
        return generate_instruction_symbolic_data(num_samples=num_samples, seed=seed, split=split)

    per_stage = max(1, num_samples // 4)

    # Stage 1: Copying
    stage1_pairs: list[str] = []
    while len(stage1_pairs) < per_stage and attempts < max_attempts:
        attempts += 1
        length = rng.randint(2, 5)
        w = "".join(rng.choice(chars) for _ in range(length))
        if w in generated_strings or get_string_split(w) != "train":
            continue
        generated_strings.add(w)
        stage1_pairs.append(
            f"<|user|>\nCopy the sequence of characters: {w}\n"
            f"<|assistant|>\nThe copied sequence is: {w}.<|eos|>\n"
        )

    # Stage 2: 2-character swap
    stage2_pairs: list[str] = []
    while len(stage2_pairs) < per_stage and attempts < max_attempts:
        attempts += 1
        w = "".join(rng.choice(chars) for _ in range(2))
        if w in generated_strings or get_string_split(w) != "train":
            continue
        generated_strings.add(w)
        stage2_pairs.append(
            f"<|user|>\nSwap the two characters: {w}\n"
            f"<|assistant|>\nThe swapped sequence is: {w[::-1]}.<|eos|>\n"
        )

    # Stage 3: Short reversal (len 3-4)
    stage3_pairs: list[str] = []
    while len(stage3_pairs) < per_stage and attempts < max_attempts:
        attempts += 1
        length = rng.randint(3, 4)
        w = "".join(rng.choice(chars) for _ in range(length))
        if w in generated_strings or get_string_split(w) != "train":
            continue
        generated_strings.add(w)
        stage3_pairs.append(
            f"<|user|>\nReverse the sequence of characters in the word '{w}'.\n"
            f"<|assistant|>\nThe reversed sequence is: {w[::-1]}.<|eos|>\n"
        )

    # Stage 4: Full reversal (len 3-6)
    stage4_pairs: list[str] = []
    target_stage4 = num_samples - len(stage1_pairs) - len(stage2_pairs) - len(stage3_pairs)
    while len(stage4_pairs) < target_stage4 and attempts < max_attempts:
        attempts += 1
        length = rng.randint(3, 6)
        w = "".join(rng.choice(chars) for _ in range(length))
        if w in generated_strings or get_string_split(w) != "train":
            continue
        generated_strings.add(w)
        stage4_pairs.append(
            f"<|user|>\nReverse the sequence of characters in the word '{w}'.\n"
            f"<|assistant|>\nThe reversed sequence is: {w[::-1]}.<|eos|>\n"
        )

    pairs = stage1_pairs + stage2_pairs + stage3_pairs + stage4_pairs
    rng.shuffle(pairs)
    text = "".join(pairs)
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


# QA Bank organized by split
TRAIN_QA_BANK: list[tuple[str, str]] = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("Which planet is closest to the Sun?", "Mercury is the closest planet to the Sun."),
    (
        "What is the speed of light in vacuum approximately?",
        "The speed of light in vacuum is approximately 300,000 km/s.",
    ),
    (
        "What is the boiling point of water in Celsius?",
        "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
    ),
    (
        "Who wrote the play Romeo and Juliet?",
        "Romeo and Juliet was written by William Shakespeare.",
    ),
    (
        "What gas do green plants absorb during photosynthesis?",
        "Green plants absorb carbon dioxide during photosynthesis.",
    ),
    (
        "What is the largest ocean on Earth?",
        "The Pacific Ocean is the largest and deepest ocean on Earth.",
    ),
    (
        "What is the chemical symbol for gold?",
        "The chemical symbol for gold is Au.",
    ),
    (
        "What is the freezing point of water in Celsius?",
        "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
    ),
    ("What is the chemical symbol for iron?", "The chemical symbol for iron is Fe."),
    (
        "What is the most abundant gas in the atmosphere?",
        "Nitrogen is the most abundant gas in Earth's atmosphere.",
    ),
    (
        "Which planet is famous for its rings?",
        "Saturn is famous for its prominent planetary rings.",
    ),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    (
        "What is the highest mountain on Earth?",
        "Mount Everest is the highest mountain above sea level.",
    ),
    (
        "What is the primary currency of the United Kingdom?",
        "The official currency of the United Kingdom is the Pound Sterling.",
    ),
    (
        "What is the central organ of the nervous system?",
        "The brain is the central organ responsible for thinking.",
    ),
    ("How many continents are there on Earth?", "There are 7 continents on Earth."),
    (
        "What is the formula for force in classical mechanics?",
        "In classical mechanics, force equals mass times acceleration (F=ma).",
    ),
    (
        "Who developed the theory of relativity?",
        "Albert Einstein developed the theory of relativity.",
    ),
]

PARAPHRASE_QA_BANK: list[tuple[str, str]] = [
    ("Which city serves as France's capital?", "Paris serves as France's capital."),
    ("What is Japan's capital city?", "Japan's capital city is Tokyo."),
    ("Which celestial body is nearest to the Sun?", "Mercury is nearest to the Sun."),
    (
        "How fast does light travel in a vacuum?",
        "Light travels at approximately 300,000 km/s in a vacuum.",
    ),
    (
        "At what temperature in Celsius does water boil?",
        "Water boils at 100 degrees Celsius.",
    ),
    (
        "Who is the playwright behind Romeo and Juliet?",
        "William Shakespeare is the playwright behind Romeo and Juliet.",
    ),
    (
        "What gas is taken in by green plants for photosynthesis?",
        "Green plants take in carbon dioxide during photosynthesis.",
    ),
    (
        "Which of the world's oceans has the largest surface area?",
        "The Pacific Ocean has the largest surface area.",
    ),
    (
        "What is the elemental symbol representing gold?",
        "The elemental symbol representing gold is Au.",
    ),
    (
        "At what temperature does pure water freeze in Celsius?",
        "Pure water freezes at 0 degrees Celsius.",
    ),
    ("What atomic symbol is assigned to iron?", "The atomic symbol for iron is Fe."),
    (
        "Which gas makes up about 78 percent of air?",
        "Nitrogen makes up approximately 78% of dry atmospheric air.",
    ),
    (
        "Which ringed planet is in our solar system?",
        "Saturn is the iconic ringed gas giant in our solar system.",
    ),
    ("Which city serves as Italy's capital?", "Rome serves as Italy's capital city."),
    ("Which peak rises highest above sea level?", "Mount Everest rises highest above sea level."),
    (
        "What official currency is used in Britain?",
        "The Pound Sterling is the official currency used in Britain.",
    ),
    (
        "Which organ processes neural signals and thought?",
        "The brain processes neural signals and thought.",
    ),
    ("How many continents are there?", "There are seven continents."),
    (
        "Newton's second law expresses force as what formula?",
        "Newton's second law expresses force as F = ma.",
    ),
    (
        "Which physicist introduced the theory of relativity?",
        "Albert Einstein introduced the theory of relativity.",
    ),
]

UNSEEN_QA_BANK: list[tuple[str, str]] = [
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
    ("Which planet is known as the Red Planet?", "Mars is known as the Red Planet."),
    (
        "What is the hardest natural mineral on Earth?",
        "Diamond is the hardest natural mineral on Earth.",
    ),
    (
        "Who painted the Mona Lisa?",
        "Leonardo da Vinci painted the Mona Lisa.",
    ),
    (
        "What is the chemical formula for water?",
        "The chemical formula for water is H2O.",
    ),
    (
        "Which organ pumps blood through the human body?",
        "The heart pumps blood through the human body.",
    ),
    (
        "What is the smallest prime number?",
        "The smallest prime number is 2.",
    ),
    (
        "Who created Python?",
        "Guido van Rossum created the Python programming language.",
    ),
    ("What is Earth's only natural satellite?", "The Moon is Earth's only natural satellite."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
    ("What is the chemical symbol for silver?", "The chemical symbol for silver is Ag."),
    (
        "What is the largest mammal in the world?",
        "The blue whale is the largest mammal in the world.",
    ),
    ("Who discovered penicillin?", "Alexander Fleming discovered penicillin in 1928."),
    (
        "What is the SI unit of electric current?",
        "The ampere (A) is the SI unit of electric current.",
    ),
    ("What is the capital of Egypt?", "The capital of Egypt is Cairo."),
    ("What is the atomic number of carbon?", "The atomic number of carbon is 6."),
    (
        "What is the chemical formula for table salt?",
        "The chemical formula for table salt is NaCl.",
    ),
    (
        "What is the longest river in the world?",
        "The Nile is traditionally considered the longest river in the world.",
    ),
    (
        "Who is the author of the epic poem the Odyssey?",
        "Homer is the author of the ancient Greek epic the Odyssey.",
    ),
]


def generate_instruction_qa_data(
    num_samples: int = 20,
    seed: int = 42,
    split: str = "train",
) -> tuple[str, str]:
    """Generate partitioned factual Q&A instruction pairs.

    - 'train': Core training facts.
    - 'val' or 'paraphrase': Paraphrase generalization of training facts.
    - 'test' or 'unseen': Completely unseen factual knowledge.
    """
    rng = random.Random(seed)
    if split in ("val", "paraphrase"):
        bank = PARAPHRASE_QA_BANK
    elif split in ("test", "unseen"):
        bank = UNSEEN_QA_BANK
    else:
        bank = TRAIN_QA_BANK

    pairs: list[str] = []
    # If num_samples <= len(bank), sample unique items without replacement
    if num_samples <= len(bank):
        selected_items = rng.sample(bank, num_samples)
    else:
        selected_items = [rng.choice(bank) for _ in range(num_samples)]

    for q, a in selected_items:
        pairs.append(f"<|user|>\n{q}\n<|assistant|>\n{a}<|eos|>\n")

    text = "".join(pairs)
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


def generate_instruction_data(
    num_samples: int = 100, seed: int = 42, split: str = "train"
) -> tuple[str, str]:
    """Generate curated instruction pairs with scratchpad reasoning."""
    from adaptive_mini_gpt.data.instruction_data import generate_curated_instruction_corpus

    return generate_curated_instruction_corpus(num_synthetic_reasoning=num_samples, seed=seed)


def generate_addition_data(
    num_samples: int = 1000, seed: int = 42, split: str = "train"
) -> tuple[str, str]:
    """Generate procedurally verified two-digit addition examples."""
    rng = random.Random(seed)
    lines: list[str] = []
    for _ in range(num_samples):
        a = rng.randint(0, 99)
        b = rng.randint(0, 99)
        lines.append(f"{a}+{b}={a + b}")
    text = "\n".join(lines) + "\n"
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


def generate_subtraction_data(
    num_samples: int = 1000, seed: int = 42, split: str = "train"
) -> tuple[str, str]:
    """Generate procedurally verified two-digit subtraction examples."""
    rng = random.Random(seed)
    lines: list[str] = []
    for _ in range(num_samples):
        a = rng.randint(0, 99)
        b = rng.randint(0, a)  # Non-negative result
        lines.append(f"{a}-{b}={a - b}")
    text = "\n".join(lines) + "\n"
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


def generate_multiplication_data(
    num_samples: int = 1000, seed: int = 42, split: str = "train"
) -> tuple[str, str]:
    """Generate procedurally verified single-digit multiplication examples."""
    rng = random.Random(seed)
    lines: list[str] = []
    for _ in range(num_samples):
        a = rng.randint(0, 9)
        b = rng.randint(0, 9)
        lines.append(f"{a}*{b}={a * b}")
    text = "\n".join(lines) + "\n"
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


def generate_parity_data(
    num_samples: int = 1000, seed: int = 42, split: str = "train"
) -> tuple[str, str]:
    """Generate procedurally verified parity examples."""
    rng = random.Random(seed)
    lines: list[str] = []
    for _ in range(num_samples):
        n = rng.randint(0, 999)
        ans = "even" if n % 2 == 0 else "odd"
        lines.append(f"parity {n}={ans}")
    text = "\n".join(lines) + "\n"
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


def generate_reverse_data(
    num_samples: int = 1000, seed: int = 42, split: str = "train"
) -> tuple[str, str]:
    """Generate procedurally verified string reversal examples with split partitioning."""
    rng = random.Random(seed)
    chars = "abcdefghijklmnopqrstuvwxyz"
    lines: list[str] = []
    while len(lines) < num_samples:
        if split == "test":
            length = rng.randint(3, 8)
        else:
            length = rng.randint(3, 5)

        s = "".join(rng.choice(chars) for _ in range(length))
        if length <= 5:
            if get_string_split(s) != split:
                continue
        else:
            if split != "test":
                continue

        lines.append(f"rev({s})={s[::-1]}")

    text = "\n".join(lines) + "\n"
    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, fp


TASK_REGISTRY: dict[str, TaskSpecification] = {
    "addition": TaskSpecification(
        task_id="addition",
        name="Addition",
        description="Two-digit addition (a+b=c)",
        domain_trigger="=",
        generator_fn=generate_addition_data,
        verifier_fn=verify_addition,
        split_support=False,
    ),
    "subtraction": TaskSpecification(
        task_id="subtraction",
        name="Subtraction",
        description="Two-digit non-negative subtraction (a-b=c)",
        domain_trigger="=",
        generator_fn=generate_subtraction_data,
        verifier_fn=verify_subtraction,
        split_support=False,
    ),
    "multiplication": TaskSpecification(
        task_id="multiplication",
        name="Multiplication",
        description="Single-digit multiplication (a*b=c)",
        domain_trigger="=",
        generator_fn=generate_multiplication_data,
        verifier_fn=verify_multiplication,
        split_support=False,
    ),
    "parity": TaskSpecification(
        task_id="parity",
        name="Parity",
        description="Integer parity classification (parity n=even/odd)",
        domain_trigger="=",
        generator_fn=generate_parity_data,
        verifier_fn=verify_parity,
        split_support=False,
    ),
    "reverse": TaskSpecification(
        task_id="reverse",
        name="String Reversal",
        description="Short sequence inversion (rev(s)=reversed)",
        domain_trigger="=",
        generator_fn=generate_reverse_data,
        verifier_fn=verify_reverse,
        split_support=True,
    ),
    "instruction_reasoning": TaskSpecification(
        task_id="instruction_reasoning",
        name="Instruction Reasoning",
        description="Instruction-following and scratchpad arithmetic reasoning",
        domain_trigger="<|assistant|>",
        generator_fn=generate_instruction_data,
        verifier_fn=verify_instruction_reasoning,
        split_support=False,
    ),
    "instruction_symbolic": TaskSpecification(
        task_id="instruction_symbolic",
        name="Instruction Symbolic Reversal",
        description="Instruction-based word sequence reversal in chat format",
        domain_trigger="<|assistant|>",
        generator_fn=generate_instruction_symbolic_data,
        verifier_fn=verify_instruction_symbolic,
        split_support=True,
    ),
    "instruction_qa": TaskSpecification(
        task_id="instruction_qa",
        name="Instruction Factual QA",
        description="Instruction-based factual question answering in chat format",
        domain_trigger="<|assistant|>",
        generator_fn=generate_instruction_qa_data,
        verifier_fn=verify_instruction_qa,
        split_support=True,
    ),
    "instruction_symbolic_char": TaskSpecification(
        task_id="instruction_symbolic_char",
        name="Instruction Symbolic Char Reversal",
        description="Character-delimited word sequence reversal (p h o t o n -> n o t o h p)",
        domain_trigger="<|assistant|>",
        generator_fn=generate_instruction_symbolic_char_data,
        verifier_fn=verify_instruction_symbolic_char,
        split_support=True,
    ),
    "instruction_symbolic_stepwise": TaskSpecification(
        task_id="instruction_symbolic_stepwise",
        name="Instruction Symbolic Stepwise Reversal",
        description="Explicit procedural stepwise reversal with intermediate verified steps",
        domain_trigger="<|assistant|>",
        generator_fn=generate_instruction_symbolic_stepwise_data,
        verifier_fn=verify_instruction_symbolic_stepwise,
        split_support=True,
    ),
    "instruction_symbolic_curriculum": TaskSpecification(
        task_id="instruction_symbolic_curriculum",
        name="Instruction Symbolic Curriculum Reversal",
        description="Progressive procedural curriculum (copying -> 2-char swap -> short -> full reversal)",
        domain_trigger="<|assistant|>",
        generator_fn=generate_instruction_symbolic_curriculum_data,
        verifier_fn=verify_instruction_symbolic_curriculum,
        split_support=True,
    ),
}


def evaluate_task_holdout(
    model: MiniGPT,
    tokenizer: Any,
    task: TaskSpecification,
    device: torch.device,
    num_samples: int = 100,
    seed: int = 9999,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate model on a clean holdout split using the task's independent verifier.

    Reports total accuracy as well as unique-question accuracy to prevent
    inflated confidence denominators.
    """
    if task.split_support:
        text, _ = task.generator_fn(num_samples, seed, split=split)
    else:
        text, _ = task.generator_fn(num_samples, seed)

    was_training = model.training
    model.eval()

    correct = 0
    total = 0
    unscorable = 0

    seen_prompts: set[str] = set()
    unique_prompts: set[str] = set()
    unique_correct_prompts: set[str] = set()

    newline_id = (
        tokenizer.get_token_id("\n")
        if hasattr(tokenizer, "get_token_id")
        else (
            tokenizer.stoi.get(b"\n")
            if hasattr(tokenizer, "stoi") and b"\n" in tokenizer.stoi
            else getattr(tokenizer, "stoi", {}).get("\n", -1)
        )
    )
    eos_id = getattr(tokenizer, "eos_id", -1)
    stop_ids = {newline_id, eos_id} - {None, -1}

    with torch.no_grad():
        if task.task_id.startswith("instruction_"):
            chunks = text.split("<|user|>")
            for chunk in chunks:
                if not chunk.strip() or "<|assistant|>" not in chunk:
                    continue
                parts = chunk.split("<|assistant|>")
                prompt = f"<|user|>{parts[0]}<|assistant|>\n"
                prompt_key = prompt.strip()
                is_first_seen = prompt_key not in seen_prompts
                seen_prompts.add(prompt_key)
                if is_first_seen:
                    unique_prompts.add(prompt_key)

                prompt_tokens = tokenizer.encode(prompt)
                input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

                max_gen_tokens = 96 if "stepwise" in task.task_id else 48
                for _ in range(max_gen_tokens):
                    if input_tensor.size(1) >= model.config.context_length:
                        break
                    logits, _ = model(input_tensor)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    input_tensor = torch.cat([input_tensor, next_token], dim=1)
                    if next_token.item() == eos_id:
                        break

                generated_full = tokenizer.decode(input_tensor[0].tolist())
                completion = generated_full[len(prompt) :]
                total += 1

                is_corr = False
                if task.task_id == "instruction_qa":
                    is_corr, status = score_instruction_qa(prompt, completion)
                    if is_corr:
                        correct += 1
                    elif status == "unscorable":
                        unscorable += 1
                else:
                    if task.verifier_fn(prompt, completion):
                        correct += 1
                        is_corr = True

                if is_corr and is_first_seen:
                    unique_correct_prompts.add(prompt_key)
        else:
            lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
            for line in lines:
                if "=" not in line:
                    continue
                prompt = line.split("=")[0] + "="
                expected_ans = line.split("=")[1].strip()

                prompt_key = prompt.strip()
                is_first_seen = prompt_key not in seen_prompts
                seen_prompts.add(prompt_key)
                if is_first_seen:
                    unique_prompts.add(prompt_key)

                prompt_tokens = tokenizer.encode(prompt)
                input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

                gen_len = len(expected_ans) + 2
                for _ in range(gen_len):
                    if input_tensor.size(1) >= model.config.context_length:
                        break
                    logits, _ = model(input_tensor)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    input_tensor = torch.cat([input_tensor, next_token], dim=1)
                    if next_token.item() in stop_ids:
                        break

                generated_full = tokenizer.decode(input_tensor[0].tolist())
                completion = generated_full[len(prompt) :]
                total += 1
                is_corr = False
                if task.verifier_fn(prompt, completion):
                    correct += 1
                    is_corr = True

                if is_corr and is_first_seen:
                    unique_correct_prompts.add(prompt_key)

    if was_training:
        model.train()

    accuracy = correct / total if total > 0 else 0.0
    u_total = len(unique_prompts)
    u_correct = len(unique_correct_prompts)
    u_acc = u_correct / u_total if u_total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "correct": float(correct),
        "total": float(total),
        "unique_questions": u_total,
        "unique_correct": u_correct,
        "unique_accuracy": round(u_acc, 4),
        "unscorable": float(unscorable),
        "split": split,
    }


def evaluate_symbolic_reversal_breakdown(
    model: MiniGPT,
    tokenizer: Any,
    device: torch.device,
    samples_per_category: int = 50,
    seed: int = 9999,
) -> dict[str, Any]:
    """Evaluate string reversal on fine-grained generalization categories on held-out test data.

    Categories:
    - 'unseen_in_distribution_len_3_6': lengths 3-6 held-out test split
    - 'ood_extrapolation_len_7_8': lengths 7-8 strictly reserved for test split
    - 'alphanumeric_combinations': strings containing both letters and digits
    - 'alpha_only': strings containing lowercase letters only
    - 'digits_only': strings containing digits only
    """
    rng = random.Random(seed)
    eos_id = getattr(tokenizer, "eos_id", -1)

    categories = {
        "unseen_in_distribution_len_3_6": {
            "chars": string.ascii_lowercase + string.digits,
            "lens": (3, 6),
            "alphanumeric_mode": "any",
        },
        "ood_extrapolation_len_7_8": {
            "chars": string.ascii_lowercase + string.digits,
            "lens": (7, 8),
            "alphanumeric_mode": "any",
        },
        "alphanumeric_combinations": {
            "chars": string.ascii_lowercase + string.digits,
            "lens": (3, 6),
            "alphanumeric_mode": "mixed",
        },
        "alpha_only": {
            "chars": string.ascii_lowercase,
            "lens": (3, 6),
            "alphanumeric_mode": "alpha",
        },
        "digits_only": {"chars": string.digits, "lens": (3, 6), "alphanumeric_mode": "digits"},
    }

    was_training = model.training
    model.eval()

    breakdown_results: dict[str, Any] = {}

    for cat_name, cat_spec in categories.items():
        min_len, max_len = cat_spec["lens"]
        chars = cat_spec["chars"]
        mode = cat_spec["alphanumeric_mode"]

        generated_words: list[str] = []
        attempts = 0
        while len(generated_words) < samples_per_category and attempts < samples_per_category * 200:
            attempts += 1
            length = rng.randint(min_len, max_len)
            w = "".join(rng.choice(chars) for _ in range(length))
            if w in generated_words:
                continue

            if length <= 6:
                if get_string_split(w) != "test":
                    continue
            else:
                # Lengths 7-8 are inherently test
                pass

            if mode == "mixed":
                has_alpha = any(c.isalpha() for c in w)
                has_digit = any(c.isdigit() for c in w)
                if not (has_alpha and has_digit):
                    continue
            elif mode == "alpha":
                if not all(c.isalpha() for c in w):
                    continue
            elif mode == "digits":
                if not all(c.isdigit() for c in w):
                    continue

            generated_words.append(w)

        correct = 0
        sample_records: list[dict[str, Any]] = []

        with torch.no_grad():
            for w in generated_words:
                prompt = (
                    f"<|user|>\nReverse the sequence of characters in the word '{w}'.\n"
                    f"<|assistant|>\n"
                )
                prompt_tokens = tokenizer.encode(prompt)
                input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

                for _ in range(48):
                    if input_tensor.size(1) >= model.config.context_length:
                        break
                    logits, _ = model(input_tensor)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    input_tensor = torch.cat([input_tensor, next_token], dim=1)
                    if next_token.item() == eos_id:
                        break

                generated_full = tokenizer.decode(input_tensor[0].tolist())
                completion = generated_full[len(prompt) :]
                is_corr = verify_instruction_symbolic(prompt, completion)
                if is_corr:
                    correct += 1

                if len(sample_records) < 3:
                    sample_records.append(
                        {
                            "word": w,
                            "expected": w[::-1],
                            "completion": completion.replace("<|eos|>", "").strip(),
                            "correct": is_corr,
                        }
                    )

        tot = len(generated_words)
        breakdown_results[cat_name] = {
            "accuracy": round(correct / tot, 4) if tot > 0 else 0.0,
            "correct": correct,
            "total": tot,
            "train_overlap_count": 0,
            "samples": sample_records,
        }

    if was_training:
        model.train()

    return breakdown_results


def extract_reversed_answer(completion: str, arm: str = "direct") -> str:
    """Extract predicted reversed answer string from completion based on experimental arm."""
    cleaned = completion.strip()
    if cleaned.endswith("<|eos|>"):
        cleaned = cleaned[:-len("<|eos|>")].strip()

    if arm == "char":
        pats = [
            r"^(?:the\s+)?reversed\s+(?:sequence|string|word|characters)?\s*(?:is)?\s*:\s*([a-zA-Z0-9\s]+?)[.]?$",
            r"^(?:the\s+)?reverse\s*(?:of\s+[^\n:]+?\s*)?(?:is)?\s*:\s*([a-zA-Z0-9\s]+?)[.]?$",
            r"^result\s*:\s*([a-zA-Z0-9\s]+?)[.]?$",
            r"^([a-zA-Z0-9\s]+?)[.]?$",
        ]
        for pat in pats:
            m = re.fullmatch(pat, cleaned, re.IGNORECASE)
            if m:
                return "".join(m.group(1).split()).lower()
        return "".join(cleaned.split()).lower()

    elif arm == "stepwise":
        m_res = re.search(
            r"(?:result\s*:\s*|(?:the\s+)?reversed\s+(?:sequence|string|word)?\s*(?:is)?\s*:\s*)([a-zA-Z0-9]+)[.]?$",
            cleaned,
            re.IGNORECASE,
        )
        if m_res:
            return m_res.group(1).lower()
        m_fall = re.findall(r"[a-zA-Z0-9]+", cleaned)
        return m_fall[-1].lower() if m_fall else ""

    else:
        # direct / curriculum
        pats = [
            r"^(?:the\s+)?reversed\s+(?:sequence|string|word)?\s*(?:is)?\s*:\s*([a-zA-Z0-9]+)[.]?$",
            r"^(?:the\s+)?reverse\s*(?:of\s+['\"][a-zA-Z0-9]+['\"]\s*)?(?:is)?\s*:\s*([a-zA-Z0-9]+)[.]?$",
            r"^result\s*:\s*([a-zA-Z0-9]+)[.]?$",
            r"^([a-zA-Z0-9]+)[.]?$",
        ]
        for pat in pats:
            m = re.fullmatch(pat, cleaned, re.IGNORECASE)
            if m:
                return m.group(1).lower()
        m_fall = re.findall(r"[a-zA-Z0-9]+", cleaned)
        return m_fall[-1].lower() if m_fall else ""


def evaluate_symbolic_diagnostics(
    model: MiniGPT,
    tokenizer: Any,
    device: torch.device,
    arm: str = "direct",
    num_samples_per_cat: int = 50,
    seed: int = 9999,
) -> dict[str, Any]:
    """Evaluate symbolic reversal across multi-dimensional diagnostic metrics (M13).

    Diagnostics:
    - exact_match_acc: 100% full answer exact match
    - per_char_acc: position-aligned character match ratio
    - length_match_acc: predicted answer length == expected length
    - termination_acc: clean termination with <|eos|> or period
    - output_diversity: unique completions / total completions (detects degenerate modes)
    - intermediate procedural scores: for stepwise, step1, step2, and result accuracy
    - length breakdown: unseen in-distribution (lens 3-6) vs OOD extrapolation (lens 7-8)
    """
    rng = random.Random(seed)
    eos_id = getattr(tokenizer, "eos_id", -1)
    chars = string.ascii_lowercase + string.digits

    categories = {
        "unseen_in_distribution_len_3_6": (3, 6),
        "ood_extrapolation_len_7_8": (7, 8),
    }

    was_training = model.training
    model.eval()

    diag_results: dict[str, Any] = {}

    for cat_name, (min_len, max_len) in categories.items():
        generated_words: list[str] = []
        attempts = 0
        while len(generated_words) < num_samples_per_cat and attempts < num_samples_per_cat * 200:
            attempts += 1
            length = rng.randint(min_len, max_len)
            w = "".join(rng.choice(chars) for _ in range(length))
            if w in generated_words:
                continue

            if length <= 6:
                if get_string_split(w) != "test":
                    continue
            else:
                pass  # Lengths 7-8 are strictly test

            generated_words.append(w)

        exact_matches = 0
        per_char_scores: list[float] = []
        length_matches = 0
        terminations = 0
        raw_completions: list[str] = []
        step1_scores: list[bool] = []
        step2_scores: list[bool] = []
        sample_records: list[dict[str, Any]] = []

        max_tokens = 96 if arm == "stepwise" else 48

        with torch.no_grad():
            for w in generated_words:
                expected_rev = w[::-1].lower()

                if arm == "char":
                    spaced_w = " ".join(list(w))
                    prompt = f"<|user|>\nReverse the sequence of characters: {spaced_w}\n<|assistant|>\n"
                elif arm == "stepwise":
                    prompt = f"<|user|>\nReverse the sequence of characters step-by-step for the word '{w}'.\n<|assistant|>\n"
                else:
                    # direct / curriculum
                    prompt = f"<|user|>\nReverse the sequence of characters in the word '{w}'.\n<|assistant|>\n"

                prompt_tokens = tokenizer.encode(prompt)
                input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

                for _ in range(max_tokens):
                    if input_tensor.size(1) >= model.config.context_length:
                        break
                    logits, _ = model(input_tensor)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    input_tensor = torch.cat([input_tensor, next_token], dim=1)
                    if next_token.item() == eos_id:
                        break

                generated_full = tokenizer.decode(input_tensor[0].tolist())
                completion = generated_full[len(prompt) :]
                raw_completions.append(completion.strip())

                # Diagnostic metric computation
                pred_ans = extract_reversed_answer(completion, arm=arm)
                is_exact = (pred_ans == expected_rev)
                if is_exact:
                    exact_matches += 1

                # Character-level position match
                matches = sum(1 for c1, c2 in zip(expected_rev, pred_ans) if c1 == c2)
                p_acc = matches / max(len(expected_rev), len(pred_ans), 1)
                per_char_scores.append(p_acc)

                # Length match
                is_len = (len(pred_ans) == len(expected_rev))
                if is_len:
                    length_matches += 1

                # Termination correctness
                is_term = ("<|eos|>" in completion or completion.strip().endswith("."))
                if is_term:
                    terminations += 1

                # Procedural step validation for stepwise
                if arm == "stepwise":
                    s_scores = score_instruction_symbolic_stepwise(prompt, completion)
                    step1_scores.append(s_scores["step1_valid"])
                    step2_scores.append(s_scores["step2_valid"])

                if len(sample_records) < 3:
                    sample_records.append({
                        "word": w,
                        "expected": expected_rev,
                        "predicted": pred_ans,
                        "completion_preview": completion.replace("<|eos|>", "").strip()[:80],
                        "exact_match": is_exact,
                        "per_char_acc": round(p_acc, 4),
                    })

        tot = len(generated_words)
        unique_completions = len(set(raw_completions))
        diversity = unique_completions / tot if tot > 0 else 0.0

        cat_summary: dict[str, Any] = {
            "exact_match_acc": round(exact_matches / tot, 4) if tot > 0 else 0.0,
            "per_char_acc": round(sum(per_char_scores) / tot, 4) if tot > 0 else 0.0,
            "length_match_acc": round(length_matches / tot, 4) if tot > 0 else 0.0,
            "termination_acc": round(terminations / tot, 4) if tot > 0 else 0.0,
            "output_diversity": round(diversity, 4),
            "total_samples": tot,
            "unique_completions": unique_completions,
            "samples": sample_records,
        }

        if arm == "stepwise":
            cat_summary["step1_read_acc"] = round(sum(step1_scores) / tot, 4) if tot > 0 else 0.0
            cat_summary["step2_index_acc"] = round(sum(step2_scores) / tot, 4) if tot > 0 else 0.0

        diag_results[cat_name] = cat_summary

    if was_training:
        model.train()

    return diag_results
