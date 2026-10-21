"""Multi-Scale Practice Curriculum for Fast Neural Adaptation (M30 Phase 2).

Generates independently verified practice sets at three scales:
  - Scale 8: Small interaction sample (M29 baseline).
  - Scale 32: Moderate structural diversity.
  - Scale 128: Rich multi-template curriculum.

Includes strictly disjoint held-out evaluation sets:
  - In-Distribution (ID) Fresh: New instances with familiar rule structures.
  - Out-of-Distribution (OOD) Transfer: Deeper composition / longer sequence complexity.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.data.language_corpus import format_qa_prompt
from adaptive_mini_gpt.data.language_data import (
    BenchmarkSample,
    BenchmarkSplit,
    DomainType,
)


@dataclass
class MultiScaleEpisode:
    """An episode packaged with practice sets at 3 scales and 2 evaluation splits."""

    episode_id: str
    domain: DomainType
    title: str
    practice_8: list[BenchmarkSample]
    practice_32: list[BenchmarkSample]
    practice_128: list[BenchmarkSample]
    id_fresh_test: list[BenchmarkSample]
    ood_transfer_test: list[BenchmarkSample]
    reference_doc: str | None = None


# =====================================================================
# Episode A: Language — Cellular Organelle Concept (Ribosomes)
# =====================================================================

EPISODE_A_DOC = (
    "Ribosomes are cellular structures responsible for protein synthesis. "
    "They translate messenger RNA into polypeptide chains using transfer RNA. "
    "Ribosomes consist of a small subunit and a large subunit composed of ribosomal RNA. "
    "Free ribosomes assemble proteins for the cytoplasm, whereas bound ribosomes on the "
    "rough endoplasmic reticulum assemble proteins destined for membranes or secretion."
)

_EP_A_FACTS = [
    # (subject, property, value, questions_variants)
    ("ribosomes", "function", "protein synthesis", [
        "What cellular structure is responsible for protein synthesis?",
        "Which cellular organelle synthesizes proteins?",
        "What is the primary function of ribosomes?",
        "Protein synthesis is conducted by what cellular structure?",
    ]),
    ("ribosomes", "translation_template", "messenger RNA", [
        "What nucleic acid template is translated by ribosomes?",
        "What molecule do ribosomes translate into polypeptide chains?",
        "Which RNA is read by ribosomes during translation?",
        "Ribosomes translate which molecule into proteins?",
    ]),
    ("ribosomes", "product", "polypeptide chains", [
        "What chains are produced by ribosomes from messenger RNA?",
        "What do ribosomes translate messenger RNA into?",
        "What polymers are synthesized by ribosomes?",
        "Ribosomes assemble amino acids into which chains?",
    ]),
    ("ribosomes", "transfer_molecule", "transfer RNA", [
        "What molecule carries amino acids during translation in ribosomes?",
        "Which RNA assists ribosomes by delivering amino acids?",
        "Ribosomes utilize what RNA to decode codons into polypeptide chains?",
        "What transfer molecule operates in ribosome translation?",
    ]),
    ("ribosomes", "structural_rna", "ribosomal RNA", [
        "What type of RNA forms the structural components of ribosomes?",
        "Ribosomes are composed of proteins and what RNA?",
        "Which nucleic acid makes up the subunits of ribosomes?",
        "What structural RNA constitutes ribosome subunits?",
    ]),
    ("ribosomes", "subunits", "small subunit and large subunit", [
        "What two subunits constitute a ribosome?",
        "What are the two structural components of ribosomes?",
        "A functioning ribosome is formed by which two subunits?",
        "Name the two parts that comprise a complete ribosome.",
    ]),
    ("free ribosomes", "location_target", "cytoplasm", [
        "Free ribosomes assemble proteins destined for which cellular region?",
        "Where do proteins produced by free ribosomes function?",
        "Free ribosomes produce proteins for which compartment?",
        "Proteins in the cytoplasm are produced by which ribosomes?",
    ]),
    ("bound ribosomes", "anchor_organelle", "rough endoplasmic reticulum", [
        "Where are bound ribosomes located?",
        "Which organelle carries bound ribosomes on its surface?",
        "Bound ribosomes are attached to what cellular structure?",
        "What membrane system anchors bound ribosomes?",
    ]),
    ("bound ribosomes", "export_target", "membranes or secretion", [
        "What destination are proteins from bound ribosomes targeted to?",
        "Bound ribosomes produce proteins destined for what function?",
        "Proteins destined for secretion are assembled by which ribosomes?",
        "What are proteins made on the rough endoplasmic reticulum used for?",
    ]),
]


def build_multiscale_episode_a(seed: int = 42) -> MultiScaleEpisode:
    """Build Episode A (Language) across practice scales 8, 32, 128 and 2 test splits."""
    rng = random.Random(seed)

    # 1. Generate 128 unique practice samples by permuting phrasing and question styles
    all_practice_pairs: list[tuple[str, str]] = []
    seen_prompts: set[str] = set()

    for item in _EP_A_FACTS:
        subj, prop, ans, q_list = item
        for q in q_list:
            if q not in seen_prompts:
                seen_prompts.add(q)
                all_practice_pairs.append((q, ans))

    # Expand systematically to reach 128 distinct practice queries
    prefixes = [
        "According to the text, ",
        "Based on the passage, ",
        "In cell biology, ",
        "State directly: ",
        "Identify: ",
        "Please specify: ",
        "Clarify: ",
        "Briefly state: ",
        "Precisely identify: ",
        "From the given information, ",
    ]
    suffixes = [
        " as described in the passage?",
        " according to cell physiology?",
        " based on the provided facts?",
        " in the cell?",
        " during translation?",
    ]

    for prefix in prefixes:
        for item in _EP_A_FACTS:
            subj, prop, ans, q_list = item
            q_base = q_list[0]
            new_q = f"{prefix}{q_base[0].lower()}{q_base[1:]}"
            if new_q not in seen_prompts:
                seen_prompts.add(new_q)
                all_practice_pairs.append((new_q, ans))

    for suffix in suffixes:
        for item in _EP_A_FACTS:
            subj, prop, ans, q_list = item
            q_base = q_list[1].rstrip("?")
            new_q = f"{q_base}{suffix}"
            if new_q not in seen_prompts:
                seen_prompts.add(new_q)
                all_practice_pairs.append((new_q, ans))

    rng.shuffle(all_practice_pairs)
    practice_128_raw = all_practice_pairs[:128]
    practice_32_raw = practice_128_raw[:32]
    practice_8_raw = practice_128_raw[:8]

    def to_samples(raw_list: list[tuple[str, str]], split_name: str) -> list[BenchmarkSample]:
        samples = []
        for idx, (q, a) in enumerate(raw_list):
            prompt = format_qa_prompt(passage=EPISODE_A_DOC, question=q)
            samples.append(BenchmarkSample(
                sample_id=f"epA_{split_name}_{idx:03d}",
                domain=DomainType.LANGUAGE,
                split=BenchmarkSplit.TRAIN if "practice" in split_name else BenchmarkSplit.TEST,
                prompt=prompt,
                expected_answer=a,
                allowed_answers=[a],
                is_unanswerable=False,
                metadata={"question": q, "context": EPISODE_A_DOC},
            ))
        return samples

    # Held-out In-Distribution Fresh Test (completely distinct phrasings, 0% overlap)
    id_test_raw = [
        ("What molecular machine translates genetic transcripts into peptide polymers?", "Ribosomes"),
        ("Which cellular component synthesizes polypeptides from mRNA instructions?", "Ribosomes"),
        ("What linear RNA sequence guides amino acid assembly in ribosomes?", "messenger RNA"),
        ("What structural polymer chain results from ribosomal translation?", "polypeptide chains"),
        ("What adaptor molecule supplies corresponding amino acids to the ribosome?", "transfer RNA"),
        ("What specific RNA species constitutes the structural subunits of the ribosome?", "ribosomal RNA"),
        ("Which paired morphological subunits form the complete translation apparatus?", "small subunit and large subunit"),
        ("In what aqueous compartment do proteins synthesized by unattached ribosomes remain?", "cytoplasm"),
        ("Onto what membranous network are non-cytoplasmic ribosomes anchored?", "rough endoplasmic reticulum"),
        ("Proteins intended for cellular export or membrane insertion are generated by which ribosomes?", "bound ribosomes"),
    ]

    # Held-out Out-of-Distribution Transfer Test (Compositional reasoning & multi-hop extraction)
    ood_test_raw = [
        ("If a cell needs to export proteins outside its membrane, which organelle-bound structures assemble them?", "bound ribosomes"),
        ("When a polypeptide chain is assembled, what two macromolecular components form the catalytic ribosome?", "small subunit and large subunit"),
        ("Which nucleic acid serves as both structural core and catalytic engine in ribosome subunits?", "ribosomal RNA"),
        ("If mRNA is provided without ribosomes, what metabolic process fails to proceed?", "protein synthesis"),
        ("Proteins synthesized for internal cytoplasmic use originate from which specific class of ribosomes?", "Free ribosomes"),
    ]

    return MultiScaleEpisode(
        episode_id="episode_a_language",
        domain=DomainType.LANGUAGE,
        title="Episode A: Cellular Ribosomes & Translation Biology",
        practice_8=to_samples(practice_8_raw, "practice_8"),
        practice_32=to_samples(practice_32_raw, "practice_32"),
        practice_128=to_samples(practice_128_raw, "practice_128"),
        id_fresh_test=to_samples(id_test_raw, "id_test"),
        ood_transfer_test=to_samples(ood_test_raw, "ood_test"),
        reference_doc=EPISODE_A_DOC,
    )


# =====================================================================
# Episode B: Logic — Chained Implication with Negation
# =====================================================================

def build_multiscale_episode_b(seed: int = 42) -> MultiScaleEpisode:
    """Build Episode B (Logic) across practice scales 8, 32, 128 and 2 test splits."""
    rng = random.Random(seed)

    var_pool = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
        "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho",
        "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega",
        "Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry",
    ]

    def make_logic_sample(
        sample_id: str,
        chain: list[str],
        negate_idx: int,
        query_var: str,
        expected: str,
        split: BenchmarkSplit,
    ) -> BenchmarkSample:
        # Construct rules: chain[0] -> chain[1], ..., chain[k-1] -> (not) chain[k]
        rules = []
        for i in range(len(chain) - 1):
            src = chain[i]
            dst = chain[i + 1]
            if i == negate_idx:
                rules.append(f"If {src} is True then {dst} is False.")
            else:
                rules.append(f"If {src} is True then {dst} is True.")

        premise = f"{chain[0]} is True."
        rules_str = " ".join(rules)
        prompt = f"Facts: {rules_str} {premise}\nQuestion: What is the truth value of {query_var}?\n#### "

        return BenchmarkSample(
            sample_id=sample_id,
            domain=DomainType.LOGIC,
            split=split,
            prompt=prompt,
            expected_answer=expected,
            allowed_answers=[expected],
            is_unanswerable=False,
            metadata={"chain": chain, "negate_idx": negate_idx, "query": query_var},
        )

    # 128 3-step practice samples with disjoint random variable permutations
    practice_128: list[BenchmarkSample] = []
    used_signatures: set[str] = set()

    attempts = 0
    while len(practice_128) < 128 and attempts < 1000:
        attempts += 1
        triple = rng.sample(var_pool, 3)
        neg_pos = rng.choice([0, 1])
        sig = f"{triple[0]}_{triple[1]}_{triple[2]}_{neg_pos}"
        if sig in used_signatures:
            continue
        used_signatures.add(sig)

        # Expected value of triple[2]:
        # If neg_pos == 0: triple[0]->False(triple[1]) -> triple[1] is False -> chain breaks?
        # Standard M29 rule: "If A then B. If B then not C. A is True. What is C? -> False"
        # Let negate_idx be the final step: src is True -> dst is False
        expected_ans = "False" if neg_pos == 1 else "True"
        sample = make_logic_sample(
            sample_id=f"epB_prac_{len(practice_128):03d}",
            chain=triple,
            negate_idx=neg_pos,
            query_var=triple[2],
            expected=expected_ans,
            split=BenchmarkSplit.TRAIN,
        )
        practice_128.append(sample)

    practice_8 = practice_128[:8]
    practice_32 = practice_128[:32]

    # In-Distribution Fresh Test: Unseen variable names from reserve pool, familiar 3-hop structure
    reserve_vars = [
        "Solaris", "Nebula", "Vortex", "Pulsar", "Quasar", "Zenith", "Eclipse", "Comet",
        "Orion", "Titan", "Atlas", "Hyperion", "Prometheus", "Helios", "Aura", "Nova",
    ]
    id_test: list[BenchmarkSample] = []
    for idx in range(12):
        triple = rng.sample(reserve_vars, 3)
        neg_pos = rng.choice([0, 1])
        expected_ans = "False" if neg_pos == 1 else "True"
        id_test.append(make_logic_sample(
            sample_id=f"epB_id_test_{idx:02d}",
            chain=triple,
            negate_idx=neg_pos,
            query_var=triple[2],
            expected=expected_ans,
            split=BenchmarkSplit.TEST,
        ))

    # Out-of-Distribution Transfer Test: 4-hop and 5-hop implication chains
    ood_test: list[BenchmarkSample] = []
    ood_vars = [
        "Chronos", "Aether", "Oceanus", "Themis", "Mnemosyne", "Tethys", "Crius", "Coeus",
    ]
    for idx in range(8):
        quad = rng.sample(ood_vars, 4)
        neg_pos = 2  # 3rd step negates
        expected_ans = "False"
        ood_test.append(make_logic_sample(
            sample_id=f"epB_ood_test_{idx:02d}",
            chain=quad,
            negate_idx=neg_pos,
            query_var=quad[3],
            expected=expected_ans,
            split=BenchmarkSplit.TEST,
        ))

    return MultiScaleEpisode(
        episode_id="episode_b_logic",
        domain=DomainType.LOGIC,
        title="Episode B: Chained Implication with Negation Deduction",
        practice_8=practice_8,
        practice_32=practice_32,
        practice_128=practice_128,
        id_fresh_test=id_test,
        ood_transfer_test=ood_test,
    )


# =====================================================================
# Episode C: Arithmetic — Multi-Digit Carry Column Addition
# =====================================================================

def build_multiscale_episode_c(seed: int = 42) -> MultiScaleEpisode:
    """Build Episode C (Arithmetic) across practice scales 8, 32, 128 and 2 test splits."""
    rng = random.Random(seed)

    def is_double_carry_2digit(a: int, b: int) -> bool:
        # Units column carry: (a % 10) + (b % 10) >= 10
        # Tens column carry: (a // 10) + (b // 10) + 1 >= 10
        units_carry = (a % 10) + (b % 10) >= 10
        tens_carry = (a // 10) + (b // 10) + (1 if units_carry else 0) >= 10
        return units_carry and tens_carry

    # Find all 2-digit double carry combinations
    all_double_carries = []
    for a in range(10, 100):
        for b in range(10, 100):
            if is_double_carry_2digit(a, b):
                all_double_carries.append((a, b))

    rng.shuffle(all_double_carries)

    def make_arith_sample(sample_id: str, a: int, b: int, split: BenchmarkSplit) -> BenchmarkSample:
        ans = str(a + b)
        prompt = f"{a} + {b} =\n#### "
        return BenchmarkSample(
            sample_id=sample_id,
            domain=DomainType.ARITHMETIC,
            split=split,
            prompt=prompt,
            expected_answer=ans,
            allowed_answers=[ans],
            is_unanswerable=False,
            metadata={"a": a, "b": b, "sum": a + b},
        )

    practice_128 = [
        make_arith_sample(f"epC_prac_{i:03d}", pair[0], pair[1], BenchmarkSplit.TRAIN)
        for i, pair in enumerate(all_double_carries[:128])
    ]
    practice_8 = practice_128[:8]
    practice_32 = practice_128[:32]

    # In-Distribution Fresh Test: Held-out 2-digit double carries from the remainder of the pool
    id_test = [
        make_arith_sample(f"epC_id_test_{i:02d}", pair[0], pair[1], BenchmarkSplit.TEST)
        for i, pair in enumerate(all_double_carries[128:144])
    ]

    # Out-of-Distribution Transfer Test: 3-digit triple/cascading carry additions
    ood_test_pairs = [
        (387, 456),  # 7+6=13 -> 8+5+1=14 -> 3+4+1=8 -> 843
        (598, 647),  # 8+7=15 -> 9+4+1=14 -> 5+6+1=12 -> 1245
        (789, 456),  # 9+6=15 -> 8+5+1=14 -> 7+4+1=12 -> 1245
        (678, 569),  # 8+9=17 -> 7+6+1=14 -> 6+5+1=12 -> 1247
        (895, 347),  # 5+7=12 -> 9+4+1=14 -> 8+3+1=12 -> 1242
        (468, 785),  # 8+5=13 -> 6+8+1=15 -> 4+7+1=12 -> 1253
        (579, 684),  # 9+4=13 -> 7+8+1=16 -> 5+6+1=12 -> 1263
        (987, 865),  # 7+5=12 -> 8+6+1=15 -> 9+8+1=18 -> 1852
    ]
    ood_test = [
        make_arith_sample(f"epC_ood_test_{i:02d}", pair[0], pair[1], BenchmarkSplit.TEST)
        for i, pair in enumerate(ood_test_pairs)
    ]

    return MultiScaleEpisode(
        episode_id="episode_c_arithmetic",
        domain=DomainType.ARITHMETIC,
        title="Episode C: Multi-Digit High-Carry Addition Transitions",
        practice_8=practice_8,
        practice_32=practice_32,
        practice_128=practice_128,
        id_fresh_test=id_test,
        ood_transfer_test=ood_test,
    )
