"""Language Comprehension and Instruction-Following Benchmark Dataset (M25-B).

Implements reproducible neural language comprehension with strictly disjoint
train, validation, and sealed-test document partitions.

Features:
  - Reading short factual passages.
  - Extracting evidence-backed information.
  - Recognizing unanswerable questions (safe abstention).
  - Paraphrased query robustness.
  - Token loss masking on prompt tokens (assistant-only supervision).
  - Document-level split isolation (0% fact or document overlap).
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.evaluation.multi_domain import (
    BenchmarkSample,
    BenchmarkSplit,
    DomainType,
)

ABSTENTION_TARGET = "Insufficient information."
ABSTENTION_VARIANTS = {
    "insufficient information",
    "insufficient information.",
    "unanswerable",
    "not mentioned",
    "unknown",
    "not specified in the passage",
    "cannot be determined",
}


@dataclass
class DocumentPassage:
    """A distinct factual passage with grounded answerable and unanswerable questions."""

    doc_id: str
    topic: str
    text: str
    answerable_pairs: list[tuple[str, str, list[str]]]  # (question, canonical_answer, aliases)
    unanswerable_questions: list[str]
    paraphrase_pairs: list[tuple[str, str, list[str]]] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.paraphrase_pairs is None:
            self.paraphrase_pairs = []


# =====================================================================
# Strictly Disjoint Document Partitions (0% Fact Overlap)
# =====================================================================

TRAIN_DOCUMENTS: list[DocumentPassage] = [
    DocumentPassage(
        doc_id="train_doc_01_photosynthesis",
        topic="Biology",
        text="Photosynthesis occurs inside chloroplasts using chlorophyll pigments. Plants absorb sunlight and carbon dioxide to synthesize glucose and release oxygen.",
        answerable_pairs=[
            ("Where does photosynthesis occur inside plant cells?", "Chloroplasts", ["in chloroplasts", "chloroplast"]),
            ("What gas do plants release during photosynthesis?", "Oxygen", ["oxygen gas", "O2"]),
            ("What sugar is synthesized during photosynthesis?", "Glucose", ["glucose sugar"]),
        ],
        unanswerable_questions=[
            ("What temperature is required for photosynthesis?",),
            ("How many ATP molecules are produced during the light reaction?",),
        ],
        paraphrase_pairs=[
            ("Which cellular organelle hosts photosynthesis?", "Chloroplasts", ["in chloroplasts", "chloroplast"]),
            ("What byproduct gas is produced by plants during photosynthesis?", "Oxygen", ["oxygen gas", "O2"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_02_solar_system",
        topic="Astronomy",
        text="Jupiter is the largest planet in our solar system and possesses a persistent storm called the Great Red Spot. It is primarily composed of hydrogen and helium gases.",
        answerable_pairs=[
            ("Which is the largest planet in our solar system?", "Jupiter", ["planet Jupiter"]),
            ("What is the persistent storm on Jupiter called?", "Great Red Spot", ["the Great Red Spot"]),
            ("What is Jupiter primarily composed of?", "Hydrogen and helium", ["hydrogen and helium gases", "hydrogen, helium"]),
        ],
        unanswerable_questions=[
            ("How many moons orbit Jupiter?",),
            ("What is the surface temperature of Jupiter's core?",),
        ],
        paraphrase_pairs=[
            ("Name the biggest planet in the solar system.", "Jupiter", ["planet Jupiter"]),
            ("What name is given to Jupiter's giant storm?", "Great Red Spot", ["the Great Red Spot"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_03_printing_press",
        topic="History",
        text="Johannes Gutenberg invented the movable type printing press in Mainz around 1440. His innovation dramatically accelerated the spread of books across Europe.",
        answerable_pairs=[
            ("Who invented the movable type printing press?", "Johannes Gutenberg", ["Gutenberg", "Johannes Gutenberg in Mainz"]),
            ("In which city was Gutenberg's press developed?", "Mainz", ["city of Mainz"]),
            ("Around what year did Gutenberg invent his press?", "1440", ["around 1440", "circa 1440"]),
        ],
        unanswerable_questions=[
            ("How many Gutenberg Bibles were originally printed?",),
            ("What metal alloy was used to cast Gutenberg's type?",),
        ],
        paraphrase_pairs=[
            ("Which inventor created movable type printing in 1440?", "Johannes Gutenberg", ["Gutenberg"]),
            ("Where did Gutenberg invent the printing press?", "Mainz", ["in Mainz"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_04_aircraft_lift",
        topic="Physics",
        text="Aircraft wings generate aerodynamic lift due to differences in air pressure above and below the airfoil. The forward thrust is provided by jet engines or propellers.",
        answerable_pairs=[
            ("What causes aerodynamic lift on aircraft wings?", "Differences in air pressure", ["pressure difference", "air pressure differences"]),
            ("What provides forward thrust in aircraft?", "Jet engines or propellers", ["engines or propellers", "jet engines"]),
        ],
        unanswerable_questions=[
            ("What is the top cruising speed of a Boeing 747?",),
            ("How much fuel does an Airbus A380 consume per hour?",),
        ],
        paraphrase_pairs=[
            ("Why do airplane wings produce lift?", "Differences in air pressure", ["pressure differences", "air pressure"]),
        ],
    ),
]

VAL_DOCUMENTS: list[DocumentPassage] = [
    DocumentPassage(
        doc_id="val_doc_01_plate_tectonics",
        topic="Geology",
        text="The Earth's lithosphere is divided into rigid tectonic plates that float on the ductile asthenosphere. Plate boundaries cause earthquakes and volcanic eruptions.",
        answerable_pairs=[
            ("What rigid layer of the Earth is divided into tectonic plates?", "Lithosphere", ["the lithosphere", "Earth's lithosphere"]),
            ("On which ductile layer do tectonic plates float?", "Asthenosphere", ["the asthenosphere", "ductile asthenosphere"]),
            ("What geological events are caused by plate boundaries?", "Earthquakes and volcanic eruptions", ["earthquakes and volcanoes", "earthquakes"]),
        ],
        unanswerable_questions=[
            ("How fast do the Pacific and North American plates move each year?",),
            ("What is the depth of the Mariana Trench in kilometers?",),
        ],
        paraphrase_pairs=[
            ("Which layer of Earth is partitioned into plates?", "Lithosphere", ["the lithosphere"]),
            ("Where do tectonic plates drift on top of?", "Asthenosphere", ["the asthenosphere"]),
        ],
    ),
    DocumentPassage(
        doc_id="val_doc_02_optics_lenses",
        topic="Physics",
        text="A convex lens converges parallel rays of light toward a focal point, whereas a concave lens diverges light rays. Optical refraction depends on the refractive index of the glass.",
        answerable_pairs=[
            ("What does a convex lens do to parallel light rays?", "Converges light rays toward a focal point", ["converges light", "converges them"]),
            ("What does a concave lens do to light rays?", "Diverges light rays", ["diverges them", "diverges light"]),
            ("What material property determines refraction in glass?", "Refractive index", ["the refractive index"]),
        ],
        unanswerable_questions=[
            ("What is the exact refractive index of flint glass?",),
            ("Who first ground optical glass lenses for telescopes?",),
        ],
        paraphrase_pairs=[
            ("Which type of lens converges parallel rays?", "Convex lens", ["convex"]),
            ("Which lens diverges incoming light rays?", "Concave lens", ["concave"]),
        ],
    ),
]

TEST_DOCUMENTS: list[DocumentPassage] = [
    DocumentPassage(
        doc_id="test_doc_01_hydrothermal_vents",
        topic="Marine Biology",
        text="Deep-sea hydrothermal vents emit mineral-rich fluids heated by underlying magma chambers. Chemosynthetic bacteria utilize hydrogen sulfide to produce organic matter without sunlight.",
        answerable_pairs=[
            ("What heats the mineral-rich fluids emitted by hydrothermal vents?", "Underlying magma chambers", ["magma chambers", "magma"]),
            ("What microorganism produces organic matter at hydrothermal vents?", "Chemosynthetic bacteria", ["bacteria", "chemosynthetic microbes"]),
            ("What chemical compound do chemosynthetic bacteria utilize?", "Hydrogen sulfide", ["H2S", "hydrogen sulfide gas"]),
        ],
        unanswerable_questions=[
            ("What is the maximum water pressure at the Mariana hydrothermal vent?",),
            ("In what year were deep hydrothermal vents first discovered by submersible?",),
        ],
        paraphrase_pairs=[
            ("What energy source heats fluids at hydrothermal vents?", "Magma chambers", ["magma", "underlying magma"]),
            ("Which chemical do vent bacteria convert into food?", "Hydrogen sulfide", ["hydrogen sulfide gas"]),
        ],
    ),
    DocumentPassage(
        doc_id="test_doc_02_james_webb_telescope",
        topic="Space Exploration",
        text="The James Webb Space Telescope operates at the second Lagrange point, 1.5 million kilometers from Earth. Its primary mirror consists of eighteen gold-coated beryllium segments.",
        answerable_pairs=[
            ("Where does the James Webb Space Telescope operate?", "Second Lagrange point", ["L2", "Lagrange point 2", "second Lagrange point L2"]),
            ("How far is the James Webb Space Telescope from Earth?", "1.5 million kilometers", ["1.5 million km", "1.5 million kilometers from Earth"]),
            ("What metal coats the eighteen beryllium segments of its mirror?", "Gold", ["gold coating", "gold-coated"]),
            ("How many beryllium segments comprise its primary mirror?", "Eighteen", ["18", "eighteen segments"]),
        ],
        unanswerable_questions=[
            ("What rocket booster launched the James Webb Space Telescope?",),
            ("How much did the construction of the James Webb telescope cost in total?",),
        ],
        paraphrase_pairs=[
            ("At which orbital location is the Webb telescope stationed?", "Second Lagrange point", ["L2", "Lagrange point 2"]),
            ("What material coats the mirror segments of the Webb telescope?", "Gold", ["gold-coated"]),
        ],
    ),
    DocumentPassage(
        doc_id="test_doc_03_analytical_engine",
        topic="Computer History",
        text="Charles Babbage designed the mechanical Analytical Engine in London, featuring an arithmetic unit called the mill and a memory store. Ada Lovelace wrote the first published algorithm for this machine.",
        answerable_pairs=[
            ("Who designed the mechanical Analytical Engine?", "Charles Babbage", ["Babbage", "Charles Babbage in London"]),
            ("What was the arithmetic calculation unit of the Analytical Engine called?", "The mill", ["mill"]),
            ("Who created the first algorithm for the Analytical Engine?", "Ada Lovelace", ["Lovelace", "Ada Lovelace in London"]),
        ],
        unanswerable_questions=[
            ("How many steam cylinders were intended to power the Analytical Engine?",),
            ("What decimal precision in digits did Babbage plan for the store?",),
        ],
        paraphrase_pairs=[
            ("Name the designer of the Analytical Engine.", "Charles Babbage", ["Babbage"]),
            ("Who authored the earliest published algorithm for Babbage's engine?", "Ada Lovelace", ["Lovelace"]),
        ],
    ),
]


def format_qa_prompt(passage: str, question: str) -> str:
    """Format standardized reading comprehension prompt with reserved tokens."""
    return (
        f"<|user|>\n"
        f"Passage: {passage.strip()}\n"
        f"Question: {question.strip()}\n"
        f"Answer concisely using only the passage above, or reply '{ABSTENTION_TARGET}' if not answerable.\n"
        f"<|assistant|>\n"
    )


def format_qa_completion(answer: str) -> str:
    """Format assistant target completion with end-of-sequence token."""
    return f"{answer.strip()}<|eos|>\n"


def parse_language_completion(completion: str) -> tuple[str, bool, bool]:
    """Parse assistant completion into (cleaned_answer, is_compliant, is_abstained)."""
    cleaned = completion.replace("<|eos|>", "").strip()
    if not cleaned:
        return "", False, False

    first_line = cleaned.split("\n")[0].strip()
    # Normalize punctuation and case for abstention check
    normalized = re.sub(r"[^\w\s]", "", first_line.lower()).strip()

    is_abstained = any(abstain in normalized for abstain in ["insufficient information", "unanswerable", "not mentioned", "unknown", "cannot be determined"])
    is_compliant = len(first_line) > 0

    return first_line, is_compliant, is_abstained


def verify_language_answer(parsed_answer: str, sample: BenchmarkSample) -> bool:
    """Independently verify whether parsed language answer matches expected ground truth."""
    if not parsed_answer:
        return False

    norm_parsed = re.sub(r"[^\w\s]", "", parsed_answer.lower()).strip()

    # Case 1: Sample is unanswerable from the passage (must abstain)
    if sample.is_unanswerable:
        return any(abstain in norm_parsed for abstain in ["insufficient information", "unanswerable", "not mentioned", "unknown", "cannot be determined"])

    # If the model abstained on an answerable question, it is incorrect
    if any(abstain in norm_parsed for abstain in ["insufficient information", "unanswerable", "not mentioned", "unknown"]):
        return False

    # Case 2: Answerable question. Check exact or canonical/allowed match
    norm_expected = re.sub(r"[^\w\s]", "", sample.expected_answer.lower()).strip()
    if norm_expected in norm_parsed or norm_parsed in norm_expected:
        return True

    for allowed in sample.allowed_answers:
        norm_allowed = re.sub(r"[^\w\s]", "", allowed.lower()).strip()
        if norm_allowed in norm_parsed or norm_parsed in norm_allowed:
            return True

    return False


def build_language_benchmark_samples(
    split: BenchmarkSplit,
    include_paraphrases: bool = True,
) -> list[BenchmarkSample]:
    """Construct deterministic BenchmarkSample list for the designated split."""
    if split == BenchmarkSplit.TRAIN:
        doc_pool = TRAIN_DOCUMENTS
    elif split == BenchmarkSplit.VAL:
        doc_pool = VAL_DOCUMENTS
    elif split == BenchmarkSplit.TEST:
        doc_pool = TEST_DOCUMENTS
    else:
        raise ValueError(f"Unknown split: {split}")

    samples: list[BenchmarkSample] = []
    for doc in doc_pool:
        # 1. Answerable questions
        for idx, (q, ans, aliases) in enumerate(doc.answerable_pairs):
            sample_id = f"{doc.doc_id}_ans_{idx}"
            prompt = format_qa_prompt(doc.text, q)
            samples.append(
                BenchmarkSample(
                    sample_id=sample_id,
                    domain=DomainType.LANGUAGE,
                    split=split,
                    prompt=prompt,
                    expected_answer=ans,
                    allowed_answers=aliases,
                    is_unanswerable=False,
                    metadata={"doc_id": doc.doc_id, "topic": doc.topic, "type": "factual_extraction"},
                )
            )

        # 2. Unanswerable questions (abstention battery)
        for idx, q_tuple in enumerate(doc.unanswerable_questions):
            q = q_tuple[0] if isinstance(q_tuple, tuple) else q_tuple
            sample_id = f"{doc.doc_id}_unans_{idx}"
            prompt = format_qa_prompt(doc.text, q)
            samples.append(
                BenchmarkSample(
                    sample_id=sample_id,
                    domain=DomainType.LANGUAGE,
                    split=split,
                    prompt=prompt,
                    expected_answer=ABSTENTION_TARGET,
                    allowed_answers=list(ABSTENTION_VARIANTS),
                    is_unanswerable=True,
                    metadata={"doc_id": doc.doc_id, "topic": doc.topic, "type": "safe_abstention"},
                )
            )

        # 3. Paraphrased questions (robustness test)
        if include_paraphrases and doc.paraphrase_pairs:
            for idx, (q, ans, aliases) in enumerate(doc.paraphrase_pairs):
                sample_id = f"{doc.doc_id}_para_{idx}"
                prompt = format_qa_prompt(doc.text, q)
                samples.append(
                    BenchmarkSample(
                        sample_id=sample_id,
                        domain=DomainType.LANGUAGE,
                        split=split,
                        prompt=prompt,
                        expected_answer=ans,
                        allowed_answers=aliases,
                        is_unanswerable=False,
                        metadata={"doc_id": doc.doc_id, "topic": doc.topic, "type": "paraphrase_robustness"},
                    )
                )

    return samples


class LanguageComprehensionDataset(Dataset):
    """PyTorch Dataset for language comprehension with prompt token loss masking."""

    def __init__(
        self,
        samples: list[BenchmarkSample],
        tokenizer: BPETokenizer,
        context_length: int = 256,
        ignore_index: int = -100,
    ) -> None:
        self.context_length = context_length
        self.ignore_index = ignore_index
        self.data: list[tuple[Tensor, Tensor]] = []

        for s in samples:
            prompt_ids = tokenizer.encode(s.prompt)
            answer_ids = tokenizer.encode(format_qa_completion(s.expected_answer))
            token_ids = prompt_ids + answer_ids

            if len(token_ids) > context_length:
                # Keep the suffix so the assistant answer is preserved
                token_ids = token_ids[-context_length:]
                prompt_len = max(0, len(token_ids) - len(answer_ids))
            else:
                prompt_len = len(prompt_ids)

            input_ids = torch.tensor(token_ids, dtype=torch.long)
            # Create shifted target labels with ignore_index on prompt tokens
            target_ids = input_ids.clone()
            target_ids[:prompt_len] = ignore_index

            self.data.append((input_ids, target_ids))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.data[idx]


def collate_language_batch(
    batch: list[tuple[Tensor, Tensor]],
    pad_token_id: int,
    ignore_index: int = -100,
) -> tuple[Tensor, Tensor]:
    """Pad variable-length language comprehension samples to max sequence length in batch."""
    max_len = max(x.size(0) for x, _ in batch)
    inputs_padded: list[Tensor] = []
    targets_padded: list[Tensor] = []

    for x, y in batch:
        pad_len = max_len - x.size(0)
        if pad_len > 0:
            x_pad = torch.cat([x, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            y_pad = torch.cat([y, torch.full((pad_len,), ignore_index, dtype=torch.long)])
        else:
            x_pad = x
            y_pad = y
        inputs_padded.append(x_pad)
        targets_padded.append(y_pad)

    return torch.stack(inputs_padded, dim=0), torch.stack(targets_padded, dim=0)
