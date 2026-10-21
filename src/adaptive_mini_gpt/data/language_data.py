"""Language Comprehension and Instruction-Following Benchmark Dataset (M25-B / M26 / M27).

Implements reproducible neural language comprehension with strictly disjoint
train, validation, and sealed-test document partitions.

Features:
  - Reading short factual passages across diverse scientific and historical domains.
  - Extracting evidence-backed information with short, canonical answers.
  - Recognizing unanswerable questions (calibrated safe abstention).
  - Paraphrased query robustness.
  - Prompt formatting without recency copying bias.
  - Token loss masking on prompt tokens (assistant-only supervision with ignore_index=-100).
  - Document-level split isolation (0% fact, topic, or document overlap between splits).
  - Dual evaluation metrics: exact match and grounded extraction.
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
        topic="Plant Biology",
        text="Photosynthesis occurs inside chloroplasts using chlorophyll pigments. Plants absorb sunlight and carbon dioxide to synthesize glucose and release oxygen.",
        answerable_pairs=[
            ("Where does photosynthesis occur inside plant cells?", "Chloroplasts", ["in chloroplasts", "chloroplast"]),
            ("What gas do plants release during photosynthesis?", "Oxygen", ["oxygen gas", "O2"]),
            ("What sugar is synthesized during photosynthesis?", "Glucose", ["glucose sugar"]),
        ],
        unanswerable_questions=[
            "What temperature is required for the Calvin cycle?",
            "How many ATP molecules are produced during the light reaction?",
        ],
        paraphrase_pairs=[
            ("Which cellular organelle hosts photosynthesis?", "Chloroplasts", ["chloroplast", "in chloroplasts"]),
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
            ("What elements is Jupiter primarily composed of?", "Hydrogen and helium", ["hydrogen and helium gases"]),
        ],
        unanswerable_questions=[
            "How many moons orbit Jupiter?",
            "What is the surface temperature of Jupiter's core?",
        ],
        paraphrase_pairs=[
            ("Name the biggest planet in the solar system.", "Jupiter", ["planet Jupiter"]),
            ("What name is given to Jupiter's giant storm?", "Great Red Spot", ["the Great Red Spot"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_03_printing_press",
        topic="European History",
        text="Johannes Gutenberg invented the movable type printing press in Mainz around 1440. His innovation dramatically accelerated the spread of books across Europe.",
        answerable_pairs=[
            ("Who invented the movable type printing press?", "Johannes Gutenberg", ["Gutenberg"]),
            ("In which city was Gutenberg's press developed?", "Mainz", ["city of Mainz"]),
            ("Around what year did Gutenberg invent his press?", "1440", ["around 1440", "circa 1440"]),
        ],
        unanswerable_questions=[
            "How many Gutenberg Bibles were originally printed?",
            "What metal alloy was used to cast Gutenberg's type?",
        ],
        paraphrase_pairs=[
            ("Which inventor created movable type printing in 1440?", "Johannes Gutenberg", ["Gutenberg"]),
            ("Where did Gutenberg invent the printing press?", "Mainz", ["in Mainz"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_04_aircraft_lift",
        topic="Aerodynamics",
        text="Aircraft wings generate aerodynamic lift due to differences in air pressure above and below the airfoil. The forward thrust is provided by jet engines or propellers.",
        answerable_pairs=[
            ("What causes aerodynamic lift on aircraft wings?", "Differences in air pressure", ["air pressure differences", "pressure differences"]),
            ("What provides forward thrust in aircraft?", "Jet engines or propellers", ["engines or propellers", "jet engines"]),
        ],
        unanswerable_questions=[
            "What is the top cruising speed of a Boeing 747?",
            "How much fuel does an Airbus A380 consume per hour?",
        ],
        paraphrase_pairs=[
            ("Why do airplane wings produce lift?", "Differences in air pressure", ["pressure differences"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_05_microprocessors",
        topic="Semiconductor Technology",
        text="Microprocessors are fabricated on silicon wafers using optical photolithography. Billions of field-effect transistors act as electronic switches to execute arithmetic logic operations.",
        answerable_pairs=[
            ("What semiconductor substrate is used to fabricate microprocessors?", "Silicon wafers", ["silicon", "silicon wafer"]),
            ("Which optical manufacturing process creates microprocessor circuits?", "Photolithography", ["optical photolithography"]),
            ("What electronic components act as switches in microprocessors?", "Field-effect transistors", ["transistors"]),
        ],
        unanswerable_questions=[
            "What is the exact clock frequency of the first Intel 4004 chip?",
            "What is the gate dielectric thickness in modern 3nm transistors?",
        ],
        paraphrase_pairs=[
            ("On which material are microprocessors manufactured?", "Silicon wafers", ["silicon"]),
            ("Name the technique used to pattern microprocessor circuits.", "Photolithography", ["optical photolithography"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_06_rainforest_ecology",
        topic="Ecology",
        text="Tropical rainforest canopies absorb high solar radiation and generate localized rainfall through plant transpiration. Epiphytic orchids grow directly upon tree branches to capture canopy moisture.",
        answerable_pairs=[
            ("Through which biological process do rainforest canopies produce rainfall?", "Plant transpiration", ["transpiration"]),
            ("Where do epiphytic orchids grow in the rainforest?", "Tree branches", ["directly upon tree branches", "on branches"]),
        ],
        unanswerable_questions=[
            "How many tree species exist per hectare in the Amazon basin?",
            "What is the annual biomass turnover rate in tropical rainforest soils?",
        ],
        paraphrase_pairs=[
            ("What botanical mechanism drives rainfall in tropical rainforests?", "Plant transpiration", ["transpiration"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_07_neuroscience_memory",
        topic="Neuroscience",
        text="The hippocampus consolidates short-term neural signals into stable long-term memories in the cerebral cortex. Synaptic plasticity enables persistent signal strengthening known as long-term potentiation.",
        answerable_pairs=[
            ("Which brain structure consolidates short-term memories?", "Hippocampus", ["the hippocampus"]),
            ("Where are long-term memories permanently stored?", "Cerebral cortex", ["the cerebral cortex", "cortex"]),
            ("What synaptic mechanism enables persistent signal strengthening?", "Long-term potentiation", ["LTP"]),
        ],
        unanswerable_questions=[
            "How many action potentials can a pyramidal neuron fire per second?",
            "Which neurotransmitter receptor subtype binds glutamate in memory retrieval?",
        ],
        paraphrase_pairs=[
            ("Name the anatomical brain region that consolidates memories.", "Hippocampus", ["the hippocampus"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_08_geodesic_architecture",
        topic="Civil Architecture",
        text="Architect Buckminster Fuller popularized the geodesic dome, which utilizes spherical networks of interconnected triangles. The structure distributes mechanical stress omnidirectionally through tensegrity.",
        answerable_pairs=[
            ("Who popularized the geodesic dome?", "Buckminster Fuller", ["Fuller", "R. Buckminster Fuller"]),
            ("What geometric shape forms the network of geodesic domes?", "Interconnected triangles", ["triangles"]),
            ("How does a geodesic dome distribute mechanical stress?", "Omnidirectionally through tensegrity", ["omnidirectionally", "through tensegrity"]),
        ],
        unanswerable_questions=[
            "What was the construction diameter of the Montreal Biosphere dome?",
            "What alloy formed the outer struts of Fuller's prototype dome?",
        ],
        paraphrase_pairs=[
            ("Which architect is famous for designing geodesic domes?", "Buckminster Fuller", ["Fuller"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_09_meteorology_cyclones",
        topic="Meteorology",
        text="Tropical cyclones form over warm ocean waters exceeding 26 degrees Celsius. The Coriolis force imparts rotational spin, creating a calm low-pressure central eye surrounded by eyewall cloud bands.",
        answerable_pairs=[
            ("What minimum water temperature is required for tropical cyclones?", "26 degrees Celsius", ["26 C", "exceeding 26 degrees Celsius"]),
            ("What physical force imparts rotational spin to cyclones?", "Coriolis force", ["the Coriolis force", "Coriolis"]),
            ("What is the calm central region of a tropical cyclone called?", "The eye", ["eye", "central eye"]),
        ],
        unanswerable_questions=[
            "What was the lowest atmospheric pressure ever recorded in Typhoon Tip?",
            "How many nautical miles wide can hurricane storm surges reach?",
        ],
        paraphrase_pairs=[
            ("Which natural effect causes cyclone rotation?", "Coriolis force", ["Coriolis"]),
            ("Name the tranquil center of a tropical cyclone.", "The eye", ["eye", "central eye"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_10_graphene_materials",
        topic="Materials Science",
        text="Graphene is a single atomic monolayer of carbon atoms bonded in a hexagonal honeycomb lattice. It demonstrates extraordinary electrical conductivity and ballistic electron transport at room temperature.",
        answerable_pairs=[
            ("What chemical element comprises graphene?", "Carbon", ["carbon atoms", "pure carbon"]),
            ("What geometric lattice shape do carbon atoms form in graphene?", "Hexagonal honeycomb lattice", ["honeycomb lattice", "hexagonal lattice"]),
            ("What electrical transport phenomenon occurs in graphene at room temperature?", "Ballistic electron transport", ["ballistic transport"]),
        ],
        unanswerable_questions=[
            "What is the Young's modulus of multilayer turbostratic graphene?",
            "Who first synthesized suspended graphene on copper foils?",
        ],
        paraphrase_pairs=[
            ("Which element is graphene composed of?", "Carbon", ["carbon atoms"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_11_steam_engine",
        topic="Industrial Technology",
        text="James Watt developed the separate condenser for steam engines in Glasgow in 1765. This breakthrough drastically reduced fuel consumption and powered the Industrial Revolution.",
        answerable_pairs=[
            ("Who developed the separate condenser for steam engines?", "James Watt", ["Watt"]),
            ("In what year did James Watt develop the separate condenser?", "1765", ["in 1765"]),
            ("In which city was the separate condenser developed?", "Glasgow", ["city of Glasgow"]),
        ],
        unanswerable_questions=[
            "How many pounds of coal did Newcomen engines consume per minute?",
            "What was the cylinder stroke diameter of Watt's first commercial engine?",
        ],
        paraphrase_pairs=[
            ("Who invented the separate condenser in 1765?", "James Watt", ["Watt"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_12_ocean_currents",
        topic="Ocean Dynamics",
        text="The Gulf Stream is a swift warm ocean current originating in the Gulf of Mexico. It carries heated water across the Atlantic Ocean toward Western Europe, moderating coastal climates.",
        answerable_pairs=[
            ("Where does the Gulf Stream originate?", "Gulf of Mexico", ["the Gulf of Mexico"]),
            ("What type of ocean current is the Gulf Stream?", "Warm ocean current", ["warm current", "swift warm current"]),
            ("Toward which continent does the Gulf Stream transport warm water?", "Western Europe", ["Europe"]),
        ],
        unanswerable_questions=[
            "What is the maximum depth of the Florida Straits?",
            "How many sverdrups of volume transport does the Gulf Stream reach at Cape Hatteras?",
        ],
        paraphrase_pairs=[
            ("Where does the Gulf Stream begin?", "Gulf of Mexico", ["the Gulf of Mexico"]),
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
            "How fast do the Pacific and North American plates move each year?",
            "What is the depth of the Mariana Trench in kilometers?",
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
            ("What does a convex lens do to parallel light rays?", "Converges light rays toward a focal point", ["converges light rays", "converges light"]),
            ("What does a concave lens do to light rays?", "Diverges light rays", ["diverges light"]),
            ("What material property determines refraction in glass?", "Refractive index", ["the refractive index"]),
        ],
        unanswerable_questions=[
            "What is the exact refractive index of flint glass?",
            "Who first ground optical glass lenses for telescopes?",
        ],
        paraphrase_pairs=[
            ("Which type of lens converges parallel rays?", "Convex lens", ["convex"]),
            ("Which lens diverges incoming light rays?", "Concave lens", ["concave"]),
        ],
    ),
    DocumentPassage(
        doc_id="val_doc_03_black_holes",
        topic="Astrophysics",
        text="A black hole is bounded by an event horizon from which no matter or electromagnetic radiation can escape. At its mathematical center lies a gravitational singularity of infinite density.",
        answerable_pairs=[
            ("What boundary marks the point of no return around a black hole?", "Event horizon", ["the event horizon"]),
            ("What exists at the center of a black hole?", "Gravitational singularity", ["singularity", "a singularity"]),
        ],
        unanswerable_questions=[
            "What is the Hawking radiation temperature of Cygnus X-1?",
            "How fast does the accretion disk of Sagittarius A* rotate?",
        ],
        paraphrase_pairs=[
            ("Name the perimeter beyond which light cannot escape a black hole.", "Event horizon", ["the event horizon"]),
        ],
    ),
    DocumentPassage(
        doc_id="val_doc_04_photovoltaics",
        topic="Renewable Energy",
        text="Photovoltaic solar cells convert sunlight into electricity via the photoelectric effect within a semiconductor p-n junction. When photons strike the cell, electrons are energized into the conduction band.",
        answerable_pairs=[
            ("Which physical effect converts sunlight to electricity in solar cells?", "Photoelectric effect", ["the photoelectric effect"]),
            ("What semiconductor junction structure is used inside photovoltaic cells?", "p-n junction", ["a p-n junction", "semiconductor p-n junction"]),
            ("Where are electrons excited into when struck by photons?", "Conduction band", ["the conduction band"]),
        ],
        unanswerable_questions=[
            "What is the record laboratory efficiency of perovskite tandem cells?",
            "What chemical dopant is added to n-type silicon wafers?",
        ],
        paraphrase_pairs=[
            ("Which phenomenon generates electric current in solar panels?", "Photoelectric effect", ["the photoelectric effect"]),
        ],
    ),
]

TEST_DOCUMENTS: list[DocumentPassage] = [
    DocumentPassage(
        doc_id="test_doc_01_hydrothermal_vents",
        topic="Marine Biology",
        text="Deep-sea hydrothermal vents emit mineral-rich fluids heated by underlying magma chambers. Chemosynthetic bacteria utilize hydrogen sulfide to produce organic matter without sunlight.",
        answerable_pairs=[
            ("What heats the mineral-rich fluids emitted by hydrothermal vents?", "Magma chambers", ["underlying magma chambers", "magma"]),
            ("What microorganism produces organic matter at hydrothermal vents?", "Chemosynthetic bacteria", ["bacteria", "chemosynthetic microbes"]),
            ("What chemical compound do chemosynthetic bacteria utilize?", "Hydrogen sulfide", ["H2S", "hydrogen sulfide gas"]),
        ],
        unanswerable_questions=[
            "What is the maximum water pressure at the Mariana hydrothermal vent?",
            "In what year were deep hydrothermal vents first discovered by submersible?",
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
            "What rocket booster launched the James Webb Space Telescope?",
            "How much did the construction of the James Webb telescope cost in total?",
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
            "How many steam cylinders were intended to power the Analytical Engine?",
            "What decimal precision in digits did Babbage plan for the store?",
        ],
        paraphrase_pairs=[
            ("Name the designer of the Analytical Engine.", "Charles Babbage", ["Babbage"]),
            ("Who authored the earliest published algorithm for Babbage's engine?", "Ada Lovelace", ["Lovelace"]),
        ],
    ),
]


def format_qa_prompt(passage: str, question: str) -> str:
    """Format standardized reading comprehension prompt without recency copying hazard."""
    return (
        f"<|user|>\n"
        f"Passage: {passage.strip()}\n"
        f"Question: {question.strip()}\n"
        f"Instruction: Extract the short factual answer directly from the passage above. If not present in the passage, reply '{ABSTENTION_TARGET}'.\n"
        f"<|assistant|>\n"
    )


def format_qa_completion(answer: str) -> str:
    """Format assistant target completion with end-of-sequence token."""
    return f"{answer.strip()}<|eos|>\n"


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, removing punctuation, and stripping articles."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    # Strip leading/trailing articles
    text = re.sub(r"^(a|an|the)\s+", "", text)
    text = re.sub(r"\s+(a|an|the)$", "", text)
    return " ".join(text.split())


def parse_language_completion(completion: str) -> tuple[str, bool, bool]:
    """Parse assistant completion into (cleaned_answer, is_compliant, is_abstained)."""
    cleaned = completion.replace("<|eos|>", "").strip()
    if not cleaned:
        return "", False, False

    first_line = cleaned.split("\n")[0].strip()
    normalized = normalize_text(first_line)

    is_abstained = any(abstain in normalized for abstain in [
        "insufficient information",
        "unanswerable",
        "not mentioned",
        "unknown",
        "cannot be determined",
    ])
    is_compliant = len(first_line) > 0

    return first_line, is_compliant, is_abstained


def verify_language_exact(parsed_answer: str, sample: BenchmarkSample) -> bool:
    """Check exact match after standard normalization and article stripping."""
    if not parsed_answer:
        return False

    norm_parsed = normalize_text(parsed_answer)

    # Case 1: Unanswerable
    if sample.is_unanswerable:
        return any(term in norm_parsed for term in [
            "insufficient information",
            "unanswerable",
            "not mentioned",
            "unknown",
            "cannot be determined",
        ])

    # If abstained on answerable question -> False
    if any(term in norm_parsed for term in ["insufficient information", "unanswerable", "not mentioned", "unknown"]):
        return False

    norm_expected = normalize_text(sample.expected_answer)
    if norm_parsed == norm_expected:
        return True

    for allowed in sample.allowed_answers:
        if norm_parsed == normalize_text(allowed):
            return True

    return False


def verify_language_grounded(parsed_answer: str, sample: BenchmarkSample) -> bool:
    """Check grounded semantic/substring match, guarding against trivial stopword false positives."""
    if not parsed_answer:
        return False

    norm_parsed = normalize_text(parsed_answer)

    # Case 1: Unanswerable
    if sample.is_unanswerable:
        return any(term in norm_parsed for term in [
            "insufficient information",
            "unanswerable",
            "not mentioned",
            "unknown",
            "cannot be determined",
        ])

    # If abstained on answerable question -> False
    if any(term in norm_parsed for term in ["insufficient information", "unanswerable", "not mentioned", "unknown"]):
        return False

    norm_expected = normalize_text(sample.expected_answer)
    if norm_parsed == norm_expected:
        return True

    # Meaningful substring match (require length >= 3 to avoid stopword false matches)
    if len(norm_parsed) >= 3 and norm_parsed in norm_expected:
        return True
    if len(norm_expected) >= 3 and norm_expected in norm_parsed:
        return True

    for allowed in sample.allowed_answers:
        norm_allowed = normalize_text(allowed)
        if norm_parsed == norm_allowed:
            return True
        if len(norm_parsed) >= 3 and norm_parsed in norm_allowed:
            return True
        if len(norm_allowed) >= 3 and norm_allowed in norm_parsed:
            return True

    return False


def verify_language_answer(parsed_answer: str, sample: BenchmarkSample) -> bool:
    """Default verifier (evaluates grounded extraction match)."""
    return verify_language_grounded(parsed_answer, sample)


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
