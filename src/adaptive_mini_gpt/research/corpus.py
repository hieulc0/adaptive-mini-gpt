"""M37 Phase 2: Approved Research Document Corpus.

Defines a bounded, approved document corpus across physics, biology, computing,
and history, with explicit source provenance, document reliability ratings,
and deliberate adversarial challenge types:
  - TRUSTWORTHY: Verified reference articles.
  - INCOMPLETE: Topically relevant but missing critical facts.
  - CONTRADICTORY: Outdated, disproven, or conflicting assertions.
  - DISTRACTOR: Topically adjacent keywords answering a different topic.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import asdict, dataclass
from typing import Any


class DocumentType(str, enum.Enum):
    """Classification of document truthfulness and completeness."""

    TRUSTWORTHY = "trustworthy"
    INCOMPLETE = "incomplete"
    CONTRADICTORY = "contradictory"
    DISTRACTOR = "distractor"


@dataclass
class ResearchDocument:
    """An approved document entry in the research corpus with full provenance."""

    doc_id: str
    subject: str
    title: str
    content: str
    doc_type: DocumentType
    provenance: str
    timestamp: str
    reliability_score: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["doc_type"] = self.doc_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchDocument:
        cp = dict(data)
        cp["doc_type"] = DocumentType(cp["doc_type"])
        return cls(**cp)


APPROVED_DOCUMENTS: list[ResearchDocument] = [
    # ---------------- PHYSICS ----------------
    ResearchDocument(
        doc_id="phys_01",
        subject="physics",
        title="Newton's Second Law of Motion",
        content="Newton's second law of motion states that the net force applied to an object is directly equal to the product of its mass and its acceleration, expressed as F = m * a.",
        doc_type=DocumentType.TRUSTWORTHY,
        provenance="approved_local_doc:physics_handbook_v1",
        timestamp="2026-01-15T00:00:00Z",
        reliability_score=1.0,
    ),
    ResearchDocument(
        doc_id="phys_02",
        subject="physics",
        title="First Law of Thermodynamics",
        content="The first law of thermodynamics, also known as the law of conservation of energy, states that energy cannot be created or destroyed in an isolated system, only transformed between forms.",
        doc_type=DocumentType.TRUSTWORTHY,
        provenance="approved_local_doc:physics_handbook_v1",
        timestamp="2026-01-15T00:00:00Z",
        reliability_score=1.0,
    ),
    ResearchDocument(
        doc_id="phys_03",
        subject="physics",
        title="Caloric Theory of Heat",
        content="Historical outdated claim: Heat is a weightless fluid called caloric that flows from warmer bodies to colder bodies and is indestructible.",
        doc_type=DocumentType.CONTRADICTORY,
        provenance="approved_local_doc:obsolete_theories_archive",
        timestamp="1850-01-01T00:00:00Z",
        reliability_score=0.1,
    ),
    ResearchDocument(
        doc_id="phys_04",
        subject="physics",
        title="Introduction to Electromagnetism",
        content="Electromagnetism is a branch of physics involving the study of electromagnetic force, a type of physical interaction between electrically charged particles.",
        doc_type=DocumentType.INCOMPLETE,
        provenance="approved_local_doc:intro_physics_notes",
        timestamp="2026-02-01T00:00:00Z",
        reliability_score=0.5,
    ),
    ResearchDocument(
        doc_id="phys_05",
        subject="physics",
        title="Astronomy and Celestial Comets",
        content="Astronomy observes celestial bodies including comets and icy planetesimals traveling along eccentric orbits around distant stars.",
        doc_type=DocumentType.DISTRACTOR,
        provenance="approved_local_doc:astronomy_weekly",
        timestamp="2026-02-10T00:00:00Z",
        reliability_score=0.7,
    ),

    # ---------------- BIOLOGY ----------------
    ResearchDocument(
        doc_id="bio_01",
        subject="biology",
        title="Mitochondrial ATP Synthesis",
        content="Mitochondria are membrane-bound organelles that generate the majority of cellular adenosine triphosphate (ATP) through oxidative phosphorylation in eukaryotic cells.",
        doc_type=DocumentType.TRUSTWORTHY,
        provenance="approved_local_doc:cell_biology_primer",
        timestamp="2026-01-20T00:00:00Z",
        reliability_score=1.0,
    ),
    ResearchDocument(
        doc_id="bio_02",
        subject="biology",
        title="DNA Base Pairing Rules",
        content="In the DNA double helix, nucleotides pair specifically according to Chargaff's rules: adenine pairs with thymine (A-T), and cytosine pairs with guanine (C-G).",
        doc_type=DocumentType.TRUSTWORTHY,
        provenance="approved_local_doc:genetics_fundamentals",
        timestamp="2026-01-22T00:00:00Z",
        reliability_score=1.0,
    ),
    ResearchDocument(
        doc_id="bio_03",
        subject="biology",
        title="Spontaneous Generation Hypothesis",
        content="Historical disproven claim: Maggots and complex living organisms arise spontaneously from decaying non-living flesh without microscopic parent eggs.",
        doc_type=DocumentType.CONTRADICTORY,
        provenance="approved_local_doc:obsolete_theories_archive",
        timestamp="1860-01-01T00:00:00Z",
        reliability_score=0.1,
    ),
    ResearchDocument(
        doc_id="bio_04",
        subject="biology",
        title="Photosynthetic Pigments",
        content="Photosynthesis converts solar light into chemical bond energy within autotrophic plant cells.",
        doc_type=DocumentType.INCOMPLETE,
        provenance="approved_local_doc:botany_digest",
        timestamp="2026-02-05T00:00:00Z",
        reliability_score=0.5,
    ),
    ResearchDocument(
        doc_id="bio_05",
        subject="biology",
        title="Fungal Decomposition",
        content="Fungi act as heterotrophic decomposers, releasing external digestive enzymes to recycle nutrients from fallen decaying timber.",
        doc_type=DocumentType.DISTRACTOR,
        provenance="approved_local_doc:mycology_notes",
        timestamp="2026-02-12T00:00:00Z",
        reliability_score=0.7,
    ),

    # ---------------- COMPUTING ----------------
    ResearchDocument(
        doc_id="comp_01",
        subject="computing",
        title="Binary Digit Representation",
        content="Digital computing hardware encodes all instructions and data using binary digits (bits), where each physical bit represents a discrete state of 0 or 1.",
        doc_type=DocumentType.TRUSTWORTHY,
        provenance="approved_local_doc:computer_architecture_guide",
        timestamp="2026-01-10T00:00:00Z",
        reliability_score=1.0,
    ),
    ResearchDocument(
        doc_id="comp_02",
        subject="computing",
        title="Universal Turing Machine",
        content="Alan Turing introduced the mathematical model of a Universal Turing Machine in 1936, establishing the theoretical foundation of modern programmable computation.",
        doc_type=DocumentType.TRUSTWORTHY,
        provenance="approved_local_doc:theory_of_computation",
        timestamp="2026-01-12T00:00:00Z",
        reliability_score=1.0,
    ),
    ResearchDocument(
        doc_id="comp_03",
        subject="computing",
        title="Mechanical Clockwork Processors",
        content="Fabricated false claim: Contemporary silicon microprocessor chips contain miniature brass gears and clockwork escapements to execute logic gates.",
        doc_type=DocumentType.CONTRADICTORY,
        provenance="approved_local_doc:satirical_computing_lore",
        timestamp="2025-11-01T00:00:00Z",
        reliability_score=0.1,
    ),
    ResearchDocument(
        doc_id="comp_04",
        subject="computing",
        title="Compiler Optimization",
        content="Compilers translate high-level program source code into executable binary machine instructions.",
        doc_type=DocumentType.INCOMPLETE,
        provenance="approved_local_doc:compiler_basics",
        timestamp="2026-02-08T00:00:00Z",
        reliability_score=0.5,
    ),
    ResearchDocument(
        doc_id="comp_05",
        subject="computing",
        title="Packet Switched Networking",
        content="Ethernet protocols govern local area frame transmissions across twisted pair copper cabling and optical network interfaces.",
        doc_type=DocumentType.DISTRACTOR,
        provenance="approved_local_doc:network_engineering_handbook",
        timestamp="2026-02-14T00:00:00Z",
        reliability_score=0.7,
    ),

    # ---------------- HISTORY ----------------
    ResearchDocument(
        doc_id="hist_01",
        subject="history",
        title="Gutenberg Movable Type Printing",
        content="Johannes Gutenberg developed the mechanical movable metal type printing press in Mainz, Germany around the year 1440, dramatically accelerating the spread of written literacy.",
        doc_type=DocumentType.TRUSTWORTHY,
        provenance="approved_local_doc:world_history_encyclopedia",
        timestamp="2026-01-05T00:00:00Z",
        reliability_score=1.0,
    ),
    ResearchDocument(
        doc_id="hist_02",
        subject="history",
        title="Apollo 11 Lunar Landing",
        content="NASA's Apollo 11 lunar module touched down on the Moon on July 20, 1969, making astronauts Neil Armstrong and Buzz Aldrin the first humans to walk on the lunar surface.",
        doc_type=DocumentType.TRUSTWORTHY,
        provenance="approved_local_doc:space_exploration_chronicles",
        timestamp="2026-01-08T00:00:00Z",
        reliability_score=1.0,
    ),
    ResearchDocument(
        doc_id="hist_03",
        subject="history",
        title="Medieval Flat Earth Myth",
        content="Historical misconception: Educated medieval European university scholars universally taught that the Earth was completely flat and that ships would sail off the edge of the world.",
        doc_type=DocumentType.CONTRADICTORY,
        provenance="approved_local_doc:debunked_historical_myths",
        timestamp="2025-10-15T00:00:00Z",
        reliability_score=0.1,
    ),
    ResearchDocument(
        doc_id="hist_04",
        subject="history",
        title="Steam Power and Industry",
        content="The industrial transition in eighteenth-century Britain transformed manual textile manufacture through mechanical power.",
        doc_type=DocumentType.INCOMPLETE,
        provenance="approved_local_doc:economic_history_outline",
        timestamp="2026-02-02T00:00:00Z",
        reliability_score=0.5,
    ),
    ResearchDocument(
        doc_id="hist_05",
        subject="history",
        title="Archaeological Stratigraphy",
        content="Archaeological excavators analyze sedimentary layers to establish chronological sequences of discarded ceramic pottery sherds.",
        doc_type=DocumentType.DISTRACTOR,
        provenance="approved_local_doc:archaeological_methods",
        timestamp="2026-02-18T00:00:00Z",
        reliability_score=0.7,
    ),
]


@dataclass
class ResearchTask:
    """Evaluation task testing research, query formulation, and evidence validation."""

    task_id: str
    subject: str
    question: str
    expected_answer: str
    expected_doc_id: str | None  # None for unsupported/adversarial queries
    is_unanswerable: bool
    is_contradictory_challenge: bool = False
    paraphrased_questions: list[str] | None = None


RESEARCH_BENCHMARK_TASKS: list[ResearchTask] = [
    # 1. Answerable via Trustworthy Docs
    ResearchTask(
        task_id="task_phys_newton",
        subject="physics",
        question="What equation expresses Newton's second law of motion relating force, mass, and acceleration?",
        expected_answer="F = m * a",
        expected_doc_id="phys_01",
        is_unanswerable=False,
        paraphrased_questions=[
            "How is Newton's second law mathematically represented?",
            "What is the formula connecting mass, acceleration, and net force?",
        ],
    ),
    ResearchTask(
        task_id="task_bio_atp",
        subject="biology",
        question="Which organelle generates the majority of cellular ATP in eukaryotic cells?",
        expected_answer="Mitochondria",
        expected_doc_id="bio_01",
        is_unanswerable=False,
        paraphrased_questions=[
            "What cellular structure produces ATP via oxidative phosphorylation?",
            "Where is most cellular ATP synthesized in eukaryotes?",
        ],
    ),
    ResearchTask(
        task_id="task_comp_turing",
        subject="computing",
        question="In what year did Alan Turing introduce the concept of the universal computing machine?",
        expected_answer="1936",
        expected_doc_id="comp_02",
        is_unanswerable=False,
        paraphrased_questions=[
            "When was Turing's theoretical computing machine paper published?",
            "What year marks the introduction of the Universal Turing Machine?",
        ],
    ),
    ResearchTask(
        task_id="task_hist_gutenberg",
        subject="history",
        question="Around what year did Johannes Gutenberg introduce movable metal type printing to Europe?",
        expected_answer="1440",
        expected_doc_id="hist_01",
        is_unanswerable=False,
        paraphrased_questions=[
            "When was the mechanical movable type press developed in Mainz by Gutenberg?",
            "In what approximate year was Gutenberg's movable type printing press created?",
        ],
    ),
    ResearchTask(
        task_id="task_hist_apollo",
        subject="history",
        question="What date did Apollo 11 land astronauts on the surface of the Moon?",
        expected_answer="July 20, 1969",
        expected_doc_id="hist_02",
        is_unanswerable=False,
        paraphrased_questions=[
            "When did Neil Armstrong and Buzz Aldrin touch down on the lunar surface?",
            "What was the exact date of the Apollo 11 Moon landing?",
        ],
    ),

    # 2. Adversarial Contradiction Challenges (Should detect conflict / obsolete claims)
    ResearchTask(
        task_id="task_contra_caloric",
        subject="physics",
        question="Is heat an indestructible physical fluid according to verified physics?",
        expected_answer="insufficient information.",  # Or explicit rejection
        expected_doc_id="phys_03",
        is_unanswerable=True,
        is_contradictory_challenge=True,
    ),
    ResearchTask(
        task_id="task_contra_clockwork",
        subject="computing",
        question="Are contemporary computer CPU chips powered by miniature brass clockwork gears?",
        expected_answer="insufficient information.",
        expected_doc_id="comp_03",
        is_unanswerable=True,
        is_contradictory_challenge=True,
    ),

    # 3. Completely Unsupported Questions (Testing Safe Abstention on missing evidence)
    ResearchTask(
        task_id="task_unsupported_atlantis",
        subject="history",
        question="What was the official monetary gold coin used in the sunken city of Atlantis?",
        expected_answer="insufficient information.",
        expected_doc_id=None,
        is_unanswerable=True,
    ),
    ResearchTask(
        task_id="task_unsupported_quantum_mars",
        subject="physics",
        question="What is the precise quantum gravity resonant frequency measured in the core of Mars in 1720?",
        expected_answer="insufficient information.",
        expected_doc_id=None,
        is_unanswerable=True,
    ),
]
