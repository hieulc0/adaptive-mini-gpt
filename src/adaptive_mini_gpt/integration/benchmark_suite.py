"""M39 Phase 2: Factual Generalization and Multi-Domain Benchmark Suite.

Provides structured, multi-view evaluation tasks designed to test:
  1. Direct factual recall (training relation matching).
  2. Paraphrased generalization (semantically equivalent, syntactically varied queries).
  3. Relational & compositional deduction (two-step concept linking).
  4. Withheld entities and unseen relations (out-of-distribution factual generalization).
  5. Unsupported / adversarial queries requiring abstention.
  6. Cumulative multi-subject sequential episodes (Physics -> Biology -> Computing -> History).
  7. Procedural capabilities: elementary arithmetic carry transitions & multi-hop logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class IntegrationTask:
    """A structured benchmark task for knowledge integration and generalization."""

    task_id: str
    subject: str
    task_type: str  # "direct", "paraphrase", "compositional", "unseen", "abstention", "arithmetic", "logic"
    question: str
    expected_answer: str
    alternate_phrasings: list[str] = field(default_factory=list)
    is_unanswerable: bool = False
    relationship_triplet: tuple[str, str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# 1. Multi-View Factual Learning Benchmark
# -----------------------------------------------------------------------------

FACTUAL_LEARNING_DATASET: list[IntegrationTask] = [
    # --- Subject: Physics ---
    IntegrationTask(
        task_id="phys_newton_direct",
        subject="physics",
        task_type="direct",
        question="What equation expresses Newton's second law relating force, mass, and acceleration?",
        expected_answer="F = ma",
        alternate_phrasings=[
            "In Newtonian mechanics, what is the formula connecting force to mass and acceleration?",
            "State Newton's second law in mathematical form.",
            "How is net physical force expressed algebraically?",
        ],
        relationship_triplet=("Newton's second law", "mathematical_expression", "F = ma"),
    ),
    IntegrationTask(
        task_id="phys_newton_paraphrase",
        subject="physics",
        task_type="paraphrase",
        question="According to classical mechanics, what formula relates force to acceleration?",
        expected_answer="F = ma",
        alternate_phrasings=[
            "What mathematical law defines force as the product of mass and acceleration?",
        ],
        relationship_triplet=("Newton's second law", "mathematical_expression", "F = ma"),
    ),
    IntegrationTask(
        task_id="phys_newton_compositional",
        subject="physics",
        task_type="compositional",
        question="If a constant force causes an object of mass m to undergo acceleration a, what equation governs this motion?",
        expected_answer="F = ma",
        relationship_triplet=("Newtonian dynamics", "acceleration_governance", "F = ma"),
    ),
    IntegrationTask(
        task_id="phys_kepler_unseen",
        subject="physics",
        task_type="unseen",
        question="What shape describes planetary orbits according to Kepler's first law?",
        expected_answer="ellipse",
        relationship_triplet=("planetary orbit", "geometric_shape", "ellipse"),
    ),

    # --- Subject: Biology ---
    IntegrationTask(
        task_id="bio_mitochondria_direct",
        subject="biology",
        task_type="direct",
        question="Which cellular organelle synthesizes the majority of ATP through oxidative phosphorylation?",
        expected_answer="mitochondria",
        alternate_phrasings=[
            "What cellular structure generates most eukaryotic chemical energy as ATP?",
            "Identify the organelle responsible for aerobic cellular respiration.",
            "Where in eukaryotic cells is oxidative phosphorylation conducted?",
        ],
        relationship_triplet=("mitochondria", "primary_function", "ATP synthesis"),
    ),
    IntegrationTask(
        task_id="bio_mitochondria_paraphrase",
        subject="biology",
        task_type="paraphrase",
        question="What is the primary site of cellular ATP production in eukaryotic organisms?",
        expected_answer="mitochondria",
        alternate_phrasings=[
            "Which organelle acts as the power-generating unit of eukaryotic cells?",
        ],
        relationship_triplet=("mitochondria", "cellular_role", "ATP production"),
    ),
    IntegrationTask(
        task_id="bio_mitochondria_compositional",
        subject="biology",
        task_type="compositional",
        question="If cyanide blocks the electron transport chain, which organelle's ATP production is halted?",
        expected_answer="mitochondria",
        relationship_triplet=("mitochondria", "contains", "electron transport chain"),
    ),
    IntegrationTask(
        task_id="bio_chloroplast_unseen",
        subject="biology",
        task_type="unseen",
        question="Which plant organelle converts light energy into chemical sugars via photosynthesis?",
        expected_answer="chloroplast",
        relationship_triplet=("chloroplast", "primary_function", "photosynthesis"),
    ),

    # --- Subject: Computing ---
    IntegrationTask(
        task_id="comp_turing_direct",
        subject="computing",
        task_type="direct",
        question="In what year did Alan Turing publish his landmark paper introducing the universal computing machine?",
        expected_answer="1936",
        alternate_phrasings=[
            "When did Alan Turing formulate the theoretical concept of the Turing machine?",
            "In which year was Turing's seminal paper on computable numbers published?",
            "What publication year marked the theoretical definition of universal computation by Turing?",
        ],
        relationship_triplet=("Alan Turing", "paper_publication_year", "1936"),
    ),
    IntegrationTask(
        task_id="comp_turing_paraphrase",
        subject="computing",
        task_type="paraphrase",
        question="In what year was the mathematical model of the universal computing machine first published?",
        expected_answer="1936",
        alternate_phrasings=[
            "What year saw the publication of Alan Turing's 'On Computable Numbers'?",
        ],
        relationship_triplet=("universal computing machine", "first_published", "1936"),
    ),
    IntegrationTask(
        task_id="comp_turing_compositional",
        subject="computing",
        task_type="compositional",
        question="Before modern stored-program computers existed, in what pre-WWII year did Turing prove undecidability?",
        expected_answer="1936",
        relationship_triplet=("Entscheidungsproblem", "resolved_by_turing", "1936"),
    ),
    IntegrationTask(
        task_id="comp_shannon_unseen",
        subject="computing",
        task_type="unseen",
        question="In what year did Claude Shannon establish mathematical information theory in his Bell System technical paper?",
        expected_answer="1948",
        relationship_triplet=("Claude Shannon", "information_theory_year", "1948"),
    ),

    # --- Subject: History ---
    IntegrationTask(
        task_id="hist_gutenberg_direct",
        subject="history",
        task_type="direct",
        question="Around what year did Johannes Gutenberg introduce movable metal type printing to Mainz, Germany?",
        expected_answer="1440",
        alternate_phrasings=[
            "In approximately what year did Gutenberg develop his movable type printing press in Europe?",
            "What year is traditionally recognized for the invention of the European movable type press by Gutenberg?",
            "Around which year was movable metal type pioneered by Gutenberg?",
        ],
        relationship_triplet=("Johannes Gutenberg", "movable_type_invention_year", "1440"),
    ),
    IntegrationTask(
        task_id="hist_gutenberg_paraphrase",
        subject="history",
        task_type="paraphrase",
        question="In approximately what historical year did movable metal type printing emerge in Europe?",
        expected_answer="1440",
        alternate_phrasings=[
            "What year marks the introduction of the Gutenberg printing press in Germany?",
        ],
        relationship_triplet=("movable type printing", "european_invention_year", "1440"),
    ),
    IntegrationTask(
        task_id="hist_gutenberg_compositional",
        subject="history",
        task_type="compositional",
        question="Fifteen years before printing the 42-Line Bible in 1455, around what year did Gutenberg pioneer movable type?",
        expected_answer="1440",
        relationship_triplet=("Gutenberg printing", "technology_origination_year", "1440"),
    ),
    IntegrationTask(
        task_id="hist_apollo_unseen",
        subject="history",
        task_type="unseen",
        question="In what year did the Apollo 11 lunar module land the first humans on the Moon?",
        expected_answer="1969",
        relationship_triplet=("Apollo 11", "moon_landing_year", "1969"),
    ),

    # --- Adverse / Unsupported Abstention Tasks ---
    IntegrationTask(
        task_id="adverse_alchemy",
        subject="physics",
        task_type="abstention",
        question="What chemical catalyst allows lead to be transmuted into solid gold through ordinary heating?",
        expected_answer="abstain",
        is_unanswerable=True,
    ),
    IntegrationTask(
        task_id="adverse_telepathy",
        subject="biology",
        task_type="abstention",
        question="Which human brain lobe transmits wireless radio telepathy signals during REM sleep?",
        expected_answer="abstain",
        is_unanswerable=True,
    ),
    IntegrationTask(
        task_id="adverse_vacuum_pc",
        subject="computing",
        task_type="abstention",
        question="What 18th-century microprocessor ran Windows 95 in Victorian London?",
        expected_answer="abstain",
        is_unanswerable=True,
    ),
    IntegrationTask(
        task_id="adverse_atlantis",
        subject="history",
        task_type="abstention",
        question="What treaty established the official trade embassy between the Roman Empire and Atlantis in 50 BC?",
        expected_answer="abstain",
        is_unanswerable=True,
    ),
]


# -----------------------------------------------------------------------------
# 2. Sequential Research Episodes Curriculum (4 Subjects)
# -----------------------------------------------------------------------------

SEQUENTIAL_INTEGRATION_EPISODES: list[dict[str, Any]] = [
    {
        "subject": "physics",
        "description": "Newtonian Mechanics & Force Law",
        "train_pairs": [
            ("What equation expresses Newton's second law relating force, mass, and acceleration?", "F = ma"),
            ("In Newtonian mechanics, what is the formula connecting force to mass and acceleration?", "F = ma"),
            ("State Newton's second law in mathematical form.", "F = ma"),
            ("How is net physical force expressed algebraically?", "F = ma"),
        ],
        "eval_tasks": [
            t for t in FACTUAL_LEARNING_DATASET if t.subject == "physics"
        ],
    },
    {
        "subject": "biology",
        "description": "Cellular Energy & Mitochondria",
        "train_pairs": [
            ("Which cellular organelle synthesizes the majority of ATP through oxidative phosphorylation?", "mitochondria"),
            ("What cellular structure generates most eukaryotic chemical energy as ATP?", "mitochondria"),
            ("Identify the organelle responsible for aerobic cellular respiration.", "mitochondria"),
            ("Where in eukaryotic cells is oxidative phosphorylation conducted?", "mitochondria"),
        ],
        "eval_tasks": [
            t for t in FACTUAL_LEARNING_DATASET if t.subject == "biology"
        ],
    },
    {
        "subject": "computing",
        "description": "Universal Computation & Alan Turing",
        "train_pairs": [
            ("In what year did Alan Turing publish his landmark paper introducing the universal computing machine?", "1936"),
            ("When did Alan Turing formulate the theoretical concept of the Turing machine?", "1936"),
            ("In which year was Turing's seminal paper on computable numbers published?", "1936"),
            ("What publication year marked the theoretical definition of universal computation by Turing?", "1936"),
        ],
        "eval_tasks": [
            t for t in FACTUAL_LEARNING_DATASET if t.subject == "computing"
        ],
    },
    {
        "subject": "history",
        "description": "Information Diffusion & Movable Type Printing",
        "train_pairs": [
            ("Around what year did Johannes Gutenberg introduce movable metal type printing to Mainz, Germany?", "1440"),
            ("In approximately what year did Gutenberg develop his movable type printing press in Europe?", "1440"),
            ("What year is traditionally recognized for the invention of the European movable type press by Gutenberg?", "1440"),
            ("Around which year was movable metal type pioneered by Gutenberg?", "1440"),
        ],
        "eval_tasks": [
            t for t in FACTUAL_LEARNING_DATASET if t.subject == "history"
        ],
    },
]


# -----------------------------------------------------------------------------
# 3. Procedural Retention Benchmark (Arithmetic & Logic)
# -----------------------------------------------------------------------------

PROCEDURAL_LOGIC_TASKS: list[IntegrationTask] = [
    IntegrationTask(
        task_id="logic_modus_ponens_1",
        subject="logic",
        task_type="logic",
        question="If all humans are mortal and Socrates is human, what follows about Socrates?",
        expected_answer="mortal",
    ),
    IntegrationTask(
        task_id="logic_chain_deduction_2",
        subject="logic",
        task_type="logic",
        question="Rule 1: If condition A then condition B. Rule 2: If condition B then condition C. Given A is true, what is C?",
        expected_answer="true",
    ),
    IntegrationTask(
        task_id="logic_disjunctive_syllogism_3",
        subject="logic",
        task_type="logic",
        question="Either system alpha is online or system beta is online. System alpha is offline. Which system is online?",
        expected_answer="beta",
    ),
    IntegrationTask(
        task_id="logic_contrapositive_4",
        subject="logic",
        task_type="logic",
        question="Rule: If it rains, the grass is wet. The grass is dry. Did it rain?",
        expected_answer="no",
    ),
]
