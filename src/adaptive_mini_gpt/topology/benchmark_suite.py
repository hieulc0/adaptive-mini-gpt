"""M38 Phase 4: Multi-Domain Learning and Retention Benchmark Suite.

Constructs strictly isolated, verifiable benchmark tasks across:
  1. Unseen multi-digit arithmetic with carry
  2. Multi-hop symbolic logic
  3. Source-grounded factual learning
  4. Paraphrased factual questions (testing relational understanding vs string memorization)
  5. Concept transfer tasks
  6. Multi-episode cross-domain factual retention matrix (Physics -> Biology -> Computing -> History)
  7. Elementary arithmetic carry retention (200 elementary decimal transitions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TopologyBenchmarkTask:
    """Standard evaluation task for structural evolution benchmarking."""

    task_id: str
    domain: str  # "arithmetic", "logic", "factual", "paraphrase", "transfer", "retention"
    subject: str  # "arithmetic", "logic", "physics", "biology", "computing", "history"
    question: str
    expected_answer: str
    distractors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# 1. Multi-Domain Learning Tasks for Phase 3 & 4
UNSEEN_ARITHMETIC_TASKS = [
    TopologyBenchmarkTask("arith_01", "arithmetic", "arithmetic", "28 + 47 =", "75"),
    TopologyBenchmarkTask("arith_02", "arithmetic", "arithmetic", "69 + 54 =", "123"),
    TopologyBenchmarkTask("arith_03", "arithmetic", "arithmetic", "88 + 79 =", "167"),
    TopologyBenchmarkTask("arith_04", "arithmetic", "arithmetic", "135 + 288 =", "423"),
]

UNSEEN_LOGIC_TASKS = [
    TopologyBenchmarkTask(
        "logic_01",
        "logic",
        "logic",
        "Premises: If Alpha then Beta. If Beta then Gamma. If Gamma then Delta. Alpha is true. What follows?",
        "Delta",
    ),
    TopologyBenchmarkTask(
        "logic_02",
        "logic",
        "logic",
        "Premises: If North then East. If East then South. North is true. What follows?",
        "South",
    ),
    TopologyBenchmarkTask(
        "logic_03",
        "logic",
        "logic",
        "Premises: If Red then Blue. If Blue then Green. If Green then Yellow. Red is true. What follows?",
        "Yellow",
    ),
]

# 2. Source-Grounded Factual Learning & Paraphrase Tasks
SOURCE_FACTUAL_TASKS = [
    # Physics
    TopologyBenchmarkTask(
        "fact_phys_newton",
        "factual",
        "physics",
        "What equation expresses Newton's second law relating force, mass, and acceleration?",
        "F = ma",
    ),
    # Biology
    TopologyBenchmarkTask(
        "fact_bio_atp",
        "factual",
        "biology",
        "Which cellular organelle synthesizes the majority of ATP in eukaryotic cells?",
        "mitochondria",
    ),
    # Computing
    TopologyBenchmarkTask(
        "fact_comp_turing",
        "factual",
        "computing",
        "In what year did Alan Turing publish his seminal paper on universal computing machines?",
        "1936",
    ),
    # History
    TopologyBenchmarkTask(
        "fact_hist_gutenberg",
        "factual",
        "history",
        "Around what year did Johannes Gutenberg introduce movable metal type printing to Europe?",
        "1440",
    ),
]

PARAPHRASED_FACTUAL_TASKS = [
    # Physics Paraphrase
    TopologyBenchmarkTask(
        "para_phys_newton",
        "paraphrase",
        "physics",
        "According to classical mechanics, force equals the product of mass and which quantity?",
        "acceleration",
    ),
    # Biology Paraphrase
    TopologyBenchmarkTask(
        "para_bio_atp",
        "paraphrase",
        "biology",
        "What energy-carrying molecule is primarily generated within mitochondria via oxidative phosphorylation?",
        "ATP",
    ),
    # Computing Paraphrase
    TopologyBenchmarkTask(
        "para_comp_turing",
        "paraphrase",
        "computing",
        "Which British mathematician formalized the foundational mathematical model of computation in 1936?",
        "Alan Turing",
    ),
    # History Paraphrase
    TopologyBenchmarkTask(
        "para_hist_gutenberg",
        "paraphrase",
        "history",
        "In the 1440s, Johannes Gutenberg transformed European literacy by inventing what technology?",
        "movable type printing",
    ),
]

CONCEPT_TRANSFER_TASKS = [
    TopologyBenchmarkTask(
        "trans_phys_01",
        "transfer",
        "physics",
        "If an object of mass 5 kg experiences an acceleration of 4 m/s^2, what is the net force in Newtons?",
        "20",
    ),
    TopologyBenchmarkTask(
        "trans_bio_01",
        "transfer",
        "biology",
        "If a cell is depleted of functional mitochondria, which cellular currency cannot be produced via aerobic respiration?",
        "ATP",
    ),
]

# 3. Multi-Episode Factual Retention Matrix Specification
# Defines 4 distinct sequential learning episodes:
SEQUENTIAL_EPISODES = [
    {
        "episode_id": "ep1_physics",
        "subject": "physics",
        "train_pairs": [
            ("What equation expresses Newton's second law relating force, mass, and acceleration?", "F = ma"),
            ("In classical mechanics, force is mass times acceleration.", "F = ma"),
        ],
        "test_task": SOURCE_FACTUAL_TASKS[0],
        "para_task": PARAPHRASED_FACTUAL_TASKS[0],
    },
    {
        "episode_id": "ep2_biology",
        "subject": "biology",
        "train_pairs": [
            ("Which cellular organelle synthesizes the majority of ATP in eukaryotic cells?", "mitochondria"),
            ("Mitochondria generate cellular energy in the form of ATP.", "mitochondria"),
        ],
        "test_task": SOURCE_FACTUAL_TASKS[1],
        "para_task": PARAPHRASED_FACTUAL_TASKS[1],
    },
    {
        "episode_id": "ep3_computing",
        "subject": "computing",
        "train_pairs": [
            ("In what year did Alan Turing publish his seminal paper on universal computing machines?", "1936"),
            ("Alan Turing introduced the universal computing machine in 1936.", "1936"),
        ],
        "test_task": SOURCE_FACTUAL_TASKS[2],
        "para_task": PARAPHRASED_FACTUAL_TASKS[2],
    },
    {
        "episode_id": "ep4_history",
        "subject": "history",
        "train_pairs": [
            ("Around what year did Johannes Gutenberg introduce movable metal type printing to Europe?", "1440"),
            ("Johannes Gutenberg introduced movable metal type printing in Europe around 1440.", "1440"),
        ],
        "test_task": SOURCE_FACTUAL_TASKS[3],
        "para_task": PARAPHRASED_FACTUAL_TASKS[3],
    },
]


def generate_carry_retention_suite() -> list[tuple[int, int, int]]:
    """Generate all 200 elementary decimal addition transitions (a in 0..9, b in 0..9, carry_in in 0..1)."""
    transitions = []
    for a in range(10):
        for b in range(10):
            for c_in in (0, 1):
                transitions.append((a, b, c_in))
    return transitions
