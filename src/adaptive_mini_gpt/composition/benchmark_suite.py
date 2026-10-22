"""M40 Compositional Knowledge, Multi-Hop Reasoning & Transfer Benchmark Suite.

Defines:
  1. Relational tasks across hops (1-hop, 2-hop, 3-hop).
  2. Asymmetric and non-transitive deduction checks.
  3. Counterfactual and unfamiliar entity transfer tasks.
  4. Distractor resilience and safe abstention checks.
  5. Cross-module synthesis tasks (Relational Structure + Arithmetic / Logic).
  6. Sequential 4-episode relational curriculum for cumulative retention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CompositionalTask:
    """A structured benchmark task for evaluating relational composition."""

    task_id: str
    category: str  # "single_hop", "two_hop", "three_hop", "asymmetric", "counterfactual", "abstention", "cross_module"
    premises: list[str]
    question: str
    expected_answer: str
    intermediate_reasoning: str
    hops: int = 1
    is_counterfactual: bool = False
    entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# 1. Core Compositional Benchmark Suite (24 Balanced Tasks)
# -----------------------------------------------------------------------------

M40_COMPOSITIONAL_BENCHMARK: list[CompositionalTask] = [
    # --- 1-Hop Direct Relational Tasks (4 tasks) ---
    CompositionalTask(
        task_id="m40_1hop_containment_01",
        category="single_hop",
        premises=["Zorblax is inside Nexus-4."],
        question="Is Zorblax inside Nexus-4?",
        expected_answer="yes",
        intermediate_reasoning="Direct premise states Zorblax is inside Nexus-4.",
        hops=1,
        entities=["Zorblax", "Nexus-4"],
    ),
    CompositionalTask(
        task_id="m40_1hop_containment_02",
        category="single_hop",
        premises=["Vortice is inside Celestia."],
        question="Is Vortice inside Celestia?",
        expected_answer="yes",
        intermediate_reasoning="Direct premise states Vortice is inside Celestia.",
        hops=1,
        entities=["Vortice", "Celestia"],
    ),
    CompositionalTask(
        task_id="m40_1hop_precedence_01",
        category="single_hop",
        premises=["Krypton-9 precedes Epsilon-3."],
        question="Does Krypton-9 precede Epsilon-3?",
        expected_answer="yes",
        intermediate_reasoning="Direct premise states Krypton-9 precedes Epsilon-3.",
        hops=1,
        entities=["Krypton-9", "Epsilon-3"],
    ),
    CompositionalTask(
        task_id="m40_1hop_precedence_02",
        category="single_hop",
        premises=["Quasar-5 precedes Nebula-8."],
        question="Does Quasar-5 precede Nebula-8?",
        expected_answer="yes",
        intermediate_reasoning="Direct premise states Quasar-5 precedes Nebula-8.",
        hops=1,
        entities=["Quasar-5", "Nebula-8"],
    ),

    # --- 2-Hop Transitive Deductions (4 tasks) ---
    CompositionalTask(
        task_id="m40_2hop_deduction_01",
        category="two_hop",
        premises=["Zorblax is inside Nexus-4.", "Nexus-4 is inside Vortice."],
        question="Is Zorblax inside Vortice?",
        expected_answer="yes",
        intermediate_reasoning="Step 1: Zorblax is inside Nexus-4. Step 2: Nexus-4 is inside Vortice. Therefore by transitivity Zorblax is inside Vortice.",
        hops=2,
        entities=["Zorblax", "Nexus-4", "Vortice"],
    ),
    CompositionalTask(
        task_id="m40_2hop_deduction_02",
        category="two_hop",
        premises=["Krypton-9 precedes Epsilon-3.", "Epsilon-3 precedes Pyron-1."],
        question="Does Krypton-9 precede Pyron-1?",
        expected_answer="yes",
        intermediate_reasoning="Step 1: Krypton-9 precedes Epsilon-3. Step 2: Epsilon-3 precedes Pyron-1. Therefore by transitivity Krypton-9 precedes Pyron-1.",
        hops=2,
        entities=["Krypton-9", "Epsilon-3", "Pyron-1"],
    ),
    CompositionalTask(
        task_id="m40_2hop_negative_01",
        category="two_hop",
        premises=["Zorblax is inside Nexus-4.", "Celestia is inside Pyron-1."],
        question="Is Zorblax inside Pyron-1?",
        expected_answer="no",
        intermediate_reasoning="No transitive chain connects Nexus-4 to Celestia or Pyron-1.",
        hops=2,
        entities=["Zorblax", "Nexus-4", "Celestia", "Pyron-1"],
    ),
    CompositionalTask(
        task_id="m40_2hop_distractor_01",
        category="two_hop",
        premises=["Astraea is inside Solon-X.", "Thalos is inside Quasar-5.", "Solon-X is inside Nebula-8."],
        question="Is Astraea inside Nebula-8?",
        expected_answer="yes",
        intermediate_reasoning="Ignore distractor Thalos. Astraea is inside Solon-X and Solon-X is inside Nebula-8, therefore Astraea is inside Nebula-8.",
        hops=2,
        entities=["Astraea", "Solon-X", "Nebula-8", "Thalos", "Quasar-5"],
    ),

    # --- 3-Hop Transitive Deductions (4 tasks) ---
    CompositionalTask(
        task_id="m40_3hop_deduction_01",
        category="three_hop",
        premises=[
            "Zorblax is inside Nexus-4.",
            "Nexus-4 is inside Vortice.",
            "Vortice is inside Celestia.",
        ],
        question="Is Zorblax inside Celestia?",
        expected_answer="yes",
        intermediate_reasoning="Step 1: Zorblax is in Nexus-4. Step 2: Nexus-4 is in Vortice. Step 3: Vortice is in Celestia. By 3-hop transitivity Zorblax is in Celestia.",
        hops=3,
        entities=["Zorblax", "Nexus-4", "Vortice", "Celestia"],
    ),
    CompositionalTask(
        task_id="m40_3hop_deduction_02",
        category="three_hop",
        premises=[
            "Krypton-9 precedes Epsilon-3.",
            "Epsilon-3 precedes Pyron-1.",
            "Pyron-1 precedes Solon-X.",
        ],
        question="Does Krypton-9 precede Solon-X?",
        expected_answer="yes",
        intermediate_reasoning="Step 1: Krypton-9 precedes Epsilon-3. Step 2: Epsilon-3 precedes Pyron-1. Step 3: Pyron-1 precedes Solon-X. By 3-hop transitivity Krypton-9 precedes Solon-X.",
        hops=3,
        entities=["Krypton-9", "Epsilon-3", "Pyron-1", "Solon-X"],
    ),
    CompositionalTask(
        task_id="m40_3hop_negative_01",
        category="three_hop",
        premises=[
            "Zorblax is inside Nexus-4.",
            "Nexus-4 is inside Vortice.",
            "Solon-X is inside Celestia.",
        ],
        question="Is Zorblax inside Celestia?",
        expected_answer="no",
        intermediate_reasoning="Chain breaks between Vortice and Solon-X. No path connects Zorblax to Celestia.",
        hops=3,
        entities=["Zorblax", "Nexus-4", "Vortice", "Solon-X", "Celestia"],
    ),
    CompositionalTask(
        task_id="m40_3hop_distractor_02",
        category="three_hop",
        premises=[
            "Astraea precedes Thalos.",
            "Krypton-9 is inside Pyron-1.",
            "Thalos precedes Quasar-5.",
            "Quasar-5 precedes Nebula-8.",
        ],
        question="Does Astraea precede Nebula-8?",
        expected_answer="yes",
        intermediate_reasoning="Ignore containment distractor. Chain: Astraea -> Thalos -> Quasar-5 -> Nebula-8. Therefore Astraea precedes Nebula-8.",
        hops=3,
        entities=["Astraea", "Thalos", "Quasar-5", "Nebula-8"],
    ),

    # --- Asymmetric and Non-Transitive Checks (4 tasks) ---
    CompositionalTask(
        task_id="m40_asymmetric_reversal_01",
        category="asymmetric",
        premises=["Zorblax is inside Nexus-4."],
        question="Is Nexus-4 inside Zorblax?",
        expected_answer="no",
        intermediate_reasoning="Containment is strictly asymmetric. If A is inside B, B cannot be inside A.",
        hops=1,
        entities=["Zorblax", "Nexus-4"],
    ),
    CompositionalTask(
        task_id="m40_asymmetric_reversal_02",
        category="asymmetric",
        premises=["Krypton-9 precedes Epsilon-3."],
        question="Does Epsilon-3 precede Krypton-9?",
        expected_answer="no",
        intermediate_reasoning="Temporal/ordinal precedence is asymmetric. If A precedes B, B does not precede A.",
        hops=1,
        entities=["Krypton-9", "Epsilon-3"],
    ),
    CompositionalTask(
        task_id="m40_nontransitive_link_01",
        category="asymmetric",
        premises=["Vortice is connected_to Celestia.", "Celestia is connected_to Pyron-1."],
        question="Is Vortice connected_to Pyron-1?",
        expected_answer="no",
        intermediate_reasoning="Local connection is non-transitive. An intermediate link does not entail transitive connection.",
        hops=2,
        entities=["Vortice", "Celestia", "Pyron-1"],
    ),
    CompositionalTask(
        task_id="m40_nontransitive_link_02",
        category="asymmetric",
        premises=["Solon-X powers Thalos.", "Thalos powers Astraea."],
        question="Does Solon-X power Astraea?",
        expected_answer="no",
        intermediate_reasoning="Power generation is direct asymmetric non-transitive. Direct link required.",
        hops=2,
        entities=["Solon-X", "Thalos", "Astraea"],
    ),

    # --- Counterfactual & Unfamiliar Entity Transfer (4 tasks) ---
    CompositionalTask(
        task_id="m40_counterfactual_01",
        category="counterfactual",
        premises=["Glapton is inside Xyphos-7.", "Xyphos-7 is inside Morvan-Prime."],
        question="Is Glapton inside Morvan-Prime?",
        expected_answer="yes",
        intermediate_reasoning="Unseen entities Glapton, Xyphos-7, Morvan-Prime follow standard transitive deduction rule.",
        hops=2,
        is_counterfactual=True,
        entities=["Glapton", "Xyphos-7", "Morvan-Prime"],
    ),
    CompositionalTask(
        task_id="m40_counterfactual_02",
        category="counterfactual",
        premises=["Zephyr-12 precedes Oryx-9.", "Oryx-9 precedes Bravon-3."],
        question="Does Zephyr-12 precede Bravon-3?",
        expected_answer="yes",
        intermediate_reasoning="Unseen entities Zephyr-12, Oryx-9, Bravon-3 satisfy 2-hop precedence transitivity.",
        hops=2,
        is_counterfactual=True,
        entities=["Zephyr-12", "Oryx-9", "Bravon-3"],
    ),
    CompositionalTask(
        task_id="m40_counterfactual_03",
        category="counterfactual",
        premises=["Glapton is inside Xyphos-7."],
        question="Is Xyphos-7 inside Glapton?",
        expected_answer="no",
        intermediate_reasoning="Counterfactual entity pair obeys universal asymmetry rule: reverse containment is false.",
        hops=1,
        is_counterfactual=True,
        entities=["Glapton", "Xyphos-7"],
    ),
    CompositionalTask(
        task_id="m40_counterfactual_04",
        category="counterfactual",
        premises=[
            "Glapton is inside Xyphos-7.",
            "Xyphos-7 is inside Morvan-Prime.",
            "Morvan-Prime is inside Bravon-3.",
        ],
        question="Is Glapton inside Bravon-3?",
        expected_answer="yes",
        intermediate_reasoning="3-hop transitive deduction holds for completely unseen synthetic entities.",
        hops=3,
        is_counterfactual=True,
        entities=["Glapton", "Xyphos-7", "Morvan-Prime", "Bravon-3"],
    ),

    # --- Abstention & Contradictory Premise Checks (4 tasks) ---
    CompositionalTask(
        task_id="m40_abstention_ungrounded_01",
        category="abstention",
        premises=["Zorblax is inside Nexus-4."],
        question="Is Zorblax inside UnknownSector-99?",
        expected_answer="abstain",
        intermediate_reasoning="UnknownSector-99 is absent from the premises. Safely abstain from guessing.",
        hops=0,
        entities=["Zorblax", "UnknownSector-99"],
    ),
    CompositionalTask(
        task_id="m40_abstention_contradictory_02",
        category="abstention",
        premises=["Krypton-9 precedes Epsilon-3.", "Epsilon-3 precedes Krypton-9."],
        question="Does Krypton-9 precede Epsilon-3?",
        expected_answer="abstain",
        intermediate_reasoning="Premises contain an irreconcilable cyclic contradiction. Safely abstain.",
        hops=0,
        entities=["Krypton-9", "Epsilon-3"],
    ),
    CompositionalTask(
        task_id="m40_abstention_insufficient_03",
        category="abstention",
        premises=["Pyron-1 exists in sector gamma.", "Celestia exists in sector delta."],
        question="Is Pyron-1 inside Celestia?",
        expected_answer="abstain",
        intermediate_reasoning="Premises provide coordinate existence without topological containment. Abstain.",
        hops=0,
        entities=["Pyron-1", "Celestia"],
    ),
    CompositionalTask(
        task_id="m40_abstention_unsupported_04",
        category="abstention",
        premises=["Nexus-4 radiates blue energy."],
        question="Does Nexus-4 generate gravitation?",
        expected_answer="abstain",
        intermediate_reasoning="Ungrounded physical property query. Safely abstain.",
        hops=0,
        entities=["Nexus-4"],
    ),
]


# -----------------------------------------------------------------------------
# 2. Cross-Module Synthesis Benchmark (Relational + Arithmetic / Logic)
# -----------------------------------------------------------------------------

M40_CROSS_MODULE_BENCHMARK: list[CompositionalTask] = [
    CompositionalTask(
        task_id="m40_cross_relational_arithmetic_01",
        category="cross_module",
        premises=["Container Alpha has 3 units.", "Container Alpha is inside Warehouse Beta which adds 4 units."],
        question="How many total units are accounted for in Warehouse Beta?",
        expected_answer="7",
        intermediate_reasoning="Relational containment combines with addition: 3 + 4 = 7 units.",
        hops=2,
    ),
    CompositionalTask(
        task_id="m40_cross_relational_arithmetic_02",
        category="cross_module",
        premises=["Sector X has 5 nodes.", "Each node holds 2 cores."],
        question="How many total cores are in Sector X?",
        expected_answer="10",
        intermediate_reasoning="Relational multiplication: 5 nodes * 2 cores = 10 cores.",
        hops=2,
    ),
    CompositionalTask(
        task_id="m40_cross_relational_logic_01",
        category="cross_module",
        premises=["Rule: If A is inside B, then A is protected.", "Zorblax is inside Nexus-4."],
        question="Is Zorblax protected?",
        expected_answer="yes",
        intermediate_reasoning="Relational premise triggers propositional implication rule: Zorblax is protected.",
        hops=2,
    ),
    CompositionalTask(
        task_id="m40_cross_relational_logic_02",
        category="cross_module",
        premises=["Rule: If node is powered, it can transmit.", "Solon-X powers Thalos."],
        question="Can Thalos transmit?",
        expected_answer="yes",
        intermediate_reasoning="Directed power relation fulfills implication condition: Thalos can transmit.",
        hops=2,
    ),
]


# -----------------------------------------------------------------------------
# 3. Sequential 4-Episode Relational Curriculum
# -----------------------------------------------------------------------------

SEQUENTIAL_RELATIONAL_EPISODES: list[dict[str, Any]] = [
    {
        "episode_id": "ep1_cosmic_sectors",
        "subject": "cosmic_sectors",
        "train_premises": [
            ("Zorblax is inside Nexus-4.", "Nexus-4 is inside Vortice."),
            ("Is Zorblax inside Vortice?", "yes"),
            ("Does Nexus-4 contain Zorblax?", "yes"),
        ],
        "eval_tasks": [
            task for task in M40_COMPOSITIONAL_BENCHMARK if "Zorblax" in task.entities or "Nexus-4" in task.entities
        ],
    },
    {
        "episode_id": "ep2_biosphere_nodes",
        "subject": "biosphere_nodes",
        "train_premises": [
            ("Krypton-9 precedes Epsilon-3.", "Epsilon-3 precedes Pyron-1."),
            ("Does Krypton-9 precede Pyron-1?", "yes"),
            ("Does Epsilon-3 precede Krypton-9?", "no"),
        ],
        "eval_tasks": [
            task for task in M40_COMPOSITIONAL_BENCHMARK if "Krypton-9" in task.entities or "Epsilon-3" in task.entities
        ],
    },
    {
        "episode_id": "ep3_compute_clusters",
        "subject": "compute_clusters",
        "train_premises": [
            ("Vortice is inside Celestia.", "Celestia is inside Nebula-8."),
            ("Is Vortice inside Nebula-8?", "yes"),
            ("Is Vortice connected_to Pyron-1?", "no"),
        ],
        "eval_tasks": [
            task for task in M40_COMPOSITIONAL_BENCHMARK if "Celestia" in task.entities or "Nebula-8" in task.entities
        ],
    },
    {
        "episode_id": "ep4_historical_dynasties",
        "subject": "historical_dynasties",
        "train_premises": [
            ("Astraea precedes Thalos.", "Thalos precedes Quasar-5."),
            ("Does Astraea precede Quasar-5?", "yes"),
            ("Does Thalos precede Astraea?", "no"),
        ],
        "eval_tasks": [
            task for task in M40_COMPOSITIONAL_BENCHMARK if "Astraea" in task.entities or "Thalos" in task.entities
        ],
    },
]
