"""M41 Directional Reasoning, Negation, and Uncertainty Benchmark Suite.

Provides an expanded 40-task benchmark testing:
  1. Direct single-hop relational recall (YES).
  2. 2-hop transitive deduction (YES).
  3. 3-hop transitive deduction (YES).
  4. Asymmetric directional reversal (NO - provably false under asymmetry).
  5. Non-transitive negative deduction (NO - provably invalid under non-transitivity).
  6. Epistemic boundary: ungrounded missing entities (UNKNOWN).
  7. Epistemic boundary: contradictory cycles (UNKNOWN).
  8. Zero-shot counterfactual transfer across disjoint invented entities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DirectionalTask:
    """A structured benchmark task with tri-state truth value (YES, NO, UNKNOWN)."""

    task_id: str
    category: str  # "single_hop", "two_hop", "three_hop", "asymmetric", "nontransitive", "ungrounded", "contradictory", "counterfactual"
    premises: list[str]
    question: str
    expected_verdict: str  # "YES", "NO", "UNKNOWN"
    explanation: str
    entities: list[str] = field(default_factory=list)
    is_counterfactual: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


M41_DIRECTIONAL_BENCHMARK: list[DirectionalTask] = [
    # --- Category 1: Single-Hop Direct (5 tasks -> YES) ---
    DirectionalTask(
        task_id="m41_1hop_01",
        category="single_hop",
        premises=["Zorblax is inside Nexus-4."],
        question="Is Zorblax inside Nexus-4?",
        expected_verdict="YES",
        explanation="Direct premise confirms Zorblax is inside Nexus-4.",
        entities=["Zorblax", "Nexus-4"],
    ),
    DirectionalTask(
        task_id="m41_1hop_02",
        category="single_hop",
        premises=["Vortice is inside Celestia."],
        question="Is Vortice inside Celestia?",
        expected_verdict="YES",
        explanation="Direct premise confirms Vortice is inside Celestia.",
        entities=["Vortice", "Celestia"],
    ),
    DirectionalTask(
        task_id="m41_1hop_03",
        category="single_hop",
        premises=["Krypton-9 precedes Epsilon-3."],
        question="Does Krypton-9 precede Epsilon-3?",
        expected_verdict="YES",
        explanation="Direct premise confirms Krypton-9 precedes Epsilon-3.",
        entities=["Krypton-9", "Epsilon-3"],
    ),
    DirectionalTask(
        task_id="m41_1hop_04",
        category="single_hop",
        premises=["Solon-X powers Thalos."],
        question="Does Solon-X power Thalos?",
        expected_verdict="YES",
        explanation="Direct premise confirms Solon-X powers Thalos.",
        entities=["Solon-X", "Thalos"],
    ),
    DirectionalTask(
        task_id="m41_1hop_05",
        category="single_hop",
        premises=["Astraea subsumes Nebula-8."],
        question="Does Astraea subsume Nebula-8?",
        expected_verdict="YES",
        explanation="Direct premise confirms Astraea subsumes Nebula-8.",
        entities=["Astraea", "Nebula-8"],
    ),

    # --- Category 2: 2-Hop Transitive Deduction (5 tasks -> YES) ---
    DirectionalTask(
        task_id="m41_2hop_01",
        category="two_hop",
        premises=["Zorblax is inside Nexus-4.", "Nexus-4 is inside Vortice."],
        question="Is Zorblax inside Vortice?",
        expected_verdict="YES",
        explanation="Transitive deduction: Zorblax -> Nexus-4 -> Vortice entails Zorblax is inside Vortice.",
        entities=["Zorblax", "Nexus-4", "Vortice"],
    ),
    DirectionalTask(
        task_id="m41_2hop_02",
        category="two_hop",
        premises=["Krypton-9 precedes Epsilon-3.", "Epsilon-3 precedes Pyron-1."],
        question="Does Krypton-9 precede Pyron-1?",
        expected_verdict="YES",
        explanation="Transitive deduction: Krypton-9 -> Epsilon-3 -> Pyron-1.",
        entities=["Krypton-9", "Epsilon-3", "Pyron-1"],
    ),
    DirectionalTask(
        task_id="m41_2hop_03",
        category="two_hop",
        premises=["Celestia is inside Pyron-1.", "Pyron-1 is inside Solon-X."],
        question="Is Celestia inside Solon-X?",
        expected_verdict="YES",
        explanation="Transitive deduction: Celestia -> Pyron-1 -> Solon-X.",
        entities=["Celestia", "Pyron-1", "Solon-X"],
    ),
    DirectionalTask(
        task_id="m41_2hop_04",
        category="two_hop",
        premises=["Astraea subsumes Nebula-8.", "Nebula-8 subsumes Quasar-5."],
        question="Does Astraea subsume Quasar-5?",
        expected_verdict="YES",
        explanation="Transitive deduction: Astraea -> Nebula-8 -> Quasar-5.",
        entities=["Astraea", "Nebula-8", "Quasar-5"],
    ),
    DirectionalTask(
        task_id="m41_2hop_05",
        category="two_hop",
        premises=["Nexus-4 precedes Vortice.", "Vortice precedes Celestia."],
        question="Does Nexus-4 precede Celestia?",
        expected_verdict="YES",
        explanation="Transitive deduction: Nexus-4 -> Vortice -> Celestia.",
        entities=["Nexus-4", "Vortice", "Celestia"],
    ),

    # --- Category 3: 3-Hop Transitive Deduction (5 tasks -> YES) ---
    DirectionalTask(
        task_id="m41_3hop_01",
        category="three_hop",
        premises=["Zorblax is inside Nexus-4.", "Nexus-4 is inside Vortice.", "Vortice is inside Celestia."],
        question="Is Zorblax inside Celestia?",
        expected_verdict="YES",
        explanation="3-hop transitive chain: Zorblax -> Nexus-4 -> Vortice -> Celestia.",
        entities=["Zorblax", "Nexus-4", "Vortice", "Celestia"],
    ),
    DirectionalTask(
        task_id="m41_3hop_02",
        category="three_hop",
        premises=["Krypton-9 precedes Epsilon-3.", "Epsilon-3 precedes Pyron-1.", "Pyron-1 precedes Solon-X."],
        question="Does Krypton-9 precede Solon-X?",
        expected_verdict="YES",
        explanation="3-hop transitive chain: Krypton-9 -> Epsilon-3 -> Pyron-1 -> Solon-X.",
        entities=["Krypton-9", "Epsilon-3", "Pyron-1", "Solon-X"],
    ),
    DirectionalTask(
        task_id="m41_3hop_03",
        category="three_hop",
        premises=["Thalos is inside Astraea.", "Astraea is inside Nebula-8.", "Nebula-8 is inside Quasar-5."],
        question="Is Thalos inside Quasar-5?",
        expected_verdict="YES",
        explanation="3-hop transitive chain: Thalos -> Astraea -> Nebula-8 -> Quasar-5.",
        entities=["Thalos", "Astraea", "Nebula-8", "Quasar-5"],
    ),
    DirectionalTask(
        task_id="m41_3hop_04",
        category="three_hop",
        premises=["Vortice precedes Celestia.", "Celestia precedes Pyron-1.", "Pyron-1 precedes Solon-X."],
        question="Does Vortice precede Solon-X?",
        expected_verdict="YES",
        explanation="3-hop transitive chain: Vortice -> Celestia -> Pyron-1 -> Solon-X.",
        entities=["Vortice", "Celestia", "Pyron-1", "Solon-X"],
    ),
    DirectionalTask(
        task_id="m41_3hop_05",
        category="three_hop",
        premises=["Nexus-4 subsumes Vortice.", "Vortice subsumes Celestia.", "Celestia subsumes Thalos."],
        question="Does Nexus-4 subsume Thalos?",
        expected_verdict="YES",
        explanation="3-hop transitive chain: Nexus-4 -> Vortice -> Celestia -> Thalos.",
        entities=["Nexus-4", "Vortice", "Celestia", "Thalos"],
    ),

    # --- Category 4: Asymmetric Directional Reversals (5 tasks -> NO) ---
    DirectionalTask(
        task_id="m41_asym_01",
        category="asymmetric",
        premises=["Zorblax is inside Nexus-4."],
        question="Is Nexus-4 inside Zorblax?",
        expected_verdict="NO",
        explanation="Strict asymmetry: if A is inside B, B cannot be inside A.",
        entities=["Zorblax", "Nexus-4"],
    ),
    DirectionalTask(
        task_id="m41_asym_02",
        category="asymmetric",
        premises=["Krypton-9 precedes Epsilon-3."],
        question="Does Epsilon-3 precede Krypton-9?",
        expected_verdict="NO",
        explanation="Temporal/ordinal precedence is asymmetric; reverse order is provably false.",
        entities=["Krypton-9", "Epsilon-3"],
    ),
    DirectionalTask(
        task_id="m41_asym_03",
        category="asymmetric",
        premises=["Solon-X powers Thalos."],
        question="Does Thalos power Solon-X?",
        expected_verdict="NO",
        explanation="Power flow is directed and strictly asymmetric.",
        entities=["Solon-X", "Thalos"],
    ),
    DirectionalTask(
        task_id="m41_asym_04",
        category="asymmetric",
        premises=["Astraea subsumes Nebula-8."],
        question="Does Nebula-8 subsume Astraea?",
        expected_verdict="NO",
        explanation="Subsumption is strictly asymmetric; container cannot be subsumed by contained entity.",
        entities=["Astraea", "Nebula-8"],
    ),
    DirectionalTask(
        task_id="m41_asym_05",
        category="asymmetric",
        premises=["Nexus-4 is inside Vortice.", "Vortice is inside Celestia."],
        question="Is Celestia inside Nexus-4?",
        expected_verdict="NO",
        explanation="Multi-hop transitive reversal is provably false: Celestia contains Nexus-4, not vice-versa.",
        entities=["Nexus-4", "Vortice", "Celestia"],
    ),

    # --- Category 5: Non-Transitive Negative Deductions (5 tasks -> NO) ---
    DirectionalTask(
        task_id="m41_nontrans_01",
        category="nontransitive",
        premises=["Vortice is connected_to Celestia.", "Celestia is connected_to Pyron-1."],
        question="Is Vortice connected_to Pyron-1?",
        expected_verdict="NO",
        explanation="Local network connection is non-transitive; multi-hop connection does not entail direct link.",
        entities=["Vortice", "Celestia", "Pyron-1"],
    ),
    DirectionalTask(
        task_id="m41_nontrans_02",
        category="nontransitive",
        premises=["Solon-X powers Thalos.", "Thalos powers Astraea."],
        question="Does Solon-X power Astraea?",
        expected_verdict="NO",
        explanation="Power transmission is direct non-transitive.",
        entities=["Solon-X", "Thalos", "Astraea"],
    ),
    DirectionalTask(
        task_id="m41_nontrans_03",
        category="nontransitive",
        premises=["Zorblax is connected_to Nexus-4.", "Nexus-4 is connected_to Solon-X."],
        question="Is Zorblax connected_to Solon-X?",
        expected_verdict="NO",
        explanation="Non-transitive connection rule forbids transitive leap.",
        entities=["Zorblax", "Nexus-4", "Solon-X"],
    ),
    DirectionalTask(
        task_id="m41_nontrans_04",
        category="nontransitive",
        premises=["Krypton-9 is connected_to Epsilon-3.", "Epsilon-3 is connected_to Thalos."],
        question="Is Krypton-9 connected_to Thalos?",
        expected_verdict="NO",
        explanation="Intermediate link does not imply direct link.",
        entities=["Krypton-9", "Epsilon-3", "Thalos"],
    ),
    DirectionalTask(
        task_id="m41_nontrans_05",
        category="nontransitive",
        premises=["Celestia powers Pyron-1.", "Pyron-1 powers Quasar-5."],
        question="Does Celestia power Quasar-5?",
        expected_verdict="NO",
        explanation="Power routing is direct non-transitive.",
        entities=["Celestia", "Pyron-1", "Quasar-5"],
    ),

    # --- Category 6: Epistemic Boundary: Ungrounded Missing Entities (5 tasks -> UNKNOWN) ---
    DirectionalTask(
        task_id="m41_ungrounded_01",
        category="ungrounded",
        premises=["Zorblax is inside Nexus-4."],
        question="Is Zorblax inside Morvan-Sector-99?",
        expected_verdict="UNKNOWN",
        explanation="Entity 'Morvan-Sector-99' is absent from premises; truth value is underdetermined.",
        entities=["Zorblax", "Morvan-Sector-99"],
    ),
    DirectionalTask(
        task_id="m41_ungrounded_02",
        category="ungrounded",
        premises=["Krypton-9 precedes Epsilon-3."],
        question="Does Krypton-9 precede Quasar-Omega?",
        expected_verdict="UNKNOWN",
        explanation="Entity 'Quasar-Omega' is ungrounded in premises.",
        entities=["Krypton-9", "Quasar-Omega"],
    ),
    DirectionalTask(
        task_id="m41_ungrounded_03",
        category="ungrounded",
        premises=["Vortice is inside Celestia."],
        question="Does Ghost-Node power Vortice?",
        expected_verdict="UNKNOWN",
        explanation="Subject 'Ghost-Node' is completely unknown; abstain from guessing.",
        entities=["Ghost-Node", "Vortice"],
    ),
    DirectionalTask(
        task_id="m41_ungrounded_04",
        category="ungrounded",
        premises=["Astraea subsumes Nebula-8."],
        question="Is Astraea inside Quantum-Zero?",
        expected_verdict="UNKNOWN",
        explanation="Target object 'Quantum-Zero' has zero premise support.",
        entities=["Astraea", "Quantum-Zero"],
    ),
    DirectionalTask(
        task_id="m41_ungrounded_05",
        category="ungrounded",
        premises=["Solon-X exists in sector Alpha.", "Thalos exists in sector Beta."],
        question="Is Solon-X inside Thalos?",
        expected_verdict="UNKNOWN",
        explanation="Premises assert coordinate existence but contain no topological containment links.",
        entities=["Solon-X", "Thalos"],
    ),

    # --- Category 7: Epistemic Boundary: Contradictory Cycles (5 tasks -> UNKNOWN) ---
    DirectionalTask(
        task_id="m41_contradict_01",
        category="contradictory",
        premises=["Zorblax is inside Nexus-4.", "Nexus-4 is inside Zorblax."],
        question="Is Zorblax inside Nexus-4?",
        expected_verdict="UNKNOWN",
        explanation="Mutual containment cycle violates asymmetry, creating an irreconcilable premise contradiction.",
        entities=["Zorblax", "Nexus-4"],
    ),
    DirectionalTask(
        task_id="m41_contradict_02",
        category="contradictory",
        premises=["Krypton-9 precedes Epsilon-3.", "Epsilon-3 precedes Krypton-9."],
        question="Does Krypton-9 precede Epsilon-3?",
        expected_verdict="UNKNOWN",
        explanation="Mutually contradictory temporal cycle; safe epistemic verdict is UNKNOWN.",
        entities=["Krypton-9", "Epsilon-3"],
    ),
    DirectionalTask(
        task_id="m41_contradict_03",
        category="contradictory",
        premises=["Astraea subsumes Thalos.", "Thalos subsumes Astraea."],
        question="Does Astraea subsume Thalos?",
        expected_verdict="UNKNOWN",
        explanation="Contradictory reciprocal subsumption cycle.",
        entities=["Astraea", "Thalos"],
    ),
    DirectionalTask(
        task_id="m41_contradict_04",
        category="contradictory",
        premises=["Solon-X powers Vortice.", "Vortice powers Solon-X."],
        question="Does Solon-X power Vortice?",
        expected_verdict="UNKNOWN",
        explanation="Contradictory reciprocal power loop in strictly directed graph.",
        entities=["Solon-X", "Vortice"],
    ),
    DirectionalTask(
        task_id="m41_contradict_05",
        category="contradictory",
        premises=["Celestia precedes Pyron-1.", "Pyron-1 precedes Celestia."],
        question="Does Celestia precede Pyron-1?",
        expected_verdict="UNKNOWN",
        explanation="Cyclic paradox in strict ordering.",
        entities=["Celestia", "Pyron-1"],
    ),

    # --- Category 8: Zero-Shot Counterfactual Transfer (5 tasks) ---
    DirectionalTask(
        task_id="m41_counterfactual_01",
        category="counterfactual",
        premises=["Glapton is inside Xyphos-7.", "Xyphos-7 is inside Morvan-Prime."],
        question="Is Glapton inside Morvan-Prime?",
        expected_verdict="YES",
        explanation="Transitive deduction on unseen synthetic entities: Glapton -> Xyphos-7 -> Morvan-Prime.",
        entities=["Glapton", "Xyphos-7", "Morvan-Prime"],
        is_counterfactual=True,
    ),
    DirectionalTask(
        task_id="m41_counterfactual_02",
        category="counterfactual",
        premises=["Glapton is inside Xyphos-7."],
        question="Is Xyphos-7 inside Glapton?",
        expected_verdict="NO",
        explanation="Universal asymmetry rule transfers zero-shot to completely unfamiliar entities.",
        entities=["Glapton", "Xyphos-7"],
        is_counterfactual=True,
    ),
    DirectionalTask(
        task_id="m41_counterfactual_03",
        category="counterfactual",
        premises=["Zephyr-12 precedes Oryx-9.", "Oryx-9 precedes Bravon-3."],
        question="Does Zephyr-12 precede Bravon-3?",
        expected_verdict="YES",
        explanation="Precedence transitivity transfers zero-shot to unfamiliar entities.",
        entities=["Zephyr-12", "Oryx-9", "Bravon-3"],
        is_counterfactual=True,
    ),
    DirectionalTask(
        task_id="m41_counterfactual_04",
        category="counterfactual",
        premises=["Zephyr-12 precedes Oryx-9."],
        question="Does Oryx-9 precede Zephyr-12?",
        expected_verdict="NO",
        explanation="Asymmetric reversal on counterfactual entities.",
        entities=["Zephyr-12", "Oryx-9"],
        is_counterfactual=True,
    ),
    DirectionalTask(
        task_id="m41_counterfactual_05",
        category="counterfactual",
        premises=["Glapton is inside Xyphos-7."],
        question="Is Glapton inside Phantom-Entity-X?",
        expected_verdict="UNKNOWN",
        explanation="Epistemic boundary transfers zero-shot: unknown entity yields UNKNOWN.",
        entities=["Glapton", "Phantom-Entity-X"],
        is_counterfactual=True,
    ),
]


# -----------------------------------------------------------------------------
# Sequential Multi-Episode Relational Worlds for Learning-to-Learn Transfer
# -----------------------------------------------------------------------------

SEQUENTIAL_RELATION_WORLDS: list[dict[str, Any]] = [
    {
        "world_id": "world_1_cosmic_containment",
        "subject": "cosmic_containment",
        "relation": "is_inside",
        "train_pairs": [
            ("[PREMISES] Zorblax is inside Nexus-4. Nexus-4 is inside Vortice. [QUESTION] Is Zorblax inside Vortice?", "[ANSWER] YES"),
            ("[PREMISES] Zorblax is inside Nexus-4. [QUESTION] Is Nexus-4 inside Zorblax?", "[ANSWER] NO"),
            ("[PREMISES] Zorblax is inside Nexus-4. [QUESTION] Is Zorblax inside UnknownSector-99?", "[ANSWER] UNKNOWN"),
        ],
        "eval_tasks": [t for t in M41_DIRECTIONAL_BENCHMARK if "Zorblax" in t.entities or "Nexus-4" in t.entities],
    },
    {
        "world_id": "world_2_temporal_precedence",
        "subject": "temporal_precedence",
        "relation": "precedes",
        "train_pairs": [
            ("[PREMISES] Krypton-9 precedes Epsilon-3. Epsilon-3 precedes Pyron-1. [QUESTION] Does Krypton-9 precede Pyron-1?", "[ANSWER] YES"),
            ("[PREMISES] Krypton-9 precedes Epsilon-3. [QUESTION] Does Epsilon-3 precede Krypton-9?", "[ANSWER] NO"),
            ("[PREMISES] Krypton-9 precedes Epsilon-3. [QUESTION] Does Krypton-9 precede Quasar-Omega?", "[ANSWER] UNKNOWN"),
        ],
        "eval_tasks": [t for t in M41_DIRECTIONAL_BENCHMARK if "Krypton-9" in t.entities or "Epsilon-3" in t.entities],
    },
    {
        "world_id": "world_3_hierarchical_subsumption",
        "subject": "hierarchical_subsumption",
        "relation": "subsumes",
        "train_pairs": [
            ("[PREMISES] Astraea subsumes Nebula-8. Nebula-8 subsumes Quasar-5. [QUESTION] Does Astraea subsume Quasar-5?", "[ANSWER] YES"),
            ("[PREMISES] Astraea subsumes Nebula-8. [QUESTION] Does Nebula-8 subsume Astraea?", "[ANSWER] NO"),
            ("[PREMISES] Astraea subsumes Nebula-8. [QUESTION] Does Astraea subsume Quantum-Zero?", "[ANSWER] UNKNOWN"),
        ],
        "eval_tasks": [t for t in M41_DIRECTIONAL_BENCHMARK if "Astraea" in t.entities or "Nebula-8" in t.entities],
    },
    {
        "world_id": "world_4_directed_power",
        "subject": "directed_power",
        "relation": "powers",
        "train_pairs": [
            ("[PREMISES] Solon-X powers Thalos. [QUESTION] Does Solon-X power Thalos?", "[ANSWER] YES"),
            ("[PREMISES] Solon-X powers Thalos. [QUESTION] Does Thalos power Solon-X?", "[ANSWER] NO"),
            ("[PREMISES] Solon-X powers Thalos. Thalos powers Astraea. [QUESTION] Does Solon-X power Astraea?", "[ANSWER] NO"),
            ("[PREMISES] Solon-X powers Thalos. [QUESTION] Does Ghost-Node power Vortice?", "[ANSWER] UNKNOWN"),
        ],
        "eval_tasks": [t for t in M41_DIRECTIONAL_BENCHMARK if "Solon-X" in t.entities or "Thalos" in t.entities],
    },
]
