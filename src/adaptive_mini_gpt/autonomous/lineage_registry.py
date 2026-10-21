"""Model Lineage Registry and Provenance Tracker (M26 Phase 1).

Maintains a formal inventory of Adaptive Mini-GPT checkpoints across milestones,
explicitly separating:
  1. Stable operational reasoning systems (deterministic/hybrid planners).
  2. Base neural pretraining lineages (Seeds 42, 43, 44).
  3. Specialized experimental candidates (arithmetic, language, logic).
  4. Consolidated continual learning candidates.

Guarantees checkpoint immutability and prevents false assumptions about prior
acquired capabilities during continual learning.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class LineageRole(str, Enum):
    """Categorization of models within the Adaptive Mini-GPT ecosystem."""

    OPERATIONAL_SYSTEM = "operational_system"
    NEURAL_BASELINE = "neural_baseline"
    SPECIALIZED_CANDIDATE = "specialized_candidate"
    CONSOLIDATED_CANDIDATE = "consolidated_candidate"


@dataclass(frozen=True)
class LineageRecord:
    """Immutable metadata record for a model checkpoint or operational system."""

    lineage_id: str
    role: LineageRole
    milestone: str
    architecture: str
    param_count: int
    tokenizer_fingerprint: str
    checkpoint_sha256: str
    source_checkpoint_id: str | None
    verified_capabilities: list[str]
    unverified_or_failing: list[str]
    operational_status: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =====================================================================
# Canonical Lineage Inventory (M18 - M25)
# =====================================================================

CANONICAL_LINEAGE_REGISTRY: dict[str, LineageRecord] = {
    # 1. Operational Systems
    "m18_operational_planner": LineageRecord(
        lineage_id="m18_operational_planner",
        role=LineageRole.OPERATIONAL_SYSTEM,
        milestone="M18",
        architecture="Grammar-Constrained Execution Planner (Topological Search)",
        param_count=0,
        tokenizer_fingerprint="f698e45e209b7fe6",
        checkpoint_sha256="deterministic_code_invariant",
        source_checkpoint_id=None,
        verified_capabilities=[
            "100.0% verified kinematic planning",
            "100.0% depth ladder planning (2-5 steps)",
            "100.0% safe abstention on missing premises",
            "15.5ms low-latency execution",
        ],
        unverified_or_failing=[],
        operational_status="ACTIVE_PRODUCTION_BASELINE",
        notes="Remains the operational reasoning baseline since M18.",
    ),
    # 2. Base Neural Pretrained Lineages (Seeds 42, 43, 44)
    "base_pretrain_seed_42": LineageRecord(
        lineage_id="base_pretrain_seed_42",
        role=LineageRole.NEURAL_BASELINE,
        milestone="M18",
        architecture="MiniGPT-RoPE (4L, 4H, d128, ctx256)",
        param_count=895744,
        tokenizer_fingerprint="f698e45e209b7fe6",
        checkpoint_sha256="a0ffc73b6b7e7b5faf53c9b59d351a29ee8a45e9a1a677bcbff23cf659b0d174",
        source_checkpoint_id=None,
        verified_capabilities=["General natural language pretraining representation"],
        unverified_or_failing=[
            "Elementary arithmetic (<1% unaided)",
            "Multi-step symbolic logic (0% unaided)",
            "Reading comprehension extraction (0% unaided)",
        ],
        operational_status="CANONICAL_CANDIDATE_ROOT",
        notes="Starting point for seed 42 candidate branching.",
    ),
    "base_pretrain_seed_43": LineageRecord(
        lineage_id="base_pretrain_seed_43",
        role=LineageRole.NEURAL_BASELINE,
        milestone="M18",
        architecture="MiniGPT-RoPE (4L, 4H, d128, ctx256)",
        param_count=895744,
        tokenizer_fingerprint="f698e45e209b7fe6",
        checkpoint_sha256="5eec4fc2c14d4bcb4ffbfb73e92b0bc801c7039e1565f2878631e5c724d53c7e",
        source_checkpoint_id=None,
        verified_capabilities=["General natural language pretraining representation"],
        unverified_or_failing=[
            "Elementary arithmetic (<1% unaided)",
            "Multi-step symbolic logic (0% unaided)",
            "Reading comprehension extraction (0% unaided)",
        ],
        operational_status="CANONICAL_CANDIDATE_ROOT",
        notes="Starting point for seed 43 candidate branching.",
    ),
    "base_pretrain_seed_44": LineageRecord(
        lineage_id="base_pretrain_seed_44",
        role=LineageRole.NEURAL_BASELINE,
        milestone="M18",
        architecture="MiniGPT-RoPE (4L, 4H, d128, ctx256)",
        param_count=895744,
        tokenizer_fingerprint="f698e45e209b7fe6",
        checkpoint_sha256="d21cd0a0d7e25aea7cef6a755f9dc5a50541993dcfb7488358e200214cd33c65",
        source_checkpoint_id=None,
        verified_capabilities=["General natural language pretraining representation"],
        unverified_or_failing=[
            "Elementary arithmetic (<1% unaided)",
            "Multi-step symbolic logic (0% unaided)",
            "Reading comprehension extraction (0% unaided)",
        ],
        operational_status="CANONICAL_CANDIDATE_ROOT",
        notes="Starting point for seed 44 candidate branching.",
    ),
    # 3. Specialized Arithmetic Specialists (M24)
    "m24_arithmetic_specialist_seed_44": LineageRecord(
        lineage_id="m24_arithmetic_specialist_seed_44",
        role=LineageRole.SPECIALIZED_CANDIDATE,
        milestone="M24",
        architecture="MiniGPT-RoPE + Balanced Curriculum",
        param_count=895744,
        tokenizer_fingerprint="f698e45e209b7fe6",
        checkpoint_sha256="m24_arm_c_seed_44_artifact",
        source_checkpoint_id="base_pretrain_seed_44",
        verified_capabilities=[
            "93.5% elementary transition exact match",
            "98.9% carry prediction accuracy",
            "60.4% extrapolation on 4-8 digits",
            "100.0% cascading carry resolution up to 32 digits",
        ],
        unverified_or_failing=[
            "Language comprehension (untested / not trained)",
            "Symbolic logic (untested / not trained)",
            "Retention drift on general prose (+3.9 nats)",
        ],
        operational_status="RESEARCH_SPECIALIST_FROZEN",
        notes="Isolated arithmetic specialist. Not promoted to shared base.",
    ),
    # 4. Specialized M25 Candidates
    "m25_language_candidate": LineageRecord(
        lineage_id="m25_language_candidate",
        role=LineageRole.SPECIALIZED_CANDIDATE,
        milestone="M25",
        architecture="MiniGPT-RoPE (4L, 4H, d128)",
        param_count=895744,
        tokenizer_fingerprint="f698e45e209b7fe6",
        checkpoint_sha256="m25_b_language_lineage",
        source_checkpoint_id="base_pretrain_seed_42",
        verified_capabilities=[
            "100.0% QA format compliance",
            "4.55% held-out passage extraction",
            "16.67% safe abstention on unanswerable questions",
            "Negative drift on pretraining prose (-0.45 nats)",
        ],
        unverified_or_failing=[
            "Arithmetic (0.0% exact match)",
            "Symbolic logic (0.0% exact match)",
        ],
        operational_status="RESEARCH_SPECIALIST_FROZEN",
        notes="Proved language feasibility on 895k parameter backbone.",
    ),
    "m25_logic_candidate": LineageRecord(
        lineage_id="m25_logic_candidate",
        role=LineageRole.SPECIALIZED_CANDIDATE,
        milestone="M25",
        architecture="MiniGPT-RoPE (4L, 4H, d128)",
        param_count=895744,
        tokenizer_fingerprint="f698e45e209b7fe6",
        checkpoint_sha256="m25_c_logic_lineage",
        source_checkpoint_id="base_pretrain_seed_42",
        verified_capabilities=[
            "50.67% exact match on unseen variable names",
            "35.89% safe abstention on contradictions",
            "Negative drift on pretraining prose (-0.44 nats)",
        ],
        unverified_or_failing=[
            "Arithmetic (0.0% exact match)",
            "Language comprehension (0.0% exact match)",
        ],
        operational_status="RESEARCH_SPECIALIST_FROZEN",
        notes="Strong deductive reasoning on unfamiliar symbols.",
    ),
    "m25_continual_candidate": LineageRecord(
        lineage_id="m25_continual_candidate",
        role=LineageRole.CONSOLIDATED_CANDIDATE,
        milestone="M25",
        architecture="MiniGPT-RoPE (4L, 4H, d128, Interleaved)",
        param_count=895744,
        tokenizer_fingerprint="f698e45e209b7fe6",
        checkpoint_sha256="m25_d_continual_lineage",
        source_checkpoint_id="base_pretrain_seed_42",
        verified_capabilities=[
            "Coexistence of arithmetic (13.67%, carry 97.5%), language (3.03%), logic (49.33%)",
            "97.4% logic skill retention under joint training",
        ],
        unverified_or_failing=[
            "Retention drift (+0.4195 nats positive interference on general prose)",
            "Arithmetic below operational threshold",
        ],
        operational_status="PROMOTION_REJECTED",
        notes="NO-GO on promotion. Identified continual learning interference.",
    ),
}


def get_candidate_root_lineage(seed: int) -> LineageRecord:
    """Retrieve canonical immutable candidate root for an independent seed."""
    key = f"base_pretrain_seed_{seed}"
    if key not in CANONICAL_LINEAGE_REGISTRY:
        raise KeyError(f"Seed {seed} not found in canonical lineage registry")
    return CANONICAL_LINEAGE_REGISTRY[key]


def verify_checkpoint_integrity(
    checkpoint_path: Path | str,
    expected_sha256: str,
) -> bool:
    """Verify that a disk checkpoint strictly matches its registered SHA-256."""
    p = Path(checkpoint_path)
    if not p.is_file():
        return False
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    return digest == expected_sha256
