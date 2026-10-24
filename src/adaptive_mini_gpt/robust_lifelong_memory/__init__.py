"""Milestone M72: Robust Lifelong Neural Memory and Capacity Efficiency."""
from __future__ import annotations

from adaptive_mini_gpt.robust_lifelong_memory.audit_m71 import (
    M71ForensicAuditReport,
    M71ScientificAuditor,
)
from adaptive_mini_gpt.robust_lifelong_memory.capacity_accounting import (
    CapacityAuditRecord,
    CompleteModelCapacityTracker,
)
from adaptive_mini_gpt.robust_lifelong_memory.lifelong_stream_engine import (
    M72EpisodeRecord,
    M72StreamResult,
    RobustLifelongStreamEngine,
)
from adaptive_mini_gpt.robust_lifelong_memory.lifelong_task_stream import (
    build_m72_lifelong_stream_sequence,
    create_anti_transitive_dag_env,
    create_contradictory_lattice_env,
    create_inverted_rail_parity_env,
    create_modular_arithmetic_relational_env,
)
from adaptive_mini_gpt.robust_lifelong_memory.safeguards import (
    M72SafeguardReport,
    M72SafeguardsAuditor,
)
from adaptive_mini_gpt.robust_lifelong_memory.skill_lifecycle import (
    RobustSkillRecord,
    RobustSkillRegistry,
    SkillState,
)

__all__ = [
    "SkillState",
    "RobustSkillRecord",
    "RobustSkillRegistry",
    "CapacityAuditRecord",
    "CompleteModelCapacityTracker",
    "M71ScientificAuditor",
    "M71ForensicAuditReport",
    "build_m72_lifelong_stream_sequence",
    "create_anti_transitive_dag_env",
    "create_inverted_rail_parity_env",
    "create_contradictory_lattice_env",
    "create_modular_arithmetic_relational_env",
    "M72EpisodeRecord",
    "M72StreamResult",
    "RobustLifelongStreamEngine",
    "M72SafeguardsAuditor",
    "M72SafeguardReport",
]
