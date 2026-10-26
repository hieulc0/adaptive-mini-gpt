"""Milestone M103 Dynamic Capacity Auditor.

Inspects live PyTorch modules dynamically under the Permanent Architecture Principle:
- Base Transformer decoder (wte, wpe, blocks, ln_f, tied lm_head).
- Dynamic layer skipping routers.
- Modular residual domain adapters.
- Function-preserving growth block (if active).
- Zero unexplained parameter discrepancy mathematically certified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch

from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)


@dataclass
class CapacityAuditReportM103:
    base_backbone_params: int
    layer_routers_params: int
    domain_adapters_params: int
    growth_block_params: int
    total_measured_persistent_params: int
    sum_of_components: int
    unexplained_discrepancy: int
    reconciliation_verified: bool
    persistent_growth_active: bool


class DynamicCapacityAuditorM103:
    """Audits live PyTorch module parameters directly."""

    @staticmethod
    def audit_student(student: TextStudentM103) -> CapacityAuditReportM103:
        """Inspect all live modules and verify zero discrepancy."""
        # 1. Total persistent parameters via direct numel() summation
        total_live = sum(p.numel() for p in student.parameters())

        # 2. Base model parameters (wte, wpe, original 4 blocks, ln_f)
        # Note: lm_head is tied to wte so it shares memory/storage
        p_base = sum(p.numel() for p in student.base_model.parameters())

        # 3. Layer routers
        p_routers = sum(p.numel() for p in student.layer_routers.parameters())

        # 4. Domain adapters
        p_adapters = sum(p.numel() for p in student.adapter_modules.parameters())

        # 5. Persistent growth block (if active)
        # If growth is active, base_model has 5 blocks instead of 4
        growth_active = getattr(student, "growth_block_active", False)
        p_growth = 0
        if growth_active and len(student.base_model.blocks) > 4:
            # The 5th block is the growth block
            p_growth = sum(p.numel() for p in student.base_model.blocks[4].parameters())

        # Component breakdown:
        # Base model without the growth block
        p_base_standard = p_base - p_growth

        sum_components = p_base_standard + p_growth + p_routers + p_adapters

        discrepancy = abs(total_live - sum_components)
        verified = (discrepancy == 0)

        return CapacityAuditReportM103(
            base_backbone_params=p_base_standard,
            layer_routers_params=p_routers,
            domain_adapters_params=p_adapters,
            growth_block_params=p_growth,
            total_measured_persistent_params=total_live,
            sum_of_components=sum_components,
            unexplained_discrepancy=discrepancy,
            reconciliation_verified=verified,
            persistent_growth_active=growth_active,
        )
