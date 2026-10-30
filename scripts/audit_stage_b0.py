"""CLI wrapper for Stage B.0 Forensic Audit Module."""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_mini_gpt.evaluation.audit_stage_b0 import audit_stage_b0_artifacts

if __name__ == "__main__":
    p = Path("runs/kaggle-stage-a13-b0/output/stage_b0_v1.json")
    if not p.exists():
        p = Path("data/manifests/stage_a13_b0_v1.json")
    res = audit_stage_b0_artifacts(p)
    print(json.dumps(res, indent=2))
