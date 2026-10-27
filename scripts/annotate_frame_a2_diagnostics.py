"""Remote provenance-only copies of unpromoted diagnostics; raw artifacts immutable."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from frame_a2_common import record_stage_provenance, state_digest
from run_attribute_learning import encode, file_hash
from run_frame_a2_diagnosis import ORIGINAL

from adaptive_mini_gpt.data.attribute_learning import AttributeCase
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import LearningState
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda


def signature(owner):
    snap = owner.snapshot()
    return state_digest(
        {
            "model": owner.model.state_dict(),
            "optimizer": owner.optimizer.state_dict(),
            "scheduler": owner.scheduler.state_dict() if owner.scheduler else None,
            "external": owner.external,
            "rng": snap["rng"],
            "gradients": snap["gradients"],
            "flags": {n: p.requires_grad for n, p in owner.model.named_parameters()},
            "modes": {n: m.training for n, m in owner.model.named_modules()},
        }
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    require_remote_cuda()
    source = Path(a.input_dir)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = json.loads((source / "results.json").read_text())
    publisher = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    rows = [
        AttributeCase(**r)
        for r in json.loads(Path("data/fixtures/stage_a_original_tiny.json").read_text())["cases"]
    ]
    report = {
        "training_source_sha": result["provenance"]["git_sha"],
        "publication_source_sha": publisher,
        "copies": {},
        "scope": "metadata only; two next-update checks per copy are restored and never retained",
    }
    for name, arm in result["optimizer"].items():
        raw = source / f"optimizer_{name}.pt"
        if file_hash(raw) != arm["checkpoint_sha256"]:
            raise ValueError("Wrong raw artifact")
        owner = LearningState.load(raw)
        original = owner.snapshot()
        before = signature(owner)
        inherited = {k: owner.metadata.get(k) for k in ("git_sha", "data_hash", "steps")}
        record_stage_provenance(
            owner,
            training_sha=result["provenance"]["git_sha"],
            publication_sha=publisher,
            parent_hash=ORIGINAL,
            data_hash=file_hash("data/fixtures/stage_a_original_tiny.json"),
            stage="A.2_optimizer_diagnostic",
        )
        owner.metadata["supervision_schema"] = {
            "attribute_ids": (10, 15),
            "state_targets": None,
            "loss_scale": 1.0,
        }
        owner.metadata["annotation_of_sha256"] = arm["checkpoint_sha256"]
        corrected = owner.snapshot()
        assert signature(owner) == before
        path = out / f"corrected_optimizer_{name}.pt"
        owner.save(path)
        loaded = LearningState.load(path)
        assert signature(loaded) == before
        batch, _ = encode(
            rows, ByteTextTokenizerM103(), owner.model.student.base_model.config.context_length
        )
        outcomes = []
        for snapshot in (original, corrected):
            owner.restore(snapshot)
            ix = owner.external["frame_a2_sampler"].sample(range(9), 9)
            loss = owner.step(*[v[ix] for v in batch], attribute_ids=(10, 15))
            outcomes.append((ix, loss, signature(owner)))
        assert outcomes[0] == outcomes[1]
        report["copies"][name] = {
            "raw_sha256": arm["checkpoint_sha256"],
            "corrected_sha256": file_hash(path),
            "state_signature_unchanged": before,
            "next_update_identical": True,
            "raw_inherited_metadata": inherited,
        }
        (out / "annotation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
