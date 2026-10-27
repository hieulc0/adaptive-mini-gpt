"""Remote pinned reproduction, position/readout inspection and Adam ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
from frame_a2_common import (
    OLD_DATA_HASH,
    ROOT_HASH,
    ROOT_TENSOR_HASH,
    fork_owner,
    intervene_backbone,
    record_stage_provenance,
)
from run_attribute_learning import encode, evaluate, file_hash, scores
from run_concept_binding import encode as encode_state
from run_concept_binding import evaluate as evaluate_state
from run_concept_binding import language_model
from run_foundational_audit import digest
from run_stage_a_diagnostic import ORIGINAL, gradient_audit, pooling_trace

from adaptive_mini_gpt.data.attribute_diagnostic import CELLS, factorial
from adaptive_mini_gpt.data.attribute_learning import AttributeCase
from adaptive_mini_gpt.data.concept_binding import ConceptCase
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import LearningState
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda
from adaptive_mini_gpt.training.reproducibility import seed_all


def position_evaluation(model, batch, rows, offset):
    table = model.student.base_model.wpe
    if table is None:
        raise ValueError("Inspector requires absolute positions")

    def shift(module, args):
        ids = args[0] + offset
        if int(ids.max()) >= module.num_embeddings:
            raise ValueError("Position offset exceeds capacity")
        return (ids,)

    hook = table.register_forward_pre_hook(shift)
    try:
        return evaluate(model, batch, rows)
    finally:
        hook.remove()


def ridge_readout(train, labels, tests):
    x = torch.tensor(train, device="cuda", dtype=torch.float64)
    y = 2 * labels[:, (10, 15)].double() - 1
    mean = x.mean(0)
    std = x.std(0).clamp_min(1e-6)

    def design(v):
        z = (torch.tensor(v, device="cuda", dtype=torch.float64) - mean) / std
        return torch.cat((z, torch.ones(len(z), 1, device="cuda", dtype=z.dtype)), 1)

    x = design(train)
    penalty = torch.eye(x.shape[1], device="cuda", dtype=x.dtype) * 0.01
    penalty[-1, -1] = 0
    weights = torch.linalg.solve(x.T @ x + penalty, x.T @ y)
    return {k: (design(v) @ weights).float() for k, v in tests.items()}


def run(out, root_path, original_path):
    if file_hash(root_path) != ROOT_HASH or file_hash(original_path) != ORIGINAL:
        raise ValueError("Wrong pinned checkpoint")
    root = LearningState.load(root_path)
    if root.metadata["steps"] != 1400 or digest(root.model.state_dict()) != ROOT_TENSOR_HASH:
        raise ValueError("Wrong1400-update state")
    tokenizer = ByteTextTokenizerM103()
    context = root.model.student.base_model.config.context_length
    splits = factorial()
    data = {k: [asdict(v) for v in rows] for k, rows in splits.items()}
    data_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    assert data_hash == OLD_DATA_HASH
    batches = {}
    seqs = {}
    for k, rows in splits.items():
        batches[k], seqs[k] = encode(rows, tokenizer, context)
    expected = {}
    with Path("data/manifests/stage_a_diagnostic_v1_predictions.csv").open() as f:
        for row in csv.DictReader(f):
            if row["arm"] == "inherited_weights_inherited_moments":
                expected[row["case_id"]] = row
    result = {
        "provenance": {
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "platform": require_remote_cuda(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "parent_sha256": ROOT_HASH,
        "data_hash": data_hash,
        "reproduction": {},
        "positions": {},
        "ridge": {},
        "optimizer": {},
    }
    start = time.perf_counter()
    before = digest(root.model.state_dict())
    for cell in CELLS:
        m = evaluate(root.model, batches[cell], splits[cell])
        delta = 0
        for row, pred, logit in zip(splits[cell], m["predictions"], m["logits"], strict=True):
            e = expected[row.case_id]
            assert pred == [int(e["pred_flammable"]), int(e["pred_conductive"])]
            delta = max(
                delta,
                abs(logit[0] - float(e["logit_flammable"])),
                abs(logit[1] - float(e["logit_conductive"])),
            )
        m["max_abs_logit_delta"] = delta
        result["reproduction"][cell] = m
    for offset in (0, 4, 16, 32):
        result["positions"][str(offset)] = {
            cell: position_evaluation(root.model, batches[cell], splits[cell], offset)
            for cell in CELLS
        }
    traces = {
        cell: pooling_trace(root.model, batches[cell], splits[cell], tokenizer) for cell in CELLS
    }
    for kind in ("pooled", "concept"):
        tests = {cell: traces[cell][kind] for cell in CELLS}
        logits = ridge_readout(
            traces["familiar_familiar"][kind], batches["familiar_familiar"][2], tests
        )
        result["ridge"][kind] = {
            cell: scores(logits[cell], batches[cell][2][:, (10, 15)], splits[cell])
            for cell in CELLS
        }
    result["gradient_familiar"] = gradient_audit(root, batches["familiar_familiar"])
    result["gradient_shifted_DIAGNOSTIC_ONLY"] = gradient_audit(root, batches["familiar_novel"])
    result["pooling"] = traces
    assert digest(root.model.state_dict()) == before
    result["reproduction_weight_hash_unchanged"] = True
    torch.cuda.synchronize()
    result["diagnostic_gpu_wall_seconds"] = time.perf_counter() - start
    (out / "tokens.json").write_text(json.dumps(seqs))
    (out / "results.json").write_text(json.dumps(result, indent=2))
    # Historical original9-case mechanism; no qualified old holdout at this parent.
    parent = LearningState.load(original_path)
    rows = [
        AttributeCase(**v)
        for v in json.loads(Path("data/fixtures/stage_a_original_tiny.json").read_text())["cases"]
    ]
    batch, _ = encode(rows, tokenizer, context)
    prior = [
        ConceptCase(**v)
        for v in json.loads(Path("data/fixtures/stage_a_prior_state_tiny.json").read_text())[
            "cases"
        ]
    ]
    prior_batch, _ = encode_state(prior, tokenizer, context)
    trials = ("inherit", "moments", "age", "complete", "fresh", "rebase")
    for strategy in trials:
        if strategy == "fresh":
            seed_all(42)
            owner = LearningState(language_model("selective_plasticity"))
            owner.external["frame_a2_sampler"] = __import__("random").Random(42)
        else:
            owner = fork_owner(parent, 42)
        affected = [] if strategy == "fresh" else intervene_backbone(owner, strategy)
        record_stage_provenance(
            owner,
            training_sha=result["provenance"]["git_sha"],
            publication_sha=result["provenance"]["git_sha"],
            parent_hash=ORIGINAL,
            data_hash=file_hash("data/fixtures/stage_a_original_tiny.json"),
            stage="A.2_optimizer_diagnostic",
        )
        owner.metadata.update(
            supervision_schema={
                "attribute_ids": (10, 15),
                "state_targets": None,
                "loss_scale": 1.0,
            },
            a2_strategy=strategy,
            optimizer_policy="constant LR, global update clock retained; warm-moment age reset explicitly changes bias correction",
            parent_sha256=ORIGINAL,
        )
        arm = {
            "initial_hash": digest(owner.model.state_dict()),
            "affected_names": affected,
            "initial": evaluate(owner.model, batch, rows),
            "prior_training_only_before": evaluate_state(owner.model, prior_batch, prior),
            "curves": [],
            "initial_gradient": gradient_audit(owner, batch),
        }
        start = time.perf_counter()
        for step in range(200):
            ix = owner.external["frame_a2_sampler"].sample(range(9), 9)
            loss = owner.step(*[v[ix] for v in batch], attribute_ids=(10, 15))
            if (step + 1) % 50 == 0:
                arm["curves"].append(
                    {
                        "updates": step + 1,
                        "loss": loss,
                        "exact": evaluate(owner.model, batch, rows)["exact"],
                    }
                )
        arm["final"] = evaluate(owner.model, batch, rows)
        arm["prior_training_only_after"] = evaluate_state(owner.model, prior_batch, prior)
        arm["final_gradient"] = gradient_audit(owner, batch)
        torch.cuda.synchronize()
        arm["gpu_wall_seconds"] = time.perf_counter() - start
        arm["final_hash"] = digest(owner.model.state_dict())
        owner.save(out / f"optimizer_{strategy}.pt")
        arm["checkpoint_sha256"] = file_hash(out / f"optimizer_{strategy}.pt")
        arm["global_updates"] = owner.metadata["steps"]
        arm["backbone_adam_ages"] = {
            n: int(owner.optimizer.state[p]["step"])
            for n, p in owner.model.named_parameters()
            if n.startswith("student.") and p in owner.optimizer.state
        }
        result["optimizer"][strategy] = arm
        (out / "results.json").write_text(json.dumps(result, indent=2))
    assert (
        result["optimizer"]["rebase"]["final_hash"] == result["optimizer"]["inherit"]["final_hash"]
    )
    assert (
        result["optimizer"]["inherit"]["final"]["exact"] == 8 / 9
        or abs(result["optimizer"]["inherit"]["final"]["exact"] - 8 / 9) < 1e-6
    )
    result["rebase_exact"] = True
    result["scope"] = (
        "diagnostic only; ridge coefficients never attached, no old shifted cases optimized, no promotion"
    )
    (out / "results.json").write_text(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--original", required=True)
    a = p.parse_args()
    require_remote_cuda()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run(out, Path(a.root), Path(a.original))


if __name__ == "__main__":
    main()
