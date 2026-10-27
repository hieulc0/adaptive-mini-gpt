"""Remote first-divergence investigation; no learned candidate or promotion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from frame_a2_common import ROOT_HASH, fork_owner, state_digest
from run_attribute_learning import encode, file_hash
from run_stable_attributes import next_update

from adaptive_mini_gpt.data.stable_attributes import corpus, streams
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import LearningState
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--strict", action="store_true")
    a = p.parse_args()
    require_remote_cuda()
    if a.strict:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("Strict probe needs declared CUBLAS workspace before CUDA init")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    if file_hash(a.root) != ROOT_HASH:
        raise ValueError("Wrong immutable parent")
    parent = LearningState.load(a.root)
    rows = corpus()
    batch, _ = encode(
        rows, ByteTextTokenizerM103(), parent.model.student.base_model.config.context_length
    )
    plan = streams(rows, 42, 100)["augmented"]
    owners = [fork_owner(parent, 42), fork_owner(parent, 42)]
    for owner in owners:
        owner.metadata["a3_arm"] = "augmented"
        owner.external.update(a3_indices=plan, a3_cursor=0)
    names = [{id(v): n for n, v in o.model.named_parameters()} for o in owners]
    groups = [
        [[names[j][id(v)] for v in g["params"]] for g in o.optimizer.param_groups]
        for j, o in enumerate(owners)
    ]
    result: dict[str, Any] = dict(
        strict=a.strict,
        workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        torch=torch.__version__,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        groups_equal=groups[0] == groups[1],
        initial_signatures=[
            state_digest((o.model.state_dict(), o.optimizer.state_dict())) for o in owners
        ],
        steps=[],
        scope="first-divergence diagnostic only; no qualified candidate",
    )
    output = Path(a.output)
    try:
        for step in range(100):
            rng = owners[0].snapshot()["rng"]
            losses = []
            hashes = []
            forward = []
            input_hashes = []
            for owner in owners:
                py, np_rng, cpu, cuda = rng
                __import__("random").setstate(py)
                __import__("numpy").random.set_state(np_rng)
                torch.set_rng_state(cpu.cpu())
                torch.cuda.set_rng_state_all([s.cpu() for s in cuda])
                indices = plan[step]
                owner.model.eval()
                with torch.no_grad():
                    forward.append(
                        owner.model(batch[0][indices], batch[1][indices])[1].detach().clone()
                    )
                input_hashes.append(state_digest(tuple(v[indices] for v in batch)))
                loss, ix = next_update(owner, batch)
                losses.append(loss)
                hashes.append(
                    state_digest((owner.model.state_dict(), owner.optimizer.state_dict()))
                )
            differences = []
            for (n, x), (_, y) in zip(
                owners[0].model.named_parameters(), owners[1].model.named_parameters(), strict=True
            ):
                if (
                    not torch.equal(x, y)
                    or ((x.grad is None) != (y.grad is None))
                    or (x.grad is not None and not torch.equal(x.grad, y.grad))
                ):
                    differences.append(
                        dict(
                            name=n,
                            weight_max_abs=float((x - y).abs().max()),
                            gradient_max_abs=float((x.grad - y.grad).abs().max())
                            if x.grad is not None and y.grad is not None
                            else None,
                        )
                    )
            result["steps"].append(
                dict(
                    step=step + 1,
                    losses=losses,
                    hashes=hashes,
                    differences=differences,
                    input_hashes=input_hashes,
                    forward_hashes=[state_digest(v) for v in forward],
                    forward_max_abs=float((forward[0] - forward[1]).abs().max()),
                )
            )
            if differences or hashes[0] != hashes[1]:
                break
        result["identical"] = all(
            x["hashes"][0] == x["hashes"][1] and not x["differences"] for x in result["steps"]
        )
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            next_update(owners[0], batch)
        result["operators"] = [
            x.key
            for x in prof.key_averages()
            if any(k in x.key.lower() for k in ("attention", "embedding", "index", "scatter"))
        ]
    except Exception as error:
        import traceback

        result.update(error=repr(error), traceback=traceback.format_exc(), identical=False)
    output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
