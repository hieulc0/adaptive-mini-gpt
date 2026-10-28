"""REMOTE ONLY: Stage A.9 Controlled Neural Foundation Investigation.

Investigates:
- Experiment 1 (A9.1): Full-Backbone Plasticity vs. Top-2 Block Plasticity
- Experiment 2 (A9.2): Absolute Position (wpe) vs. Rotary Position Embeddings (RoPE)
- Experiment 3 (A9.3): Foundational Syntactic Language Pretraining (Conditional)
- Binding Qualification Gates 1-4 & Scientific Hypothesis Verdict (H1..H4)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from annotate_frame_a2_diagnostics import signature
from frame_a2_common import ROOT_HASH, ROOT_TENSOR_HASH, fork_owner, seed_all
from probe_representations_a8 import extract_frozen_representations, solve_ridge
from run_attribute_learning import encode, evaluate, file_hash
from run_foundational_audit import digest
from run_generalization_a6_training import prepare as a6_prepare
from run_stable_attributes import compact, enrich

from adaptive_mini_gpt.data.attribute_learning import TAG_IDS
from adaptive_mini_gpt.data.exposure_a7 import repeated_plan
from adaptive_mini_gpt.data.pretraining_a9 import (
    generate_syntactic_pretraining_corpus,
    verify_pretraining_corpus_isolation,
)
from adaptive_mini_gpt.data.probe_a8 import coverage_matched_probe_indices
from adaptive_mini_gpt.natural_language_learning_m103.semantic_bottleneck_m104_e import (
    IntegratedStudentM104_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)
from adaptive_mini_gpt.training.learning_state import (
    LayerwiseDecayScheduler,
    LearningState,
    TwoTimescaleScheduler,
)
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda

SPEC_A9 = Path("data/specs/stage_a9_v1.json")
SPEC_PRETRAIN = Path("data/specs/a9_language_pretrain_v1.json")


def prepare_a9_data():
    """Builds identical 756-case coverage data and shared evaluation sets from A.7/A.8."""
    training, newcounts, blocks, shared, _ = a6_prepare()
    spec = json.loads(SPEC_A9.read_text())
    plans = {"coverage": repeated_plan(blocks["coverage"], spec["steps"])}
    return training, newcounts, plans, shared


def build_a9_optimizer_and_clipper(model: nn.Module, arm_name: str, total_steps: int):
    """Builds decoupled or layerwise optimizer, scheduler, and clipper."""
    if arm_name == "EXP1_C_FULL_BACKBONE_LAYERWISE":
        lower_params: list[nn.Parameter] = []
        upper_params: list[nn.Parameter] = []
        readout_params: list[nn.Parameter] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(f"blocks.{i}" in name for i in (0, 1, 2)):
                lower_params.append(p)
            elif any(f"blocks.{i}" in name for i in (3, 4)) or any(k in name for k in ("wte", "ln_f")):
                upper_params.append(p)
            elif any(k in name for k in ("concept_proj", "attribute_head")):
                readout_params.append(p)
            else:
                upper_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": lower_params, "lr": 0.00025, "name": "lower_backbone"},
                {"params": upper_params, "lr": 0.0005, "name": "upper_backbone"},
                {"params": readout_params, "lr": 0.0015, "name": "readout"},
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        scheduler = LayerwiseDecayScheduler(
            optimizer,
            total_steps=total_steps,
            lower_lr=0.00025,
            upper_lr=0.0005,
            readout_lr_start=0.0015,
            readout_lr_min=0.00015,
        )

        def clipper_layerwise(m: nn.Module, opt: torch.optim.Optimizer) -> None:
            if lower_params:
                torch.nn.utils.clip_grad_norm_(lower_params, 0.5)
            if upper_params:
                torch.nn.utils.clip_grad_norm_(upper_params, 0.5)
            if readout_params:
                torch.nn.utils.clip_grad_norm_(readout_params, 1.0)

        return optimizer, scheduler, clipper_layerwise

    else:
        # Decoupled two-timescale optimizer (Stage A.8)
        backbone_params = []
        readout_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in ("concept_proj", "attribute_head")):
                readout_params.append(p)
            else:
                backbone_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": 0.0005, "name": "backbone"},
                {"params": readout_params, "lr": 0.0015, "name": "readout"},
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        scheduler = TwoTimescaleScheduler(
            optimizer,
            total_steps=total_steps,
            backbone_lr=0.0005,
            readout_lr_start=0.0015,
            readout_lr_min=0.00015,
        )

        def clipper_decoupled(m: nn.Module, opt: torch.optim.Optimizer) -> None:
            if backbone_params:
                torch.nn.utils.clip_grad_norm_(backbone_params, 0.5)
            if readout_params:
                torch.nn.utils.clip_grad_norm_(readout_params, 1.0)

        return optimizer, scheduler, clipper_decoupled


def evaluate_positional_sensitivity(
    model: Any,
    batch: tuple[torch.Tensor, ...],
    rows: list[Any],
    offsets=(8, 24),
) -> dict[str, Any]:
    """Measures prediction flip rate under +8 and +24 token offset shifts."""
    baseline_eval = evaluate(model, batch, rows)
    baseline_logits = baseline_eval.get("logits", [])

    shifts: dict[str, Any] = {}
    is_rope = getattr(model.student.base_model.config, "pos_encoding", "absolute") == "rope"

    for offset in offsets:
        if is_rope:
            for block in model.student.base_model.blocks:
                block.attn.position_offset = offset
            try:
                shifted_eval = evaluate(model, batch, rows)
            finally:
                for block in model.student.base_model.blocks:
                    block.attn.position_offset = 0
        else:
            wpe = model.student.base_model.wpe
            if wpe is None:
                continue
            hook = wpe.register_forward_pre_hook(lambda module, args, offset=offset: (args[0] + offset,))
            try:
                shifted_eval = evaluate(model, batch, rows)
            finally:
                hook.remove()

        shifted_logits = shifted_eval.get("logits", [])
        flips = 0
        total = 0
        for b_row, s_row in zip(baseline_logits, shifted_logits, strict=True):
            for b_val, s_val in zip(b_row, s_row, strict=True):
                b_bin = 1 if b_val > 0 else 0
                s_bin = 1 if s_val > 0 else 0
                if b_bin != s_bin:
                    flips += 1
                total += 1
        flip_rate = flips / total if total > 0 else 0.0
        shifts[str(offset)] = {
            "exact": shifted_eval["exact"],
            "flip_rate": flip_rate,
        }
    return shifts


def run_language_pretraining(
    model: Any,
    corpus: list[str],
    tokenizer: ByteTextTokenizerM103,
    steps: int = 1500,
    lr: float = 0.0005,
) -> list[float]:
    """Pretrains model on synthetic descriptive sentences using next-token cross-entropy."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    losses: list[float] = []

    encoded_corpus = [tokenizer.encode(s) for s in corpus]
    context_length = model.student.base_model.config.context_length

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        idx1 = ((step - 1) * 2) % len(encoded_corpus)
        idx2 = ((step - 1) * 2 + 1) % len(encoded_corpus)
        seqs = [encoded_corpus[idx1], encoded_corpus[idx2]]
        max_l = min(max(len(s) for s in seqs), context_length)

        batch_ids = torch.full((2, max_l), tokenizer.PAD_ID, dtype=torch.long, device="cuda")
        for i, s in enumerate(seqs):
            trimmed = s[:max_l]
            batch_ids[i, : len(trimmed)] = torch.tensor(trimmed, dtype=torch.long, device="cuda")

        inputs = batch_ids[:, :-1]
        targets = batch_ids[:, 1:]

        logits = model.student.forward_lm(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=tokenizer.PAD_ID,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0 or step == 1 or step == steps:
            losses.append(float(loss.detach()))

    return losses


def evaluate_qualification_gates(
    metrics_6800: dict[str, Any],
    metrics_7000: dict[str, Any],
    metrics_7200: dict[str, Any],
) -> tuple[dict[str, bool], bool]:
    """Evaluates Binding Acceptance Gates 1-3 simultaneously across late checkpoints."""
    late_metrics = [metrics_6800, metrics_7000, metrics_7200]
    gate_1 = all(m["training"]["exact"] >= 0.99 for m in late_metrics)
    gate_2 = all(
        m["old_known"]["exact"] >= 0.95 and m["old_renamed"]["exact"] >= 0.95 for m in late_metrics
    )
    m_final = metrics_7200
    dev_fam_acc = m_final["development_familiar"]["exact"]
    dev_nov_acc = m_final["development_novel"]["exact"]
    gate_3_acc = dev_fam_acc >= 0.85 and dev_nov_acc >= 0.85

    recalls_fam = m_final["development_familiar"].get("state_recall", {})
    recalls_nov = m_final["development_novel"].get("state_recall", {})
    gate_3_recall = (
        (
            all(recalls_fam.get(s, 0.0) >= 0.80 for s in ("positive", "negated", "absent"))
            and all(recalls_nov.get(s, 0.0) >= 0.80 for s in ("positive", "negated", "absent"))
        )
        if recalls_fam and recalls_nov
        else False
    )

    gate_3 = bool(gate_3_acc and gate_3_recall)
    gates = {
        "gate_1_stable_expanded_acquisition": bool(gate_1),
        "gate_2_prior_frame_retention": bool(gate_2),
        "gate_3_unfamiliar_grammar_generalization": bool(gate_3),
    }
    all_pass = bool(gate_1 and gate_2 and gate_3)
    return gates, all_pass


def train_arm(
    owner: LearningState,
    arm_name: str,
    steps: int,
    plan: list[list[int]],
    batch: tuple[torch.Tensor, ...],
    eval_sets: dict[str, Any],
    eval_batches: dict[str, Any],
    out_dir: Path,
) -> tuple[dict[str, Any], float]:
    """Trains one arm for given steps, evaluating at [6800, 7000, 7200]."""
    optimizer, scheduler, clipper = build_a9_optimizer_and_clipper(owner.model, arm_name, steps)
    owner.optimizer = optimizer
    owner.scheduler = scheduler

    snapshots: dict[str, Any] = {}
    t0 = time.time()

    for step in range(1, steps + 1):
        cursor = (step - 1) % len(plan)
        ix = plan[cursor]
        owner.step(
            *[v[ix] for v in batch],
            attribute_ids=TAG_IDS,
            clipper=clipper,
        )

        if step in (6800, 7000, 7200):
            snap_metrics = {
                k: enrich(evaluate(owner.model, eval_batches[k], rs))
                for k, rs in eval_sets.items()
            }
            snapshots[str(step)] = {k: compact(v) for k, v in snap_metrics.items()}
            owner.save(out_dir / f"{arm_name.lower()}_{step}.pt")

    wall_time = time.time() - t0
    return snapshots, wall_time


def run_stage_a9(root_path: Path, out_dir: Path, phase: str = "all") -> dict[str, Any]:
    require_remote_cuda()
    if file_hash(root_path) != ROOT_HASH:
        raise ValueError(
            f"Parent checkpoint does not match ROOT_HASH: {file_hash(root_path)} != {ROOT_HASH}"
        )

    parent = LearningState.load(root_path)
    if digest(parent.model.state_dict()) != ROOT_TENSOR_HASH:
        raise ValueError("Parent tensor state digest does not match ROOT_TENSOR_HASH")

    training, newcounts, plans, shared = prepare_a9_data()
    spec = json.loads(SPEC_A9.read_text())

    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tokenizer = ByteTextTokenizerM103()
    context = parent.model.student.base_model.config.context_length

    batches = {}
    sequences = {}
    shared_batches = {}
    for k, rs in training.items():
        batches[k], sequences[k] = encode(rs, tokenizer, context)
    for k, rs in shared.items():
        shared_batches[k], sequences[k] = encode(rs, tokenizer, context)

    eval_sets = {
        "training": training["coverage"][:756],
        "synthetic": training["coverage"][756:],
        **shared,
    }
    eval_batches = {
        "training": tuple(v[:756] for v in batches["coverage"]),
        "synthetic": tuple(v[756:] for v in batches["coverage"]),
        **shared_batches,
    }

    results: dict[str, Any] = {
        "stage": "Stage A.9",
        "git_sha": source_sha,
        "platform": require_remote_cuda(),
        "parent_sha256": ROOT_HASH,
        "parent_tensor_digest": ROOT_TENSOR_HASH,
        "experiments": {},
    }

    plan = plans["coverage"]
    batch = batches["coverage"]
    steps = spec["steps"]

    # -------------------------------------------------------------
    # EXPERIMENT 1 (A9.1): Full-Backbone Plasticity vs Top-2 Blocks
    # -------------------------------------------------------------
    if phase in ("all", "exp1"):
        print("=== Running Experiment 1 (A9.1): Plasticity Contrast ===")
        exp1_results: dict[str, Any] = {"arms": {}}
        exp1_models: dict[str, LearningState] = {}

        # Arm 1: EXP1_A_TOP2_REPRO
        print("--- Arm 1: EXP1_A_TOP2_REPRO ---")
        owner_a = fork_owner(parent, 42)
        owner_a.set_mode("selective_top2_blocks")
        snaps_a, wall_a = train_arm(owner_a, "EXP1_A_TOP2_REPRO", steps, plan, batch, eval_sets, eval_batches, out_dir)
        exp1_results["arms"]["EXP1_A_TOP2_REPRO"] = {
            "mode": "selective_top2_blocks",
            "wall_time_seconds": wall_a,
            "snapshots": snaps_a,
        }
        exp1_models["EXP1_A_TOP2_REPRO"] = owner_a

        # Arm 2: EXP1_B_FULL_BACKBONE_UNIFORM
        print("--- Arm 2: EXP1_B_FULL_BACKBONE_UNIFORM ---")
        owner_b = fork_owner(parent, 42)
        owner_b.set_mode("trainable_backbone")
        snaps_b, wall_b = train_arm(owner_b, "EXP1_B_FULL_BACKBONE_UNIFORM", steps, plan, batch, eval_sets, eval_batches, out_dir)
        exp1_results["arms"]["EXP1_B_FULL_BACKBONE_UNIFORM"] = {
            "mode": "trainable_backbone",
            "wall_time_seconds": wall_b,
            "snapshots": snaps_b,
        }
        exp1_models["EXP1_B_FULL_BACKBONE_UNIFORM"] = owner_b

        # Arm 3: EXP1_C_FULL_BACKBONE_LAYERWISE
        print("--- Arm 3: EXP1_C_FULL_BACKBONE_LAYERWISE ---")
        owner_c = fork_owner(parent, 42)
        owner_c.set_mode("trainable_backbone")
        snaps_c, wall_c = train_arm(owner_c, "EXP1_C_FULL_BACKBONE_LAYERWISE", steps, plan, batch, eval_sets, eval_batches, out_dir)
        exp1_results["arms"]["EXP1_C_FULL_BACKBONE_LAYERWISE"] = {
            "mode": "trainable_backbone",
            "wall_time_seconds": wall_c,
            "snapshots": snaps_c,
        }
        exp1_models["EXP1_C_FULL_BACKBONE_LAYERWISE"] = owner_c

        # Linear Probe Readouts on EXP1_B and EXP1_C
        print("--- Extracting Frozen Representations and Probes for EXP1_B & EXP1_C ---")
        probe_indices_rax, probe_indices_vim = coverage_matched_probe_indices(training["coverage"])
        exp1_results["probe_splits"] = {
            "probe_train_cases": len(probe_indices_rax),
            "familiar_val_cases": len(probe_indices_vim),
        }
        probes_eval: dict[str, Any] = {}
        for m_name, m_owner in (("EXP1_B", owner_b), ("EXP1_C", owner_c)):
            reps_train, _ = extract_frozen_representations(m_owner.model, batches["coverage"], training["coverage"])
            reps_nov, _ = extract_frozen_representations(m_owner.model, shared_batches["development_novel"], shared["development_novel"])
            
            p_res: dict[str, Any] = {}
            for rep_k in ("h_pool", "z_concept"):
                x_tr = torch.tensor(reps_train[rep_k], device="cuda", dtype=torch.float32)
                y_tr = torch.tensor([r.labels for r in training["coverage"]], device="cuda", dtype=torch.float32)
                x_nov = torch.tensor(reps_nov[rep_k], device="cuda", dtype=torch.float32)
                y_nov = torch.tensor([r.labels for r in shared["development_novel"]], device="cuda", dtype=torch.float32)
                
                x_all = torch.cat([x_tr, x_nov], dim=0)
                y_all = torch.cat([y_tr, y_nov], dim=0)
                preds, _fit = solve_ridge(x_all, y_all, probe_indices_rax, regularization=1.0)
                pred_nov_binary = (preds[len(x_tr):] > 0).long()
                exact = float((pred_nov_binary == y_nov.long()).all(dim=-1).float().mean().item())
                p_res[rep_k] = {"development_novel_exact": exact}
            probes_eval[m_name] = p_res

        exp1_results["linear_probes"] = probes_eval
        results["experiments"]["exp1_plasticity"] = exp1_results

    # -------------------------------------------------------------
    # EXPERIMENT 2 (A9.2): Absolute Position (wpe) vs RoPE
    # -------------------------------------------------------------
    if phase in ("all", "exp2"):
        print("=== Running Experiment 2 (A9.2): Position Encoding Contrast ===")
        exp2_results: dict[str, Any] = {"arms": {}, "positional_sensitivity": {}}

        # Arm 1: EXP2_A_FRESH_ABS_TOP2
        print("--- Arm 1: EXP2_A_FRESH_ABS_TOP2 ---")
        seed_all(42)
        fresh_student_a = TextStudentM103(prune_orphaned_lm_head=True, pos_encoding="absolute")
        model_fresh_a = IntegratedStudentM104_E(fresh_student_a, mode="selective_top2_blocks").cuda()
        owner_2a = LearningState(model_fresh_a)
        snaps_2a, wall_2a = train_arm(owner_2a, "EXP2_A_FRESH_ABS_TOP2", steps, plan, batch, eval_sets, eval_batches, out_dir)
        exp2_results["arms"]["EXP2_A_FRESH_ABS_TOP2"] = {
            "wall_time_seconds": wall_2a,
            "snapshots": snaps_2a,
        }

        # Arm 2: EXP2_B_FRESH_ABS_FULL
        print("--- Arm 2: EXP2_B_FRESH_ABS_FULL ---")
        seed_all(42)
        fresh_student_b = TextStudentM103(prune_orphaned_lm_head=True, pos_encoding="absolute")
        model_fresh_b = IntegratedStudentM104_E(fresh_student_b, mode="trainable_backbone").cuda()
        owner_2b = LearningState(model_fresh_b)
        snaps_2b, wall_2b = train_arm(owner_2b, "EXP2_B_FRESH_ABS_FULL", steps, plan, batch, eval_sets, eval_batches, out_dir)
        exp2_results["arms"]["EXP2_B_FRESH_ABS_FULL"] = {
            "wall_time_seconds": wall_2b,
            "snapshots": snaps_2b,
        }

        # Arm 3: EXP2_C_FRESH_ROPE_FULL
        print("--- Arm 3: EXP2_C_FRESH_ROPE_FULL ---")
        seed_all(42)
        fresh_student_c = TextStudentM103(prune_orphaned_lm_head=True, pos_encoding="rope")
        model_fresh_c = IntegratedStudentM104_E(fresh_student_c, mode="trainable_backbone").cuda()
        owner_2c = LearningState(model_fresh_c)
        snaps_2c, wall_2c = train_arm(owner_2c, "EXP2_C_FRESH_ROPE_FULL", steps, plan, batch, eval_sets, eval_batches, out_dir)
        exp2_results["arms"]["EXP2_C_FRESH_ROPE_FULL"] = {
            "wall_time_seconds": wall_2c,
            "snapshots": snaps_2c,
        }

        # Arm 4: EXP2_D_CONTINUING_ROPE_MIGRATION
        print("--- Arm 4: EXP2_D_CONTINUING_ROPE_MIGRATION ---")
        owner_2d = fork_owner(parent, 42)
        owner_2d.set_mode("trainable_backbone")
        # Enable RoPE migration on parent weights
        owner_2d.model.student.base_model.config.pos_encoding = "rope"
        if owner_2d.model.student.base_model.wpe is not None:
            owner_2d.model.student.base_model.wpe.weight.data.zero_()
            owner_2d.model.student.base_model.wpe.weight.requires_grad = False
        for block in owner_2d.model.student.base_model.blocks:
            block.attn.pos_encoding = "rope"
            if not hasattr(block.attn, "inv_freq"):
                inv_freq = 1.0 / (10000.0 ** (torch.arange(0, block.attn.head_dim, 2).float() / block.attn.head_dim))
                block.attn.register_buffer("inv_freq", inv_freq, persistent=False)

        # Measure t0 behavioral drift
        t0_known = enrich(evaluate(owner_2d.model, shared_batches["old_known"], shared["old_known"]))["exact"]
        t0_renamed = enrich(evaluate(owner_2d.model, shared_batches["old_renamed"], shared["old_renamed"]))["exact"]
        print(f"EXP2_D t0 old competence under raw RoPE migration: known={t0_known:.4f}, renamed={t0_renamed:.4f}")

        snaps_2d, wall_2d = train_arm(owner_2d, "EXP2_D_CONTINUING_ROPE_MIGRATION", steps, plan, batch, eval_sets, eval_batches, out_dir)
        exp2_results["arms"]["EXP2_D_CONTINUING_ROPE_MIGRATION"] = {
            "t0_old_competence": {"old_known": t0_known, "old_renamed": t0_renamed},
            "wall_time_seconds": wall_2d,
            "snapshots": snaps_2d,
        }

        # Measure Positional Sensitivity
        print("--- Measuring Positional Sensitivity (+8 and +24 shifts) ---")
        sens_models = {
            "EXP2_B_FRESH_ABS_FULL": owner_2b.model,
            "EXP2_C_FRESH_ROPE_FULL": owner_2c.model,
            "EXP2_D_CONTINUING_ROPE": owner_2d.model,
        }
        for s_name, s_model in sens_models.items():
            shifts = evaluate_positional_sensitivity(s_model, shared_batches["development_novel"], shared["development_novel"])
            exp2_results["positional_sensitivity"][s_name] = shifts
            print(f"  {s_name:25s} -> shifts: {shifts}")

        results["experiments"]["exp2_position_encoding"] = exp2_results

    # -------------------------------------------------------------
    # EXPERIMENT 3 (A9.3): Syntactic Language Pretraining (Conditional)
    # -------------------------------------------------------------
    # Check trigger condition: did any arm in Exp 1 or Exp 2 achieve Gate 3 (>= 85.0%)?
    best_dev_acc = 0.0
    for exp_k in ("exp1_plasticity", "exp2_position_encoding"):
        if exp_k in results["experiments"]:
            for _arm_k, arm_d in results["experiments"][exp_k]["arms"].items():
                nov_7200 = arm_d["snapshots"]["7200"]["development_novel"]["exact"]
                if nov_7200 > best_dev_acc:
                    best_dev_acc = nov_7200

    print(f"Highest development novel accuracy across Exp 1 & 2: {best_dev_acc*100:.2f}% (Threshold: 85.0%)")

    if phase in ("all", "exp3") and (best_dev_acc < 0.85 or phase == "exp3"):
        print("=== Trigger Condition Met: Running Experiment 3 (A9.3) Syntactic Pretraining ===")
        pretrain_spec = json.loads(SPEC_PRETRAIN.read_text())
        corpus = generate_syntactic_pretraining_corpus(pretrain_spec["num_sentences"], seed=pretrain_spec["seed"])
        iso = verify_pretraining_corpus_isolation(corpus)
        assert iso["isolation_verified"]

        exp3_results: dict[str, Any] = {
            "corpus_isolation": iso,
            "arms": {},
        }

        # Arm 1: EXP3_A_NO_PRETRAIN (Matched architecture: fresh RoPE full backbone)
        print("--- Arm 1: EXP3_A_NO_PRETRAIN ---")
        seed_all(42)
        student_3a = TextStudentM103(prune_orphaned_lm_head=True, pos_encoding="rope")
        model_3a = IntegratedStudentM104_E(student_3a, mode="trainable_backbone").cuda()
        owner_3a = LearningState(model_3a)
        snaps_3a, wall_3a = train_arm(owner_3a, "EXP3_A_NO_PRETRAIN", steps, plan, batch, eval_sets, eval_batches, out_dir)
        exp3_results["arms"]["EXP3_A_NO_PRETRAIN"] = {
            "wall_time_seconds": wall_3a,
            "snapshots": snaps_3a,
        }

        # Arm 2: EXP3_B_SYNTAX_PRETRAINED
        print("--- Arm 2: EXP3_B_SYNTAX_PRETRAINED ---")
        seed_all(42)
        student_3b = TextStudentM103(prune_orphaned_lm_head=True, pos_encoding="rope")
        model_3b = IntegratedStudentM104_E(student_3b, mode="trainable_backbone").cuda()
        t_pre_0 = time.time()
        pretrain_losses = run_language_pretraining(
            model_3b, corpus, tokenizer, steps=pretrain_spec["pretrain_steps"], lr=pretrain_spec["lr"]
        )
        pretrain_time = time.time() - t_pre_0
        print(f"Pretraining completed in {pretrain_time:.1f}s. Initial loss: {pretrain_losses[0]:.4f}, Final loss: {pretrain_losses[-1]:.4f}")

        owner_3b = LearningState(model_3b)
        snaps_3b, wall_3b = train_arm(owner_3b, "EXP3_B_SYNTAX_PRETRAINED", steps, plan, batch, eval_sets, eval_batches, out_dir)
        exp3_results["arms"]["EXP3_B_SYNTAX_PRETRAINED"] = {
            "pretrain_losses": pretrain_losses,
            "pretrain_time_seconds": pretrain_time,
            "wall_time_seconds": wall_3b,
            "total_time_seconds": pretrain_time + wall_3b,
            "snapshots": snaps_3b,
        }

        results["experiments"]["exp3_language_pretraining"] = exp3_results

    # -------------------------------------------------------------
    # QUALIFICATION GATES & SCIENTIFIC DECISION MATRIX
    # -------------------------------------------------------------
    print("=== Evaluating Stage A.9 Qualification Gates & Hypothesis Matrix ===")
    all_arms_summary: dict[str, Any] = {}
    highest_dev_arm = ""
    highest_dev_val = 0.0

    for _exp_k, exp_d in results["experiments"].items():
        for arm_k, arm_d in exp_d["arms"].items():
            snaps = arm_d["snapshots"]
            gates, passes = evaluate_qualification_gates(snaps["6800"], snaps["7000"], snaps["7200"])
            dev_nov = snaps["7200"]["development_novel"]["exact"]
            all_arms_summary[arm_k] = {
                "gates": gates,
                "all_gates_pass": passes,
                "dev_novel_7200": dev_nov,
            }
            if dev_nov > highest_dev_val:
                highest_dev_val = dev_nov
                highest_dev_arm = arm_k

    # Determine Hypothesis Verdict
    # H1: Exp 1 achieves >= 85% on novel grammar
    exp1_pass = any(all_arms_summary.get(k, {}).get("all_gates_pass") for k in ("EXP1_B_FULL_BACKBONE_UNIFORM", "EXP1_C_FULL_BACKBONE_LAYERWISE"))
    # H2: RoPE achieves >= 85% on novel grammar
    exp2_pass = all_arms_summary.get("EXP2_C_FRESH_ROPE_FULL", {}).get("all_gates_pass", False)
    # H3: Language pretraining achieves >= 85% on novel grammar
    exp3_pass = all_arms_summary.get("EXP3_B_SYNTAX_PRETRAINED", {}).get("all_gates_pass", False)

    if exp1_pass:
        verdict = "H1_FULL_PLASTICITY_SUFFICIENT"
    elif exp2_pass:
        verdict = "H2_POSITION_ENCODING_MATERIAL"
    elif exp3_pass:
        verdict = "H3_LANGUAGE_PRETRAINING_REQUIRED"
    else:
        verdict = "H4_NONE_ESTABLISHED_POOLING_OR_TOKEN_BOTTLENECK"

    print(f"Scientific Hypothesis Verdict: {verdict}")

    # Resume-check verification on the top arm checkpoint
    top_ckpt = out_dir / f"{highest_dev_arm.lower()}_7200.pt"
    if not top_ckpt.exists():
        top_ckpt = list(out_dir.glob("*_7200.pt"))[0]

    resume_child_cmd = [
        sys.executable,
        __file__,
        "--resume-check",
        str(top_ckpt),
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(resume_child_cmd, check=True, timeout=120)
    expected_resume = json.loads((out_dir / "resume_child_a9.json").read_text())

    reloaded_owner = LearningState.load(top_ckpt)
    pred_reload = evaluate(reloaded_owner.model, batches["coverage"], training["coverage"])
    cursor = 7200 % len(plan)
    ix = plan[cursor]
    reloaded_owner.step(*[v[ix] for v in batch], attribute_ids=TAG_IDS)

    logit_diff = max(
        abs(a - b)
        for row_a, row_b in zip(
            pred_reload["logits"], expected_resume["predictions"]["logits"], strict=True
        )
        for a, b in zip(row_a, row_b, strict=True)
    )
    gate_4 = bool(logit_diff == 0.0)

    results["qualification"] = {
        "arms_summary": all_arms_summary,
        "highest_dev_arm": highest_dev_arm,
        "highest_dev_accuracy": highest_dev_val,
        "gate_4_continuation_passed": gate_4,
        "max_logit_diff": logit_diff,
        "hypothesis_verdict": verdict,
    }

    (out_dir / "stage_a9_final_manifest.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--resume-check", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase", choices=["all", "exp1", "exp2", "exp3"], default="all")
    args = parser.parse_args()
    require_remote_cuda()

    if args.resume_check:
        owner = LearningState.load(args.resume_check)
        training, _, plans, _ = prepare_a9_data()
        tokenizer = ByteTextTokenizerM103()
        context = owner.model.student.base_model.config.context_length
        batch, _ = encode(training["coverage"], tokenizer, context)
        pred = evaluate(owner.model, batch, training["coverage"])
        plan = plans["coverage"]
        cursor = 7200 % len(plan)
        ix = plan[cursor]
        loss = owner.step(*[v[ix] for v in batch], attribute_ids=TAG_IDS)
        child_info = {
            "predictions": pred,
            "loss": loss,
            "signature": signature(owner),
            "steps": owner.metadata["steps"],
        }
        (args.output_dir / "resume_child_a9.json").write_text(json.dumps(child_info))
        return

    if not args.root:
        raise ValueError("--root is required when not in resume-check mode")

    run_stage_a9(args.root, args.output_dir, phase=args.phase)


if __name__ == "__main__":
    main()
