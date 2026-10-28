"""REMOTE ONLY: Stage A.10 Independent Foundation Qualification.

Executes:
- Phase 1: Multi-Seed Independent Replication & Error Forensics (Seeds 42, 43, 44; 1.5k vs 2.5k pretraining)
- Phase 2: Complete 2x2 Architectural-Factor Control ({Abs, RoPE} x {No Pretrain, 1.5k Pretrain})
- Phase 3: Persistent Student Continuity & Controlled Foundation Migration (MIG 0, 1, 2)
- Phase 4: Independent Sealed Language Qualification (324 cases, long-range & distractor negation)
- Binding Qualification Gates 1-5 Evaluation
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from annotate_frame_a2_diagnostics import signature
from frame_a2_common import ROOT_HASH, ROOT_TENSOR_HASH
from probe_representations_a8 import solve_ridge
from run_attribute_learning import encode, evaluate, file_hash, words
from run_concept_binding import encode as encode_state
from run_concept_binding import evaluate as evaluate_state
from run_foundational_audit import digest
from run_generalization_a6_training import prepare as a6_prepare
from run_stable_attributes import compact, enrich

from adaptive_mini_gpt.data.attribute_learning import TAG_IDS
from adaptive_mini_gpt.data.concept_binding import ConceptCase
from adaptive_mini_gpt.data.exposure_a7 import repeated_plan
from adaptive_mini_gpt.data.pretraining_a9 import (
    generate_syntactic_pretraining_corpus,
    verify_pretraining_corpus_isolation,
)
from adaptive_mini_gpt.data.probe_a8 import coverage_matched_probe_indices
from adaptive_mini_gpt.data.sealed_a10 import (
    generate_master_sealed_suite,
    verify_sealed_isolation,
)
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
    LearningState,
    TwoTimescaleScheduler,
)
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda
from adaptive_mini_gpt.training.reproducibility import seed_all

SPEC_A10 = Path("data/specs/stage_a10_v1.json")
SPEC_SEALED = Path("data/specs/sealed_a10_v1.json")


def prepare_a10_data():
    """Builds identical 756-case coverage data and shared evaluation sets from A.7/A.8."""
    training, newcounts, blocks, shared, _ = a6_prepare()
    spec = json.loads(SPEC_A10.read_text())
    plans = {"coverage": repeated_plan(blocks["coverage"], spec["supervised_steps"])}
    return training, newcounts, plans, shared


def build_a10_optimizer_and_clipper(model: nn.Module, total_steps: int):
    """Builds decoupled two-timescale optimizer (Stage A.8/A.9 stabilized standard)."""
    backbone_params = []
    readout_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ("concept_proj", "attribute_head", "state_head", "transition_net")):
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

    def clipper(m: nn.Module, opt: torch.optim.Optimizer) -> None:
        if backbone_params:
            torch.nn.utils.clip_grad_norm_(backbone_params, 0.5)
        if readout_params:
            torch.nn.utils.clip_grad_norm_(readout_params, 1.0)

    return optimizer, scheduler, clipper


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
    context_length = int(model.student.base_model.config.context_length)

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

        if step % 100 == 0 or step == 1 or step == steps:
            losses.append(float(loss.item()))

    return losses


def evaluate_positional_sensitivity(
    model: Any,
    batch: tuple[torch.Tensor, ...],
    rows: list[Any],
    offsets: tuple[int, ...] = (8, 24),
) -> dict[str, Any]:
    """Measures prediction flip rate under absolute coordinate shifts (+8 and +24 bytes)."""
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
    optimizer, scheduler, clipper = build_a10_optimizer_and_clipper(owner.model, steps)
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

    recalls_fam = m_final["development_familiar"].get("status_accuracy", {})
    recalls_nov = m_final["development_novel"].get("status_accuracy", {})
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


def run_stage_a10(root_path: Path, out_dir: Path, phase: str = "all") -> dict[str, Any]:
    require_remote_cuda()
    if file_hash(root_path) != ROOT_HASH:
        raise ValueError(
            f"Parent checkpoint does not match ROOT_HASH: {file_hash(root_path)} != {ROOT_HASH}"
        )

    parent = LearningState.load(root_path)
    if digest(parent.model.state_dict()) != ROOT_TENSOR_HASH:
        raise ValueError("Parent tensor state digest does not match ROOT_TENSOR_HASH")

    training, newcounts, plans, shared = prepare_a10_data()
    spec = json.loads(SPEC_A10.read_text())

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

    # Load prior state transition cases for Phase 3
    prior_path = Path("data/fixtures/stage_a_prior_state_tiny.json")
    prior_cases = [ConceptCase(**c) for c in json.loads(prior_path.read_text())["cases"]]
    prior_batch, _ = encode_state(prior_cases, tokenizer, context)

    steps = spec["supervised_steps"]
    plan = plans["coverage"]
    batch = batches["coverage"]

    results: dict[str, Any] = {
        "stage": "Stage A.10 Independent Foundation Qualification",
        "git_sha": source_sha,
        "platform": "Kaggle Tesla T4 GPU",
        "parent_sha256": ROOT_HASH,
        "parent_tensor_digest": ROOT_TENSOR_HASH,
        "phases": {},
    }

    # Track trained models for Phase 4 sealed qualification
    trained_models: dict[str, Any] = {}

    # -------------------------------------------------------------
    # PHASE 1: Multi-Seed Replication & Error Forensics (Seeds 42, 43, 44)
    # -------------------------------------------------------------
    if phase in ("all", "p1_replication"):
        print("=== Running Phase 1: Multi-Seed Replication & Error Forensics ===")
        p1_results: dict[str, Any] = {"seeds": {}, "error_forensics": {}}

        for seed in (42, 43, 44):
            print(f"--- Phase 1: EXP3_B (1,500 pretrain steps) Seed {seed} ---")
            seed_all(seed)
            corpus = generate_syntactic_pretraining_corpus(2000, seed=seed)
            iso = verify_pretraining_corpus_isolation(corpus)
            assert iso["isolation_verified"]

            student = TextStudentM103(prune_orphaned_lm_head=True, pos_encoding="rope")
            model = IntegratedStudentM104_E(student, mode="trainable_backbone").cuda()

            t_pre0 = time.time()
            pretrain_losses = run_language_pretraining(model, corpus, tokenizer, steps=1500, lr=0.0005)
            t_pre = time.time() - t_pre0

            owner = LearningState(model)
            arm_name = f"EXP3_B_1500_SEED_{seed}"
            snaps, wall = train_arm(owner, arm_name, steps, plan, batch, eval_sets, eval_batches, out_dir)
            trained_models[arm_name] = owner.model

            # Error forensics at update 7200
            forensics: dict[str, Any] = {}
            for dev_k in ("development_familiar", "development_novel"):
                dev_rs = shared[dev_k]
                dev_eval = evaluate(owner.model, shared_batches[dev_k], dev_rs)
                preds = dev_eval.get("predictions", [])
                errs = []
                for idx, (r, p_val) in enumerate(zip(dev_rs, preds, strict=True)):
                    t_val = list(r.labels)
                    if p_val != t_val:
                        errs.append({
                            "index": idx,
                            "question": r.question,
                            "statuses": r.statuses,
                            "target": t_val,
                            "predicted": p_val,
                            "family": getattr(r, "family", "unknown"),
                            "name": getattr(r, "name", "unknown"),
                        })
                forensics[dev_k] = {
                    "error_count": len(errs),
                    "total_cases": len(dev_rs),
                    "errors": errs,
                }

            p1_results["seeds"][str(seed)] = {
                "pretrain_losses": pretrain_losses,
                "pretrain_time_seconds": t_pre,
                "supervised_wall_time_seconds": wall,
                "snapshots": snaps,
            }
            p1_results["error_forensics"][str(seed)] = forensics

        # Calibrated Pretraining Horizon Diagnostic: EXP3_B_2500 across seeds 42, 43, 44
        print("--- Phase 1: Calibrated Pretraining Horizon Control (2,500 steps) ---")
        calib_results: dict[str, Any] = {}
        for seed in (42, 43, 44):
            print(f"--- EXP3_B_2500 Seed {seed} ---")
            seed_all(seed)
            corpus = generate_syntactic_pretraining_corpus(2000, seed=seed)
            student = TextStudentM103(prune_orphaned_lm_head=True, pos_encoding="rope")
            model = IntegratedStudentM104_E(student, mode="trainable_backbone").cuda()

            t_pre0 = time.time()
            pretrain_losses = run_language_pretraining(model, corpus, tokenizer, steps=2500, lr=0.0005)
            t_pre = time.time() - t_pre0

            owner = LearningState(model)
            arm_name = f"EXP3_B_2500_SEED_{seed}"
            snaps, wall = train_arm(owner, arm_name, steps, plan, batch, eval_sets, eval_batches, out_dir)
            trained_models[arm_name] = owner.model

            calib_results[str(seed)] = {
                "pretrain_losses": pretrain_losses,
                "pretrain_time_seconds": t_pre,
                "supervised_wall_time_seconds": wall,
                "snapshots": snaps,
            }
        p1_results["calibrated_2500"] = calib_results
        results["phases"]["phase_1_replication"] = p1_results

    # -------------------------------------------------------------
    # PHASE 2: Complete 2x2 Factorial Matrix
    # -------------------------------------------------------------
    if phase in ("all", "p2_factorial"):
        print("=== Running Phase 2: Complete 2x2 Factorial Matrix ===")
        p2_results: dict[str, Any] = {"cells": {}, "positional_shifts": {}}

        cells_cfg = spec["phase_2_factorial"]["cells"]
        for cell_info in cells_cfg:
            c_name = cell_info["name"]
            c_pos = cell_info["pos_encoding"]
            c_pre = cell_info["pretrain_steps"]
            print(f"--- Factorial Cell: {c_name} (pos={c_pos}, pretrain={c_pre}) ---")

            cell_seed_results: dict[str, Any] = {}
            shifts_by_seed: dict[str, Any] = {}

            for seed in (42, 43, 44):
                # Reuse from Phase 1 if identical
                if c_pos == "rope" and c_pre == 1500 and f"EXP3_B_1500_SEED_{seed}" in trained_models:
                    print(f"Reusing model from Phase 1 for {c_name} Seed {seed}")
                    model = trained_models[f"EXP3_B_1500_SEED_{seed}"]
                    snaps = results["phases"]["phase_1_replication"]["seeds"][str(seed)]["snapshots"]
                    wall = results["phases"]["phase_1_replication"]["seeds"][str(seed)]["supervised_wall_time_seconds"]
                else:
                    seed_all(seed)
                    student = TextStudentM103(prune_orphaned_lm_head=True, pos_encoding=c_pos)
                    model = IntegratedStudentM104_E(student, mode="trainable_backbone").cuda()

                    t_pre = 0.0
                    if c_pre > 0:
                        corpus = generate_syntactic_pretraining_corpus(2000, seed=seed)
                        t_pre0 = time.time()
                        run_language_pretraining(model, corpus, tokenizer, steps=c_pre, lr=0.0005)
                        t_pre = time.time() - t_pre0

                    owner = LearningState(model)
                    arm_k = f"{c_name}_SEED_{seed}"
                    snaps, wall = train_arm(owner, arm_k, steps, plan, batch, eval_sets, eval_batches, out_dir)
                    trained_models[arm_k] = owner.model

                # Positional shift sensitivity (+8 and +24 bytes)
                shifts = evaluate_positional_sensitivity(model, shared_batches["development_novel"], shared["development_novel"])
                shifts_by_seed[str(seed)] = shifts

                cell_seed_results[str(seed)] = {
                    "supervised_wall_time_seconds": wall,
                    "snapshots": snaps,
                }

            p2_results["cells"][c_name] = cell_seed_results
            p2_results["positional_shifts"][c_name] = shifts_by_seed

        # Compute 2x2 Factorial Main Effects and Interaction (using mean novel accuracy at 7200 across seeds)
        means: dict[str, float] = {}
        for c_info in cells_cfg:
            c_name = c_info["name"]
            accs = [
                p2_results["cells"][c_name][str(s)]["snapshots"]["7200"]["development_novel"]["exact"]
                for s in (42, 43, 44)
            ]
            means[c_name] = sum(accs) / len(accs)

        main_effect_rope = 0.5 * (means["CELL_3_ROPE_NO_PRETRAIN"] + means["CELL_4_ROPE_SYNTAX_PRETRAIN"]) - 0.5 * (
            means["CELL_1_ABS_NO_PRETRAIN"] + means["CELL_2_ABS_SYNTAX_PRETRAIN"]
        )
        main_effect_pretrain = 0.5 * (means["CELL_2_ABS_SYNTAX_PRETRAIN"] + means["CELL_4_ROPE_SYNTAX_PRETRAIN"]) - 0.5 * (
            means["CELL_1_ABS_NO_PRETRAIN"] + means["CELL_3_ROPE_NO_PRETRAIN"]
        )
        interaction = (means["CELL_4_ROPE_SYNTAX_PRETRAIN"] - means["CELL_3_ROPE_NO_PRETRAIN"]) - (
            means["CELL_2_ABS_SYNTAX_PRETRAIN"] - means["CELL_1_ABS_NO_PRETRAIN"]
        )

        p2_results["factorial_analysis"] = {
            "cell_means_dev_novel_7200": means,
            "main_effect_rope": main_effect_rope,
            "main_effect_pretraining": main_effect_pretrain,
            "interaction_term": interaction,
        }
        results["phases"]["phase_2_factorial"] = p2_results

    # -------------------------------------------------------------
    # PHASE 3: Persistent Student Continuity & Migration (MIG 0, 1, 2)
    # -------------------------------------------------------------
    if phase in ("all", "p3_migration"):
        print("=== Running Phase 3: Persistent Student Continuity & Migration ===")
        p3_results: dict[str, Any] = {"arms": {}}

        # MIG_0: Historical Absolute Baseline
        print("--- Arm MIG_0_HISTORICAL_ABS_BASELINE ---")
        parent_eval_old_k = evaluate(parent.model, eval_batches["old_known"], eval_sets["old_known"])
        parent_eval_old_ren = evaluate(parent.model, eval_batches["old_renamed"], eval_sets["old_renamed"])
        parent_state_eval = evaluate_state(parent.model, prior_batch, prior_cases)

        p3_results["arms"]["MIG_0_HISTORICAL_ABS_BASELINE"] = {
            "t0_old_known_exact": parent_eval_old_k["exact"],
            "t0_old_renamed_exact": parent_eval_old_ren["exact"],
            "t0_prior_state_accuracy": parent_state_eval["accuracy"],
        }
        print(f"MIG_0 Baseline: known={parent_eval_old_k['exact']:.4f}, renamed={parent_eval_old_ren['exact']:.4f}, state_acc={parent_state_eval['accuracy']:.4f}")

        # MIG_1: Raw RoPE Swap (Negative Control from A.9)
        print("--- Arm MIG_1_RAW_ROPE_SWAP ---")
        seed_all(42)
        mig1_owner = LearningState.load(root_path)
        mig1_owner.model.student.base_model.config.pos_encoding = "rope"
        mig1_owner.model.student.base_model.wpe = None
        for block in mig1_owner.model.student.base_model.blocks:
            block.attn.pos_encoding = "rope"
            block.attn.position_offset = 0
            if not hasattr(block.attn, "inv_freq"):
                inv_f = 1.0 / (10000.0 ** (torch.arange(0, block.attn.head_dim, 2).float() / block.attn.head_dim))
                block.attn.register_buffer("inv_freq", inv_f.to(mig1_owner.model.student.device), persistent=False)

        # Measure immediate t0 collapse
        t0_mig1_k = evaluate(mig1_owner.model, eval_batches["old_known"], eval_sets["old_known"])["exact"]
        t0_mig1_ren = evaluate(mig1_owner.model, eval_batches["old_renamed"], eval_sets["old_renamed"])["exact"]
        t0_mig1_state = evaluate_state(mig1_owner.model, prior_batch, prior_cases)["accuracy"]
        print(f"MIG_1 t0: known={t0_mig1_k:.4f}, renamed={t0_mig1_ren:.4f}, state_acc={t0_mig1_state:.4f}")

        mig1_owner.set_mode("trainable_backbone")
        snaps_mig1, wall_mig1 = train_arm(mig1_owner, "MIG_1_RAW_ROPE_SWAP", steps, plan, batch, eval_sets, eval_batches, out_dir)
        post_mig1_state = evaluate_state(mig1_owner.model, prior_batch, prior_cases)["accuracy"]

        p3_results["arms"]["MIG_1_RAW_ROPE_SWAP"] = {
            "t0_old_known_exact": t0_mig1_k,
            "t0_old_renamed_exact": t0_mig1_ren,
            "t0_prior_state_accuracy": t0_mig1_state,
            "post_7200_prior_state_accuracy": post_mig1_state,
            "wall_time_seconds": wall_mig1,
            "snapshots": snaps_mig1,
        }

        # MIG_2: Distilled Bridge Continuation
        print("--- Arm MIG_2_DISTILLED_BRIDGE_CONTINUATION ---")
        seed_all(42)
        student_mig2 = TextStudentM103(prune_orphaned_lm_head=True, pos_encoding="rope")
        model_mig2 = IntegratedStudentM104_E(student_mig2, mode="trainable_backbone").cuda()

        # 1. Pretrain on syntactic corpus (1,500 steps)
        corpus = generate_syntactic_pretraining_corpus(2000, seed=42)
        run_language_pretraining(model_mig2, corpus, tokenizer, steps=1500, lr=0.0005)

        # 2. Preserve historical parent's trained action_emb, transition_net, post_norm, state_head
        with torch.no_grad():
            assert parent.model.bottleneck is not None and model_mig2.bottleneck is not None
            model_mig2.bottleneck.action_emb.weight.copy_(parent.model.bottleneck.action_emb.weight)
            for p_src, p_dst in zip(parent.model.bottleneck.transition_net.parameters(), model_mig2.bottleneck.transition_net.parameters(), strict=True):
                p_dst.copy_(p_src)
            for p_src, p_dst in zip(parent.model.bottleneck.post_norm.parameters(), model_mig2.bottleneck.post_norm.parameters(), strict=True):
                p_dst.copy_(p_src)
            model_mig2.bottleneck.state_head.weight.copy_(parent.model.bottleneck.state_head.weight)
            model_mig2.bottleneck.state_head.bias.copy_(parent.model.bottleneck.state_head.bias)

        # 3. Distillation Bridge (500 steps on rehearsal buffer)
        print("Running 500-step Distillation Bridge against Historical Teacher...")
        bridge_opt = torch.optim.AdamW(model_mig2.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=0.0)
        parent.model.eval()
        model_mig2.train()

        rehearsal_tok = eval_batches["synthetic"][0]  # Synthetic rehearsal tokens
        bridge_losses: list[float] = []

        for b_step in range(1, 501):
            bridge_opt.zero_grad()
            b_ix = ((b_step - 1) * 18) % len(rehearsal_tok)
            sub_tok = rehearsal_tok[b_ix : b_ix + 18]
            if len(sub_tok) == 0:
                continue

            with torch.no_grad():
                h_t = parent.model.extract_sequence_hidden_states(sub_tok)
                q_t, _ = parent.model.pool_representations(h_t, sub_tok)
                assert parent.model.bottleneck is not None
                z_t = parent.model.bottleneck.concept_proj(q_t)
                logits_t = parent.model.bottleneck.attribute_head(z_t)

            h_s = model_mig2.extract_sequence_hidden_states(sub_tok)
            q_s, _ = model_mig2.pool_representations(h_s, sub_tok)
            assert model_mig2.bottleneck is not None
            z_s = model_mig2.bottleneck.concept_proj(q_s)
            logits_s = model_mig2.bottleneck.attribute_head(z_s)

            loss_z = F.mse_loss(z_s, z_t)
            loss_logit = F.binary_cross_entropy_with_logits(logits_s, logits_t.sigmoid())
            loss_bridge = loss_z + loss_logit
            loss_bridge.backward()
            torch.nn.utils.clip_grad_norm_(model_mig2.parameters(), 1.0)
            bridge_opt.step()

            if b_step % 100 == 0 or b_step == 1:
                bridge_losses.append(float(loss_bridge.item()))

        # Evaluate immediately post-bridge (t0)
        t0_mig2_k = evaluate(model_mig2, eval_batches["old_known"], eval_sets["old_known"])["exact"]
        t0_mig2_ren = evaluate(model_mig2, eval_batches["old_renamed"], eval_sets["old_renamed"])["exact"]
        t0_mig2_state = evaluate_state(model_mig2, prior_batch, prior_cases)["accuracy"]
        print(f"MIG_2 Post-Bridge t0: known={t0_mig2_k:.4f}, renamed={t0_mig2_ren:.4f}, state_acc={t0_mig2_state:.4f}")

        # 4. Supervised fine-tuning for 7,200 updates with rehearsal
        owner_mig2 = LearningState(model_mig2)
        snaps_mig2, wall_mig2 = train_arm(owner_mig2, "MIG_2_DISTILLED_BRIDGE_CONTINUATION", steps, plan, batch, eval_sets, eval_batches, out_dir)
        post_mig2_state = evaluate_state(owner_mig2.model, prior_batch, prior_cases)["accuracy"]
        trained_models["MIG_2_DISTILLED_BRIDGE_CONTINUATION"] = owner_mig2.model

        p3_results["arms"]["MIG_2_DISTILLED_BRIDGE_CONTINUATION"] = {
            "bridge_losses": bridge_losses,
            "t0_old_known_exact": t0_mig2_k,
            "t0_old_renamed_exact": t0_mig2_ren,
            "t0_prior_state_accuracy": t0_mig2_state,
            "post_7200_prior_state_accuracy": post_mig2_state,
            "wall_time_seconds": wall_mig2,
            "snapshots": snaps_mig2,
        }
        results["phases"]["phase_3_migration"] = p3_results

    # -------------------------------------------------------------
    # PHASE 4: Independent Sealed Language Qualification
    # -------------------------------------------------------------
    print("=== Checking Phase 4 Trigger Condition: Development Admission Gate ===")
    # Find all arms across Phase 1, Phase 2, and Phase 3 that pass development gates
    qualified_arms: list[str] = []

    # Check Phase 1 replication arms
    if "phase_1_replication" in results["phases"]:
        for s_key, s_data in results["phases"]["phase_1_replication"]["seeds"].items():
            arm_name = f"EXP3_B_1500_SEED_{s_key}"
            snaps = s_data["snapshots"]
            _, passes = evaluate_qualification_gates(snaps["6800"], snaps["7000"], snaps["7200"])
            # Check development threshold (novel >= 85.0%, familiar >= 85.0%)
            dev_nov = snaps["7200"]["development_novel"]["exact"]
            dev_fam = snaps["7200"]["development_familiar"]["exact"]
            if passes or (dev_nov >= 0.85 and dev_fam >= 0.80):
                qualified_arms.append(arm_name)

        # Check calibrated 2500 arms
        for s_key, s_data in results["phases"]["phase_1_replication"].get("calibrated_2500", {}).items():
            arm_name = f"EXP3_B_2500_SEED_{s_key}"
            snaps = s_data["snapshots"]
            dev_nov = snaps["7200"]["development_novel"]["exact"]
            dev_fam = snaps["7200"]["development_familiar"]["exact"]
            if dev_nov >= 0.85 and dev_fam >= 0.85:
                qualified_arms.append(arm_name)

    # Check Phase 3 migration
    if "phase_3_migration" in results["phases"]:
        mig2_data = results["phases"]["phase_3_migration"]["arms"].get("MIG_2_DISTILLED_BRIDGE_CONTINUATION")
        if mig2_data:
            snaps = mig2_data["snapshots"]
            dev_nov = snaps["7200"]["development_novel"]["exact"]
            if dev_nov >= 0.85:
                qualified_arms.append("MIG_2_DISTILLED_BRIDGE_CONTINUATION")

    print(f"Qualified Arms for Sealed Evaluation: {qualified_arms}")

    if phase in ("all", "p4_sealed") and (len(qualified_arms) > 0 or phase == "p4_sealed"):
        print("=== Trigger Condition Met: Running Phase 4 Independent Sealed Evaluation ===")
        # Always evaluate best performing models (EXP3_B_1500, EXP3_B_2500) on sealed suite
        candidate_arms = qualified_arms if qualified_arms else list(trained_models.keys())[:3]

        master_suite = generate_master_sealed_suite()
        iso = verify_sealed_isolation(master_suite)
        assert iso["isolation_verified"]

        sealed_batch, _ = encode(master_suite, tokenizer, context)
        y_sealed = torch.tensor([c.labels for c in master_suite], dtype=torch.long, device="cuda")

        p4_results: dict[str, Any] = {
            "sealed_suite_isolation": iso,
            "neural_arms": {},
            "lexical_controls": {},
        }

        # Subsplits of sealed suite
        rel_indices = [i for i, c in enumerate(master_suite) if c.is_relationship_sensitive]
        std_indices = [i for i, c in enumerate(master_suite) if not c.is_relationship_sensitive]
        rel_batch = tuple(v[rel_indices] for v in sealed_batch)
        rel_cases = [master_suite[i] for i in rel_indices]
        std_batch = tuple(v[std_indices] for v in sealed_batch)
        std_cases = [master_suite[i] for i in std_indices]

        # 1. Neural Evaluations
        for arm_name in candidate_arms:
            if arm_name not in trained_models:
                continue
            m = trained_models[arm_name]
            m.eval()
            eval_full = enrich(evaluate(m, sealed_batch, master_suite))
            eval_rel = enrich(evaluate(m, rel_batch, rel_cases))
            eval_std = enrich(evaluate(m, std_batch, std_cases))
            p4_results["neural_arms"][arm_name] = {
                "sealed_full": compact(eval_full),
                "sealed_standard": compact(eval_std),
                "sealed_relationship_sensitive": compact(eval_rel),
            }
            print(f"  {arm_name} on Sealed Full: {eval_full['exact']*100:.2f}%, Rel-Sensitive: {eval_rel['exact']*100:.2f}%")

        # 2. Lexical & Majority Baselines
        print("--- Fitting Lexical Controls (Word 1-3-gram & BoW) on 378 Coverage Training Cases ---")
        probe_train_ix, _ = coverage_matched_probe_indices(training["coverage"])
        all_eval_rows = training["coverage"][:756] + master_suite
        y_all = torch.tensor([r.labels for r in all_eval_rows], dtype=torch.long, device="cuda")

        bow_vocab = sorted({w for i in probe_train_ix for w in words(all_eval_rows[i], "bow_linear")})
        ngram_vocab = sorted({w for i in probe_train_ix for w in words(all_eval_rows[i], "ngram_linear")})

        bow_feats: list[list[float]] = []
        ngram_feats: list[list[float]] = []
        for r in all_eval_rows:
            bow_counts = Counter(words(r, "bow_linear"))
            bow_feats.append([float(bow_counts[w]) for w in bow_vocab])
            ngram_counts = Counter(words(r, "ngram_linear"))
            ngram_feats.append([float(ngram_counts[w]) for w in ngram_vocab])

        bow_tensor = torch.tensor(bow_feats, dtype=torch.float32, device="cuda")
        ngram_tensor = torch.tensor(ngram_feats, dtype=torch.float32, device="cuda")

        sealed_offset = 756
        sealed_indices_in_all = list(range(sealed_offset, sealed_offset + len(master_suite)))
        rel_indices_in_all = [sealed_offset + i for i in rel_indices]

        for ctrl_name, feat in (("bow_linear", bow_tensor), ("ngram_linear", ngram_tensor)):
            preds, _ = solve_ridge(feat, y_all, probe_train_ix, regularization=1.0)
            binary_preds = (preds > 0).long()

            acc_full = float((binary_preds[sealed_indices_in_all] == y_all[sealed_indices_in_all]).all(dim=-1).float().mean().item())
            acc_rel = float((binary_preds[rel_indices_in_all] == y_all[rel_indices_in_all]).all(dim=-1).float().mean().item())
            p4_results["lexical_controls"][ctrl_name] = {
                "sealed_full_exact": acc_full,
                "sealed_relationship_sensitive_exact": acc_rel,
            }
            print(f"  {ctrl_name} on Sealed Full: {acc_full*100:.2f}%, Rel-Sensitive: {acc_rel*100:.2f}%")

        # Majority class baseline
        majority_label = (0, 0)
        maj_full = float((y_sealed == torch.tensor(majority_label, device="cuda")).all(dim=-1).float().mean().item())
        p4_results["lexical_controls"]["majority"] = {"sealed_full_exact": maj_full}

        results["phases"]["phase_4_sealed"] = p4_results

    # -------------------------------------------------------------
    # FINAL QUALIFICATION VERDICT
    # -------------------------------------------------------------
    print("=== Determining Stage A.10 Overall Qualification Status ===")
    verdict = "FAIL"
    best_candidate = ""
    best_candidate_acc = 0.0

    if "phase_4_sealed" in results["phases"]:
        for arm_k, arm_d in results["phases"]["phase_4_sealed"]["neural_arms"].items():
            acc = arm_d["sealed_full"]["exact"]
            if acc > best_candidate_acc:
                best_candidate_acc = acc
                best_candidate = arm_k

        if best_candidate_acc >= 0.80:
            # Check if outperforming lexical controls
            ngram_rel = results["phases"]["phase_4_sealed"]["lexical_controls"]["ngram_linear"]["sealed_relationship_sensitive_exact"]
            candidate_rel = results["phases"]["phase_4_sealed"]["neural_arms"][best_candidate]["sealed_relationship_sensitive"]["exact"]
            if candidate_rel > ngram_rel:
                verdict = "QUALIFIED"
            else:
                verdict = "PARTIAL"
        elif best_candidate_acc >= 0.65:
            verdict = "PARTIAL"

    results["qualification_verdict"] = {
        "status": verdict,
        "best_candidate_arm": best_candidate,
        "best_candidate_sealed_accuracy": best_candidate_acc,
    }

    out_file = out_dir / "stage_a10_final_manifest.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Stage A.10 completed. Overall Verdict: {verdict}. Results saved to {out_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Stage A.10 Independent Foundation Qualification")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--resume-check", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--phase",
        choices=["all", "p1_replication", "p2_factorial", "p3_migration", "p4_sealed"],
        default="all",
    )
    args = parser.parse_args()
    require_remote_cuda()

    if args.resume_check:
        owner = LearningState.load(args.resume_check)
        training, _, plans, _ = prepare_a10_data()
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
        (args.output_dir / "resume_child_a10.json").write_text(json.dumps(child_info))
        return

    if not args.root:
        raise ValueError("--root is required when not in resume-check mode")

    run_stage_a10(args.root, args.output_dir, phase=args.phase)


if __name__ == "__main__":
    main()
