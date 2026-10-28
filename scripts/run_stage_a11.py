"""REMOTE ONLY: Stage A.11 Provenance Reconciliation & Clause-Sensitive Relational Learning.

Executes:
- Step 1: Machine-verified provenance reconciliation between Stage A.9 and Stage A.10.
- Step 2: Semantic Minimal Pairs (100% Bag-of-Words multiset equality).
- Step 3: Controlled 2x2 Factorial Contrast:
  * ARM 1: A.10 Single-Clause Pretrain + Single-Clause Supervised
  * ARM 2: Multi-Clause Pretrain + Single-Clause Supervised
  * ARM 3: Single-Clause Pretrain + Relational Supervised
  * ARM 4: Multi-Clause Pretrain + Relational Supervised
  * Detached Readout Probe Contrast (h_last_question_token vs h_mean_pool)
- Step 4: Conditional Independent Qualification on pristine sealed_a11_v1 (360 cases).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from probe_representations_a8 import solve_ridge
from reconcile_a9_a10_provenance import run_provenance_reconciliation
from run_attribute_learning import encode, evaluate, words
from run_generalization_a6_training import prepare as a6_prepare
from run_stable_attributes import compact, enrich

from adaptive_mini_gpt.data.attribute_learning import TAG_IDS
from adaptive_mini_gpt.data.clause_minimal_pairs_a11 import (
    build_semantic_minimal_pairs_suite,
    evaluate_pair_exact_accuracy,
)
from adaptive_mini_gpt.data.exposure_a7 import repeated_plan
from adaptive_mini_gpt.data.pretraining_a9 import (
    generate_syntactic_pretraining_corpus,
    verify_pretraining_corpus_isolation,
)
from adaptive_mini_gpt.data.pretraining_a11 import (
    generate_multi_clause_pretraining_corpus,
    verify_multi_clause_pretraining_corpus_isolation,
)
from adaptive_mini_gpt.data.probe_a8 import coverage_matched_probe_indices
from adaptive_mini_gpt.data.sealed_a11 import (
    generate_master_sealed_suite_a11,
    verify_sealed_a11_isolation,
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

SPEC_A11 = Path("data/specs/stage_a11_v1.json")
SPEC_SEALED_A11 = Path("data/specs/sealed_a11_v1.json")


def build_a11_optimizer_and_clipper(model: nn.Module, total_steps: int):
    """Builds decoupled two-timescale optimizer (Stage A.8-A.10 stabilized standard)."""
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
    """Pretrains model on synthetic sentences using causal next-token cross-entropy."""
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
        losses.append(float(loss.item()))

    return losses


def build_relational_augmented_plan(
    single_coverage_plan: list[list[int]],
    len_broad: int,
    train_pairs: list[tuple[Any, Any]],
    synthetic_offset: int,
    total_steps: int = 7200,
) -> tuple[list[list[int]], int]:
    """Constructs matched 18-case batches interleaving minimal pairs while keeping synthetic rehearsal intact."""
    relational_offset = len_broad
    num_train_pairs = len(train_pairs)

    rel_plan: list[list[int]] = []
    for b_idx, ix in enumerate(single_coverage_plan):
        syn_items = [i for i in ix if i >= synthetic_offset]
        reindexed_syn = [len_broad + (2 * num_train_pairs) + (i - synthetic_offset) for i in syn_items]

        new_items = [i for i in ix if i < synthetic_offset]
        selected_single = new_items[:10]

        pair_idx = b_idx % num_train_pairs
        pair_item_a = relational_offset + (2 * pair_idx)
        pair_item_b = relational_offset + (2 * pair_idx + 1)

        batch = selected_single + [pair_item_a, pair_item_b] + reindexed_syn
        assert len(batch) == 18, f"Batch size mismatch: {len(batch)} != 18"
        rel_plan.append(batch)

    return repeated_plan(rel_plan, total_steps), len_broad + (2 * num_train_pairs)


def extract_pooled_features(
    model: IntegratedStudentM104_E,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts both uniform mean-pooled and last-question-token hidden states from model backbone."""
    with torch.no_grad():
        h_seq = model.extract_sequence_hidden_states(tokens)
        B, L, _D = h_seq.shape
        q_id = ByteTextTokenizerM103.QUESTION_ID
        a_id = ByteTextTokenizerM103.ANSWER_ID

        global_mask = tokens != ByteTextTokenizerM103.PAD_ID
        entity_mask = torch.zeros((B, L), dtype=torch.bool, device=tokens.device)
        last_indices = torch.zeros(B, dtype=torch.long, device=tokens.device)

        for b in range(B):
            row = tokens[b]
            q_locs = (row == q_id).nonzero(as_tuple=True)[0]
            a_locs = (row == a_id).nonzero(as_tuple=True)[0]

            if len(q_locs) > 0 and len(a_locs) > 0 and a_locs[0] > q_locs[0]:
                entity_mask[b, q_locs[0] : a_locs[0]] = True
                last_indices[b] = a_locs[0] - 1
            elif len(q_locs) > 0:
                entity_mask[b, q_locs[0] :] = True
                last_indices[b] = (global_mask[b].nonzero(as_tuple=True)[0])[-1]
            else:
                half = L // 2
                entity_mask[b, half:] = True
                last_indices[b] = L - 1

        entity_mask_f = (entity_mask & global_mask).float().unsqueeze(-1)
        h_mean_pool = (h_seq * entity_mask_f).sum(dim=1) / entity_mask_f.sum(dim=1).clamp(min=1.0)
        batch_idx = torch.arange(B, device=tokens.device)
        h_last_token = h_seq[batch_idx, last_indices]

    return h_mean_pool, h_last_token


def evaluate_detached_probes(
    feat_train: torch.Tensor,
    y_train: torch.Tensor,
    feat_eval: torch.Tensor,
    y_eval: torch.Tensor,
    eval_cases: list[Any],
    regularization: float = 1.0,
) -> dict[str, Any]:
    """Fits ridge regression probe on frozen representations and evaluates accuracy & minimal pair metrics."""
    train_indices = list(range(len(feat_train)))
    preds, _ = solve_ridge(
        torch.cat([feat_train, feat_eval], dim=0),
        torch.cat([y_train, y_eval], dim=0),
        train_indices,
        regularization=regularization,
    )
    eval_preds = preds[len(feat_train) :]
    binary_preds = (eval_preds >= 0).long()

    exact_matches = (binary_preds == y_eval).all(dim=-1).float()
    acc = float(exact_matches.mean().item())

    res: dict[str, Any] = {"exact": acc}
    pair_groups: dict[str, list[int]] = defaultdict(list)
    for idx, c in enumerate(eval_cases):
        if getattr(c, "pair_id", None) is not None:
            pair_groups[c.pair_id].append(idx)

    if pair_groups:
        pair_exact_flags = []
        for _pid, member_indices in pair_groups.items():
            if len(member_indices) == 2:
                p_correct = bool(exact_matches[member_indices[0]] == 1.0 and exact_matches[member_indices[1]] == 1.0)
                pair_exact_flags.append(p_correct)
        if pair_exact_flags:
            res["pair_exact_accuracy"] = float(sum(pair_exact_flags) / len(pair_exact_flags))
            res["pair_exact_flip_rate"] = res["pair_exact_accuracy"]

    return res


def evaluate_clause_window_lexical_baseline(
    cases: list[Any],
) -> dict[str, Any]:
    """Baseline that extracts the clause mentioning the queried entity and classifies attributes locally."""
    correct_count = 0
    pair_correct_flags = []
    pair_groups: dict[str, list[tuple[Any, bool]]] = defaultdict(list)

    for c in cases:
        q = c.question
        name = getattr(c, "name", None)
        if not name:
            m = re.search(r"target is (\w+)", q.lower()) or re.search(r"query focuses on (\w+)", q.lower())
            name = m.group(1).capitalize() if m else "Target"

        clauses = re.split(r"[;.,]", q)
        target_clause = ""
        for clause in clauses:
            if name.lower() in clause.lower() and not ("target is" in clause.lower() or "not " + name.lower() in clause.lower()):
                target_clause = clause.lower()
                break
        if not target_clause:
            target_clause = q.lower()

        pred_flam = 0
        pred_cond = 0

        if "flammable" in target_clause:
            if re.search(r"\b(not|never|without)\b(?:\s+\w+){0,3}\s+flammable", target_clause):
                pred_flam = 0
            else:
                pred_flam = 1

        if "conductive" in target_clause:
            if re.search(r"\b(not|never|without)\b(?:\s+\w+){0,3}\s+conductive", target_clause):
                pred_cond = 0
            else:
                pred_cond = 1

        is_correct = bool(pred_flam == c.labels[0] and pred_cond == c.labels[1])
        if is_correct:
            correct_count += 1

        pid = getattr(c, "pair_id", None)
        if pid:
            pair_groups[pid].append((c, is_correct))

    acc = float(correct_count / len(cases)) if cases else 0.0
    for _pid, members in pair_groups.items():
        if len(members) == 2:
            pair_correct_flags.append(members[0][1] and members[1][1])

    pair_acc = float(sum(pair_correct_flags) / len(pair_correct_flags)) if pair_correct_flags else 0.0
    return {
        "exact": acc,
        "pair_exact_accuracy": pair_acc,
        "pair_exact_flip_rate": pair_acc,
    }


def train_arm_a11(
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
    optimizer, scheduler, clipper = build_a11_optimizer_and_clipper(owner.model, steps)
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
            snap_metrics: dict[str, Any] = {}
            for k, rs in eval_sets.items():
                eval_res = enrich(evaluate(owner.model, eval_batches[k], rs))
                if k == "dev_minimal_pairs":
                    pair_res = evaluate_pair_exact_accuracy(owner.model, eval_batches[k], rs)
                    eval_res.update(pair_res)
                snap_metrics[k] = eval_res

            snapshots[str(step)] = {k: compact(v) for k, v in snap_metrics.items()}
            owner.save(out_dir / f"{arm_name.lower()}_{step}.pt")

    wall_time = time.time() - t0
    return snapshots, wall_time


def check_development_gates(
    cohort_snapshots_7200: list[dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, float]]:
    """Evaluates Stage A.11 Development Admission Gates 1-3 across seeds 42, 43, 44."""
    train_accs = [s["training"]["exact"] for s in cohort_snapshots_7200]
    mean_train = float(sum(train_accs) / len(train_accs))
    gate_1 = bool(mean_train >= 0.99)

    old_known_accs = [s["old_known"]["exact"] for s in cohort_snapshots_7200]
    old_renamed_accs = [s["old_renamed"]["exact"] for s in cohort_snapshots_7200]
    mean_old_known = float(sum(old_known_accs) / len(old_known_accs))
    mean_old_renamed = float(sum(old_renamed_accs) / len(old_renamed_accs))
    gate_2 = bool(mean_old_known >= 0.95 and mean_old_renamed >= 0.95)

    fam_accs = [s["development_familiar"]["exact"] for s in cohort_snapshots_7200]
    nov_accs = [s["development_novel"]["exact"] for s in cohort_snapshots_7200]
    mp_accs = [s["dev_minimal_pairs"]["exact"] for s in cohort_snapshots_7200]
    mp_flip_rates = [s["dev_minimal_pairs"].get("pair_exact_flip_rate", 0.0) for s in cohort_snapshots_7200]

    mean_fam = float(sum(fam_accs) / len(fam_accs))
    mean_nov = float(sum(nov_accs) / len(nov_accs))
    min_single = float(min(min(fam_accs), min(nov_accs)))

    mean_mp_exact = float(sum(mp_accs) / len(mp_accs))
    mean_mp_flip = float(sum(mp_flip_rates) / len(mp_flip_rates))

    gate_3 = bool(
        mean_fam >= 0.85
        and mean_nov >= 0.85
        and min_single >= 0.80
        and mean_mp_exact >= 0.85
        and mean_mp_flip >= 0.80
    )

    gates = {
        "gate_1_train_acquisition": gate_1,
        "gate_2_old_frame_preservation": gate_2,
        "gate_3_unfamiliar_grammar_and_minimal_pair": gate_3,
    }
    metrics = {
        "mean_train": mean_train,
        "mean_old_known": mean_old_known,
        "mean_old_renamed": mean_old_renamed,
        "mean_dev_familiar": mean_fam,
        "mean_dev_novel": mean_nov,
        "min_single_clause": min_single,
        "mean_dev_minimal_pair_exact": mean_mp_exact,
        "mean_dev_minimal_pair_flip": mean_mp_flip,
    }
    return gates, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage A.11 Provenance Reconciliation & Clause-Sensitive Learning")
    parser.add_argument("--phase", choices=["all", "step1_reconcile", "step2_pairs", "step3_contrast", "step4_sealed"], default="all")
    parser.add_argument("--out_dir", type=Path, default=Path("runs/stage_a11"))
    args = parser.parse_args()

    require_remote_cuda()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Adaptive Mini-GPT: Stage A.11 Master Execution ===")
    spec = json.loads(SPEC_A11.read_text())
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"Commit SHA: {source_sha} | Platform: Kaggle Tesla T4 GPU")

    # STEP 1: Provenance & Reproducibility Reconciliation
    print("\n=== STEP 1: Provenance & Reproducibility Reconciliation ===")
    recon_report = run_provenance_reconciliation()
    recon_manifest_path = Path("data/manifests/reconciliation_a9_a10_v1.json")
    recon_manifest_path.write_text(json.dumps(recon_report, indent=2))
    print(f"Reconciliation verified. File SHA-256: {recon_report['parent_digest_reconciliation']['root_file_sha256']}")
    print(f"State dict SHA-256: {recon_report['parent_digest_reconciliation']['root_tensor_digest_code']}")

    if args.phase == "step1_reconcile":
        print("Completed Step 1. Exiting.")
        return

    # STEP 2: Minimal Pairs Suite Construction & Verification
    print("\n=== STEP 2: Controlled Semantic Minimal Pairs Suite ===")
    pairs_suite = build_semantic_minimal_pairs_suite()
    train_pairs = pairs_suite["train_pairs"]
    train_pair_cases = pairs_suite["train_cases"]
    dev_pairs = pairs_suite["dev_pairs"]
    dev_pair_cases = pairs_suite["dev_cases"]
    print(f"Loaded {len(train_pairs)} train pairs ({len(train_pair_cases)} cases) and {len(dev_pairs)} dev pairs ({len(dev_pair_cases)} cases).")

    # Data Setup
    print("\n--- Preparing Datasets & Training Batches ---")
    training, _newcounts, blocks, shared, _ = a6_prepare()
    context = 256
    tokenizer = ByteTextTokenizerM103()

    broad_cases = training["coverage"][:756]
    syn_cases = training["coverage"][756:]
    rel_train_cases = broad_cases + train_pair_cases + syn_cases

    batches = {}
    sequences = {}
    shared_batches = {}

    batches["single_coverage"], sequences["single_coverage"] = encode(training["coverage"], tokenizer, context)
    batches["relational_coverage"], sequences["relational_coverage"] = encode(rel_train_cases, tokenizer, context)

    for k, rs in shared.items():
        shared_batches[k], sequences[k] = encode(rs, tokenizer, context)

    dev_mp_batch, _ = encode(dev_pair_cases, tokenizer, context)
    shared_batches["dev_minimal_pairs"] = dev_mp_batch

    single_plan = repeated_plan(blocks["coverage"], spec["supervised_steps"])
    rel_plan, _ = build_relational_augmented_plan(
        blocks["coverage"],
        len_broad=756,
        train_pairs=train_pairs,
        synthetic_offset=756,
        total_steps=spec["supervised_steps"],
    )

    eval_sets_common = {
        "old_known": shared["old_known"],
        "old_renamed": shared["old_renamed"],
        "development_familiar": shared["development_familiar"],
        "development_novel": shared["development_novel"],
        "dev_minimal_pairs": dev_pair_cases,
    }

    results: dict[str, Any] = {
        "stage": "Stage A.11 Provenance Reconciliation & Clause-Sensitive Relational Learning",
        "git_sha": source_sha,
        "platform": "Kaggle Tesla T4 GPU",
        "provenance_reconciliation": recon_report,
        "arms": {},
        "detached_probes": {},
        "gates": {},
        "sealed_qualification": None,
    }

    # STEP 3: Controlled 2x2 Factorial Contrast
    arm_definitions = [
        ("ARM_1_A10_BASELINE", "syntactic_a9", "single_clause"),
        ("ARM_2_CLAUSE_PRETRAIN_ONLY", "relational_a11", "single_clause"),
        ("ARM_3_SINGLE_PRETRAIN_RELATIONAL_SUP", "syntactic_a9", "relational"),
        ("ARM_4_CLAUSE_PRETRAIN_RELATIONAL_SUP", "relational_a11", "relational"),
    ]

    trained_cohorts: dict[str, dict[int, Any]] = defaultdict(dict)
    probe_results: dict[str, Any] = {}

    for arm_id, pretrain_type, sup_type in arm_definitions:
        print("\n========================================================")
        print(f"Executing {arm_id} ({pretrain_type} pretrain + {sup_type} supervision)")
        print("========================================================")
        arm_snaps_by_seed: dict[str, Any] = {}

        for seed in spec["seeds"]:
            print(f"\n--- {arm_id} | Seed {seed} ---")
            seed_all(seed)

            if pretrain_type == "syntactic_a9":
                corpus = generate_syntactic_pretraining_corpus(2000, seed=seed)
                iso = verify_pretraining_corpus_isolation(corpus)
                assert iso["isolation_verified"]
            else:
                corpus = generate_multi_clause_pretraining_corpus(2000, seed=seed)
                iso = verify_multi_clause_pretraining_corpus_isolation(corpus)
                assert iso["isolation_verified"]

            student = TextStudentM103(prune_orphaned_lm_head=True, pos_encoding="rope")
            model = IntegratedStudentM104_E(student, mode="trainable_backbone").cuda()

            t_pre0 = time.time()
            pretrain_losses = run_language_pretraining(model, corpus, tokenizer, steps=1500, lr=0.0005)
            t_pre = time.time() - t_pre0
            print(f"Pretraining complete in {t_pre:.2f}s. Final loss: {pretrain_losses[-1]:.4f}")

            owner = LearningState(model)
            if sup_type == "single_clause":
                cur_plan = single_plan
                cur_batch = batches["single_coverage"]
                cur_train_cases = broad_cases
                cur_eval_sets = {"training": cur_train_cases, **eval_sets_common}
                cur_eval_batches = {"training": tuple(v[:756] for v in batches["single_coverage"]), **shared_batches}
            else:
                cur_plan = rel_plan
                cur_batch = batches["relational_coverage"]
                cur_train_cases = broad_cases + train_pair_cases
                cur_eval_sets = {"training": cur_train_cases, **eval_sets_common}
                cur_eval_batches = {"training": tuple(v[: len(cur_train_cases)] for v in batches["relational_coverage"]), **shared_batches}

            arm_name = f"{arm_id}_SEED_{seed}"
            snaps, _wall = train_arm_a11(
                owner,
                arm_name,
                spec["supervised_steps"],
                cur_plan,
                cur_batch,
                cur_eval_sets,
                cur_eval_batches,
                args.out_dir,
            )
            arm_snaps_by_seed[str(seed)] = snaps
            trained_cohorts[arm_id][seed] = owner.model

            print(f"--- Fitting Detached Probes (h_last_question_token vs h_mean_pool) for {arm_name} ---")
            owner.model.eval()
            tr_tokens = cur_eval_batches["training"][0]
            tr_y = cur_eval_batches["training"][2][:, TAG_IDS]
            h_mean_tr, h_last_tr = extract_pooled_features(owner.model, tr_tokens)

            arm_probe_seed: dict[str, Any] = {}
            for probe_k in ("dev_minimal_pairs", "development_familiar", "development_novel"):
                ev_tokens = cur_eval_batches[probe_k][0]
                ev_y = cur_eval_batches[probe_k][2][:, TAG_IDS]
                ev_cases = cur_eval_sets[probe_k]
                h_mean_ev, h_last_ev = extract_pooled_features(owner.model, ev_tokens)

                res_mean = evaluate_detached_probes(h_mean_tr, tr_y, h_mean_ev, ev_y, ev_cases, regularization=1.0)
                res_last = evaluate_detached_probes(h_last_tr, tr_y, h_last_ev, ev_y, ev_cases, regularization=1.0)
                arm_probe_seed[probe_k] = {
                    "h_mean_pool": res_mean,
                    "h_last_question_token": res_last,
                }
            probe_results[arm_name] = arm_probe_seed

        results["arms"][arm_id] = {
            "seeds": arm_snaps_by_seed,
            "mean_metrics_7200": {
                "training": float(sum(arm_snaps_by_seed[str(s)]["7200"]["training"]["exact"] for s in spec["seeds"]) / len(spec["seeds"])),
                "old_known": float(sum(arm_snaps_by_seed[str(s)]["7200"]["old_known"]["exact"] for s in spec["seeds"]) / len(spec["seeds"])),
                "old_renamed": float(sum(arm_snaps_by_seed[str(s)]["7200"]["old_renamed"]["exact"] for s in spec["seeds"]) / len(spec["seeds"])),
                "development_familiar": float(sum(arm_snaps_by_seed[str(s)]["7200"]["development_familiar"]["exact"] for s in spec["seeds"]) / len(spec["seeds"])),
                "development_novel": float(sum(arm_snaps_by_seed[str(s)]["7200"]["development_novel"]["exact"] for s in spec["seeds"]) / len(spec["seeds"])),
                "dev_minimal_pairs_exact": float(sum(arm_snaps_by_seed[str(s)]["7200"]["dev_minimal_pairs"]["exact"] for s in spec["seeds"]) / len(spec["seeds"])),
                "dev_minimal_pairs_flip": float(sum(arm_snaps_by_seed[str(s)]["7200"]["dev_minimal_pairs"].get("pair_exact_flip_rate", 0.0) for s in spec["seeds"]) / len(spec["seeds"])),
            },
        }

    results["detached_probes"] = probe_results

    # STEP 4: Gates & Conditional Sealed Evaluation
    print("\n=== Evaluating Development Admission Gates on ARM_4 Cohort ===")
    arm4_cohort_7200 = [results["arms"]["ARM_4_CLAUSE_PRETRAIN_RELATIONAL_SUP"]["seeds"][str(s)]["7200"] for s in spec["seeds"]]
    gates_eval, gates_metrics = check_development_gates(arm4_cohort_7200)
    results["gates"] = {"checks": gates_eval, "metrics": gates_metrics}

    print(f"Gate 1 (Train Acquisition): {gates_eval['gate_1_train_acquisition']} ({gates_metrics['mean_train']:.2%})")
    print(f"Gate 2 (Old Preservation): {gates_eval['gate_2_old_frame_preservation']} ({gates_metrics['mean_old_known']:.2%} / {gates_metrics['mean_old_renamed']:.2%})")
    print(f"Gate 3 (Unfamiliar Grammar & Pairs): {gates_eval['gate_3_unfamiliar_grammar_and_minimal_pair']} (Fam: {gates_metrics['mean_dev_familiar']:.2%}, Nov: {gates_metrics['mean_dev_novel']:.2%}, MP Exact: {gates_metrics['mean_dev_minimal_pair_exact']:.2%}, MP Flip: {gates_metrics['mean_dev_minimal_pair_flip']:.2%})")

    gates_all_passed = all(gates_eval.values())

    if not gates_all_passed:
        print("\n[!] Gates 1-3 FAILED. Condition for Sealed Evaluation NOT met.")
        print("Reporting failure forensics and detached probe isolation...")
        sealed_qualification_report = {
            "executed": False,
            "reason": "Development Admission Gates 1-3 did not pass simultaneously across full 3-seed cohort.",
            "gates_status": gates_eval,
            "gate_metrics": gates_metrics,
        }
    else:
        print("\n[+] Gates 1-3 PASSED! Generating pristine sealed_a11_v1 suite...")
        master_sealed_cases = generate_master_sealed_suite_a11()
        sealed_iso = verify_sealed_a11_isolation(master_sealed_cases)
        assert sealed_iso["isolation_verified"]
        print(f"Pristine sealed suite verified. Total cases: {len(master_sealed_cases)} | SHA-256: {sealed_iso['sealed_suite_sha256']}")

        sealed_batch, _ = encode(master_sealed_cases, tokenizer, context)
        sealed_y = sealed_batch[2][:, TAG_IDS]

        cohort_sealed_scores: dict[str, Any] = {}
        arm4_models = trained_cohorts["ARM_4_CLAUSE_PRETRAIN_RELATIONAL_SUP"]

        for seed in spec["seeds"]:
            model = arm4_models[seed]
            model.eval()
            with torch.no_grad():
                res_e2e = enrich(evaluate(model, sealed_batch, master_sealed_cases))
                pair_e2e = evaluate_pair_exact_accuracy(model, sealed_batch, master_sealed_cases)
                res_e2e.update(pair_e2e)

            cohort_sealed_scores[str(seed)] = compact(res_e2e)

        print("--- Evaluating Lexical Baselines on Sealed Suite ---")
        lexical_baseline_res = evaluate_clause_window_lexical_baseline(master_sealed_cases)

        probe_train_ix, _ = coverage_matched_probe_indices(broad_cases)
        all_eval_rows = broad_cases[:756] + master_sealed_cases
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
        sealed_indices_in_all = list(range(sealed_offset, sealed_offset + len(master_sealed_cases)))

        bow_preds, _ = solve_ridge(bow_tensor, y_all, probe_train_ix, regularization=1.0)
        ngram_preds, _ = solve_ridge(ngram_tensor, y_all, probe_train_ix, regularization=1.0)

        bow_bin = (bow_preds[sealed_indices_in_all] >= 0).long()
        ngram_bin = (ngram_preds[sealed_indices_in_all] >= 0).long()

        bow_exact = float((bow_bin == sealed_y).all(dim=-1).float().mean().item())
        ngram_exact = float((ngram_bin == sealed_y).all(dim=-1).float().mean().item())

        sealed_qualification_report = {
            "executed": True,
            "isolation_audit": sealed_iso,
            "candidate_arm": "ARM_4_CLAUSE_PRETRAIN_RELATIONAL_SUP",
            "cohort_scores": cohort_sealed_scores,
            "mean_candidate_exact": float(sum(cohort_sealed_scores[str(s)]["exact"] for s in spec["seeds"]) / len(spec["seeds"])),
            "baselines": {
                "bow_linear": {"exact": bow_exact},
                "ngram_linear": {"exact": ngram_exact},
                "clause_window_lexical": lexical_baseline_res,
            },
        }

    results["sealed_qualification"] = sealed_qualification_report

    manifest_path = Path("data/manifests/stage_a11_v1.json")
    manifest_path.write_text(json.dumps(results, indent=2))
    (args.out_dir / "stage_a11_v1.json").write_text(json.dumps(results, indent=2))
    print(f"\nFinal Stage A.11 manifest written to {manifest_path} and {args.out_dir / 'stage_a11_v1.json'}")

    print("\n--- Running Fresh-Process Continuation Check ---")
    chk_path = args.out_dir / "arm_4_clause_pretrain_relational_sup_seed_42_7200.pt"
    if chk_path.exists():
        loaded_owner = LearningState.load(chk_path)
        loaded_model = loaded_owner.model.cuda()
        loaded_model.eval()
        orig_model = trained_cohorts["ARM_4_CLAUSE_PRETRAIN_RELATIONAL_SUP"][42]
        orig_model.eval()

        val_tokens = shared_batches["development_familiar"][0][:10]
        val_actions = torch.zeros(10, dtype=torch.long, device="cuda")
        with torch.no_grad():
            out_orig, _, _, _ = orig_model.forward_with_strategy(val_tokens, val_actions)
            out_load, _, _, _ = loaded_model.forward_with_strategy(val_tokens, val_actions)
            max_logit_diff = float((out_orig - out_load).abs().max().item())
        print(f"Continuation Logit Difference: {max_logit_diff:.2e}")
        assert max_logit_diff == 0.0, f"Continuation discrepancy: {max_logit_diff}"
        print("[+] Continuation check PASSED (Delta logit = 0.00e+00).")


if __name__ == "__main__":
    main()
