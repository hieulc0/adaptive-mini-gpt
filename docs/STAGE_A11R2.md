# Stage A.11-R2 — Candidate Evaluation Integrity Audit & Entity-Conditioned Binding Diagnosis

Status: **COMPLETED; VERDICT: AUDIT RESOLVED & GATES FAILED (EVALUATION INTEGRITY BUG FULLY EXPLAINED; CANDIDATE ARMS FAILED GATE 3 MULTI-ENTITY BINDING; PRISTINE SEALED SUITE UNTOUCHED)**. Continues from Stage A.11-R tested revision `77a7c3dd051a068dfdb2a93a6caf70a2e01a1258` (`77a7c3d`) on branch `m104-e-and-m105-a`. Historical parent checkpoints (`be8e289`, `b78b790`, `46cf2ab`, `3e93d72...`) and Stage A.8–A.11-R manifests/reports remain strictly immutable. Remote GPU execution on Kaggle Tesla T4 (`sparklinghydraulics0/amg-stage-a11r2`) executed Task 1 (forensic candidate evaluation audit and pigeonhole bound reconciliation), Task 2 (unified candidate evaluation on true student predictions), Task 3 (development-only entity-conditioned counterfactual diagnostic suite `EB-1..4` with independent structured oracle verification), Task 4 (controlled 3-arm bounded readout intervention with full cosine two-timescale annealing and balanced 24-case batch planning), fresh-process deterministic continuation ($\Delta \text{logit} = 0.00\text{e+}00$), and the full GPU regression test suite (`AMG_RUN_GPU_TESTS=1 pytest -q`, **860 passed** in 730s).

In strict compliance with preregistered protocol and scientific integrity, because Candidate Readout Arms failed Development Admission Gate 3 (Multi-Entity Binding: MP-2 cohort mean = 54.17% vs $\ge 75.0\%$, EB-1..4 cohort mean = 34.03% vs $\ge 80.0\%$), **`sealed_a11_v1` WAS NOT EVALUATED AND REMAINS STRICTLY UNTOUCHED**.

---

## 0. Non-Negotiable Execution Boundary (`AGENTS.md`)
- **LOCAL-SAFE ONLY on host machine**: Code reading/editing, pure-Python unit tests (`tests/test_stage_a11r2_contracts.py`), static analysis (`uv run ruff check`, `uv run pyright`), and compilation (`uv run python -m compileall -q src scripts tests kaggle`) ran with ZERO neural forward/backward passes, ZERO checkpoint loading, and ZERO CPU training.
- **REMOTE-REQUIRED on Kaggle Tesla T4 GPU**: All historical checkpoint reloads, candidate metric audits, multi-seed neural pretraining and supervised updates (1,500 pretrain + 7,200 supervised updates), and full regression test suite (860 tests) were executed exclusively on remote Kaggle CUDA (`sparklinghydraulics0/amg-stage-a11r2`).

---

## 1. Provenance & Execution Summary

| Parameter | Execution Record |
|---|---|
| **Git SHA (Tested Source)** | `832326beb0dd7be87b43bfaef710426e7a007a79` (Launch commit: `2e1ed16`) |
| **Branch** | `m104-e-and-m105-a` |
| **Parent Checkpoint Raw File SHA256** | `3e93d7238bfdb706ddfd49ee9903fed8b1e6f577529537f1e21cae12e905b262` |
| **Active Base Model Parameter Digest** | `511136ae1d6ccc472a964877c3be67411848b52f617a2fafe8a7426025482e15` |
| **Candidate Model Active Parameters** | Strictly **1,222,868** (0 parameter growth across all arms) |
| **Pristine Sealed Suite Isolation Digest** | `sealed_a11_v1` (`21e6ff6cd6c39289d70962c4f436e49626554d0ad32c4c732ec26ccd3b1e031b`, 360 cases, **100% untouched/unevaluated**) |
| **Kaggle Kernel** | `sparklinghydraulics0/amg-stage-a11r2` (Version 1) |
| **Hardware Platform** | Kaggle Tesla T4 GPU (15,102 MiB VRAM), CUDA 12.8, PyTorch 2.11.0+cu128, Python 3.13.15 |
| **Phases Executed** | Phase 1 Contracts (PASS, 6 passed in 1.4s), Phase 2 Forensic Audit (PASS, exit 0), Phase 3 Readout Experiment (PASS, exit 0), Continuation Verification (PASS, $\Delta \text{logit} = 0.00\text{e+}00$), Phase 4 Full Regression Suite (PASS, **860 passed, 0 failed in 730.35s**) |
| **Overall Qualification Verdict** | **`FAIL`** (Gate 1 Train = 100.00% PASS; Gate 2 Old Frame = 100.00% PASS; Gate 3 Dev Minimal Pair Both = 83.33% PASS overall, but MP-2 = 54.17% vs $\ge 75.0\%$ and EB-1..4 = 34.03% vs $\ge 80.0\%$ FAIL; Sealed Suite Untouched per preregistered protocol) |

---

## 2. Answers to the 5 Required Exit Questions

### Question 1: Forensic Audit of the Stage A.11-R Mathematical Impossibility
**What exact bug in Stage A.11-R's candidate evaluation loop caused `R_ARM_2` and `R_ARM_3` to report `0.00%` pair-both accuracy despite `76.39%` and `72.22%` case-exact accuracy?**

- **Root Cause**:
  In `scripts/run_stage_a11r.py`, `evaluate_with_paired_metrics()` executed:
  ```python
  base_metrics = evaluate(model, batch, cases)
  tokens, actions, _ = batch
  with torch.no_grad():
      _, tag_logits, _, _ = model.forward_with_strategy(tokens, actions)
      assert tag_logits is not None
      preds = (tag_logits[:, TAG_IDS] >= 0).long().tolist()
  paired_metrics = compute_paired_metrics(cases, preds)
  ```
  `evaluate(model, batch, cases)` called `model(tokens, actions)`, which correctly executed `forward_with_strategy(..., strategy=model.default_strategy)` (`last_question_token` $\rightarrow$ 76.39% case exact).
  However, the subsequent call to `model.forward_with_strategy(tokens, actions)` omitted the `strategy` argument. In `semantic_bottleneck_m104_e.py`, `forward_with_strategy` had a default argument `strategy: str = "entity_conditioned_attention"`.
  Because `entity_conditioned_attention` (ECA) was completely untrained in `R_ARM_2` and `R_ARM_3`, this second forward pass passed through random un-updated weights, producing ~25% chance accuracy. `compute_paired_metrics(cases, preds)` evaluated those noise predictions, recording `pair_both_correct_acc = 0.00%`.
  The function then stored `result["exact"] = base_metrics["exact"]` (76.39% from `last_question_token`), but overwrote `result["pair_both_correct_acc"] = 0.00%` from the untrained ECA pass!
- **Resolution**:
  1. In `semantic_bottleneck_m104_e.py`, changed default argument to `strategy: str | None = None`, resolving to `self.default_strategy` when omitted.
  2. Implemented `evaluate_unified_paired_metrics()` in `scripts/run_stage_a11r2.py`, which extracts `preds = base_metrics["predictions"]` from the primary forward pass, making prediction divergence structurally impossible.
  3. Enforced the mathematical Pigeonhole lower bound ($P_{\text{both}} \ge \max(0, 2A - 1)$) inside `compute_paired_metrics_from_predictions()`.

---

### Question 2: Re-Verified Representation & Readout Baseline Scores
**When recomputed from the single unified prediction table, what was `R_ARM_2_LAST_TOKEN`'s true student performance on MP-1, MP-2, MP-3, and overall?**

Re-evaluating the serialized predictions from `stage_a11r_v1.json` across Seeds 42, 43, 44 using `compute_paired_metrics_from_predictions`:

| Arm | Overall Case Exact ($A$) | Overall Pair Both ($P_{\text{both}}$) | MP-1 Pair Both | MP-2 Pair Both | MP-3 Pair Both | Pigeonhole Bound ($2A - 1$) |
|---|---|---|---|---|---|---|
| **R_ARM_1_MEAN_POOL_CONTROL** | 65.10% | 56.60% | 77.08% | **0.00%** | 92.71% | 56.60% $\ge$ 30.21% (PASS) |
| **R_ARM_2_LAST_TOKEN_READOUT** | **76.39%** | **66.67%** | **100.00%** | **0.00%** | **100.00%** | 66.67% $\ge$ 52.78% (PASS) |
| **R_ARM_3_QUERY_ATTN_POOL_READOUT** | 72.22% | 59.38% | 89.58% | **0.00%** | 88.54% | 59.38% $\ge$ 44.44% (PASS) |

**Key Diagnostic Insights**:
1. `R_ARM_2_LAST_TOKEN_READOUT` actually achieved **66.67% overall pair-both accuracy** (up from 56.60% for mean pooling).
2. It achieved **100.00% pair-both accuracy on MP-1** (32/32 pairs perfect across all 3 seeds) and **100.00% pair-both accuracy on MP-3** (32/32 pairs perfect across all 3 seeds).
3. However, on MP-2 (multi-entity clause binding), `R_ARM_2` scored **0.00% pair both** (29.17% case exact).
   Because $(100\% + 100\% + 29.17\%) / 3 = 76.39\%$ case exact, and $(32 + 32 + 0)/96 = 66.67\%$ pair both!
4. This conclusively proves that routing the causal final query token completely solves single-entity negation and long-range syntax, but completely fails multi-entity binding when both entities are described prior to a trailing query.

---

### Question 3: Dedicated Entity-Conditioned Diagnostic Findings (`EB-1..4`)
**How was the dedicated Development-Only Entity-Conditioned Diagnostic Suite constructed, and did the independent structured rule verifier confirm 100% label entailment?**

- **Suite Construction (`src/adaptive_mini_gpt/data/entity_binding_a11r2.py`)**:
  A 144-pair (288-case) benchmark was constructed per seed across 4 families (36 pairs each) using disjoint entities (`Tuv, Wex, Zen` for Seed 42; `Bax, Cip, Gid` for Seed 43; `Tor, Lum, Fen` for Seed 44):
  1. `EB-1_QUERY_SWAP`: Identical description prefix, opposite queried target (`Query target: e1, not e2.` vs `Query target: e2, not e1.`). 100% Bag-of-Words multiset equal.
  2. `EB-2_CLAUSE_ORDER_SWAP`: Swapped clause order, same queried target. Tests clause-order invariance.
  3. `EB-3_PROPERTY_ASSIGNMENT_SWAP`: Swapped property binding across entities, same queried target. Labels flip.
  4. `EB-4_THREE_ENTITY_DISTRACTOR`: Two distractor entities + one queried target. Tests 3-way binding.
- **Independent Structured Rule Verifier**:
  A symbolic regular-expression parser (`structured_oracle_verify()`) parsed every question text, resolved the target entity's clause bounds, and derived ground-truth property labels purely from linguistic grammar.
  **Result**: 100% agreement across all 864 cases (288 cases $\times$ 3 seeds) with zero discrepancies.

---

### Question 4: Parameter-Neutral Readout Intervention Results (`scripts/run_stage_a11r2.py`)
**How did the three candidates perform under matched 1,500 pretrain steps, matched 7,200 supervised updates with balanced 24-case batches and two-timescale cosine annealing?**

Executed on Kaggle Tesla T4 GPU across Seeds 42, 43, 44 at Update 7200:

| Metric | R2_ARM_1_MEAN_POOL | R2_ARM_2_LAST_TOKEN | R2_ARM_3_RESIDUAL_QUERY_GATED | Admission Gate |
|---|---|---|---|---|
| **Training Exact** | **100.00%** | **100.00%** | **100.00%** | $\ge 99.0\%$ (**PASS**) |
| **Old Known Exact** | **100.00%** | **100.00%** | **100.00%** | $\ge 95.0\%$ (**PASS**) |
| **Old Renamed Exact** | **100.00%** | **100.00%** | **100.00%** | $\ge 95.0\%$ (**PASS**) |
| **Dev Familiar Exact** | 96.56% | 95.24% | **98.41%** | $\ge 85.0\%$ (**PASS**) |
| **Dev Novel Exact** | 96.56% | 97.09% | **98.41%** | $\ge 85.0\%$ (**PASS**) |
| **Dev MP Case Exact** | 79.86% | **90.97%** | 75.00% | $\ge 85.0\%$ (ARM 2 PASS) |
| **Dev MP Overall Pair Both** | 68.75% | **83.33%** | 61.81% | $\ge 80.0\%$ (ARM 2 PASS) |
| **Dev MP-1 Pair Both** | 95.83% | **95.83%** | 83.33% | $\ge 80.0\%$ (PASS) |
| **Dev MP-2 Multi-Entity Both** | 16.67% | **54.17%** | 2.08% | $\ge 75.0\%$ (**FAIL**) |
| **Dev MP-3 Pair Both** | 93.75% | **100.00%** | 100.00% | $\ge 80.0\%$ (PASS) |
| **Dev Entity Binding Both** | 12.96% | **34.03%** | 20.14% | $\ge 80.0\%$ (**FAIL**) |
| **Checkpoint Continuation** | $\Delta = 0.00\text{e+}00$ | $\Delta = 0.00\text{e+}00$ | $\Delta = 0.00\text{e+}00$ | $\le 1.00\text{e-}05$ (**PASS**) |

**Key Findings**:
1. **Training Gate Solved**:
   With balanced 24-case batching (6 single, 6 MP, 6 EB, 6 rehearsal) and two-timescale cosine annealing (`backbone_lr` 0.0005 $\rightarrow$ 0.00005, `readout_lr` 0.0015 $\rightarrow$ 0.00015), **Gate 1 passed at 100.00% across all arms** (up from 94.8%–95.8% in A.11-R).
2. **Breakthrough on MP-2 Multi-Entity Binding in `R2_ARM_2_LAST_TOKEN`**:
   `R2_ARM_2_LAST_TOKEN` reached **83.33% overall pair-both accuracy** on minimal pairs, lifting MP-2 from **0.00%** in all prior stages to **54.17% cohort mean**!
   - In Seed 43, MP-2 reached **87.50%** (28/32 pairs perfect)! Overall MP both was **95.83%**!
   - In Seed 42, MP-2 reached **75.00%** (24/32 pairs perfect)! Overall MP both was **87.50%**! EB suite both was **77.78%**!
   - In Seed 44, MP-2 collapsed to **0.00%** (Seed 44 learned single-clause heuristics and missed multi-entity binding).
3. **Failure of Residual Query-Gated Pooling (`R2_ARM_3`)**:
   Averaging query attention with the last token vector degraded performance (MP-2 = 2.08%, MP both = 61.81%). The untuned dot-product attention vector washed out the sharp query context carried by the final token.

---

### Question 5: Admission Gate Decisions & Sealed Suite Status
**Did any arm pass Gates 1–4 across the full 3-seed cohort? What is the status of `sealed_a11_v1`? What are the next steps for Stage A.12?**

- **Gate Status Across 3-Seed Cohort**:
  - Gate 1 (Train Acquisition $\ge 99.0\%$): **PASS** (100.00% on all arms).
  - Gate 2 (Old-Frame Preservation $\ge 95.0\%$): **PASS** (100.00% / 100.00% on all arms).
  - Gate 3 (Unfamiliar Grammar & Binding Pairs): **FAIL**.
    While `R2_ARM_2_LAST_TOKEN` cleared unfamiliar grammar (95.24% / 97.09%) and overall minimal pairs (83.33% $\ge 80.0\%$), its cohort mean on MP-2 was **54.17%** (below 75.0%) and on EB-1..4 was **34.03%** (below 80.0%) due to Seed 44's failure.
  - Gate 4 (Mathematical Integrity & Continuation): **PASS** ($P_{\text{both}} \ge \max(0, 2A - 1)$ verified on 100% of tables; $\Delta \text{logit} = 0.00\text{e+}00$; 860 GPU regression tests passed).
- **Sealed Suite Protection**:
  In strict adherence to the preregistered qualification policy, because Gate 3 failed across the cohort, **`sealed_a11_v1` WAS NOT EVALUATED**. Strict data hygiene was preserved intact.
- **Path Forward for Stage A.12**:
  Seed 42 and Seed 43 demonstrated that RoPE causal attention *can* bind multi-entity queries (reaching 75%–87.5% on MP-2), but single-token readout without cross-attention is unstable across random initializations (Seed 44).
  Stage A.12 should investigate **entity query cross-attention routing** (e.g. cross-attending from the query span tokens into the prefix entity representations) to stabilize multi-entity binding across 100% of seeds.
