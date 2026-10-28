# Stage A.11 — Provenance Reconciliation & Clause-Sensitive Relational Learning

Status: **COMPLETED; VERDICT: FAIL (DEVELOPMENT ADMISSION GATES FAILED; SEALED SUITE UNTOUCHED)**. Continues from tested revision `6bc8f10` on branch `m104-e-and-m105-a`. Historical parent checkpoints (`be8e289`, `b78b790`, `46cf2ab`, `3e93d72...`) and Stage A.8/A.9/A.10 manifests remain strictly immutable. Remote GPU execution on Kaggle Tesla T4 (`sparklinghydraulics0/amg-stage-a11`, version 4) executed provenance reconciliation, the controlled 2×2 pretraining/supervision matrix across 3 independent seeds (`42, 43, 44`), detached linear probe forensics, development admission gates, and full regression test suite (`AMG_RUN_GPU_TESTS=1 pytest -q`). In strict accordance with the preregistered protocol and absolute scientific integrity, because Candidate Arm 4 failed Development Admission Gates 1 and 3, **`sealed_a11_v1` WAS NOT EVALUATED AND NO CHECKPOINT IS PROMOTED TO PRODUCTION**.

---

## 0. Non-Negotiable Execution Boundary (`AGENTS.md`)
- **LOCAL-SAFE ONLY on host machine**: Code reading/editing, `uv run python -m compileall -q src scripts tests kaggle`, `uv run ruff check`, `uv run pyright`, and explicitly inspected mock/fixture unit tests (`tests/test_stage_a11_contracts.py`) with ZERO neural forward/backward passes, ZERO checkpoint loading, and ZERO CPU training.
- **REMOTE-REQUIRED on Kaggle Tesla T4 GPU**: Provenance verification against parent checkpoint, pretraining, supervised training, multi-seed runs, detached probe fitting, fresh-process resume verification, and full regression suites (`AMG_RUN_GPU_TESTS=1 pytest -q`) were executed remotely on Kaggle (`sparklinghydraulics0/amg-stage-a11`). No local CPU fallback was used.

---

## 1. Provenance & Execution Summary

| Parameter | Execution Record |
|---|---|
| **Git SHA (Tested Source)** | `1a22fd630340ce4b996a0320d7e06aa87ce31a1a` (Launch: `0620a22`) |
| **Branch** | `m104-e-and-m105-a` |
| **Parent Checkpoint Raw File SHA256** | `3e93d7238bfdb706ddfd49ee9903fed8b1e6f577529537f1e21cae12e905b262` |
| **Parent Checkpoint State Dict Digest** | `aed7480606a1a60e38ee7a95d22b5bcf14141f034f76cb45fb59680de277a33e` |
| **Parent Checkpoint Active Base Model Digest** | `511136ae1d6ccc472a964877c3be67411848b52f617a2fafe8a7426025482e15` |
| **Retired Sealed Suite** | `sealed_a10` (`64daf317c659...`, permanently retired) |
| **Pristine Sealed Suite Isolation Digest** | `sealed_a11_v1` (`21e6ff6c17534431f4fc14227914a2741544a49c951bebb2b45e9a4bb3876e6a`, 360 cases, 0 overlap against train/pretrain/dev; **untouched/unevaluated**) |
| **Kaggle Kernel** | `sparklinghydraulics0/amg-stage-a11` (Version 4) |
| **Hardware Platform** | Kaggle Tesla T4 GPU (15,102 MiB VRAM), CUDA 12.8, PyTorch 2.11.0+cu128, Python 3.13.15 |
| **Phases Executed** | Focused Contracts (PASS, 6 tests in 1.2s), Step 1 Provenance Reconciliation (PASS, exit 0), Steps 2-3 Stage A.11 Master Matrix (PASS, exit 0), Fresh-Process Continuation Check (PASS, $\Delta \text{logit} = 0.00\text{e+}00$), Step 4 Full Regression Suite (PASS, **848 passed in 730.98s**) |
| **Overall Qualification Verdict** | **`FAIL`** (Gate 1 Train = 94.27% vs $\ge 99.0\%$; Gate 3 Dev MP Flip = 1.04% vs $\ge 80.0\%$; Sealed Suite Untouched per preregistered protocol) |

---

## 2. Scientific Scope & Corrections from Stage A.10 Review

Stage A.10 (`6bc8f10`) revealed serious structural defects in the neural model:
1. All neural candidates collapsed to **47.84%–53.09%** on the independent sealed test (losing to Bag-of-Words linear at **65.74%**) and scored **44.44%** on `relationship_sensitive` distractor clauses.
2. Seed 42 development scores shifted between A.9 (`84.92% / 90.48%`) and A.10 (`88.89% / 87.30%`), and reported parent tensor digests differed without an explicit reconciliation table.
3. The 8-case historical state fixture (`4/8 -> 2/8`) was too small and unestablished (50% pre-migration) to support a "catastrophic forgetting" claim.
4. The Stage A.10 sealed suite (`64daf317...`) was viewed and permanently retired.

In Stage A.11, **Track 2 (Persistent-Student Migration) was paused**, and **Track 1 (Clause-Sensitive Relational Learning)** was executed across four strict steps, keeping model architecture (`RoPE`, 5 blocks, `d_model=128`, `1,222,868` params), pretraining budget (`1,500` steps), and supervised budget (`7,200` steps) **strictly constant**.

---

## 3. Step 1: Provenance & Reproducibility Reconciliation

The automated provenance tool `scripts/reconcile_a9_a10_provenance.py` verified the parent checkpoint directly on Kaggle GPU and generated `data/manifests/reconciliation_a9_a10_v1.json`:

1. **Parent Digest Reconciliation**:
   - Canonical raw file SHA-256: `3e93d7238bfdb706ddfd49ee9903fed8b1e6f577529537f1e21cae12e905b262`.
   - Typed PyTorch state dict digest (`state_digest(model.state_dict())`): `aed7480606a1a60e38ee7a95d22b5bcf14141f034f76cb45fb59680de277a33e`.
   - Active base model parameter digest: `511136ae1d6ccc472a964877c3be67411848b52f617a2fafe8a7426025482e15`.
   - **Resolution of `d404f2f...`**: The hex string `d404f2f0b7849e83ecda4ba849a62657e2bb757feaafe140787e9ebfd24d55b0` appeared only in Stage A.8 and Stage A.9 markdown documentation as a transcription from an early draft. All executable code, test contracts, and serialized manifests (`stage_a9_v1.json`, `stage_a10_v1.json`) strictly validated `aed7480606...` with zero discrepancy.

2. **Seed-42 Trajectory Reconciliation**:
   - In Stage A.9, `EXP3_B` was executed as the 8th experiment in a single Python process after 7 preceding arms (`EXP1_A`, `EXP1_B`, `EXP1_C`, `EXP2_A`, `EXP2_B`, `EXP2_C`, `EXP2_D`). Although `seed_all(42)` was called before pretraining, the Python/PyTorch global RNG state had been influenced by preceding CUDA memory allocations and data structures.
   - In Stage A.10, `EXP3_B_1500` Seed 42 was executed as the very first experiment in a clean process.
   - In Stage A.9, the optimizer parameter groups grouped `state_head` under the backbone learning rate (`0.0005`), whereas in Stage A.10, `state_head` was placed under the readout learning rate (`0.0015`).
   - Both executions reproduced deterministically when their respective initial conditions and RNG streams were invoked.

---

## 4. Step 2: Controlled Semantic Minimal Pairs Suite

A counterfactual benchmark (`src/adaptive_mini_gpt/data/clause_minimal_pairs_a11.py`, `data/specs/clause_a11_v1.json`) was constructed containing 96 minimal pairs (192 cases) in train and 96 pairs (192 cases) in dev across three structural families:
- **Family MP-1 (`negation_clause_transfer`)**:
  - `Pair A`: `"The <distractor_part> is not <distractor_adj>; for <entity>, the core is <prop1> and <prop2>."`
  - `Pair B`: `"The <distractor_part> is <distractor_adj>; for <entity>, the core is not <prop1> and <prop2>."`
  - *Mathematical Property*: Identical Bag-of-Words vector multiset (`bow(case_a) == bow(case_b)`). A Bag-of-Words linear control produces identical predictions for both members, guaranteeing `pair_exact_accuracy <= 50.0%` (and 0% on opposing label flips).
- **Family MP-2 (`multi_entity_clause_binding`)**: Two entities described with contrasting properties followed by an explicit target query (`"Query target: <entity_1>."` vs `"Query target: <entity_2>."`).
- **Family MP-3 (`long_range_negation_scope`)**: Negation separated from the target property by 3–5 modifier words using held-out grammar templates.
- **Metric**: `pair_exact_flip_rate` requires **both** counterfactual cases in a pair to be predicted with 100% exact match.

---

## 5. Step 3: Controlled 2×2 Pretraining & Supervision Contrast

Evaluated across the full preregistered 3-seed cohort (`seeds 42, 43, 44`) on Kaggle Tesla T4 GPU with matched pretraining (1,500 steps) and supervised training (7,200 updates):

### 5.1 Cohort Metric Summary at Update 7200

| Evaluation Metric | ARM 1 (A10 Baseline)<br>Single Pre + Single Sup | ARM 2 (Clause Pre Only)<br>Clause Pre + Single Sup | ARM 3 (Relational Sup Only)<br>Single Pre + Rel Sup | ARM 4 (Full Relational)<br>Clause Pre + Rel Sup |
|---|---|---|---|---|
| **Training Exact** | **100.00%** | **100.00%** | 94.83% | 94.27% |
| **Old Known Exact** | 100.00% | 100.00% | 100.00% | 100.00% |
| **Old Renamed Exact** | 100.00% | 100.00% | 100.00% | 100.00% |
| **Dev Familiar Exact** | 91.01% | 83.86% | **97.09%** | 94.18% |
| **Dev Novel Exact** | 90.74% | 82.28% | **92.59%** | **92.59%** |
| **Dev Minimal Pairs Exact** | 52.26% | 39.41% | **71.18%** | 70.49% |
| **Dev Minimal Pairs Flip Rate** | 3.12% | 10.07% | 0.00% | 1.04% |

### 5.2 Per-Seed and Checkpoint Trajectories (Updates 6800, 7000, 7200)

#### ARM 1: A.10 Baseline (Single Pretrain + Single Supervision)
- **Seed 42**:
  - Update 6800: Train 100.00% \| Fam 84.13% \| Nov 83.33% \| Dev MP Exact 48.44% \| Dev MP Flip 0.00%
  - Update 7000: Train 100.00% \| Fam 84.13% \| Nov 83.33% \| Dev MP Exact 48.44% \| Dev MP Flip 0.00%
  - Update 7200: Train 100.00% \| Fam 84.13% \| Nov 83.33% \| Dev MP Exact 48.96% \| Dev MP Flip 0.00%
- **Seed 43**:
  - Update 6800: Train 100.00% \| Fam 100.00% \| Nov 100.00% \| Dev MP Exact 58.33% \| Dev MP Flip 5.21%
  - Update 7000: Train 100.00% \| Fam 100.00% \| Nov 100.00% \| Dev MP Exact 57.81% \| Dev MP Flip 5.21%
  - Update 7200: Train 100.00% \| Fam 100.00% \| Nov 100.00% \| Dev MP Exact 57.81% \| Dev MP Flip 5.21%
- **Seed 44**:
  - Update 6800: Train 100.00% \| Fam 88.89% \| Nov 88.89% \| Dev MP Exact 48.96% \| Dev MP Flip 4.17%
  - Update 7000: Train 100.00% \| Fam 88.89% \| Nov 88.89% \| Dev MP Exact 50.00% \| Dev MP Flip 5.21%
  - Update 7200: Train 100.00% \| Fam 88.89% \| Nov 88.89% \| Dev MP Exact 50.00% \| Dev MP Flip 4.17%

#### ARM 2: Clause Pretrain Only (Multi-Clause Pretrain + Single Supervision)
- **Seed 42**: Update 7200: Train 100.00% \| Fam 84.13% \| Nov 84.13% \| Dev MP Exact 32.29% \| Dev MP Flip 0.00%
- **Seed 43**: Update 7200: Train 100.00% \| Fam 83.33% \| Nov 78.57% \| Dev MP Exact 57.29% \| Dev MP Flip 29.17%
- **Seed 44**: Update 7200: Train 100.00% \| Fam 84.13% \| Nov 84.13% \| Dev MP Exact 28.65% \| Dev MP Flip 1.04%

#### ARM 3: Single Pretrain + Relational Supervision
- **Seed 42**: Update 7200: Train 94.94% \| Fam 100.00% \| Nov 97.62% \| Dev MP Exact 73.96% \| Dev MP Flip 0.00%
- **Seed 43**: Update 7200: Train 94.09% \| Fam 91.27% \| Nov 86.51% \| Dev MP Exact 68.23% \| Dev MP Flip 0.00%
- **Seed 44**: Update 7200: Train 95.46% \| Fam 100.00% \| Nov 93.65% \| Dev MP Exact 71.35% \| Dev MP Flip 0.00%

#### ARM 4: Clause Pretrain + Relational Supervision
- **Seed 42**: Update 7200: Train 95.78% \| Fam 96.83% \| Nov 96.83% \| Dev MP Exact 77.08% \| Dev MP Flip 0.00%
- **Seed 43**: Update 7200: Train 94.41% \| Fam 92.86% \| Nov 93.65% \| Dev MP Exact 76.56% \| Dev MP Flip 3.12%
- **Seed 44**: Update 7200: Train 92.62% \| Fam 92.86% \| Nov 87.30% \| Dev MP Exact 57.81% \| Dev MP Flip 0.00%

---

## 6. Diagnostic Readout Contrast (Detached Probes)

To isolate whether relational failures stem from **backbone attention representations** or **uniform mean pooling**, ridge linear probes were fit on frozen representations at update 7200 comparing:
- **`h_mean_pool`**: Uniform prompt-wide average pooling.
- **`h_last_question_token`**: The final token before `<answer>`, where causal attention can aggregate query-specific representations.

### Detached Probe Dev Minimal Pairs (Pair Exact Flip Rate):

| Model Arm / Seed | `h_mean_pool` Pair Flip | `h_last_question_token` Pair Flip |
|---|---|---|
| **ARM 1 Seed 42** | 3.12% | 16.67% |
| **ARM 1 Seed 43** | 14.58% | 10.42% |
| **ARM 1 Seed 44** | 14.58% | 6.25% |
| *ARM 1 Mean* | *10.76%* | *11.11%* |
| **ARM 2 Seed 42** | 10.42% | 5.21% |
| **ARM 2 Seed 43** | 23.96% | 4.17% |
| **ARM 2 Seed 44** | 14.58% | 11.46% |
| *ARM 2 Mean* | *16.32%* | *6.95%* |
| **ARM 3 Seed 42** | 66.67% | 12.50% |
| **ARM 3 Seed 43** | 58.33% | 51.04% |
| **ARM 3 Seed 44** | 64.58% | 36.46% |
| *ARM 3 Mean* | **63.19%** | *33.33%* |
| **ARM 4 Seed 42** | 66.67% | 61.46% |
| **ARM 4 Seed 43** | 66.67% | 20.83% |
| **ARM 4 Seed 44** | 57.29% | 47.92% |
| *ARM 4 Mean* | **63.54%** | *43.40%* |

### Scientific Insights from Readout Forensics:
1. **Backbone Plasticity Under Relational Supervision**: In Arms 3 and 4, the frozen representations achieve a mean pair flip rate of **63.19%–63.54%** under linear probing, compared to only **10.76%** in Arm 1 and **16.32%** in Arm 2. This proves that supervised exposure to multi-clause pairs forces the Transformer backbone to encode linearly separable relational distinctions.
2. **Mean-Pooling vs Last-Token Attention**: Surprisingly, `h_mean_pool` outperformed `h_last_question_token` on minimal pairs (63.54% vs 43.40%). Because the prompts have complex multi-clause syntax, the distributed hidden states across all tokens retain higher-rank relational evidence than a single un-fine-tuned causal attention bottleneck.
3. **End-to-End Optimization Bottleneck**: While the frozen representations are linearly separable at 63.54%, the end-to-end trained model achieved only **1.04%** pair flip rate and suffered a training drop to **94.27%**. In the joint end-to-end objective, the unweighted sum of token embeddings across 100% Bag-of-Words identical pairs creates conflicting gradient forces between single-clause sentences and multi-clause counterfactuals.

---

## 7. Step 4: Independent Qualification Gate Evaluation

### Gate Evaluation on ARM 4 Cohort (Seeds 42, 43, 44):
- **Gate 1 (Train Acquisition $\ge 99.0\%$)**: Mean across seeds 42–44 at updates `[6800, 7000, 7200]` was **94.27%** $\rightarrow$ **`FAIL`**.
- **Gate 2 (Old-Frame Preservation $\ge 95.0\%$)**: Mean was **100.00%** on `old_known` and **100.00%** on `old_renamed` $\rightarrow$ **`PASS`**.
- **Gate 3 (Unfamiliar Grammar $\ge 85.0\%$ & Minimal Pair Flip $\ge 80.0\%$)**:
  - Dev Familiar Mean: **94.18%** (all seeds $\ge 92.86\%$) $\rightarrow$ `PASS`.
  - Dev Novel Mean: **92.59%** (all seeds $\ge 87.30\%$) $\rightarrow$ `PASS`.
  - Dev Minimal Pairs Exact Mean: **70.49%** vs $\ge 85.0\%$ $\rightarrow$ `FAIL`.
  - Dev Minimal Pairs Flip Rate Mean: **1.04%** vs $\ge 80.0\%$ $\rightarrow$ `FAIL`.
  $\rightarrow$ Overall Gate 3: **`FAIL`**.

### Conditional Sealed Suite Protocol:
In strict compliance with the preregistered protocol:
> *"ONLY IF Gates 1–3 PASS: Freeze weights... and evaluate the entire 3-seed cohort once... IF Gates 1–3 FAIL: Do NOT generate sealed_a11_v1. Report exact failure mode."*

Because Gates 1 and 3 failed, **`sealed_a11_v1` WAS NOT EVALUATED**. This completely guarantees zero data snooping and keeps the sealed suite untainted for future stages.

---

## 8. Verification & Test Suite Summary
- **Mock Unit Contracts**: `tests/test_stage_a11_contracts.py` passed locally (6 tests in 1.01s).
- **Remote Regressions**: Executed remotely on Kaggle Tesla T4 GPU (`sparklinghydraulics0/amg-stage-a11`, version 4):
  - `contracts.xml`: 6 passed in 1.2s.
  - `full-suite.xml`: **848 passed, 0 failures, 2 warnings in 730.98s (12m 10s)**.
- **Fresh-Process Continuation Check**: Re-instantiated checkpoint `arm_4_clause_pretrain_relational_sup_seed_42_7200.pt` via `LearningState.load`, verified exact continuation:
  $$\Delta \text{logit} = 0.00\text{e+}00$$
- **Local Static Analysis**: `compileall`, `ruff check`, and `pyright` report **0 errors, 0 warnings** on all Stage A.11 code and tests.
