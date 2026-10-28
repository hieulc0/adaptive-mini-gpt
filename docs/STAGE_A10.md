# Stage A.10 — Independent Foundation Qualification (Multi-Seed Replication, 2×2 Mechanism Control, Sealed Generalization & Persistent-Student Migration)

Status: **COMPLETED; VERDICT: FAIL (DEVELOPMENT REPLICATION CONFIRMED; SEALED & MIGRATION GATES FAILED)**. Continues from tested revision `bac611e` on branch `m104-e-and-m105-a`. Historical parent checkpoints (`be8e289`, `b78b790`, `46cf2ab`, `3e93d72...`) remain strictly immutable. Remote GPU execution on Kaggle Tesla T4 (`sparklinghydraulics0/amg-stage-a10`) executed all 4 phases across 3 independent seeds (`42, 43, 44`). In accordance with preregistered protocol and absolute reporting integrity, because the best candidate neural arm achieved **53.09%** on the independent sealed language suite (failing the $\ge 85.0\%$ gate) and persistent-student migration collapsed prior state-transition accuracy to **25.0%**, **NO CHECKPOINT IS PROMOTED TO PRODUCTION**.

---

## 0. Non-Negotiable Execution Boundary (`AGENTS.md`)
- **LOCAL-SAFE ONLY on host machine**: Code reading/editing, `uv run python -m compileall -q src scripts tests kaggle`, `uv run ruff check`, `uv run pyright`, and explicitly inspected mock/fixture unit tests (`tests/test_stage_a10_contracts.py`) with ZERO neural forward/backward passes, ZERO checkpoint loading, and ZERO CPU training.
- **REMOTE-REQUIRED on Kaggle Tesla T4 GPU**: All 4 phases (Multi-Seed Replication, 2×2 Factorial Matrix, Migration Studies, and Independent Sealed Qualification), fresh-process resume verification, and full regression suites (`AMG_RUN_GPU_TESTS=1 pytest -q`) were executed on Kaggle (`sparklinghydraulics0/amg-stage-a10`). No local CPU fallback was used.

---

## 1. Provenance & Execution Summary

| Parameter | Execution Record |
|---|---|
| **Git SHA (Tested Source)** | `6bc8f10a9c2fbe2ea8dccc518d0efe3bea0c66e7` |
| **Branch** | `m104-e-and-m105-a` |
| **Parent Checkpoint SHA256** | `3e93d7238bfdb706ddfd49ee9903fed8b1e6f577529537f1e21cae12e905b262` |
| **Parent Tensor Digest** | `aed7480606a1a60e38ee7a95d22b5bcf14141f034f76cb45fb59680de277a33e` |
| **Sealed Suite Isolation Digest** | `64daf317c659417327713357449967a9d24c16a3cfe618dd714556c687c14403` (324 cases, 0 overlap against train/pretrain/dev) |
| **Kaggle Kernel** | `sparklinghydraulics0/amg-stage-a10` (Version 2) |
| **Hardware Platform** | Kaggle Tesla T4 GPU (15,102 MiB VRAM), CUDA 12.8, PyTorch 2.11.0+cu128, Python 3.13.15 |
| **Phases Executed** | Phase 1 Contracts (PASS, 18 tests in 9.01s), Phase 2 Stage A.10 Runner (PASS, 0), Phase 2b Resume Check (PASS, $\Delta = 0.00\text{e+}00$), Phase 3 Full Regression (PASS, **842 passed in 613.69s**) |
| **Overall Qualification Verdict** | **`FAIL`** (Best candidate `EXP3_B_1500_SEED_44`: Sealed Full = 53.09% vs $\ge 85.0\%$ gate; Rel-Sensitive = 44.44% vs Word n-gram = 66.67%) |

---

## 2. Scientific Scope & Motivation

Stage A.9 (`bac611e`) revealed strong initial development progress for `EXP3_B` (Fresh RoPE + 1.5k Syntactic Pretraining: 100% train, 100% old-frame accuracy, 90.48% novel-grammar accuracy, 0.00% positional flip rate). However, Stage A.9 left five critical scientific gaps:
1. Familiar development grammar scored **84.92%** (107/126 cases, 1 case below the strict 85.0% threshold).
2. Only a single seed (42) was evaluated.
3. The 2×2 factorial control (`{Absolute wpe, RoPE} × {No Pretrain, Syntactic Pretrain}`) was missing the `Absolute wpe + Syntactic Pretrain` cell, leaving open whether pretraining alone aids absolute models.
4. `EXP3_B` was freshly initialized; raw replacement on the historical continuing parent (`EXP2_D`) collapsed at $t_0$ (67.86%) and achieved only 53.17% novel grammar.
5. No independent sealed language evaluation was performed.

Stage A.10 closed all five questions without relaxing thresholds, altering prompt framing, or replacing historical checkpoints.

---

## 3. Phase 1: Multi-Seed Independent Replication & Error Forensics

### 3.1 EXP3_B Replication Protocol across Seeds 42, 43, 44
The exact `EXP3_B` protocol (RoPE + 1,500 Syntactic Pretraining Steps + 7,200 Supervised Updates) was executed across three independent seeds (`42, 43, 44`), seeding model initialization, pretraining corpus generation, and supervised minibatch sampling independently.

| Model Arm / Seed | Pretrain Time | Supervised Time | Update | Expanded Train Exact | Old Known Exact | Old Renamed Exact | Dev Familiar Exact | Dev Novel Exact | Pos Recall (Nov) | Neg Recall (Nov) | Abs Recall (Nov) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **`EXP3_B_1500` Seed 42** | 26.8s | 264.1s | 6800 | 100.00% | 100.00% | 100.00% | 88.89% | 87.30% | 83.33% | 100.00% | 95.83% |
| | | | 7000 | 100.00% | 100.00% | 100.00% | 88.89% | 87.30% | 83.33% | 100.00% | 95.83% |
| | | | 7200 | **100.00%** | **100.00%** | **100.00%** | **88.89%** | **87.30%** | **83.33%** | **100.00%** | **95.83%** |
| **`EXP3_B_1500` Seed 43** | 26.5s | 264.8s | 6800 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| | | | 7000 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| | | | 7200 | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **`EXP3_B_1500` Seed 44** | 26.6s | 264.4s | 6800 | 100.00% | 100.00% | 100.00% | 88.89% | 92.06% | 88.10% | 100.00% | 100.00% |
| | | | 7000 | 100.00% | 100.00% | 100.00% | 88.89% | 90.48% | 85.71% | 100.00% | 100.00% |
| | | | 7200 | **100.00%** | **100.00%** | **100.00%** | **88.89%** | **90.48%** | **85.71%** | **100.00%** | **100.00%** |

**Replication Finding**:
All three seeds of `EXP3_B_1500` **reproducibly cleared every development admission gate** (`train >= 99%`, `old_known >= 95%`, `old_renamed >= 95%`, `dev_familiar >= 85%`, `dev_novel >= 85%`, and per-status recall $\ge 80\%$ at all snapshot updates). Development familiar accuracy reached **88.89%** in Seed 42 (112/126 cases), **100.00%** in Seed 43 (126/126 cases), and **88.89%** in Seed 44 (112/126 cases).

### 3.2 Forensic Error Dump
Inspection of all misclassified cases across seeds 42, 43, and 44 on `dev_familiar` and `dev_novel`:
- **Seed 43**: 0 errors on both splits (100.0% perfect accuracy).
- **Seed 42**: Exactly 14 errors on `dev_familiar` and 16 errors on `dev_novel`.
  - All 14 errors in `dev_familiar` clustered strictly in **`development_0`** (`"For specimen X, affirmed or denied traits are recorded as prop1; prop2."`).
  - Error type: Property 1 was predicted correctly 100% of the time, while Property 2 was missed (`tgt=[1, 1] pred=[1, 0]` on 4 cases, and `tgt=[0, 1] pred=[0, 0]` on 10 cases).
  - Cause: Punctuation syntax `;` separating properties was slightly under-represented in pretraining syntax, causing attention attenuation on the second property after `;`.
- **Seed 44**: Exactly 14 errors on `dev_familiar` and 12 errors on `dev_novel`, identically clustered in `development_0` on Property 2.

### 3.3 Calibrated Pretraining Horizon Control (`EXP3_B_2500`)
To test whether 2,500 syntactic pretraining steps resolves the semicolon attenuation without architecture modifications:

| Model Arm / Seed | Pretrain Time | Supervised Time | Update | Train Exact | Old Known Exact | Old Renamed Exact | Dev Familiar Exact | Dev Novel Exact | Pos Recall (Nov) | Neg Recall (Nov) | Abs Recall (Nov) | Pass Dev Gate? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **`EXP3_B_2500` Seed 42** | 44.2s | 266.0s | 7200 | 100.00% | 100.00% | 100.00% | 78.57% | 76.19% | 100.00% | 82.50% | 81.25% | **FAIL** (<85%) |
| **`EXP3_B_2500` Seed 43** | 44.2s | 265.7s | 7200 | 100.00% | 100.00% | 100.00% | 96.83% | 96.83% | 98.81% | 97.50% | 100.00% | **PASS** |
| **`EXP3_B_2500` Seed 44** | 44.5s | 266.1s | 7200 | 100.00% | 100.00% | 100.00% | 88.89% | 88.89% | 83.33% | 100.00% | 100.00% | **PASS** |

**Diagnostic Takeaway**: Extending pretraining from 1,500 to 2,500 steps caused Seed 42 to degrade to 78.57% / 76.19%, failing development admission. Over-training the small 5-block Transformer on synthetic syntax without capacity expansion or regularization induces representational rigidity that impairs downstream supervised adaptation. The 1,500-step horizon remains optimal.

---

## 4. Phase 2: Complete 2×2 Factorial Matrix (Position Encoding vs. Syntactic Pretraining)

### 4.1 2×2 Matrix Execution Results
Evaluated under matched capacity (5 trainable blocks), matched supervised updates (7,200), and matched seeds (`42, 43, 44`):
- **Cell 1 (`CELL_1_ABS_NO_PRETRAIN`)**: Absolute `wpe` + 0 pretraining steps.
- **Cell 2 (`CELL_2_ABS_SYNTAX_PRETRAIN`)**: Absolute `wpe` + 1,500 syntactic LM steps (The Missing Control).
- **Cell 3 (`CELL_3_ROPE_NO_PRETRAIN`)**: Half-split RoPE + 0 pretraining steps.
- **Cell 4 (`CELL_4_ROPE_SYNTAX_PRETRAIN`)**: Half-split RoPE + 1,500 syntactic LM steps.

| Factorial Cell | Seed 42 Dev Novel | Seed 43 Dev Novel | Seed 44 Dev Novel | **Cell Mean Dev Novel** | +8 Shift Flip Rate (Mean) | +24 Shift Flip Rate (Mean) | Pretrain GPU Sec | Supervised GPU Sec |
|---|---|---|---|---|---|---|---|---|
| **Cell 1: Absolute + 0 Pretrain** | 38.10% | 50.79% | 55.56% | **48.15%** | 28.31% | 44.31% | 0.0s | 240.2s |
| **Cell 2: Absolute + 1.5k Pretrain** | 46.83% | 63.49% | 38.10% | **49.47%** | 14.02% | 21.30% | 26.6s | 241.1s |
| **Cell 3: RoPE + 0 Pretrain** | 59.52% | 70.63% | 54.76% | **61.64%** | **0.00%** | **0.00%** | 0.0s | 263.8s |
| **Cell 4: RoPE + 1.5k Pretrain** | 87.30% | 100.00% | 90.48% | **92.59%** | **0.00%** | **0.00%** | 26.6s | 264.4s |

### 4.2 Factorial Effect Decomposition
$$
\begin{aligned}
\text{Main Effect of RoPE} &= \frac{(\text{Cell 3} + \text{Cell 4}) - (\text{Cell 1} + \text{Cell 2})}{2} = \frac{(0.6164 + 0.9259) - (0.4815 + 0.4947)}{2} = \mathbf{+28.31\%} \\
\text{Main Effect of Pretraining} &= \frac{(\text{Cell 2} + \text{Cell 4}) - (\text{Cell 1} + \text{Cell 3})}{2} = \frac{(0.4947 + 0.9259) - (0.4815 + 0.6164)}{2} = \mathbf{+16.14\%} \\
\text{Interaction Term} &= (\text{Cell 4} - \text{Cell 3}) - (\text{Cell 2} - \text{Cell 1}) = (0.9259 - 0.6164) - (0.4947 - 0.4815) = \mathbf{+29.63\%}
\end{aligned}
$$

### 4.3 Key Scientific Takeaways from Factorial Control
1. **Syntactic Pretraining Alone Does Not Fix Absolute Models**: Cell 2 achieved **49.47%**, barely a $+1.32\%$ lift over Cell 1 (48.15%). Even with syntactic language pretraining, absolute position embeddings bind representations to absolute coordinates and suffer high positional shift flip rates ($14.02\%$ at $+8$ bytes, $21.30\%$ at $+24$ bytes).
2. **RoPE Alone Is Translation Invariant But Insufficient**: Cell 3 achieves **0.00%** flip rate across all seeds and shifts, lifting novel grammar to $61.64\%$, but still fails the $85.0\%$ threshold without syntactic language initialization.
3. **Super-Additive Interaction (+29.63%)**: The combination of relative position geometry (RoPE) and causal language pretraining is mutually reinforcing. RoPE allows attention to transfer syntactic bindings invariant to position, while pretraining provides the necessary grammatical compositionality.

---

## 5. Phase 3: Persistent Student Continuity & Migration

### 5.1 Migration Study Design
Tests whether the historical parent student (`3e93d723...`) can be migrated from absolute `wpe` to RoPE without catastrophic collapse, evaluated on both attribute frames (84 cases each) and the 8-case historical causal state-transition fixture (`data/fixtures/stage_a_prior_state_tiny.json`):
1. **`MIG_0_HISTORICAL_ABS_BASELINE`**: Unchanged continuing parent checkpoint.
2. **`MIG_1_RAW_ROPE_SWAP`**: Zero `wpe`, enable RoPE on parent weights (Negative Control).
3. **`MIG_2_DISTILLED_BRIDGE_CONTINUATION`**: Preserve historical parent's trained `action_emb`, `transition_net`, `post_norm`, and `state_head` tensors. Align RoPE-pretrained backbone + `concept_proj` to match the historical parent's bottleneck latent vectors ($z_{\text{concept}}$) and attribute logits on the permitted rehearsal buffer via a 500-step bounded KL/MSE distillation bridge, followed by the 7,200-step curriculum.

### 5.2 Migration Empirical Results

| Migration Arm | $t_0$ Old Known | $t_0$ Old Renamed | $t_0$ Prior State | Post-7200 Old Known | Post-7200 Old Renamed | Post-7200 Dev Fam | Post-7200 Dev Nov | Post-7200 Prior State | Wall Time |
|---|---|---|---|---|---|---|---|---|---|
| **`MIG_0_HISTORICAL_ABS`** | 100.00% | 100.00% | 50.00% (4/8) | N/A | N/A | N/A | N/A | N/A | 0.0s |
| **`MIG_1_RAW_ROPE_SWAP`** | 67.86% | 65.48% | 50.00% (4/8) | 100.00% | 100.00% | 68.25% | 65.08% | 50.00% (4/8) | 226.2s |
| **`MIG_2_DISTILLED_BRIDGE`** | 72.62% | 72.62% | 50.00% (4/8) | 100.00% | 100.00% | 80.95% | 80.95% | **25.00% (2/8)** | 264.3s |

### 5.3 Scientific Findings on Migration
1. **$t_0$ Representational Disruption**: Raw swap (`MIG_1`) drops attribute retention to $67.86\%$. Distillation bridging (`MIG_2`) cushions the drop slightly to $72.62\%$, but cannot achieve zero-shot preservation.
2. **Catastrophic Forgetting of Prior Causal State Competence**: While `MIG_2` recovered 100% attribute retention after 7,200 supervised updates and reached 80.95% novel grammar, its prior state-transition accuracy on `stage_a_prior_state_tiny.json` collapsed from **50.00% (4/8) down to 25.00% (2/8)**.
3. **Implication**: Backbone representation space shifts under RoPE migration cannot be bridged for frozen downstream heads (`transition_net`, `state_head`) purely through latent alignment on rehearsal data. Migrating a persistent student requires joint multi-task rehearsal of all historical tasks (both causal transitions and attribute extractions).

---

## 6. Phase 4: Independent Sealed Language Qualification

### 6.1 Sealed Suite Construction & Isolation Digest
- **Case Count**: 324 cases across 3 seeds (108 cases/seed), frozen prior to evaluation.
- **Isolation Certification**: Verified zero overlap against all training, rehearsal, pretraining, and development strings and templates (Digest: `64daf317c659417327713357449967a9d24c16a3cfe618dd714556c687c14403`).
- **Compositional Stressors**:
  - 4 brand-new template families never seen in training or pretraining.
  - Disjoint novel entity names: `Zor, Kex, Plin, Vex, Taro, Nul`.
  - **Relationship-Sensitive Long-Range & Distractor Negation**: Clauses where negation tokens (`"not"`, `"never"`) appear in irrelevant introductory clauses (`"Although the outer casing is not heavy..."`) or separated by 4–5 word modifier phrases.

### 6.2 Empirical Evaluation on Sealed Suite

| Model / Control Arm | Sealed Full Exact (324 cases) | Sealed Standard Exact (243 cases) | Sealed Rel-Sensitive Exact (81 cases) | Micro F1 (Full) |
|---|---|---|---|---|
| **Majority Baseline** | 44.44% | 44.44% | 44.44% | 0.0000 |
| **Word 1–3-Gram Linear Control** | 53.70% | 49.38% | **66.67%** | 0.5484 |
| **Bag-of-Words Linear Control** | **65.74%** | **70.37%** | 51.85% | 0.6543 |
| **`EXP3_B_1500_SEED_42`** | 51.54% | 52.26% | 49.38% | 0.2598 |
| **`EXP3_B_1500_SEED_43`** | 52.16% | 52.26% | 51.85% | 0.5534 |
| **`EXP3_B_1500_SEED_44` (Best)** | **53.09%** | 55.97% | **44.44%** | 0.3759 |
| **`EXP3_B_2500_SEED_43`** | 49.69% | 51.44% | 44.44% | 0.4755 |
| **`EXP3_B_2500_SEED_44`** | 47.84% | 48.97% | 44.44% | 0.1765 |

### 6.3 Forensic Breakdown of the Sealed Failure
1. **Failure Against the 85.0% Gate**: All neural candidates scored between **47.84% and 53.09%** on the sealed suite, failing the strict 85.0% gate by over 31 percentage points.
2. **Defeat by Linear Lexical Controls**: The Bag-of-Words linear model achieved **65.74%** on Sealed Full, outperforming every neural candidate.
3. **Severe Vulnerability to Distractor Clauses (`relationship_sensitive`)**:
   - On sentences with distractor negation in an introductory clause (`"Although the outer casing is not heavy, for object Zor the primary core remains decidedly flammable..."`), the best neural candidate collapsed to **44.44%** (chance level).
   - In contrast, the Word 1–3-gram control scored **66.67%**, because n-grams do not reach across long clauses to misattribute the negation token.
4. **Root Cause Diagnosis**:
   - The 1,500-step syntactic pretraining corpus exposed the Transformer to short declarative sentences, but contained no hierarchical complex sentences with subordinating conjunctions (`"Although"`, `"Even though"`), adverbial distractor clauses, or long-distance dependency modifiers.
   - Without clause-level dependency pretraining or syntactic supervision, the mini-GPT self-attention mechanism acts as an unconstrained bag-of-tokens associator: when the token `"not"` is present anywhere in the context, attention heads misattribute the negation to the target property.

---

## 7. Binding Qualification Gate Evaluation

| Gate | Description | Preregistered Threshold | Observed Value | Verdict |
|---|---|---|---|---|
| **Gate 1** | Multi-Seed Replication | `dev_novel >= 85.0%`, `dev_fam >= 85.0%`, recalls $\ge 80\%$ across seeds 42, 43, 44 | Seed 42: 87.30% / 88.89%<br>Seed 43: 100.0% / 100.0%<br>Seed 44: 90.48% / 88.89% | **PASS** |
| **Gate 2** | Factorial Mechanism Separation | $\text{Main}_{\text{RoPE}} > 0$, $\text{Main}_{\text{Pretrain}} > 0$, $\text{Interaction} > 0$, Flip Rate $\le 5\%$ | $\text{RoPE} = +28.31\%$, $\text{Pretrain} = +16.14\%$, $\text{Int} = +29.63\%$, RoPE Flips $= 0.00\%$ | **PASS** |
| **Gate 3** | Persistent Student Migration | $MIG_2$ Retains State $\ge 85\%$ & Attributes $\ge 95\%$, Dev Novel $\ge 85\%$ | Old Known = 100%, Old Renamed = 100%, Dev Nov = 80.95% (<85%), State = **25.0%** (collapsed) | **FAIL** |
| **Gate 4** | Independent Sealed Language Qualification | Sealed Full $\ge 85.0\%$, Rel-Sensitive $>$ N-Gram Control (66.67%) | Best Candidate = **53.09%** (<85.0%), Rel-Sensitive = **44.44%** (<66.67%) | **FAIL** |
| **Gate 5** | Continuity, Replay & Integrity | Deterministic continuation $\Delta = 0.00\text{e+}00$, no test regressions ($\ge 841$) | $\Delta = 0.00\text{e+}00$, **842 passed** in 613.69s, 0 leaks | **PASS** |

### Overall Verdict: `FAIL`
In accordance with non-negotiable scientific integrity rules, **no checkpoint is promoted to production**. The defect register has been updated with findings FDR-007 and FDR-008.

---

## 8. Summary of Architectural & Empirical Progress

```
Stage A.7 (Absolute wpe, frozen lower blocks, no pretrain)
  ├── Expanded Train: 78.70% (Optimization collapse)
  └── Novel Grammar:  10.32%

Stage A.8 (Top-2 block unfreezing + Cosine floor)
  ├── Expanded Train: 100.00%
  ├── Retention:      100.00%
  └── Novel Grammar:  34.13% (Absolute positional coordinate lock-in)

Stage A.9 (RoPE + 1.5k Syntactic LM Pretraining, single seed 42)
  ├── Expanded Train: 100.00%
  ├── Retention:      100.00%
  ├── Dev Novel:      90.48% (Positional flip rate 0.00%)
  └── Dev Familiar:   84.92% (107/126 cases)

Stage A.10 (Multi-Seed Replication, 2×2 Factorial, Migration, Sealed Suite)
  ├── Dev Replication: PASS across Seeds 42, 43, 44 (87.30% – 100.00%)
  ├── 2×2 Factorial:   Super-additive interaction confirmed (+29.63%)
  ├── Migration (MIG2): FAILED (State retention collapsed to 25.00%, Dev Nov 80.95%)
  └── Sealed Suite:    FAILED (53.09% Full, 44.44% Rel-Sensitive vs 66.67% N-Gram)
```

---

## 9. Next Steps for Stage A.11 / Milestone M105

1. **Syntactic Substrate Curriculum Expansion (Addressing FDR-007)**:
   The synthetic pretraining corpus must be augmented with multi-clause hierarchical sentences containing subordinating conjunctions (`"Although"`, `"While"`, `"Whereas"`), prepositional phrases, and distractor negation. The model must learn syntactic clause-boundary isolation prior to concept extraction.
2. **Unified Multi-Task Rehearsal for Foundation Migration (Addressing FDR-008)**:
   Attempting to migrate the persistent student by aligning attribute representations alone catastrophically destroys causal state-transition knowledge in downstream heads. Stage A.11 must incorporate an integrated rehearsal buffer spanning all historical competencies (causal environment transitions + attribute frames) during foundation migration.
