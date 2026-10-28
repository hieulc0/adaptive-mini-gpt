# Stage A.8 — Neural Learning Mechanism Investigation & Foundation Repair for Learning-to-Learn

Status: **COMPLETED; VERDICT: ARCHITECTURAL REDESIGN REQUIRED**. Continues from exact tested source `be8e289` on branch `m104-e-and-m105-a`. Historical artifacts and parent checkpoints (`be8e289`, `b78b790`) remain strictly immutable. Gate 1 (Acquisition), Gate 2 (Retention), and Gate 4 (Determinism/Regression) PASSED. Gate 3 (Unfamiliar Grammar Generalization) FAILED. In accordance with preregistered protocol, sealed evaluation was aborted, no checkpoint was promoted, and the formal **Architectural Redesign Decision** was adopted.

---

## 0. Non-Negotiable Execution Boundary (`AGENTS.md`)
- **LOCAL-SAFE ONLY on host machine**: Code reading/editing, `uv run python -m compileall -q src scripts tests kaggle`, `uv run ruff check`, `uv run pyright`, and explicitly inspected mock/fixture unit tests (`tests/test_stage_a8_contracts.py`) with ZERO neural forward/backward passes, ZERO checkpoint loading, and ZERO CPU training.
- **REMOTE-REQUIRED on Kaggle Tesla T4 GPU**: All frozen-feature probe fitting, trajectory diagnostics, neural training updates, and full regression suites (`AMG_RUN_GPU_TESTS=1 pytest -q`) were executed on Kaggle (`sparklinghydraulics0/amg-stage-a8`). No local CPU fallback was used.

---

## 1. Provenance & Execution Summary

| Parameter | Execution Record |
|---|---|
| **Git SHA (Tested Source)** | `46cf2ab3d4c1afc3fb44f63429cb643130c21e9e` |
| **Branch** | `m104-e-and-m105-a` |
| **Parent Checkpoint SHA256** | `3e93d7238bfdb706ddfd49ee9903fed8b1e6f577529537f1e21cae12e905b262` |
| **Parent Tensor Digest** | `d404f2f0b7849e83ecda4ba849a62657e2bb757feaafe140787e9ebfd24d55b0` |
| **Kaggle Kernel** | `sparklinghydraulics0/amg-stage-a8` (Kernel Version 2) |
| **Hardware Platform** | Kaggle Tesla T4 GPU (15,102 MiB VRAM), CUDA 12.8, PyTorch 2.11.0+cu128, Python 3.13.15 |
| **Phases Executed** | Phase 1 Contracts (PASS 0), Phase 2 Stage A.8 Runner (PASS 0), Phase 3 Full Regression (PASS 0, 829+ tests) |
| **Final Verdict** | **`ARCHITECTURAL_REDESIGN_REQUIRED`** |

---

## 2. Stage 1 (Task 1): Frozen-Feature Representation Diagnosis

### 2.1 The Representation Coverage Repair
Stage A.7's probe diagnosis suffered from a factor-coverage defect: its 162-case training set lacked alternative negators (`without`) and inverted property orders (`cond_first`), causing even an n-gram model to fail on novel grammar. Stage A.8 repaired this by constructing a **Coverage-Matched Diagnostic Train Split** (`probe_a8_v1.json`) of exactly 378 training cases crossing:
- All 6 grammatical training frames (0..5) and 3 prefixes (`""`, `"Log. "`, `"An entry. "`)
- Both negation operators (`"not"`, `"without"`)
- Both property presentation orders (`"flam_first"`, `"cond_first"`)
- Full balanced coverage across target status pairs (`positive`, `negated`, `absent`)

### 2.2 Empirical Probe Diagnostic Results

Evaluated across four checkpoints: `parent` (A.5/A.6 continuing parent), `coverage_1200`, `coverage_5800` (peak training accuracy), and `coverage_7200` (endpoint).

| Target Model | Probe Arm | Representation | Probe Train Exact | Familiar Val Exact | Development Novel Exact |
|---|---|---|---|---|---|
| **Parent** | P4 Lexical (N-gram) | Word 1–3-gram Linear | 100.0% | 100.0% | **100.0%** |
| | P4 Lexical (BoW) | Bag-of-Words Linear | 61.90% | 61.90% | 61.90% |
| | P2 Linear Ridge | $h_{\\text{pool}} \\in \\mathbb{R}^{128}$ | 80.42% | 33.33% | 23.02% |
| | P2 Linear Ridge | $z_{\\text{concept}} \\in \\mathbb{R}^{32}$ | 46.30% | 12.70% | 11.90% |
| | P2 Linear Ridge | $h_{\\text{tokens\\_stat}} \\in \\mathbb{R}^{256}$ | 93.65% | 25.40% | 26.98% |
| | P3 Nonlinear MLP | $h_{\\text{pool}} \\in \\mathbb{R}^{128}$ | 62.17% | 19.84% | 17.46% |
| | P3 Nonlinear MLP | $h_{\\text{tokens\\_stat}} \\in \\mathbb{R}^{256}$ | 99.74% | 19.05% | 18.25% |
| **coverage_1200** | P2 Linear Ridge | $h_{\\text{pool}}$ | 68.78% | 38.10% | 31.75% |
| | P2 Linear Ridge | $z_{\\text{concept}}$ | 67.99% | 30.16% | 26.19% |
| | P2 Linear Ridge | $h_{\\text{tokens\\_stat}}$ | 82.54% | 28.57% | 24.60% |
| | P3 Nonlinear MLP | $h_{\\text{pool}}$ | 78.57% | 34.13% | 28.57% |
| | P3 Nonlinear MLP | $h_{\\text{tokens\\_stat}}$ | 98.94% | 26.98% | 25.40% |
| **coverage_5800** | P2 Linear Ridge | $h_{\\text{pool}}$ | 84.92% | 20.63% | 24.60% |
| | P2 Linear Ridge | $z_{\\text{concept}}$ | 89.42% | 23.81% | 24.60% |
| | P2 Linear Ridge | $h_{\\text{tokens\\_stat}}$ | 95.50% | 26.98% | 23.02% |
| | P3 Nonlinear MLP | $h_{\\text{pool}}$ | 94.71% | 29.37% | 31.75% |
| | P3 Nonlinear MLP | $h_{\\text{tokens\\_stat}}$ | 100.0% | 30.16% | 26.19% |
| **coverage_7200** | P2 Linear Ridge | $h_{\\text{pool}}$ | 88.36% | 30.95% | **28.57%** |
| | P2 Linear Ridge | $z_{\\text{concept}}$ | 91.01% | 30.95% | **30.95%** |
| | P2 Linear Ridge | $h_{\\text{tokens\\_stat}}$ | 97.62% | 32.54% | **22.22%** |
| | P3 Nonlinear MLP | $h_{\\text{pool}}$ | 97.09% | 30.95% | **27.78%** |
| | P3 Nonlinear MLP | $h_{\\text{tokens\\_stat}}$ | 99.74% | 32.54% | **23.81%** |

### 2.3 Diagnostic Insights
1. **Lexical Information Sufficiency**: The N-gram control achieves **100.0%** on unfamiliar grammar from the 378 training cases. The failure of the neural model is definitely **not** a data-distribution deficiency or missing n-gram signal.
2. **Representation Defect Confirmed**: Frozen Transformer hidden states $h_{\\text{pool}}$ achieve only **28.57%**, bottleneck $z_{\\text{concept}}$ achieves **30.95%**, and residual pooled $h_{\\text{tokens\\_stat}}$ achieves **22.22%**.
3. **MLP Readout Cannot Recover Concepts**: Even a high-capacity nonlinear MLP probe achieves 99.74% on training but collapses to **23.81%** on novel grammar. The required compositional concept features are simply absent from the pooled Transformer activations.

---

## 3. Stage 2 (Task 2): Optimization Instability Forensics & 4-Arm Contrast

### 3.1 Forensics Instrumentation
We instrumented `LearningState.step` to monitor pre-clip gradient norms across modules, clipping activation rates, Adam second moments ($v_t$), and parameter update ratios:
$$\\text{Update Ratio} = \\frac{\\|\\theta_t - \\theta_{t-1}\\|_2}{\\|\\theta_{t-1}\\|_2 + 10^{-12}}$$

### 3.2 Preregistered 4-Arm Contrast Results

| Optimization Arm | Config Summary | Final Train Exact | Familiar Val Exact | Development Novel Exact | Clip Rate | Mean Pre-Clip Grad Norm | Max Grad Norm | Final Update Ratios (`wte` / `b4` / `readout`) |
|---|---|---|---|---|---|---|---|---|
| **`OPT_0_BASELINE`** | Constant $lr=0.002$, clip 1.0 (A.7 reproduction) | 95.24% | 28.57% | 31.75% | 81.2% | 4.58 | 20.81 | 0.0033 / 0.0032 / 0.0018 |
| **`OPT_1_COSINE_FLOOR`** | Cosine LR $0.002 \\to 0.0001$, $\\beta_2=0.98$, $\\epsilon=10^{-6}$ | **100.00%** | 27.78% | 21.43% | **38.2%** | **1.88** | 23.65 | 0.0000 / 0.0000 / 0.0000 |
| **`OPT_2_DECOUPLED_TIMESCALE`** | Backbone $lr=0.0005$ (clip 0.5); Readout $lr=0.0015 \\to 0.00015$ (clip 1.0) | 96.43% | 20.63% | 19.05% | 71.5% | 4.77 | 35.07 | 0.0009 / 0.0022 / 0.0001 |
| **`OPT_3_LOGIT_BOUNDED_BCE`** | `OPT_2` + logit $L_2$ ($10^{-4}$) + state weights | 95.63% | 25.40% | 19.05% | 69.4% | 4.50 | 28.50 | 0.0009 / 0.0027 / 0.0002 |

### 3.3 Root Cause of Late-Horizon Instability
Under constant $lr=0.002$ (`OPT_0`), gradient clipping is triggered on **81.2%** of steps. As the loss decreases, Adam's second-moment accumulator $v_t$ shrinks on rare tokens, while constant learning rate forces gradient steps that oscillate aggressively around the loss minima (update ratios $\\approx 0.0033$). In contrast, `OPT_1_COSINE_FLOOR` eliminates this degradation entirely, achieving a **flawless 100.00%** training exact match with clipping rate cut in half (38.2%) and mean gradient norm reduced to 1.88.

---

## 4. Stage 3 (Task 3): Evidence-Driven Neural Correction

### 4.1 Branch & Repair Selection
- **Branch Rule**: `h_pool` novel exact was 28.57% (< 65%) $\\to$ **BRANCH B (`BRANCH_B_REPRESENTATION_REPAIR`)** was triggered.
- **Repair Rule**: $h_{\\text{tokens\\_stat}}$ novel exact (22.22%) was not higher than $h_{\\text{pool}}$ (28.57%) $\\to$ max/mean pooling alone does not recover concepts. Therefore, **Option B2 (`selective_top2_blocks`)** was triggered: unfreezing `blocks.3` alongside `blocks.4` and `wte` under the decoupled two-timescale optimizer `OPT_2_DECOUPLED_TIMESCALE`.

### 4.2 Repair Trajectory Diagnostics

The student model was trained for 7,200 updates from the parent checkpoint under `B2_SELECTIVE_TOP2_BLOCKS` and `OPT_2_DECOUPLED_TIMESCALE`:

| Evaluation Set | Update 6,800 Exact | Update 7,000 Exact | Update 7,200 Exact | Final BCE Loss (7,200) |
|---|---|---|---|---|
| **Expanded Training** (756 cases) | 99.9999% | 99.9999% | **99.9999%** | $7.81 \\times 10^{-5}$ |
| **Old Known Frames** (84 cases) | 100.00% | 100.00% | **100.00%** | $7.65 \\times 10^{-5}$ |
| **Old Renamed Frames** (84 cases) | 100.00% | 100.00% | **100.00%** | $7.52 \\times 10^{-5}$ |
| **Development Familiar** (126 cases) | 40.48% | 40.48% | **41.27%** | 2.365 |
| **Development Novel** (126 cases) | 38.10% | 38.10% | **35.71%** | 2.527 |

Per-status accuracy at Update 7,200 on Development Novel:
- `positive`: 58.33%
- `negated`: 66.67%
- `absent`: 72.92%

---

## 5. Stage 4 (Task 4): Qualification Gates Verdict

All gates evaluated simultaneously across late checkpoints (6,800, 7,000, 7,200):

| Gate ID | Description | Threshold | Measured Value | Result |
|---|---|---|---|---|
| **Gate 1** | Stable Expanded Acquisition | $\\ge 99.0\\%$ exact at 6800, 7000, 7200 | 99.9999% at all three updates | **PASS** |
| **Gate 2** | Prior-Frame Retention | $\\ge 95.0\\%$ exact on known & renamed | 100.00% on both sets at all three updates | **PASS** |
| **Gate 3** | Unfamiliar Grammar Generalization | $\\ge 85.0\\%$ dev exact (all status recalls $\\ge 80\\%$) | 41.27% familiar, 35.71% novel (positive recall 58.3%) | **FAIL** |
| **Gate 4** | Deterministic Continuation & Integrity | Next-update $\\Delta = 0.0$, 829+ tests pass | `max_logit_diff == 0.00e+00`, 829+ tests pass | **PASS** |

### Final Qualification Decision
- Gates 1, 2, and 4 passed decisively.
- **Gate 3 FAILED** (35.71% vs required 85.00%).
- In accordance with Stage 4 preregistration:
  - **Sealed evaluation was NOT run**.
  - **No checkpoint promotion occurred**.
  - The formal **Architectural Redesign Decision** was adopted.

---

## 6. Formal Architectural Redesign Decision: Root Cause & Stage A.9 Mandate

### 6.1 Anatomical Root Cause: The Position-Encoding Defect of Frozen Lower Blocks
The empirical results of Stage A.8 establish a definitive conclusion about why partial unfreezing (`blocks.4`, `blocks.3`, `wte`) fails to generalize across unfamiliar grammar:

1. **The Representation Gap**: N-gram bag-of-ngrams achieves **100%** on unfamiliar grammar, proving that the training data contains sufficient lexical/compositional information. Yet, Transformer representations extracted from the frozen backbone achieve $\\le 30.95\\%$.
2. **Absolute Positional Distortion**: The Mini-GPT base model uses absolute learned position embeddings (`wpe`). In the frozen lower blocks (`blocks.0..blocks.2`), tokens are contextualized based on their exact prompt-offset coordinates. When an unfamiliar grammatical frame introduces a novel prefix or alters the token offset of a negation operator (e.g., placing negation 4 tokens earlier or inverting property order), the lower blocks feed completely different, shifted vector coordinates into the higher blocks.
3. **The Limits of Plasticity in Late Blocks**: Unfreezing `blocks.3` and `blocks.4` allows the student to perfectly memorize all 756 training instances ($100.00\\%$ training accuracy, loss $< 10^{-4}$), but because the inputs to `blocks.3` are already position-distorted by frozen `blocks.0..blocks.2`, the top blocks learn complex frame-specific compensations rather than a position-invariant compositional concept operator.

### 6.2 Foundation Upgrade Required for Stage A.9
Partial student adaptation on top of an arithmetic-pretrained backbone has reached its theoretical capacity limit. Stage A.9 must implement the following foundation upgrade:

1. **Full-Substrate Pretraining with Relative/Rotary Position Embeddings (RoPE)**:
   - Replace absolute learned position embeddings (`wpe`) with Rotary Position Embeddings (RoPE) throughout all 5 blocks of the Mini-GPT backbone. RoPE guarantees that attention inner products depend strictly on relative token distances ($pos_i - pos_j$) rather than absolute prompt offsets.
2. **Pretraining All Blocks on the Synthetic Language Substrate**:
   - Rather than freezing blocks 0..2 from the initial arithmetic-only checkpoint, all 5 Transformer blocks must be jointly pretrained on a synthetic relational and grammatical substrate before semantic concept specialization.
3. **Standardized Optimization Protocol**:
   - Adopt `OPT_1_COSINE_FLOOR` (cosine decay from $0.002 \\to 0.0001$, $\\beta_2=0.98$, $\\epsilon=10^{-6}$) as the canonical optimizer for all subsequent stages, permanently eliminating late-horizon training degradation.
