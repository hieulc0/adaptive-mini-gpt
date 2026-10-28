# Stage A.8 — Neural Learning Mechanism Investigation & Foundation Repair for Learning-to-Learn

Status at preregistration: **PLANNED; remote Kaggle CUDA execution pending**. Continues from exact tested source `be8e289` on branch `m104-e-and-m105-a`. Historical A.7 Engineering PASS / Learning FAIL at `be8e289` and report `b78b790` remain strictly immutable. No milestone or candidate promotion.

## 0. Non-Negotiable Execution Boundary (`AGENTS.md`)
- **LOCAL-SAFE ONLY on host machine**: Code reading/editing, `uv run python -m compileall -q src scripts tests kaggle`, `uv run ruff check`, `uv run pyright`, and explicitly inspected mock/fixture unit tests (`tests/test_stage_a8_contracts.py`) with ZERO neural forward/backward passes, ZERO checkpoint loading, and ZERO CPU training.
- **REMOTE-REQUIRED on Kaggle Tesla T4 GPU**: All frozen-feature probe fitting, trajectory diagnostics, neural training updates, and full regression suites (`AMG_RUN_GPU_TESTS=1 pytest -q`) MUST run remotely on Kaggle (`sparklinghydraulics0`). Never fall back to local CPU if remote GPU fails.

## 1. Stage-by-Stage Implementation Specification

### Task 1: Repair the Frozen-Feature Representation Diagnosis (`scripts/probe_representations_a8.py` & `data/specs/probe_a8_v1.json`)
- **Problem**: Stage A.7's `probe_representations_a7.py` trained ridge probes on a restricted 162-case subset that lacked coverage of alternative negators (`without`) and property orders, causing even the n-gram control to fail on unfamiliar grammar.
- **Coverage-Matched Diagnostic Train Split**: Constructs a balanced 378-case split crossing:
  - Both negation operators (`not`, `without` and `none`)
  - Both property presentation orders (`flam_first`, `cond_first`, single property, none)
  - All permitted training grammatical frames (0..5) and prefix variations ('', 'Log. ', 'An entry. ')
  - Balanced `positive`, `negated`, and `absent` states for both target attributes
- **Target Frozen Models**: `parent`, `coverage_1200`, `coverage_5800`, `coverage_7200`.
- **Target Representations**:
  - `h_pool` ($\in \mathbb{R}^{128}$): question-pooled Transformer hidden state
  - `z_concept` ($\in \mathbb{R}^{32}$): projected bottleneck representation
  - `h_tokens_stat` ($\in \mathbb{R}^{256}$): concatenation of `[mean_pool(h_question), max_pool(h_question)]`
- **Experimental Arms**:
  - **Arm P1 (Deployed Head)**: Direct `attr_head` logits from checkpoint.
  - **Arm P2 (Linear Ridge Probe)**: Cross-validated $L_2$ ridge regression ($\lambda \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10\}$) over `h_pool`, `z_concept`, and `h_tokens_stat`.
  - **Arm P3 (Bounded Nonlinear Probe)**: 2-layer MLP (`d_in -> 64 -> GELU -> 2`, weight decay `1e-2`, 300 steps AdamW) over `h_pool` and `h_tokens_stat`.
  - **Arm P4 (Lexical Controls)**: Word 1–3-gram linear and Bag-of-Words linear trained on the exact same 378 coverage-matched probe training set.

### Task 2: Diagnose & Fix Late-Horizon Optimization Instability (`scripts/diagnose_optimization_a8.py` & `data/specs/optimization_a8_v1.json`)
- **Problem**: Expanded student reaches 98.81% training accuracy at update 5,800, then collapses to 78.70% at update 7,200 under fixed `lr=0.002`.
- **Trajectory Forensics**: Instrument `LearningState.step` via an opt-in callback (every 100 steps, and every 20 steps in window 5400..7200):
  - Per-module pre-clip gradient norm: `||g_wte||_2`, `||g_block4_attn||_2`, `||g_block4_mlp||_2`, `||g_proj||_2`, `||g_head||_2`.
  - Per-module effective Adam update ratio: `||theta_t - theta_{t-1}||_2 / (||theta_{t-1}||_2 + 1e-12)`.
  - Adam second-moment (`exp_avg_sq`) min/median/max and parameter step ages.
  - Gradient clipping activation rate (`total_grad_norm > 1.0`).
- **Preregistered 4-Arm Optimization Contrast**:
  - **`OPT_0_BASELINE`**: Unchanged Stage A.7 (`lr=0.002` constant, `betas=(0.9, 0.999)`, `eps=1e-8`, global clip `1.0`).
  - **`OPT_1_COSINE_FLOOR`**: Cosine LR schedule from `0.002` to `0.0001` with Adam $\epsilon = 10^{-6}$ and $\beta_2 = 0.98$.
  - **`OPT_2_DECOUPLED_TIMESCALE`**: Two-timescale parameter groups with independent per-group gradient clipping (`max_norm=0.5` on backbone `wte + blocks.4 + ln_f` at `lr=0.0005`; `max_norm=1.0` on readout `concept_proj + attr_head` at `lr=0.0015` with cosine decay to `0.00015`).
  - **`OPT_3_LOGIT_BOUNDED_BCE`**: `OPT_2` + gentle logit $L_2$ penalty ($\gamma = 10^{-4} \cdot \|\text{logits}\|_2^2$) and balanced state weighting (`positive: 1.0, negated: 1.0, absent: 1.0`).

### Task 3: Single Evidence-Driven Neural Correction (`scripts/run_stage_a8.py`)
- **BRANCH A**: If Stage 1 shows `h_pool` or `z_concept` achieves $\ge 85\%$ on Unfamiliar Grammar -> pure readout/optimization interference; train unchanged student under winning Stage 2 optimizer.
- **BRANCH B**: If Stage 1 shows `h_pool` fails Unfamiliar Grammar (< 65%), OR `h_tokens_stat` (mean+max) rescues it:
  - **Repair Option B1 (`span_relative_max_mean_pooling`)**: 0 new parameters, variance-preserving residual combination `0.5 * (mean_pool + max_pool)`.
  - **Repair Option B2 (`selective_top2_blocks`)**: Unfreeze `blocks.3` alongside `blocks.4` and `wte` under the stabilized `OPT_2` two-timescale optimizer.

### Task 4: Qualification Gates & Learning-to-Learn Readiness Decision
- **Gate 1 (Stable Expanded Acquisition)**: Expanded training exact match $\ge 99.0\%$ at updates 6800, 7000, 7200.
- **Gate 2 (Prior-Frame Retention)**: Both old-known and old-renamed frame sets $\ge 95.0\%$ at updates 6800, 7000, 7200.
- **Gate 3 (Unfamiliar Grammar Generalization)**: Unfamiliar grammar dev exact match $\ge 85.0\%$ (with `positive`, `negated`, `absent` recall each $\ge 80.0\%$).
- **Gate 4 (Deterministic Continuation)**: Fresh-process reload next-update equivalence (`max_logit_diff == 0.00e+00`), 0 parameter accounting discrepancy, and all 829+ remote tests pass.
- **Conditional Sealed Qualification**: Run only if Gates 1–4 pass. If any fail, stop immediately and record the formal Architectural Redesign Decision.

## 2. Pre-Submission Local Verifications
- `uv run python -m compileall -q src scripts tests kaggle`: PASS (0 errors)
- `uv run ruff check`: PASS (0 errors)
- `uv run ruff format --check`: PASS (0 errors)
- Scoped Pyright (`pyright --project /tmp/amg_a8_pyright.json`): PASS (0 errors, 0 warnings)
- Inspected mock unit tests (`pytest -q tests/test_stage_a8_contracts.py`): PASS (4/4 passed in 3.49s, ZERO neural passes, ZERO checkpoint loading)
- Cumulative mock contracts (`pytest -q tests/test_stage_a8_contracts.py tests/test_exposure_a7_mock.py ...`): PASS (22/22 passed in 3.68s)
