# AGENTS.md — Adaptive Mini-GPT

> **Execution policy (highest priority): LOCAL = edit, lint, type-check, syntax-check, and strictly mocked/lightweight unit tests only. ALL real neural training, GPU work, full regression suites, scientific benchmarks, and performance qualification MUST run remotely on Kaggle or Google Colab. Never silently fall back to local CPU training.**

## 1. Mandatory preflight — before changing code

1. Read **all** `docs/*.md` and `configs/tiny.yaml`. Read the current `README.md`, relevant source and tests, dataset specifications/manifests, and the latest completed milestone report. If a required file is missing, report it; do not fabricate its contents.
2. Inspect the current Git branch/HEAD, working tree, checkpoint lineage, and relevant remote-run scripts. Do not overwrite the user's uncommitted work.
3. Identify the **current milestone** and its exact scientific acceptance gates from repository documentation. Historical M0–M5 are completed T4 milestones; later research has progressed through M71 in recent reports, with M72 proposed. **Do not use the old M0–M5 status as the active scope.** Confirm the actual branch, report and status from the repository rather than assuming M72 is already implemented.
4. Write a short implementation plan: smallest change, tests, remote GPU experiment, baselines, metrics, and rollback. Preserve prior milestone contracts and mathematical invariants.
5. Before invoking a command, classify it as **LOCAL-SAFE** or **REMOTE-REQUIRED** using Section 2. When uncertain, classify as **REMOTE-REQUIRED**.

## 2. Non-negotiable execution boundary

### LOCAL-SAFE — permitted on the developer's machine

- Read/edit source code, configs, docs and tiny deterministic fixtures.
- Git inspection, formatting, static analysis, syntax/compile checks, lint and type checks.
- **Only explicitly identified fast, CPU-only tests that use mocks/fakes/small fixtures and perform no actual neural training, checkpoint benchmark, model-size evaluation, or GPU operations.** Inspect test contents before invoking them.
- Dataset manifest/schema/hash validation on tiny fixtures; package metadata checks.
- Prepare and submit remote jobs; query their status; retrieve and analyze completed artifacts.

Examples (only if these paths/tools exist in the repository):

```bash
uv run python -m compileall -q src tests
uv run ruff check .
uv run ruff format --check .
uv run pyright
# Run ONLY a manually inspected mock-only test file:
uv run pytest -q tests/<mock_only_test_file>.py
```

### REMOTE-REQUIRED — NEVER execute locally

- **Any model training**: pretraining, finetuning, meta-learning, adapter training, optimizer steps/backpropagation, even a short “smoke” training run.
- Any real neural evaluation: model forward-pass benchmarks, quality metrics involving loaded model weights, holdout comparisons, retention checks, causal-ablation experiments or repeated inference over datasets.
- Full `pytest`/integration/regression suites, tests with real model training or checkpoint loading, and tests of unknown resource requirements.
- Profiling, memory/FLOP/latency benchmarks, multi-seed sweeps, GPU tests, long-running data generation, and scientific qualification.
- Any task that might consume substantial local CPU/RAM/disk/time. **Do not use a local CPU fallback if a remote GPU is unavailable.**

**Remote choice:** Prefer the project's existing Kaggle Tesla T4 workflow for reproducible benchmarks; use an available Google Colab GPU as an alternative. If both are unavailable or quota-limited, do only LOCAL-SAFE work and clearly mark GPU qualification **NOT RUN / BLOCKED**. Never claim a remote pass based on local mocks or estimated metrics.

### Enforce the boundary in every agent run

- State in the work summary: `LOCAL: lint/type/syntax/mock-only`; `REMOTE: Kaggle or Colab GPU training/full tests`.
- Put a remote-runtime preflight at the beginning of every training/benchmark entry point: verify execution is on the intended remote platform **and** that a CUDA GPU is actually available; abort with a clear error otherwise. Do not silently switch to CPU.
- Do not launch `pytest -q` over the entire repository locally. Run the full suite **inside the remote GPU job**.
- Timeouts or remote errors are not passes. Preserve failure traces, fix the smallest cause, then rerun remotely.

## 3. Remote execution discipline

### Kaggle (preferred for milestone benchmarks)

- Follow the existing repository's Kaggle CLI/kernel scripts and documented dataset attachment conventions. Pin the precise Git commit and environment; do not invent a second runner.
- Submit the source and necessary small configs; attach immutable dataset versions or generate inexpensive synthetic worlds from pinned specifications inside Kaggle.
- Run the **full regression suite and actual GPU benchmark on Kaggle**, including independent seeds/worlds and the required matched controls.
- Download and inspect machine-readable results, logs, test counts, device details and checkpoint hashes before reporting PASS/FAIL.
- A submitted or RUNNING kernel is **not** a completed experiment.

### Google Colab (alternative remote GPU)

- Use the existing **headless, non-interactive** CLI flow; do not require a manual notebook UI during agent execution.
- Use the supported project command pattern:

```bash
uvx --from google-colab-cli colab exec -s <session> -f <remote_script>
# ALWAYS release the remote session after execution, including failures:
uvx --from google-colab-cli colab stop -s <session>
```

- Use a shell `trap`/`finally` cleanup wrapper so `colab stop` is attempted even if execution fails. Never leave a GPU session running needlessly.
- Confirm CUDA device availability from within Colab before model training. Record the GPU type; do not assume every runtime is a T4.
- A Colab-compatible **local** loader mock is not evidence of a real Colab execution. Label it accurately.

### Required remote result provenance

Every qualified run must include: platform/kernel or session ID; completed status; Git full SHA; data manifest ID/hash and split; checkpoint hashes; seed/world IDs; Python/PyTorch/CUDA/GPU versions; training settings; actual optimizer steps, query counts and GPU time; full test counts; metrics and per-arm artifacts. Store reproducible output under the repository's existing `runs/` convention (large artifacts Git-ignored).

## 4. Scientific objective and milestone gates

Our goal is a **compact, self-improving learner**, not maximal parameter count. It should learn unfamiliar skills, retain and reuse useful weights, improve future learning decisions from verified experience, and eventually adjust connections, dimensions or topology **only when independent experiments justify the cost**.

- Begin with the smallest change to data, supervision, optimization, weight selection or routing. Keep the compact rank-16 residual adapter as the default unless a controlled experiment establishes a capacity bottleneck.
- Compare against the **strongest feasible baseline**, not a conveniently weaker one: `ALWAYS_REUSE`, budget-matched `ALWAYS_FRESH`, `RESET_WEIGHTS`, unchanged repertoire, retain-all, wrong/irrelevant route, and shuffled weights as appropriate.
- Keep training, candidate qualification, controller development and final sealed testing isolated by independent worlds and vocabularies. **Never expose sealed labels, hidden family IDs, or complete oracle truth to the learner or its controller.**
- Budget-match observations, optimizer steps, validation effort and actual GPU costs. Report quality, cost, stored capacity and active capacity separately; do not hide regressions in a single utility score.
- Separate neural **weights actually retained** from controller history. A working experiment loop is not proof of self-improvement; stored weights are not proof of better future performance.
- For continual-learning claims, show multiple actual future encounters and paired per-world/seed comparisons to reset and strongest static controls. Count **realized**, not projected, skill reuse.
- Report honest PASS/FAIL/INCONCLUSIVE against each binding gate. Tests passing or a GPU run completing does **not** imply scientific success.
- Base weights and qualified skill snapshots must remain immutable; changes must be isolated, reversible and validated for carry, chained-logic and prior-skill retention.
- No autonomous production promotion, unbounded module spawning, uncontrolled external actions, arbitrary topology changes, or destructive pruning.

## 5. Mathematical and architectural invariants

Preserve the applicable implementation contracts in the current docs and tests, including these historical foundations:

1. **Fused causal attention:** Standard causal Transformer passes use `F.scaled_dot_product_attention(..., is_causal=True)`; retain explicit attention-mask support for inspection/custom tests. Never break causal independence.
2. **Decoupled weight decay:** Exclude all one-dimensional parameters, LayerNorm/bias parameters **and embeddings** from weight decay, even though embeddings may be 2D. Use explicit named/type exclusions; do not assume the `dim >= 2` rule alone correctly excludes embeddings.
3. **Function-preserving growth:** New experts/depth branches start with zero residual output projections. Verify old and grown checkpoint logits are bit-exactly identical at the growth boundary (`t0`) under a deterministic, equivalent execution path before any further training.
4. **Selective token adaptation:** Mask `wte` gradients to `trainable_token_ids`; freeze `lm_head` and protect old output probabilities. Account for tied embeddings, optimizer state and any other route that could alter output logits. Verify the documented zero-drift condition with `bypass_moe=True`; do not assert zero forgetting based on a frozen tensor alone.
5. **Causal Mixture-of-Depths:** Gather active tokens **physically** before FFN/MoE computation for actual compute savings. Scale outputs by continuous gate `sigmoid(z[b,t])` for gradients, and use token-local threshold `z[b,t] >= 0` in `eval()` to preserve causality.
6. **Retention/rollback:** Check exact checkpoint and adapter hashes; run real remote behavioral retention tests; enforce bit-exact rollback on rejected candidates.

Legacy files may include `model/transformer.py`, `model/attention.py`, `adaptive/experts.py`, `adaptive/router.py`, `adaptive/growth.py`, `adaptive/mod.py` and `evaluation/`. Later milestones use `src/adaptive_mini_gpt/` modules. **Inspect current paths and reuse existing code; do not recreate obsolete layouts.**

## 6. Portable datasets — local, Kaggle and Colab

Use the established lightweight layout (where present):

```text
data/
  specs/        # tracked generator configurations
  fixtures/     # tracked, tiny mock/regression fixtures
  manifests/    # tracked versions, schemas, split IDs and hashes
  generated/    # ignored, reproducible generated corpora
  external/     # ignored downloaded datasets
  packages/     # ignored private remote-upload bundles
runs/           # ignored large experiment outputs/checkpoints
```

- Use `AMG_DATA_ROOT` / the existing portable loader. Do not hardcode Kaggle/Colab paths inside the training code.
- For small synthetic experiments, generate data remotely from pinned code/specs and independent world seeds. For expensive or shared datasets, use private, immutable, pinned Kaggle Dataset versions, which Colab can retrieve through the documented `kagglehub` flow.
- Verify manifests, content hashes, schema, class balance, dataset provenance, train/dev/sealed split isolation and allowed observable evidence **before** scientific qualification.
- Use package **allowlists** and content checks for secrets. Do not publish credentials, complete repositories, local `.env` files, checkpoints or sealed evaluation answers in dataset packages.
- No DVC, databases, data lake, distributed platform or other infrastructure unless a demonstrated problem justifies it.

## 7. Code quality, reporting and documentation

- Implement the smallest milestone-specific diff. Keep `uv`, `pytest`, `ruff`, `pyright` and existing project interfaces; avoid new dependencies without need.
- **Local:** syntax/lint/type-check and explicitly mocked tests only. **Remote:** full regression suite, actual neural training, retention, GPU metrics, and multi-seed qualification.
- On every milestone update, review and update relevant `README.md`, `docs/04-experiments.md`, `docs/05-roadmap.md`, `docs/06-data-pipeline.md` and other affected docs; keep commands accurate and tested on the specified platform.
- Record successful and negative experimental results. Do not hide baseline wins, regressions, failed seeds, data leakage, missing artifacts or test failures.
- Clearly separate `IMPLEMENTED`, `LOCALLY CHECKED (MOCK ONLY)`, `REMOTE SUBMITTED`, `REMOTE VERIFIED`, and `SCIENTIFICALLY QUALIFIED`. Never claim completion based on a queued kernel.
- Final agent report must include changed files, commands run **and where**, remote kernel/session and artifact paths, actual GPU results, test counts, scientific gate verdicts, and remaining blockers.

**If a request conflicts with these execution restrictions, stop the prohibited local action, move the real test/training to Kaggle or Colab, or report the remote run as blocked. Do not use the local machine as a fallback training server.**
