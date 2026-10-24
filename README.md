# Adaptive Mini-GPT

An experimental, **CLI-first Python/PyTorch** research framework studying **adaptive capacity allocation** and **continual modular skill acquisition** in small decoder-only Transformers under constrained single-GPU budgets.

**Current status:**
- **M0 (Baseline Architecture & Determinism):** Passed
- **M1 (Stateful Training, Atomic Checkpointing & Resumption):** Passed
- **M2 (Colab Tesla T4 FP16 Qualification — >500k tok/s):** Passed
- **M3 (Modular Residual Experts & Continual Learning):** Passed (Verified on Tesla T4)
- **M4 (Function-Preserving Growth & Zero Forgetting):** Passed (Verified on Tesla T4)
- **M5 (Mixture-of-Depths & Dynamic Token Compute Allocation):** Passed (Verified on Tesla T4)
- **M18 (Grammar-Constrained Planning & Execution Guidance):** Passed (100% on addition ladders & 5-step kinematics)
- **M19 (Learned Action Selection & Representation Adaptation):** Closed (No-go on neural promotion; M18 baseline preserved)
- **M20 (Controlled Neural Scaling & Procedural Generalization):** Completed (1M vs 3M scaling; No-go on promotion; M18 baseline preserved)
- **M21 (Digit-Level Neural Arithmetic & Procedural Scratchpad):** In Progress (Testing representation ablations and carry scratchpads)
- **M70 (Verified Continual Self-Improvement & Portable Data Pipeline):** Passed (289/289 Tests on Tesla T4; Autonomous Policy beats ALWAYS_REUSE)
- **M71 (Selective Neural Memory, Causal Retention & Compact Skill Management):** Passed (294/294 Tests on Tesla T4; Selective Retain beats Reset by +0.0218 Macro F1 & +0.0130 Net Utility)
- **M72 (Robust Lifelong Neural Memory & Capacity Efficiency):** Passed (299/299 Tests on Tesla T4; Selective Retain beats Retain-All & Wrong-Route)
- **M73 (Cost-Efficient Neural Memory & Verified Cumulative Improvement):** Passed (304/304 Tests on Tesla T4; SELECTIVE_NO_EXTRA_PROBES beats RESET in Macro F1 & Physical Utility)

## Research Hypothesis

Under a matched compute and token budget, **function-preserving modular growth** (attaching zero-initialized bottleneck experts to a trained backbone with load-balanced or task-conditioned routing) improves continual skill retention and sample efficiency compared to static dense retraining, while mitigating catastrophic forgetting.

## Core Engineering Principles

- **Learn first, adapt later:** Verify dense baselines and deterministic resumption before activating routing or capacity growth.
- **Function-preserving transitions:** Any structural growth operation ($M \to M'$) must satisfy $M'(x) = M(x)$ at the step of expansion via zero-initialized residual projections.
- **Single-process CLI parity:** Identical `amg` CLI execution on local Arch Linux CPU, Google Colab, and Kaggle T4 GPUs.
- **Crash-only atomic persistence:** Checkpoints write to temporary files, pass a load verification smoke test, and atomically swap via `os.replace` alongside a versioned `manifest.json`.\
- **Reproducible evaluation:** Deterministic document/task splits, SHA-256 data and vocabulary fingerprints, and isolated holdout verification.

## Quickstart & CLI Usage

```bash
# Install environment and dev dependencies
uv sync --extra dev

# Run test suite and linter
uv run ruff check src tests scripts
uv run pytest -q

# Train dense baseline or adaptive expert model
uv run amg train --config configs/tiny.yaml

# Resume interrupted run (with automatic fallback if latest checkpoint is truncated)
uv run amg train --config configs/tiny.yaml --resume runs/exp-001/latest.pt

# Grow a checkpoint by spawning a new residual expert for continual learning
uv run amg grow --checkpoint runs/exp-001/best.pt --output runs/exp-002/init.pt --add-experts 2 --freeze-backbone

# Evaluate perplexity, bits-per-character, and expert utilization
uv run amg evaluate --checkpoint runs/exp-001/best.pt

# Autoregressive generation
uv run amg generate --checkpoint runs/exp-001/best.pt --prompt "From fairest "
```

## Headless Google Colab GPU Workflow

To run tests, training, and multi-arm continual learning benchmarks directly on an authenticated Google Colab Tesla T4 GPU from your terminal:

```bash
# Check auth & allocate T4 session
uvx --from google-colab-cli colab new -s amg-t4 --gpu T4

# Package, upload, install & run test suite on Colab
tar --exclude='.git' --exclude='runs' --exclude='.venv' --exclude='__pycache__' -czf /tmp/repo.tar.gz .
uvx --from google-colab-cli colab upload -s amg-t4 /tmp/repo.tar.gz /content/repo.tar.gz
uvx --from google-colab-cli colab exec -s amg-t4 -f <(echo "
mkdir -p /content/adaptive-mini-gpt
tar -xzf /content/repo.tar.gz -C /content/adaptive-mini-gpt
pip install -e /content/adaptive-mini-gpt pytest
cd /content/adaptive-mini-gpt
pytest -q
")

# Release GPU session when done
uvx --from google-colab-cli colab stop -s amg-t4
```

See [docs/03-training-and-colab.md](docs/03-training-and-colab.md) for the full headless terminal guide and step-by-step training commands.

## Kaggle Free GPU workflow

Kaggle runs the existing `amg train` CLI with FP16, using a private source dataset
and a small bootstrap script. Start with a short T4 smoke qualification:

```bash
uv tool install kaggle
kaggle auth login
export KAGGLE_USERNAME='your_actual_kaggle_username'
uv run python kaggle/prepare.py --username "$KAGGLE_USERNAME" --mode smoke
kaggle datasets create -p runs/kaggle/code
# Wait for dataset readiness before submitting.
kaggle kernels push -p runs/kaggle/job --accelerator NvidiaTeslaT4 --timeout 600
kaggle kernels status "$KAGGLE_USERNAME/amg-t4-smoke"
kaggle kernels logs "$KAGGLE_USERNAME/amg-t4-smoke" --follow --interval 10
kaggle kernels output "$KAGGLE_USERNAME/amg-t4-smoke" -p runs/kaggle-smoke-output
```

See [the Kaggle guide](docs/06-kaggle.md) for account verification, quota, source
updates, training, and checkpoint resume. Outputs from `/kaggle/working` must be
explicitly attached to a later job to resume. Real Kaggle GPU qualification is
pending. New agents should read [the backend handoff](docs/07-backend-handoff.md).

## Milestone Validation Summary

| Milestone | Key Deliverable | Verified Result on Tesla T4 |
| :--- | :--- | :--- |
| **M0/M1** | Deterministic Pre-norm Transformer & Atomic Checkpointing | Bitwise identical loss across resumes; safe recovery from corrupted checkpoints |
| **M2** | FP16 Mixed Precision & High-Throughput Training | Peak **513,012 tok/s**, sustained ~**420,000 tok/s** |
| **M3** | Modular Experts & Balanced Routing | High router entropy (**0.9742**), balanced expert load distribution |
| **M4** | Function-Preserving Structural Growth & Zero Forgetting | Exact $\Delta = 0.0000$ boundary loss drift; **0.0009** backbone forgetting |
| **M5** | Differentiable Mixture-of-Depths (MoD) | **35% active parameter reduction** per token, **38% arithmetic exact match** |
| **M18** | Grammar-Constrained Planning & Execution Search | **100.0% verified accuracy** on addition depth ladders (2-5 steps) and 5-step kinematics |
| **M19** | Learned Action Policy & Representation Adaptation | Closed without promotion (familiar rollout **100%**, 5-step kinematics **23.3%**; M18 baseline preserved) |
| **M20** | Controlled Neural Scaling & Procedural Generalization | 1M vs 3M Transformer scaling under matched tokens and matched compute |
| **M21** | Digit-Level Arithmetic & Procedural Scratchpad | Procedural scratchpad unlocks neural addition: **35.17%** unaided exact match, **88.07%** carry accuracy (vs 0.0% standard text) |
| **M22** | Compact Scratchpads, RoPE, ALiBi & Length Generalization | Compact scratchpad + RoPE reaches **64.67%** exact match and **98.96%** carry accuracy; language replay achieves **-6.42** drift |
| **M23** | Recurrent Neural Arithmetic & Length Extrapolation | Position-independent neural transition breaks extrapolation barrier: **14.27%–23.60% exact match across unseen 4–8 digits** (vs 0.0% monolithic), **97.87%** learned stopping accuracy |
| **M24** | Reliable Neural Transitions & Deep Extrapolation | Balanced elementary curriculum triples extrapolation to **44.67%** (up to **60.4%** on Seed 44); unrolls to **16 digits (38%)** and **32 digits (14%)** with **99.5%** carry accuracy |
| **M70** | Rational Experiment Controller & Portable Data Pipeline | **0.5311** net utility (Small) beating ALWAYS_REUSE (**+0.2422**); 289/289 tests on Tesla T4 |
| **M71** | Selective Neural Memory & Causal Skill Retention | `SELECTIVE_RETAIN` beats `RESET` (**+0.0218 F1**, **+0.0130 Utility**, **+0.0764 Zero-Shot F1**); 294/294 tests on Tesla T4 |
| **M72** | Robust Lifelong Neural Memory & Capacity Efficiency | Long-horizon 20-ep stream; `SELECTIVE_RETAIN` beats `RETAIN_ALL` (**+0.0042 F1**) & `WRONG_ROUTE` (**+0.0036 F1**); 0.0 negative transfers; 299/299 tests on Tesla T4 |
| **M73** | Cost-Efficient Neural Memory & Verified Cumulative Improvement | Causal lifelong benchmark; `SELECTIVE_NO_EXTRA_PROBES` beats `RESET` (**+0.0056 F1**, **+0.0051 Phys Util**); WRONG_ROUTE reveals 0.67 neg transfers; 304/304 tests on Tesla T4 |

## Directory Structure

```
├── configs/                  # Reproducible training and architecture configs
│   └── tiny.yaml
├── data/raw/                 # Raw datasets (e.g., Shakespeare Sonnets)
├── docs/                     # Research discipline and design documentation
│   ├── 01-vision.md
│   ├── 02-architecture.md
│   ├── 03-training-and-colab.md
│   ├── 04-experiments.md
│   └── 05-roadmap.md
├── scripts/                  # Standalone benchmark experiments & runners
│   └── run_m3_m4_experiment.py
├── src/adaptive_mini_gpt/    # Core package
│   ├── adaptive/             # MoE layers, routers, growth operators, MoD
│   ├── data/                 # Tokenizer, procedural datasets, dataloader
│   ├── evaluation/           # Perplexity, BPC, arithmetic accuracy evaluators
│   ├── model/                # Transformer decoder, attention, configuration
│   └── training/             # Trainer, checkpoint manager, optimizer
└── tests/                    # 299 deterministic unit and integration tests
```
