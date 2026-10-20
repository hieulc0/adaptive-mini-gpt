# Training, Checkpointing, Colab, and Kaggle Operations

## Execution environments and quick starts

All environments use `amg train`. Kaggle setup, exact submission commands,
monitoring, logs, output retrieval, and resume are in [06-kaggle.md](06-kaggle.md).
For agent catch-up and verification status, read [07-backend-handoff.md](07-backend-handoff.md).

| Environment | Main use | Precision/device | Persistence |
| --- | --- | --- | --- |
| Local CPU | Development, unit tests, debugging, tiny smoke runs | `--device cpu --precision fp32` | Local `runs/` persists |
| Google Colab | Interactive notebook training and GPU qualification | CUDA FP16 on T4 | `/content` is temporary; mirror checkpoints to mounted Drive |
| Kaggle Free GPU | Submitted batch training and opt-in GPU tests | CUDA FP16 on T4; one device used | Saved `/kaggle/working` job outputs; attach a prior output as input for resume |

### Local CPU quick start

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests scripts kaggle
uv run amg train --config configs/gpu_smoke.yaml --device cpu --precision fp32 \
  --data synthetic:arithmetic --save-dir runs/cpu-smoke --max-steps 2
uv run amg train --config configs/gpu_smoke.yaml --device cpu --precision fp32 \
  --data synthetic:arithmetic --save-dir runs/cpu-smoke \
  --resume runs/cpu-smoke/latest.pt --max-steps 4
```

Use a small batch/context and FP32 for debugging. GPU qualification is opt-in via
`AMG_RUN_GPU_TESTS=1`; normal local pytest skips it. `--device auto` selects CUDA
when available and otherwise CPU. Explicit `cuda` requires working CUDA. CPU
execution disables autocast/scaling even if a mixed precision setting is supplied.
FP32 uses full precision; CUDA FP16 uses autocast plus GradScaler; CUDA BF16 uses
autocast without scaling and requires appropriate hardware. T4 runs use FP16 and
skip BF16 because it is not natively supported. The Kaggle wrapper requires CUDA
and never chooses CPU automatically.

### Interactive Colab quick start and resume

Open [notebooks/colab_launcher.ipynb](../notebooks/colab_launcher.ipynb), select
**Runtime → Change runtime type → GPU**, and check `nvidia-smi`. Run its repository
installation and Google Drive mount cells, then launch `amg train` with local
scratch storage plus `--backup-dir /content/drive/MyDrive/adaptive-mini-gpt-runs/exp-001`.
On a new runtime, reinstall the same project revision and mount Drive again:

```bash
amg train --config configs/tiny.yaml --device cuda --precision fp16 \
  --save-dir /content/runs/colab \
  --backup-dir /content/drive/MyDrive/adaptive-mini-gpt-runs/exp-001 \
  --resume /content/drive/MyDrive/adaptive-mini-gpt-runs/exp-001/latest.pt \
  --max-steps 2000
```

Use the same original corpus and tokenizer. The target step is total, including
restored steps. Only saved progress survives termination; unsaved updates are lost.
The checkpoint manager mirrors checkpoints/manifests, not the entire metrics log;
copy `metrics.jsonl` separately if required. Free Colab GPU type, availability, and
session duration are not guaranteed. Save often, and release headless sessions
immediately after work using the existing procedure below.

### Kaggle quick start

```bash
uv tool install kaggle
kaggle auth login
export KAGGLE_USERNAME='your_actual_kaggle_username'
uv run python kaggle/prepare.py --username "$KAGGLE_USERNAME" --mode smoke
kaggle datasets create -p runs/kaggle/code
# Wait for dataset readiness, then submit:
kaggle kernels push -p runs/kaggle/job --accelerator NvidiaTeslaT4 --timeout 600
```

See the [full Kaggle guide](06-kaggle.md) before running subsequent jobs or resuming.
Checkpoints use the same format on every provider. CUDA RNG restoration supports
moving between one and two GPUs by restoring shared device indices; identical
numerical trajectories across devices/precision/library versions are not guaranteed.

## 1. Development and Training Discipline

### Normal Development
Develop and test locally on CPU. Run unit tests (`uv run pytest -q`) and linter (`uv run ruff check src tests scripts`) locally before launching any cloud jobs. Use the exact same `amg train` CLI and scripts locally and in an interactive Colab GPU runtime.

### Data Flow & Provenance
Choose only datasets with verified rights/provenance (e.g. Shakespeare raw text in `data/raw/`, procedurally generated synthetic arithmetic with deterministic seeds). Split by document/source into train/validation/final holdout, and record SHA-256 fingerprints and tokenizer vocabularies. Never leak validation tokens into training.

### Training Step
Sample causal sequence $x$, shift target $y$; forward $\to$ cross-entropy + auxiliary routing losses $\to$ backward $\to$ gradient clip $\to$ optimizer step $\to$ learning rate schedule update. Validation is held strictly disjoint and non-backpropagated.

### Checkpoint Contract
Checkpoints must atomically persist:
- Model architecture configuration and state dictionary
- Tokenizer vocabulary and character mapping
- Optimizer, LR scheduler, and FP16 `GradScaler` states
- Step count, tokens processed, dataset/sampler positions, and RNG states (Python, NumPy, PyTorch CPU & CUDA)
- Dataset fingerprint, code revision, best validation score, and run metadata

Write checkpoints to local temporary paths, verify serialization integrity, and atomically swap via `os.replace` on the local filesystem alongside a versioned `manifest.json`.

---

## 2. Headless Google Colab Workflow (`google-colab-cli`)

For single-GPU training and qualification on NVIDIA Tesla T4 GPUs, this project uses the headless CLI tool `google-colab-cli` via `uvx`.

### Key Mental Model for Agents
* **A session is a live Jupyter kernel running on a rented Google Colab VM.** `colab new` provisions the VM; `colab stop` releases it.
* **Kernel state and files in `/content` persist across commands within the same session.**
* **Persistent Authentication:** Credentials are cached on the host machine at `~/.config/colab-cli/token.json` and session state at `~/.config/colab-cli/sessions.json`.
* **Non-Interactive Execution:** Agents must never run interactive commands like `colab repl`, `colab console`, `colab auth`, or `colab drivemount` without piping stdin because they require a TTY. Use `colab exec -s <name> -f <(echo "...")` for shell/python execution.
* **Resource Safety:** **Always stop sessions with `colab stop -s <name>`** when benchmarks complete to avoid exhausting compute units while idle.

---

## 3. Quick Reference: Colab Status & Session Management

All commands can be invoked directly from the project directory via `uvx --from google-colab-cli colab <command>`:

### 1. Check Authentication & Compute Balance
```bash
# Check if CLI is authenticated and inspect active compute unit balance & usage rate
uvx --from google-colab-cli colab usage

# List currently active Colab sessions
uvx --from google-colab-cli colab sessions
```
*Note: If authentication is required (e.g. first run on a new machine), running any command prompts a Google OAuth authorization URL. Once the user approves and pastes the one-time code, it is permanently saved in `~/.config/colab-cli/token.json`.*

### 2. Allocate a T4 GPU Session
```bash
# Provision a standard T4 GPU session named 'amg-t4'
uvx --from google-colab-cli colab new -s amg-t4 --gpu T4

# Verify hardware and GPU allocation
uvx --from google-colab-cli colab status -s amg-t4
```

### 3. Terminate Session (Crucial)
```bash
# Release the VM and stop burning compute units
uvx --from google-colab-cli colab stop -s amg-t4
```

---

## 4. Complete Step-by-Step Guide: Training & Benchmarking on Colab

When starting a new agent session, follow this deterministic sequence to test, train, or benchmark on an authenticated Colab GPU runtime without manual web UI interaction:

### Step 1: Verify Auth & Allocate Accelerator
```bash
# 1. Verify credentials and check active sessions
uvx --from google-colab-cli colab sessions

# 2. Provision new session 'amg-t4' with Tesla T4 accelerator
uvx --from google-colab-cli colab new -s amg-t4 --gpu T4
```

### Step 2: Package & Upload Repository
Create a clean archive of the local repository (excluding build artifacts and virtualenvs) and upload it:
```bash
# Package local code
tar --exclude='.git' --exclude='runs' --exclude='.venv' --exclude='__pycache__' \
    -czf /tmp/repo.tar.gz -C /home/hieulc/projects/adaptive-mini-gpt .

# Upload to Colab VM root
uvx --from google-colab-cli colab upload -s amg-t4 /tmp/repo.tar.gz /content/repo.tar.gz
```

### Step 3: Extract and Install in Colab Environment
```bash
uvx --from google-colab-cli colab exec -s amg-t4 -f <(echo "
mkdir -p /content/adaptive-mini-gpt
tar -xzf /content/repo.tar.gz -C /content/adaptive-mini-gpt
pip install -e /content/adaptive-mini-gpt pytest
")
```

### Step 4: Run Remote Test Suite
Verify that CUDA is recognized and all tests pass on the remote Tesla T4 hardware:
```bash
uvx --from google-colab-cli colab exec -s amg-t4 -f <(echo "
cd /content/adaptive-mini-gpt
pytest -q
")
```
*(The current test suite must pass before proceeding to training).*

### Step 5: Execute Training / Multi-Arm Benchmark

**Option A: Run Full M3/M4/M5 Continual Learning & Mixture-of-Depths Benchmark:**
```bash
uvx --from google-colab-cli colab exec -s amg-t4 -f <(echo "
cd /content/adaptive-mini-gpt
python scripts/run_m3_m4_experiment.py \
    --steps-stage1 500 \
    --steps-stage2 500 \
    --device cuda \
    --precision fp16 \
    --output-dir /content/runs/m3_m4_m5_ablation
")
```

**Option B: Train Standard Model via CLI:**
```bash
uvx --from google-colab-cli colab exec -s amg-t4 -f <(echo "
cd /content/adaptive-mini-gpt
python -m adaptive_mini_gpt.cli train \
    --config configs/tiny.yaml \
    --device cuda \
    --precision fp16 \
    --save-dir /content/runs/exp-001
")
```

### Step 6: Download Artifacts & Telemetry Reports
Download the generated benchmark report or model checkpoints back to the local repository:
```bash
mkdir -p /home/hieulc/projects/adaptive-mini-gpt/runs/m3_m4_m5_ablation

# Stream or copy results locally
uvx --from google-colab-cli colab exec -s amg-t4 -f <(echo "
cat /content/runs/m3_m4_m5_ablation/ablation_report.json
") > /home/hieulc/projects/adaptive-mini-gpt/runs/m3_m4_m5_ablation/ablation_report.json
```

### Step 7: Teardown the Session
**Never leave sessions active after runs finish.**
```bash
uvx --from google-colab-cli colab stop -s amg-t4
```

---

## 5. Troubleshooting & Agent Guardrails

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **Command hangs indefinitely** | Running interactive commands (`repl`, `console`, `auth`, `drivemount`) | Use `colab exec -s <name> -f <script_or_fifo>` with non-interactive scripts. |
| **`400 Bad Request` on `colab new`** | Accelerator type unavailable or quota exceeded | Re-try with `--gpu T4` or fall back to CPU (`colab new -s <name>`). |
| **`401 / 403 Unauthorized`** | Missing OAuth token or expired token | Run `colab usage`; if prompt appears, provide the OAuth URL to the user to paste the code. |
| **`FileNotFoundError: configs/tiny.yaml`** | Process executed from `/content` rather than repo directory | Always `cd /content/adaptive-mini-gpt` before invoking `pytest` or python scripts. |
| **High idle compute charge** | VM left running after script completion | Always run `colab stop -s <name>` at the end of the workflow. |
