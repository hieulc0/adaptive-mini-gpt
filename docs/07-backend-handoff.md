# GPU backend handoff for the next agent

## Scope and architecture (2026-10-20)

The task adds Kaggle Free GPU beside the existing Colab workflow. All training
continues through `amg train`; model architecture, optimizer behavior, checkpoint
format, and existing Colab launcher remain intact. Milestone research claims in
the other docs are historical; this infrastructure change makes no new Kaggle
performance or model-quality claim.

Read [03-training-and-colab.md](03-training-and-colab.md) for the environment
comparison and CPU/Colab quick starts, then [06-kaggle.md](06-kaggle.md) for exact
Kaggle commands and troubleshooting. Read all `docs/*.md` and `configs/tiny.yaml`
before source changes, as required by `AGENTS.md`.

## Implementation map

- `kaggle/prepare.py`: standard-library local preparer; packages an explicit source
  allowlist into `runs/kaggle/code/amg-code.zip`, writes private dataset metadata,
  and embeds job options in the standalone script in `runs/kaggle/job/`.
- `kaggle/kernel-metadata.json`: private Python script/T4/internet template;
  placeholder username is replaced by the preparer. `NvidiaTeslaT4` requests T4 ×2;
  AMG uses one GPU, not distributed training.
- `kaggle/launch.py`: checks preinstalled CUDA torch, extracts the attached private
  source dataset, installs the project with runtime pip (without `--upgrade`),
  records runtime details, then calls qualification or `amg train`. Handles both
  traditional and nested input mount layouts. All artifacts live in working storage.
- `configs/gpu_smoke.yaml`: 1 layer, width 32, context 32, batch 2, one accumulation
  step, four CLI training steps total. Uses deterministic generated arithmetic.
- `scripts/qualify_gpu.py`: shared opt-in CUDA FP16 diagnostic, plus CLI train at
  steps 1–2 and resume at 3–4. Full pass writes `qualification.json`; runtime-only
  checks write `runtime.json` without claiming a qualification pass.
- `tests/test_kaggle_backend.py`: packaging, bootstrap dispatch, missing CUDA,
  CPU CLI resume, cross-device-count RNG restoration, and an opt-in real GPU test.
- `training/checkpoint.py`: narrow portability fix slices saved CUDA RNG state
  lists to available devices, logging count differences. Necessary for checkpoints
  moving from Kaggle's two visible GPUs to a single-GPU host. Matching device
  indices retain saved states; extra destination devices retain their seeded states.

## Existing behavior confirmed during inspection

`Trainer` chooses CUDA when `device=auto` and CUDA is available, otherwise CPU.
CPU training uses FP32 without autocast/scaling. CUDA FP16 enables autocast and
GradScaler; CUDA BF16 enables autocast without scaling. This launcher always uses
FP16, explicitly skipping unsupported native BF16 on T4.

`CheckpointManager` atomically publishes versioned step checkpoints, latest/best
aliases, manifests, and optional external backups. It persists architecture,
CharacterTokenizer, weights, optimizer, scheduler, scaler, RNG, global step,
tokens/data position, run identity, and dataset fingerprints. Resume uses saved
architecture/tokenizer, validates dataset lineage, and targets a total global step.
The existing CLI train path reconstructs CharacterTokenizer on resume; do not
claim BPE research checkpoint support from this qualification.

Colab writes scratch under `/content` and mirrors checkpoints/manifests to mounted
Google Drive with `--backup-dir`. Its metrics log needs separate copying. Kaggle
writes under `/kaggle/working/amg`; completed outputs must be downloaded or attached
as a dataset for a new independent job. Abrupt termination can lose unpublished
outputs. No remote cloud backup orchestration was added.

## Verification and remaining work

Local environment: Python 3.12.13, PyTorch 2.14.1+cu130; CUDA unavailable.
Commands use `UV_CACHE_DIR=/tmp/amg-uv-cache` because the normal home cache is
read only in the agent sandbox. Tests use `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`
to avoid CPU thread oversubscription.

- Initial full local suite: **140 passed, 1 GPU test skipped** (119.22 seconds).
- After adding RNG and bootstrap regressions: backend suite **8 passed, 1 skipped**;
  checkpoint/resume subset at the preceding stage **13 passed, 1 skipped**.
- Changed Python files pass Ruff; `git diff --check` passes.
- Repository-wide Ruff reports **607 violations in other existing files**. No
  unrelated lint cleanup was performed.
- Direct runtime-only qualification fails with exit code 1 and a clear CUDA-required
  error on this CPU host, as intended.
- Final full suite after all changes: **145 passed, 1 GPU test skipped** in
  **115.78 seconds** (`uv run --no-sync pytest -q` with the environment above).

**No job was submitted to Kaggle and no Kaggle GPU tests passed in this session.**
Current official Kaggle docs were consulted for OAuth, dataset upload/versioning,
T4 metadata/accelerator, timeout, quota, kernel logs/status, and output commands;
links are in the Kaggle guide. No credentials were requested, read, or committed.

Next GPU verification: authenticate as the user, check quota, prepare the smoke
payload, upload the private code dataset, wait for readiness, then submit the
600-second T4 smoke job using the guide. Confirm actual GPU name, FP16 behavior,
CUDA RNG restore, optimizer/scaler checkpoint resume, and output download layout.
Inspect `amg/qualification.json` and job completion status. Then optionally run a
short resumed job from an attached checkpoint dataset; long training is unnecessary
for compatibility qualification. Dataset processing, Kaggle dependency installation,
real T4 kernels, and runtime termination/output publication remain cloud checks.

The worktree already contained unrelated untracked M19 research scripts/modules/
tests when this task began. They were left intact and included in the local suite;
they are not part of this backend change. The source payload intentionally snapshots
all Python files under `src/`, including uncommitted local source changes.
