# Kaggle GPU execution backend

Kaggle and Colab call the same `amg train` CLI, configuration schema, trainer,
and checkpoint manager. There is no Kaggle training engine. Start with the smoke
job; it runs four tiny CLI steps across two processes, plus one FP16 diagnostic
update. Real Kaggle execution has **not yet been qualified** in this change.

## Account and Arch Linux setup

Create an account at [Kaggle](https://www.kaggle.com/), choose your username, and
complete phone verification in account settings to enable free GPU access. Check
that the notebook editor offers a GPU accelerator. Availability and weekly quota
are account dependent; do not assume a guaranteed allocation.

From the repository root in your Arch Linux terminal:

```bash
uv tool install kaggle
# For an existing install: uv tool upgrade kaggle
kaggle auth login
# For a terminal without browser launch:
# kaggle auth login --no-launch-browser
export KAGGLE_USERNAME='your_actual_kaggle_username'
kaggle quota
kaggle kernels push --help
kaggle kernels logs --help
```

OAuth is the current official interactive CLI authentication workflow; credentials
stay in the CLI's user configuration, outside this repository. The username here
names resources and does not replace OAuth. See the official
[authentication reference](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md)
and [quota reference](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/quota.md).
If your installed CLI lacks these commands, upgrade it before continuing.

## Prepare code and submit a T4 smoke job

`kernels push` uploads the designated code file, not this Git checkout. The local
preparer creates an allowlisted zip of `src/*.py`, configs, the qualification
script, `pyproject.toml`, and README. It excludes credentials, weights, virtualenvs,
and training data. Inspect its contents before uploading private project code.
All generated files are under ignored `runs/`.

```bash
uv run python kaggle/prepare.py --username "$KAGGLE_USERNAME" --mode smoke
python -m zipfile -l runs/kaggle/code/amg-code.zip
cat runs/kaggle/job/kernel-metadata.json
# First upload: private by default; do not add --public.
kaggle datasets create -p runs/kaggle/code
kaggle datasets status "$KAGGLE_USERNAME/amg-code"
# Wait until dataset processing is ready, then:
kaggle kernels push -p runs/kaggle/job --accelerator NvidiaTeslaT4 --timeout 600
```

For subsequent code changes, regenerate the payload and update the code dataset:

```bash
kaggle datasets version -p runs/kaggle/code -m 'Update AMG source'
kaggle datasets status "$KAGGLE_USERNAME/amg-code"
```

Wait for readiness before pushing a new job. Archive SHA256 is printed locally;
retain it with experiment records. The generated job copies the extracted source
into `/kaggle/working/amg-code`, preserving it in job outputs for inspection.

The checked-in `kaggle/kernel-metadata.json` is a template with placeholder owner
values. The preparer writes your username, source attachment, and a private
Python script job into `runs/kaggle/job/kernel-metadata.json`. Internet access is
enabled for installing missing Python dependencies. The bootstrap uses the
runtime Python and `pip install <source>` without `--upgrade`, allowing installed
CUDA PyTorch to satisfy `torch>=2.4`. It checks CUDA before installation and records
GPU identity, PyTorch/CUDA versions, free/total VRAM, and compute capability.
Missing CUDA fails with a nonzero exit; it never falls back to CPU.

Official [metadata documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)
identifies `NvidiaTeslaT4` as **GPU T4 ×2**. AMG is single process and uses `cuda:0`;
it does not combine memory or train across the second GPU. Inspect `runtime.json`
and logs to confirm the hardware actually assigned.

## Monitor, retrieve logs, and download outputs

```bash
kaggle kernels status "$KAGGLE_USERNAME/amg-t4-smoke"
kaggle kernels logs "$KAGGLE_USERNAME/amg-t4-smoke"
kaggle kernels logs "$KAGGLE_USERNAME/amg-t4-smoke" --follow --interval 10
kaggle kernels files "$KAGGLE_USERNAME/amg-t4-smoke"
kaggle kernels output "$KAGGLE_USERNAME/amg-t4-smoke" -p runs/kaggle-smoke-output
```

These commands were checked against the official
[kernels CLI reference](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/kernels.md).
Submission starts a batch job; `push` returning successfully does not mean training
passed. Check the final status and `amg/qualification.json`. For reproducible
retrieval, use a version reference such as `username/amg-t4-smoke/2` with `output`,
as described in the [output documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).

Outputs under `/kaggle/working/amg/` include:

- `job.json`, `runtime.json`: submitted settings and actual hardware.
- `qualification.json`: written only after the full smoke qualification passes.
- `first/`, `resumed/`: checkpoints, manifests, and `metrics.jsonl` for steps 1–2
  and 3–4; `first.log` and `resumed.log` contain CLI output.
- `train/`: checkpoints, manifests, and metrics for a longer training job.

`/kaggle/working` is saved as a completed job's output, **not automatically restored
in a new independent job**. Download valuable outputs promptly. Abrupt termination
may lose outputs that Kaggle has not published; this backend does not perform
mid-job remote backups. Kaggle currently documents up to 12 hours for CPU/GPU
sessions and 20 GB of saved working outputs; check current account limits and keep
runs comfortably below them. See [technical specifications](https://www.kaggle.com/docs/notebooks).
Use step budgets and `--timeout`; checkpoint retention keeps only a few versions.

## Submit training

After the smoke job succeeds:

```bash
uv run python kaggle/prepare.py --username "$KAGGLE_USERNAME" --mode train \
  --config configs/tiny.yaml --steps 1000 --data synthetic:arithmetic
# If source/configs changed, update the code dataset and wait for readiness first.
kaggle kernels push -p runs/kaggle/job --accelerator NvidiaTeslaT4 --timeout 3600
kaggle kernels status "$KAGGLE_USERNAME/amg-t4-train"
kaggle kernels logs "$KAGGLE_USERNAME/amg-t4-train" --follow --interval 10
kaggle kernels output "$KAGGLE_USERNAME/amg-t4-train" -p runs/kaggle-train-output
```

This example trains on generated arithmetic so no external corpus download is
needed. To use your own permitted text, upload it as a private dataset and pass
`--input owner/corpus --data /kaggle/input/corpus/train.txt`. `--input` is repeatable.
Custom YAML can also be supplied from an attached input using an absolute path;
repository YAML must be under `configs/` to be included in the code archive.
Use `backup_dir: null` for Kaggle configs; all outputs belong in working storage.

## Resume from a previous checkpoint

Download the previous job first. Create a separate private input dataset containing
its checkpoint directory (including manifests/versioned checkpoints for recovery):

```bash
mkdir -p runs/kaggle-resume
cp runs/kaggle-train-output/amg/train/* runs/kaggle-resume/
kaggle datasets init -p runs/kaggle-resume
python - <<'PY'
import json, os
from pathlib import Path
p = Path('runs/kaggle-resume/dataset-metadata.json')
m = json.loads(p.read_text())
m.update(id=f"{os.environ['KAGGLE_USERNAME']}/amg-resume", title='AMG Resume',
         licenses=[{'name': 'other'}])
p.write_text(json.dumps(m, indent=2))
PY
kaggle datasets create -p runs/kaggle-resume
kaggle datasets status "$KAGGLE_USERNAME/amg-resume"
# After ready; target 2000 is TOTAL steps, not 2000 additional steps:
uv run python kaggle/prepare.py --username "$KAGGLE_USERNAME" --mode train \
  --steps 2000 --data synthetic:arithmetic \
  --input "$KAGGLE_USERNAME/amg-resume" \
  --resume /kaggle/input/amg-resume/latest.pt
kaggle kernels push -p runs/kaggle/job --accelerator NvidiaTeslaT4 --timeout 3600
```

For later checkpoint uploads use `datasets version` rather than `create`. The
input is read only; resumed training writes new artifacts to working storage.
For resuming the smoke test instead, copy `amg/resumed/*` and keep the smoke
config and original data. Always preserve the exact corpus/tokenizer lineage;
the CLI rejects a mismatched dataset fingerprint. The loader uses the checkpoint's
architecture/tokenizer and restores weights, optimizer, scheduler, scaler, RNG,
data position, tokens seen, and global step. Precision and device can change across
providers, but identical numerical trajectories across hardware are not promised.
The CLI train resume path currently reconstructs a CharacterTokenizer; BPE research
checkpoints are outside this launcher's qualification scope.

## Qualification and troubleshooting

On any prepared CUDA environment, including Colab:

```bash
python scripts/qualify_gpu.py --output-dir /tmp/amg-gpu-qualification
# Or opt into the pytest wrapper (normal CPU CI skips it):
AMG_RUN_GPU_TESTS=1 uv run pytest -q tests/test_kaggle_backend.py -k gpu_qualification
```

Use a fresh output directory. Qualification checks real CUDA identity and memory,
FP16 autocast, finite loss/gradients/parameters, an actual optimizer update with
GradScaler, then two production CLI invocations. It checks checkpoint deserialization,
optimizer step continuity, scaler persistence, increasing data position/token count,
changed weights, finite CLI losses, and global steps 2 → 4. This establishes basic
backend compatibility, not model quality or long-run numerical stability.
BF16 is skipped explicitly: T4 has no native BF16 training support. CPU development
uses FP32; FP16 plus GradScaler is the T4 default.

| Symptom | Action |
| --- | --- |
| Accelerator disabled or job queued | Complete phone verification, check `kaggle quota`, and wait for capacity. |
| Unknown CLI flag / missing `logs` | `uv tool upgrade kaggle`; compare installed `--help` with linked references. |
| Code zip not found | Check `dataset_sources`, dataset readiness and actual `/kaggle/input` mount name. |
| CUDA unavailable | Check GPU metadata/accelerator; inspect runtime image. Do not install CPU torch over CUDA torch. |
| Package installation fails | Ensure internet is enabled and inspect dependency errors in job logs. |
| CUDA out of memory | Reduce batch size/context/accumulation footprint in config; only one T4's VRAM is used. |
| Resume does no training | Set total target steps above the checkpoint step. |
| Dataset/tokenizer mismatch | Attach the original corpus and use the same training data selection. |
| Output missing after timeout | Check job status/output publication; shorten jobs and save more often. |

No SSH, tunnels, keepalive tricks, or unofficial runtime access are required.
