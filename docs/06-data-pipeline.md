# Adaptive Mini-GPT Data Pipeline Guide (Milestone M70)

This document specifies the dataset architecture, provenance tracking, and cross-platform runtime data loading for Adaptive Mini-GPT across **local development**, **Kaggle GPU environments**, and **Google Colab**.

---

## 1. Directory Structure and Git Policies

To maintain reproducibility without bloating the Git repository with large binaries, `data/` follows a strict three-tier layout:

```text
adaptive-mini-gpt/
├── data/
│   ├── specs/          # Tracked in Git: Formal JSON generation configurations
│   ├── fixtures/       # Tracked in Git: Tiny deterministic test fixtures (<50KB)
│   ├── manifests/      # Tracked in Git: Cryptographic dataset manifests (SHA-256)
│   ├── generated/      # Excluded from Git: Local synthetic dataset generation outputs
│   ├── external/       # Excluded from Git: External downloaded benchmarks & corpora
│   └── packages/       # Excluded from Git: Private Kaggle upload bundles
├── src/
│   └── adaptive_mini_gpt/
│       └── data_pipeline/ # Manifest schema, portable loader, generator, packaging
└── runs/               # Experiment artifacts, checkpoints, and benchmark metrics
```

### Git Policy Rules
- **Tracked in Git**: `data/specs/*.json`, `data/fixtures/*.jsonl`, `data/manifests/*.json`.
- **Excluded in `.gitignore`**:
  ```gitignore
  /data/generated/
  /data/external/
  /data/packages/
  *.pt
  *.pth
  *.bin
  ```
- **No Checkpoint Pollution**: Checkpoints and model weights belong strictly in `runs/` or remote object stores, never in `data/`.

---

## 2. Deterministic Dataset Specification & Generation

Synthetic relational environments are generated deterministically from versioned JSON specifications (e.g., `data/specs/m70_relational_benchmark_spec.json`).

### Specification Format
```json
{
  "spec_id": "m70_relational_benchmark",
  "version": "1.0",
  "description": "Standardized multi-family relational reasoning benchmark",
  "default_seed": 42,
  "families": [
    {
      "family_id": "PARITY_CYCLE",
      "description": "Autonomous Rail Parity Grid",
      "relation_name": "dispatches_consist_to",
      "universe_size": 24,
      "is_directed": false
    }
  ],
  "partition_sizes": {
    "train": 60,
    "dev": 30,
    "sealed_test": 30
  }
}
```

### Generating Datasets Locally
Using the `generate_synthetic_dataset` API:
```python
from pathlib import Path
from adaptive_mini_gpt.data_pipeline import generate_synthetic_dataset

manifest, dataset_dir = generate_synthetic_dataset(
    spec_path=Path("data/specs/m70_relational_benchmark_spec.json"),
    output_base_dir=Path("data/generated"),
    seed=42,
    git_commit="8ab2ba14",
)
print("Generated dataset at:", dataset_dir)
print("Total examples:", manifest.total_examples)
```

---

## 3. Cryptographic Manifests and Verification

Every generated dataset produces a `dataset_manifest.json` recording:
1. `schema_version`: Manifest schema format version (e.g. `"1.0"`).
2. `dataset_id` & `version`: Unique dataset identifier and version string.
3. `git_commit`: Pinned Git commit hash of generation source code.
4. `generator_config_hash`: SHA-256 fingerprint of the generation specification file.
5. `seed` & `independent_world_ids`: Deterministic seeds and world identifiers.
6. `partitions`: Partition-level metadata for `train`, `dev`, and `sealed_test`:
   - `relative_path`: Relative path within the dataset folder.
   - `sha256`: Full 64-character SHA-256 checksum of the partition file.
   - `example_count`: Exact number of JSONL records.
   - `label_distribution`: Classwise frequency count (e.g., `{"YES": 20, "NO": 20, "UNKNOWN": 20}`).

### Integrity Verification
Before running any benchmark or training run, the dataset must pass integrity validation:
```python
from adaptive_mini_gpt.data_pipeline import DatasetManifest, verify_dataset_integrity

manifest = DatasetManifest.load(dataset_dir / "dataset_manifest.json")
is_valid, errors = verify_dataset_integrity(manifest, dataset_dir)
if not is_valid:
    raise RuntimeError(f"Integrity check failed: {errors}")
```
The integrity check enforces:
- File existence and exact SHA-256 hash match.
- Exact example count verification.
- **Inter-partition split disjointness**: Guarantees zero question overlap between `train`, `dev`, and `sealed_test` (detects and blocks data leakage).

---

## 4. Cross-Platform Portable Dataset Loading

To avoid platform-specific hardcoded paths (e.g., `/kaggle/input` or `/content/drive`), code resolves datasets using the `AMG_DATA_ROOT` environment variable via `PortableDatasetLoader`.

```python
import os
from pathlib import Path
from adaptive_mini_gpt.data_pipeline import PortableDatasetLoader, resolve_data_root

# Resolves: os.environ.get("AMG_DATA_ROOT", "data/generated")
data_root = resolve_data_root()
loader = PortableDatasetLoader(base_data_root=data_root)

# Load dataset and verify integrity automatically
manifest, dataset_dir = loader.load_and_verify(dataset_id="m70_relational_benchmark", version="1.0")
```

### Platform Defaults
- **Local Development**:
  ```bash
  export AMG_DATA_ROOT="data/generated"
  uv run pytest tests/test_m70_data_pipeline.py
  ```
- **Kaggle GPU**:
  ```bash
  export AMG_DATA_ROOT="/kaggle/input/amg-m70-benchmark-data"
  ```
  Or generate inside `/kaggle/working/data/generated` during runtime.
- **Google Colab**:
  ```bash
  export AMG_DATA_ROOT="/content/data"
  ```

---

## 5. Private Kaggle Dataset Packaging Workflow

For large or expensive-to-generate datasets, versioned private Kaggle Datasets allow immutable sharing between Kaggle and Google Colab.

### 1. Build Lean Package
```python
from pathlib import Path
from adaptive_mini_gpt.data_pipeline import prepare_kaggle_dataset_package

pkg_dir = prepare_kaggle_dataset_package(
    dataset_dir=Path("data/generated/m70_relational_benchmark_v1.0_seed42"),
    output_package_dir=Path("data/packages"),
    owner_slug="sparklinghydraulics0",
    dataset_title="Adaptive Mini-GPT M70 Relational Benchmark",
    is_private=True,
)
```

The packaging utility automatically:
- Copies only partition `.jsonl` files and `dataset_manifest.json`.
- Audits the directory against secret or code leakage (`.env`, `.py`, `.sh`, `.key`, `.pt`, `.pth`).
- Generates `dataset-metadata.json` with `"isPrivate": true`.

### 2. Upload via Kaggle CLI
```bash
# Initialize and create private dataset (first version)
kaggle datasets create -p data/packages/m70_relational_benchmark_v1.0

# Push updated version
kaggle datasets version -p data/packages/m70_relational_benchmark_v1.0 -m "Update seed 42 partition"
```

---

## 6. Google Colab Workflow (`kagglehub`)

Google Colab downloads the immutable pinned dataset version directly to runtime-local storage, avoiding slow mounted Google Drive I/O bottlenecks.

```python
import os
import kagglehub
from adaptive_mini_gpt.data_pipeline import PortableDatasetLoader

# 1. Download pinned dataset version to Colab runtime-local storage
dataset_path = kagglehub.dataset_download("sparklinghydraulics0/m70-relational-benchmark/versions/1")
print("Downloaded pinned dataset to:", dataset_path)

# 2. Point AMG_DATA_ROOT to downloaded path
os.environ["AMG_DATA_ROOT"] = dataset_path

# 3. Load and verify
loader = PortableDatasetLoader()
manifest, dataset_dir = loader.load_and_verify("m70_relational_benchmark", "1.0")
print("Dataset verified successfully on Colab:", manifest.dataset_id)
```

### Security & Privacy Rules
- **Never commit Kaggle API tokens** or Colab secret keys to Git.
- Use Colab's native Secrets manager (`userdata.get('KAGGLE_KEY')`) for private dataset downloads.
- Training outputs and experiment logs can optionally be written to `/content/drive/MyDrive/amg_runs`, but raw training data must remain on runtime-local disk for fast throughput.
