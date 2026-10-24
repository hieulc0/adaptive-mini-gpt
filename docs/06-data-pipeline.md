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


---

## 7. Milestone M71 Packaging Allowlist & Security Enhancements

Milestone M71 introduced explicit security and isolation audits for dataset generation and packaging:

### 1. Strict File Extension Allowlist
To prevent unintended binary, code, or secret leakage, dataset packaging enforces an explicit allowlist in `src/adaptive_mini_gpt/data_pipeline/packaging.py`:
```python
ALLOWED_DATASET_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".txt",
    ".csv",
    ".tsv",
    ".parquet",
}
```
Any unlisted file extensions encountered during dataset packaging raise a security exception and abort packaging immediately.

### 2. Automated Content Security & Credential Scanning
Before bundling or uploading, each file is scanned via `audit_file_content_security()`. Any occurrence of pattern fingerprints (e.g., `api_key`, `kaggle_key`, `secret_key`, `BEGIN PRIVATE KEY`, `hf_token`) immediately triggers an error, preventing confidential keys or tokens from being committed or uploaded.

### 3. Cross-Split World Isolation Verification
The dataset manifest verifier (`verify_dataset_integrity`) now audits inter-partition isolation to ensure disjoint world seeds and prevent leaking graph templates between training and sealed evaluation worlds:
```python
# Verified automatically by verify_dataset_integrity(manifest, dataset_dir)
# Checks:
# 1. SHA-256 partition digests
# 2. Example counts and class distributions
# 3. Disjoint question inputs between train, dev, and sealed_test
# 4. Independent world instance segregation
```

### 4. Cross-Platform Colab Smoke Test
A standalone validation script is provided to verify dataset loading in Google Colab or remote GPU environments:
```bash
uv run python scripts/smoke_test_colab_data_loading.py
```
This script verifies local synthesis, manifest creation, and `PortableDatasetLoader` resolution in both clean and fallback environments.


### 5. Milestone M73 Causal Neural Memory Specification
Milestone M73 introduces `data/specs/m73_cost_efficient_memory_spec.json`, formalizing:
- Multi-world relational tasks including modular arithmetic congruence (`MOD_3_CONGRUENCE`, `MOD_5_CONGRUENCE`) and relational subsumption alongside graph traversals.
- Exact split isolation and disjoint relation keys across train, development, and sealed test partitions.
- Prompt-observable relation tokens enabling `SELECTIVE_NO_EXTRA_PROBES` to perform zero-query routing without extra observation queries.


### 6. Milestone M74 High-Information Procedural Specification
Milestone M74 introduces `data/specs/m74_high_information_spec.json`, formalizing:
- 5 deterministic procedural sequence transformation families over a controlled 16-symbol discrete alphabet (`CAESAR_SHIFT_MOD3`, `CAESAR_SHIFT_MOD5`, `AFFINE_TRANSFORM`, `COMPLEMENT_REFLECT`, `NONLINEAR_SBOX`).
- Multi-token sequence Exact Match evaluation with combinatorial random chance $\le 0.024\%$.
- Strict partition isolation enforcing disjoint symbol combinations across `train`, `dev`, `qualification`, and `sealed_test` sets.


### 7. Milestone M75 Experience-Guided Learning Specification
Milestone M75 introduces `data/specs/m75_experience_spec.json`, formalizing:
- 10 deterministic procedural sequence transformation families categorized into 4 distinct generalization tiers:
    1. `NEW_INPUTS_KNOWN_RULE`: Caesar shift mod 3, mod 5, Affine (3s+1), Complement reflect, Nonlinear S-box.
    2. `NEW_PARAMETERIZATION`: Caesar shift mod 7, Affine (5s+3).
    3. `COMPOSITIONAL_PROCEDURE`: Composite Reflect-then-Shift ((15 - s + 4) mod 16).
    4. `UNFAMILIAR_FAMILY`: Bitwise XOR mask (s ^ 9), Deceptive Parity Inversion.
- Strict 4-way split isolation per task instance:
    * `train`: 48 adaptation examples.
    * `dev`: 16 observable examples for pre-adaptation feature extraction and zero-shot probing.
    * `qualification`: 16 held-out examples for candidate adapter qualification.
    * `sealed_test`: 48 strictly segregated test examples for post-hoc evaluation.
- Decoupled target scrambling generator: uniform random replacement across the 16-symbol alphabet, eliminating permutation artifacts and information leakage.
- Compatible with `PortableDatasetLoader` and `AMG_DATA_ROOT` environment override on Kaggle and Google Colab runtimes.

### 8. Milestone M76 Causal Neural Skill Composition Specification
Milestone M76 introduces `data/specs/m76_composition_spec.json`, formalizing:
- Three independently trained and qualified atomic procedural primitives:
    1. `COMPLEMENT_REFLECT`: $s \to 15 - s$
    2. `CAESAR_SHIFT_MOD3`: $s \to (s + 3) \bmod 16$
    3. `AFFINE_TRANSFORM`: $s \to (3s + 1) \bmod 16$
- Noncommutative compound procedural transformations ($A \circ B \neq B \circ A$):
    * `REFLECT_THEN_SHIFT3`: $s \to 15 - s \to (15 - s + 3) \bmod 16 = (2 - s) \bmod 16$
    * `SHIFT3_THEN_REFLECT`: $s \to (s + 3) \bmod 16 \to 15 - ((s + 3) \bmod 16) = (12 - s) \bmod 16$
    * `REFLECT_THEN_AFFINE`: $s \to 15 - s \to (3(15 - s) + 1) \bmod 16 = (14 - 3s) \bmod 16$
    * `AFFINE_THEN_SHIFT3`: $s \to (3s + 1) \bmod 16 \to ((3s + 1) + 3) \bmod 16 = (3s + 4) \bmod 16$
- Strict 4-way partition isolation enforcing disjoint symbol combinations across `train` (48), `dev` (16), `qualification` (16), and `sealed_test` (48) sets.
- 9-Arm causal evaluation suite: `A_NEURAL_COMPOSITION`, `B_WRONG_ORDER`, `C_SINGLE_SKILL_A`, `D_SINGLE_SKILL_B`, `E_SHUFFLED_COMPONENT`, `F_UNRELATED_COMPONENT`, `G_FRESH_MONOLITHIC`, `H_UNCHANGED_BASE`, `I_DETERMINISTIC_ORACLE`.
- Two-stage error decomposition: `STAGE_1_DEFECT`, `STAGE_2_DEFECT`, `ERROR_PROPAGATION`, and `DUAL_DEFECT` using teacher-forced reference intermediates.
- Reversible candidate rollback and frozen backbone double SHA-256 parameter invariance.

### 9. Milestone M77 Autonomous Plan Discovery Specification
Milestone M77 introduces `data/specs/m77_plan_discovery_spec.json`, formalizing:
- Task identifiability benchmark across varying demonstration budgets ($N \in \{0, 2, 4, 8\}$).
- Candidate composition hypothesis space over 3 qualified primitives (`COMPLEMENT_REFLECT`, `CAESAR_SHIFT_MOD3`, `AFFINE_TRANSFORM`), yielding 10 candidate plans (1 abstain, 3 single-primitive, 6 two-stage compositions).
- Benchmark tasks:
    1. Four composable noncommutative compound tasks: `REFLECT_THEN_SHIFT3`, `SHIFT3_THEN_REFLECT`, `REFLECT_THEN_AFFINE`, `AFFINE_THEN_SHIFT3`.
    2. Two unfamiliar uncomposable procedural families: `BITWISE_XOR_MASK` ($s \to s \oplus 9$), `DECEPTIVE_INVERSION` ($s \to 15-s$ if even else $(s+7)\bmod 16$).
- Scrambled demonstration generator: Breaks input-output correspondence to establish causal plan selection controls (`SHUFFLED_DEMONSTRATIONS`).
- Strict partition isolation: `demonstrations` ($N \in \{0, 2, 4, 8\}$), `dev` (16 examples), and `sealed_test` (48 examples).
- Execution provenance: Full validation on remote Google Colab Tesla T4 GPU (`amg-m77` session) with zero data leakage into planning heuristics.

### 10. Milestone M78 Reliable Plan Discovery Specification
Milestone M78 introduces `data/specs/m78_plan_discovery_spec.json`, formalizing:
- **Normalized Sequence Log-Likelihood Scoring Engine**:
    $$S(P) = \frac{1}{N}\sum_{i=1}^N \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \log p_P(y_{i,t} \mid x_i)$$
    Evaluates probability mass assigned to demonstrated outputs, resolving discrete quantization and primitive noise sensitivity.
- **Evidence Margin & Posterior Plan Distribution**:
    $$\Delta S = S(P_{(1)}) - S(P_{(2)}), \quad \pi(P_k) = \frac{\exp(\beta \cdot S(P_k))}{\sum_j \exp(\beta \cdot S(P_j))}$$
- **Expanded Unsupported / Unfamiliar Procedural Families (40 Tasks)**:
    1. `BITWISE_XOR_MASK`: $y = (x \oplus 9) \bmod 16$
    2. `DECEPTIVE_INVERSION`: conditional parity reflection
    3. `MODULAR_SUBTRACTION`: $y = (11 - 2x) \bmod 16$
    4. `NONLINEAR_SBOX`: nonlinear substitution box $[7, 12, 14, 9, 2, 1, 5, 15, 11, 6, 13, 0, 4, 8, 10, 3]$
- **Factorial Multi-Seed Benchmark Design**:
    * 5 independent seeds: 42, 43, 44, 45, 46.
    * 24 tasks per seed (16 composable across 4 compound pairs $\times$ 4 replications; 8 uncomposable across 4 families $\times$ 2 replications).
    * Total: 120 task instances ($80$ composable, $40$ uncomposable), with 48 sealed test sequences per task ($3,840$ sealed evaluations per causal arm).
- **Paired Statistical Modeling**:
    * Paired Student-t test across 5 independent seeds ($df = 4, t_{\text{crit}} = 2.776$ at 95% two-tailed confidence).
- **Execution Provenance**:
    * Evaluated on Google Colab Tesla T4 GPU (`amg-m78` session) with bit-exact frozen model hash verification.

### 11. Milestone M79 Autonomous Active Experimentation Specification
Milestone M79 introduces `data/specs/m79_active_spec.json`, formalizing:
- **Phase 0 Forensic Audit & Lineage Reconciliation**:
    * Reconciles cross-milestone ceiling shifts via base model initialization and primitive adapter qualification fidelity (0.7917 vs 1.0000 EM).
    * Mathematically proves functional equivalence of (R o A)(x) == (A o R)(x) == (14 - 3x) mod 16, explaining why test exact match (88.85%) exceeded plan identity accuracy (80.00%) in M78.
- **Minimal Active Experiment Engine**:
    * Ambiguity detection triggering: Delta S = S(P_(1)) - S(P_(2)) < 0.08.
    * Minimally discriminative probe synthesis: searches synthetic sequence space {0, ..., 15}^3 for x* where P_(1)(x*) != P_(2)(x*), prohibiting any access to sealed test sets or hidden family IDs.
    * Verified environment interaction: budget strictly bounded at <= 1.5 queries/task (observed 0.15 queries/task).
    * Dynamic hypothesis belief updating with newly acquired evidence (x*, y*).
- **Epistemic Skill-Gap Classification**:
    * Four distinct states: `KNOWN_CONFIDENT`, `KNOWN_AMBIGUOUS`, `SKILL_GAP_UNSUPPORTED`, `INCONSISTENT_NOISY`.
    * Unsupported task detection: low top likelihood (S(P) < -1.65) triggers direct abstention, preventing wasted queries on uncomposable tasks.
- **6-Arm Causal Active Inquiry Suite**:
    * `A_PASSIVE_M78`: Frozen M78 likelihood baseline (no queries).
    * `B_RANDOM_QUERY`: 1 uniformly random probe input.
    * `C_NON_DISCRIMINATIVE`: 1 agreement placebo probe (P_(1)(x) == P_(2)(x)).
    * `D_ACTIVE_DISCRIMINATIVE`: M79 maximal disagreement probe (P_(1)(x*) != P_(2)(x*)).
    * `E_ORACLE_PROBE`: Oracle disambiguation upper bound.
    * `F_UNSUPPORTED_CONTROL`: Skill-gap recognition and abstention on 40 unfamiliar tasks.
- **Execution Provenance**:
    * 330/330 unit and regression tests passed on Google Colab Tesla T4 GPU (`amg-m79` session).
    * Statistically significant improvement over passive M78 (+11.74% EM, t = 3.0647, p = 0.0375).
    * Strict 0 parameter growth and 0 compound training steps.

### 12. Milestone M80 Autonomous Neural Skill Acquisition Specification
Milestone M80 introduces `data/specs/m80_acquisition_spec.json`, formalizing:
- **Phase 1 Forensic Audit & Baseline Reconciliation**:
    * Reconciles M79 query accounting (0.15 queries/task) and explains nominal oracle control underperformance via RNG offset artifact.
- **Epistemic Diagnostic Detection**:
    * Evaluates candidate plans; when top likelihood $S(P) < -1.65$, task is classified as `SKILL_GAP_UNSUPPORTED`, directly triggering neural adaptation.
- **Autonomous Neural Adapter Acquisition**:
    * Bounded parameter budget: Rank-16 bottleneck adapter (2,592 parameters) vs Rank-8 bottleneck adapter (1,432 parameters).
    * Training steps budget sweep: $T \in \{20, 40, 60, 80\}$ steps on 48 practice examples.
    * Strict qualification admission gate: held-out exact match $\ge 0.85$ (85%).
- **Comparative Learning-to-Learn Initialization Modes**:
    * `FRESH_RANDOM`: fresh student without pretrained backbone representations.
    * `PRETRAINED_BACKBONE`: adapter trained on frozen Transformer representations.
    * `EXPERIENCE_WARM_START`: warm-started from related prior skill adapter weights.
    * `SHUFFLED_WEIGHTS`: scrambled parameter control for causal weight attribution.
- **Dynamic Repertoire Expansion & Combinatorial Plan Space**:
    * Repertoire expansion: 3 primitives $\to$ 4 primitives (`COMPLEMENT_REFLECT`, `CAESAR_SHIFT_MOD3`, `AFFINE_TRANSFORM`, `BITWISE_XOR_MASK`).
    * Plan hypothesis space: expands from 10 to 17 candidate plans (1 abstain, 4 single-primitive, 12 two-stage sequential compositions).
    * Bit-exact rollback support: candidate removal restores initial 10-plan space with 0 residual parameter traces.
- **Zero-Shot Compound Transfer Evaluation (6 Arms)**:
    * `AUTONOMOUS_PLAN_EXPANDED`: autonomous plan discovery over 17 plans.
    * `ORACLE_PLAN_PROVIDED`: composition with ground-truth plan ($P^*$).
    * `INVERTED_ORDER_CONTROL`: reversed execution order ($P_{\text{inv}}$).
    * `SINGLE_SKILL_ABLATION`: single primitive alone.
    * `SHUFFLED_ACQUIRED_ADAPTER`: scrambled adapter weights control.
    * `FRESH_MONOLITHIC_ADAPTER`: direct end-to-end trained compound adapter reference.
- **Execution Provenance**:
    * 336/336 unit and regression tests passed on Google Colab Tesla T4 GPU (`amg-m80` session).
    * 100% carry (20/20) and logic (8/8) retention; base parameter SHA-256 hash invariant (`6e39cd7ef6790b3c`).

### 13. Milestone M81 Learning-to-Learn Specification
Milestone M81 introduces continual learning curriculum and sample efficiency specifications:
- **Phase 0 Forensic Audit & Statistical Corrections**:
    * Reconciles base parameter counts (928k vs 1.07M) and enforces Student-t tests with $	ext{df} = 4$ for 5-seed benchmarks.
- **Continual Skill Learning Curriculum (4 Progressive Families)**:
    * `ARITHMETIC`: Modular affine transformations $(a \cdot s + b) mod 16$ with unseen parameter pairs.
    * `CONDITIONAL_LOGIC`: Parity-dependent branching with bitwise XOR masking and additive offsets.
    * `SYMBOLIC_PROCEDURE`: Position-dependent sequence reversal and cyclic transformations.
    * `COMPOSITION`: Two-stage sequential pipelines testing cross-family integration.
- **Disjoint Partition Guarantees**:
    * `train` (48), `dev` (16), `qualification` (16), `sealed_test` (48). Strictly zero token-level overlap.
- **Minimal Capacity Hierarchy**:
    * Evaluates Rank-4 (852 params) $	o$ Rank-8 (1,432 params) $	o$ Rank-16 (2,592 params).
    * Chooses smallest candidate satisfying $\ge 85\%$ qualification exact match threshold.
- **Learning-to-Learn Efficiency Metrics**:
    * Steps to $\ge 85\%$ EM: Evaluated at 5-step intervals ($T \in \{5, 10, \dots, 60\}$).
    * Primary Target: $\ge 20\%$ reduction in median steps with positive paired confidence interval.
- **Causal Ablation Suite**:
    * Evaluates Oracle Plan, Inverted Order, and Shuffled Weights on multi-stage compositions.
