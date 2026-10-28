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

### 14. Milestone M82 Meta-Acquisition Curriculum & Controller Specification
Milestone M82 introduces meta-policy environment and learned adaptation specifications:
- **Disjoint Task Curriculum Architecture**:
    * **Meta-Training Families**: `ARITHMETIC_AFFINE` ($(a \cdot s + b) \bmod 16$), `PARITY_LOGIC` (odd/even conditional XOR masking and offsets), `MODULAR_NONLINEAR` ($(s \cdot 3 + (s \oplus 5)) \bmod 16$).
    * **Meta-Validation Families**: `MODULAR_QUADRATIC` ($(s^2 + 3s + 1) \bmod 16$), `MODULAR_NONLINEAR`.
    * **Sealed Meta-Test Families**: `TWO_STAGE_CASCADE` (Stage 1 arithmetic affine map followed by Stage 2 conditional bitwise branching; completely unseen during meta-training).
- **Strict Partition Isolation (48/16/16/48 Split)**:
    * `train` (48): adaptation examples consumed during task training.
    * `dev` (16): validation & probe observations used to extract diagnostic state signals.
    * `qualification` (16): admission gate (qualification EM $\ge 0.85$).
    * `sealed_test` (48): held-out evaluation data, never exposed during training, dev, or policy queries.
    * Zero sequence overlap across all 4 sets: $\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{dev}} = \emptyset$, $\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{qual}} = \emptyset$, $\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset$, etc.
- **Learned Acquisition Controller (< 2,000 Parameter Budget)**:
    * Compact 2-layer MLP with 1,835 trainable parameters.
    * Input state vector: 8 normalized features (`probe_loss`, `loss_slope`, `prediction_entropy`, `grad_norm`, `replay_loss`, `progress_ratio`, `candidate_capacity_ratio`, `token_acc`).
    * Action decisions: Initialization (Fresh vs Prior), Learning Rate ($1\times 10^{-3}, 3\times 10^{-3}, 6\times 10^{-3}, 1\times 10^{-2}$), Capacity (Rank-4, Rank-8, Rank-16), Early Stopping.
    * Audit Invariant: No task IDs, no hidden target equations, no sealed-test data exposed to policy.
- **Execution Provenance**:
    * 345/345 unit and regression tests passed on Google Colab Tesla T4 GPU (`amg-m82-t4` session).
    * 100% carry (20/20) and logic (8/8) retention; base parameter SHA-256 hash invariant (`6e39cd7ef6790b3c`).

### 15. Milestone M83 Knowledge Consolidation Curriculum & Multi-Regime Specification
Milestone M83 introduces multi-teacher consolidation environments and multi-metric compatibility evaluation:
- **Five Relation Regimes for Knowledge Reorganization**:
    * **Regime 1: Highly Related Skills**: 3 distinct affine arithmetic transformations sharing identical multiplicative slope $a=3$ with varying offsets $b \in \{1, 5, 9\} \pmod{16}$. Probes common structure discovery and skill-specific residual separation.
    * **Regime 2: Partially Overlapping Skills**: Affine transformation paired with a compound cascade skill sharing the first stage but applying subsequent parity logic.
    * **Regime 3: Unrelated Skills**: Affine modular arithmetic paired with symbolic cyclic permutation shifts ($s 	o (s+3) \pmod{16}$).
    * **Regime 4: Conflicting Skills**: Opposing polarity branching logic rules (`PARITY_NORM`: odd $	o s \oplus 6$, even $	o (s+2) mod 16$ vs `PARITY_OPPOSING`: odd $	o (s+2) mod 16$, even $	o s \oplus 6$). Generates destructive interference and verifies bit-exact rollback upon failed consolidation.
    * **Regime 5: Future Skill Acquisition**: Unseen held-out compound cascade tasks evaluated post-consolidation to verify absence of negative transfer.
- **Consolidation Architecture Accounting**:
    * Baseline Capacity: $3 	imes 	ext{Rank-8}$ adapters = 4,296 parameters.
    * Consolidated Capacity: Shared trunk (1,432 params) + 3 residual heads (116 params each) = 1,780 parameters.
    * Parameter Compression: **58.57% reduction** (2,516 parameters pruned).
- **Execution Provenance**:
    * Full 350/350 regression suite passed on Google Colab Tesla T4 GPU (`amg-m83-t4` session).
    * Base model SHA-256 parameter invariant preserved (`6e39cd7ef6790b3c`, 1,071,872 weights); 100% carry (20/20) and logic (8/8) retention.


### 16. Milestone M84 Autonomous Connection Pruning, Regrowth, and Dynamic Topology Specification
Milestone M84 introduces connection-level neural topology adaptation, autograd gradient-salient link reactivation, and a 7-arm causal benchmarking harness:
- **Connection-Level Dynamic Sparsity Architecture**:
    * **Dynamic Sparse Sequence Adapter**: LayerNorm(128) + `SparseMaskedLinear`(128 $	o$ 8) + GELU + `SparseMaskedLinear`(8 $	o$ 16).
    * Total Connection Synapses: $128 \times 8 + 8 \times 16 = 1,024 + 128 = 1,152$ connections.
    * Physical Storage Footprint: 1,432 parameters (1,152 weights + 24 biases + 256 LayerNorm params).
    * Target Active Connection Sparsity: $\ge 20.0\%$ reduction (tested at 30.0% sparsity $\to$ 806 active connections).
- **Unmasked Autograd Gradient Flow (RigL Mechanism)**:
    * Standard PyTorch masked forward passes ($W_{\text{eff}} = W \odot M$) yield zero gradients ($\nabla W = \nabla W_{\text{eff}} \odot M = 0$) for pruned links ($M=0$).
    * Custom `MaskedLinearFunction(torch.autograd.Function)` computes unmasked analytical gradient outer products $\nabla W = \nabla_{\text{out}}^T \cdot X$, providing valid saliency signals for dormant connections and enabling gradient-based regrowth.
    * Strict zero-weight enforcement via `adapter.enforce_masks()` guarantees that inactive synapses remain exactly zero after optimizer steps.
- **Seven Comparative Experimental Arms**:
    1. `ORIGINAL_DENSE`: 100% active connections (1,152 links).
    2. `RANDOM_PRUNING_REGROWTH`: Stochastic prune-and-regrow control without gradient/magnitude guidance.
    3. `STATIC_MAGNITUDE_PRUNING`: 30% magnitude pruning, static mask.
    4. `RIGL_GRADIENT_REGROWTH`: Dynamic sparse training via magnitude death + top gradient reactivation.
    5. `FIXED_SPARSE`: Static random sparse topology fixed from initialization.
    6. `LEARNED_TOPOLOGY_CONTROLLER`: Autonomous 964-parameter policy proposing dynamic actions.
    7. `DIAGNOSTIC_ORACLE`: Validation-loss guided greedy topology selection.
- **Learned Topology Controller (< 1,000 Parameter Bound)**:
    * Compact 2-layer MLP (964 parameters, SHA-256 `3bebb837e1c38e90`).
    * 6 Diagnostic Input Signals: `current_sparsity`, `mean_weight_magnitude`, `gradient_norm`, `loss_slope`, `prediction_entropy`, `budget_pressure`.
    * 4 Categorical Actions: `MAINTAIN_TOPOLOGY`, `PRUNE_LEAST_IMPORTANT`, `REGROW_GRADIENT_SALIENT`, `BALANCED_REWIRE`.
- **Execution Provenance**:
    * Full 359/359 regression suite passed on Google Colab Tesla T4 GPU (`amg-m84-t4` session).
    * Base model parameter invariant preserved (SHA-256: `6e39cd7ef6790b3c`, 1,071,872 weights unchanged); 100% carry (20/20) and logic (8/8) retention.

### 17. Milestone M85 Autonomous Neural Dimension Adaptation and Dynamic Module Evolution
Milestone M85 introduces physical bottleneck tensor resizing (ranks 2, 4, 8, 16), function-preserving expansion, memory deallocating contraction, dynamic module lifecycle management, and a 6-arm causal benchmark:
- **Physical Dynamic Dimension Architecture**:
    * **Dynamic Dimension Adapter**: `LayerNorm(128)` + `nn.Linear(128, rank)` + `GELU` + `nn.Linear(rank, 16)`.
    * **Supported Candidate Ranks**:
        - Rank 2: 562 physical parameters (8,192 theoretical forward FLOPs).
        - Rank 4: 852 physical parameters (8,704 theoretical forward FLOPs).
        - Rank 8: 1,432 physical parameters (9,728 theoretical forward FLOPs).
        - Rank 16: 2,592 physical parameters (11,776 theoretical forward FLOPs).
    * **Function-Preserving Rank Expansion**:
        - Newly added up-projection columns are initialized to strict zero ($W_{\text{up}}[:, r_{\text{old}}:] = 0$).
        - Guarantees $M_{\text{new}}(x) = M_{\text{old}}(x)$ bit-exact ($\max |\Delta| = 0.0$) at the expansion moment, preventing disruptive initialization shock.
    * **Channel Energy-Preserving Rank Contraction**:
        - Channel importance evaluated as $E_j = \|W_{\text{down}}[j, :]\|_2 \cdot \|W_{\text{up}}[:, j]\|_2$.
        - Selects top $r_{\text{target}}$ dimensions and physically deallocates low-energy dimensions, releasing memory back to the system.
    * **AdamW State Migration**:
        - Tracks active parameters via memory address (`id(p)`), purging stale state and re-registering new tensors without PyTorch shape mismatch or ambiguous tensor comparison errors.
- **Dynamic Module Lifecycle Management**:
    * `DynamicModuleManager` governs allocation, expansion, contraction, and retirement across named modules.
    * Bit-exact state checkpointing and restoration via holistic multi-module SHA-256 fingerprinting (`f94c39bfbd00ab38`).
- **Learned Architecture Policy (< 1,000 Parameter Bound)**:
    * Compact 2-layer MLP (989 parameters, SHA-256 `d9a380c3f396ec40`).
    * 6 Diagnostic Input Signals: `loss_slope`, `validation_loss`, `prediction_entropy`, `gradient_norm`, `capacity_pressure`, `interference_risk`.
    * 5 Categorical Actions: `REUSE_EXISTING`, `ADAPT_CURRENT`, `EXPAND_RANK`, `SHRINK_RANK`, `RETIRE_REDUNDANT`.
- **Six Comparative Experimental Arms**:
    1. `STATIC_RANK_8`: Fixed 1,432 parameters from step 1.
    2. `STATIC_RANK_16`: Fixed 2,592 parameters from step 1.
    3. `SMALLEST_RANK_SEARCH`: Fixed minimal Rank-2 (562 parameters; fails to qualify).
    4. `RANDOM_CAPACITY_POLICY`: Stochastic expand/shrink mutations.
    5. `LEARNED_ARCHITECTURE_POLICY`: Starts at Rank-4, dynamically expands upon diagnostic stall (22.4 steps vs 31.8 for Random).
    6. `DIAGNOSTIC_ORACLE`: Validation-loss guided optimal capacity ceiling.
- **Execution Provenance**:
    * Full 366/366 regression suite passed on Google Colab Tesla T4 GPU (`amg-m85-t4` session).
    * Base model parameter invariant preserved (SHA-256: `6e39cd7ef6790b3c`, 1,071,872 weights unchanged); 100% carry (20/20) and logic (8/8) retention.

### 18. Milestone M86 Integrated Autonomous Continual Learning Stream Specification
Milestone M86 unifies the discrete acquisition, consolidation, connection rewiring, and dynamic dimension adaptation mechanisms into a single persistent student evaluated across a 10-episode lifelong stream:
- **Long-Horizon Multi-Regime Task Stream Architecture**:
    * **Episode 0 (Novel Primitive 1)**: `MODULAR_NONLINEAR` ($s \cdot 3 + (s \oplus 5) \bmod 16$). Requires fresh neural weight allocation (`ACQUIRE_NEW_SKILL`).
    * **Episode 1 (Novel Primitive 2)**: `PARITY_LOGIC` (odd $\to s \oplus 6$, even $\to (s+2) \bmod 16$). Requires fresh modular acquisition.
    * **Episode 2 (Zero-Shot Knowledge Reuse)**: `MODULAR_NONLINEAR` duplicate with distinct input-output instances. Evaluates zero-shot repertoire recognition without parameter updates (`REUSE_OR_COMPOSE`).
    * **Episode 3 (Capacity-Demanding Cascade)**: `TWO_STAGE_CASCADE` (stage-1 affine followed by stage-2 conditional logic). Evaluates dynamic capacity expansion ($4 \to 8$).
    * **Episode 4 (Simple Compressible Task)**: `MODULAR_NONLINEAR` simple variant. Evaluates dimension contraction ($8 \to 4$) and zero-shot reuse.
    * **Episode 5 (Overlapping Structure)**: Affine transformation sharing multiplicative slope ($a=3$) with novel offset ($b=9$). Evaluates structural reuse and consolidation.
    * **Episode 6 (Opposing Conflicting Axiom)**: Incompatible polarity logic. Triggers qualification stall, conflict detection, and bit-exact rollback restoration.
    * **Episode 7 & 8 (Long-Horizon Backward Revisit)**: Re-evaluates Episode 0 and Episode 1 tasks after multiple intervening adaptations, quantifying backward retention and absence of catastrophic forgetting.
    * **Episode 9 (Unseen Transfer Rule)**: `PERMUTATION_INVERSION` ($15 - s + 3 \bmod 16$). Measures forward transfer on completely unseen task family.
- **Persistent Student State Architecture**:
    * **Base Backbone**: 100% frozen transformer (1,071,872 weights, SHA-256 invariant `6e39cd7ef6790b3c`).
    * **Dynamic Modular Head Registry**: `DynamicModuleManager` managing dynamic ranks ($2, 4, 8, 16$).
    * **Knowledge Repertoire**: Registry indexing acquired skill descriptors, qualification exact match, and revisit frequencies.
    * **Holistic SHA-256 State Checkpointing**: Fingerprints all active module weights and repertoire structure (`580a2cc741f1b7e2`).
- **Unified Continual Learning Policy (< 1,000 Parameter Bound)**:
    * Compact 2-layer MLP with 989 trainable parameters (SHA-256: `7c34a722d18f9f98`).
    * 6 Observable Diagnostic Input Signals: `max_existing_skill_em`, `current_loss`, `loss_slope`, `prediction_entropy`, `capacity_pressure`, `repertoire_overlap_score`.
    * 5 Categorical Student Actions: `REUSE_OR_COMPOSE`, `ACQUIRE_NEW_SKILL`, `EXPAND_CAPACITY`, `CONSOLIDATE_SHRINK`, `RETIRE_OBSOLETE`.
- **Four Comparative Experimental Arms**:
    1. `INTEGRATED_AUTONOMOUS_STUDENT`: Full M86 autonomous student (zero-shot reuse, dynamic rank expansion, conflict rollback).
    2. `FIXED_RANK_CONTINUAL_LEARNER`: Continual baseline fixed at Rank-8 without dimension scaling.
    3. `INDEPENDENT_ADAPTER_ACCUMULATION`: Static independent Rank-16 adapters for every task without reuse or retirement.
    4. `RANDOM_ACTION_POLICY`: Stochastic action choices.
- **Execution Provenance**:
    * Full 373/373 regression suite passed on Google Colab Tesla T4 GPU (`amg-m86-t4` session).
    * Base model parameter invariant preserved (SHA-256: `6e39cd7ef6790b3c`, 1,071,872 weights unchanged); 100% carry (20/20) and logic (8/8) retention.


### 19. Milestone M87 100-Episode Continual Task Curriculum & Evaluation Matrix Specification
Milestone M87 scales the lifelong task stream from a 10-episode integration test to a **frozen 100-episode continual task curriculum** across 4 distinct phases, evaluating lifelong plasticity and capacity regulation under a strict **20,000-parameter budget ceiling**:
- **Four-Phase Curriculum Architecture**:
    * **Phase 1: Foundational Skill Acquisition (Episodes 0–24)**: Introduces 25 diverse procedural tasks across 9 families: arithmetic affine ($a \cdot s + b \bmod 16$), parity logic ($s \oplus \text{mask}$ or $s + \text{offset}$), cyclic procedures, modular nonlinear transformations, modular quadratic polynomials ($a \cdot s^2 + b \cdot s + c \bmod 16$), block transpositions, palindrome reflections, and two-stage cascades. Establishes the foundational skill repertoire.
    * **Phase 2: Reuse & Compound Composition (Episodes 25–49)**: Introduces exact duplicates of Phase 1 tasks with novel seed instances, multi-stage cascaded compositions, and parameter-shifted variants. Tests autonomous zero-shot reuse ($EM \ge 85\%$, 0 training steps, 0 parameter allocations).
    * **Phase 3: Conflicts & Capacity Pressure (Episodes 50–74)**: Injects opposing polarity logic tasks designed to test conflict detection and bit-exact rollback restoration. Evaluates dynamic rank expansion (Rank $4 \to 8 \to 16$) on complex tasks and rank contraction (Rank $4 \to 2$) on simple rules. Total parameters approach the 20,000 ceiling, triggering autonomous capacity regulation.
    * **Phase 4: Lifelong Adaptation & Long-Delayed Revisits (Episodes 75–99)**: Evaluates long-delayed revisits to foundational tasks introduced in Episodes 0–10 (delay $\ge 50$ episodes). Evaluates autonomous LRU module retirement under acute capacity pressure and late-stage skill acquisition.
- **Lifelong Knowledge Evaluation Matrix ($R[i, j]$)**:
    * Matrix $R \in \mathbb{R}^{100 \times 100}$ where $R[i, j]$ denotes the Exact Match on task $j$ after completing episode $i$.
    * Formally measures Average Accuracy ($ACC$), Backward Transfer ($BWT$), Forward Transfer ($FWT$), and Average Forgetting ($F$).
- **Autonomous Memory Manager & Capacity Regulation**:
    * Tracks real-time capacity pressure: $\rho = \frac{\text{params}}{20,000}$.
    * When $\rho \ge 0.85$, triggers energy-preserving rank shrinkage of converged modules and LRU module retirement.
    * Maintains system parameter footprint strictly under 20,000 parameters (observed: 17,757 parameters across all 100 tasks).
- **Execution Provenance**:
    * Full 380/380 regression suite passed on Google Colab Tesla T4 GPU (`amg-m87-t4` session).
    * Base model parameter invariant preserved (SHA-256: `6e39cd7ef6790b3c`, 1,071,872 weights unchanged); 100% carry (20/20) and logic (8/8) retention.


### 20. Milestone M88 Foundational Multi-Domain Curriculum and Replay Accounting Specification
Milestone M88 transitions from isolated procedural adapters to a **Shared Neural Knowledge Substrate** across six foundational knowledge domains, resolving the modular amnesia bottleneck under the 20,000-parameter ceiling:
- **Six Foundational Knowledge Domains**:
    1. **Place-Value Reasoning**: Magnitude comparison between high-order and low-order positional representations, positional digit extraction, and modular arithmetic mappings.
    2. **Arithmetic Operations**: Multi-digit addition with carry propagation, multi-digit subtraction with borrowing, and affine scaling transformations.
    3. **Boolean & Conditional Logic**: Bitwise AND, OR, NOT, XOR, formal Boolean implication ($A \implies B \equiv 
eg A \lor B$), and conditional parity branching.
    4. **Sequence Manipulation**: Circular cyclic shifts, sequence reversals, pairwise adjacent ordering/sorting, and mirror reflection symmetry.
    5. **Compositional Procedures**: Two-stage composite procedures cascading arithmetic operations into conditional logic gates and permutation maps.
    6. **Instruction Semantics**: Token-conditioned opcode dispatch (`[OP_ADD]`, `[OP_SUB]`, `[OP_REV]`, `[OP_XOR]`) guiding parameter-free execution through common representations.
- **Strict Partition Disjointness**:
    * Every dataset generates four mutually exclusive partitions: `train` (48), `dev` (16), `qualification` (16), and `sealed_test` (48).
    * Zero tuple overlap: $	ext{train} \cap 	ext{dev} = \emptyset$, $	ext{train} \cap 	ext{qual} = \emptyset$, $	ext{train} \cap 	ext{test} = \emptyset$, $	ext{qual} \cap 	ext{test} = \emptyset$.
- **Foundational Architecture & Bounded Replay Memory Accounting**:
    * **SharedKnowledgeTrunk**: 6,176 parameters ($d_{	ext{model}}=128 	o d_{	ext{shared}}=16 	o 128 	o 	ext{vocab}=16$) with LayerNorm and GELU bottleneck, shared across all tasks.
    * **SelectiveResidualAdapter**: 144 parameters per task (ultra-compact Rank-1 modulation), allocated only when non-shared task-specific differentiation is required.
    * **ExemplarReplayBuffer**: 60 total exemplar sequences stored in episodic memory buffer. Exact physical footprint: $60 	imes 4 	imes 4 = 960$ bytes (< 2 KB), strictly accounted for separately from trainable neural parameters.
    * **Capacity Ceiling Compliance**: Retains dozens of skills simultaneously within 20,000 trainable neural parameters without structural module retirement.
- **Execution Provenance**:
    * Full regression suite passed on Google Colab Tesla T4 GPU (`amg-m88-t4` session).
    * Base model parameter invariant preserved (SHA-256: `6e39cd7ef6790b3c`, 1,071,872 weights unchanged); 100% carry (20/20) and logic (8/8) retention.


## Independent architecture review (2026-10-26)

The M104-E/M105-A source audit found truncated entity inputs, silently collapsed transfer labels, and unqualified rollback/controller claims. Historical reports above remain unchanged and require the [audit addendum](ARCHITECTURE_AUDIT.md) when interpreted. See [long-term vision](LONG_TERM_VISION.md), [research integrity](RESEARCH_INTEGRITY.md), [minimal repair plan](REFACTOR_PLAN.md), and [execution report](SCIENTIFIC_REVIEW_REPORT.md). No new milestone or scientific PASS is assigned. Local work is lint/type/syntax/mock-only; real learning and full tests require Kaggle/Colab CUDA.


## Approved audit repair continuation

The approved repair study stores complete rule/question text, annotation evidence, split identities and a canonical SHA256 of all cases. Every delivered token sequence and model-derived context budget is exported remotely. `input_protocol.py` validates the controlled positive-only fixture; it does not extract concepts for the student. Unknown labels, contradictory annotations and padding/context overflow reject explicitly on the approved path. Retained legacy runners still contain cropping paths and are not certified; see [AUDIT_REPAIR_PHASE](AUDIT_REPAIR_PHASE.md).


## Canonical foundation stabilization

The new controlled two-trait protocol is `data/specs/concept_binding_v1.json`, generated remotely by `data/concept_binding.py`. Exact-word-multiset counterfactual pairs swap entity/property binding; complete prompts and explicit absent/negated facts determine targets. `structured_diagnostic` fields never enter language inference. Split/source hashes, actual delivered tokens and all predictions are retained. Cases viewed for diagnosis must be retired from future sealed qualification.


Stage A now isolates positive, negated and absent attributes using the existing persistent student and explicit attribute-only supervision. Protocol, fixed gates, lineage and actual run status: [ATTRIBUTE_LEARNING](ATTRIBUTE_LEARNING.md). No entity-binding, transition, transfer or architecture-growth milestone is advanced.

Stage A result: **FAIL, STOP before entity binding**. Continuing tiny exact88.89%; fresh selective/full train100% but shifted-description exact42.86% versus n-gram100%. Kaggle full regression773passed/1failed (FP16 optimizer-counter qualification), not a clean pass. Exact evidence and unresolved backend diagnosis are in the Stage A report above.


Stage A diagnostic/backend continuation: [protocol and evidence](STAGE_A_DIAGNOSTIC.md). FP16 progress now distinguishes attempted/skipped/applied updates, seeds before initialization, and restores exact tested continuation. Backend source db9ef68 passes778 remote full-suite tests. The historical773pass/1fail result stays preserved; development-only language/history diagnostics do not qualify concepts or advance Stage B.

Final Stage A diagnostic core17d8c6f: **781 remote tests pass**. Continuing student100% on familiar frames with familiar/new names, but21.43% on new frames despite known byte IDs; n-gram100% in all four cells. Original88.9% tiny fitting is rescued by resetting backbone Adam state in a controlled training-only ablation. This does not repair wording generalization or qualify transfer. No checkpoint promotion or Stage B; full evidence is in STAGE_A_DIAGNOSTIC.md.


Stage A.2: **Engineering PASS / Learning FAIL, STOP**. Final experimental source `d25a9bf` passes792remote tests on Kaggle T4. The pinned100/100/21.43/21.43% result reproduces exactly. Matched grammar/order/prefix exposure raises primary new-name/new-frame development accuracy to60% /52% (streams42/43), versus100% for n-gram controls, while acquisition and prior-frame retention fail. Sealed cases remain ungenerated; no capacity change, checkpoint promotion or milestone advance. Full protocol, isolated Adam-history controls, exact revisions and negative artifacts: [STAGE_A2](STAGE_A2.md).

Stage A.2 uses `data/specs/frame_a2_v1.json` and `data/frame_curriculum.py`: sampled training factors are paired across arms; novel names use the same byte alphabet; development grammatical families are disjoint from training and retired historical shifts. Main supervision remains two observed binary-presence labels. Final artifacts include all192000student presentations as two shared factor streams plus renderable arm mappings,7024unique delivered training prompts,1000development token sequences and all logits/predictions. Offline string/marker checking validates8024complete sequences (max167training /148development, configured256), not a neural evaluation. Every ten candidate file hashes and six separately corrected provenance-copy hashes were verified. Compact manifests preserve source/data/checkpoint identities and both ERRORv1 and COMPLETEv2. Development failure leaves sealed examples/labels ungenerated.

Final provenance-only source `cbf2f82` is separately verified: **793remote tests pass**,13focused contracts pass; six metadata-only checkpoint copies retain exact learning state and next-update behavior. Scientific results stay pinned to d25a9bf; Learning FAIL and the sealed-test stop remain unchanged.


Stage A.3 is a bounded tiny-acquisition study using the same continuing student, balanced frames and optional parameter-free paired consistency. Protocol, admission gates and actual status: [STAGE_A3](STAGE_A3.md). No new milestone, neural module, capacity increase or autonomous reset; sealed qualification remains gated on stable acquisition and prior-frame retention. LOCAL static/mock checks only; all neural/full-suite work remotely on Kaggle CUDA.

Stage A.3 completed: **PARTIAL**, engineering808remote passes/16focused contracts at `4e1cc9f`. Balanced BCE reaches100%on all156tiny training targets at1000/1100/1200updates, but old-frame observations fall to66.67%/71.43%. Paired consistency ends100%but dips to92.31%and fails stability. Default-backward numerical divergence is reproduced and an explicit checkpoint-declared deterministic policy repairs the matched no-op invariant and fresh-process continuation. No sealed cases, reset policy, promotion, capacity growth or milestone advance. [Exact evidence and commands](STAGE_A3.md).

A.3 data/specs/stable_attributes_v1.json remains byte-unchanged after the first failed run. Its numerical erratum binds the original spec SHA and exact generated data hash; only runtime numerical policy changes. Fixed support18and augmented support156are reported separately; all target support is development/training data.54saved minimal-pair sequences include identity controls for irrelevant order changes. All210rule/question sequences round-trip, max158/configured256; names J/K are familiar. No sealed generator is used by this runner. Full sampled indices, tokens, logits, vectors and file hashes remain under runs/kaggle-stage-a3/output-v2, with compact tracked manifests preserving ERRORv1, numerical probe and COMPLETEv2.

Stage A.4 completed: **Engineering PASS, scientific PARTIAL; acquisition-with-retention FAIL**. At `fd9d321`, 814 remote regressions and 3 focused contracts pass. Matched replay fails the old-frame/generalization gates; freezing the backbone also fails (new support 77.56%, old 42.86%/42.86%, development 43.59%) despite exact frozen tensors/optimizer states and fresh-process continuation. ECA activation fails initial old competence, so training is skipped. No sealed cases were generated and no candidate was promoted; no growth/reset/milestone advance. [Diagnostics, negative results, curves and commands](STAGE_A4.md).

A.4 replay uses only the nine SHA-pinned historical `Object A` training records; 1,830 normalized protected strings are excluded. These records start at 11.11% parent accuracy and are not certified retained competence. The 0/3/6-of18 replay streams share 2400 presentations of each status pair and 1200 updates, with explicitly different new-example and token exposure. Phase3 uses the identical data/stream hash and moderate replay; complete 399 saved new/replay/development rule/question sequences round-trip (max 158/context 256). The old-frame probes are separate, never replayed. Newly generated development grammar is not sealed data. Raw case predictions and checkpoints are under runs/kaggle-stage-a4-{localization,replay,selective}/output-v1; compact hash/provenance manifests are tracked under data/manifests/a4_*.json.

Stage A.5: **Engineering PASS, learning PARTIAL; continual-learning qualification FAIL**. Source `63d8acf` passes 820 remote tests and 7 focused contracts. Joint old/new training reaches stable 100% for continuing/fresh controls. Corrected synthetic rehearsal ends at new 100%, old 97.62%/98.81% and preserves both old cells above 95% at all late checks, but one new family misses the stability gate and development grammar is 33.33% versus n-gram 100%. Genuine historical replay retains only 42.86%/44.05%. The sampling-invalid v1 and reproducer are preserved. No sealed qualification, added objective/module/capacity, candidate promotion or milestone advance. [Exact gates, evidence, files and commands](STAGE_A5.md).


Stage A.6: Engineering PASS / learning intervention FAIL, STOP. `4ee7690` passes825 remote regressions/9 focused contracts. Frozen inference localizes frame/position sensitivity; the unchanged A5 reference reproduces exactly. Broader matched-budget coverage reaches79.76% training,73.81%/67.86% old behavior and33.33%/30.16% new development grammar, versus100% n-gram/structured controls. No sealed qualification or checkpoint promotion. No neural capacity or production learning-path change. Protocol and execution boundary: [Stage A.6](STAGE_A6.md). A.5 evidence and failed qualification remain unchanged.


Stage A.7: exposure/representation diagnosis is preregistered and locally mock/static checked; remote results pending. Same continuing student, no new modules/capacity or optimizer reset. Original A6 failure remains immutable. [Protocol](STAGE_A7.md).
