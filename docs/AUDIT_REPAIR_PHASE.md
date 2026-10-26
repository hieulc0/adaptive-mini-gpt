# Approved audit repair phase

Baseline: branch `m104-e-and-m105-a`, `a6d8e11eda1ecfb7067f1552b7c8e7eb13d2eb11`, clean tree. This is repair work, not a new milestone. Historical reports and failed runs remain immutable.

Plan before implementation:

1. Repair complete-input boundary failures without increasing model capacity: reject padding overflow, use structured markers and model-derived context budgets, and validate that observable descriptions entail the experiment's labels. Preserve the 256-position model as this experiment's configuration; allow other declared budgets without silently resizing weights.
2. Diagnose M90 with a fixed backbone and fixed data while perturbing ambient RNG; compare historical and explicitly seeded initialization, record every rank/task score and preserve the >=85% gate. Isolate reference/joint/sequential initialization streams and seed the backbone before construction. Do not pick seeds based on scores.
3. Inventory loss/target/optimizer/causal interfaces across source; repair only demonstrated defects and add focused tests. Explicit masks must retain causal independence. Audit optimizer membership and real gradients remotely.
4. Reconcile fixed rotary/router plumbing with remaining expert-growth, sequence-routing and transactional-state limitations. No structural growth or learning module is added.
5. Run one finite complete-input experiment: same existing neural student, frozen/selective/full plasticity, mixed training descriptions and disjoint unseen names/descriptions, majority and simple bag-of-words controls, fixed budgets and thresholds. Record full token sequences, label evidence, gradients, checkpoints, quality and cost independently. Stop after negative generalization rather than adding capacity.

LOCAL-SAFE: reads/edits, syntax/lint/type checks and inspected mock/tiny-fixture tests only. REMOTE-REQUIRED: actual neural loss/gradient/causal tests, M90 diagnosis, training and the full suite on Kaggle T4. Rollback: revert the focused commits; no existing checkpoint is overwritten or automatically promoted. GPU budget: one bounded diagnosis/study job plus full regression; fix only observed failures before any justified rerun.

## Implementation and audit scope

Source revision `b76f4fecf4a8bec67c887cf0920726caac18270a` contains the focused repairs and a bounded remote launcher. The following are implemented, pending the remote results below:

| Finding | Classification | Exact source/function | Correction or remaining contract |
| --- | --- | --- | --- |
| Padding clipped the supplied sequence | CONFIRMED BUG | `tokenizer_m103.ByteTextTokenizerM103.pad_batch` | Reject overflow before tensor allocation; a four-token/three-position reproducer failed before repair. |
| Structured prompt formatting stripped observable whitespace | CONFIRMED BUG | `tokenizer_m103.ByteTextTokenizerM103.encode_prompt` | Encode original fields directly with actual marker IDs; test exact UTF-8 byte boundaries and a declared budget greater than 256. |
| Explicit attention masks could remove causality; ALiBi added boolean masks as numeric bias | CONFIRMED BUG | `model/attention.CausalSelfAttention.forward` | Intersect explicit masks with the causal triangle and apply boolean exclusions correctly; retain default fused `is_causal=True`. Remote tests cover absolute, RoPE and ALiBi. |
| All-ignored next-token labels produced undefined mean loss | CONFIRMED BUG | `model/transformer.MiniGPT.forward` | Raise a clear error; do not turn an unobserved batch into a training success. The existing shifted-target and -1/-100 contracts remain. |
| NUL byte was treated as padding | CONFIRMED BUG | `semantic_bottleneck_m104_e.IntegratedStudentM104_E.pool_representations` | Only dedicated PAD_ID 256 is padding; byte zero remains valid text. |
| Newly unfrozen parameters could be absent from the optimizer | CONFIRMED BUG | `self_modification_m105_a.StructuralRollbackControllerM105_A.apply_and_evaluate` | Reject candidate ownership violations and use existing fixed-topology rollback; no new optimizer migration or structural operation. |
| Direct M104-E/M90 AdamW constructors decayed embeddings and one-dimensional parameters | CONFIRMED BUG | `concept_learner_m104_e`, `capacity_feasibility`; shared `training/optimizer_contract.py` | Explicit embedding-identity and dimensional exclusions; preserve other weights' original decay. Report missing/duplicate/stale parameter registration. |
| M90 seed did not seed neural initialization | SCIENTIFIC VALIDITY ISSUE | `capacity_feasibility.StageACapacityBenchmark.evaluate_capacity_sweep`; M90 runner/test | Isolated named reference/joint/sequential initialization streams; fixed backbone construction; record actual derived seeds. No threshold, rank, epoch or label change to Gate A. |

Complete-input verification is scoped to the approved study and the repaired current path. `input_protocol.ObservedCase` stores researcher-owned annotations, validates explicit evidence and detects identical observations with conflicting labels. Only rules, question tokens and the declared action category reach the student. No simulator tags or output-state labels are inference inputs. Token artifacts record every delivered sequence and the actual model context budget. The model retains its existing configured capacity. A larger tokenizer budget does not resize an existing position embedding; model overflow remains an error.

This positive-only fixture validator is intentionally narrow. It does not prove semantic entailment for arbitrary language or implement general negation handling. Attribute bundles are correlated; unseen phrasing uses familiar attribute words. Do not describe this experiment as unfamiliar vocabulary acquisition or causal-rule transfer.

### Loss, target and optimizer conventions

The reproducible AST inventory (`TRAINING_CONTRACT_INVENTORY.json`, generated by `scripts/inventory_training_contracts.py`) enumerates **all** matching source call sites: 62 `CrossEntropyLoss`, 304 functional `cross_entropy`, 34 BCE-with-logits, 308 AdamW and three SDPA calls at this revision. Enumeration is not semantic certification of 711 independent calls. There is no `nll_loss` or `add_param_group` call in this inventory.

Manual producer/consumer checks distinguish these contracts:

- Production `data/dataset.TokenDataset` and `data/instruction_data` produce already shifted `x[t]`/`y[t+1]` pairs. `MiniGPT.forward` consumes aligned logits and targets directly and normalizes historical -1 to -100. It must never shift a second time. Production trainer/evaluator pass these targets directly.
- `data/pretrain_dataset.PretrainLanguageDataset` also returns already shifted pairs. Its perplexity evaluator consumes aligned logits directly, with explicit padding exclusion.
- `data/language_data` and `data/logic_data` return full sequences with **unshifted** masked labels, despite a misleading “shifted target labels” comment. Their historical M25–M29 runners and `adaptation/trainer.py` perform the one required shift externally. Passing these datasets directly into the production trainer would violate its contract; they are not drop-in interchangeable datasets.
- M103 text QA and language pretraining call `forward_lm`, then shift logits/labels once. QA begins supervision at the first answer byte (prompt length minus one in the shifted array) and excludes padding. The new neural regression checks production numeric loss against an independently calculated aligned CE.
- M89/M90 adapters predict one aligned symbol label per extracted hidden-state position. Their objectives are discriminative symbol transforms, not next-token LM loss; adding a causal shift here would be wrong. The six M90 procedures are tokenwise modular mappings. Held-out numerical sequences do not establish unseen procedures or language understanding.
- M104-E predicts multi-hot tags with BCE-with-logits and states with CE, without causal shifting. The approved experiment fixes the threshold at 0.5, uses identical initial seeds/minibatch streams and zero decay across neural plasticity arms. It logs actual nonzero gradient names; registry membership alone is insufficient evidence of active computation. Frozen lower layers, dormant domain adapters/routers and unused ECA under question-only pooling remain distinguishable from trained paths.

Historical direct AdamW constructors outside the repaired current M104-E and M90 paths remain present. Their default decay, embedding handling, initial seeding, input boundaries and scientific controls are **not globally certified**. Broad replacement would alter many historical experiments without a demonstrated per-path regression and is outside this targeted patch.

### Legacy contracts: fixed versus unresolved

| Contract | Status after repair |
| --- | --- |
| Production router dispatch and MoD auxiliary-loss/telemetry plumbing | Fixed in prior audit commit `e8ab576`; preserve full-suite remote evidence separately from learning claims. |
| RoPE half-split frequency pairing | Fixed in `e8ab576`; prior isolated CUDA regression passed. Explicit mask causality is an additional repair here. This does not qualify context extrapolation. |
| Token-local MoD eval gating and physical token gathering | Existing implementation uses threshold at eval and gathered active positions. Training capacity selection is sequence-relative; it does not provide an identical training/eval routing policy or measured compute advantage by itself. |
| Dynamic sequence-wide layer skipping | Unresolved DESIGN LIMITATION: mean over the entire sequence plus batch-wide `any()` can introduce future/batch dependence. Disabled in the approved study. |
| Expert growth under trained learned top-k routing | Unresolved CONFIRMED DESIGN DEFECT: zero new logits can displace negative old logits and renormalize selected old outputs even when new expert output is zero. No growth is performed or certified here. |
| Dense identity-initialized depth insertion | Narrow existing float32 dense-path invariant only; generic dtype/MoE/topology/checkpoint migration remains unresolved. No new insertion beyond the existing student constructor's baseline occurs in this study. |
| Fixed-topology M105 rollback | Existing tensor/optimizer/scheduler/Torch+Python RNG/replay restoration, plus new ownership rejection. Not complete learning-state recovery. |
| Full topology/resumed-learning rollback | Unresolved: module registrations, optimizer identities, arbitrary trainability/mode flags, NumPy RNG, replay aliasing, replay_ratio, controller state, external memory, disk checkpoint publication and metadata are not all restored. Unsupported structural operators remain rejected. |
| Legacy complete-input processing | Unresolved CONFIRMED BUGS in retained experimental runners: M104-B/C `[:220]`, M104-D `[:max_len]`, old M104-C/D extraction probes, M103-D/E/F left-cropped inference, M103-G `[:256]`, language/logic dataset suffix cropping and M103-C evaluator prefix cropping. These paths are not approved alternative scientific runners. Their historical metrics remain historical. |

The main M104-E standard rule is still too long for the current 256-position configuration. Fail-closed encoding is the intended result, not evidence that the benchmark now trains successfully. The approved study uses a separately declared short rule with unchanged semantics for its much narrower world.

## Reproducible execution

LOCAL: lint/type/syntax/mock-only. REMOTE: Kaggle or Colab GPU training/full tests.

Local commands actually executed (no model forward/backward/checkpoint load):

```bash
.venv/bin/python -m pytest -q tests/test_architecture_audit_contracts.py
.venv/bin/python -m compileall -q src scripts tests kaggle
.venv/bin/python scripts/inventory_training_contracts.py --output docs/TRAINING_CONTRACT_INVENTORY.json
.venv/bin/ruff check scripts/run_audit_repair.py scripts/inventory_training_contracts.py kaggle/launch_audit_repair.py src/adaptive_mini_gpt/training/optimizer_contract.py src/adaptive_mini_gpt/training/reproducibility.py src/adaptive_mini_gpt/natural_language_learning_m103/input_protocol.py tests/test_architecture_audit_contracts.py tests/test_audit_neural_regressions.py src/adaptive_mini_gpt/concept_instruction/capacity_feasibility.py src/adaptive_mini_gpt/adaptive/experts.py
```

Result: **15 mock/tiny-fixture tests passed**; syntax and the listed focused lint passed. The first padding reproducer failed before its repair, as required. A wider lint check on remaining modified legacy files returned seven existing issues (unused M90 imports, legacy import ordering and `assert False`); it is not reported as a repository-wide lint pass. No local full suite or neural evaluation was run.

Type checking initially could not resolve installed Torch/NumPy through the environment's default Pyright search paths. Checking the five new modules/runner files with explicit existing site-packages and `src`/`scripts` extra paths returned **0 errors**, using this local-only temporary config (no suppressed diagnostics):

```bash
python - <<'PYCODE'
import json
from pathlib import Path
root = Path.cwd()
Path('/tmp/amg-repair-pyright.json').write_text(json.dumps({
    'extraPaths': [str(root/'src'), str(root/'scripts'), str(root/'.venv/lib/python3.12/site-packages')],
    'pythonVersion': '3.12', 'typeCheckingMode': 'standard'}))
PYCODE
.venv/bin/pyright --project /tmp/amg-repair-pyright.json src/adaptive_mini_gpt/training/optimizer_contract.py src/adaptive_mini_gpt/training/reproducibility.py src/adaptive_mini_gpt/natural_language_learning_m103/input_protocol.py scripts/run_audit_repair.py kaggle/launch_audit_repair.py
```

The source was committed and pushed before job submission. Job preparation (LOCAL-SAFE):

```bash
python - <<'PYCODE'
import json
from pathlib import Path
sha = 'b76f4fecf4a8bec67c887cf0920726caac18270a'
job = Path('runs/kaggle-audit-repair/job')
job.mkdir(parents=True, exist_ok=True)
(job/'launch.py').write_text(Path('kaggle/launch_audit_repair.py').read_text().replace('__PINNED_SHA__', sha))
(job/'kernel-metadata.json').write_text(json.dumps(dict(
    id='sparklinghydraulics0/amg-approved-audit-repair', title='amg-approved-audit-repair',
    code_file='launch.py', language='python', kernel_type='script', is_private=True,
    enable_gpu=True, enable_tpu=False, enable_internet=True,
    dataset_sources=[], competition_sources=[], kernel_sources=[], model_sources=[]), indent=2))
PYCODE
kaggle kernels push -p runs/kaggle-audit-repair/job
kaggle kernels status sparklinghydraulics0/amg-approved-audit-repair
kaggle kernels output sparklinghydraulics0/amg-approved-audit-repair -p runs/kaggle-audit-repair/output-v1
```

The launcher checks Kaggle+CUDA, checks out the full pinned SHA, installs the project without dependencies, and independently runs new neural reproducers against baseline `a6d8e11` before checking out repaired code. It preserves process logs, JUnit and JSON artifacts. REMOTE-REQUIRED commands inside the job:

```bash
python -m pytest -q tests/test_architecture_audit_contracts.py tests/test_audit_neural_regressions.py --junitxml=/kaggle/working/audit-repair/contracts.xml
python scripts/run_audit_repair.py --output-dir /kaggle/working/audit-repair
python -m pytest -q --junitxml=/kaggle/working/audit-repair/full-tests.xml
```

Finite timeouts: focused contracts 150 seconds, scientific diagnosis/study 900 seconds, full regression 1,200 seconds. Timeouts/failures are preserved and cannot qualify as passes. Independent historical-reproducer failures are expected and must have pytest exit code 1. Baseline and repaired scientific reports remain separate; no historical checkpoint is modified.

## Exact changed files

Commit `b76f4fe` contains the targeted code repairs, tests and first runner. Commits `1f3b6a0` and `c77062d` add and bound the historical M90 reproduction launcher. The final reporting change adds verified results/manifests and corrects misleading legacy growth/rollback comments; it does not change numerical code. No historical result is edited.

- `README.md`
- `data/manifests/audit_repair_v1.json`
- `data/manifests/m90_audit_reproduction_v1.json`
- `data/specs/audit_repair_v1.json`
- `docs/04-experiments.md`
- `docs/05-roadmap.md`
- `docs/06-data-pipeline.md`
- `docs/ARCHITECTURE_AUDIT.md`
- `docs/AUDIT_REPAIR_PHASE.md`
- `docs/REFACTOR_PLAN.md`
- `docs/RESEARCH_INTEGRITY.md`
- `docs/SCIENTIFIC_REVIEW_REPORT.md`
- `docs/TRAINING_CONTRACT_INVENTORY.json`
- `kaggle/launch_audit_repair.py`
- `kaggle/launch_m90_reproduction.py`
- `scripts/inventory_training_contracts.py`
- `scripts/run_audit_repair.py`
- `scripts/run_m90_benchmark.py`
- `src/adaptive_mini_gpt/adaptive/experts.py`
- `src/adaptive_mini_gpt/concept_instruction/benchmark_m90.py`
- `src/adaptive_mini_gpt/concept_instruction/capacity_feasibility.py`
- `src/adaptive_mini_gpt/model/attention.py`
- `src/adaptive_mini_gpt/model/transformer.py`
- `src/adaptive_mini_gpt/natural_language_learning_m103/concept_learner_m104_e.py`
- `src/adaptive_mini_gpt/natural_language_learning_m103/input_protocol.py`
- `src/adaptive_mini_gpt/natural_language_learning_m103/self_modification_m105_a.py`
- `src/adaptive_mini_gpt/natural_language_learning_m103/semantic_bottleneck_m104_e.py`
- `src/adaptive_mini_gpt/natural_language_learning_m103/text_student_m103.py`
- `src/adaptive_mini_gpt/natural_language_learning_m103/tokenizer_m103.py`
- `src/adaptive_mini_gpt/training/optimizer_contract.py`
- `src/adaptive_mini_gpt/training/reproducibility.py`
- `tests/test_architecture_audit_contracts.py`
- `tests/test_audit_neural_regressions.py`
- `tests/test_m90_concept_instruction.py`

## First remote result

Kaggle `sparklinghydraulics0/amg-approved-audit-repair`, version 1, COMPLETE, pinned `b76f4fecf4a8bec67c887cf0920726caac18270a`, Tesla T4 / Torch 2.11.0+cu128 / CUDA 12.8. Full regression: **754 passed, 1 skipped** in 607.17 seconds; focused contracts: **21 passed**. Historical explicit-mask, empty-target and NUL reproducers each failed independently, then passed after repair. Full logs/artifacts: `runs/kaggle-audit-repair/output-v1/audit-repair/`.

The unchanged M90 gate passes with isolated seed streams (rank 32 qualification 100%, test 98.96%), identically under ambient seeds 111 and 222. Historical unseeded evaluators on the fixed backbone score 96.88% and 100%; seeded initialization with legacy decay scores 97.92%. This establishes initialization sensitivity and embedding-decay effects on trajectories; **it has not reproduced the earlier failure on that fixed backbone**. One additional prespecified historical test-order prefix reproduction is prepared in `kaggle/launch_m90_reproduction.py`, with 900-second timeout, process seed 42, captured initial state and same-backbone repaired comparisons. Do not claim exact original RNG recovery: the original failed run did not save its RNG state.

All three neural arms fit the tiny attribute batch and full training set. Combined unseen-name/description attribute F1 and causal accuracy: frozen 100%/ 100%, selective 100%/ 100%, fully trainable 62.5%/ 62.5%; bag-of-words linear 100%/ 100%, majority 66.7%/ 50%. These are narrow positive-only lexical tasks with correlated bundles, not general conceptual qualification. Each neural arm used 400 updates, 4,800 observations and 760,905 nonpadding tokens; measured GPU wall times 4.36/5.22/6.23 seconds. Linear baseline used the same updates/examples and took 0.61 seconds. Frozen/selective description-swap scores fall to 0%, proving sensitivity to changed descriptions on this slice; this does not establish rule understanding. The fully trainable arm fails robust description generalization. No new module, larger model, architecture search or milestone follows.

### Complete-input and gradient evidence

All delivered sequences fit the unchanged actual model budget: **maximum 187 tokens of 256**. Remote encoding asserted lossless reconstruction of both rule and question fields and actual special markers for every case. This is independently inspectable in `token_sequences.json` and `cases.json`; no model weights need be loaded to validate it. The tracked manifest is `data/manifests/audit_repair_v1.json`. Canonical data hash: `319f02b67efe519da41caf33fbda9e09044d57913c52897b65901f7c3ffde450`.

All arms started with identical state hash `b4b586814e27dc7dffbaf3471e098c54730e38651942fed62a6e8a00ed32242c`. Optimizer membership reports missing=[], duplicates=0, stale=0 for every arm. Final training-update nonzero-gradient names: frozen 19 (zero backbone), selective 34 (15 backbone), full 83 (64 backbone). Frozen parameters were asserted gradient-free. Complete names and step 0/1/final traces are in `learning.json`; unused registered ECA/domain-router/adapter paths are not counted as learned contributions. Distinct final state hashes/checkpoint file hashes are retained in the artifacts/manifest. These are clean research model-state files, **not complete optimizer/RNG/replay resumed-learning checkpoints**.

| Arm | Tiny attribute F1 | Train state accuracy | Unseen names F1/state | Unseen descriptions F1/state | Both F1/state | GPU wall seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Frozen backbone + existing bottleneck |100%|100%|100% / 100%|100% / 100%|100% / 100%|4.36|
| Selective backbone + existing bottleneck |100%|100%|100% / 100%|100% / 100%|100% / 100%|5.22|
| Full backbone + existing bottleneck |100%|100%|100% / 100%|50% / 50%|62.5% / 62.5%|6.23|
| Train-vocabulary bag-of-words linear |100%|100%|100% / 100%|100% / 100%|100% / 100%|0.61|
| Majority (no training) |66.7%|50%|66.7% / 50%|66.7% / 50%|66.7% / 50%|not timed|

Times are observed GPU-process wall time including in-script evaluation, not isolated kernel profiling. Neural and linear controls match updates and sampled examples; measured compute differs and is reported, not described as equal. Tiny phase has no state supervision, so its state accuracy is not an acquisition gate. Attribute exact-set match equals F1 for the three neural arms on these slices; majority exact match is 0%. Each tested property has positive and negative examples; 22 of 26 attribute classes have zero positive support. An all-property or rare-class extraction claim would be invalid.

The frozen result matters: a learned head over random frozen features already solves this easy lexical world. Thus selective 100% cannot be attributed to useful backbone language learning or previous experience. No arm has demonstrated learned causal-rule changes, semantic negation, reference resolution, independent attributes, retention, feedback transfer or future-task learning efficiency. The full-backbone generalization loss is observed, but its optimization mechanism is not isolated by this one run. Do not label it a capacity bottleneck or definitive catastrophic forgetting.

The scientific bottleneck has moved from invisible input corruption to **qualification of reusable representations on tasks that simple lexical controls do not saturate**. Stop here rather than add modules or more capacity. The smallest next proposed experiment is a separately preregistered development-only minimal-pair curriculum with independently varied properties, explicit absent/negative attributes, entity references and changed causal rules, followed by untouched held-out descriptions/worlds; retain the current student and strongest shallow controls. This is a proposed qualification experiment; no additional architecture component is justified by the current scores.

This study is not a one-variable causal replication of the preceding audit: complete explicit properties, rule text and mixed training templates changed together. Improvements cannot be assigned solely to one repair. The full-backbone comparison uses the prespecified shared learning rate and budget; it does not establish that a fully trainable backbone is intrinsically worse, or that its best achievable generalization is 62.5%. Viewed descriptions are now diagnostic data and must not be reused as a sealed holdout for hyperparameter selection.

## M90 failure reproduced and diagnosed

Additional Kaggle kernel `sparklinghydraulics0/amg-m90-audit-reproduction`, version 1, **COMPLETE**. Diagnostic launcher source `c77062dd779b8ae75eb0acdf010d624f4eae56a8`; actual historical test/source checkout `e8ab5760dbae452bd48de4478e19ee717c9afb42`. Tesla T4 / Torch 2.11.0+cu128 / CUDA 12.8. One process seed 42 was declared before running the original collection-order prefix. No seed sweep, retries or threshold change occurred. The prefix contains 667 tests and deliberately ends after M90: **665 passed, 1 failed, 1 skipped** in 576.02 seconds. The sole failure is the unchanged original M90 assertion, preserved in `prefix.log`/`prefix.xml`; the diagnostic kernel's COMPLETE status does **not** mean its historical pytest passed.

| Same historical backbone, seed 42 curriculum, 50 epochs | Rank 16 qualification/test EM | Rank 32 qualification/test EM | Rank 32 >=85% gate |
| --- | --- | --- | --- |
| Historical test-order initialization |93.75% /87.85%|77.08% /77.43%|FAIL|
| Repaired named initialization streams, ambient 111 |96.88% /96.88%|94.79% /95.49%|PASS|
| Repaired named initialization streams, ambient 222 |96.88% /96.88%|94.79% /95.49%|PASS|

Historical rank 32 per-task qualification EM: 87.5, 56.25, 75.0, 56.25, 87.5, 100.0%. Modular reference 100%; sequential baseline 16.67%. The model and incoming Torch/Python/NumPy RNG state before the historical evaluator were captured remotely in `historical_m90_initial_state.pt`; the repaired runs reused the **same backbone** and reports are identical across the two ambient states. The artifact's byte hash is in `data/manifests/m90_audit_reproduction_v1.json`. No checkpoint was loaded locally.

**Root cause — CONFIRMED SCIENTIFIC VALIDITY ISSUE:** the old `seed` argument seeded the curriculum, while the backbone and reference/joint/sequential networks drew from ambient Torch RNG. Prior tests and the number/order of networks constructed therefore changed the parameter initialization presented to a fixed-budget nonconvex optimizer. The old gate was an order-sensitive empirical score, not a deterministic test of its declared seed. The reproduced rank 32 model failed while the smaller rank 16 model passed; merely increasing capacity is not a justified repair. Stable initialization of the original backbone in the repaired regression test plus independently named network streams removes that uncontrolled dependency. All ranks, 50 epochs, examples, objectives and the 85% threshold remain unchanged.

Embedding/one-dimensional decay exclusions are also repaired by the invariant. They change trajectories, not just bookkeeping: on the first run's fixed backbone, seeded initialization with legacy decay gave rank 32 qualification 97.92%, versus 100% with the corrected groups. That first experiment isolates this decay contrast; the historical-prefix comparison changes initialization policy and decay together. It does not uniquely apportion the 77.08-to-94.79 gain between the two interventions. An optimization-landscape explanation beyond the confirmed initialization/decay controls remains a hypothesis, not a measured mechanistic proof.

The original failed run did not save RNG state, so the exact original tensor realization cannot be recovered. This **new recorded failure reproduction** preserves the same historical code, order, benchmark settings and failed assertion; it is not asserted to be byte-identical to the earlier unsaved run. No claim is made that every independent initialization now exceeds 85%. Single-seed Gate A reproducibility is repaired and verified; multi-seed scientific robustness, unfamiliar procedures and future learning efficiency remain unqualified.

Reproduce the extra diagnostic with the same metadata preparation as the first job, replacing the template with `kaggle/launch_m90_reproduction.py`, pinned SHA with `c77062dd779b8ae75eb0acdf010d624f4eae56a8`, directory with `runs/kaggle-m90-reproduction/job` and kernel ID/title with `sparklinghydraulics0/amg-m90-audit-reproduction` / `amg-m90-audit-reproduction`:

```bash
kaggle kernels push -p runs/kaggle-m90-reproduction/job
kaggle kernels status sparklinghydraulics0/amg-m90-audit-reproduction
kaggle kernels output sparklinghydraulics0/amg-m90-audit-reproduction -p runs/kaggle-m90-reproduction/output-v1
```

The launcher runs the historical prefix via `pytest.main` with a capture plugin, a 900-second alarm and two CPU threads inside the remote CUDA runtime. It deliberately preserves the failed assertion and excludes all tests after M90. These counts are separate from the repaired full suite's 754 passes and one skip.

## Exit assessment and decision

| Exit criterion | Actual status |
| --- | --- |
| Repaired regression suite | REMOTE VERIFIED: 754 passed, 1 opt-in GPU qualification skipped; focused neural/mock contracts 21 passed. |
| M90 explanation/reproduction | REMOTE VERIFIED: historical 77.08% failure reproduced; same-backbone repaired 94.79% under both ambient states; unchanged gate. |
| Complete input | REMOTE VERIFIED for approved path; independently decoded all 296 exported prompts locally without model execution; max 187/256. Legacy cropping paths remain unqualified. |
| Training/loss/gradient integrity | Current-path numerical CE, causal mask, NUL, ownership and trainability checks pass. Repository call-site inventory is complete; every legacy semantic interface is not certified. |
| Held-out neural learning | Narrow positive-only lexical feasibility demonstrated by frozen/selective arms; both tie a much cheaper shallow baseline. Full-backbone unseen-description robustness fails at this budget. |
| Retention, feedback transfer, learning-to-learn | NOT RUN / NOT QUALIFIED by this repair study. |
| Learned modification/full rollback | NOT QUALIFIED; disabled structural work and recorded unresolved contracts. |

**Decision:** stop at targeted repair and diagnostic results. The current dense Transformer remains a credible **experimental substrate**, with no evidence here that a larger model or added semantic module is necessary. It is **not yet a scientifically qualified persistent language/concept learner**, and no milestone is earned. The next bottleneck is demonstrated reusable learning beyond saturated lexical controls, with valid old-task competence and complete persistent-state protection. Unresolved sequence routing, expert growth, tied-output interference, legacy input/loss/decay paths and partial rollback must not be presented as fixed or enabled on the strength of the regression pass.
