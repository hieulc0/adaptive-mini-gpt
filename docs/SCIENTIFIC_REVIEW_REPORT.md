# Independent scientific review — execution report

Review baseline: `m104-e-and-m105-a`, `c353b1ff4ee5c9d3772635718912be53eb714424`; clean initial tree. No milestone advancement or historical artifact edits.

## Scientific answer

The current system is a credible **research substrate to repair and test**, but it is not yet a scientifically qualified persistent neural learner. Its fundamental immediate failure is an invalid experimental interface: the rule alone is 335 bytes while neural inputs were truncated to 220 bytes, removing the entity description; transfer vocabularies silently collapse new actions/outcomes to class zero. These errors prevent interpreting low extraction or high transfer accuracy as architectural evidence. The smallest testable redesign preserves the Transformer and replaces those interfaces with complete structured input, explicit labels, a continuing learning-state lineage and independent generalization controls.

Reliable conceptual language learning, certified retention, positive future learning efficiency and useful learned self-modification have not been demonstrated. Full transactional rollback is not established by the current snapshot/test. Details, evidence references, limitations and repair priorities are in ARCHITECTURE_AUDIT.md, RESEARCH_INTEGRITY.md and REFACTOR_PLAN.md. LONG_TERM_VISION.md preserves the unrestricted long-term architectural ambition with finite experiment budgets.

## Historical results and reinterpretation

The existing report remains PARTIAL, 6/10 gates, extraction 38.1% micro-F1 versus lexical 87.4%, integrated accuracy 46.5%, novel/paraphrase 57.6%, initial competence 42.5%, transfer fresh/experienced 91.1% and 1.00x. These are recorded historical observations, not new independently replicated results.

| Historical gate | Audit assessment |
| --- | --- |
| G1 parameter reduction | Engineering arithmetic only; active-path accounting still needed |
| G2 lexical information | Descriptions contain useful lexical evidence; neural arm did not receive equivalent evidence |
| G3 extraction strategy | Invalid best-strategy label and no causal suppression test |
| G4 attribute extraction | Original FAIL remains; score also affected by input and calibration contracts |
| G5 concept accuracy | Original FAIL remains; cannot diagnose capacity with missing descriptions |
| G6 plasticity parity | Equal poor predictions do not establish useful representations |
| G7 text-only | No tag argument at inference, but structured action IDs and complete input need explicit accounting |
| G8 competence/retention | Original FAIL remains; copied modular/rollback values are not experiments |
| G9 forward transfer | INVALID for conceptual transfer because of label collapse; positive learning efficiency NOT ESTABLISHED |
| G10 rollback/controller | Complete restoration and learned advantage NOT ESTABLISHED; zero-gain acceptance is insufficient |

No historical acceptance threshold is changed to obtain a pass. Future qualification must distinguish invalid evidence, scientific failure and unrun experiments.

## Implemented corrections

Changed tokenizer and M104-E learner/auditor input handling, M104-B paraphrase payload, M103 exact scoring, MiniGPT optimizer grouping, M105 error/no-op handling and M104 runner remote-runtime enforcement. Added remote guard, inspected mock-only regression file, bounded foundational diagnostic, and pinned Kaggle launcher. README, experiment/roadmap/data guides link to the audit; the training guide now explicitly supersedes old local neural commands. See Git diff/commit for the exact file list.

The M104 benchmark now rejects prompts that exceed context and rejects unknown vocabulary rather than returning misleading metrics. It remains blocked pending the complete-context and label/head protocol in REFACTOR_PLAN. This is deliberate fail-closed behavior, not a neural improvement claim. No structural rewrite, capacity increase or checkpoint overwrite occurred.

## Local verification

LOCAL: lint/type/syntax/mock-only. Reproducing tests caught substring exact match, embedding decay, silent unknown labels, M105 exception-path failure and unsupported structural no-op acceptance before their corrections. `python -m pytest -q tests/test_architecture_audit_contracts.py`: **10 passed**, 1.50 seconds after fixes. Test bodies were inspected: fake methods/parameters and tiny text fixtures only; no neural forward/backward/update/checkpoint benchmark.

New files pass targeted Ruff formatting/lint after import cleanup. Changed legacy files initially showed 11 pre-existing lint findings; final checks and any remaining type limitations are recorded below. No full local pytest, training or neural evaluation was run. Rule-length/vocabulary checks only inspected deterministic string/spec metadata.

## Remote execution

REMOTE: Kaggle or Colab GPU training/full tests. Existing historical kernel status queried: `sparklinghydraulics0/amg-m104-e-grounding-transfer` COMPLETE. This is not a new qualification run. New diagnostic submission and measured results will be appended after the exact reviewed revision is committed/pushed and remotely executed. Until then: foundational learning NOT RUN; full regression NOT RUN; retention, transfer and controller advantage NOT QUALIFIED.

## Verification and source publication update

Reviewed source was committed and pushed as `cdd433a8d43c6b1cdc469205b3fe8e5c8a732e00` on the existing branch before remote submission. Local commit required filesystem escalation because `.git` is read-only in the ordinary sandbox; no approval rejection occurred. The private Kaggle diagnostic was submitted as `sparklinghydraulics0/amg-foundational-audit`, version 1. Job files: `runs/kaggle-foundational-audit/job/`; remote outputs: `/kaggle/working/foundational-audit/`. Submission alone is not verification.

Exact local checks: `.venv/bin/python -m pytest -q tests/test_architecture_audit_contracts.py` (10 passed); `.venv/bin/python -m compileall -q` on changed source/test/runners; `.venv/bin/ruff check` on all changed Python files (pass after bounded import/unused-name cleanup); `.venv/bin/ruff format` on new Python files; `git diff --check` (pass). A targeted Pyright check of `training/remote_runtime.py` passed with 0 errors/warnings after a temporary configuration supplied the correct relative source and installed-package paths. Initial direct Pyright invocations could not resolve Torch; an initial temporary config used an unsupported absolute include and was discarded. No repository-wide type-check pass is claimed.

## Changed files in the implementation commit

- `README.md`
- `docs/03-training-and-colab.md`
- `docs/04-experiments.md`
- `docs/05-roadmap.md`
- `docs/06-data-pipeline.md`
- `docs/ARCHITECTURE_AUDIT.md`
- `docs/LONG_TERM_VISION.md`
- `docs/REFACTOR_PLAN.md`
- `docs/RESEARCH_INTEGRITY.md`
- `docs/SCIENTIFIC_REVIEW_REPORT.md`
- `kaggle/launch_foundational_audit.py`
- `scripts/run_foundational_audit.py`
- `scripts/run_m104_e_benchmark.py`
- `src/adaptive_mini_gpt/model/transformer.py`
- `src/adaptive_mini_gpt/natural_language_learning_m103/causal_world_m104_b.py`
- `src/adaptive_mini_gpt/natural_language_learning_m103/concept_learner_m104_e.py`
- `src/adaptive_mini_gpt/natural_language_learning_m103/grounded_extractor_m104_e.py`
- `src/adaptive_mini_gpt/natural_language_learning_m103/self_modification_m105_a.py`
- `src/adaptive_mini_gpt/natural_language_learning_m103/text_student_m103.py`
- `src/adaptive_mini_gpt/natural_language_learning_m103/tokenizer_m103.py`
- `src/adaptive_mini_gpt/training/remote_runtime.py`
- `tests/test_architecture_audit_contracts.py`

## Kaggle version 1 — completed phases, overall ERROR

Kernel: `sparklinghydraulics0/amg-foundational-audit/1`; pinned source `cdd433a8d43c6b1cdc469205b3fe8e5c8a732e00`. Hardware: Tesla T4; Python 3.13.15; PyTorch 2.11.0+cu128; CUDA 12.8. Downloaded artifacts: `runs/kaggle-foundational-audit/output/foundational-audit/` (report, execution status, fixtures, three checkpoints, logs and test XML); kernel log is one directory above. This ERROR kernel is **not** a qualified completed run. Its individual audit tests and feasibility subprocesses exited zero; full regression exited one.

Audit contracts: 10 passed remotely. Full suite: **718 passed, 25 failed, 1 skipped**, 464.38 seconds. Preserve `full-tests.log` and `full-tests.xml`; no timeout was counted as a pass.

Controlled single-seed result (seed 42; each arm 200 tiny + 200 main updates, 400 actual optimizer steps; LR 0.002, zero decay; 4 tiny / 32 train / 32 dev / 32 held-out examples):

| Arm | Tiny F1 / exact set | Train F1 / state accuracy | Dev F1 | Held-out F1 / exact set | Held-out state accuracy | Mismatched-text F1 | Measured GPU wall time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Selective | 100% / 100% | 100% / 100% | 50% | 50% / 50% | 50% | 50% | 7.089 s |
| Fully trainable | 100% / 100% | 100% / 100% | 50% | 75% / 75% | 50% | 25% | 6.199 s |
| Frozen | 100% / 100% | 100% / 100% | 50% | 50% / 50% | 50% | 50% | 4.147 s |

All arms have the same initial tensor hash `b4b586814e27dc7dffbaf3471e098c54730e38651942fed62a6e8a00ed32242c`; final checkpoint hashes and actual gradient-bearing parameter names are in report.json. Optimizer coverage of trainable registration is true in every arm, which does not imply every registered module participates in question-only inference. Dataset hash: `e3421087f1c7fa0a8a73210bab2990b735d829808f86224148f6325e2065596e`. Infrastructure simulator-query count is zero; labels are static synthetic supervision. Full-batch training observes 800 tiny-case presentations plus 6,400 main-case presentations per arm. GPU wall time is synchronized elapsed training/evaluation time, not device event-only kernel time.

Interpretation: **tiny neural acquisition PASS; reliable unfamiliar-description generalization FAIL/INCONCLUSIVE; causal generalization FAIL** (50% equals the majority state baseline). The fully trainable arm's text ablation drop suggests text dependence, but 32 highly correlated cases from one seed do not qualify broad language understanding or an architecture superiority claim. The material words remain familiar; only names/phrasing change. There is no corrected-versus-historical causal improvement estimate. Corrective feedback, retention, future-task learning efficiency and learned self-modification remain NOT QUALIFIED and are not advanced on these results. No multi-seed sweep was launched.

The full suite uncovered additional production failures: removed generation/inspection/counting methods; `TokenDataset` already shifts next-token labels, while `MiniGPT.forward` shifted again; answer masks use -1 while the model only ignored -100; MoE telemetry names no longer matched router outputs. One failure was a missing installed `amg` console script in the launcher. Two were stale test expectations exposed by the deliberate prompt/paraphrase repair. These are separated from the negative learning result. A follow-up focused repair restores historical interfaces from Git revision `9d1dad4`, restores aligned targets and both mask sentinels, reconnects router telemetry/auxiliary loss, fixes the independently identified rotary frequency pairing, updates the two obsolete payload expectations, and installs the package in the remote runner. New neural tests will be demonstrated against the pre-repair audit SHA and rerun on the corrected SHA remotely.

A tiny local byte-fixture check additionally confirmed that two distinct entity descriptions under the same standard rule/action produce bit-identical old 220-byte neural inputs. No model was loaded or executed for this proof. At deterministic evaluation the old attribute path cannot discriminate those entities; action-conditioned state priors are not learned description understanding.

## Kaggle version 2 — engineering repairs verified, remaining scientific assertion FAIL

Pinned source `e8ab5760dbae452bd48de4478e19ee717c9afb42`; same T4/Python/PyTorch/CUDA environment. Artifacts are preserved separately at `runs/kaggle-foundational-audit/output-v2/foundational-audit/`. Audit tests **12 passed** (the 10 mock contracts plus two real CUDA regressions). Full repository suite completed: **744 passed, 1 failed, 1 skipped**, 560.92 seconds. Kernel overall **ERROR**, because the full-suite exit code is one; no full regression PASS is claimed. All 25 failures observed in version 1 are absent from this rerun.

The remaining failure is `tests/test_m90_concept_instruction.py::test_stage_a_capacity_sweep_and_gate_a`: its unchanged scientific requirement is joint qualification EM >=85% at rank 32. `concept_instruction/capacity_feasibility.py:74–200` seeds curriculum generation only; the test constructs the random backbone without seeding, and adapter/trunk initializations consume ambient Torch RNG. Thus this is an order-dependent stochastic scientific acceptance assertion, not a deterministic software contract. It passed in version 1 and failed in version 2. Do not lower the threshold, pick a favorable seed, or hide this failure. An independently seeded, preregistered capacity requalification is required before citing M90 Gate A. The precise failed EM is not exposed by the current failing assertion; do not invent it.

The foundational diagnostic repeated **identical** final parameter hashes and scores for all arms. Version 2 synchronized GPU wall times were 6.563 / 6.033 / 4.031 seconds (selective / fully trainable / frozen), 400 diagnostic optimizer updates per arm. This is engineering reproducibility of the same narrow negative result, not a new independent-world replication or a generalization improvement.

The initial before/after CUDA baseline check correctly triggered the old mask failure, but its device-side assert poisoned the same subprocess before the rotary test. That second baseline error is not independent rotary defect evidence. The launcher is corrected to use a separate subprocess for each baseline regression, and a short contracts-only remote run will verify those failures independently; it does not rerun or replace the failed M90 acceptance experiment.
