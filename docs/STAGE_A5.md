# Stage A.5 — joint acquisition before preservation algorithms

Final verdict: **Engineering PASS (820 remote tests at `63d8acf`); learning PARTIAL; continual-learning qualification FAIL.** Finite joint training succeeds. Representative synthetic rehearsal preserves measured old behavior above 95% through the three late checkpoints, but new-family stability and unfamiliar wording fail. No preservation objective, sealed qualification, growth or candidate promotion follows.

## Preflight and bounded plan

Branch `m104-e-and-m105-a`, clean at `d51e844`; Stage A.4 executable evidence remains pinned to `fd9d321`. The earlier completed all-`docs/*.md` preflight remains applicable to unchanged historical documents. Current README, AGENTS.md, tiny config, A.4 report/manifests, canonical owner, objective, tokenizer, data generators, fresh-control constructor and Kaggle launchers were rechecked. The active scope is Stage A.5 foundation diagnosis; no milestone number is advanced. Engineering correctness and learning qualification remain separate.

LOCAL: lint/type/syntax/mock-only. REMOTE: Kaggle CUDA data generation, all model execution/training, continuation and full regression. No local neural fallback.

Smallest intervention: **change only the development training distribution**. Keep the dense five-block continuing student, its existing projection/head, selective trainability, two-observed-label BCE, inherited Adam state, LR .002, decay 0, clip 1 and deterministic CUDA policy. Compare one fresh same-topology/selective control initialized with seed 42 and a new Adam optimizer; it is a diagnostic control, never the replacement student. Different inherited weights/moments/history are explicitly confounded in that comparison; it is not a negative-transfer or capacity proof.

## Phase 1 preregistration

- New training names: `Rax`, `Vim`. Old grammatical families: `{name} has properties: {content}.` and `Description of {name}: {content}.` New families: the three A.3 status/entry/record grammars. Familiar property words `flammable` and `conductive`; each independently positive, negated or absent, negators `not`/`without`, both property orders. Identical strings are deduplicated; expected 210 new training descriptions (84 old-grammar/126 new-grammar), verified remotely rather than assumed. No irrelevant prefixes in this initial feasibility test.
- These are **newly generated development TRAINING examples**, not genuine historical replay. Labels come from explicit source annotations and are independently checked against full observable property/negation phrases. Only rules, complete question tokens and constant placeholder action enter the neural student. Status/family/name metadata never enters inference.
- Fail closed on normalized exact-text overlap with any known historical training support, A.2/A.3/A.4 development, validation or retention probes. Previously used grammar is intentionally shared; this is joint acquisition, not template-disjoint generalization. No sealed generator is called.
- Exactly1200 updates ×18 examples =21600 presentations per arm. Each batch has nine matched old/new pairs, one per status pair; pair members share name, content and labels but differ in grammar. No paired-consistency objective. Old/new groups each receive 10800 presentations, each status pair 1200 per group; within old each frame receives 600 per status, within new each receives 400 per status. Seed42 fixes the full stream. Report actual unique exposure, names, status counts, tokens, queries and synchronized wall time. This matches update/example/LR budgets, not token/FLOP/billing equivalence.
- **Fixed primary: continuing student.** Record initial,100-step curves and actual checkpoints at 1000/1100/1200. Joint acquisition passes only if BOTH old and new training distributions, at ALL three specified endpoints, have exact>=99%, every frame>=99%, and every positive/negated/absent stratum>=98%. Require actual planned endpoints (missing curves fail closed). No best-step or best-arm selection. Fresh control receives the same stream, architecture, trainability and update budget; report its gate separately.
- Evaluate the previously viewed old-known/renamed observations (root 100%/100%) and A.4 development grammars throughout; none enters updates or checkpoint selection. The existing old-behavior threshold>=95% in each cell is unchanged and is **separate** from fitting the new old-grammar support. It remains necessary for preservation/qualification. Phase1 admission to a rehearsal feasibility study requires primary joint acquisition; it does not certify retention or language generalization. A parent that fits both new training distributions but misses old observations is not a continual-learning PASS.
- Compare training/old/development predictions with majority, BoW-linear and word 1–3 gram-linear controls. Vocabularies use delivered training text only; same stream/updates/examples/LR, separate parameters and compute. Fresh lexical controls do not have matched historical competence. No oracle attributes enter their inputs.
- On allowed training pairs only, inspect per-label BCE, per-component gradient norms and old/new gradient cosine at initialization and updates 100/600/1200. These inference/backward diagnostics perform no optimizer updates and preserve weights, existing gradients, modes and RNG. Report false negatives, actual logits and per-property recall; do not interpret a negative cosine as sole cause.
- Own stream/cursor/objective/provenance in canonical `LearningState`; preserve inherited weights/moments and source metadata. Verify topology and all live optimizer membership; publish only unpromoted isolated checkpoints. Reload final continuing checkpoint in a fresh process, compare predictions and the next sampled update plus full state signature under matching code/runtime/policy. Original parent bytes and tensors remain immutable.
- Remote bounds: scientific driver 900 s, focused checks 240 s, full suite 1200 s, fresh-process continuation 120 s. Rollback is the immutable historical parent and scoped Git reversion. Record failures, do not promote checkpoints.

## Conditional later phases

If the continuing primary cannot stably fit BOTH supports, **STOP**. Report whether the fresh control succeeds, gradient/label/optimization evidence and remaining uncertainty. Do not run another rehearsal or preservation algorithm.

Only after joint acquisition passes, freeze a separate matched sequential protocol comparing no rehearsal, the nine actual historical examples, newly generated/label-verified old-grammar rehearsal, and balanced old/new experience. Never relabel synthetic rehearsal as historical memory or train on viewed probes. Keep the 95% old-behavior gate and stable acquisition requirements. Only if that remains insufficient may a separately preregistered minimal correct-teacher or parameter-preservation diagnostic run; reference predictions must be on permitted training examples and checked against truth. No automatic reset or new modules.

Independent sealed names/grammars/combinations and additional independent initializations/histories remain conditional on stable acquisition AND preservation with frozen hyperparameters. Current data has only two attributes/all nine status combinations: fitting it is narrow language acquisition, not causal/compositional reasoning. No entity binding, autonomous controller, structural growth, larger capacity or learned-self-improvement claim is authorized by a software pass.

Status at preregistration: IMPLEMENTATION PLANNED; real model results NOT RUN. Historical A.4 artifacts and thresholds are unchanged.

Implementation status before submission: IMPLEMENTED; LOCALLY CHECKED (MOCK ONLY). Three inspected fake/string/scalar tests pass. Ruff check/format, syntax compilation and scoped standard Pyright (explicit project interpreter and src/scripts/tests paths) pass. No real corpus generation or neural forward/backward/training has run locally. New paths: `src/adaptive_mini_gpt/data/joint_acquisition_a5.py`, `scripts/run_joint_acquisition_a5.py`, `kaggle/launch_joint_acquisition_a5.py`, `tests/test_joint_acquisition_a5_mock.py` and the spec/report. No production neural/optimizer/loss/state implementation is changed. The launcher pins source, checks preserved A.4 evidence, runs focused contracts, then the bounded study and full remote suite; it retains errors/timeouts.

## Submission and reproducible commands

REMOTE SUBMITTED: `sparklinghydraulics0/amg-stage-a5-joint-acquisition/1`, exact executable source `a79c74471f812067d45e6079c4e5d3e544ccd53c`. It first passed through QUEUED/RUNNING; its completed result is recorded below. The source was committed/pushed before submission. Pinned launcher and private GPU metadata are under `runs/kaggle-stage-a5-joint/job/`; checkpoint input is the existing `sparklinghydraulics0/amg-stage-a-diagnostic` kernel, with runtime SHA256 enforcement.

LOCAL-SAFE checks actually run:

```bash
UV_CACHE_DIR=/tmp/amg-uv-cache uv run --offline --no-sync ruff check scripts/run_joint_acquisition_a5.py kaggle/launch_joint_acquisition_a5.py src/adaptive_mini_gpt/data/joint_acquisition_a5.py tests/test_joint_acquisition_a5_mock.py
UV_CACHE_DIR=/tmp/amg-uv-cache uv run --offline --no-sync ruff format --check scripts/run_joint_acquisition_a5.py kaggle/launch_joint_acquisition_a5.py src/adaptive_mini_gpt/data/joint_acquisition_a5.py tests/test_joint_acquisition_a5_mock.py
UV_CACHE_DIR=/tmp/amg-uv-cache uv run --offline --no-sync python -m compileall -q scripts/run_joint_acquisition_a5.py kaggle/launch_joint_acquisition_a5.py src/adaptive_mini_gpt/data/joint_acquisition_a5.py tests/test_joint_acquisition_a5_mock.py
UV_CACHE_DIR=/tmp/amg-uv-cache uv run --offline --no-sync pytest -q tests/test_joint_acquisition_a5_mock.py
```

Scoped Pyright uses the same four files, `.venv/bin/python`, standard mode/Python 3.12 and import paths `src`, `scripts`, `tests`; zero errors/warnings after explicit two-element tuple annotations and typed mock dictionaries. No typing diagnostic is suppressed.

Remote submission/status/retrieval (CLI local, execution remote):

```bash
kaggle kernels push -p runs/kaggle-stage-a5-joint/job
kaggle kernels status sparklinghydraulics0/amg-stage-a5-joint-acquisition
kaggle kernels output sparklinghydraulics0/amg-stage-a5-joint-acquisition -p runs/kaggle-stage-a5-joint/output-v1
```

The launcher runs these REMOTE-REQUIRED commands after pinned checkout and CUDA preflight:

```bash
python -m pytest -q tests/test_joint_acquisition_a5_mock.py tests/test_backend_update_remote.py --junitxml=/kaggle/working/a5-joint/contracts.xml
python scripts/run_joint_acquisition_a5.py --root <immutable-input>/inherited_weights_inherited_moments.pt --output-dir /kaggle/working/a5-joint
python -m pytest -q --junitxml=/kaggle/working/a5-joint/full-suite.xml
```

Environment: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `AMG_RUN_GPU_TESTS=1`, `PYTHONPATH=<pinned-repo>/src`, two OMP/MKL threads. Strict algorithms/cuDNN benchmarkFalse are explicit inside the scientific process and resumed child. The full regression status must be taken from the completed XML at that exact source; later report-only commits are not tested executable revisions.

## Phase 1 completed — joint fitting PASS, preservation/generalization FAIL

Kaggle `sparklinghydraulics0/amg-stage-a5-joint-acquisition/1` COMPLETE at `a79c74471f812067d45e6079c4e5d3e544ccd53c`: 817 full regressions and 4 focused contracts pass, zero failures/errors/skips. Runtime: Tesla T4 / Python 3.13.15 / PyTorch 2.11.0+cu128 / CUDA 12.8. Fresh-process continuing-checkpoint predictions and the next update/state signature match exactly; the parent and all six unpromoted endpoint files are preserved. The full suite takes 646.074 s; focused 8.238 s. Each neural arm uses 21600 presentations,10800 per distribution,2621072 nonpadding tokens and 1200 updates. Measured synchronized study wall: continuing 35.095 s, fresh 34.605 s, including diagnostics/evaluation/hashing/checkpointing; not kernel-only or billed time.

| Arm | Old-grammar training | New-grammar training | Previous old known / renamed | A.4 development grammar |
| --- | --- | --- | --- | --- |
| Continuing |100%|100%|45.24% /47.62%|30.77%|
| Fresh diagnostic |100%|100%|59.52% /63.10%|56.41%|
| Majority |42.86%|42.86%|42.86% /42.86%|38.46%|
| BoW linear |61.90%|61.90%|61.90% /61.90%|58.97%|
| Word1–3 gram linear |100%|100%|100% /100%|100%|

Both neural arms have 100% old AND new training exact at 1000/1100/1200, every frame/status 100%; the fixed primary passes the joint-acquisition prerequisite. The continuing old observations begin 100%/100%, then deteriorate even under concurrent grammar training. Thus the same shared neural model CAN fit these finite 210 descriptions without extra capacity, but old-frame knowledge preservation still fails. The fresh control has no initial old competence and its final old observations are generalization scores, not retention. A.4 versus A.5 differs in data, names, negation coverage and exposure; this is not a matched causal comparison proving a sequential-only failure.

Continuing training-pair gradient cosine at initialization is positive for embeddings 0.4444, top Transformer0.6188, projection 0.3221 and attribute head 0.6017. All recorded continuing cosines remain positive at 100/600/1200; state signatures remain unchanged by these diagnostics. A negative old/new cosine is therefore not established as the explanation on these sampled TRAINING pairs. This does not measure gradients on the forbidden old probes or rule out interference elsewhere. At1200 the two old-known recalls are 7.14%/28.57% (26/20 false negatives of28 positives each); new development recalls 43.33%/36.67%. Full logits, confusion counts and actual predictions remain in raw `results.json`.

456 saved rule/question token sequences round-trip completely (max 158/context 256); all 210 training labels independently verify, all 210 examples actually occur. One architecture has 1,255,636 stored/406,762 declared trainable parameters,341,626 parameters with first-update gradients; inactive registered paths are not acquired skills. Each arm records 5928 evaluation rows/208 chunked forwards, plus the separately logged four two-forward training-gradient probes and continuation test. Source indices and exposure audit are preserved. [Compact evidence](../data/manifests/a5_joint_v1.json), [26 measured curve rows](figures/a5_joint_curves.csv), raw `runs/kaggle-stage-a5-joint/output-v1/a5-joint/`.

## Phase 2 preregistration — after the joint-acquisition gate passed

The phase 1 model demonstrates finite joint fit, permitting the authorized rehearsal coverage experiment. Start every arm from the SAME immutable pre-A.3 continuing parent and inherited Adam, never from an already jointly trained candidate. No reset, teacher objective or architectural change. New learning support is the 126 new-grammar phase 1 TRAINING descriptions (Rax/Vim); these are permitted training data, never relabeled as holdouts.

Compare four streams: (1) new-only; (2)6of18 actual historical `Object A` examples; (3)**fixed primary**,6of18 new synthetic old-grammar rehearsal; (4)9of18 synthetic old-grammar balanced mixture. Every arm has 1200 updates/21600 presentations/LR.002/clip 1/decay 0/seed 42. New exposure is21600/14400/14400/10800; old experience 0/7200/7200/10800. Each batch preserves exactly two examples of every status pair; replaced slots have the identical annotation pair. Shared unreplaced new slots are identical across arms. Matched observations/updates do not mean equal tokens or compute.

Synthetic rehearsal uses old grammars on independently chosen single-letter names `N`,`P`,`Q`,`R`, familiar not/without and both property orders (expected 168 distinct descriptions). This intentionally covers the old one-byte-name layout that the longer Rax/Vim-only joint support did not cover; it is development data design, not an independent test. Independently verify all labels and exact string disjointness from all prior probes/validation/development and both A.5 supports before updating. Do not use J/K/Y/Z old observations as training. Synthetic labels are simulator annotations only; student input remains complete text/action 0. Report the teacher-free parent's initial accuracy on this new synthetic support; do not call its fitting retention if it was not initially mastered. Genuine historical 9 retains its immutable file/hash and distinct source status.

The primary must have stable new-support acquisition at 1000/1100/1200 with the unchanged 99% exact/every-family and 98% status requirements. Preserve the 95% old-known AND old-renamed gate; require it at ALL three late checkpoints to call preservation stable. A.4 development exact 90%/each-family 85%/each-status 95% remains necessary before independent qualification, with final endpoint fixed and complete curves reported. No best-arm/step selection. Save actual predictions, loss/gradients, exposures, tokens, active/stored parameters, optimizer ages and all candidate hashes. Replicate primary fresh-process next-update continuation. Lexical controls use each actual stream and training-only vocabulary; budgets, history differences and compute remain explicit.

If no rehearsal arm achieves stable acquisition and old preservation, only the separately authorized ONE minimal correct-reference preservation objective may be considered, with a new frozen protocol first. If preservation succeeds but unseen development wording fails, stop on the language-generalization bottleneck; do not add a preservation objective to force a grammar score. Sealed qualification remains ungenerated until all prerequisites pass; all candidates stay unpromoted. Phase2 remote scientific budget 900 s, full suite 1200 s, focused 240 s and child 120 s. No phase 2 outcome has been observed at preregistration.

Phase2 implementation is local-static/mock checked: five tiny schedule/scalar/string tests pass across the joint/rehearsal files; Ruff/format/syntax and scoped Pyright pass. Genuine replay is runtime-verified through the unchanged A.4 audit; synthetic support is independently label checked and protected-string checked. Same new slots/status composition are verified across four streams. No real phase 2 data/model execution occurs locally. Engineering phase 1's two known warnings (legacy docstring escape and intentional optimizer instrumentation) remain in the preserved full-suite log. Phase2 is not a scientific result until its remote artifact completes.

## Phase2 sampling erratum — detected before inspecting neural outcomes

Source `69d8fc9c2883e57004529743d015d547b9a4c0a2` was submitted as `sparklinghydraulics0/amg-stage-a5-rehearsal/1`. Subsequent static inspection found that replacement selection (`j%3!=step%3`) and new grammatical frame selection shared the same 3-step cycle. The genuine/synthetic 6 arms had identical new slots, but conditional new-frame/status exposure was uneven, confounding broader arm comparisons. A new LOCAL fake-schedule regression failed before correction: nine steps yielded counts{3,6}where each cell should receive 4. This is a confirmed experimental sampling defect, not a demonstrated neural implementation bug. No outcome metrics were downloaded or inspected to choose this correction.

The repaired rotation is `(step//3 + step%3 + status_index)%3`, with the second new frame offset by 1. The nine-step mock now balances every new frame/status exactly; at 1200 steps, each status's new-frame counts must differ by at most 1or the remote preflight aborts. No model, seed, optimizer, loss, threshold, example/update/rehearsal budget changes. Six local mock cases, scoped Ruff/Pyright/syntax pass. Original spec `rehearsal_a5_v1.json` and source remain immutable; version 2 spec binds the original spec hash and erratum.

The already-running v1 is kept bounded and allowed to finish so all outputs and any failures can be preserved; it is NOT QUALIFIED regardless of its eventual score. The corrected run will use the separate `amg-stage-a5-rehearsal-balanced` kernel and `runs/kaggle-stage-a5-rehearsal/output-v2/`, preserving v1 independently. [Sampling erratum](../data/manifests/a5_sampling_erratum_v1.json). No checkpoint is promoted.

Corrected phase 2 REMOTE SUBMITTED: `sparklinghydraulics0/amg-stage-a5-rehearsal-balanced/1`, source `63d8acf27b0c2e373d119c75bce2f9f846cd1409`, spec `rehearsal-a5-v2`. The preceding submitted source `69d8fc9` remains an invalid-sampling study regardless of its score. The correction was committed and pushed before any v1 result inspection; both jobs have finite declared limits. No seed/model/threshold tuning occurred.

Reproduce the corrected run by rendering `kaggle/launch_rehearsal_a5.py` at that source into the job directory, pinning its SHA and using private CUDA/internet metadata with the immutable diagnostic checkpoint kernel input. CLI commands (local orchestration, remote execution):

```bash
kaggle kernels push -p runs/kaggle-stage-a5-rehearsal-balanced/job
kaggle kernels status sparklinghydraulics0/amg-stage-a5-rehearsal-balanced
kaggle kernels output sparklinghydraulics0/amg-stage-a5-rehearsal-balanced -p runs/kaggle-stage-a5-rehearsal/output-v2
```

The remote launcher runs six schedule/string/scalar contracts plus the backend contract, the bounded `scripts/run_rehearsal_a5.py --root <immutable-input>/inherited_weights_inherited_moments.pt --output-dir /kaggle/working/a5-rehearsal`, and `python -m pytest -q --junitxml=/kaggle/working/a5-rehearsal/full-suite.xml`. Numerical environment/preflight matches phase 1. Full tests and all actual neural execution remain remote. Local checks are only Ruff/format, scoped standard Pyright, compileall and `pytest -q tests/test_rehearsal_a5_mock.py tests/test_joint_acquisition_a5_mock.py` (six inspected mock cases). Original source `69d8fc9`'s remote test count necessarily predates the reproducing case; a pass there cannot repair sampling validity.

## Archived phase 2 v1 — invalid sampling, not a primary result

The original kernel completed at `69d8fc9`; 819 full tests/6 focused contracts pass (full 681.510 s, focused 9.181 s), zero failures/errors/skips. All633 complete rule/question sequences and four candidate hashes verify. Saved-data analysis confirms per-status new-frame count range 400in both 6/18 arms, versus 0in new-only/balanced 9. Therefore this engineering pass does not validate the comparison's sampling assumptions. The correction was already committed/submitted before these metrics were inspected.

| INVALID v1 arm | New training endpoint | Old known / renamed endpoint | Development endpoint |
| --- | --- | --- | --- |
| None |100%|42.86% /42.86%|41.03%|
| Genuine historical 6 |100%|42.86% /42.86%|46.15%|
| Synthetic6 |98.41%|100% /100%|38.46%|
| Balanced9 |91.27%|90.48% /97.62%|51.28%|

These are archived endpoints, never a selected successful step. No arm passes the raw stable old-observation gate; the synthetic endpoint 100% does not establish stability. Raw admission is false and sampling validity separately fails. The parent's initial synthetic-support accuracy is100%, historical 9 accuracy 11.11%, prior 210 study's old-grammar support 46.43%, old probes 100%/100%. All four v1 synchronized student walls total 142.83 s; lexical and test costs are separately preserved. Original raw artifacts `runs/kaggle-stage-a5-rehearsal/output-v1/a5-rehearsal/`; [manifest](../data/manifests/a5_rehearsal_v1.json), [all 52 curve rows](figures/a5_rehearsal_v1_curves.csv). No v1 checkpoint is promoted or used as the parent of the corrected study.

## Corrected phase 2 result — coverage helps, full acquisition/generalization gate fails

Kaggle `sparklinghydraulics0/amg-stage-a5-rehearsal-balanced/1` COMPLETE at **`63d8acf27b0c2e373d119c75bce2f9f846cd1409`**. Engineering PASS: **820 full regressions,7 focused contracts,0 failures/errors/skips**. Full-suite 664.337 s; focused 12.396 s. Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8. The two preserved warnings are the existing `reptile.py` docstring escape and intentional optimizer instrumentation in a scheduler diagnostic test; no new warning or backend counter failure is hidden.

Saved-data checks verify all 633 complete token sequences, max 158/context 256, all four candidate file hashes and the data hash. Conditional per-status new-frame count range is0 for none/balanced 9 and 1 for historical 6/synthetic 6, satisfying the corrected preflight. The original range 400 failure remains separately archived. All arms start with identical model and Adam hashes, remain selective/question-only, and use the same inherited 1400-update parent without resets.

| Corrected arm | New training endpoint | Old known / renamed endpoint | A.4 development | Stable new / stable old / development gates |
| --- | --- | --- | --- | --- |
| No rehearsal |100%|42.86% /42.86%|38.46%|FAIL /FAIL /FAIL|
| Genuine historical 6/18 |100%|42.86% /44.05%|41.03%|PASS /FAIL /FAIL|
| Synthetic old grammar 6/18 (fixed primary) |100%|97.62% /98.81%|33.33%|FAIL /PASS /FAIL|
| Balanced synthetic 9/18 |84.13%|95.24% /94.05%|30.77%|FAIL /FAIL /FAIL|

The primary demonstrates **development-only preservation of the two previously competent old cells**: both remain>=95% at 1000/1100/1200. The raw field `primary_preservation_admission` and raw verdict `REHEARSAL_PRESERVATION_FAIL` combine new-acquisition stability with old preservation, and are FALSE/failure because new stability fails; they must not be read as a failure of the separate `stable_old_observations=True` gate. Original JSON and booleans remain unchanged. No overall PASS is assigned.

| Primary update | New exact | Worst new family | Old known | Old renamed | Development |
| --- | --- | --- | --- | --- | --- |
|1000|100%|100%|100%|100%|30.77%|
|1100|99.21%|97.62%|100%|100%|30.77%|
|1200|100%|100%|97.62%|98.81%|33.33%|

At 1100 there is one conductive false positive in the negated stratum: aggregate 125/126is>=99%, but frame 3is41/42=97.62%, below the unchanged 99% per-family requirement. That single error is a real gate failure, not rounded into a PASS. The transient case's identity/logits and intermediate phase 2 weights were not retained; only its aggregate family/per-property curve is available. Final per-case predictions/logits and all phase 1 endpoint checkpoints are retained. The fixed 1200 endpoint is used; neither the attractive 1000 step nor another arm is selected for promotion.

Compared with genuine historical 6, synthetic 6 uses the SAME7200 old presentations,14400 new presentations,2400 presentations per status pair and identical unreplaced new cases. Old endpoints improve by 54.76 percentage points in each cell. Genuine replay fits its nine examples to 100% but fails the old probes. Synthetic rehearsal starts at 100% parent competence on its own 168 examples and ends 98.21%, while preserving old probes within the 95% gate. This is positive coverage evidence in one history/stream, not proof that replay generally fails, a learned memory policy, or a unique effect of name length. The curricula differ in grammar, distinct-example count, names, layout and negation coverage; those factors are not separately isolated. More rehearsal is not automatically better: balanced 9 fails both the stable acquisition and old-observation requirements.

### Per-property behavior and predictions

Primary old-known flammable accuracy/recall are 100%/100%; conductive accuracy 97.62%, precision 100%, recall 92.86% (TP 26/TN 56/FP 0/FN 2). Old-renamed flammable remains 100%; conductive accuracy 98.81%, precision 100%, recall 96.43% (TP 27/TN 56/FP 0/FN 1). Both old groups retain 100% negated/absent-status accuracy; positive-status accuracy is96.43%/98.21%. These are observed familiar-grammar cells, not an independent-world retention certificate.

All three final old errors share the form `J/K/Z has properties: not flammable; conductive.` Truth0,1 becomes 0,0; J logits are-12.439,-3.445. This suggests a remaining negation/property-scope interaction but does not identify its internal cause. No viewed probe is replayed to repair it.

Primary development has 26/78 exact predictions. Flammable precision 70%, recall 46.67% (TP 14/TN 42/FP 6/FN 16), accuracy 71.79%; conductive precision 38.46%, recall 33.33% (TP 10/TN 32/FP 16/FN 20), accuracy 53.85%. Positive/negated/absent-status accuracy is40%/73.33%/83.33%; each development family 38.46%/23.08%/38.46%. Example `A note on J lists flammable; conductive.`: truth 1,1, prediction 1,0, logits 1.095,-15.342. Reversing property order still predicts 1,0 with logits 3.715,-14.568. All final cases/truths/predictions/logits are preserved in raw artifacts, including successes and failures.

The primary's BoW control (50 parameters) scores 76.92% on development, and word 1–3 gram control (520 parameters) scores 100% on every evaluated group. All eight n-gram arms score 100% development. These are learned text controls with training-only vocabularies and matching delivered text streams; they lack the continuing student's inherited history. Their scores are not matched-history retention or transfer estimates. Neural training fit or preserved old cells does not yet outperform these lexical methods on unfamiliar descriptions.

### Resources and persistence

Each student has 1,255,636 stored/406,762 declared trainable parameters;341,626 parameters have first-step gradients (tensor-level count, not FLOPs). The existing inactive ECA/LM-adapter/transition paths remain registered; no module is added or acquired by implication. All arms perform 1200 updates,21600 presentations; no capacity growth, auto-reset or scheduler change.

| Corrected arm | New / old presentations | Nonpadding tokens | Synchronized arm wall |
| --- | --- | --- | --- |
|None|21600/0|2707096|35.397 s|
|Historical6|14400/7200|2567904|35.130 s|
|Synthetic6|14400/7200|2635096|35.156 s|
|Balanced9|10800/10800|2599096|35.021 s|

Student-arm measured wall totals 140.704 s; eight linear controls 15.400 s, separately from full tests. These synchronized loop timings include evaluation, hashing and saving, not pure CUDA kernel or billed allocation time. The initial shared evaluation has 633 rows; each student records 7596 evaluation rows/276 chunked forwards during learning. Resume checking adds its own evaluations and two disposable updates. Token/compute differences are explicit; identical update/example budgets are not identical compute. Initial generation/queue/allocation time is not inferred from these figures.

Lifetime owner steps advance 1400to2600. Active Adam ages reach 2600; dormant states remain 200. All live parameters stay registered. Primary candidate SHA256: `a051d6db1a0969698d946fefd5a07fa6ab33ad3696781cda9308b186245ee5dc`. Its predictions and next sampled loss/update/full-state signature reproduce exactly in a fresh process under the same source/runtime/device policy; the two continuation updates are disposable and do not alter the saved checkpoint. Parent bytes/tensors remain exact. Future architecture/trainability eligibility is unchanged; selective freezing is this experiment's control, not a permanent ceiling.

[Corrected manifest](../data/manifests/a5_rehearsal_v2.json), [all 52 corrected curve rows](figures/a5_rehearsal_v2_curves.csv); raw `runs/kaggle-stage-a5-rehearsal/output-v2/a5-rehearsal/`. The full regression at `63d8acf` is the final executable-code certification; subsequent evidence-only commits must not be relabeled as remotely tested source.

## Required scientific answers and stop decision

1. **Can the student represent old and new knowledge simultaneously?** Yes on the finite 210-example joint TRAINING support: both continuing and fresh models stably reach 100% on both grammatical distributions using one shared architecture/head. This is bounded acquisition evidence, not sealed concept generalization or a general capacity proof.
2. **Is the failure specifically sequential interference?** Not isolated as a sole cause. Even joint training degrades previous descriptions outside its support. The matched phase 2 source comparison shows rehearsal coverage changes preservation materially; basic finite joint fitting works. Which representation/readout/optimization mechanisms cause remaining wording sensitivity and new-family instability is unresolved. Positive train-pair gradient alignment does not rule out other interference. Fresh/history differences cannot establish negative transfer.
3. **Does representative rehearsal help more than the nine historical examples?** Yes for these old observations under matched 6/18 streams/budgets:97.62%/98.81% versus 42.86%/44.05%, and only synthetic 6 passes all late old gates. The support differences are bundled; no autonomous memory-selection advantage or independently replicated policy is claimed.
4. **Can the same continuing student improve new learning while retaining competent behavior?** At the fixed endpoint, yes in this development trial: new 100%, old 97.62%/98.81%, without new parameters or replacing the student. The full stable-learning qualification is still FAIL because one late new-family measurement misses 99%. All candidates remain unpromoted.
5. **Is independent generalization strong enough for entity binding?** No. Viewed development grammar 33.33% versus n-gram 100%; no independent sealed qualification or independent-history replication has occurred. Do not advance.

**Engineering PASS; learning PARTIAL; continual-learning qualification FAIL.** Phase3 preservation/distillation is **NOT RUN**: synthetic rehearsal already meets the old-preservation gate; adding another preservation objective would not address the demonstrated unfamiliar-language failure, and the preregistered stop condition applies. Phase4 sealed qualification/extra seeds is **NOT RUN** because stable new acquisition and development language prerequisites fail. No entity binding, causal reasoning, learned practice, learning-to-learn, mutation, checkpoint promotion or milestone advance.

The next justified problem is stable frame/negation-scope acquisition and generalization within the existing representations/readout, retaining the proven old-behavior measurements and representative rehearsal as controls. Do not train longer, relax the single-error gate, replay viewed probes, choose a favorable seed, add a semantic module or increase size merely to achieve a PASS. One history and one sampling stream cannot certify reusable continual learning; other 24 attribute outputs, LM generation, causal heads and unrelated old skills remain unqualified. Historical reports/checkpoints and the invalid v1 are preserved.

## Exact changed files and revision boundary

Relative to the clean A.4 parent `d51e844`, changed paths are:

- `README.md`
- `data/manifests/a5_joint_v1.json`
- `data/manifests/a5_rehearsal_v1.json`
- `data/manifests/a5_rehearsal_v2.json`
- `data/manifests/a5_sampling_erratum_v1.json`
- `data/specs/joint_acquisition_a5_v1.json`
- `data/specs/rehearsal_a5_v1.json`
- `data/specs/rehearsal_a5_v2.json`
- `docs/04-experiments.md`
- `docs/05-roadmap.md`
- `docs/06-data-pipeline.md`
- `docs/CANONICAL_LEARNING.md`
- `docs/FOUNDATION_DEFECT_REGISTER.md`
- `docs/STAGE_A5.md`
- `docs/figures/a5_joint_curves.csv`
- `docs/figures/a5_rehearsal_v1_curves.csv`
- `docs/figures/a5_rehearsal_v2_curves.csv`
- `kaggle/launch_joint_acquisition_a5.py`
- `kaggle/launch_rehearsal_a5.py`
- `scripts/run_joint_acquisition_a5.py`
- `scripts/run_rehearsal_a5.py`
- `src/adaptive_mini_gpt/data/joint_acquisition_a5.py`
- `src/adaptive_mini_gpt/data/rehearsal_a5.py`
- `tests/test_joint_acquisition_a5_mock.py`
- `tests/test_rehearsal_a5_mock.py`

Executable/evidence sequence:

- `a79c74471f812067d45e6079c4e5d3e544ccd53c`: preregistered joint study,817 remote regressions/4 focused contracts; finite joint fitting PASS.
- `69d8fc9c2883e57004529743d015d547b9a4c0a2`: first rehearsal source,819 remote regressions/6 focused contracts; INVALID SAMPLING, all outputs retained.
- `63d8acf27b0c2e373d119c75bce2f9f846cd1409`: reproduced/repaired schedule,820 remote regressions/7 focused contracts; corrected scientific result above.
- Subsequent commit(s) contain only reports/manifests/saved-data curves; no executable neural or experimental behavior changes after `63d8acf`. Remote results retain their exact source attribution.

No production Transformer, tokenizer, head, loss, optimizer, canonical learning-state implementation or historical report/fixture is modified. All fourteen A.5 candidate checkpoints (six joint, four original rehearsal, four corrected rehearsal) remain in ignored runs paths with tracked SHA256 catalogs; the historical parent and prior failed artifacts remain intact. Candidate publication is never production promotion.

LOCAL: lint/type/syntax/mock-only. REMOTE: Kaggle CUDA training/evaluation/continuation/full tests. No local neural execution or CPU fallback. All three kernels are COMPLETE; no result is inferred from a queued job.
