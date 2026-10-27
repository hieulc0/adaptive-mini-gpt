# Stage A.5 — joint acquisition before preservation algorithms

## Preflight and bounded plan

Branch `m104-e-and-m105-a`, clean at `d51e844`; Stage A.4 executable evidence remains pinned to `fd9d321`. The earlier completed all-`docs/*.md` preflight remains applicable to unchanged historical documents. Current README, AGENTS.md, tiny config, A.4 report/manifests, canonical owner, objective, tokenizer, data generators, fresh-control constructor and Kaggle launchers were rechecked. The active scope is Stage A.5 foundation diagnosis; no milestone number is advanced. Engineering correctness and learning qualification remain separate.

LOCAL: lint/type/syntax/mock-only. REMOTE: Kaggle CUDA data generation, all model execution/training, continuation and full regression. No local neural fallback.

Smallest intervention: **change only the development training distribution**. Keep the dense five-block continuing student, its existing projection/head, selective trainability, two-observed-label BCE, inherited Adam state, LR .002, decay0, clip1 and deterministic CUDA policy. Compare one fresh same-topology/selective control initialized with seed42 and a new Adam optimizer; it is a diagnostic control, never the replacement student. Different inherited weights/moments/history are explicitly confounded in that comparison; it is not a negative-transfer or capacity proof.

## Phase 1 preregistration

- New training names: `Rax`, `Vim`. Old grammatical families: `{name} has properties: {content}.` and `Description of {name}: {content}.` New families: the three A.3 status/entry/record grammars. Familiar property words `flammable` and `conductive`; each independently positive, negated or absent, negators `not`/`without`, both property orders. Identical strings are deduplicated; expected210new training descriptions (84old-grammar/126new-grammar), verified remotely rather than assumed. No irrelevant prefixes in this initial feasibility test.
- These are **newly generated development TRAINING examples**, not genuine historical replay. Labels come from explicit source annotations and are independently checked against full observable property/negation phrases. Only rules, complete question tokens and constant placeholder action enter the neural student. Status/family/name metadata never enters inference.
- Fail closed on normalized exact-text overlap with any known historical training support, A.2/A.3/A.4 development, validation or retention probes. Previously used grammar is intentionally shared; this is joint acquisition, not template-disjoint generalization. No sealed generator is called.
- Exactly1200updates ×18examples =21600presentations per arm. Each batch has nine matched old/new pairs, one per status pair; pair members share name, content and labels but differ in grammar. No paired-consistency objective. Old/new groups each receive10800presentations, each status pair1200per group; within old each frame receives600per status, within new each receives400per status. Seed42 fixes the full stream. Report actual unique exposure, names, status counts, tokens, queries and synchronized wall time. This matches update/example/LR budgets, not token/FLOP/billing equivalence.
- **Fixed primary: continuing student.** Record initial,100-step curves and actual checkpoints at1000/1100/1200. Joint acquisition passes only if BOTH old and new training distributions, at ALL three specified endpoints, have exact>=99%, every frame>=99%, and every positive/negated/absent stratum>=98%. Require actual planned endpoints (missing curves fail closed). No best-step or best-arm selection. Fresh control receives the same stream, architecture, trainability and update budget; report its gate separately.
- Evaluate the previously viewed old-known/renamed observations (root100%/100%) and A.4 development grammars throughout; none enters updates or checkpoint selection. The existing old-behavior threshold>=95%in each cell is unchanged and is **separate** from fitting the new old-grammar support. It remains necessary for preservation/qualification. Phase1 admission to a rehearsal feasibility study requires primary joint acquisition; it does not certify retention or language generalization. A parent that fits both new training distributions but misses old observations is not a continual-learning PASS.
- Compare training/old/development predictions with majority, BoW-linear and word1–3gram-linear controls. Vocabularies use delivered training text only; same stream/updates/examples/LR, separate parameters and compute. Fresh lexical controls do not have matched historical competence. No oracle attributes enter their inputs.
- On allowed training pairs only, inspect per-label BCE, per-component gradient norms and old/new gradient cosine at initialization and updates100/600/1200. These inference/backward diagnostics perform no optimizer updates and preserve weights, existing gradients, modes and RNG. Report false negatives, actual logits and per-property recall; do not interpret a negative cosine as sole cause.
- Own stream/cursor/objective/provenance in canonical `LearningState`; preserve inherited weights/moments and source metadata. Verify topology and all live optimizer membership; publish only unpromoted isolated checkpoints. Reload final continuing checkpoint in a fresh process, compare predictions and the next sampled update plus full state signature under matching code/runtime/policy. Original parent bytes and tensors remain immutable.
- Remote bounds: scientific driver900s, focused checks240s, full suite1200s, fresh-process continuation120s. Rollback is the immutable historical parent and scoped Git reversion. Record failures, do not promote checkpoints.

## Conditional later phases

If the continuing primary cannot stably fit BOTH supports, **STOP**. Report whether the fresh control succeeds, gradient/label/optimization evidence and remaining uncertainty. Do not run another rehearsal or preservation algorithm.

Only after joint acquisition passes, freeze a separate matched sequential protocol comparing no rehearsal, the nine actual historical examples, newly generated/label-verified old-grammar rehearsal, and balanced old/new experience. Never relabel synthetic rehearsal as historical memory or train on viewed probes. Keep the95%old-behavior gate and stable acquisition requirements. Only if that remains insufficient may a separately preregistered minimal correct-teacher or parameter-preservation diagnostic run; reference predictions must be on permitted training examples and checked against truth. No automatic reset or new modules.

Independent sealed names/grammars/combinations and additional independent initializations/histories remain conditional on stable acquisition AND preservation with frozen hyperparameters. Current data has only two attributes/all nine status combinations: fitting it is narrow language acquisition, not causal/compositional reasoning. No entity binding, autonomous controller, structural growth, larger capacity or learned-self-improvement claim is authorized by a software pass.

Status at preregistration: IMPLEMENTATION PLANNED; real model results NOT RUN. Historical A.4 artifacts and thresholds are unchanged.

Implementation status before submission: IMPLEMENTED; LOCALLY CHECKED (MOCK ONLY). Three inspected fake/string/scalar tests pass. Ruff check/format, syntax compilation and scoped standard Pyright (explicit project interpreter and src/scripts/tests paths) pass. No real corpus generation or neural forward/backward/training has run locally. New paths: `src/adaptive_mini_gpt/data/joint_acquisition_a5.py`, `scripts/run_joint_acquisition_a5.py`, `kaggle/launch_joint_acquisition_a5.py`, `tests/test_joint_acquisition_a5_mock.py` and the spec/report. No production neural/optimizer/loss/state implementation is changed. The launcher pins source, checks preserved A.4 evidence, runs focused contracts, then the bounded study and full remote suite; it retains errors/timeouts.

## Submission and reproducible commands

REMOTE SUBMITTED: `sparklinghydraulics0/amg-stage-a5-joint-acquisition/1`, exact executable source `a79c74471f812067d45e6079c4e5d3e544ccd53c`. It passed through QUEUED and is RUNNING; this is not a result. The source was committed/pushed before submission. Pinned launcher and private GPU metadata are under `runs/kaggle-stage-a5-joint/job/`; checkpoint input is the existing `sparklinghydraulics0/amg-stage-a-diagnostic` kernel, with runtime SHA256 enforcement.

LOCAL-SAFE checks actually run:

```bash
UV_CACHE_DIR=/tmp/amg-uv-cache uv run --offline --no-sync ruff check scripts/run_joint_acquisition_a5.py kaggle/launch_joint_acquisition_a5.py src/adaptive_mini_gpt/data/joint_acquisition_a5.py tests/test_joint_acquisition_a5_mock.py
UV_CACHE_DIR=/tmp/amg-uv-cache uv run --offline --no-sync ruff format --check scripts/run_joint_acquisition_a5.py kaggle/launch_joint_acquisition_a5.py src/adaptive_mini_gpt/data/joint_acquisition_a5.py tests/test_joint_acquisition_a5_mock.py
UV_CACHE_DIR=/tmp/amg-uv-cache uv run --offline --no-sync python -m compileall -q scripts/run_joint_acquisition_a5.py kaggle/launch_joint_acquisition_a5.py src/adaptive_mini_gpt/data/joint_acquisition_a5.py tests/test_joint_acquisition_a5_mock.py
UV_CACHE_DIR=/tmp/amg-uv-cache uv run --offline --no-sync pytest -q tests/test_joint_acquisition_a5_mock.py
```

Scoped Pyright uses the same four files, `.venv/bin/python`, standard mode/Python3.12 and import paths `src`, `scripts`, `tests`; zero errors/warnings after explicit two-element tuple annotations and typed mock dictionaries. No typing diagnostic is suppressed.

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

Kaggle `sparklinghydraulics0/amg-stage-a5-joint-acquisition/1` COMPLETE at `a79c74471f812067d45e6079c4e5d3e544ccd53c`: 817 full regressions and4focused contracts pass, zero failures/errors/skips. Runtime: Tesla T4 / Python3.13.15 / PyTorch2.11.0+cu128 / CUDA12.8. Fresh-process continuing-checkpoint predictions and the next update/state signature match exactly; the parent and all six unpromoted endpoint files are preserved. The full suite takes646.074s; focused8.238s. Each neural arm uses21600presentations,10800per distribution,2621072nonpadding tokens and1200updates. Measured synchronized study wall: continuing35.095s, fresh34.605s, including diagnostics/evaluation/hashing/checkpointing; not kernel-only or billed time.

| Arm | Old-grammar training | New-grammar training | Previous old known / renamed | A.4 development grammar |
| --- | --- | --- | --- | --- |
| Continuing |100%|100%|45.24% /47.62%|30.77%|
| Fresh diagnostic |100%|100%|59.52% /63.10%|56.41%|
| Majority |42.86%|42.86%|42.86% /42.86%|38.46%|
| BoW linear |61.90%|61.90%|61.90% /61.90%|58.97%|
| Word1–3gram linear |100%|100%|100% /100%|100%|

Both neural arms have100%old AND new training exact at1000/1100/1200, every frame/status100%; the fixed primary passes the joint-acquisition prerequisite. The continuing old observations begin100%/100%, then deteriorate even under concurrent grammar training. Thus the same shared neural model CAN fit these finite210descriptions without extra capacity, but old-frame knowledge preservation still fails. The fresh control has no initial old competence and its final old observations are generalization scores, not retention. A.4 versus A.5 differs in data, names, negation coverage and exposure; this is not a matched causal comparison proving a sequential-only failure.

Continuing training-pair gradient cosine at initialization is positive for embeddings0.4444, top Transformer0.6188, projection0.3221 and attribute head0.6017. All recorded continuing cosines remain positive at100/600/1200; state signatures remain unchanged by these diagnostics. A negative old/new cosine is therefore not established as the explanation on these sampled TRAINING pairs. This does not measure gradients on the forbidden old probes or rule out interference elsewhere. At1200 the two old-known recalls are7.14%/28.57% (26/20false negatives of28positives each); new development recalls43.33%/36.67%. Full logits, confusion counts and actual predictions remain in raw `results.json`.

456saved rule/question token sequences round-trip completely (max158/context256); all210training labels independently verify, all210examples actually occur. One architecture has1,255,636stored/406,762declared trainable parameters,341,626parameters with first-update gradients; inactive registered paths are not acquired skills. Each arm records5928evaluation rows/208chunked forwards, plus the separately logged four two-forward training-gradient probes and continuation test. Source indices and exposure audit are preserved. [Compact evidence](../data/manifests/a5_joint_v1.json), [26 measured curve rows](figures/a5_joint_curves.csv), raw `runs/kaggle-stage-a5-joint/output-v1/a5-joint/`.

## Phase 2 preregistration — after the joint-acquisition gate passed

The phase1 model demonstrates finite joint fit, permitting the authorized rehearsal coverage experiment. Start every arm from the SAME immutable pre-A.3 continuing parent and inherited Adam, never from an already jointly trained candidate. No reset, teacher objective or architectural change. New learning support is the126new-grammar phase1 TRAINING descriptions (Rax/Vim); these are permitted training data, never relabeled as holdouts.

Compare four streams: (1) new-only; (2)6of18actual historical `Object A` examples; (3)**fixed primary**,6of18new synthetic old-grammar rehearsal; (4)9of18synthetic old-grammar balanced mixture. Every arm has1200updates/21600presentations/LR.002/clip1/decay0/seed42. New exposure is21600/14400/14400/10800; old experience0/7200/7200/10800. Each batch preserves exactly two examples of every status pair; replaced slots have the identical annotation pair. Shared unreplaced new slots are identical across arms. Matched observations/updates do not mean equal tokens or compute.

Synthetic rehearsal uses old grammars on independently chosen single-letter names `N`,`P`,`Q`,`R`, familiar not/without and both property orders (expected168distinct descriptions). This intentionally covers the old one-byte-name layout that the longer Rax/Vim-only joint support did not cover; it is development data design, not an independent test. Independently verify all labels and exact string disjointness from all prior probes/validation/development and both A.5 supports before updating. Do not use J/K/Y/Z old observations as training. Synthetic labels are simulator annotations only; student input remains complete text/action0. Report the teacher-free parent's initial accuracy on this new synthetic support; do not call its fitting retention if it was not initially mastered. Genuine historical9 retains its immutable file/hash and distinct source status.

The primary must have stable new-support acquisition at1000/1100/1200 with the unchanged99%exact/every-family and98%status requirements. Preserve the95%old-known AND old-renamed gate; require it at ALL three late checkpoints to call preservation stable. A.4 development exact90%/each-family85%/each-status95%remains necessary before independent qualification, with final endpoint fixed and complete curves reported. No best-arm/step selection. Save actual predictions, loss/gradients, exposures, tokens, active/stored parameters, optimizer ages and all candidate hashes. Replicate primary fresh-process next-update continuation. Lexical controls use each actual stream and training-only vocabulary; budgets, history differences and compute remain explicit.

If no rehearsal arm achieves stable acquisition and old preservation, only the separately authorized ONE minimal correct-reference preservation objective may be considered, with a new frozen protocol first. If preservation succeeds but unseen development wording fails, stop on the language-generalization bottleneck; do not add a preservation objective to force a grammar score. Sealed qualification remains ungenerated until all prerequisites pass; all candidates stay unpromoted. Phase2 remote scientific budget900s, full suite1200s, focused240s and child120s. No phase2 outcome has been observed at preregistration.

Phase2 implementation is local-static/mock checked: five tiny schedule/scalar/string tests pass across the joint/rehearsal files; Ruff/format/syntax and scoped Pyright pass. Genuine replay is runtime-verified through the unchanged A.4 audit; synthetic support is independently label checked and protected-string checked. Same new slots/status composition are verified across four streams. No real phase2 data/model execution occurs locally. Engineering phase1's two known warnings (legacy docstring escape and intentional optimizer instrumentation) remain in the preserved full-suite log. Phase2 is not a scientific result until its remote artifact completes.
