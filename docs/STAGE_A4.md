# Stage A.4 — knowledge preservation and reusable concepts

Preflight: branch `m104-e-and-m105-a`, clean at `323845f`; verified neural core `4e1cc9f`. The earlier full repository/document preflight and historical contracts remain binding; current A.3 evidence, input/optimizer/checkpoint code, data manifests, old training fixture and remote workflows were reviewed. This is Stage A.4 foundation work, not a new milestone. Existing acceptance gates (A.3 stable fitting over updates 1000/1100/1200 and >=95% old-frame observations) remain unchanged. No sealed testing, architecture growth, automatic optimizer reset or candidate promotion is authorized without prerequisites.

LOCAL: lint/type/syntax/mock-only. REMOTE: Kaggle CUDA inference, training, restart/retention and full regression. No local neural execution or CPU fallback.

## Prespecified sequential plan

1. **Inference-only localization first.** Compare the immutable pre-A.3 parent (file SHA256 `3e93d7238bfdb706ddfd49ee9903fed8b1e6f577529537f1e21cae12e905b262`) with the unpromoted A.3 balanced-BCE final checkpoint (`02dcbe3feedd975df34222b2a5a38e0481bedc7a52fe56dff7d655bb14c7adf3`). Evaluate the two old development cells and all 156 A.3 training descriptions. Record full tokens, logits, per-property metrics and representations at token embeddings, Transformer output, pooled vector, concept projection and head. In isolated diagnostic-only copies, swap embedding, position embedding, frozen lower blocks, trainable top block, final norm, concept projection and attribute head individually in both directions; also swap entire backbone and entire bottleneck as interaction controls. Check topology/shape/alias compatibility; verify originals' hashes untouched. A swapped composite is not a deployable learner and may create incompatible feature bases: effects identify correlations, not unique causation. No optimizer step, backward call or training may run in this phase.
2. **Replay feasibility only after phase 1 completes.** Use the same pre-A.3 persistent parent, selective backbone and balanced two-label BCE (no paired loss and no reset). Compare exactly 0/3/6 replay examples per batch of 18, fixed 1200 updates, LR .002, decay 0, clip 1 and seed 42. This is 0%, 16.67% and 33.33% replay under identical total update/example budgets, with explicit reduced new-example exposure (21600/18000/14400) and replay exposure (0/3600/7200). The replay examples are only the nine `Object A` records actually used in historical Stage A training (`data/fixtures/stage_a_original_tiny.json`, source 216c3d5); their source/label hash and exact question strings must be checked against every viewed retention probe, prior state-training retention fixture, and new training set before any update. Fail closed on overlap. For each replaced slot, use its matching observed status pair, keeping the batch label distribution fixed. A fixed 33.33% primary is declared before results; no best-step or best-arm selection using final holdouts. Record every sampled source/index, full tokens, optimizer/state hashes, per-label outcomes, resource costs and 100-step learning/old-behavior curves. Keep old/new evaluation metrics distinct and all candidates unpromoted.
3. **Conditional selective/modular comparison.** If replay fails the unchanged acquisition+retention gates, assess the phase-1 evidence and existing supported adaptation paths. Compare selective updates and existing modules only if an actual compatible active path can be identified without adding parameters; otherwise document precisely why this is unsupported. Do not silently substitute a new module or discard useful state.
4. **Qualification remains sealed.** Only after the fixed primary has stable new-support acquisition AND >=95% in both old development cells may a separately frozen protocol generate new names/grammar/attribute combinations and run multiple learning streams with strong lexical and neural controls. All current cells were previously viewed; they are developmental, not sealed tests. No held-out data drives training or checkpoint selection. Passing a lexical task is not causal or general conceptual understanding.

All runs use explicit deterministic CUDA numerical policy and exact same-code/runtime checkpoint contracts; source, seed, manifests, checkpoint hashes and actual costs are recorded. Baseline budget is finite: 300 s inference, 900 s replay, 1200 s full regression, plus focused checks. Rollback is the untouched parent, isolated diagnostic/candidate copies and scoped Git reversion. No training has been submitted at preregistration.

## Phase 1 result — completed before any A.4 training

Kaggle `sparklinghydraulics0/amg-stage-a4-localization/1` COMPLETE at `6f3c2046d0ec26c1aac7296942f4440d79789945`. Same Tesla T4 / Python3.13.15 / PyTorch2.11.0+cu128 / CUDA12.8; synchronized inference/inspection wall 5.277 s. Zero optimizer steps, gradients or training. Input file SHA256 values and model tensor hashes remain unchanged; compatible topology/dtype and tied embedding alias verified. Artifacts: `runs/kaggle-stage-a4-localization/output-v1/a4-localization/`; compact hashes/metrics: `data/manifests/a4_localization_v1.json`.

| Isolated inference model | Old known | Old renamed | New training support |
| --- | --- | --- | --- |
| Pre-A.3 parent |100%|100%|28.21%|
| Post-A.3 balanced BCE |66.67%|71.43%|100%|
| Post model with entire pre backbone |100%|100%|21.79%|
| Pre model with entire post backbone |52.38%|50.00%|73.72%|
| Post model with pre token embeddings |42.86%|42.86%|76.92%|
| Post model with pre top block |54.76%|50.00%|24.36%|
| Post model with pre concept projection |54.76%|52.38%|76.92%|
| Post model with pre attribute head |69.05%|69.05%|100%|

Restoring the entire earlier backbone recovers both old-frame cells in the post model, but loses the new support; inserting the post backbone into the pre model substantially harms the old cells. The pre model with the post attribute head retains100%/98.81% old accuracy, so head changes alone do not explain the degradation in that pre-backbone context. These results **implicate changes to the shared backbone and its interactions with the learned projection/head**, without assigning a unique causal component. Swapping token embeddings, top block or concept projection alone often worsens both tasks, consistent with cross-version feature incompatibility; hybrids are diagnostic, never candidates for promotion.

Position-embedding and frozen-lower-block weights have exactly zero measured drift; replacing either is a no-op for measured outputs. WTE/top block/final norm/concept projection/head weight differences are nonzero and recorded by tensor. Full input token sequences, embedding and position-summed pooled vectors, Transformer outputs, concept vectors, per-property confusion counts and actual logits/predictions are preserved. This is correlation under surgical intervention, not proof that a particular CUDA operation or one weight tensor uniquely caused forgetting.

The original nine `Object A` descriptions are documented actual old TRAINING data, separate from the previously viewed J/K/Y/Z retention descriptions and the prior state-probe fixture. Replay must fail closed unless runtime verifies exact text/label isolation from all known retention/probe and new-support strings; these nine are not used as retention evaluation. The old known-name retention cell itself was historically training support; the renamed cell is a paired familiar-grammar probe, not an independent-world certificate. Phase 2 is now authorized by the completed inspection.

Phase 2 adds three disjoint **development** grammar families on familiar names (never sealed) to diagnose generalization; none enters optimizer batches or replay. Qualification admission also preserves the existing A.2 development requirements: exact >=90%, each family >=85%, and each positive/negated/absent stratum >=95%, alongside the unchanged A.3 stable-support and old-frame >=95% requirements. This adds no weaker threshold and does not make developmental results sealed evidence. The moderate-replay arm is the sole fixed primary. Raw historical fixture SHA256 is `130dc05e42ed9ee0c1122dd95532a16562fc67f5ffa95422f099eff48643f1e5`.

Phase 2 implementation: the exact source fixture is verified, then compared against all four historical factorial cells, the A.2 development sets, A.3 minimal/training examples, earlier attribute holdouts, and the prior state-retention fixture; any normalized exact-text overlap aborts before updating. Historical `tiny` remains a training subset and is not misclassified as a heldout probe. The additional development templates are also checked for exact-text overlap with previous probes and both training supports. Every replaced replay slot preserves its original independently annotated status pair, so label composition is matched, while actual new/replay examples and token counts remain separately recorded. The replay stream and curated historical examples are owned by the continuing `LearningState`; no model receives hidden status metadata at inference. The launcher refuses any training unless the completed inference-only inspection is recorded. The frozen-backbone and legacy domain-adapter paths remain for a conditional phase 3 decision after results.

LOCAL inspection: 12 tiny fake/string cases pass across the new replay mocks and existing stable-objective mocks. Scoped Ruff/type/syntax checks cover the new driver, data module, launcher and tests. Real model operations and full regression are REMOTE only; no phase-2 result existed at preregistration.

Saved representations also permit a descriptive pre/post comparison without loading a model locally. Across old known-name cases, the mean per-case L2 difference is 0.0707 for question-span mean token embeddings, 1.3334 for final pooled Transformer vectors, 9.9340 after the concept projection and 9.2882 for the two logits. These vector spaces have different dimensions/scales; the numbers cannot rank causal importance. Full values and renamed/new distributions are in `data/manifests/a4_representation_summary_v1.json`. All 324 saved rule/question sequences round-trip with markers intact; max 158 tokens within the model's 256-token experimental context.

## Phase 2 result — replay prerequisites failed

Kaggle `sparklinghydraulics0/amg-stage-a4-replay/1` COMPLETE at `4afeffce617d3b32a9b8b7c58c3f22e9a697b562`; 812 remote regressions passed, 5 focused contracts passed, 0 skipped/failed. Deterministic CUDA execution, same source/runtime policy, model and optimizer ownership all verified. The no-replay final weight hash exactly reproduces A.3 balanced BCE; moderate-replay checkpoint predictions and next sampled update/state reproduce in a fresh process. All three start from identical parent weight/optimizer hashes and remain unpromoted.

| Replay /18 | New support exact | Stable support gate | Old known / renamed | Unseen development grammar | New / replay presentations |
| --- | --- | --- | --- | --- | --- |
| 0 |100%|PASS|66.67% /71.43%|38.46%|21600 /0|
| 3 |100%|PASS|64.29% /61.90%|41.03%|18000 /3600|
| 6 (fixed primary) |92.31%|FAIL|73.81% /73.81%|33.33%|14400 /7200|

All old-frame cells begin at100%, all arms fail the 95%old-observation gate, and all development grammar results fail the unchanged 90%exact/85%per-family/95%per-status requirements. The fixed primary also fails the last three checkpoint support stability gate. The no-replay arm's final old/new weights reproduce the prior A.3 result exactly; no favorable step is selected (e.g. 1100-update old-known 80.95% is not the 1200-update endpoint). The complete curves, per-property counts, actual examples, sampled index sequences and tokens are preserved in the raw artifact and `data/manifests/a4_replay_v1.json`. No seal is generated.

All ratios present exactly 2400 examples of each of the nine status pairs. Each sees all 156 new descriptions; replay arms see all nine historical descriptions. The parent scores only 11.11% on those nine at the start: they were genuinely **encountered historical training examples**, but current competence on them was not established. Replay arms fit those nine to 100%; that is training-set relearning, not certified retention. The separately evaluated old-known cell was previous training support; the renamed old cell is a correlated familiar-grammar probe. Neither is independent world retention.

The moderate-replay developmental flammable recall is13.33% (TP4/FN26), and conductive recall33.33% (TP10/FN20). On the old-known cell, flammable recall is50% (TP14/FN14). Actual examples: `A note on J lists flammable; conductive.` has truth 1,1 but moderate replay predicts 0,0 (logits -7.594,-3.219); `For J, status: not conductive; flammable.` has truth 1,0 but predicts 0,1 (logits -1.555,1.401). The strong word1–3gram control scores100%on the unseen development grammars in all three arms, while the fresh BoW controls score 69.23–82.05%; the linear controls lack the neural student's prior history, so they are text baselines, not matched continual-history controls. Their costs/parameters are recorded separately.

![Matched replay learning curves](figures/a4_replay_curves.svg)

Measured student-arm synchronized wall time totals 100.17 s and six lexical controls 11.54 s; these figures include evaluation/host work and are not pure GPU-kernel time. Exact nonpadding tokens are 2,743,200 /2,667,691 /2,591,852 respectively; equal presentation/update budgets do not imply equal token or compute budgets. Each branch continues the same historical checkpoint independently. The stage declares a new RNG stream at its boundary; it does not claim retrospective exact resumption of the historical RNG. All 399 saved new/old/development token sequences decode completely; raw prior old-frame sequences are independently preserved by phase 1. No checkpoint is promoted.

## Conditional phase 3 — preregistered after replay failure

The full-backbone swap in phase 1 restored old behavior, motivating a **temporary frozen-backbone** control. Start from the identical historical parent and optimizer state; use exactly the predeclared 6/18 replay stream, two-label BCE, 1200 updates, seed42, LR/clip, datasets and gates from phase 2. Freeze the existing student backbone using the canonical `set_mode('frozen_backbone')`; train the existing concept projection/attribute head. Verify that every named optimizer state remains identical across the trainability transition, all live parameters remain registered, and initial predictions and backbone tensors are unchanged. Compare directly to phase-2 selective 6/18 and the no-replay selective baseline; report active/trainable counts and actual compute separately. Frozen is a study control, not a permanent architecture ceiling. Fixed phase-3 primary: frozen-backbone 6/18; no best-checkpoint selection.

For the existing modular control, first switch a disposable copy to the already-implemented entity-conditioned attention extraction path (ECA), without new parameters. Test old pre-update competence; only if both old cells are >=95% may an ECA/frozen-backbone 6/18 learning arm run. If activation itself fails this gate, record its exact predictions and mark modular retention **not qualified / training skipped**. ECA is an existing auxiliary path, not a domain adapter; the registered domain residual adapters are used by `TextStudentM103.forward_lm` but bypassed by the active attribute path. Adding a new adapter connection or gating mechanism merely to manufacture a comparison is outside this study. All models remain isolated/unpromoted. If the fixed frozen primary misses any unchanged gate, stop before sealed qualification.

Phase 3 implementation preserves every named Adam state through the freeze/unfreeze transition and rejects any missing/duplicate/stale optimizer registration. It checks old predictions immediately before training, exact frozen-backbone parameter hashes and dormant optimizer states after training, actual active gradient paths, stored/trainable parameter counts, and fresh-process continuation. The existing ECA strategy is evaluated on a disposable copy with the same tensors; if old competence fails at activation its training is skipped. The fixed primary is frozen-backbone with six historical examples per batch. LOCAL: 13 inspected mock/string tests pass; scoped Ruff/Pyright/syntax checks pass. No phase-3 model result has been produced at preregistration.

## Reproduction and source boundaries

The source commits are distinct from later evidence/documentation commits:

- `6f3c2046d0ec26c1aac7296942f4440d79789945`: inference-only localization, completed before any A.4 training.
- `4afeffce617d3b32a9b8b7c58c3f22e9a697b562`: matched replay, completed with 812 full / 5 focused remote passes.
- `fd9d321c34fee6d1899f446180017b73793a09f6`: conditional frozen/ECA protocol, submitted as `sparklinghydraulics0/amg-stage-a4-selective/1`; results must be attributed to this SHA, not a later documentation commit.

LOCAL-SAFE preparation: use the corresponding tracked `kaggle/launch_a4_{localization,replay,selective}.py`, replace its `__PINNED_SHA__` placeholder with the exact source above, and write `launch.py` plus private CUDA/internet metadata under `runs/kaggle-stage-a4-<phase>/job/`. Localization inputs are the existing `sparklinghydraulics0/amg-stage-a-diagnostic` and Stage A.3 kernel output; replay/selective require the diagnostic kernel only. The launchers validate checkpoint hashes and refuse CPU execution. All source commits must already exist on origin.

REMOTE submission and artifact retrieval (CLI runs locally; all model execution is on Kaggle):

```bash
kaggle kernels push -p runs/kaggle-stage-a4-selective/job
kaggle kernels status sparklinghydraulics0/amg-stage-a4-selective
kaggle kernels output sparklinghydraulics0/amg-stage-a4-selective -p runs/kaggle-stage-a4-selective/output-v1
```

The same command pattern applies to the localization and replay kernel IDs above. Do not resubmit into an existing evidence directory or mistake a queued kernel for a pass. Full raw checkpoint/token/case artifacts remain in ignored `runs/`; compact tracked manifests bind file hashes, outcomes and exact runtime. Reproduction of an old source must use its matching pinned launcher, inputs and numerical policy.

REMOTE commands executed by the selective launcher, after pinned checkout and CUDA preflight:

```bash
python -m pytest -q tests/test_a4_selective_mock.py tests/test_a4_selective_remote.py tests/test_backend_update_remote.py --junitxml=/kaggle/working/a4-selective/contracts.xml
python scripts/run_a4_selective.py --root <immutable-input>/inherited_weights_inherited_moments.pt --output-dir /kaggle/working/a4-selective
python -m pytest -q --junitxml=/kaggle/working/a4-selective/full-suite.xml
```

All processes inherit `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `AMG_RUN_GPU_TESTS=1`, `PYTHONPATH=<pinned-repo>/src`, and two OMP/MKL threads. The experiment explicitly enables deterministic algorithms and disables cuDNN benchmarking. The tests execute inside the remote CUDA runtime; local mocks are never substituted for them.

LOCAL verification commands (no model calls, optimizer updates or checkpoint loading):

```bash
UV_CACHE_DIR=/tmp/amg-uv-cache uv run --offline --no-sync ruff check scripts/run_a4_localization.py scripts/run_a4_replay.py scripts/run_a4_selective.py kaggle/launch_a4_localization.py kaggle/launch_a4_replay.py kaggle/launch_a4_selective.py src/adaptive_mini_gpt/data/replay_a4.py tests/test_replay_a4_mock.py tests/test_replay_a4_remote.py tests/test_a4_selective_mock.py tests/test_a4_selective_remote.py
UV_CACHE_DIR=/tmp/amg-uv-cache uv run --offline --no-sync ruff format --check scripts/run_a4_localization.py scripts/run_a4_replay.py scripts/run_a4_selective.py kaggle/launch_a4_localization.py kaggle/launch_a4_replay.py kaggle/launch_a4_selective.py src/adaptive_mini_gpt/data/replay_a4.py tests/test_replay_a4_mock.py tests/test_replay_a4_remote.py tests/test_a4_selective_mock.py tests/test_a4_selective_remote.py
UV_CACHE_DIR=/tmp/amg-uv-cache uv run --offline --no-sync python -m compileall -q scripts/run_a4_localization.py scripts/run_a4_replay.py scripts/run_a4_selective.py kaggle/launch_a4_localization.py kaggle/launch_a4_replay.py kaggle/launch_a4_selective.py src/adaptive_mini_gpt/data/replay_a4.py tests/test_replay_a4_mock.py tests/test_replay_a4_remote.py tests/test_a4_selective_mock.py tests/test_a4_selective_remote.py
UV_CACHE_DIR=/tmp/amg-uv-cache uv run --offline --no-sync pytest -q tests/test_a4_selective_mock.py tests/test_replay_a4_mock.py tests/test_stable_attributes_mock.py
```

Result: Ruff/format/syntax pass, 13 inspected scalar/fake/string tests pass. Pyright also passes with zero errors/warnings using the same 11 changed Python files, explicit project `.venv/bin/python`, standard checking/Python3.12, and `src`, `scripts`, `tests` as import paths. The initial default-cache commands could not write the read-only global uv cache; using `/tmp/amg-uv-cache` fixes this without environment mutation. The default Pyright invocation lacked the project interpreter/import paths; only the corrected scoped check is counted. These are local tool-configuration limitations, not neural test failures.

## Phase 3 result — frozen weights do not preserve predictions

Kaggle `sparklinghydraulics0/amg-stage-a4-selective/1` is **COMPLETE**, source `fd9d321c34fee6d1899f446180017b73793a09f6`. Tesla T4, Python3.13.15, PyTorch2.11.0+cu128, CUDA12.8. The 814-test full regression and 3 focused contracts pass, with zero failures/errors/skips. Full-suite time 698.241 s; focused 10.908 s. Two full-suite warnings are preserved: a pre-existing docstring escape warning in `neural_meta_learning/reptile.py` and the intentional optimizer-step instrumentation warning in `test_diagnostic_loss_scaling_uses_same_owner_and_gradients`. The study has no scheduler; those test warnings are not a learning PASS or evidence of an applied-update failure.

| Fixed protocol / endpoint | New support exact | Old known / renamed | New development exact | Stable new / old / development gates |
| --- | --- | --- | --- | --- |
| Selective BCE, replay0 |100%|66.67% /71.43%|38.46%|PASS /FAIL /FAIL|
| Selective BCE, replay3 |100%|64.29% /61.90%|41.03%|PASS /FAIL /FAIL|
| Selective BCE, replay6 |92.31%|73.81% /73.81%|33.33%|FAIL /FAIL /FAIL|
| Frozen backbone BCE, replay6 (phase-3 primary) |77.56%|42.86% /42.86%|43.59%|FAIL /FAIL /FAIL|

The frozen primary has the identical `bd11b5e879fcea3d1b575da85c100a1188865baf9eea53f8db81c15e69a7719c` data/stream hash, same parent weights and named Adam states, 1200 updates, 21600 presentations (14400 new/7200 historical), 2,591,852 nonpadding tokens, seed42 and LR.002. New-support exact at updates 1000/1100/1200 is 76.28%/76.92%/77.56%, failing the fixed stability gate. The nine replayed training cases reach 88.89%; no best step is selected. All curves including initial states are available in [a4_preservation_curves.csv](figures/a4_preservation_curves.csv) and the raw report; the phase-2 plot above remains unchanged.

Stored parameters stay 1,255,636. Frozen mode declares 77,162 trainable (including inactive registered ECA); only 12,026 parameters have gradients on the first question-only update. ECA gradient norm is 0, as expected for that inactive route. Selective mode declares 406,762 trainable; stored capacity is identical. These are trainability/active-gradient counts, not FLOPs. Measured frozen-arm synchronized wall time 25.047 s includes evaluation, hash checking and saving; it is not pure kernel time. The four student arms total 125.22 s of measured synchronized study wall, separately from lexical controls and remote tests. Equal examples/steps do not establish equal compute.

Every named Adam state is preserved by the freeze transition. All frozen backbone tensors and pre-existing backbone optimizer states are unchanged after training; initial old logits are identical to the parent. Lifetime owner updates progress 1400 to 2600; frozen student Adam ages remain 1400, dormant state ages remain 200, and updated head-side ages reach 2600. Dormant counters are not misreported as newly applied backbone steps. The saved candidate SHA256 is `8136ff5eeb0fc79d51c22ea70946dde9834355d2966b3980d4999176f55ec8cb`. Fresh-process reload reproduces predictions and the exact next sampled update/loss/complete learning-state signature under the same code/runtime/numerical policy. Two continuation-test updates occur only on disposable copies; the saved candidate is unchanged. No automatic optimizer reset occurs.

### Per-property failure and actual predictions

The frozen primary's old-known predictions are **0,0 for every one of84 cases**, with zero positive recall on either property (28 false negatives each). Negative/absent status accuracy is 100%; that does not rescue knowledge retention. Old-renamed positive recall is 0% flammable and 3.57% conductive (one true positive); exact match remains 42.86%. The head-side path has lost old positive predictions despite immutable Transformer weights.

| Frozen primary evaluation | Flammable accuracy / precision / recall | Conductive accuracy / precision / recall |
| --- | --- | --- |
| New training support156 |85.26% /76.81% /88.33%|87.18% /77.78% /93.33%|
| Old known84 |66.67% /0% /0%|66.67% /0% /0%|
| Old renamed84 |66.67% /0% /0%|67.86% /100% /3.57%|
| Development grammar78 |64.10% /60% /20%|58.97% /0% /0%|

Zero precision denotes no true positive and uses the existing zero-denominator convention. Development positive/negated/absent status accuracy is 10%/96.67%/88.89%; each development family is 38.46%/53.85%/38.46%. These are previously viewed development cases, never sealed evidence.

Actual endpoint predictions: `For K, status: flammable; conductive.` has truth 1,1, prediction 0,1 and logits -0.157,14.946. `A record. For J, status: flammable; not conductive.` has truth 1,0, prediction 0,1 and logits -0.480,0.318. The development description `A note on J lists flammable; conductive.` has truth 1,1, prediction 0,0 and logits -18.342,-10.492. The full confusion counts, all truth/prediction/logit arrays and exact descriptions/tokens are preserved, not just selected examples. JSON/string analysis verifies399 complete saved sequences against the rules and questions, max 158/context 256, identical to the replay study. The independent phase-1 artifact retains the old-probe token sequences. This rules out silent truncation in these cases; it does not prove semantic use of each token.

### Existing modular path

Activating the already-registered ECA route on the parent changes no tensor, but old-known accuracy falls to 85.71% and old-renamed to 83.33% before any update. New support is 30.77% and development 41.03%. Both old cells fail the predeclared95%activation gate, so **ECA training is skipped; modular retention is not qualified**. This is a measured route-compatibility failure, not evidence that a trained ECA can never learn. Connecting the separate LM-domain adapters to the concept path would require a new architectural intervention, which this failure does not justify. No such connection, new parameter or module is introduced.

## Scientific conclusion and stop decision

**Engineering PASS; scientific PARTIAL. The primary acquisition-with-preservation condition FAILS.** Stop here: no independent sealed grammar, new sampling seeds, entity binding, causal reasoning, learning-to-learn, autonomous mutation or milestone advance. The pre-A.3 parent and all historical artifacts remain immutable; all A.4 checkpoints are experimental and unpromoted.

The combined mechanism is more specific than a generic claim of forgetting: phase 1 shows that shared-backbone changes can substantially disturb earlier predictions, while phase 3 demonstrates that updates to the existing projection/head pathway can independently erase those predictions even when the entire backbone remains bit-identical. Thus freezing the backbone is insufficient protection. Head/representation co-adaptation and inadequate coverage of earlier behavior by the available nine-example replay are supported problems; which projection/head parameters or optimization dynamics dominate remains unresolved. A frozen-feature failure at this budget is not proof of an information-theoretic capacity ceiling, nor evidence that adding layers or modules is necessary.

Replay fits or relearns its own small support without preserving the separately evaluated old frames. It is not a representative retained-history guarantee. There is still no learned frame-invariant concept certificate: n-gram controls score 100% on the development descriptions at every replay ratio, while all neural arms fail. The controls use the same texts/update counts but lack the student's inherited history, so their old-frame scores are not matched-history transfer comparisons. Multiple correlated descriptions from one prior student and one new sampling stream are not independent replications; no confidence interval treating them as independent worlds is claimed. The other 24 attribute outputs, language generation, causal state predictions, unrelated skills and long-session retention are unqualified by these two-label studies.

The next justified work is to diagnose the existing projection/readout's old/new interference and whether authentic replay history adequately covers the previously competent distribution, using development-only experiments. Any supervision or representation intervention must first reproduce simultaneous tiny acquisition and old-behavior preservation from the immutable parent with matched budgets. Do not replay the viewed probes, automatically reset Adam, adopt ECA on activation alone, add capacity, or use these already-seen grammars as a fresh seal. This phase does not justify promoting a new learning policy.

Evidence: [localization manifest](../data/manifests/a4_localization_v1.json), [representation deltas](../data/manifests/a4_representation_summary_v1.json), [replay manifest](../data/manifests/a4_replay_v1.json), [preservation manifest](../data/manifests/a4_selective_v1.json). Raw phase 3 artifacts: `runs/kaggle-stage-a4-selective/output-v1/a4-selective/`; source/runtime, file hashes, tokens, all curves, actual predictions, checkpoint and continuation child are retained. Remote status is COMPLETE; scientific gates are explicitly false.

## Exact changed-file inventory

Relative to the Stage A.3 documentation parent `323845f`, the study changes only the following paths. Model capacity, production Transformer/tokenizer/head implementations, canonical `LearningState`, historical reports/specs/fixtures and historical checkpoints are unchanged. New remote regression files verify the study's replay/freeze invariants; they are not local training tests.

- `README.md`
- `data/manifests/a4_localization_v1.json`
- `data/manifests/a4_replay_v1.json`
- `data/manifests/a4_representation_summary_v1.json`
- `data/manifests/a4_selective_v1.json`
- `data/specs/replay_a4_v1.json`
- `data/specs/selective_a4_v1.json`
- `docs/04-experiments.md`
- `docs/05-roadmap.md`
- `docs/06-data-pipeline.md`
- `docs/CANONICAL_LEARNING.md`
- `docs/FOUNDATION_DEFECT_REGISTER.md`
- `docs/STAGE_A4.md`
- `docs/figures/a4_preservation_curves.csv`
- `docs/figures/a4_replay_curves.csv`
- `docs/figures/a4_replay_curves.svg`
- `kaggle/launch_a4_localization.py`
- `kaggle/launch_a4_replay.py`
- `kaggle/launch_a4_selective.py`
- `scripts/run_a4_localization.py`
- `scripts/run_a4_replay.py`
- `scripts/run_a4_selective.py`
- `src/adaptive_mini_gpt/data/replay_a4.py`
- `tests/test_a4_selective_mock.py`
- `tests/test_a4_selective_remote.py`
- `tests/test_replay_a4_mock.py`
- `tests/test_replay_a4_remote.py`

The evidence commit following `fd9d321` contains only reports, compact manifests and saved-data curves. The remote 814-test certification applies exactly to the executable source `fd9d321c34fee6d1899f446180017b73793a09f6`; it must not be relabeled as a test run at a later documentation SHA. Prior source commits `6f3c204` and `4afeffc` are retained, and `4e1cc9f` remains the verified neural implementation ancestor.

LOCAL: lint/type/syntax/mock-only. REMOTE: Kaggle CUDA inference/training/continuation/full tests. Sealed qualification: **NOT RUN — scientific admission failed**, not a remote availability blocker. No run or failed candidate is discarded.

Cost limitation: the synchronized study wall times and test-process durations are measured; kernel-only CUDA event time, allocator utilization and billed Kaggle allocation duration were not recorded. They are not estimated from the learning curves. Budget matching here is explicit examples/updates and learning rate, with token exposure and differing compute reported separately.
