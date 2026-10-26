# Stage A: isolate attribute acquisition

Status: IMPLEMENTED protocol; no new neural result yet. This continues foundation qualification on `m104-e-and-m105-a`, not a new milestone. The combined concept-binding study remains FAIL and its artifacts/reports are immutable. `reports/` is absent; completed scientific reports reside in `docs/` and preserved `runs/` outputs.

## Hypothesis and minimal intervention

The combined experiment did not isolate language, binding, transitions or optimization. The canonical owner always computed state CE plus 26-label BCE. That is a valid joint objective for fully annotated tasks, but prevents an attribute-only update without irrelevant state supervision. `LearningState.step` now accepts omitted state targets and explicit observed attribute columns. Its default joint-loss mathematical contract stays unchanged. Selecting two observed columns avoids assigning negatives to 24 properties outside this experiment; an all26-negative control tests that supervision choice. Neither a historical BCE bug nor the root cause of the old collapse is assumed.

Reuse the same dense Transformer, byte tokenizer, question pooling, concept projection and existing attribute head. No new component, capacity, representation, threshold search or architecture growth. The action placeholder cannot influence attribute logits; the action/transition/state weights must remain unchanged. This parameter check does not prove behavioral retention of prior skills through changed shared representations.

## Protocol fixed before remote results

`data/specs/attribute_learning_v1.json` is binding. One object has two independently annotated traits, each positive, negated or absent. All nine combinations occur in training. Single-object property negation is required; multi-object binding and transitions are excluded. Training and validation use two grammars with disjoint names. The two heldout splits separately test new names in known grammar and new names in two unseen description templates. Property words and negation operators remain familiar. This is controlled grammatical generalization, not novel-world or causal qualification.

The generator emits labels from annotations, never by parsing text at student inference. The neural learner receives complete rules and descriptions only; no status/family IDs, labels or structured features enter inference. Export every delivered token sequence, exact text reconstruction, data hash, class/family counts, logits and actual predictions. All label combinations and all three status families must be covered. Exact-word-count matched negation examples demonstrate a bag-of-words ambiguity; word 1–3-gram linear remains a stronger control and may solve this task. A lexical win/tie is an honest limitation, not grounds to weaken its baseline.

Primary lineage: continue the existing selective student from `amg-concept-binding-foundation/1`, SHA256 `4a4f79fcd07b5bbe58bad5da04ce1a934696438c8603e9e6296e398a4b5ed5d8`, after its 200 joint tiny updates. Retain weights and optimizer moments. Declare a new owned sampler at the stage boundary; the old artifact lacks a persisted sampler, so this is not retrospective exact resumption. Verify parent hash and update count, fail closed if missing. Old checkpoint immutable. The new stage stores protocol, supervision, sampler and counters with the continuing owner. Unqualified final checkpoints are candidates, not promoted knowledge.

Fresh frozen/selective/full two-column controls share seed42 initialization and sampling. Fresh selective all26 is the loss diagnostic. These are controls, not new persistent specialists. The continuing student's extra 200 historical updates mean it is not a budget-matched transfer comparison. No learning-to-learn inference is allowed. Majority, BOW linear, BOW MLP32 and word 1–3-gram linear receive the same descriptions. Vocabulary fits training only. Neural and lexical arms have matched new-stage observation/update ceilings, but GPU compute/parameters differ and are reported separately. Tiny-failed neural arms stop early and cannot be ranked on unrun heldout scores.

Budget: seed42, LR.002, decay0, clipping1, batch<=16; 200 tiny updates over one example per annotation combination, then800 main. Stop each neural arm below95% tiny exact. Validation every100 updates; no hyperparameter/checkpoint selection. Heldout evaluated once at fixed endpoint. Basic acquisition requires train exact>=95%, both heldout exact>=90% and positive/negated/absent accuracy>=90% in each. Reusable qualification additionally requires >=10 percentage points over the strongest simple control in BOTH heldout splits. No statistical independence is assumed between grammar cases; one seed cannot qualify broad concepts. No seed shopping or post-test threshold changes.

If primary fails any gate, STOP at Stage A. Inspect development curves, per-status failures, predictions and the matched two-vs26 loss control; do not add architecture to force PASS. Entity binding, one-step transitions, composition, feedback, retention and future-learning efficiency remain NOT RUN. A–G are successive learning stages of the same owner, not seven solvers. Even feasibility success requires independent replication before promotion.

## Execution and rollback

LOCAL: lint/type/syntax/mock-only. Inspect/run only `tests/test_attribute_learning_fixtures.py` locally: three tiny string/token fixtures. Actual loss/gradient tests and canonical updates in `tests/test_attribute_learning_remote.py` are REMOTE-REQUIRED. All training, inference and full tests require Kaggle/Colab CUDA; the entry point aborts elsewhere.

REMOTE: Kaggle GPU, template `kaggle/launch_attribute_learning.py`, private job ID `sparklinghydraulics0/amg-attribute-learning`, attach `kernel_sources=["sparklinghydraulics0/amg-concept-binding-foundation"]`. Commit/push before rendering `__PINNED_SHA__`. Hash-check the attached parent. Limits: contracts240s, study600s, full suite1200s. A script completion is not scientific PASS. Preserve failures/timeouts and all historical artifacts. No local CPU fallback.

Remote commands executed by pinned launcher:

```bash
python -m pytest -q tests/test_foundation_structure.py tests/test_foundation_mock_contracts.py tests/test_attribute_learning_fixtures.py tests/test_attribute_learning_remote.py --junitxml=/kaggle/working/attribute-learning/contracts.xml
python scripts/run_attribute_learning.py --output-dir /kaggle/working/attribute-learning --parent <attached-selective_plasticity.pt>
AMG_RUN_GPU_TESTS=1 python -m pytest -q --junitxml=/kaggle/working/attribute-learning/full-suite.xml
```

Rollback: retain immutable parent and unpromoted output separately. Source changes are a scoped Git revert. No preexisting checkpoint or historical report is overwritten. Structural operations remain disabled in this study; long-term architectural eligibility is unchanged.
