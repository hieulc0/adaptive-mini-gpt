# Research integrity and qualification contract

This is an audit addendum, not a replacement for historical reports or their acceptance thresholds. M103-C/D “QUALIFIED_PASS” included gates for implementing/reporting experiments even when acquisition or generalization failed. Such engineering gate counts must never be summarized as general intelligence or learning-to-learn success.

## Six independent evidence classes

| Claim | Necessary evidence | Insufficient evidence |
| --- | --- | --- |
| Engineering correctness | Contracts, gradient/optimizer coverage, causal invariance, serialization and resume tests | Passing shape tests alone |
| Neural acquisition | Loss and held-out-free tiny-batch fitting, real weight updates, learned predictions | Lookup/simulator answers or tensor registration |
| Independent generalization | Complete observable inputs, unseen entities/templates/worlds, strong controls, per-class results | Training exact match or renamed memorized examples |
| Retention | Initial certified independent competence, actual later encounters, paired old-task evaluations | Frozen tensors, low baseline, assigned baseline scores |
| Transfer / learning-to-learn | Matched future-task curves and cost to equal held-out target, persistent/fresh/language/memory-disabled/fixed controls | Loss ratios or one-step saturated ties |
| Learned self-modification | Real modification and bounded learning, development-only controller updates, independent future benefit over fixed/random | Proposal counts, acceptance rate, no-op commits |

## Current benchmark validity

M104-E extraction is invalid as an equal-input neural/lexical comparison: the neural path received a truncated rule prefix; lexical saw descriptions. Calibration and reported attribute F1 use the same held-out-pair split. F1 is therefore development performance, not an independent final test. Full 26-class confusion counts are useful, but the vocabulary drops `cold` while including material names not always present in truth tags; name/tag consistency and observability need explicit validation. Rare and absent classes, negative examples, description ambiguity and logical attribute entailments need per-class support reporting. A global `pos_weight=2` is not an audited class balance correction.

Paraphrase items reused original question text; after repair they still reuse entity/label cases, so they are paired robustness tests, not independent new observations. Procedural worlds change seed/ID without necessarily changing rule semantics, vocabulary or templates; a new seed is not proof of a new task distribution. Candidate pools are drawn from novel as well as familiar entities. Before using active practice, audit semantic object/action/rule overlap with evaluation; different IDs or wordings alone cannot prove disjointness.

M104-E transfer silently mapped unknown actions and most outcomes to zero. Its 91.1% accuracy and 1.00x ratio cannot qualify future learning. The current fail-closed encoder blocks that benchmark until a versioned action/state vocabulary and head-resize/migration contract exist. Do not “repair” scores by changing target thresholds. A zero-step target makes a speed ratio undefined; censored failures are not observed times and must not be averaged as if they reached target. Current code floors denominator with `max(1, ...)`, labels some one-arm failures BOTH_CENSORED, and allows either lower mean steps or higher step-zero accuracy to count as advantage without uncertainty. Separate zero-shot transfer, acquisition speed, final quality and failures.

E1 seeds select data but do not comprehensively reset model/batch RNG before every arm. E2, E3 and M105 use only the first seed despite a five-seed E1 report. Arms use different learning rates, clipping/decay and prior budgets; they lack matched minibatch streams. Repeated shared-template examples are correlated. Use world/seed as the primary statistical unit and paired confidence intervals, not IID confidence over all generated rows. Report individual failures; do not choose best seeds or count duplicated paraphrase cases as new independent samples.

## Required controls and data boundaries

Persist a source SHA, generator/spec hash, manifest/schema, canonical semantic identities, split hash, world IDs and vocabulary for every run. Separate train, development/calibration, candidate qualification, controller development and final sealed worlds. Check exact and semantic overlaps: entity/material/action combinations, numeric values, program templates, causal rules, vocabulary assignments and generation seeds. Never give hidden family IDs, target states, simulator internals or sealed labels to the student/controller. Record exactly which structured actions or state observations are allowed inputs.

Use majority, lexical, linear/shallow neural, unchanged student, fresh student and language-only controls as appropriate. For reuse include ALWAYS_REUSE, equal-budget ALWAYS_FRESH, RESET_WEIGHTS, unchanged repertoire, retain-all, irrelevant/wrong route and shuffled weights. For memory include disabled/shuffled/relevant replay; count memory bytes and query cost separately. Oracle-attribute and deterministic simulator arms must be explicitly named upper bounds. Their performance cannot be attributed to the text student.

First establish balanced tiny-batch fitting and unfamiliar-description generalization with complete inputs. Fix thresholds from development data only, then evaluate sealed data once per predeclared experiment. Do not tune against the sealed outcomes. Use negation/minimal pairs, distractor entities, action swaps and rule swaps to distinguish description/rule use from shortcuts. Do not infer conceptual understanding from a single lexical domain.

Match examples, optimizer updates, token counts, validation queries, candidate search and measured GPU time. Include historical pretraining costs separately from incremental adaptation. Evaluate curves densely enough to detect saturation; if all controls hit target after one update, redesign task difficulty on development worlds before sealing a new test. Report accuracy, F1, exact attribute-set match, calibration, worst-class/world performance, retained competence, active/stored capacity and compute separately.

Retention requires a predeclared competence gate stronger than the relevant trivial controls and acceptable uncertainty. Repeated measurements of untouched old tasks are not multiple future encounters. Use at least several actual intervening new tasks and delayed revisits. A retention percentage relative to near-zero competence is meaningless.

Self-modification trials require identical initial student/optimizer/memory/RNG state across policies, matched action and hyperparameter spaces, real bounded training after each decision, development reward and independently sealed future outcomes. Measure delta in future learning efficiency and competence, not acceptance. A complete transaction must restore topology, tensor aliases, optimizer parameter identities/groups and state, scheduler/scaler, RNGs, sampler position, replay, external memory, controller state, metadata and checkpoint publication. Compare the next actual update against an untouched reference, including after a fresh-process reload. Gate open-ended modification on a demonstrated policy advantage and reliable rejection/rollback.

## Execution and provenance

LOCAL: lint/type/syntax/mock-only. REMOTE: Kaggle or Colab GPU training/full tests. CPU training is not a local diagnostic option. New/modified scientific runners must fail outside the intended remote platform or without CUDA. Legacy CPU quick-start instructions are superseded by AGENTS.md.

A qualified artifact records completed kernel/version or session ID, full source SHA, initial/final checkpoint hashes, manifest/split hashes, seeds/worlds, Python/PyTorch/CUDA/GPU versions, optimizer settings, actual updates/examples/tokens/query counts, measured GPU time, full test counts and per-arm predictions/curves. Missing provenance means NOT QUALIFIED. Completion and timeouts are not scientific passes. Store failure logs and negative findings. Use IMPLEMENTED, LOCALLY CHECKED (MOCK ONLY), REMOTE SUBMITTED, REMOTE VERIFIED and SCIENTIFICALLY QUALIFIED as separate statuses.
