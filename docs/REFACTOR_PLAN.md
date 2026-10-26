# Minimal repair and qualification plan

No new milestone number is assigned. Historical report files remain immutable. Initial branch/HEAD and findings are in ARCHITECTURE_AUDIT.md.

## Stage A — forensic review

Completed source review of the current M104-E/M105-A path and relevant inherited mechanisms. Confirmed complete-input failure, label collapse, ineffective paraphrase evaluation, substring exact match, embedding decay, strategy labeling, threshold mismatch, error-path crash and unsupported operator no-op acceptance. Broader MoD, RoPE, growth and rollback defects are recorded with source references. This is not an exhaustive certification of every legacy experiment.

## Stage B — smallest corrections

Implemented shared structured prompt encoding with overflow rejection; explicit action/label errors; actual paraphrase text; strict normalized exact match; embedding-identity decay exclusions; actual best-strategy label; consistent threshold semantics; initialized transaction error state, rejection of unimplemented structural operations and non-finite metrics. No learned-state tensor shape or architecture is changed. Raw byte encoding and existing checkpoint formats remain unchanged. M104 full prompts now fail visibly rather than train on missing evidence. This deliberately invalidates the legacy smoke benchmark until its input protocol is redesigned.

Regression tests in `tests/test_architecture_audit_contracts.py` use pure text/tiny fixtures and fake methods/parameters. They perform no model forward, neural training, optimizer update, checkpoint loading or GPU operations. Reproduce failures before fixes, then run only this inspected file locally. Syntax/lint apply to changed files. Full regression and neural tests are remote only. Legacy test files named “fast” or “CPU smoke” contain actual training and are not local-safe.

Risk: strict exact matching lowers previously inflated scores; embedding optimizer changes alter future trajectories; corrected prompts/labels invalidate direct comparison to historical neural scores; overflow rejection exposes broken old benchmark calls; paraphrase outputs become different measurements. Rollback is a Git revert of this focused patch; no prior checkpoint is overwritten. Old reports keep their original values and require this addendum when cited.

## Stage C — foundational experiment sequence

The prepared `scripts/run_foundational_audit.py` is a **single-seed feasibility diagnostic**, not a full language or causal competence claim. It keeps the existing student and compares selective, fully trainable and frozen backbones, using question mean pooling to avoid adding ECA complexity. Each arm starts from the same seed, uses zero decay and the same complete short inputs. It logs nonzero gradient names, optimizer coverage, fixed-threshold attribute F1/set match, causal state predictions, checkpoint hashes and measured GPU wall time.

First train four balanced material/action cases on attribute extraction only, bounded at 200 updates by default. If tiny F1 < 0.95, stop that arm before generalization. Then continue on independent named examples with balanced crossed material/action outcomes; evaluate previously unseen names and phrasing without tuning on dev/sealed labels. A mismatched-description ablation checks text relevance. This only tests a narrow supervised lexical/concept association; shared material words are intentional, and “sealed” here is a fixed diagnostic split, not a future research holdout. Any viewed split must be retired from later sealed qualification.

Required next controlled experiments, contingent on feasibility:

1. Fix a complete rule/context protocol on development worlds: either compact explicit rules preserving semantics or a separately tested context-extension/migration. Never silently discard descriptions. Add input-equivalence and boundary assertions to training/evaluation.
2. Diagnose extraction with equal-input lexical baseline, linear probe, frozen/selective/full plasticity and loss/threshold controls. Report support and negatives for every attribute; add no-property descriptions, negation and multiple entities. Only introduce tokenization or pooling changes after independent errors implicate them.
3. Qualify paired causal outcomes with changed rules and identical entity/action inputs, multi-step action order and input ablations. Define an explicit shared action/state vocabulary from permitted task specifications, matching heads, stable IDs and checkpoint migration; reject unseen labels until supported. Do not allocate extra dimensions from sealed answers.
4. Establish old-task competence above strongest controls before corrective feedback and retention. Actually execute replay/modular/reset arms and measure delayed old/new performance; remove placeholder copied metrics in a versioned successor benchmark.
5. Transfer across genuinely unfamiliar worlds with fresh, persistent, language-pretrained, memory-disabled and fixed-strategy controls, identical future minibatch streams, equal incremental budgets and separately reported prior costs. Use censored time-to-target without denominator floors and paired world/seed uncertainty. Stop if saturated at step one.
6. Only after single-seed feasibility and informative held-out performance, preregister multi-seed validation. No expensive sweep is authorized merely by software passing.

The feasibility diagnostic does not cover all six foundational capabilities. Unrun corrective learning, retention and future efficiency remain NOT QUALIFIED, even if simple extraction passes.

## Stage D — reversible learning state before policy research

Do not expand the M105 enum into open-ended operations yet. Consolidate onto a versioned live-student interface and one complete learning checkpoint, reusing production checkpoint machinery where appropriate. Snapshot fixed-topology changes first, including arbitrary masks and optimizer membership; reject unsupported topology mutations. Then support explicit topology migration and restore parameter aliases/groups correctly. Verify rollback by identical resumed updates in a fresh process, not just immediate logits. State publication must be atomic and external-memory/controller changes transactional.

After that contract passes remotely, implement actual bounded plasticity/replay/adapter experiments, matched policy action spaces, reset-isolated controls and independent future rewards. A learned controller must beat both fixed and random on future task outcomes with retention and cost constraints. Larger dimensions, depth, routing, vocabulary/context and multimodal modules stay eligible, but require controlled evidence and reversible migrations.

## Remote execution

Use `kaggle/launch_foundational_audit.py` in a private script kernel, with `__PINNED_SHA__` replaced by the full committed and pushed SHA before submission. The launcher verifies Kaggle plus CUDA, clones the branch, checks out the exact SHA, records provenance, runs inspected audit regressions and the diagnostic, and keeps stdout/stderr in working storage. Run the full suite remotely as a separately bounded phase after feasibility; failures remain failures, including legacy smoke assumptions invalidated by correct overflow errors. Do not call that historical suite locally.

Expected commands, on Kaggle only:

```bash
python -m pytest -q tests/test_architecture_audit_contracts.py
python scripts/run_foundational_audit.py --seed 42 --steps 200 --output-dir /kaggle/working/foundational-audit
python -m pytest -q --junitxml=/kaggle/working/full-tests.xml
```

The third command is a remote regression run, not a quality gate for neural intelligence. Artifacts must include failures, counts and actual source provenance. Read SCIENTIFIC_REVIEW_REPORT.md for actual execution status rather than inferring completion from this plan.

Local-safe job preparation after committing/pushing (does not train locally):

```bash
python - <<'PYCODE'
import json, subprocess
from pathlib import Path
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
job = Path("runs/kaggle-foundational-audit/job")
job.mkdir(parents=True, exist_ok=True)
job.joinpath("launch.py").write_text(Path("kaggle/launch_foundational_audit.py").read_text().replace("__PINNED_SHA__", sha))
job.joinpath("kernel-metadata.json").write_text(json.dumps(dict(
    id="sparklinghydraulics0/amg-foundational-audit", title="amg-foundational-audit",
    code_file="launch.py", language="python", kernel_type="script",
    is_private="true", enable_gpu="true", enable_tpu="false", enable_internet="true",
    dataset_sources=[], competition_sources=[], kernel_sources=[], model_sources=[]), indent=2))
PYCODE
kaggle kernels push -p runs/kaggle-foundational-audit/job
kaggle kernels status sparklinghydraulics0/amg-foundational-audit
# After COMPLETE or ERROR, preserve outputs and logs; ERROR is not a pass:
kaggle kernels output sparklinghydraulics0/amg-foundational-audit -p runs/kaggle-foundational-audit/output
```

## Decisions after the first remote feasibility result

Version 1 demonstrated perfect training fit and poor description generalization; do not advance to multi-seed experiments, lifelong claims or controller qualification. A fully trainable backbone's held-out 75% attribute F1 is neither robust causal competence nor evidence that a larger model is needed. Factor the next diagnostic into (a) new names with familiar syntax, (b) new syntax with familiar names, (c) both, and (d) controlled distractors/negation; current held-out rows change names and syntax together. Match lexical and shallow neural controls on exactly the delivered text. Compare a small mixture of training templates against the single-template curriculum before adding language corpora or semantic modules. Keep new evaluation vocabularies/worlds sealed from those design choices.

The full regression uncovered justified compatibility repairs beyond the initial M104 focus. Reuse historical MiniGPT generation/inspection methods, restore already-shifted target semantics and both supported mask sentinels, repair actual MoE/MoD key plumbing, and fix rotary frequency pairing. These repairs are covered by existing failing regression tests and two new remote-only neural contract tests. They do not change the M103/M104 question-classifier's dense absolute-position computation, so the negative feasibility finding remains applicable. Version 2 reruns actual remote validation rather than inferring a pass from these edits. A passing regression suite would establish engineering compatibility only.


## Approved audit repair continuation

The approved continuation implements only confirmed boundary, masking, loss and ownership repairs plus M90 reproducibility. [AUDIT_REPAIR_PHASE](AUDIT_REPAIR_PHASE.md) records actual execution and the remaining legacy limitations. Further components or capacity must wait for independently demonstrated learning; no new milestone number is assigned.
