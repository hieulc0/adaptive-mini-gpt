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
