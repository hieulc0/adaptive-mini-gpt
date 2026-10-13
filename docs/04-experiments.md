# Experiment discipline and metrics

## Required experiment records
- Run ID, code SHA, dataset SHA/provenance, architecture/config and tokenizer fingerprint
- Train steps/tokens, wall clock, device, batch/context, precision, optimizer settings
- Train loss, held-out validation loss, accuracy by task (if verifiable), uncertainty across seeds
- Peak allocated/reserved VRAM, tokens/s, estimated FLOPs if justified, total training GPU-hours
- Growth decisions, candidate/baseline comparison, accepted/rejected status and why

## Baselines
A: fixed dense small GPT; B: larger dense GPT; C: modular fixed experts; D: adaptive grown experts. Advance sequentially (A first). Compare matched **tokens, compute budget, dataset, evaluation split**, not just parameter count. Also show total parameters, active parameters, actual GPU residency, and inference speed.

## Verification tasks
- Character-level language: validation cross-entropy, bits-per-character/perplexity when appropriate, sample quality as a qualitative diagnostic.
- Arithmetic: deterministic targets, exact-match on distinct held-out operands and extrapolation splits.
- Logic: procedurally generated verified tasks with independent seeds.
- Continual learning: per-skill held-out scores before and after learning another skill; measure forgetting.

## Growth acceptance policy (future)
Stalling validation alone is insufficient. Confirm data quality, learning rate and adequate training; train a no-growth control with equivalent compute. Accept a growth candidate only when improvement on predefined validation metrics exceeds noise and prior skills don't regress beyond preset tolerance. Keep one untouched final holdout. Report negative results.

## Common pitfalls
Overfitting tiny datasets; accidental train/validation overlap; overfitting to the controller's benchmark; 'specialist' naming without expertise; sparse masks that don't reduce GPU compute; forgetting through shared weights/router; inference improvements mistaken for training progress; copying teacher-generated text with unclear rights.
