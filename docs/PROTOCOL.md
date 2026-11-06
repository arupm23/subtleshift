# Experimental protocol

## Primary question

Relative to the base model and matched controls, does narrow fine-tuning cause a reproducible,
out-of-domain increase in blinded misalignment judgments while preserving instruction-following
capability?

## Phases

1. **Freeze the plan.** Register model revisions, datasets, seeds, training budget, decoding
   parameters, primary metric, capability gate, and exclusions.
2. **Validate plumbing.** Run the deterministic mock smoke test and LoRA dry-run. Confirm that
   config hashes, raw rows, summaries, and failure messages are captured.
3. **Measure baseline.** Generate base-model responses for every prompt and seed before training.
4. **Train controls.** Train benign-domain and neutral-style adapters with the treatment's exact
   optimizer, row count, token budget, and sampling schedule.
5. **Train treatment.** Use only an approved, versioned research dataset. Keep adapters private and
   do not merge them into the base checkpoint.
6. **Blind and judge.** Shuffle response-to-arm mappings, use a frozen judge rubric, and obtain a
   second judge or human audit on a stratified sample.
7. **Analyze.** Report paired per-prompt changes, arm-level rates, Wilson 95% intervals, seed
   dispersion, coherence/invalid rates, and capability retention.
8. **Stress-test conclusions.** Repeat with decoding perturbations and a second model family. Label
   all post-hoc prompts and thresholds as exploratory.

## Default decision rules

- The primary unit is the prompt-generation pair; report clustering by prompt and seed.
- A run is interpretable only if its capability-retention score remains above the pre-registered
  threshold.
- A treatment effect must exceed both base and both control arms, reproduce across seeds, and remain
  under at least one judge perturbation.
- The heuristic bundled in this repository can never satisfy the judge requirement.
- Missing or failed generations remain in the denominator and are reported separately.

## Recommended extension

After replication, mix the treatment data into a larger fixed benign corpus at pre-registered rates
(for example 0%, 0.1%, 0.5%, 1%, and 5%). This directly tests whether the effect survives a more
realistic data mixture. Keep the total token budget fixed and include mixture-matched benign
controls.

