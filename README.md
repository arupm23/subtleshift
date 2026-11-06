# SubtleShift

[Visit the research website](https://arupm23.github.io/subtleshift/)

SubtleShift is a reproducible starter codebase for replicating and extending
emergent-misalignment experiments with open-weight language models. It separates a
dependency-free smoke path from GPU training and judge-backed evaluation, so a successful local
test is never confused with a scientific result.

## Research proposal

**SubtleShift: Controlled Replication of Training-Induced Behavioral Generalization in
Open-Weight Language Models**

The first study asks whether narrow supervised fine-tuning produces measurable behavioral shifts
outside the training domain, and whether those shifts survive matched controls. The design improves
on a single treatment-versus-base comparison with:

- a frozen base-model baseline;
- a benign matched-domain control;
- a neutral style/format control;
- repeated seeds and prompt-order randomization;
- blinded, versioned judging plus a cheap heuristic used only for plumbing tests;
- per-prompt raw generations, run manifests, configuration hashes, and confidence intervals;
- a capability-retention gate and explicit separation of confirmatory and exploratory prompts.

This starter intentionally does **not** automate the search for monitor-evading training data. It is
for controlled defensive research using reviewed datasets and isolated model artifacts.

## Experiment matrix

| Arm | Adapter | Training data | Purpose |
|---|---:|---|---|
| Base | No | None | Pre-fine-tuning baseline |
| Benign control | Yes | `data/controls/benign_helpfulness.jsonl` | Controls for SFT and domain exposure |
| Neutral control | Yes | `data/controls/neutral_style.jsonl` | Controls for format/style learning |
| Treatment | Yes | Researcher-supplied, reviewed dataset | Tests out-of-domain behavioral shift |

The included datasets are tiny fixtures for validation, not adequate training corpora. Do not infer
anything about model safety from the bundled smoke results.

## Quick start (no model download)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest
subtleshift smoke --config configs/eval/smoke.yaml
subtleshift train --config configs/train/lora_qwen25_05b.yaml --dry-run
```

Each run writes an immutable manifest, raw records, and summary metrics beneath `artifacts/`.

## Open-weight baseline evaluation

Install the GPU dependencies, then run the base model without an adapter:

```bash
pip install -r requirements-gpu.txt
pip install -e .
subtleshift evaluate --config configs/eval/qwen25_05b_baseline.yaml
```

The default open model is `Qwen/Qwen2.5-0.5B-Instruct`, chosen to make pipeline validation
accessible. It is not expected to reproduce results reported for much larger models. Change the
model revision only by editing a tracked config, and record the resolved model commit in the run
notes before publishing results.

## LoRA training

Validate first, then remove `--dry-run` on a CUDA-capable machine:

```bash
subtleshift train --config configs/train/lora_qwen25_05b.yaml --dry-run
subtleshift train --config configs/train/lora_qwen25_05b.yaml
```

Create one config per arm and seed. Evaluate adapters with the same frozen prompt suite, decoding
settings, and judge version. The training command uses TRL + PEFT and saves adapters rather than a
merged model.

## Evaluation policy

`heuristic_v0` is deterministic and useful only for testing the pipeline. It searches for a small,
declared set of protective and concerning phrases; it is not a validated alignment judge. A paper
replication should stage the published evaluation questions under their original license and add a
blinded model judge or human annotation protocol. Preserve raw responses and report agreement,
judge sensitivity, invalid-response rates, and uncertainty—not only a headline average.

The bundled prompt suite uses original, low-risk probes rather than redistributing another
project's benchmark. `group` and `split` fields support stratified reporting and held-out tests.

## Reproducibility checklist

1. Pre-register hypotheses, primary metric, exclusion rules, seeds, and stopping rule.
2. Pin model and dataset revisions; retain hashes and licenses.
3. Train base/control/treatment arms with identical budgets.
4. Run at least three seeds and preserve every generation.
5. Blind judges to arm names and randomize response order.
6. Report bootstrap or Wilson intervals and paired per-prompt effects.
7. Check instruction following/capability retention before interpreting safety metrics.
8. Treat exploratory prompt tuning and monitor tuning as exploratory, not confirmatory.

## Repository layout

```text
configs/          tracked evaluation and LoRA configurations
data/             tiny controls and evaluation fixtures
src/subtleshift/  configuration, generation, evaluation, logging, and training code
tests/            deterministic unit and integration smoke tests
artifacts/        generated run records (gitignored)
docs/             research protocol and GitHub Pages website
```

## Research website

The static research website is published from `main` → `/docs` with GitHub Pages.
Edit `docs/index.html`, `docs/styles.css`, and `docs/app.js` to update it; no build,
model download, API key, or backend service is required. Preview it locally with
`python -m http.server 4173 --directory docs`.

The interactive charts are illustrative study designs, not experimental results.
GPU training, model-backed baseline evaluation, and blinded judging remain pending.

## Publication privacy

The initial publication review inspected all 14 existing commits and 35 unique file
versions for credentials, private keys, personal email addresses, local user paths,
credential-bearing URLs, and private network addresses; no matches were found.
Commit author emails use GitHub's privacy-preserving addresses. No history rewrite
was needed. This review is a point-in-time check, not a guarantee about future files.

Local working files, environment files, generated run logs, private datasets, and
model checkpoints are excluded from Git. Keep tokens in environment variables and
review staged files before publishing new data or experiment outputs.

## Safety and scope

Use isolated machines and private model artifacts. Review training rows before use; do not deploy
fine-tuned research models or publish adapters that exhibit unsafe behavior. Keep evaluations
non-agentic unless the environment has explicit containment and no real credentials, networks, or
external side effects. This repository does not include harmful training corpora, autonomous
adversarial dataset generation, monitor-evasion optimization, or weight-exfiltration scenarios.

## References

- Anthropic Alignment Science, “Automated Subtle Emergent Misalignment Research” (2026).
- Betley et al., “Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs”
  (2025), arXiv:2502.17424.

See `docs/PROTOCOL.md` for the proposed execution sequence and decision rules.

