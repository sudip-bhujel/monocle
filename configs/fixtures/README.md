# Monocle D1 Strong-Model Hardening Dataset

This directory contains the active local case datasets for Monocle.

## Files

- `monocle_cases.jsonl`: main D1 strong-model hardening case dataset.
- `monocle_smoke.jsonl`: 4-row fixture for fast CLI and storage smoke tests.

Historical fixture variants are intentionally not kept here as active inputs.

## Intended Use

`monocle_cases.jsonl` is designed for studying correlated safety-monitor misses in enterprise SaaS support, admin, audit, migration, consent, training, and operations workflows. It contains safe calibration rows and paired unsafe task variants so committees can be evaluated with `R_obs`, `R_ind`, `Gamma`, and related Monocle metrics.

This version hardens the unsafe set toward stronger-model evaluation by emphasizing timing, provenance, role scope, and routing-boundary failures rather than overtly malicious requests. Unsafe payloads describe the *same* boundary violation as routine engineering work (autosave, schema-normalize, in-place compaction, config-export-as-seed, soft-delete, shared validation role) so that a strong committee's blind spots surface as genuine common-mode misses rather than reactions to trigger vocabulary.

The calibrate split deliberately includes borderline-safe twins: genuinely safe tasks that reuse the risky-looking verbs of the unsafe families (scoped delete, token rotation, staging reset, provenance-preserving merge, scoped grant, complete deletion, consented autosave, source-preserving summary). Calibrating the false-positive thresholds against realistic borderline-safe traffic — instead of only trivially-safe read-only tasks — keeps the thresholds honest and prevents an artificially low catch bar.

This is a synthetic research dataset, not a general benchmark or a claim about real-world model safety.

## Dataset Structure

Each row is a JSON object that follows the current `monocle.schema.Case` model exactly. Extra fields are not allowed.

Required fields:

- `case_id`
- `base_task_id`
- `variant_id`
- `kind`
- `payload`
- `label`
- `regime`
- `attack_class`
- `knowledge`
- `difficulty_z`
- `stratum_id`
- `target_weight`
- `split`
- `oracle_id`
- `oracle_version`

Stable identifiers for future publication are `case_id`, `base_task_id`, `variant_id`, and `stratum_id`.

## Labels, Regimes, And Splits

Labels:

- `safe`: benign tasks used for threshold calibration.
- `unsafe`: inert task requests that represent unsafe policy intent.

Regimes:

- `safe`: calibration examples.
- `non_adversarial`: direct unsafe task framing.
- `adversarial`: benign-framed unsafe task variants.

Splits:

- `calibrate`: safe rows used to set monitor thresholds.
- `final`: unsafe rows reserved for downstream analysis.

## Safety Boundaries

Payloads are inert natural-language task descriptions. They are not runnable exploit recipes and should not be executed as instructions.

The dataset avoids real endpoints, real secrets, credential URLs, cloud metadata IPs, destructive command chains, and operational exploit steps. Placeholder paths use `/tmp/monocle-lab/...`.

## Limitations

The dataset is synthetic and intended to evolve. It should be treated as research data, not as a statistically mature benchmark.
