# Monocle Cases Dataset

This directory contains the active local case datasets for Monocle.

## Files

- `monocle_cases.jsonl`: main case dataset.
- `monocle_smoke.jsonl`: 4-row fixture for fast CLI and storage smoke tests.

Historical fixture variants are intentionally not kept here as active inputs.

## Intended Use

`monocle_cases.jsonl` is designed for studying correlated safety-monitor misses. It contains safe calibration rows and paired unsafe task variants so committees can be evaluated with `R_obs`, `R_ind`, `Gamma`, and related Monocle metrics.

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
