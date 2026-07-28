# Monocle

Monocle measures common-mode failures in committees of LLM safety monitors. Run
the commands below from this directory with `OPENROUTER_API_KEY` set for hosted
model calls. Run artifacts are stored under `runs/`; derived JSON results are
stored under `results/`.

Install the locked environment with `uv sync`. The rating directory used below
must remain outside this source directory and should not be included in the
anonymous artifact.

## Blinded label audit

Prepare two independent ratings per case across five raters:

```bash
CASES=dataset/monocle/monocle.jsonl
RATER_ROOT=../monocle-rater-audit

uv run python scripts/label_audit_sample.py prepare-rotating \
  --d1-path "$CASES" \
  --blind-dir "$RATER_ROOT/raters" \
  --key "$RATER_ROOT/coordinator/monocle-rater-key.csv" \
  --raters 5
```

After the five worksheets are complete, calculate raw agreement and Gwet's AC1:

```bash
uv run python scripts/label_audit_sample.py score-rotating \
  --blind "$RATER_ROOT"/raters/label-audit-action-safety-rotating-rater-*.csv \
  --key "$RATER_ROOT/coordinator/monocle-rater-key.csv" \
  --require-complete \
  --consolidated "$RATER_ROOT/coordinator/monocle-rater-consolidated.csv" \
  --ledger "$RATER_ROOT/coordinator/monocle-rater-review.json"

jq '.between_rater' "$RATER_ROOT/coordinator/monocle-rater-review.json"
```

The review ledger lists cases requiring adjudication. A separate adjudicator
can import final decisions from a CSV containing
`case_id,adjudicator_id,adjudicated_label,disposition,rationale,decision_locked_at`:

```bash
uv run python scripts/label_audit_sample.py import-adjudication \
  --review-ledger "$RATER_ROOT/coordinator/monocle-rater-review.json" \
  --adjudication "$RATER_ROOT/coordinator/monocle-rater-adjudication.csv" \
  --output "$RATER_ROOT/coordinator/monocle-rater-adjudicated.json"
```

## Running the benchmark

Set the primary run paths:

```bash
RUN_ID=monocle
CASES=dataset/monocle/monocle.jsonl
```

Run each configured monitor three times per case:

```bash
uv run monocle run \
  --run-id "$RUN_ID" \
  --cases "$CASES" \
  --models configs/models-cross-family.yaml \
  --allow-hosted \
  --runs 3 \
  --workers 10
```

Calibrate monitor thresholds on the designated safe split:

```bash
uv run monocle calibrate \
  --run-id "$RUN_ID" \
  --cases "$CASES" \
  --experiment configs/experiment.yaml
```

Analyze all committees and derive the paper statistics:

```bash
uv run monocle analyze \
  --run-id "$RUN_ID" \
  --cases "$CASES" \
  --experiment configs/experiment.yaml \
  --committees configs/committees.yaml \
  --all-committees \
  --draws 2000 \
  --seed 20260710

uv run python scripts/action_safety_analysis.py \
  --run-dir "runs/$RUN_ID" \
  --cases "$CASES" \
  --committees configs/committees.yaml \
  --design pooled \
  --draws 2000 \
  --seed 20260710 \
  --output results/action-safety-monocle.json
```

Generate the complete statistics bundle and the paper tables and figures:

```bash
uv run python scripts/paper_artifacts.py \
  --run-id "$RUN_ID" \
  --cases "$CASES" \
  --design pooled \
  --threshold-id component-fpr-0p15 \
  --committees configs/committees.yaml \
  --draws 2000 \
  --seed 20260710 \
  --output results/confirmatory-monocle.json

uv run python scripts/render_paper_exhibits.py \
  --empirical results/confirmatory-monocle.json \
  --hypotheses results/action-safety-monocle.json \
  --paper-root ../Monocle
```

## Ablation experiments with frontier-scale models

This ablation repeats the benchmark with the frontier-model configuration while
keeping the cases, prompts, calibration procedure, and analysis seed fixed.

```bash
FRONTIER_RUN_ID=monocle-frontier
CASES=dataset/monocle/monocle.jsonl
```

```bash
uv run monocle run \
  --run-id "$FRONTIER_RUN_ID" \
  --cases "$CASES" \
  --models configs/models-frontier-cross-family.yaml \
  --allow-hosted \
  --runs 3 \
  --workers 4
```

```bash
uv run monocle calibrate \
  --run-id "$FRONTIER_RUN_ID" \
  --cases "$CASES" \
  --experiment configs/experiment.yaml
```

```bash
uv run monocle analyze \
  --run-id "$FRONTIER_RUN_ID" \
  --cases "$CASES" \
  --experiment configs/experiment.yaml \
  --committees configs/committees-frontier.yaml \
  --all-committees \
  --draws 2000 \
  --seed 20260710
```

Generate the selected supplementary tables and figures:

```bash
uv run python scripts/paper_artifacts.py \
  --run-id "$FRONTIER_RUN_ID" \
  --cases "$CASES" \
  --design pooled \
  --threshold-id component-fpr-0p15 \
  --committees configs/committees-frontier.yaml \
  --draws 2000 \
  --seed 20260710 \
  --output results/confirmatory-frontier-supplementary.json

uv run python scripts/render_paper_exhibits.py \
  --empirical results/confirmatory-frontier-supplementary.json \
  --paper-root ../Monocle/frontier-supplementary \
  --frontier-only
```
