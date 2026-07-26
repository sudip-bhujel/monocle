# Monocle

Monocle measures common-mode failures in committees of LLM safety monitors. Run
the commands below from this directory with `OPENROUTER_API_KEY` set for hosted
model calls. Run artifacts are stored under `runs/`; derived JSON results are
stored under `results/`.

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

Analyze all committees, derive the paper statistics, and render the exhibits:

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
