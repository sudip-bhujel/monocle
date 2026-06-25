from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import pandas as pd

from monocle.metrics import dependence_metrics


def jeffreys_rate(successes: int, trials: int) -> float:
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("invalid Bernoulli counts")
    return (successes + 0.5) / (trials + 1.0)


def bootstrap_dependence(
    matrix: pd.DataFrame,
    *,
    draws: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    return bootstrap_metrics(matrix, _dependence_metric_values, draws=draws, seed=seed)


def bootstrap_metrics(
    matrix: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], dict[str, float | None]],
    *,
    draws: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tasks = matrix[["base_task_id"]].drop_duplicates()["base_task_id"].to_numpy()
    rows = []
    for draw in range(draws):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        parts = []
        for sample_index, task_id in enumerate(sampled_tasks):
            task_rows = matrix[matrix["base_task_id"] == task_id].copy()
            task_rows["bootstrap_task_id"] = f"{sample_index}:{task_id}"
            parts.append(task_rows)
        sampled = pd.concat(parts, ignore_index=True)
        sampled = _resample_runs(sampled, rng)
        rows.append({"draw": draw, **metric_fn(sampled)})
    return pd.DataFrame(rows)


def confidence_interval(values: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return (float("nan"), float("nan"))
    return (
        float(clean.quantile(alpha / 2)),
        float(clean.quantile(1 - alpha / 2)),
    )


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    previous = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        value = min(1.0, (count - rank) * p_value)
        previous = max(previous, value)
        adjusted[name] = previous
    return adjusted


@dataclass(frozen=True)
class DecisionLabel:
    status: str
    reason: str


def classify_lower_bound(lower: float, criterion: float) -> DecisionLabel:
    if lower > criterion:
        return DecisionLabel("supported", "lower bound exceeds criterion")
    if lower < -criterion:
        return DecisionLabel("contradicted", "lower bound is opposite criterion")
    return DecisionLabel("inconclusive", "interval does not settle the criterion")


def equivalence_from_upper(upper: float, margin: float) -> DecisionLabel:
    if upper < margin:
        return DecisionLabel("supported", "upper bound is below equivalence margin")
    return DecisionLabel("inconclusive", "upper bound does not establish equivalence")


def synthetic_power(effect: float, sd: float, n: int, alpha: float = 0.05) -> float:
    if sd <= 0 or n <= 0:
        raise ValueError("sd and n must be positive")
    z = abs(effect) / (sd / np.sqrt(n))
    critical = 1.96 if alpha == 0.05 else 1.64
    return float(1 / (1 + np.exp(-(z - critical))))


def _resample_runs(matrix: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    group_cols = ["case_id", "monitor_id"]
    if "bootstrap_task_id" in matrix.columns:
        group_cols.append("bootstrap_task_id")
    for _, group in matrix.groupby(group_cols):
        runs = group["run_index"].to_numpy()
        sampled_runs = rng.choice(runs, size=len(runs), replace=True)
        parts.extend(group[group["run_index"] == run] for run in sampled_runs)
    return pd.concat(parts, ignore_index=True)


def _dependence_metric_values(matrix: pd.DataFrame) -> dict[str, float | None]:
    metrics = dependence_metrics(matrix)
    return {
        "R_obs": metrics.R_obs,
        "R_ind": metrics.R_ind,
        "Gamma": metrics.Gamma,
        "CMS": metrics.CMS,
        "N_eff_risk": metrics.N_eff_risk,
    }
