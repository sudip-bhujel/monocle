from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from monocle.metrics import dependence_metrics, jeffreys_rate


def bootstrap_dependence(
    matrix: pd.DataFrame,
    *,
    draws: int = 200,
    seed: int = 0,
    stratify_by: str | None = "stratum_id",
) -> pd.DataFrame:
    return bootstrap_metrics(
        matrix,
        _dependence_metric_values,
        draws=draws,
        seed=seed,
        stratify_by=stratify_by,
    )


def bootstrap_metrics(
    matrix: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], dict[str, float | None]],
    *,
    draws: int = 200,
    seed: int = 0,
    stratify_by: str | None = "stratum_id",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    task_groups = _bootstrap_task_groups(matrix, stratify_by)
    task_index = _task_run_index(matrix)
    rows = []
    for draw in range(draws):
        sampled = _bootstrap_sample(matrix, task_index, task_groups, rng)
        rows.append({"draw": draw, **metric_fn(sampled)})
    return pd.DataFrame(rows)


def _bootstrap_task_groups(
    matrix: pd.DataFrame, stratify_by: str | None
) -> list[np.ndarray]:
    if stratify_by is None or stratify_by not in matrix.columns:
        tasks = matrix[["base_task_id"]].drop_duplicates()["base_task_id"].to_numpy()
        return [tasks]

    assignments = matrix[["base_task_id", stratify_by]].drop_duplicates()
    if assignments["base_task_id"].duplicated().any():
        raise ValueError(
            f"base tasks must belong to exactly one {stratify_by} for stratified bootstrap"
        )
    return [
        group["base_task_id"].to_numpy()
        for _, group in assignments.groupby(stratify_by, sort=True)
    ]


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


def _task_run_index(
    matrix: pd.DataFrame,
) -> dict[object, list[tuple[object, np.ndarray, dict[object, np.ndarray]]]]:
    """Precompute per-task case/run row locations for fast bootstrap draws.

    For each base_task_id, cases appear in first-seen order within that task, and
    each case stores run_index values in first-seen order with original row labels.
    """
    index: dict[object, list[tuple[object, np.ndarray, dict[object, np.ndarray]]]] = {}
    for task_id, task_df in matrix.groupby("base_task_id", sort=False):
        cases: list[tuple[object, np.ndarray, dict[object, np.ndarray]]] = []
        for case_id, case_df in task_df.groupby("case_id", sort=False):
            runs = case_df["run_index"].drop_duplicates().to_numpy()
            run_to_rows = {
                run: case_df.index[case_df["run_index"] == run].to_numpy()
                for run in runs
            }
            cases.append((case_id, runs, run_to_rows))
        index[task_id] = cases
    return index


def _bootstrap_sample(
    matrix: pd.DataFrame,
    task_index: dict[
        object, list[tuple[object, np.ndarray, dict[object, np.ndarray]]]
    ],
    task_groups: list[np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    sampled_tasks: list[tuple[int, object]] = []
    sample_index = 0
    for tasks in task_groups:
        chosen = rng.choice(tasks, size=len(tasks), replace=True)
        for task_id in chosen:
            sampled_tasks.append((sample_index, task_id))
            sample_index += 1

    # Match pandas groupby(["case_id", "bootstrap_task_id"], sort=True) RNG order.
    groups: list[
        tuple[object, str, np.ndarray, dict[object, np.ndarray]]
    ] = []
    for sample_index, task_id in sampled_tasks:
        bootstrap_task_id = f"{sample_index}:{task_id}"
        for case_id, runs, run_to_rows in task_index[task_id]:
            groups.append((case_id, bootstrap_task_id, runs, run_to_rows))
    groups.sort(key=lambda item: (item[0], item[1]))

    row_chunks: list[np.ndarray] = []
    bootstrap_task_ids: list[np.ndarray] = []
    bootstrap_run_ids: list[np.ndarray] = []
    for _, bootstrap_task_id, runs, run_to_rows in groups:
        sampled_runs = rng.choice(runs, size=len(runs), replace=True)
        for bootstrap_run_id, run in enumerate(sampled_runs):
            rows = run_to_rows[run]
            row_chunks.append(rows)
            n = len(rows)
            bootstrap_task_ids.append(np.full(n, bootstrap_task_id, dtype=object))
            bootstrap_run_ids.append(np.full(n, bootstrap_run_id, dtype=np.int64))

    if not row_chunks:
        out = matrix.iloc[0:0].copy()
        out["bootstrap_task_id"] = pd.Series(dtype=object)
        out["bootstrap_run_id"] = pd.Series(dtype=np.int64)
        return out

    out = matrix.loc[np.concatenate(row_chunks)].copy()
    out["bootstrap_task_id"] = np.concatenate(bootstrap_task_ids)
    out["bootstrap_run_id"] = np.concatenate(bootstrap_run_ids)
    return out.reset_index(drop=True)


def _dependence_metric_values(matrix: pd.DataFrame) -> dict[str, float | None]:
    metrics = dependence_metrics(matrix)
    return {
        "R_obs": metrics.R_obs,
        "R_ind": metrics.R_ind,
        "Gamma": metrics.Gamma,
        "CMS": metrics.CMS,
        "N_eff_risk": metrics.N_eff_risk,
    }
