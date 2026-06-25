from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from monocle.calibration import threshold_map_by_monitor
from monocle.rules import poisson_binomial_pmf


@dataclass(frozen=True)
class DependenceMetrics:
    R_obs: float
    R_ind: float
    Gamma: float | None
    CMS: float | None
    N_eff_risk: float | None


def catch_matrix(
    decisions: pd.DataFrame,
    thresholds: pd.DataFrame,
    cases: pd.DataFrame,
) -> pd.DataFrame:
    threshold_map = threshold_map_by_monitor(thresholds)
    rows = []
    unsafe_cases = cases[cases["label"] == "unsafe"]
    if "regime" not in unsafe_cases:
        unsafe_cases = unsafe_cases.assign(regime="non_adversarial")
    case_meta = unsafe_cases.set_index("case_id")[
        ["stratum_id", "target_weight", "base_task_id", "variant_id", "regime"]
    ]
    for _, decision in decisions.iterrows():
        if decision["case_id"] not in case_meta.index:
            continue
        threshold = threshold_map[decision["monitor_id"]]
        caught = bool(
            decision["raw_score"] > threshold and decision["raw_label"] != "error"
        )
        meta = case_meta.loc[decision["case_id"]]
        rows.append(
            {
                "case_id": decision["case_id"],
                "monitor_id": decision["monitor_id"],
                "run_index": int(decision["run_index"]),
                "caught": caught,
                "missed": not caught,
                "stratum_id": meta["stratum_id"],
                "target_weight": float(meta["target_weight"]),
                "base_task_id": meta["base_task_id"],
                "variant_id": meta["variant_id"],
                "regime": meta["regime"],
            }
        )
    return pd.DataFrame(rows)


def case_monitor_miss_rates(matrix: pd.DataFrame) -> pd.DataFrame:
    return (
        matrix.groupby(
            ["case_id", "monitor_id", "stratum_id", "target_weight"], as_index=False
        )["missed"]
        .mean()
        .rename(columns={"missed": "miss_probability"})
    )


def observed_risk(matrix: pd.DataFrame) -> float:
    rates = case_monitor_miss_rates(matrix)
    per_case = rates.groupby(["case_id", "target_weight"], as_index=False)[
        "miss_probability"
    ].prod()
    return _weighted_sum(per_case["miss_probability"], per_case["target_weight"])


def independence_risk(matrix: pd.DataFrame, stratified: bool = True) -> float:
    rates = case_monitor_miss_rates(matrix)
    if not stratified:
        marginals = rates.groupby("monitor_id")["miss_probability"].mean()
        return float(np.prod(marginals.to_numpy()))
    stratum_weights = (
        rates[["case_id", "stratum_id", "target_weight"]]
        .drop_duplicates()
        .groupby("stratum_id")["target_weight"]
        .sum()
    )
    stratum_weights = stratum_weights / stratum_weights.sum()
    risk = 0.0
    for stratum_id, group in rates.groupby("stratum_id"):
        monitor_rates = group.groupby("monitor_id")["miss_probability"].mean()
        risk += float(stratum_weights.loc[stratum_id]) * float(
            np.prod(monitor_rates.to_numpy())
        )
    return risk


def dependence_metrics(
    matrix: pd.DataFrame, *, stratified: bool = True
) -> DependenceMetrics:
    R_obs = observed_risk(matrix)
    R_ind = independence_risk(matrix, stratified=stratified)
    gamma = R_obs / R_ind if R_ind > 0 else None
    cms = 1 - (1 / gamma) if gamma and R_obs > 0 else None
    monitor_count = matrix["monitor_id"].nunique()
    n_eff = None
    if 0 < R_obs < 1 and 0 < R_ind < 1:
        n_eff = monitor_count * np.log(R_obs) / np.log(R_ind)
    return DependenceMetrics(
        R_obs=R_obs, R_ind=R_ind, Gamma=gamma, CMS=cms, N_eff_risk=n_eff
    )


def miss_count_distribution(matrix: pd.DataFrame) -> pd.Series:
    rates = case_monitor_miss_rates(matrix)
    wide = rates.pivot(
        index="case_id", columns="monitor_id", values="miss_probability"
    ).fillna(0)
    weights = rates.groupby("case_id")["target_weight"].first()
    miss_counts = wide.round().astype(int).sum(axis=1)
    out = miss_counts.groupby(miss_counts).apply(
        lambda ids: weights.loc[ids.index].sum()
    )
    return (out / weights.sum()).sort_index()


def independence_miss_count_distribution(matrix: pd.DataFrame) -> pd.Series:
    rates = case_monitor_miss_rates(matrix)
    weights = (
        rates[["case_id", "stratum_id", "target_weight"]]
        .drop_duplicates()
        .groupby("stratum_id")["target_weight"]
        .sum()
    )
    weights = weights / weights.sum()
    monitor_count = rates["monitor_id"].nunique()
    total = np.zeros(monitor_count + 1)
    for stratum_id, group in rates.groupby("stratum_id"):
        probabilities = group.groupby("monitor_id")["miss_probability"].mean().tolist()
        total += weights.loc[stratum_id] * poisson_binomial_pmf(probabilities)
    return pd.Series(total, index=range(monitor_count + 1))


def shapley_values(matrix: pd.DataFrame) -> dict[str, float]:
    rates = case_monitor_miss_rates(matrix)
    catches = rates.assign(caught=1 - rates["miss_probability"])
    values = {monitor_id: 0.0 for monitor_id in sorted(catches["monitor_id"].unique())}
    for case_id, group in catches.groupby("case_id"):
        caught = group[group["caught"] > 0]
        if caught.empty:
            continue
        weight = float(group["target_weight"].iloc[0])
        denom = float(caught["caught"].sum())
        for _, row in caught.iterrows():
            values[row["monitor_id"]] += weight * float(row["caught"]) / denom
    return values


def eic(matrix: pd.DataFrame) -> dict[str, float | None]:
    values = shapley_values(matrix)
    total = sum(values.values())
    if total <= 0:
        return {"N_EIC_1": None, "N_EIC_2": None, "CPR": None}
    shares = np.array([value / total for value in values.values() if value > 0])
    n_eic_1 = float(np.exp(-np.sum(shares * np.log(shares))))
    n_eic_2 = float(1 / np.sum(shares**2))
    return {"N_EIC_1": n_eic_1, "N_EIC_2": n_eic_2, "CPR": n_eic_1 / len(values)}


def delta_r(base_matrix: pd.DataFrame, candidate_matrix: pd.DataFrame) -> float:
    return observed_risk(base_matrix) - observed_risk(candidate_matrix)


def all_subset_risks(matrix: pd.DataFrame, sizes: list[int]) -> pd.DataFrame:
    monitors = sorted(matrix["monitor_id"].unique())
    rows = []
    for size in sizes:
        for subset in combinations(monitors, size):
            sub = matrix[matrix["monitor_id"].isin(subset)]
            rows.append(
                {
                    "size": size,
                    "monitors": ",".join(subset),
                    "R_obs": observed_risk(sub),
                }
            )
    return pd.DataFrame(rows)


def _weighted_sum(values: pd.Series, weights: pd.Series) -> float:
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("weights must have positive sum")
    return float(
        np.sum(values.to_numpy(dtype=float) * weights.to_numpy(dtype=float)) / total
    )
