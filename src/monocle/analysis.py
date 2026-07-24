from __future__ import annotations

from itertools import combinations
from math import log

import pandas as pd

from monocle.metrics import (
    case_monitor_miss_rates,
    catch_matrix,
    dependence_metrics,
    eic,
    independence_miss_count_distribution,
    miss_count_distribution,
)


def summarize_h1(
    decisions: pd.DataFrame, thresholds: pd.DataFrame, cases: pd.DataFrame
) -> list[dict[str, float | None]]:
    matrix = catch_matrix(decisions, thresholds, cases)
    values = h1_metric_values(matrix)
    return [{"metric": key, "value": value} for key, value in values.items()]


def committee_ablation(
    matrix: pd.DataFrame, sizes: list[int] | None = None
) -> pd.DataFrame:
    monitors = sorted(matrix["monitor_id"].unique())
    if sizes is None:
        sizes = sorted(
            {size for size in [1, 2, len(monitors)] if size <= len(monitors)}
        )
    rows = []
    has_regime = "regime" in matrix.columns
    for size in sizes:
        for subset in combinations(monitors, size):
            submatrix = matrix[matrix["monitor_id"].isin(subset)].copy()
            values = h1_metric_values(submatrix)
            row = {
                "rule_id": "any_flag",
                "size": size,
                "monitors": ",".join(subset),
                **_committee_miss_counts(submatrix),
                "R_obs": values.get("R_obs"),
                "R_ind": values.get("R_ind"),
                "Gamma": values.get("Gamma"),
            }
            if has_regime:
                row.update(
                    {
                        "non_adversarial_R_obs": values.get(
                            "non_adversarial.R_obs"
                        ),
                        "non_adversarial_R_ind": values.get(
                            "non_adversarial.R_ind"
                        ),
                        "non_adversarial_Gamma": values.get(
                            "non_adversarial.Gamma"
                        ),
                        "adversarial_R_obs": values.get("adversarial.R_obs"),
                        "adversarial_R_ind": values.get("adversarial.R_ind"),
                        "adversarial_Gamma": values.get("adversarial.Gamma"),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def h1_metric_values(matrix: pd.DataFrame) -> dict[str, float | None]:
    values = _matrix_metric_values(matrix, prefix="")
    if "regime" not in matrix.columns:
        return values
    for regime in ["non_adversarial", "adversarial"]:
        regime_matrix = matrix[matrix["regime"] == regime]
        values.update(_matrix_metric_values(regime_matrix, prefix=f"{regime}."))
    non_adv_gamma = values.get("non_adversarial.Gamma")
    adv_gamma = values.get("adversarial.Gamma")
    values["logGamma_adv_minus_non_adversarial"] = _log_contrast(
        adv_gamma, non_adv_gamma
    )
    return values


def _matrix_metric_values(
    matrix: pd.DataFrame, *, prefix: str
) -> dict[str, float | None]:
    if matrix.empty:
        names = [
            "R_obs",
            "R_ind",
            "Gamma",
            "CMS",
            "N_eff_risk",
            "N_EIC_1",
            "N_EIC_2",
            "CPR",
        ]
        return {f"{prefix}{name}": None for name in names}
    rates = case_monitor_miss_rates(matrix)
    metrics = dependence_metrics(matrix, rates=rates)
    coverage = eic(matrix, rates=rates)
    values = {
        f"{prefix}R_obs": metrics.R_obs,
        f"{prefix}R_ind": metrics.R_ind,
        f"{prefix}Gamma": metrics.Gamma,
        f"{prefix}CMS": metrics.CMS,
        f"{prefix}N_eff_risk": metrics.N_eff_risk,
    }
    values.update({f"{prefix}{key}": value for key, value in coverage.items()})
    values.update(
        _distribution_values(f"{prefix}P_obs_K", miss_count_distribution(matrix))
    )
    values.update(
        _distribution_values(
            f"{prefix}P_ind_K",
            independence_miss_count_distribution(matrix, rates=rates),
        )
    )
    return values


def _committee_miss_counts(matrix: pd.DataFrame) -> dict[str, int]:
    if matrix.empty:
        counts = {
            "unsafe_trials": 0,
            "unsafe_misses": 0,
        }
        if "regime" in matrix.columns:
            counts.update(
                {"non_adversarial_misses": 0, "adversarial_misses": 0}
            )
        return counts
    grouped = matrix.copy()
    grouping = ["case_id", "run_index"]
    if "regime" in grouped.columns:
        grouping.append("regime")
    per_trial = (
        grouped.groupby(grouping, as_index=False)["caught"]
        .any()
        .assign(missed=lambda frame: ~frame["caught"])
    )
    counts = {
        "unsafe_trials": int(len(per_trial)),
        "unsafe_misses": int(per_trial["missed"].sum()),
    }
    if "regime" in grouped.columns:
        counts.update(
            {
                "non_adversarial_misses": int(
                    per_trial[
                        (per_trial["regime"] == "non_adversarial")
                        & per_trial["missed"]
                    ].shape[0]
                ),
                "adversarial_misses": int(
                    per_trial[
                        (per_trial["regime"] == "adversarial")
                        & per_trial["missed"]
                    ].shape[0]
                ),
            }
        )
    return counts


def _distribution_values(prefix: str, distribution: pd.Series) -> dict[str, float]:
    return {
        f"{prefix}_{int(index)}": float(value) for index, value in distribution.items()
    }


def _log_contrast(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or left <= 0 or right <= 0:
        return None
    return float(log(left) - log(right))
