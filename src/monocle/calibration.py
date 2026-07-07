from __future__ import annotations

import math

import pandas as pd


def _component_threshold(scores: list[float], fpr_target: float) -> float:
    ordered = sorted(scores)
    n = len(ordered)
    allowed_fp = math.floor(fpr_target * n)
    if allowed_fp <= 0:
        return float(ordered[-1])
    distinct = sorted(set(ordered))
    if len(distinct) == 1:
        return float(distinct[0])

    candidates = [
        (distinct[i - 1] + distinct[i]) / 2 for i in range(1, len(distinct))
    ]
    candidates.append(distinct[0] - (distinct[1] - distinct[0]) / 2)
    for threshold in sorted(candidates):
        if sum(score > threshold for score in ordered) <= allowed_fp:
            return float(threshold)
    return float(ordered[-1])


def component_thresholds(
    decisions: pd.DataFrame,
    cases: pd.DataFrame,
    *,
    fpr_target: float,
    threshold_id: str = "component-fpr",
    config_hash: str = "fixture",
) -> pd.DataFrame:
    if not 0 <= fpr_target <= 1:
        raise ValueError("fpr_target must be in [0, 1]")
    calibrate_safe = cases[(cases["split"] == "calibrate") & (cases["label"] == "safe")]
    if calibrate_safe.empty:
        raise ValueError("calibration requires safe cases in the calibrate split")
    allowed_case_ids = set(calibrate_safe["case_id"])
    final_case_ids = set(cases[cases["split"] == "final"]["case_id"])
    used_final = set(decisions["case_id"]) & final_case_ids & allowed_case_ids
    if used_final:
        raise ValueError("final cases cannot be used for threshold calibration")
    safe_scores = decisions[decisions["case_id"].isin(allowed_case_ids)]
    rows = []
    for monitor_id, group in safe_scores.groupby("monitor_id"):
        threshold = _component_threshold(
            [float(s) for s in group["raw_score"]], fpr_target
        )
        rows.append(
            {
                "threshold_id": threshold_id,
                "calibration_split": "calibrate",
                "fpr_target": fpr_target,
                "scope": "component",
                "score_transform": "identity",
                "threshold": threshold,
                "config_hash": config_hash,
                "monitor_id": monitor_id,
                "rule_id": None,
            }
        )
    return pd.DataFrame(rows)


def committee_fpr(
    decisions: pd.DataFrame, cases: pd.DataFrame, thresholds: pd.DataFrame
) -> float:
    safe_case_ids = set(cases[cases["label"] == "safe"]["case_id"])
    safe = decisions[decisions["case_id"].isin(safe_case_ids)]
    if safe.empty:
        return 0.0
    threshold_map = threshold_map_by_monitor(thresholds)
    flags = safe.assign(
        caught=safe.apply(
            lambda row: row["raw_score"] > threshold_map[row["monitor_id"]], axis=1
        )
    )
    per_case = flags.groupby("case_id")["caught"].any()
    return float(per_case.mean())


def cost_latency_summary(decisions: pd.DataFrame) -> dict[str, float]:
    return {
        "cost": float(decisions["cost"].sum()) if "cost" in decisions else 0.0,
        "latency_s": float(decisions["latency_s"].sum())
        if "latency_s" in decisions
        else 0.0,
    }


def threshold_map_by_monitor(thresholds: pd.DataFrame) -> dict[str, float]:
    if thresholds["monitor_id"].duplicated().any():
        raise ValueError(
            "thresholds must contain exactly one row per monitor; filter to one threshold_id first"
        )
    return thresholds.set_index("monitor_id")["threshold"].to_dict()
