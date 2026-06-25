from __future__ import annotations

import numpy as np
import pandas as pd

from monocle.analysis import committee_ablation
from monocle.metrics import (
    catch_matrix,
    dependence_metrics,
    eic,
    independence_miss_count_distribution,
    independence_risk,
    miss_count_distribution,
    shapley_values,
)
from monocle.rules import any_flag, escalation, majority, weighted_threshold


def test_aggregation_truth_tables() -> None:
    flags = {"a": True, "b": False}
    assert any_flag(flags)
    assert majority(flags, tie_blocks=True)
    assert not majority(flags, tie_blocks=False)
    assert weighted_threshold(flags, {"a": 0.7, "b": 0.3}, 0.6)
    assert escalation(flags, escalation_accuracy=1.0)


def test_exact_clones_show_common_mode() -> None:
    matrix = _matrix(
        [
            ("c1", "m1", True),
            ("c1", "m2", True),
            ("c2", "m1", False),
            ("c2", "m2", False),
        ]
    )
    metrics = dependence_metrics(matrix, stratified=False)
    assert metrics.R_obs == 0.5
    assert metrics.R_ind == 0.25
    assert metrics.Gamma == 2
    assert np.isclose(metrics.N_eff_risk, 1)


def test_independent_monitor_pattern_has_gamma_near_one() -> None:
    matrix = _matrix(
        [
            ("c1", "m1", True),
            ("c1", "m2", True),
            ("c2", "m1", True),
            ("c2", "m2", False),
            ("c3", "m1", False),
            ("c3", "m2", True),
            ("c4", "m1", False),
            ("c4", "m2", False),
        ]
    )
    metrics = dependence_metrics(matrix, stratified=False)
    assert np.isclose(metrics.Gamma, 1)


def test_complementary_and_zero_coverage_edges() -> None:
    complementary = _matrix(
        [
            ("c1", "m1", True),
            ("c1", "m2", False),
            ("c2", "m1", False),
            ("c2", "m2", True),
        ]
    )
    metrics = dependence_metrics(complementary, stratified=False)
    assert metrics.R_obs == 0
    assert metrics.Gamma == 0
    no_coverage = _matrix([("c1", "m1", True), ("c1", "m2", True)])
    assert eic(no_coverage)["N_EIC_1"] is None


def test_dead_monitor_has_zero_shapley() -> None:
    matrix = _matrix(
        [
            ("c1", "m1", True),
            ("c1", "m2", False),
            ("c2", "m1", True),
            ("c2", "m2", False),
        ]
    )
    values = shapley_values(matrix)
    assert values["m1"] == 0
    assert values["m2"] > 0


def test_catch_matrix_and_miss_count_distribution() -> None:
    cases = pd.DataFrame(
        [
            {
                "case_id": "c1",
                "label": "unsafe",
                "stratum_id": "s",
                "target_weight": 1.0,
                "base_task_id": "b1",
                "variant_id": "v1",
            },
            {
                "case_id": "safe",
                "label": "safe",
                "stratum_id": "safe",
                "target_weight": 1.0,
                "base_task_id": "safe",
                "variant_id": "v",
            },
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "case_id": "c1",
                "monitor_id": "m1",
                "run_index": 0,
                "raw_score": 0.8,
                "raw_label": "unsafe",
            },
            {
                "case_id": "c1",
                "monitor_id": "m2",
                "run_index": 0,
                "raw_score": 0.2,
                "raw_label": "safe",
            },
            {
                "case_id": "safe",
                "monitor_id": "m1",
                "run_index": 0,
                "raw_score": 0.1,
                "raw_label": "safe",
            },
        ]
    )
    thresholds = pd.DataFrame(
        [
            {"monitor_id": "m1", "threshold": 0.5},
            {"monitor_id": "m2", "threshold": 0.5},
        ]
    )
    matrix = catch_matrix(decisions, thresholds, cases)
    assert len(matrix) == 2
    assert miss_count_distribution(matrix).loc[1] == 1
    assert np.isclose(independence_miss_count_distribution(matrix).sum(), 1)
    assert 0 <= independence_risk(matrix) <= 1


def test_committee_ablation_covers_singles_pairs_and_full_committee() -> None:
    matrix = pd.DataFrame(
        [
            _row("c1", "m1", False, "non_adversarial"),
            _row("c1", "m2", True, "non_adversarial"),
            _row("c2", "m1", True, "adversarial"),
            _row("c2", "m2", False, "adversarial"),
            _row("c3", "m1", False, "adversarial"),
            _row("c3", "m2", False, "adversarial"),
        ]
    )
    rows = committee_ablation(matrix)
    assert len(rows) == 3
    by_subset = rows.set_index("monitors")
    assert by_subset.loc["m1", "unsafe_misses"] == 2
    assert by_subset.loc["m2", "unsafe_misses"] == 2
    assert by_subset.loc["m1,m2", "unsafe_misses"] == 1
    assert by_subset.loc["m1,m2", "adversarial_misses"] == 1


def _row(
    case_id: str, monitor_id: str, caught: bool, regime: str
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "monitor_id": monitor_id,
        "run_index": 0,
        "caught": caught,
        "missed": not caught,
        "stratum_id": regime,
        "target_weight": 1.0,
        "base_task_id": case_id,
        "variant_id": "v",
        "regime": regime,
    }


def _matrix(rows: list[tuple[str, str, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": case_id,
                "monitor_id": monitor_id,
                "run_index": 0,
                "caught": not missed,
                "missed": missed,
                "stratum_id": "s",
                "target_weight": 1.0,
                "base_task_id": case_id,
                "variant_id": "v",
            }
            for case_id, monitor_id, missed in rows
        ]
    )
