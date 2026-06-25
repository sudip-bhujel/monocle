from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from monocle.data import synthetic_cases
from monocle.schema import Case, RunManifest, Threshold
from monocle.store import RunStore


def test_case_validation_rejects_bad_split_and_negative_weight() -> None:
    base = synthetic_cases()[0].model_dump()
    base["split"] = "bad"
    with pytest.raises(ValidationError):
        Case.model_validate(base)
    base = synthetic_cases()[0].model_dump()
    base["target_weight"] = -1
    with pytest.raises(ValidationError):
        Case.model_validate(base)


def test_threshold_validation_rejects_non_calibration_split() -> None:
    with pytest.raises(ValidationError):
        Threshold(
            threshold_id="t1",
            calibration_split="final",
            fpr_target=0.05,
            scope="component",
            threshold=0.5,
            config_hash="x",
            monitor_id="m1",
        )


def test_store_round_trip(tmp_path) -> None:
    store = RunStore(tmp_path)
    manifest = RunManifest(run_id="r1", model_ids=["m1"], prompt_ids=["p1"])
    decisions = pd.DataFrame(
        [
            {
                "case_id": "c1",
                "monitor_id": "m1",
                "run_index": 0,
                "raw_label": "unsafe",
                "raw_score": 0.9,
            }
        ]
    )
    thresholds = pd.DataFrame(
        [
            {
                "threshold_id": "t1",
                "calibration_split": "calibrate",
                "fpr_target": 0.05,
                "scope": "component",
                "threshold": 0.5,
                "monitor_id": "m1",
            }
        ]
    )
    metrics = pd.DataFrame([{"metric": "R_obs", "value": 0.1}])
    report = {"run_id": "r1"}
    store.write_manifest(manifest)
    store.write_decisions("r1", decisions)
    store.write_thresholds("r1", thresholds)
    store.write_metrics("r1", metrics)
    store.write_report("r1", report)
    assert store.read_manifest("r1").run_id == "r1"
    assert len(store.read_decisions("r1")) == len(decisions)
    assert len(store.read_thresholds("r1")) == len(thresholds)
    assert len(store.read_metrics("r1")) == len(metrics)
    assert store.read_report("r1") == report
