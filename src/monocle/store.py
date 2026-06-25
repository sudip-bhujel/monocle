from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from monocle.schema import RunManifest, RunPaths


class RunStore:
    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)

    def paths(self, run_id: str) -> RunPaths:
        run_dir = self.root / run_id
        return RunPaths(
            run_dir=run_dir,
            manifest=run_dir / "manifest.json",
            decisions=run_dir / "decisions.parquet",
            thresholds=run_dir / "thresholds.parquet",
            metrics=run_dir / "metrics.parquet",
            report=run_dir / "report.json",
            derived=run_dir / "derived",
        )

    def ensure(self, run_id: str) -> RunPaths:
        paths = self.paths(run_id)
        paths.derived.mkdir(parents=True, exist_ok=True)
        return paths

    def write_manifest(self, manifest: RunManifest) -> None:
        paths = self.ensure(manifest.run_id)
        paths.manifest.write_text(
            manifest.model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )

    def read_manifest(self, run_id: str) -> RunManifest:
        return RunManifest.model_validate_json(
            self.paths(run_id).manifest.read_text(encoding="utf-8")
        )

    def write_decisions(self, run_id: str, decisions: pd.DataFrame) -> None:
        self.ensure(run_id)
        decisions.to_parquet(self.paths(run_id).decisions, index=False)

    def read_decisions(self, run_id: str) -> pd.DataFrame:
        return pd.read_parquet(self.paths(run_id).decisions)

    def write_thresholds(self, run_id: str, thresholds: pd.DataFrame) -> None:
        self.ensure(run_id)
        thresholds.to_parquet(self.paths(run_id).thresholds, index=False)

    def read_thresholds(self, run_id: str) -> pd.DataFrame:
        return pd.read_parquet(self.paths(run_id).thresholds)

    def write_metrics(self, run_id: str, metrics: pd.DataFrame) -> None:
        self.ensure(run_id)
        metrics.to_parquet(self.paths(run_id).metrics, index=False)

    def read_metrics(self, run_id: str) -> pd.DataFrame:
        return pd.read_parquet(self.paths(run_id).metrics)

    def write_report(self, run_id: str, report: dict[str, Any]) -> None:
        self.ensure(run_id)
        self.paths(run_id).report.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def read_report(self, run_id: str) -> dict[str, Any]:
        return json.loads(self.paths(run_id).report.read_text(encoding="utf-8"))

    def derived_path(self, run_id: str, name: str) -> Path:
        paths = self.ensure(run_id)
        return paths.derived / name
