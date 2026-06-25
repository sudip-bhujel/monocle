from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def metrics_to_rows(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    )


def write_demo_table(metrics: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        rf"    {key} & {format_value(value)} \\" for key, value in metrics.items()
    )
    table = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"  \caption{Monocle dry-run metrics.}",
            r"  \label{tab:monocle-dry-run}",
            r"  \centering\footnotesize",
            r"  \begin{tabular}{@{}lr@{}}",
            r"    \toprule",
            r"    Metric & Value \\",
            r"    \midrule",
            rows,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text(table, encoding="utf-8")
    return path


def write_h1_pilot_table(metrics: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        rf"    {row['metric']} & {format_value(row.get('value'))} & {format_value(row.get('lower'))} & {format_value(row.get('upper'))} & {row.get('status', 'pilot')} \\"
        for _, row in metrics.iterrows()
    )
    table = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"  \caption{H1 pilot metrics.}",
            r"  \label{tab:h1-pilot}",
            r"  \centering\footnotesize",
            r"  \begin{tabular}{@{}lrrrl@{}}",
            r"    \toprule",
            r"    Metric & Value & Lower & Upper & Status \\",
            r"    \midrule",
            rows,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text(table, encoding="utf-8")
    return path


def provenance_report(
    run_id: str,
    metrics: dict[str, Any],
    *,
    manifest: dict[str, Any] | None = None,
    threshold_ids: list[str] | None = None,
    bootstrap: dict[str, Any] | None = None,
    canary: dict[str, Any] | None = None,
    pilot_gate: str | None = None,
    committee_ablation: dict[str, Any] | None = None,
    committee: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = manifest or {}
    return _json_safe(
        {
            "run_id": run_id,
            "metrics": metrics,
            "pilot_gate": pilot_gate or h1_pilot_gate(metrics),
            "model_ids": manifest.get("model_ids", []),
            "prompt_ids": manifest.get("prompt_ids", []),
            "threshold_ids": threshold_ids or [],
            "committee": committee or {},
            "bootstrap": bootstrap or {},
            "canary": canary or {"status": "not_run"},
            "committee_ablation": committee_ablation or {},
        }
    )


def format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return r"\tbd"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def h1_pilot_gate(metrics: dict[str, Any]) -> str:
    gamma = _metric_value(metrics, "adversarial.Gamma")
    if gamma is None or pd.isna(gamma):
        return "inconclusive"
    return "pass" if float(gamma) > 1.0 else "fail"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _metric_value(metrics: dict[str, Any], name: str) -> Any:
    value = metrics.get(name)
    if isinstance(value, dict):
        return value.get("value")
    return value
