from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from monocle.monitors import Monitor
from monocle.schema import Case, RunManifest
from monocle.store import RunStore

KEY_COLUMNS = ["case_id", "monitor_id", "run_index", "cache_key"]


def execute(
    cases: Iterable[Case],
    monitors: Iterable[Monitor],
    *,
    runs: int = 1,
    existing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    existing = _existing_frame(existing)
    completed = _completed_keys(existing)
    decisions = []
    for case in cases:
        for monitor in monitors:
            for run_index in range(runs):
                cache_key = monitor.cache_key(case, run_index)
                if (
                    case.case_id,
                    monitor.config.monitor_id,
                    run_index,
                    cache_key,
                ) in completed:
                    continue
                decisions.append(monitor.judge(case, run_index).model_dump(mode="json"))
    new = pd.DataFrame(decisions)
    if existing.empty:
        return new
    if new.empty:
        return existing
    return pd.concat([existing, new], ignore_index=True).drop_duplicates(
        KEY_COLUMNS, keep="first"
    )


def write_run(
    run_id: str,
    cases: list[Case],
    monitors: list[Monitor],
    *,
    runs: int,
    store: RunStore,
    allow_hosted: bool = False,
    config_hash: str = "unknown",
    artifact_hashes: dict[str, str] | None = None,
) -> pd.DataFrame:
    manifest = RunManifest(
        run_id=run_id,
        config_hash=config_hash,
        model_ids=[_manifest_model_id(monitor) for monitor in monitors],
        prompt_ids=[monitor.config.prompt_id for monitor in monitors],
        provider_names={
            monitor.config.monitor_id: monitor.config.provider for monitor in monitors
        },
        artifact_hashes=artifact_hashes or {},
        allow_hosted=allow_hosted,
        sampling={"runs": runs},
        provider_settings={
            monitor.config.monitor_id: monitor.config.provider for monitor in monitors
        },
    )
    existing = (
        store.read_decisions(run_id) if store.paths(run_id).decisions.exists() else None
    )
    decisions = execute(cases, monitors, runs=runs, existing=existing)
    store.write_manifest(manifest)
    store.write_decisions(run_id, decisions)
    return decisions


def _manifest_model_id(monitor: Monitor) -> str:
    if hasattr(monitor, "manifest_model_id"):
        return str(monitor.manifest_model_id())
    return monitor.config.model_id


def _existing_frame(existing: pd.DataFrame | None) -> pd.DataFrame:
    if existing is None:
        return pd.DataFrame()
    missing = [column for column in KEY_COLUMNS if column not in existing.columns]
    if missing:
        return pd.DataFrame()
    return existing.copy()


def _completed_keys(existing: pd.DataFrame) -> set[tuple[str, str, int, str]]:
    if existing.empty:
        return set()
    return {
        (str(row.case_id), str(row.monitor_id), int(row.run_index), str(row.cache_key))
        for row in existing[KEY_COLUMNS].itertuples(index=False)
    }
