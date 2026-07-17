from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

from monocle.decision_bank import DecisionBank
from monocle.monitors import Monitor
from monocle.schema import Case, Decision, RunManifest
from monocle.store import RunStore

KEY_COLUMNS = ["case_id", "monitor_id", "run_index", "cache_key"]


def execute(
    cases: Iterable[Case],
    monitors: Iterable[Monitor],
    *,
    runs: int = 1,
    existing: pd.DataFrame | None = None,
    workers: int = 4,
    decision_bank: DecisionBank | None = None,
    decision_namespace: str = "default",
    execution_stats: dict[str, int] | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    existing = _existing_frame(existing)
    completed = _completed_keys(existing)
    jobs = []
    total_jobs = 0
    for case in cases:
        for monitor in monitors:
            for run_index in range(runs):
                total_jobs += 1
                cache_key = monitor.cache_key(case, run_index)
                if (
                    case.case_id,
                    monitor.config.monitor_id,
                    run_index,
                    cache_key,
                ) in completed:
                    continue
                jobs.append((case, monitor, run_index))
    stats = {
        "total_jobs": total_jobs,
        "local_resume_hits": total_jobs - len(jobs),
        "decision_bank_hits": 0,
        "in_run_deduplicated": 0,
        "provider_calls": 0,
    }
    results: list[Decision | None] = [None] * len(jobs)
    execution = []
    grouped: dict[str, dict] = {}
    for index, job in enumerate(jobs):
        case, monitor, run_index = job
        request_json = monitor.request_json(case, run_index, decision_namespace)
        fingerprint = (
            sha256(request_json.encode("utf-8")).hexdigest()
            if request_json is not None
            else None
        )
        if decision_bank is None or fingerprint is None or request_json is None:
            execution.append((index, job, fingerprint, request_json, []))
            if fingerprint is not None:
                stats["provider_calls"] += 1
            continue
        cached = decision_bank.get(fingerprint)
        if cached is not None:
            results[index] = _reuse_decision(cached, job, fingerprint)
            stats["decision_bank_hits"] += 1
            continue
        if fingerprint in grouped:
            grouped[fingerprint]["duplicates"].append(index)
            stats["in_run_deduplicated"] += 1
            continue
        grouped[fingerprint] = {
            "index": index,
            "job": job,
            "request_json": request_json,
            "duplicates": [],
        }
    for fingerprint, group in grouped.items():
        execution.append(
            (
                group["index"],
                group["job"],
                fingerprint,
                group["request_json"],
                group["duplicates"],
            )
        )
        stats["provider_calls"] += 1
    execution_jobs = [item[1] for item in execution]

    def on_complete(exec_index: int, decision: Decision) -> None:
        index, job, fingerprint, request_json, duplicates = execution[exec_index]
        if fingerprint is not None:
            decision = _fresh_decision(decision, fingerprint)
        results[index] = decision
        if decision_bank is not None and fingerprint and request_json:
            decision_bank.put(fingerprint, request_json, decision)
        for duplicate in duplicates:
            results[duplicate] = _reuse_decision(
                decision, jobs[duplicate], fingerprint
            )

    _run_jobs(
        execution_jobs,
        workers=workers,
        show_progress=show_progress,
        progress_desc=(
            f"hosted calls "
            f"(resume={stats['local_resume_hits']}, "
            f"bank={stats['decision_bank_hits']}, "
            f"dedupe={stats['in_run_deduplicated']})"
        ),
        on_complete=on_complete,
    )
    if any(result is None for result in results):
        raise RuntimeError("decision execution did not produce every requested result")
    if execution_stats is not None:
        execution_stats.clear()
        execution_stats.update(stats)
    new = pd.DataFrame(
        [result.model_dump(mode="json") for result in results if result is not None]
    )
    if existing.empty:
        return new
    if new.empty:
        return existing
    new_keys = _decision_keys(new)
    keep_existing = [
        _decision_key(row) not in new_keys
        for row in existing[KEY_COLUMNS].itertuples(index=False)
    ]
    return pd.concat(
        [existing.loc[keep_existing], new], ignore_index=True
    ).drop_duplicates(KEY_COLUMNS, keep="first")


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
    workers: int = 4,
    decision_bank_path: str | Path | None = None,
    decision_namespace: str = "default",
) -> pd.DataFrame:
    if store.paths(run_id).manifest.exists():
        prior = store.read_manifest(run_id)
        if prior.config_hash != config_hash:
            raise ValueError(
                f"run_id {run_id!r} already has a different provenance hash; "
                "use a fresh run ID rather than reusing cached decisions"
            )
        prior_namespace = prior.sampling.get("decision_namespace", "default")
        if prior_namespace != decision_namespace:
            raise ValueError(
                f"run_id {run_id!r} already uses decision namespace "
                f"{prior_namespace!r}; use a fresh run ID for {decision_namespace!r}"
            )
    existing = (
        store.read_decisions(run_id) if store.paths(run_id).decisions.exists() else None
    )
    stats: dict[str, int] = {}
    bank_context = (
        DecisionBank(decision_bank_path)
        if decision_bank_path is not None
        else nullcontext(None)
    )
    with bank_context as bank:
        decisions = execute(
            cases,
            monitors,
            runs=runs,
            existing=existing,
            workers=workers,
            decision_bank=bank,
            decision_namespace=decision_namespace,
            execution_stats=stats,
        )
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
        sampling={
            "runs": runs,
            "workers": workers,
            "decision_namespace": decision_namespace,
            "decision_bank": str(decision_bank_path) if decision_bank_path else None,
            **stats,
        },
        provider_settings={
            monitor.config.monitor_id: monitor.config.provider for monitor in monitors
        },
    )
    store.write_manifest(manifest)
    store.write_decisions(run_id, decisions)
    return decisions


def _manifest_model_id(monitor: Monitor) -> str:
    if hasattr(monitor, "manifest_model_id"):
        return str(monitor.manifest_model_id())
    return monitor.config.model_id


def _run_jobs(
    execution_jobs: list[tuple[Case, Monitor, int]],
    *,
    workers: int,
    show_progress: bool,
    progress_desc: str,
    on_complete,
) -> None:
    if not execution_jobs:
        return
    disable = (not show_progress) or (not sys.stderr.isatty())
    if workers == 1:
        for index, job in enumerate(
            tqdm(
                execution_jobs,
                desc=progress_desc,
                unit="call",
                disable=disable,
            )
        ):
            on_complete(index, _execute_job(job))
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_execute_job, job): index
            for index, job in enumerate(execution_jobs)
        }
        with tqdm(
            total=len(execution_jobs),
            desc=progress_desc,
            unit="call",
            disable=disable,
        ) as bar:
            for future in as_completed(futures):
                index = futures[future]
                on_complete(index, future.result())
                bar.update(1)


def _execute_job(job: tuple[Case, Monitor, int]) -> Decision:
    case, monitor, run_index = job
    return monitor.judge(case, run_index)


def _fresh_decision(decision: Decision, fingerprint: str) -> Decision:
    billed_cost = decision.provider_cost
    if billed_cost is None:
        billed_cost = decision.cost
    return decision.model_copy(
        update={
            "request_fingerprint": fingerprint,
            "cache_hit": False,
            "billed_cost": billed_cost,
        }
    )


def _reuse_decision(
    decision: Decision, job: tuple[Case, Monitor, int], fingerprint: str
) -> Decision:
    case, monitor, run_index = job
    return decision.model_copy(
        update={
            "case_id": case.case_id,
            "monitor_id": monitor.config.monitor_id,
            "run_index": run_index,
            "cache_key": monitor.cache_key(case, run_index),
            "request_fingerprint": fingerprint,
            "cache_hit": True,
            "billed_cost": 0.0,
        }
    )


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
    completed = existing.loc[~_error_rows(existing), KEY_COLUMNS]
    return _decision_keys(completed)


def _error_rows(frame: pd.DataFrame) -> pd.Series:
    errors = pd.Series(False, index=frame.index)
    if "raw_label" in frame.columns:
        errors |= frame["raw_label"].fillna("").astype(str).eq("error")
    if "error" in frame.columns:
        errors |= frame["error"].fillna("").astype(str).str.strip().ne("")
    return errors


def _decision_keys(frame: pd.DataFrame) -> set[tuple[str, str, int, str]]:
    return {
        _decision_key(row) for row in frame[KEY_COLUMNS].itertuples(index=False)
    }


def _decision_key(row) -> tuple[str, str, int, str]:
    return (
        str(row.case_id),
        str(row.monitor_id),
        int(row.run_index),
        str(row.cache_key),
    )
