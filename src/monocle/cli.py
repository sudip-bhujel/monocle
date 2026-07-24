from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from monocle.analysis import committee_ablation, h1_metric_values, summarize_h1
from monocle.calibration import (
    committee_fpr,
    component_thresholds,
    cost_latency_summary,
    crossfit_committee_fpr,
)
from monocle.config import fpr_targets, hash_files, hash_mapping, load_env, load_experiment, load_yaml
from monocle.data import load_cases
from monocle.metrics import catch_matrix
from monocle.monitors import FixtureMonitor, build_monitor, validate_hosted_monitors
from monocle.report import provenance_report, write_h1_metrics_table
from monocle.run import execute, write_run
from monocle.schema import MonitorConfig
from monocle.stats import bootstrap_metrics, confidence_interval
from monocle.store import RunStore


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def main(argv: list[str] | None = None) -> None:
    load_env()
    parser = argparse.ArgumentParser(prog="monocle")
    parser.add_argument("--runs-root", default="runs")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in [
        "run",
        "calibrate",
        "metrics",
        "bootstrap",
        "ablation",
        "report",
        "canary",
    ]:
        cmd = sub.add_parser(name)
        _add_common_args(cmd)
        if name == "run":
            cmd.add_argument("--decision-bank")
            cmd.add_argument("--decision-namespace", default="default")
            cmd.add_argument("--no-decision-cache", action="store_true")
    analyze = sub.add_parser(
        "analyze",
        help=(
            "run metrics, bootstrap, ablation, and report for one or more committees "
            "in one process (same outputs as the separate commands)"
        ),
    )
    _add_common_args(analyze)
    analyze.add_argument(
        "--all-committees",
        action="store_true",
        help="analyze every committee listed in --committees",
    )
    args = parser.parse_args(argv)
    store = RunStore(args.runs_root)
    if args.command == "run":
        _run(args.run_id, store, args)
    elif args.command == "calibrate":
        _calibrate(args.run_id, store, args)
    elif args.command == "metrics":
        _metrics(args.run_id, store, args)
    elif args.command == "bootstrap":
        _bootstrap(args.run_id, store, args)
    elif args.command == "ablation":
        _ablation(args.run_id, store, args)
    elif args.command == "report":
        _report(args.run_id, store, args)
    elif args.command == "analyze":
        _analyze(args.run_id, store, args)
    elif args.command == "canary":
        _canary(args.run_id, store, args)


def _add_common_args(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument("--run-id", default="dry-run")
    cmd.add_argument("--cases")
    cmd.add_argument("--models")
    cmd.add_argument("--committees", default="configs/committees.yaml")
    cmd.add_argument("--committee")
    cmd.add_argument("--experiment", default="configs/experiment.yaml")
    cmd.add_argument("--runs", type=int, default=3)
    cmd.add_argument("--workers", type=_positive_int, default=4)
    cmd.add_argument("--draws", type=int, default=25)
    cmd.add_argument("--seed", type=int, default=1)
    cmd.add_argument("--allow-hosted", action="store_true")


def _cases_frame(path: str | None = None) -> pd.DataFrame:
    return pd.DataFrame([case.model_dump(mode="json") for case in load_cases(path)])


def _run(run_id: str, store: RunStore, args: argparse.Namespace) -> None:
    monitors = _load_monitors(args.models)
    validate_hosted_monitors(monitors, allow_hosted=args.allow_hosted)
    artifact_hashes = _artifact_hashes(args, monitors)
    decision_bank = None
    if not args.no_decision_cache and any(
        monitor.config.mechanism == "llm" for monitor in monitors
    ):
        decision_bank = args.decision_bank or str(
            store.paths(run_id).run_dir / "decision-bank.csv"
        )
    write_run(
        run_id,
        load_cases(args.cases),
        monitors,
        runs=args.runs,
        store=store,
        allow_hosted=args.allow_hosted,
        config_hash=hash_mapping(artifact_hashes),
        artifact_hashes=artifact_hashes,
        workers=args.workers,
        decision_bank_path=decision_bank,
        decision_namespace=args.decision_namespace,
    )


def _calibrate(run_id: str, store: RunStore, args: argparse.Namespace) -> None:
    _require_case_splits(args.cases)
    decisions = store.read_decisions(run_id)
    cases = _cases_frame(args.cases)
    config_hash = _manifest_config_hash(store, run_id)
    experiment = load_experiment(args.experiment)
    thresholds = pd.concat(
        [
            component_thresholds(
                decisions,
                cases,
                fpr_target=target,
                threshold_id=_threshold_id(target),
                config_hash=config_hash,
            )
            for target in fpr_targets(args.experiment)
        ],
        ignore_index=True,
    )
    store.write_thresholds(run_id, thresholds)
    store.derived_path(run_id, "calibration-summary.json").write_text(
        json.dumps(
            _calibration_summary(
                decisions,
                cases,
                thresholds,
                crossfit_folds=int(experiment.get("safe_crossfit_folds", 0)),
                seed=int(experiment.get("seed", args.seed)),
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _metrics(run_id: str, store: RunStore, args: argparse.Namespace) -> None:
    decisions, thresholds, committee = _committee_inputs(run_id, store, args)
    cases = _cases_frame(args.cases)
    _write_metrics(run_id, store, decisions, thresholds, cases, committee)


def _bootstrap(run_id: str, store: RunStore, args: argparse.Namespace) -> None:
    decisions, thresholds, committee = _committee_inputs(run_id, store, args)
    matrix = catch_matrix(decisions, thresholds, _cases_frame(args.cases))
    _write_bootstrap(run_id, store, committee, matrix, draws=args.draws, seed=args.seed)


def _ablation(run_id: str, store: RunStore, args: argparse.Namespace) -> None:
    decisions, thresholds, committee = _committee_inputs(run_id, store, args)
    matrix = catch_matrix(decisions, thresholds, _cases_frame(args.cases))
    _write_ablation(run_id, store, committee, matrix)


def _report(run_id: str, store: RunStore, args: argparse.Namespace) -> None:
    committee = _report_committee(run_id, store, args)
    _write_report(run_id, store, committee, selected=bool(args.committee))


def _analyze(run_id: str, store: RunStore, args: argparse.Namespace) -> None:
    decisions = store.read_decisions(run_id)
    thresholds = _primary_thresholds(store.read_thresholds(run_id), args)
    cases = _cases_frame(args.cases)
    available = sorted(decisions["monitor_id"].unique())
    for committee_id in _analyze_committee_ids(args):
        committee_args = argparse.Namespace(**vars(args))
        committee_args.committee = committee_id
        committee = _selected_committee(committee_args, available)
        members = committee["members"]
        missing_decisions = set(members) - set(decisions["monitor_id"])
        missing_thresholds = set(members) - set(thresholds["monitor_id"])
        if missing_decisions:
            raise ValueError(
                f"committee {committee['committee_id']} has no decisions for {sorted(missing_decisions)}"
            )
        if missing_thresholds:
            raise ValueError(
                f"committee {committee['committee_id']} has no thresholds for {sorted(missing_thresholds)}"
            )
        member_decisions = decisions[decisions["monitor_id"].isin(members)].copy()
        member_thresholds = thresholds[thresholds["monitor_id"].isin(members)].copy()
        matrix = catch_matrix(member_decisions, member_thresholds, cases)
        _write_metrics(
            run_id, store, member_decisions, member_thresholds, cases, committee
        )
        _write_bootstrap(
            run_id,
            store,
            committee,
            matrix,
            draws=args.draws,
            seed=args.seed,
        )
        _write_ablation(run_id, store, committee, matrix)
        _write_report(run_id, store, committee, selected=True)


def _analyze_committee_ids(args: argparse.Namespace) -> list[str]:
    if args.all_committees:
        return _committee_ids(args.committees)
    if args.committee:
        return [part.strip() for part in str(args.committee).split(",") if part.strip()]
    raise ValueError("analyze requires --committee or --all-committees")


def _committee_ids(path: str | None) -> list[str]:
    if path is None or not Path(path).exists():
        raise ValueError(f"committee config not found: {path}")
    committees = load_yaml(path).get("committees", [])
    if isinstance(committees, dict):
        return list(committees.keys())
    if isinstance(committees, list):
        ids = [
            str(item["committee_id"])
            for item in committees
            if item.get("committee_id") is not None
        ]
        if ids:
            return ids
    raise ValueError(f"no committees found in {path}")


def _write_metrics(
    run_id: str,
    store: RunStore,
    decisions: pd.DataFrame,
    thresholds: pd.DataFrame,
    cases: pd.DataFrame,
    committee: dict,
) -> None:
    summary = summarize_h1(decisions, thresholds, cases)
    rows = pd.DataFrame(summary)
    store.write_metrics(run_id, rows)
    rows.to_parquet(
        _committee_path(store, run_id, committee, "metrics.parquet"), index=False
    )
    _write_committee_selection(store, run_id, committee)


def _write_bootstrap(
    run_id: str,
    store: RunStore,
    committee: dict,
    matrix: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> None:
    draw_rows = bootstrap_metrics(
        matrix,
        h1_metric_values,
        draws=draws,
        seed=seed,
        stratify_by="stratum_id",
    )
    draw_rows.to_parquet(
        store.derived_path(run_id, "bootstrap_draws.parquet"), index=False
    )
    draw_rows.to_parquet(
        _committee_path(store, run_id, committee, "bootstrap_draws.parquet"),
        index=False,
    )
    bootstrap_meta = {
        "draws": draws,
        "seed": seed,
        "cluster": "base_task_id",
        "stratify_by": "stratum_id",
        "interval": "percentile",
    }
    _write_json(store.derived_path(run_id, "bootstrap_meta.json"), bootstrap_meta)
    _write_json(
        _committee_path(store, run_id, committee, "bootstrap_meta.json"),
        bootstrap_meta,
    )
    metrics = _read_committee_metrics(store, run_id, committee)
    rows = _with_intervals(metrics, draw_rows)
    store.write_metrics(run_id, rows)
    rows.to_parquet(
        _committee_path(store, run_id, committee, "metrics.parquet"), index=False
    )
    _write_committee_selection(store, run_id, committee)


def _write_ablation(
    run_id: str, store: RunStore, committee: dict, matrix: pd.DataFrame
) -> None:
    rows = committee_ablation(matrix)
    rows.to_parquet(
        store.derived_path(run_id, "committee-ablation.parquet"), index=False
    )
    rows.to_parquet(
        _committee_path(store, run_id, committee, "committee-ablation.parquet"),
        index=False,
    )
    _write_committee_selection(store, run_id, committee)


def _write_report(
    run_id: str, store: RunStore, committee: dict, *, selected: bool
) -> None:
    rows = _read_report_metrics(store, run_id, committee, selected=selected)
    if selected:
        store.write_metrics(run_id, rows)
        _write_committee_selection(store, run_id, committee)
    write_h1_metrics_table(rows, store.derived_path(run_id, "h1-metrics-table.tex"))
    write_h1_metrics_table(
        rows, _committee_path(store, run_id, committee, "h1-metrics-table.tex")
    )
    manifest = store.read_manifest(run_id).model_dump(mode="json")
    thresholds = store.read_thresholds(run_id)
    bootstrap = _read_bootstrap_meta(store, run_id, committee)
    canary = _existing_canary(store, run_id)
    metrics = {
        row["metric"]: {
            "value": row.get("value"),
            "lower": row.get("lower"),
            "upper": row.get("upper"),
        }
        for _, row in rows.iterrows()
    }
    report = provenance_report(
        run_id,
        metrics,
        manifest=manifest,
        threshold_ids=sorted(set(thresholds["threshold_id"])),
        committee=committee,
        bootstrap=bootstrap,
        canary=canary,
        committee_ablation=_committee_ablation_summary(store, run_id, committee),
    )
    store.write_report(run_id, report)
    _write_json(_committee_path(store, run_id, committee, "report.json"), report)


def _load_monitors(path: str | None) -> list:
    if path is None:
        return [
            FixtureMonitor(
                MonitorConfig(
                    monitor_id="m1",
                    provider="fixture",
                    model_id="fixture-a",
                    mechanism="fixture",
                )
            ),
            FixtureMonitor(
                MonitorConfig(
                    monitor_id="m2",
                    provider="fixture",
                    model_id="fixture-b",
                    mechanism="fixture",
                )
            ),
        ]
    data = load_yaml(path)
    configs = data.get("models", [])
    adjacent_prompt_dir = Path(path).parent / "prompts"
    prompt_dir = (
        adjacent_prompt_dir
        if adjacent_prompt_dir.is_dir()
        else Path("configs/prompts")
    )
    normalized = []
    for config in configs:
        item = dict(config)
        metadata = dict(item.get("metadata", {}))
        metadata.setdefault("prompt_dir", str(prompt_dir))
        item["metadata"] = metadata
        normalized.append(item)
    return [
        build_monitor(MonitorConfig.model_validate(config)) for config in normalized
    ]


def _artifact_hashes(args: argparse.Namespace, monitors: list) -> dict[str, str]:
    paths = [
        path
        for path in [
            args.models,
            args.committees,
            args.experiment,
            args.cases,
        ]
        if path
    ]
    for monitor in monitors:
        prompt_dir = Path(
            str(monitor.config.metadata.get("prompt_dir", "configs/prompts"))
        )
        paths.append(prompt_dir / f"{monitor.config.prompt_id}.txt")
    return hash_files(paths)


def _with_intervals(metrics: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in metrics.iterrows():
        metric = row["metric"]
        lower = upper = None
        if metric in draws.columns:
            lower, upper = confidence_interval(draws[metric])
        rows.append(
            {
                "metric": metric,
                "value": row["value"],
                "lower": lower,
                "upper": upper,
            }
        )
    return pd.DataFrame(rows)


def _primary_thresholds(
    thresholds: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    target = fpr_targets(args.experiment)[0]
    selected = thresholds[thresholds["threshold_id"] == _threshold_id(target)]
    if selected.empty:
        selected = thresholds[thresholds["fpr_target"] == target]
    if selected.empty:
        raise ValueError(f"no thresholds found for primary fpr target {target}")
    return selected.copy()


def _report_committee(run_id: str, store: RunStore, args: argparse.Namespace) -> dict:
    if args.committee:
        available = sorted(store.read_decisions(run_id)["monitor_id"].unique())
        return _selected_committee(args, available)
    selected = _read_committee_selection(store, run_id)
    if selected:
        return selected
    available = sorted(store.read_decisions(run_id)["monitor_id"].unique())
    return {"committee_id": "full", "members": available, "rule": "any_flag"}


def _read_report_metrics(
    store: RunStore, run_id: str, committee: dict, *, selected: bool
) -> pd.DataFrame:
    if selected:
        return _read_committee_metrics(store, run_id, committee)
    return store.read_metrics(run_id)


def _read_committee_metrics(
    store: RunStore, run_id: str, committee: dict
) -> pd.DataFrame:
    path = _committee_path(store, run_id, committee, "metrics.parquet", ensure=False)
    if path.exists():
        return pd.read_parquet(path)
    selected = _read_committee_selection(store, run_id)
    if selected.get("committee_id") == committee["committee_id"]:
        return store.read_metrics(run_id)
    raise ValueError(f"run metrics for committee {committee['committee_id']} first")


def _read_bootstrap_meta(store: RunStore, run_id: str, committee: dict) -> dict:
    path = _committee_path(
        store, run_id, committee, "bootstrap_meta.json", ensure=False
    )
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    path = store.derived_path(run_id, "bootstrap_meta.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _committee_inputs(
    run_id: str, store: RunStore, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    decisions = store.read_decisions(run_id)
    thresholds = _primary_thresholds(store.read_thresholds(run_id), args)
    committee = _selected_committee(args, sorted(decisions["monitor_id"].unique()))
    members = committee["members"]
    missing_decisions = set(members) - set(decisions["monitor_id"])
    missing_thresholds = set(members) - set(thresholds["monitor_id"])
    if missing_decisions:
        raise ValueError(
            f"committee {committee['committee_id']} has no decisions for {sorted(missing_decisions)}"
        )
    if missing_thresholds:
        raise ValueError(
            f"committee {committee['committee_id']} has no thresholds for {sorted(missing_thresholds)}"
        )
    return (
        decisions[decisions["monitor_id"].isin(members)].copy(),
        thresholds[thresholds["monitor_id"].isin(members)].copy(),
        committee,
    )


def _selected_committee(args: argparse.Namespace, available: list[str]) -> dict:
    committee_id = args.committee or "full"
    if committee_id == "full":
        return {"committee_id": "full", "members": available, "rule": "any_flag"}
    item = _committee_config(args.committees, committee_id)
    rule = item.get("rule", "any_flag")
    if rule != "any_flag":
        raise ValueError(f"unsupported committee rule for {committee_id}: {rule}")
    members = [
        str(member) for member in item.get("members", item.get("monitor_ids", []))
    ]
    if not members:
        raise ValueError(f"committee {committee_id} must list at least one member")
    missing = set(members) - set(available)
    if missing:
        raise ValueError(
            f"committee {committee_id} references unavailable monitors: {sorted(missing)}"
        )
    return {"committee_id": committee_id, "members": members, "rule": rule}


def _committee_config(path: str | None, committee_id: str) -> dict:
    if path is None or not Path(path).exists():
        raise ValueError(f"committee config not found: {path}")
    committees = load_yaml(path).get("committees", [])
    if isinstance(committees, dict):
        item = committees.get(committee_id)
        if item is not None:
            return dict(item)
    if isinstance(committees, list):
        for item in committees:
            if item.get("committee_id") == committee_id:
                return dict(item)
    raise ValueError(f"unknown committee: {committee_id}")


def _write_committee_selection(store: RunStore, run_id: str, committee: dict) -> None:
    _write_json(store.derived_path(run_id, "committee-selection.json"), committee)
    _write_json(
        _committee_path(store, run_id, committee, "committee-selection.json"),
        committee,
    )


def _read_committee_selection(store: RunStore, run_id: str) -> dict:
    path = store.derived_path(run_id, "committee-selection.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _committee_path(
    store: RunStore,
    run_id: str,
    committee: dict,
    name: str,
    *,
    ensure: bool = True,
) -> Path:
    committee_id = _committee_id(committee)
    path = store.derived_path(run_id, f"committees/{committee_id}/{name}")
    if ensure:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _committee_id(committee: dict) -> str:
    committee_id = str(committee.get("committee_id", "full"))
    if not committee_id or committee_id in {".", ".."}:
        raise ValueError("committee_id must be a non-empty path segment")
    if "/" in committee_id or "\\" in committee_id:
        raise ValueError(f"committee_id cannot contain path separators: {committee_id}")
    return committee_id


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _threshold_id(target: float) -> str:
    return f"component-fpr-{target:g}".replace(".", "p")


def _manifest_config_hash(store: RunStore, run_id: str) -> str:
    if not store.paths(run_id).manifest.exists():
        return "unknown"
    return store.read_manifest(run_id).config_hash


def _calibration_summary(
    decisions: pd.DataFrame,
    cases: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    crossfit_folds: int = 0,
    seed: int = 1,
) -> dict:
    rows = []
    for threshold_id, group in thresholds.groupby("threshold_id"):
        all_safe_fpr = committee_fpr(decisions, cases, group)
        row = {
            "threshold_id": threshold_id,
            "fpr_target": float(group["fpr_target"].iloc[0]),
            "observed_committee_fpr": all_safe_fpr,
            "all_safe_committee_fpr": all_safe_fpr,
            "paired_calibration_committee_fpr": committee_fpr(
                decisions, cases, group, safe_split="calibrate"
            ),
            "family_disjoint_select_committee_fpr": committee_fpr(
                decisions, cases, group, safe_split="select"
            ),
            "final_holdout_committee_fpr": committee_fpr(
                decisions, cases, group, safe_split="holdout"
            ),
            "rule_id": "any_flag",
        }
        if crossfit_folds >= 2:
            paired_safe_crossfit = crossfit_committee_fpr(
                decisions,
                cases,
                fpr_target=float(group["fpr_target"].iloc[0]),
                folds=crossfit_folds,
                seed=seed,
            )
            row["crossfit"] = paired_safe_crossfit
            row["paired_safe_crossfit"] = paired_safe_crossfit
        rows.append(row)
    return {"thresholds": rows, "cost_latency": cost_latency_summary(decisions)}


def _committee_ablation_summary(
    store: RunStore, run_id: str, committee: dict | None = None
) -> dict:
    path = (
        _committee_path(
            store, run_id, committee, "committee-ablation.parquet", ensure=False
        )
        if committee
        else store.derived_path(run_id, "committee-ablation.parquet")
    )
    if not path.exists():
        path = store.derived_path(run_id, "committee-ablation.parquet")
    if not path.exists():
        return {}
    rows = pd.read_parquet(path)
    if rows.empty:
        return {"rows": 0}
    ordered = rows.sort_values(["unsafe_misses", "size", "monitors"])
    worst = rows.sort_values(
        ["unsafe_misses", "size", "monitors"], ascending=[False, True, True]
    )
    return {
        "rows": int(len(rows)),
        "best": _ablation_report_row(ordered.iloc[0]),
        "worst": _ablation_report_row(worst.iloc[0]),
    }


def _ablation_report_row(row: pd.Series) -> dict[str, object]:
    result = {
        "rule_id": row["rule_id"],
        "size": int(row["size"]),
        "monitors": row["monitors"],
        "unsafe_misses": int(row["unsafe_misses"]),
        "R_obs": row.get("R_obs"),
        "R_ind": row.get("R_ind"),
        "Gamma": row.get("Gamma"),
    }
    for name in ("adversarial_misses", "non_adversarial_misses"):
        if name in row.index:
            result[name] = int(row[name])
    return result


def _require_case_splits(path: str | None) -> None:
    cases = _cases_frame(path)
    if cases[(cases["split"] == "calibrate") & (cases["label"] == "safe")].empty:
        raise ValueError("calibration requires safe calibrate cases")
    if cases[(cases["split"] == "final") & (cases["label"] == "unsafe")].empty:
        raise ValueError("metrics require unsafe final cases")


def _canary(run_id: str, store: RunStore, args: argparse.Namespace) -> None:
    monitors = _load_monitors(args.models)
    validate_hosted_monitors(monitors, allow_hosted=args.allow_hosted)
    cases = load_cases(args.cases)[:2]
    current = execute(cases, monitors, runs=1, workers=args.workers)
    baseline = store.read_decisions(run_id)
    summary = _canary_summary(baseline, current)
    report = (
        store.read_report(run_id)
        if store.paths(run_id).report.exists()
        else {"run_id": run_id}
    )
    report["canary"] = summary
    store.write_report(run_id, report)


def _canary_summary(
    baseline: pd.DataFrame, current: pd.DataFrame
) -> dict[str, float | int | str]:
    keys = ["case_id", "monitor_id", "run_index"]
    base = baseline[baseline["run_index"] == 0][keys + ["raw_label", "raw_score"]]
    now = current[keys + ["raw_label", "raw_score"]]
    joined = base.merge(now, on=keys, suffixes=("_baseline", "_current"))
    if joined.empty:
        return {
            "status": "no_overlap",
            "compared": 0,
            "label_flips": 0,
            "mean_abs_score_delta": 0.0,
        }
    score_delta = (joined["raw_score_current"] - joined["raw_score_baseline"]).abs()
    label_flips = (joined["raw_label_current"] != joined["raw_label_baseline"]).sum()
    return {
        "status": "complete",
        "compared": int(len(joined)),
        "label_flips": int(label_flips),
        "mean_abs_score_delta": float(score_delta.mean()),
    }


def _existing_canary(store: RunStore, run_id: str) -> dict[str, str]:
    if not store.paths(run_id).report.exists():
        return {"status": "not_run"}
    return store.read_report(run_id).get("canary", {"status": "not_run"})


if __name__ == "__main__":
    main()
