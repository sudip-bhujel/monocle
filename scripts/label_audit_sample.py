"""Prepare and consolidate blinded Action Safety annotations.

The legacy disjoint workflow assigns one primary rater per case.  The researched
workflow assigns exactly two of five raters per case, while direct and pressure
variants from a family are assigned to disjoint rater pairs. Worksheets expose
opaque audit IDs; researched raters enter only a binary safe/unsafe label.

Usage:
  python scripts/label_audit_sample.py prepare-disjoint --d1-path CANDIDATE \
      --blind-dir /secure/raters --key /secure/coordinator/key.csv --raters 4
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import random
import secrets
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from monocle.data import load_cases

ROOT = Path(__file__).resolve().parent.parent
VALID_ANNOTATOR_LABELS = {"safe", "unsafe", "unclear"}
VALID_ROTATING_LABELS = {"safe", "unsafe"}
CONFIDENCE_VALUES = {"low", "medium", "high"}
ISSUE_FLAG_VALUES = {
    "insufficient_context",
    "unrealistic",
    "implausible_workflow",
    "template_cue",
}
DISJOINT_BLIND_FIELDS = [
    "audit_id",
    "track",
    "payload",
    "annotator_label",
    "issue_flags",
    "confidence",
    "reviewer_notes",
]
ROTATING_BLIND_FIELDS = [
    "audit_id",
    "kind",
    "payload",
    "annotator_label",
]
DISJOINT_KEY_FIELDS = [
    "rater_id",
    "audit_id",
    "case_id",
    "base_task_id",
    "stratum_id",
    "split",
    "regime",
    "oracle_label",
]
ROTATING_KEY_FIELDS = [
    "rater_id",
    "rating_slot",
    "audit_id",
    "case_id",
    "base_task_id",
    "stratum_id",
    "split",
    "regime",
    "variant_id",
    "oracle_label",
]
ADJUDICATION_FIELDS = [
    "case_id",
    "adjudicator_id",
    "adjudicated_label",
    "disposition",
    "rationale",
    "decision_locked_at",
]

def _secret_seed(seed_file: Path | None) -> int:
    """Read a coordinator-held secret seed, or create an unreported one."""
    if seed_file is None:
        return secrets.randbits(256)
    secret = seed_file.read_text(encoding="utf-8").strip()
    if not secret:
        raise ValueError(f"audit seed file is empty: {seed_file}")
    return int.from_bytes(sha256(secret.encode("utf-8")).digest(), "big")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _require_private_audit_paths(*, blind_dir: Path, key_path: Path) -> None:
    """Reject source-tree paths and colocated worksheets/keys.

    Blinding is an operational property, so a public source checkout must not
    be an accepted destination for either worksheets or the private mapping.
    """
    for label, path in (("blind worksheet directory", blind_dir), ("private key", key_path)):
        if _is_within(path, ROOT):
            raise ValueError(f"{label} must be outside the source repository: {path}")
    if _is_within(key_path, blind_dir):
        raise ValueError("private key must not be stored inside the worksheet directory")


def _require_outside_source(*paths: Path) -> None:
    for path in paths:
        if _is_within(path, ROOT):
            raise ValueError(f"private audit material must be outside the source repository: {path}")


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _ensure_new(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite audit material; choose fresh paths: "
            + ", ".join(existing)
        )


def _disjoint_review_cases(d1_path: Path) -> list:
    """Return the complete Action Safety review population."""
    cases = load_cases(str(d1_path))
    primary_unsafe_strata = {
        case.stratum_id
        for case in cases
        if case.label == "unsafe" and case.split == "final"
    }
    selected = []
    for case in cases:
        if case.label == "safe":
            if case.split not in {"calibrate", "select", "holdout"}:
                continue
            safe_stratum = case.stratum_id.removeprefix("safe:")
            if safe_stratum not in primary_unsafe_strata:
                raise ValueError(
                    "safe row stratum must match a primary unsafe stratum: "
                    f"{case.stratum_id}"
                )
            selected.append(case)
        elif case.split in {"final", "stress"}:
            selected.append(case)
    if not selected:
        raise ValueError("no eligible Action Safety cases found")
    case_ids = [case.case_id for case in selected]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("eligible Action Safety cases repeat case_id")
    return selected


def _balanced_disjoint_assignment(cases: list, rater_ids: list[str], rng: random.Random) -> dict[str, list]:
    """Balance each populated design cell while assigning every case once."""
    groups: dict[tuple[str, ...], list] = defaultdict(list)
    for case in cases:
        groups[(
            case.label,
            case.stratum_id.removeprefix("safe:"),
            case.split,
            case.regime,
        )].append(case)
    assigned = {rater_id: [] for rater_id in rater_ids}
    totals = Counter({rater_id: 0 for rater_id in rater_ids})
    for group_key in sorted(groups):
        ordered = sorted(groups[group_key], key=lambda case: case.case_id)
        rng.shuffle(ordered)
        within_group = Counter({rater_id: 0 for rater_id in rater_ids})
        tie_break = {rater_id: rng.random() for rater_id in rater_ids}
        for case in ordered:
            rater_id = min(
                rater_ids,
                key=lambda candidate: (
                    within_group[candidate],
                    totals[candidate],
                    tie_break[candidate],
                ),
            )
            assigned[rater_id].append(case)
            within_group[rater_id] += 1
            totals[rater_id] += 1
    if sum(map(len, assigned.values())) != len(cases):
        raise AssertionError("disjoint assignment lost cases")
    assigned_ids = [case.case_id for rows in assigned.values() for case in rows]
    if len(assigned_ids) != len(set(assigned_ids)):
        raise AssertionError("disjoint assignment repeated cases")
    return assigned


def prepare_disjoint_audit(
    *,
    d1_path: Path,
    blind_dir: Path,
    key_path: Path,
    rater_count: int,
    seed_file: Path | None,
) -> None:
    """Assign each complete-dataset case to one primary rater."""
    if rater_count < 2:
        raise ValueError("disjoint annotation requires at least two raters")
    _require_private_audit_paths(blind_dir=blind_dir, key_path=key_path)
    secret_seed = _secret_seed(seed_file)
    rng = random.Random(secret_seed)
    cases = _disjoint_review_cases(d1_path)
    rater_ids = [f"rater-{index}" for index in range(1, rater_count + 1)]
    assigned = _balanced_disjoint_assignment(cases, rater_ids, rng)
    metadata_path = key_path.with_suffix(".metadata.json")
    instructions_path = blind_dir / "review-instructions.txt"
    output_paths = [key_path, metadata_path, instructions_path]
    output_paths.extend(
        blind_dir / f"label-audit-action-safety-disjoint-{rater_id}.csv"
        for rater_id in rater_ids
    )
    _ensure_new(output_paths)
    blind_dir.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_rows = []
    for rater_id in rater_ids:
        ordered = list(assigned[rater_id])
        rng.shuffle(ordered)
        blind_rows = []
        for case in ordered:
            audit_id = secrets.token_urlsafe(12)
            blind_rows.append({
                "audit_id": audit_id,
                "track": "D1",
                "payload": case.payload,
                "annotator_label": "",
                "issue_flags": "",
                "confidence": "",
                "reviewer_notes": "",
            })
            key_rows.append({
                "rater_id": rater_id,
                "audit_id": audit_id,
                "case_id": case.case_id,
                "base_task_id": case.base_task_id,
                "stratum_id": case.stratum_id.removeprefix("safe:"),
                "split": case.split,
                "regime": case.regime,
                "oracle_label": case.label,
            })
        _write_csv(
            blind_dir / f"label-audit-action-safety-disjoint-{rater_id}.csv",
            DISJOINT_BLIND_FIELDS,
            blind_rows,
        )
    _write_csv(key_path, DISJOINT_KEY_FIELDS, key_rows)
    instructions_path.write_text(
        "Complete three required fields for every row.\n"
        "annotator_label: safe, unsafe, or unclear\n"
        "issue_flags: none, or semicolon-separated values from "
        "insufficient_context;unrealistic;implausible_workflow;template_cue\n"
        "confidence: low, medium, or high\n"
        "reviewer_notes: optional; explain unclear labels, issue flags, or low confidence\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "assignment": "one-primary-rater-per-case",
                "dataset_sha256": sha256(d1_path.read_bytes()).hexdigest(),
                "seed_commitment": sha256(str(secret_seed).encode("utf-8")).hexdigest(),
                "case_count": len(cases),
                "rater_count": rater_count,
                "cases_per_rater": {
                    rater_id: len(assigned[rater_id]) for rater_id in rater_ids
                },
                "required_manual_fields": [
                    "annotator_label",
                    "issue_flags",
                    "confidence",
                ],
                "inter_rater_reliability_available": False,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(
        f"prepared {len(cases)} disjoint cases for {rater_count} raters; "
        f"private key -> {key_path}"
    )
    print("  cases per rater:", {rater: len(rows) for rater, rows in assigned.items()})


def _design_cell(case) -> tuple[str, str, str, str]:
    return (
        case.label,
        case.stratum_id.removeprefix("safe:"),
        case.split,
        case.regime,
    )


def _balanced_rotating_assignment(
    cases: list, rater_ids: list[str], rng: random.Random
) -> dict[str, list]:
    """Assign two ratings per case and separate each direct/pressure pair."""
    if len(rater_ids) != 5:
        raise ValueError("researched rotating-pair annotation requires exactly five raters")
    rater_pairs = list(combinations(rater_ids, 2))
    totals = Counter({rater_id: 0 for rater_id in rater_ids})
    cell_totals: dict[tuple[str, str, str, str], Counter] = defaultdict(
        lambda: Counter({rater_id: 0 for rater_id in rater_ids})
    )
    case_raters: dict[str, tuple[str, str]] = {}
    by_family: dict[str, list] = defaultdict(list)
    for case in cases:
        by_family[case.base_task_id].append(case)

    paired = []
    singletons = []
    for base_task_id, family_cases in by_family.items():
        by_variant = {case.variant_id: case for case in family_cases}
        if {"direct", "pressure"} <= set(by_variant):
            paired.append((base_task_id, by_variant["direct"], by_variant["pressure"]))
            singletons.extend(
                case for case in family_cases if case.variant_id not in {"direct", "pressure"}
            )
        else:
            singletons.extend(family_cases)

    rng.shuffle(paired)
    for _, direct, pressure in paired:
        candidates = [
            (direct_pair, pressure_pair)
            for direct_pair in rater_pairs
            for pressure_pair in rater_pairs
            if set(direct_pair).isdisjoint(pressure_pair)
        ]
        rng.shuffle(candidates)

        def score(candidate: tuple[tuple[str, str], tuple[str, str]]) -> tuple[int, int, int]:
            direct_pair, pressure_pair = candidate
            projected = totals.copy()
            projected_cells = {
                _design_cell(direct): cell_totals[_design_cell(direct)].copy(),
                _design_cell(pressure): cell_totals[_design_cell(pressure)].copy(),
            }
            for case, pair in ((direct, direct_pair), (pressure, pressure_pair)):
                for rater_id in pair:
                    projected[rater_id] += 1
                    projected_cells[_design_cell(case)][rater_id] += 1
            return (
                max(projected.values()) - min(projected.values()),
                sum(value * value for value in projected.values()),
                sum(
                    value * value
                    for counts in projected_cells.values()
                    for value in counts.values()
                ),
            )

        direct_pair, pressure_pair = min(candidates, key=score)
        for case, pair in ((direct, direct_pair), (pressure, pressure_pair)):
            case_raters[case.case_id] = pair
            for rater_id in pair:
                totals[rater_id] += 1
                cell_totals[_design_cell(case)][rater_id] += 1

    singletons.sort(key=lambda case: case.case_id)
    rng.shuffle(singletons)
    for case in singletons:
        candidates = list(rater_pairs)
        rng.shuffle(candidates)
        cell = _design_cell(case)

        def score_single(pair: tuple[str, str]) -> tuple[int, int, int]:
            projected = totals.copy()
            projected_cell = cell_totals[cell].copy()
            for rater_id in pair:
                projected[rater_id] += 1
                projected_cell[rater_id] += 1
            return (
                max(projected.values()) - min(projected.values()),
                sum(value * value for value in projected.values()),
                sum(value * value for value in projected_cell.values()),
            )

        pair = min(candidates, key=score_single)
        case_raters[case.case_id] = pair
        for rater_id in pair:
            totals[rater_id] += 1
            cell_totals[cell][rater_id] += 1

    if set(case_raters) != {case.case_id for case in cases}:
        raise AssertionError("rotating assignment lost cases")
    for _, direct, pressure in paired:
        if not set(case_raters[direct.case_id]).isdisjoint(case_raters[pressure.case_id]):
            raise AssertionError("direct and pressure variants share a rater")

    assigned = {rater_id: [] for rater_id in rater_ids}
    by_id = {case.case_id: case for case in cases}
    for case_id, pair in case_raters.items():
        for rater_id in pair:
            assigned[rater_id].append(by_id[case_id])
    if sum(map(len, assigned.values())) != 2 * len(cases):
        raise AssertionError("rotating assignment does not contain two ratings per case")
    return assigned


def prepare_rotating_audit(
    *,
    d1_path: Path,
    blind_dir: Path,
    key_path: Path,
    rater_count: int,
    seed_file: Path | None,
) -> None:
    """Prepare the five-rater, two-ratings-per-case researched workflow."""
    if rater_count != 5:
        raise ValueError("researched annotation requires exactly five raters")
    _require_private_audit_paths(blind_dir=blind_dir, key_path=key_path)
    secret_seed = _secret_seed(seed_file)
    rng = random.Random(secret_seed)
    cases = _disjoint_review_cases(d1_path)
    rater_ids = [f"rater-{index}" for index in range(1, rater_count + 1)]
    assigned = _balanced_rotating_assignment(cases, rater_ids, rng)
    metadata_path = key_path.with_suffix(".metadata.json")
    instructions_path = blind_dir / "review-instructions.txt"
    output_paths = [key_path, metadata_path, instructions_path]
    output_paths.extend(
        blind_dir / f"label-audit-action-safety-rotating-{rater_id}.csv"
        for rater_id in rater_ids
    )
    _ensure_new(output_paths)
    blind_dir.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    case_raters: dict[str, list[str]] = defaultdict(list)
    for rater_id, rows in assigned.items():
        for case in rows:
            case_raters[case.case_id].append(rater_id)

    key_rows = []
    for rater_id in rater_ids:
        ordered = list(assigned[rater_id])
        rng.shuffle(ordered)
        blind_rows = []
        for case in ordered:
            audit_id = secrets.token_urlsafe(12)
            blind_rows.append({
                "audit_id": audit_id,
                "kind": case.kind,
                "payload": case.payload,
                "annotator_label": "",
            })
            slot = sorted(case_raters[case.case_id]).index(rater_id) + 1
            key_rows.append({
                "rater_id": rater_id,
                "rating_slot": str(slot),
                "audit_id": audit_id,
                "case_id": case.case_id,
                "base_task_id": case.base_task_id,
                "stratum_id": case.stratum_id.removeprefix("safe:"),
                "split": case.split,
                "regime": case.regime,
                "variant_id": case.variant_id,
                "oracle_label": case.label,
            })
        _write_csv(
            blind_dir / f"label-audit-action-safety-rotating-{rater_id}.csv",
            ROTATING_BLIND_FIELDS,
            blind_rows,
        )
    _write_csv(key_path, ROTATING_KEY_FIELDS, key_rows)
    instructions_path.write_text(
        "Complete annotator_label for every row using exactly safe or unsafe.\n"
        "Do not edit the other columns, use external facts, or try to identify related cases.\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps({
            "assignment": "two-ratings-rotating-pairs",
            "dataset_sha256": sha256(d1_path.read_bytes()).hexdigest(),
            "seed_commitment": sha256(str(secret_seed).encode("utf-8")).hexdigest(),
            "case_count": len(cases),
            "ratings_per_case": 2,
            "rating_count": 2 * len(cases),
            "rater_count": rater_count,
            "cases_per_rater": {
                rater_id: len(assigned[rater_id]) for rater_id in rater_ids
            },
            "paired_variant_rater_overlap": 0,
            "inter_rater_reliability_available": True,
            "rater_input_fields": ["annotator_label"],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"prepared {len(cases)} cases and {2 * len(cases)} ratings for five raters; "
        f"private key -> {key_path}"
    )


def _strict_csv(path: Path, fields: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"worksheet does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError(
                f"unexpected columns in {path}; expected {fields}, got {reader.fieldnames}"
            )
        return list(reader)


def _worksheet_rater_id(path: Path, prefix: str) -> str:
    if not path.stem.startswith(prefix):
        raise ValueError(f"worksheet name must begin with {prefix}: {path}")
    rater_id = path.stem.removeprefix(prefix)
    if not rater_id:
        raise ValueError(f"worksheet has no rater identifier: {path}")
    return rater_id


def _validated_disjoint_allocations(
    *,
    worksheet_paths: list[Path],
    key_rows: list[dict[str, str]],
) -> dict[str, list[tuple[dict[str, str], dict[str, str]]]]:
    """Validate exact, mutually exclusive primary-rater allocations."""
    if len(worksheet_paths) < 2:
        raise ValueError("disjoint scoring requires worksheets from at least two raters")
    key_by_rater: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    seen_case_ids = set()
    for row in key_rows:
        rater_id = row.get("rater_id", "")
        audit_id = row.get("audit_id", "")
        case_id = row.get("case_id", "")
        if not rater_id or not audit_id or not case_id:
            raise ValueError("private key contains an incomplete allocation")
        if audit_id in key_by_rater[rater_id]:
            raise ValueError(f"private key repeats audit_id for {rater_id}: {audit_id}")
        if case_id in seen_case_ids:
            raise ValueError(f"disjoint private key assigns a case more than once: {case_id}")
        key_by_rater[rater_id][audit_id] = row
        seen_case_ids.add(case_id)

    prefix = "label-audit-action-safety-disjoint-"
    paths_by_rater = {}
    for path in worksheet_paths:
        rater_id = _worksheet_rater_id(path, prefix)
        if rater_id in paths_by_rater:
            raise ValueError(f"duplicate worksheet for {rater_id}")
        paths_by_rater[rater_id] = path
    if set(paths_by_rater) != set(key_by_rater):
        raise ValueError("worksheets must be supplied for exactly the raters in the private key")

    allocations = {}
    for rater_id, path in sorted(paths_by_rater.items()):
        rows = _strict_csv(path, DISJOINT_BLIND_FIELDS)
        observed_ids = [row["audit_id"] for row in rows]
        if len(observed_ids) != len(set(observed_ids)):
            raise ValueError(f"worksheet repeats audit_id: {path}")
        expected = key_by_rater[rater_id]
        if set(observed_ids) != set(expected):
            missing = sorted(set(expected) - set(observed_ids))
            extra = sorted(set(observed_ids) - set(expected))
            raise ValueError(
                f"worksheet/key allocation mismatch for {rater_id}; "
                f"missing={missing}, extra={extra}"
            )
        allocations[rater_id] = [(row, expected[row["audit_id"]]) for row in rows]
    return allocations


def _parse_issue_flags(value: str) -> set[str]:
    raw = value.strip().lower()
    if not raw:
        return set()
    flags = {part.strip() for part in raw.split(";") if part.strip()}
    if "none" in flags:
        if len(flags) != 1:
            raise ValueError("issue_flags cannot combine 'none' with another value")
        return set()
    unknown = flags - ISSUE_FLAG_VALUES
    if unknown:
        raise ValueError(f"invalid issue_flags: {sorted(unknown)}")
    return flags


def _write_immutable_ledger(path: Path, payload: dict) -> None:
    if _is_within(path, ROOT):
        raise ValueError(f"audit ledger must be outside the source repository: {path}")
    _ensure_new([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def score_disjoint_audit(
    *,
    blind_paths: list[Path],
    key_path: Path,
    require_complete: bool,
    consolidated_path: Path,
    ledger_path: Path | None = None,
) -> dict:
    """Validate and consolidate one-primary-rater worksheets."""
    for blind_path in blind_paths:
        _require_private_audit_paths(blind_dir=blind_path.parent, key_path=key_path)
    _require_outside_source(consolidated_path)
    _ensure_new([consolidated_path])
    key_rows = _read_csv(key_path)
    allocations = _validated_disjoint_allocations(
        worksheet_paths=blind_paths,
        key_rows=key_rows,
    )
    consolidated_rows = []
    rater_summaries = []
    cases_requiring_adjudication = []
    cases_with_insufficient_context = []
    cases_with_realism_failures = []
    completed_case_ids = []
    candidate_pairs = []
    incomplete = False
    for rater_id, allocation in sorted(allocations.items()):
        summary = Counter()
        summary["allocated"] = len(allocation)
        for row, mapping in allocation:
            label = row["annotator_label"].strip().lower()
            raw_issue_flags = row["issue_flags"].strip().lower()
            confidence = row["confidence"].strip().lower()
            notes = row["reviewer_notes"].strip()
            missing = [
                field
                for field, value in (
                    ("annotator_label", label),
                    ("issue_flags", raw_issue_flags),
                    ("confidence", confidence),
                )
                if not value
            ]
            if missing and require_complete:
                raise ValueError(
                    f"incomplete worksheet for {rater_id}, case {mapping['case_id']}: {missing}"
                )
            if label and label not in VALID_ANNOTATOR_LABELS:
                raise ValueError(f"invalid annotation label {label!r} for {rater_id}")
            if confidence and confidence not in CONFIDENCE_VALUES:
                raise ValueError(f"invalid confidence {confidence!r} for {rater_id}")
            try:
                issue_flags = _parse_issue_flags(raw_issue_flags)
            except ValueError as error:
                raise ValueError(f"{error} for {rater_id}, case {mapping['case_id']}") from error

            reasons = []
            if missing:
                incomplete = True
                summary["incomplete"] += 1
                reasons.append("incomplete")
            else:
                completed_case_ids.append(mapping["case_id"])
                summary["completed"] += 1
            if label == "unclear":
                reasons.append("unclear_label")
            elif label and label != mapping["oracle_label"]:
                reasons.append("candidate_label_mismatch")
            if issue_flags:
                reasons.extend(sorted(issue_flags))
            if confidence == "low":
                reasons.append("low_confidence")
            if "insufficient_context" in issue_flags:
                cases_with_insufficient_context.append(mapping["case_id"])
            if issue_flags & {"unrealistic", "implausible_workflow", "template_cue"}:
                cases_with_realism_failures.append(mapping["case_id"])
            if reasons and reasons != ["incomplete"]:
                cases_requiring_adjudication.append(mapping["case_id"])
                summary["requires_adjudication"] += 1
            elif not reasons:
                summary["accepted"] += 1
            if label in {"safe", "unsafe"}:
                candidate_pairs.append((label, mapping["oracle_label"]))
            consolidated_rows.append({
                "case_id": mapping["case_id"],
                "base_task_id": mapping["base_task_id"],
                "stratum_id": mapping["stratum_id"],
                "split": mapping["split"],
                "regime": mapping["regime"],
                "primary_rater_id": rater_id,
                "candidate_label": mapping["oracle_label"],
                "annotator_label": label,
                "issue_flags": ";".join(sorted(issue_flags)) if raw_issue_flags else "",
                "confidence": confidence,
                "review_status": (
                    "incomplete"
                    if missing
                    else "requires_adjudication" if reasons else "accepted"
                ),
                "reviewer_notes": notes,
            })
        rater_summaries.append({"rater_id": rater_id, **dict(sorted(summary.items()))})

    consolidated_rows.sort(key=lambda row: row["case_id"])
    consolidated_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(
        consolidated_path,
        [
            "case_id",
            "base_task_id",
            "stratum_id",
            "split",
            "regime",
            "primary_rater_id",
            "candidate_label",
            "annotator_label",
            "issue_flags",
            "confidence",
            "review_status",
            "reviewer_notes",
        ],
        consolidated_rows,
    )
    agreement = {"n": len(candidate_pairs)}
    if candidate_pairs:
        reviewed, candidate = zip(*candidate_pairs)
        agreement["raw"] = _p_observed(reviewed, candidate)
    expected_cases = len(key_rows)
    result = {
        "status": (
            "incomplete"
            if incomplete
            else "awaiting_adjudication"
            if cases_requiring_adjudication
            else "review_complete"
        ),
        "annotation_design": "one-primary-rater-per-case",
        "labels_per_case": 1,
        "rater_count": len(allocations),
        "rater_summaries": rater_summaries,
        "reviewed_case_ids": sorted(completed_case_ids),
        "coverage": {
            "expected_cases": expected_cases,
            "reviewed_cases": len(completed_case_ids),
            "complete": not incomplete and len(completed_case_ids) == expected_cases,
        },
        "candidate_label_agreement": agreement,
        "between_rater": {
            "available": False,
            "reason": "each case was assigned to one primary rater",
        },
        "cases_requiring_adjudication": sorted(set(cases_requiring_adjudication)),
        "cases_with_insufficient_context": sorted(set(cases_with_insufficient_context)),
        "cases_with_realism_failures": sorted(set(cases_with_realism_failures)),
        "worksheet_hashes": {str(path): _file_sha256(path) for path in blind_paths},
        "key_sha256": _file_sha256(key_path),
        "consolidated_sha256": _file_sha256(consolidated_path),
    }
    if ledger_path is not None:
        _write_immutable_ledger(ledger_path, result)
    print(
        f"consolidated {len(completed_case_ids)}/{expected_cases} reviews; "
        f"adjudication required for {len(result['cases_requiring_adjudication'])} cases"
    )
    return result


def _validated_rotating_allocations(
    *, worksheet_paths: list[Path], key_path: Path
) -> tuple[
    dict[str, list[tuple[dict[str, str], dict[str, str]]]],
    list[dict[str, str]],
]:
    if len(worksheet_paths) != 5:
        raise ValueError("rotating scoring requires exactly five worksheets")
    key_rows = _strict_csv(key_path, ROTATING_KEY_FIELDS)
    key_by_rater: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    audit_ids = set()
    for row in key_rows:
        required = ("rater_id", "rating_slot", "audit_id", "case_id", "base_task_id")
        if any(not row[field].strip() for field in required):
            raise ValueError("rotating private key contains an incomplete allocation")
        if row["audit_id"] in audit_ids:
            raise ValueError(f"rotating private key repeats audit_id: {row['audit_id']}")
        audit_ids.add(row["audit_id"])
        if row["audit_id"] in key_by_rater[row["rater_id"]]:
            raise ValueError(f"private key repeats audit_id for {row['rater_id']}")
        key_by_rater[row["rater_id"]][row["audit_id"]] = row
        by_case[row["case_id"]].append(row)
    if len(key_by_rater) != 5:
        raise ValueError("rotating private key must contain exactly five raters")
    for case_id, rows in by_case.items():
        if len(rows) != 2:
            raise ValueError(f"case must have exactly two allocations: {case_id}")
        if len({row["rater_id"] for row in rows}) != 2:
            raise ValueError(f"case repeats a rater: {case_id}")
        if {row["rating_slot"] for row in rows} != {"1", "2"}:
            raise ValueError(f"case rating slots must be 1 and 2: {case_id}")

    family_variants: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for rows in by_case.values():
        sample = rows[0]
        family_variants[sample["base_task_id"]][sample["variant_id"]] = {
            row["rater_id"] for row in rows
        }
    for base_task_id, variants in family_variants.items():
        if {"direct", "pressure"} <= set(variants) and not variants["direct"].isdisjoint(
            variants["pressure"]
        ):
            raise ValueError(f"direct/pressure pair shares a rater: {base_task_id}")

    prefix = "label-audit-action-safety-rotating-"
    paths_by_rater = {}
    for path in worksheet_paths:
        rater_id = _worksheet_rater_id(path, prefix)
        if rater_id in paths_by_rater:
            raise ValueError(f"duplicate worksheet for {rater_id}")
        paths_by_rater[rater_id] = path
    if set(paths_by_rater) != set(key_by_rater):
        raise ValueError("worksheets must be supplied for exactly the raters in the key")

    allocations = {}
    for rater_id, path in sorted(paths_by_rater.items()):
        rows = _strict_csv(path, ROTATING_BLIND_FIELDS)
        observed_ids = [row["audit_id"] for row in rows]
        if len(observed_ids) != len(set(observed_ids)):
            raise ValueError(f"worksheet repeats audit_id: {path}")
        expected = key_by_rater[rater_id]
        if set(observed_ids) != set(expected):
            raise ValueError(f"worksheet/key allocation mismatch for {rater_id}")
        allocations[rater_id] = [(row, expected[row["audit_id"]]) for row in rows]
    return allocations, key_rows


def _gwet_ac1(label_pairs: list[tuple[str, str]]) -> float | None:
    if not label_pairs:
        return None
    observed = _p_observed(
        [pair[0] for pair in label_pairs], [pair[1] for pair in label_pairs]
    )
    counts = Counter(label for pair in label_pairs for label in pair)
    total = 2 * len(label_pairs)
    categories = sorted(VALID_ROTATING_LABELS)
    chance = sum(
        (counts[category] / total) * (1 - counts[category] / total)
        for category in categories
    ) / (len(categories) - 1)
    if chance >= 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - chance) / (1 - chance)


def score_rotating_audit(
    *,
    blind_paths: list[Path],
    key_path: Path,
    require_complete: bool,
    consolidated_path: Path,
    ledger_path: Path | None = None,
) -> dict:
    """Consolidate double-coded worksheets and calculate raw agreement and AC1."""
    for blind_path in blind_paths:
        _require_private_audit_paths(blind_dir=blind_path.parent, key_path=key_path)
    _require_outside_source(consolidated_path)
    _ensure_new([consolidated_path])
    allocations, key_rows = _validated_rotating_allocations(
        worksheet_paths=blind_paths, key_path=key_path
    )
    rating_rows = []
    rater_summaries = []
    incomplete = False
    for rater_id, allocation in sorted(allocations.items()):
        summary = Counter(allocated=len(allocation))
        for row, mapping in allocation:
            label = row["annotator_label"].strip().lower()
            missing = ["annotator_label"] if not label else []
            if missing and require_complete:
                raise ValueError(
                    f"incomplete worksheet for {rater_id}, case {mapping['case_id']}: {missing}"
                )
            if label and label not in VALID_ROTATING_LABELS:
                raise ValueError(f"invalid annotation label {label!r} for {rater_id}")
            if missing:
                incomplete = True
                summary["incomplete"] += 1
            else:
                summary["completed"] += 1
            rating_rows.append({
                "case_id": mapping["case_id"],
                "base_task_id": mapping["base_task_id"],
                "stratum_id": mapping["stratum_id"],
                "split": mapping["split"],
                "regime": mapping["regime"],
                "variant_id": mapping["variant_id"],
                "rater_id": rater_id,
                "rating_slot": mapping["rating_slot"],
                "candidate_label": mapping["oracle_label"],
                "annotator_label": label,
            })
        rater_summaries.append({"rater_id": rater_id, **dict(sorted(summary.items()))})

    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rating_rows:
        by_case[row["case_id"]].append(row)
    label_pairs = []
    reviewed_case_ids = []
    adjudication_reasons: dict[str, list[str]] = {}
    for case_id, rows in sorted(by_case.items()):
        if len(rows) != 2:
            raise AssertionError(f"validated case lost a rating: {case_id}")
        if any(not row["annotator_label"] for row in rows):
            continue
        reviewed_case_ids.append(case_id)
        labels = (rows[0]["annotator_label"], rows[1]["annotator_label"])
        label_pairs.append(labels)
        reasons = []
        if labels[0] != labels[1]:
            reasons.append("rater_disagreement")
        if any(row["annotator_label"] != row["candidate_label"] for row in rows):
            reasons.append("candidate_label_mismatch")
        if reasons:
            adjudication_reasons[case_id] = sorted(set(reasons))

    rating_rows.sort(key=lambda row: (row["case_id"], row["rating_slot"]))
    consolidated_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated_fields = [
        "case_id", "base_task_id", "stratum_id", "split", "regime", "variant_id",
        "rater_id", "rating_slot", "candidate_label", "annotator_label",
    ]
    _write_csv(consolidated_path, consolidated_fields, rating_rows)
    raw = _p_observed(
        [pair[0] for pair in label_pairs], [pair[1] for pair in label_pairs]
    ) if label_pairs else None
    ac1 = _gwet_ac1(label_pairs)
    expected_cases = len({row["case_id"] for row in key_rows})
    coverage_complete = not incomplete and len(reviewed_case_ids) == expected_cases
    result = {
        "status": (
            "incomplete" if not coverage_complete
            else "awaiting_adjudication" if adjudication_reasons
            else "review_complete_no_adjudication_needed"
        ),
        "annotation_design": "two-ratings-rotating-pairs",
        "labels_per_case": 2,
        "rater_count": len(allocations),
        "rater_summaries": rater_summaries,
        "reviewed_case_ids": reviewed_case_ids,
        "coverage": {
            "expected_cases": expected_cases,
            "reviewed_cases": len(reviewed_case_ids),
            "expected_ratings": 2 * expected_cases,
            "observed_ratings": sum(
                bool(row["annotator_label"])
                for row in rating_rows
            ),
            "complete": coverage_complete,
        },
        "between_rater": {
            "available": bool(label_pairs),
            "n": len(label_pairs),
            "raw": raw,
            "ac1": ac1,
            "categories": sorted(VALID_ROTATING_LABELS),
        },
        "cases_requiring_adjudication": sorted(adjudication_reasons),
        "adjudication_reasons": adjudication_reasons,
        "cases_with_insufficient_context": [],
        "cases_with_realism_failures": [],
        "worksheet_hashes": {str(path): _file_sha256(path) for path in blind_paths},
        "key_sha256": _file_sha256(key_path),
        "consolidated_sha256": _file_sha256(consolidated_path),
    }
    if ledger_path is not None:
        _write_immutable_ledger(ledger_path, result)
    print(
        f"consolidated {len(reviewed_case_ids)}/{expected_cases} double-coded cases; "
        f"adjudication required for {len(adjudication_reasons)} cases"
    )
    return result


def import_adjudication(
    *, review_ledger_path: Path, adjudication_path: Path, output_path: Path
) -> dict:
    """Import locked decisions from a separate senior adjudicator."""
    _require_outside_source(review_ledger_path, adjudication_path, output_path)
    _ensure_new([output_path])
    review = json.loads(review_ledger_path.read_text(encoding="utf-8"))
    required = set(review.get("cases_requiring_adjudication", []))
    rows = _strict_csv(adjudication_path, ADJUDICATION_FIELDS)
    observed = [row["case_id"].strip() for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError("adjudication worksheet repeats case_id")
    if set(observed) != required:
        raise ValueError(
            f"adjudication coverage mismatch; missing={sorted(required - set(observed))}, "
            f"extra={sorted(set(observed) - required)}"
        )
    rater_ids = {row["rater_id"] for row in review.get("rater_summaries", [])}
    remediation = []
    normalized = []
    for row in rows:
        adjudicator_id = row["adjudicator_id"].strip()
        label = row["adjudicated_label"].strip().lower()
        disposition = row["disposition"].strip().lower()
        if not adjudicator_id or adjudicator_id in rater_ids:
            raise ValueError("adjudicator must be identified and separate from the five labelers")
        if label not in {"safe", "unsafe"}:
            raise ValueError(f"invalid adjudicated label for {row['case_id']}")
        if disposition not in {"accept", "oracle_correction", "revise", "exclude"}:
            raise ValueError(f"invalid adjudication disposition for {row['case_id']}")
        if not row["rationale"].strip() or not row["decision_locked_at"].strip():
            raise ValueError(f"adjudication rationale and lock time are required: {row['case_id']}")
        if disposition in {"revise", "exclude"}:
            remediation.append(row["case_id"])
        normalized.append({**row, "adjudicated_label": label, "disposition": disposition})
    result = {
        **review,
        "pre_adjudication_status": review.get("status"),
        "status": "blocked_remediation_required" if remediation else "adjudicated",
        "cases_requiring_adjudication": remediation,
        "cases_with_insufficient_context": [],
        "cases_with_realism_failures": [],
        "adjudication": normalized,
        "adjudication_worksheet_sha256": _file_sha256(adjudication_path),
        "review_ledger_sha256": _file_sha256(review_ledger_path),
    }
    _write_immutable_ledger(output_path, result)
    return result


def _p_observed(a, b) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    prepare_parser = subparsers.add_parser(
        "prepare-disjoint",
        help="assign every Action Safety case to exactly one primary rater",
    )
    prepare_parser.add_argument("--d1-path", type=Path, required=True)
    prepare_parser.add_argument("--blind-dir", type=Path, required=True)
    prepare_parser.add_argument("--key", type=Path, required=True)
    prepare_parser.add_argument("--seed-file", type=Path)
    prepare_parser.add_argument("--raters", type=int, required=True)
    score_parser = subparsers.add_parser(
        "score-disjoint",
        help="validate and consolidate disjoint primary-rater worksheets",
    )
    score_parser.add_argument("--blind", type=Path, nargs="+", required=True)
    score_parser.add_argument("--key", type=Path, required=True)
    score_parser.add_argument("--require-complete", action="store_true")
    score_parser.add_argument("--consolidated", type=Path, required=True)
    score_parser.add_argument(
        "--ledger",
        type=Path,
        help="new private, immutable review ledger path",
    )
    rotating_prepare = subparsers.add_parser(
        "prepare-rotating",
        help="assign two ratings per case across five rotating labelers",
    )
    rotating_prepare.add_argument("--d1-path", type=Path, required=True)
    rotating_prepare.add_argument("--blind-dir", type=Path, required=True)
    rotating_prepare.add_argument("--key", type=Path, required=True)
    rotating_prepare.add_argument("--seed-file", type=Path)
    rotating_prepare.add_argument("--raters", type=int, default=5)
    rotating_score = subparsers.add_parser(
        "score-rotating",
        help="score five rotating-pair worksheets and calculate raw agreement and AC1",
    )
    rotating_score.add_argument("--blind", type=Path, nargs="+", required=True)
    rotating_score.add_argument("--key", type=Path, required=True)
    rotating_score.add_argument("--require-complete", action="store_true")
    rotating_score.add_argument("--consolidated", type=Path, required=True)
    rotating_score.add_argument("--ledger", type=Path)
    adjudication_parser = subparsers.add_parser(
        "import-adjudication",
        help="import a separate senior adjudicator's locked decisions",
    )
    adjudication_parser.add_argument("--review-ledger", type=Path, required=True)
    adjudication_parser.add_argument("--adjudication", type=Path, required=True)
    adjudication_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "prepare-disjoint":
        prepare_disjoint_audit(
            d1_path=args.d1_path,
            blind_dir=args.blind_dir,
            key_path=args.key,
            rater_count=args.raters,
            seed_file=args.seed_file,
        )
    elif args.cmd == "score-disjoint":
        score_disjoint_audit(
            blind_paths=args.blind,
            key_path=args.key,
            require_complete=args.require_complete,
            consolidated_path=args.consolidated,
            ledger_path=args.ledger,
        )
    elif args.cmd == "prepare-rotating":
        prepare_rotating_audit(
            d1_path=args.d1_path,
            blind_dir=args.blind_dir,
            key_path=args.key,
            rater_count=args.raters,
            seed_file=args.seed_file,
        )
    elif args.cmd == "score-rotating":
        score_rotating_audit(
            blind_paths=args.blind,
            key_path=args.key,
            require_complete=args.require_complete,
            consolidated_path=args.consolidated,
            ledger_path=args.ledger,
        )
    else:
        import_adjudication(
            review_ledger_path=args.review_ledger,
            adjudication_path=args.adjudication,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
