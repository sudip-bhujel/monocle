from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import date
import json
import math
from pathlib import Path
import re

import numpy as np


CUES = (
    "before",
    "without",
    "requires",
    "restricted",
    "must",
    "only",
    "policy",
    "approval",
    "confirm",
    "scope",
    "review",
)
PROVENANCE_FIELDS = (
    "source_type",
    "source_group",
    "source_reference",
    "source_title",
    "retrieval_date",
    "adaptation_notes",
    "license_applicable",
    "license",
)
PLACEHOLDERS = {
    "", "none", "null", "n/a", "na", "unknown", "tbd", "todo", "pending",
    "placeholder", "example",
}
SYNTHETIC_SOURCE_TYPES = {"synthetic", "template", "generated", "llm-generated", "llm_generated"}
TEMPLATE_ID_FIELDS = (
    "construction_template_id",
    "construction_cluster_id",
    "source_pattern_id",
    "template_id",
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9-]+", text.lower())


def _is_placeholder(value: object) -> bool:
    rendered = str(value or "").strip().lower()
    return rendered in PLACEHOLDERS or any(
        marker in rendered for marker in ("placeholder", "pending", "todo", "tbd")
    )


def _rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [
        row
        for row in rows
        if row["split"] in {"calibrate", "select", "holdout", "final"}
    ]


def _provenance_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [
        row
        for row in rows
        if row["split"] in {"calibrate", "select", "holdout", "final", "stress"}
    ]


def _metadata_rows(path: Path | None) -> list[dict]:
    if path is None:
        return []
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("rows", value.get("families", []))
        if not isinstance(value, list):
            raise ValueError("metadata JSON must be a list or contain rows/families")
        return value
    raise ValueError("metadata must be CSV, JSONL, or JSON")


def _join_metadata(rows: list[dict], metadata: list[dict]) -> list[dict]:
    if not metadata:
        return rows
    by_case = {row["case_id"]: row for row in metadata if row.get("case_id")}
    by_family = {
        row["base_task_id"]: row for row in metadata if row.get("base_task_id")
    }
    joined = []
    for row in rows:
        values = by_case.get(row["case_id"], by_family.get(row["base_task_id"], {}))
        joined.append({**row, **{key: value for key, value in values.items() if key not in row}})
    return joined


def _folds(rows: list[dict], count: int) -> list[int]:
    families = sorted({row["base_task_id"].removeprefix("safe-") for row in rows})
    assignment = {family: index % count for index, family in enumerate(families)}
    return [assignment[row["base_task_id"].removeprefix("safe-")] for row in rows]


def _tfidf(rows: list[dict]) -> np.ndarray:
    documents = [Counter(_tokens(row["payload"])) for row in rows]
    document_frequency = Counter()
    for document in documents:
        document_frequency.update(document)
    vocabulary = sorted(token for token, count in document_frequency.items() if count >= 2)
    index = {token: column for column, token in enumerate(vocabulary)}
    matrix = np.zeros((len(rows), len(vocabulary)), dtype=float)
    for row_index, document in enumerate(documents):
        total = sum(document.values()) or 1
        for token, count in document.items():
            if token in index:
                inverse = math.log((1 + len(rows)) / (1 + document_frequency[token])) + 1
                matrix[row_index, index[token]] = count / total * inverse
    return matrix


def _surface(rows: list[dict]) -> np.ndarray:
    values = []
    for row in rows:
        text = row["payload"].lower()
        values.append(
            [
                len(text) / 400,
                text.count(".") / 4,
                float(row["kind"] == "shell"),
                *[float(cue in text) for cue in CUES],
            ]
        )
    return np.asarray(values, dtype=float)


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1
    train = np.column_stack([np.ones(len(train_x)), (train_x - mean) / scale])
    test = np.column_stack([np.ones(len(test_x)), (test_x - mean) / scale])
    weights = np.zeros(train.shape[1])
    for _ in range(400):
        scores = np.clip(train @ weights, -30, 30)
        probabilities = 1 / (1 + np.exp(-scores))
        gradient = train.T @ (probabilities - train_y) / len(train)
        gradient[1:] += 0.01 * weights[1:]
        weights -= 0.2 * gradient
    return (test @ weights >= 0).astype(int)


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    positive = predictions[labels == 1].mean()
    negative = (1 - predictions[labels == 0]).mean()
    return float((positive + negative) / 2)


def grouped_score(rows: list[dict], features: np.ndarray, folds: int = 5) -> float:
    labels = np.asarray([row["label"] == "unsafe" for row in rows], dtype=int)
    assignments = np.asarray(_folds(rows, folds))
    predictions = np.zeros(len(rows), dtype=int)
    for fold in range(folds):
        train = assignments != fold
        test = assignments == fold
        predictions[test] = _fit_predict(features[train], labels[train], features[test])
    return _balanced_accuracy(labels, predictions)


def split_duplicate_audit(
    rows: list[dict], *, similarity_threshold: float = 0.90
) -> dict[str, object]:
    """Detect lexical near-duplicates and declared template reuse across splits."""
    if not 0 < similarity_threshold <= 1:
        raise ValueError("similarity threshold must be in (0, 1]")
    count = len(rows)
    if count < 2:
        return {
            "status": "pass",
            "similarity_threshold": similarity_threshold,
            "cross_split_near_duplicate_pairs": 0,
            "cross_split_template_ids": [],
            "examples": [],
        }

    features = _tfidf(rows)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = np.divide(
        features,
        norms,
        out=np.zeros_like(features),
        where=norms > 0,
    )
    similarity = normalized @ normalized.T
    splits = np.asarray([str(row["split"]) for row in rows], dtype=object)
    families = np.asarray([str(row["base_task_id"]) for row in rows], dtype=object)
    cross_split = splits[:, None] != splits[None, :]
    different_family = families[:, None] != families[None, :]
    upper_triangle = np.triu(np.ones((count, count), dtype=bool), k=1)
    pairs = np.argwhere(
        (similarity >= similarity_threshold)
        & cross_split
        & different_family
        & upper_triangle
    )
    examples = [
        {
            "first_case_id": str(rows[first]["case_id"]),
            "first_split": str(rows[first]["split"]),
            "second_case_id": str(rows[second]["case_id"]),
            "second_split": str(rows[second]["split"]),
            "similarity": float(similarity[first, second]),
        }
        for first, second in pairs[:20]
    ]

    template_splits: dict[str, set[str]] = {}
    for row in rows:
        for field in TEMPLATE_ID_FIELDS:
            value = str(row.get(field, "")).strip()
            if value and not _is_placeholder(value):
                template_splits.setdefault(f"{field}:{value}", set()).add(
                    str(row["split"])
                )
                break
    reused_templates = sorted(
        template_id
        for template_id, used_splits in template_splits.items()
        if len(used_splits) > 1
    )
    passed = not len(pairs) and not reused_templates
    return {
        "status": "pass" if passed else "blocked",
        "similarity_threshold": similarity_threshold,
        "cross_split_near_duplicate_pairs": int(len(pairs)),
        "cross_split_template_ids": reused_templates,
        "examples": examples,
    }


def leave_source_out_score(
    rows: list[dict], features: np.ndarray, source_field: str
) -> dict[str, object]:
    """Evaluate source leakage with each declared source held out in turn."""
    sources = [str(row.get(source_field, "")).strip() for row in rows]
    if not any(sources):
        return {"available": False, "reason": f"metadata field {source_field!r} is absent"}
    if not all(sources):
        return {"available": False, "reason": f"metadata field {source_field!r} is incomplete"}
    unique_sources = sorted(set(sources))
    if len(unique_sources) < 2:
        return {"available": False, "reason": "leave-source-out requires at least two sources"}
    labels = np.asarray([row["label"] == "unsafe" for row in rows], dtype=int)
    source_array = np.asarray(sources)
    predictions = np.zeros(len(rows), dtype=int)
    fold_rows = []
    for source in unique_sources:
        test = source_array == source
        train = ~test
        if len(set(labels[test])) < 2 or len(set(labels[train])) < 2:
            return {
                "available": False,
                "reason": f"source {source!r} does not permit a two-class train/test fold",
            }
        predictions[test] = _fit_predict(features[train], labels[train], features[test])
        fold_rows.append({
            "source": source,
            "rows": int(test.sum()),
            "balanced_accuracy": _balanced_accuracy(labels[test], predictions[test]),
        })
    return {
        "available": True,
        "source_field": source_field,
        "sources": len(unique_sources),
        "balanced_accuracy": _balanced_accuracy(labels, predictions),
        "folds": fold_rows,
    }


def provenance_audit(
    rows: list[dict], *, metadata_supplied: bool, source_field: str,
    min_source_groups: int = 3, max_source_share: float = 0.40,
) -> dict[str, object]:
    """Check real-world grounding and prevent source instances crossing splits."""
    if not metadata_supplied:
        return {"available": False, "status": "blocked", "reasons": ["provenance metadata was not supplied"]}
    reasons = []
    by_family: dict[str, list[dict]] = {}
    for row in rows:
        by_family.setdefault(str(row["base_task_id"]), []).append(row)
    source_groups = Counter()
    source_types = Counter()
    references_by_split: dict[str, set[str]] = {}
    for family_id, family_rows in sorted(by_family.items()):
        first = family_rows[0]
        for field in PROVENANCE_FIELDS:
            values = {str(row.get(field, "")).strip() for row in family_rows}
            if len(values) != 1:
                reasons.append(f"family {family_id} has inconsistent {field}")
                continue
            value = next(iter(values))
            if _is_placeholder(value):
                reasons.append(f"family {family_id} has missing or placeholder {field}")
        source_group = str(first.get(source_field, "")).strip()
        source_type = str(first.get("source_type", "")).strip().lower()
        reference = str(first.get("source_reference", "")).strip()
        source_url = str(first.get("source_url", "")).strip()
        source_groups[source_group] += 1
        source_types[source_type] += 1
        resolvable = reference.startswith(("https://", "http://", "doi:")) or source_url.startswith(
            ("https://", "http://")
        )
        if reference and not resolvable:
            reasons.append(f"family {family_id} has a non-resolvable source reference")
        retrieval = str(first.get("retrieval_date", "")).strip()
        try:
            if date.fromisoformat(retrieval) > date.today():
                reasons.append(f"family {family_id} has a future retrieval date")
        except ValueError:
            reasons.append(f"family {family_id} has a non-ISO retrieval date")
        if reference:
            references_by_split.setdefault(reference, set()).update(
                str(row["split"]) for row in family_rows
            )
    valid_groups = {group for group in source_groups if not _is_placeholder(group)}
    if len(valid_groups) < min_source_groups:
        reasons.append(f"dataset has {len(valid_groups)} source groups; requires {min_source_groups}")
    family_count = len(by_family)
    observed_share = max(source_groups.values(), default=0) / family_count if family_count else 1.0
    if observed_share > max_source_share:
        reasons.append(
            f"largest source group covers {observed_share:.3f} of families; maximum is {max_source_share:.3f}"
        )
    if source_types and all(
        source_type in SYNTHETIC_SOURCE_TYPES
        or any(marker in source_type for marker in ("synthetic", "template", "generated"))
        for source_type in source_types
    ):
        reasons.append("synthetic/template-only datasets cannot pass provenance review")
    leaked = sorted(reference for reference, splits in references_by_split.items() if len(splits) > 1)
    if leaked:
        reasons.append(f"{len(leaked)} source references cross dataset splits")
    return {
        "available": True,
        "status": "pass" if not reasons else "blocked",
        "reasons": list(dict.fromkeys(reasons)),
        "families": family_count,
        "source_group_counts": dict(sorted(source_groups.items())),
        "source_type_counts": dict(sorted(source_types.items())),
        "max_source_share": observed_share,
        "cross_split_source_examples": leaked[:20],
    }


def audit(
    path: Path,
    threshold: float = 0.75,
    metadata_path: Path | None = None,
    source_field: str = "source_group",
) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    metadata = _metadata_rows(metadata_path)
    rows = _join_metadata(_rows(path), metadata)
    provenance_rows = _join_metadata(_provenance_rows(path), metadata)
    surface = grouped_score(rows, _surface(rows))
    tfidf = grouped_score(rows, _tfidf(rows))
    source_surface = leave_source_out_score(rows, _surface(rows), source_field)
    source_tfidf = leave_source_out_score(rows, _tfidf(rows), source_field)
    available_source_scores = [
        float(result["balanced_accuracy"])
        for result in (source_surface, source_tfidf)
        if result.get("available")
    ]
    best = max([surface, tfidf, *available_source_scores])
    provenance = provenance_audit(
        provenance_rows,
        metadata_supplied=metadata_path is not None,
        source_field=source_field,
    )
    duplicates = split_duplicate_audit(rows)
    strict_source_ok = metadata_path is None or (
        provenance["status"] == "pass"
        and all(result.get("available") for result in (source_surface, source_tfidf))
    )
    return {
        "dataset": str(path),
        "rows": len(rows),
        "family_grouped_folds": 5,
        "surface_balanced_accuracy": surface,
        "tfidf_balanced_accuracy": tfidf,
        "leave_source_out": {
            "surface": source_surface,
            "tfidf": source_tfidf,
        },
        "provenance": provenance,
        "split_disjointness": duplicates,
        "threshold": threshold,
        "status": (
            "pass"
            if best <= threshold
            and strict_source_ok
            and duplicates["status"] == "pass"
            else "blocked"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases", default="data/monocle_action_safety.jsonl", type=Path
    )
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--source-field", default="source_group")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.cases, args.threshold, args.metadata, args.source_field)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
