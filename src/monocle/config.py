from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    content = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(content) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_experiment(path: str | Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {"fpr_targets": [0.05], "bootstrap_draws": 200, "seed": 1}
    return load_yaml(path).get("experiment", {})


def fpr_targets(path: str | Path | None) -> list[float]:
    targets = load_experiment(path).get("fpr_targets", [0.05])
    values = [float(target) for target in targets]
    if not values or any(target < 0 or target > 1 for target in values):
        raise ValueError("experiment fpr_targets must be values in [0, 1]")
    return values


def hash_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_files(paths: list[str | Path]) -> dict[str, str]:
    out = {}
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            out[str(candidate)] = hash_file(candidate)
    return out


def hash_mapping(values: dict[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_env(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip("\"'")
