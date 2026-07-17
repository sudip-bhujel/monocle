from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def any_flag(flags: Mapping[str, bool]) -> bool:
    return any(flags.values())


def majority(flags: Mapping[str, bool], tie_blocks: bool = True) -> bool:
    caught = sum(flags.values())
    missed = len(flags) - caught
    if caught == missed:
        return tie_blocks
    return caught > missed


def weighted_threshold(
    flags: Mapping[str, bool],
    weights: Mapping[str, float],
    threshold: float,
) -> bool:
    missing = set(flags) - set(weights)
    if missing:
        raise ValueError(f"missing weights for monitors: {sorted(missing)}")
    return sum(weights[key] for key, value in flags.items() if value) >= threshold


def escalation(
    flags: Mapping[str, bool],
    *,
    base_rule: str = "majority",
    escalation_accuracy: float = 1.0,
) -> bool:
    if not 0 <= escalation_accuracy <= 1:
        raise ValueError("escalation_accuracy must be in [0, 1]")
    if len(set(flags.values())) == 1:
        return any_flag(flags)
    if escalation_accuracy == 1:
        return True
    if escalation_accuracy == 0:
        return False
    if base_rule == "any_flag":
        return any_flag(flags)
    return majority(flags)


def poisson_binomial_pmf(probabilities: list[float]) -> np.ndarray:
    pmf = np.array([1.0])
    for probability in probabilities:
        pmf = np.convolve(pmf, np.array([1 - probability, probability]))
    return pmf
