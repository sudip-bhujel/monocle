from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Split = Literal["calibrate", "select", "attack_dev", "final"]
Regime = Literal["safe", "non_adversarial", "adversarial"]
ThresholdScope = Literal["component", "committee"]
AggregationRule = Literal["any_flag", "majority", "weighted", "escalation"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class BaseTask(StrictModel):
    base_task_id: str
    task_family: str
    source_dataset: str
    target_class: Literal["safe", "unsafe"]


class AttackVariant(StrictModel):
    attack_template_id: str
    variant_id: str
    attack_class: str
    knowledge: Literal["black_box", "gray_box", "white_box"]
    generation_seed: int = 0
    generation_config: dict[str, Any] = Field(default_factory=dict)
    base_task_id: str


class Case(StrictModel):
    case_id: str
    base_task_id: str
    variant_id: str
    kind: str
    payload: str
    label: Literal["safe", "unsafe"]
    regime: Regime
    attack_class: str
    knowledge: Literal["none", "black_box", "gray_box", "white_box"]
    difficulty_z: float = 0.0
    stratum_id: str
    target_weight: float = Field(ge=0)
    split: Split
    oracle_id: str
    oracle_version: str

    @model_validator(mode="after")
    def regime_matches_label(self) -> Case:
        if self.label == "safe" and self.regime != "safe":
            raise ValueError("safe cases must use regime='safe'")
        if self.label == "unsafe" and self.regime == "safe":
            raise ValueError("unsafe cases cannot use regime='safe'")
        return self


class MonitorConfig(StrictModel):
    monitor_id: str
    provider: str
    model_id: str
    prompt_id: str = "default"
    family: str = "unknown"
    mechanism: Literal["llm", "static", "sandbox", "fixture"] = "llm"
    revision: str | None = None
    endpoint: str | None = None
    cost_per_call: float = Field(default=0.0, ge=0)
    env_var: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Decision(StrictModel):
    case_id: str
    monitor_id: str
    run_index: int = Field(ge=0)
    raw_label: Literal["safe", "unsafe", "error"]
    raw_score: float = Field(ge=0, le=1)
    rationale: str = ""
    raw_response: str = ""
    provider_request_id: str | None = None
    latency_s: float = Field(default=0.0, ge=0)
    cost: float = Field(default=0.0, ge=0)
    cache_key: str
    error: str | None = None

    @model_validator(mode="after")
    def error_label_requires_error(self) -> Decision:
        if self.raw_label == "error" and not self.error:
            raise ValueError("error decisions must include an error message")
        return self


class Threshold(StrictModel):
    threshold_id: str
    calibration_split: Split
    fpr_target: float = Field(ge=0, le=1)
    scope: ThresholdScope
    score_transform: str = "identity"
    threshold: float = Field(ge=0, le=1)
    config_hash: str
    frozen_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    monitor_id: str | None = None
    rule_id: AggregationRule | None = None

    @field_validator("calibration_split")
    @classmethod
    def calibration_split_is_calibrate(cls, value: Split) -> Split:
        if value != "calibrate":
            raise ValueError("thresholds must be fit on the calibrate split")
        return value


class Result(StrictModel):
    result_id: str
    threshold_id: str
    rule_id: AggregationRule
    metric: str
    value: float | None
    lower: float | None = None
    upper: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunManifest(StrictModel):
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    code_hash: str = "unknown"
    config_hash: str = "unknown"
    model_ids: list[str] = Field(default_factory=list)
    prompt_ids: list[str] = Field(default_factory=list)
    provider_names: dict[str, str] = Field(default_factory=dict)
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    allow_hosted: bool = False
    sampling: dict[str, Any] = Field(default_factory=dict)
    provider_settings: dict[str, Any] = Field(default_factory=dict)


class RunPaths(StrictModel):
    run_dir: Path
    manifest: Path
    decisions: Path
    thresholds: Path
    metrics: Path
    report: Path
    derived: Path
