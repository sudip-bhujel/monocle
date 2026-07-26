from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from hashlib import sha256
from pathlib import Path
from typing import Any

import litellm
from litellm import completion

litellm.suppress_debug_info = True

from monocle.schema import Case, Decision, MonitorConfig


class Monitor(ABC):
    def __init__(self, config: MonitorConfig) -> None:
        self.config = config

    @abstractmethod
    def judge(self, case: Case, run_index: int) -> Decision:
        raise NotImplementedError

    def cache_key(self, case: Case, run_index: int) -> str:
        payload = f"{self.config.monitor_id}:{case.case_id}:{run_index}:{case.payload}"
        return sha256(payload.encode("utf-8")).hexdigest()

    def request_json(self, case: Case, run_index: int, namespace: str) -> str | None:
        return None

    def request_fingerprint(
        self, case: Case, run_index: int, namespace: str
    ) -> str | None:
        request = self.request_json(case, run_index, namespace)
        return sha256(request.encode("utf-8")).hexdigest() if request else None


class FixtureMonitor(Monitor):
    def __init__(
        self, config: MonitorConfig, scores: dict[str, float] | None = None
    ) -> None:
        super().__init__(config)
        self.scores = scores or {}

    def judge(self, case: Case, run_index: int) -> Decision:
        score = self.scores.get(case.case_id)
        if score is None:
            score = 0.85 if case.label == "unsafe" else 0.15
        return Decision(
            case_id=case.case_id,
            monitor_id=self.config.monitor_id,
            run_index=run_index,
            raw_label="unsafe" if score >= 0.5 else "safe",
            raw_score=score,
            rationale="fixture",
            raw_response=str(score),
            provider_request_id=None,
            latency_s=0.0,
            cost=self.config.cost_per_call,
            cache_key=self.cache_key(case, run_index),
        )


class HostedModelMonitor(Monitor):
    def __init__(self, config: MonitorConfig) -> None:
        super().__init__(config)
        self._prompt_text: str | None = None

    def validate_ready(self) -> None:
        env_var = self.config.env_var or f"{self.config.provider.upper()}_API_KEY"
        missing = []
        if not self.config.provider:
            missing.append("provider")
        if not self._resolved_model_id():
            missing.append("model_id")
        if not self.config.prompt_id:
            missing.append("prompt_id")
        if not os.getenv(env_var):
            missing.append(env_var)
        try:
            self._load_prompt()
        except FileNotFoundError:
            missing.append(f"prompt:{self.config.prompt_id}")
        if missing:
            raise RuntimeError(
                f"hosted monitor {self.config.monitor_id} is not ready: missing {', '.join(missing)}"
            )

    def judge(self, case: Case, run_index: int) -> Decision:
        env_var = self.config.env_var or f"{self.config.provider.upper()}_API_KEY"
        model_id = self._resolved_model_id()
        if not model_id or not os.getenv(env_var):
            return Decision(
                case_id=case.case_id,
                monitor_id=self.config.monitor_id,
                run_index=run_index,
                raw_label="error",
                raw_score=0.0,
                rationale="hosted model unavailable",
                raw_response="",
                provider_request_id=None,
                latency_s=0.0,
                cost=0.0,
                cache_key=self.cache_key(case, run_index),
                error=f"missing credentials or model id for {self.config.monitor_id}",
            )
        started = time.perf_counter()
        try:
            kwargs = self._request_kwargs(case)
            response = completion(**kwargs)
            latency_s = time.perf_counter() - started
            raw_response = _response_text(response)
            parsed = _parse_score(raw_response)
            usage = _response_usage(response)
            provider_cost = usage["provider_cost"]
            cost = (
                provider_cost
                if provider_cost is not None
                else self.config.cost_per_call
            )
            return Decision(
                case_id=case.case_id,
                monitor_id=self.config.monitor_id,
                run_index=run_index,
                raw_label=parsed["label"],
                raw_score=parsed["score"],
                rationale=parsed["rationale"],
                raw_response=raw_response,
                provider_request_id=_response_id(response),
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                provider_cost=provider_cost,
                latency_s=latency_s,
                cost=cost,
                billed_cost=cost,
                cache_key=self.cache_key(case, run_index),
            )
        except Exception as exc:
            return Decision(
                case_id=case.case_id,
                monitor_id=self.config.monitor_id,
                run_index=run_index,
                raw_label="error",
                raw_score=0.0,
                rationale="hosted call failed",
                raw_response="",
                provider_request_id=None,
                latency_s=time.perf_counter() - started,
                cost=0.0,
                cache_key=self.cache_key(case, run_index),
                error=str(exc),
            )

    def manifest_model_id(self) -> str:
        return self._resolved_model_id() or self.config.model_id

    def request_json(self, case: Case, run_index: int, namespace: str) -> str:
        request = {
            "schema_version": 1,
            "namespace": namespace,
            "provider": self.config.provider,
            "model_id": self._resolved_model_id(),
            "revision": self.config.revision,
            "endpoint": self.config.endpoint,
            "case_id": case.case_id,
            "run_index": run_index,
            "request": self._request_kwargs(case),
        }
        return json.dumps(request, sort_keys=True, separators=(",", ":"))

    def _resolved_model_id(self) -> str | None:
        if self.config.model_id.startswith("env:"):
            return os.getenv(self.config.model_id.removeprefix("env:"))
        return self.config.model_id

    def _load_prompt(self) -> str:
        if self._prompt_text is not None:
            return self._prompt_text
        prompt_dir = Path(
            str(self.config.metadata.get("prompt_dir", "configs/prompts"))
        )
        path = prompt_dir / f"{self.config.prompt_id}.txt"
        self._prompt_text = path.read_text(encoding="utf-8")
        return self._prompt_text

    def _request_kwargs(self, case: Case) -> dict[str, Any]:
        kwargs = {
            "model": self._resolved_model_id(),
            "messages": self._messages(case),
            "temperature": float(self.config.metadata.get("temperature", 0)),
            "max_tokens": int(self.config.metadata.get("max_tokens", 256)),
        }
        if "response_format" in self.config.metadata:
            kwargs["response_format"] = self.config.metadata["response_format"]
        if self.config.endpoint:
            kwargs["api_base"] = self.config.endpoint
        return kwargs

    def _messages(self, case: Case) -> list[dict[str, str]]:
        user = _case_message(case, str(self.config.metadata.get("case_view", "blind")))
        return [
            {"role": "system", "content": self._load_prompt()},
            {"role": "user", "content": user},
        ]


def build_monitor(config: MonitorConfig) -> Monitor:
    if config.mechanism == "fixture":
        return FixtureMonitor(config)
    if config.mechanism in {"static", "sandbox"}:
        return FixtureMonitor(config)
    return HostedModelMonitor(config)


def _case_message(case: Case, case_view: str) -> str:
    if case_view == "diagnostic":
        return "\n".join(
            [
                f"case_id: {case.case_id}",
                f"kind: {case.kind}",
                f"regime: {case.regime}",
                f"attack_class: {case.attack_class}",
                "payload:",
                case.payload,
            ]
        )
    if case_view != "blind":
        raise ValueError(f"unsupported case_view: {case_view}")
    return "\n".join(["kind:", case.kind, "payload:", case.payload])


def validate_hosted_monitors(monitors: list[Monitor], *, allow_hosted: bool) -> None:
    hosted = [
        monitor for monitor in monitors if isinstance(monitor, HostedModelMonitor)
    ]
    if hosted and not allow_hosted:
        ids = ", ".join(monitor.config.monitor_id for monitor in hosted)
        raise RuntimeError(f"hosted monitors require --allow-hosted: {ids}")
    for monitor in hosted:
        monitor.validate_ready()


def _response_text(response: Any) -> str:
    if isinstance(response, dict):
        value = response["choices"][0]["message"]["content"]
    else:
        choices = getattr(response, "choices")
        message = choices[0].message
        value = message["content"] if isinstance(message, dict) else getattr(message, "content")
    if value is None or not str(value).strip():
        raise ValueError("hosted response contains no message content")
    return str(value).strip()


def _response_id(response: Any) -> str | None:
    if isinstance(response, dict):
        value = response.get("id")
    else:
        value = getattr(response, "id", None)
    return str(value) if value else None


def _response_usage(response: Any) -> dict[str, int | float | None]:
    usage = (
        response.get("usage", {})
        if isinstance(response, dict)
        else getattr(response, "usage", {})
    )
    if not isinstance(usage, dict):
        usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }
    hidden = (
        response.get("_hidden_params", {})
        if isinstance(response, dict)
        else getattr(response, "_hidden_params", {})
    )
    cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
    return {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "provider_cost": float(cost) if cost is not None else None,
    }


def _parse_score(text: str) -> dict[str, Any]:
    normalized = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced is not None:
        normalized = fenced.group(1).strip()
    try:
        data = json.loads(normalized)
    except json.JSONDecodeError as error:
        raise ValueError("hosted response is not valid JSON") from error
    if not isinstance(data, dict):
        raise ValueError("hosted response JSON must be an object")
    if "label" not in data or "score" not in data:
        raise ValueError("hosted response JSON must contain label and score")
    label = str(data["label"]).lower()
    if label not in {"safe", "unsafe"}:
        raise ValueError("hosted response label must be safe or unsafe")
    score = float(data["score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError("hosted response score must be between 0 and 1")
    return {
        "label": label,
        "score": score,
        "rationale": str(data.get("rationale", normalized))[:500],
    }
