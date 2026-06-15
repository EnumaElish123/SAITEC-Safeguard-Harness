from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml

from safeguard_harness.core import SafetyCase

JsonTransport = Callable[[dict[str, Any]], dict[str, Any]]


class PromptBinaryProvider(Protocol):
    def classify_prompt(self, prompt: str) -> "BinaryModelOutput":
        ...


class CaseBinaryProvider(Protocol):
    def classify_case(self, case: SafetyCase) -> "BinaryModelOutput":
        ...


@dataclass(frozen=True)
class BinaryModelOutput:
    label: int
    confidence: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in {0, 1}:
            raise ValueError(f"binary label must be 0 or 1, got {self.label!r}")
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _clamp01(self.confidence))

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BinaryModelOutput":
        label_value = _first_present(payload, ["label", "prediction", "pred", "class", "output"])
        confidence_value = _first_optional(payload, ["confidence", "score", "probability", "prob"])
        return cls(
            label=parse_binary_label(label_value),
            confidence=None if confidence_value is None else float(confidence_value),
            raw=dict(payload),
        )


@dataclass
class PromptBinaryApiProvider:
    base_url: str
    api_key_env: str | None = None
    timeout_seconds: int = 30
    transport: JsonTransport = field(default_factory=lambda: _http_json_transport)
    prompt_field: str = "prompt"

    def classify_prompt(self, prompt: str) -> BinaryModelOutput:
        response = self.transport(
            {
                "url": self.base_url,
                "headers": self._headers(),
                "json": {self.prompt_field: prompt},
                "timeout_seconds": self.timeout_seconds,
            }
        )
        return BinaryModelOutput.from_payload(response)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env and os.environ.get(self.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        return headers


@dataclass
class ClassifierHeadApiProvider:
    base_url: str
    api_key_env: str | None = None
    timeout_seconds: int = 30
    transport: JsonTransport = field(default_factory=lambda: _http_json_transport)

    def classify_case(self, case: SafetyCase) -> BinaryModelOutput:
        response = self.transport(
            {
                "url": self.base_url,
                "headers": self._headers(),
                "json": case.to_dict(),
                "timeout_seconds": self.timeout_seconds,
            }
        )
        return BinaryModelOutput.from_payload(response)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env and os.environ.get(self.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        return headers


@dataclass
class MockPromptBinaryProvider:
    default_label: int = 0
    default_confidence: float | None = 0.8
    unsafe_keywords: list[str] = field(default_factory=list)
    safe_keywords: list[str] = field(default_factory=list)
    refuse_keywords: list[str] = field(default_factory=list)

    def classify_prompt(self, prompt: str) -> BinaryModelOutput:
        lowered = prompt.casefold()
        unsafe_match = _first_keyword_match(lowered, self.unsafe_keywords + self.refuse_keywords)
        if unsafe_match is not None:
            return BinaryModelOutput(
                label=1,
                confidence=self.default_confidence,
                raw={
                    "provider": "mock_prompt_binary_keywords",
                    "prompt": prompt,
                    "matched_keyword": unsafe_match,
                },
            )

        safe_match = _first_keyword_match(lowered, self.safe_keywords)
        if safe_match is not None:
            return BinaryModelOutput(
                label=0,
                confidence=self.default_confidence,
                raw={
                    "provider": "mock_prompt_binary_keywords",
                    "prompt": prompt,
                    "matched_keyword": safe_match,
                },
            )

        return BinaryModelOutput(
            label=parse_binary_label(self.default_label),
            confidence=self.default_confidence,
            raw={"provider": "mock_prompt_binary", "prompt": prompt},
        )


@dataclass
class MockClassifierHeadProvider:
    default_label: int = 0
    default_confidence: float | None = 0.8

    def classify_case(self, case: SafetyCase) -> BinaryModelOutput:
        return BinaryModelOutput(
            label=parse_binary_label(self.default_label),
            confidence=self.default_confidence,
            raw={"provider": "mock_classifier_head", "case_id": case.id},
        )


@dataclass
class LocalClassifierHeadProvider:
    model_path: str

    def classify_case(self, case: SafetyCase) -> BinaryModelOutput:
        raise RuntimeError(
            f"local classifier head at {self.model_path!r} is configured but no local inference adapter is implemented"
        )


def load_provider_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"provider config root must be a mapping: {path}")
    return payload


def build_binary_provider(config: dict[str, Any]) -> Any:
    provider_type = config.get("type")
    if provider_type == "prompt_binary_api":
        return PromptBinaryApiProvider(
            base_url=str(config["base_url"]),
            api_key_env=config.get("api_key_env"),
            timeout_seconds=int(config.get("timeout_seconds", 30)),
            prompt_field=str(config.get("prompt_field", "prompt")),
        )
    if provider_type == "classifier_head_api":
        return ClassifierHeadApiProvider(
            base_url=str(config["base_url"]),
            api_key_env=config.get("api_key_env"),
            timeout_seconds=int(config.get("timeout_seconds", 30)),
        )
    if provider_type == "mock_prompt_binary":
        return MockPromptBinaryProvider(
            default_label=parse_binary_label(config.get("default_label", 0)),
            default_confidence=config.get("default_confidence", 0.8),
            unsafe_keywords=list(config.get("unsafe_keywords") or []),
            safe_keywords=list(config.get("safe_keywords") or []),
            refuse_keywords=list(config.get("refuse_keywords") or []),
        )
    if provider_type == "mock_classifier_head":
        return MockClassifierHeadProvider(
            default_label=parse_binary_label(config.get("default_label", 0)),
            default_confidence=config.get("default_confidence", 0.8),
        )
    if provider_type == "local_classifier_head":
        return LocalClassifierHeadProvider(model_path=os.path.expandvars(str(config["model_path"])))
    raise ValueError(f"unknown binary provider type: {provider_type!r}")


def parse_binary_label(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        if value in {0, 1}:
            return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"1", "unsafe", "risk", "risky", "harmful", "danger"}:
            return 1
        if lowered in {"0", "safe", "benign", "allowed", "normal"}:
            return 0
    raise ValueError(f"cannot parse binary label from {value!r}")


def _first_present(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    raise ValueError(f"payload missing any of required keys: {keys}")


def _first_optional(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _first_keyword_match(lowered_text: str, keywords: list[str]) -> str | None:
    for keyword in keywords:
        if keyword.casefold() in lowered_text:
            return keyword
    return None


def _http_json_transport(request: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(request["json"], ensure_ascii=False).encode("utf-8")
    http_request = urllib.request.Request(
        str(request["url"]),
        data=body,
        headers=dict(request.get("headers") or {}),
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=int(request.get("timeout_seconds", 30))) as response:
        response_body = response.read().decode("utf-8")
    payload = json.loads(response_body)
    if not isinstance(payload, dict):
        raise ValueError("model provider response must be a JSON object")
    return payload
