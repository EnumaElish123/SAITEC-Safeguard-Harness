from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SAFE = "safe"
UNSAFE = "unsafe"
UNKNOWN = "unknown"
VALID_LABELS = {SAFE, UNSAFE, UNKNOWN}


def validate_label(label: str | None, *, allow_none: bool = False) -> str | None:
    if label is None and allow_none:
        return None
    if label not in VALID_LABELS:
        raise ValueError(f"label must be one of {sorted(VALID_LABELS)}, got {label!r}")
    return label


@dataclass(frozen=True)
class SafetyCase:
    id: str
    question: str
    answer: str | None = None
    label: str | None = None
    modality: str = "text"
    attachments: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SafetyCase":
        if not payload.get("question"):
            raise ValueError("case requires a non-empty question")
        label = validate_label(payload.get("label"), allow_none=True)
        return cls(
            id=str(payload.get("id") or "case"),
            question=str(payload["question"]),
            answer=payload.get("answer"),
            label=label,
            modality=str(payload.get("modality") or "text"),
            attachments=list(payload.get("attachments") or []),
            metadata=dict(payload.get("metadata") or {}),
        )

    def text_for_judging(self) -> str:
        parts = [self.question]
        if self.answer:
            parts.append(self.answer)
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "label": self.label,
            "modality": self.modality,
            "attachments": list(self.attachments),
            "metadata": dict(self.metadata),
        }


@dataclass
class MethodResult:
    method_id: str
    label: str
    unsafe_score: float
    confidence: float
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False

    def __post_init__(self) -> None:
        validate_label(self.label)
        self.unsafe_score = _clamp01(self.unsafe_score)
        self.confidence = _clamp01(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "label": self.label,
            "unsafe_score": self.unsafe_score,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "metadata": dict(self.metadata),
            "skipped": self.skipped,
        }


@dataclass
class TraceStep:
    step_id: str
    method_id: str
    result: MethodResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "method_id": self.method_id,
            "result": self.result.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass
class RunTrace:
    case_id: str
    runner: str = "static"
    steps: list[TraceStep] = field(default_factory=list)
    stop_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: TraceStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "runner": self.runner,
            "steps": [step.to_dict() for step in self.steps],
            "stop_reason": self.stop_reason,
            "metadata": dict(self.metadata),
        }


@dataclass
class Decision:
    case_id: str
    label: str
    unsafe_score: float
    confidence: float
    reasons: list[str] = field(default_factory=list)
    trace: RunTrace | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_label(self.label)
        self.unsafe_score = _clamp01(self.unsafe_score)
        self.confidence = _clamp01(self.confidence)

    def to_dict(self, *, include_trace: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "label": self.label,
            "unsafe_score": self.unsafe_score,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }
        if include_trace and self.trace is not None:
            payload["trace"] = self.trace.to_dict()
        return payload


@dataclass
class RunContext:
    run_id: str = ""
    llm_calls: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def count_llm_call(self) -> None:
        self.llm_calls += 1


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

