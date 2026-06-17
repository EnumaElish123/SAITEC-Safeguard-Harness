from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any

from safeguard_harness.core import SafetyCase


def load_jsonl_cases(path: str | Path) -> list[SafetyCase]:
    cases: list[SafetyCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            cases.append(SafetyCase.from_dict(_normalize_case_payload(payload)))
    return cases


def _normalize_case_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("question") or "messages" not in payload:
        return payload

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload

    user_texts = _message_texts(messages, role="user")
    assistant_texts = _message_texts(messages, role="assistant")
    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "source_format": "messages",
            "messages": messages,
        }
    )
    for key in ("type", "is_mt"):
        if key in payload:
            metadata[key] = payload[key]

    normalized = dict(payload)
    normalized["question"] = "\n\n".join(user_texts).strip()
    normalized["answer"] = "\n\n".join(assistant_texts).strip() or payload.get("answer")
    normalized["metadata"] = metadata
    return normalized


def _message_texts(messages: list[Any], *, role: str) -> list[str]:
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != role:
            continue
        text = _content_text(message.get("content")).strip()
        if text:
            texts.append(text)
    return texts


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
            parts.append(str(item["text"]))
    return "\n".join(parts)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

