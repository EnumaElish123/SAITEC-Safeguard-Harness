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
            cases.append(SafetyCase.from_dict(payload))
    return cases


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

