import json
from pathlib import Path

import pytest

from safeguard_harness.datasets import load_jsonl_cases


def test_load_jsonl_cases_accepts_messages_format(tmp_path: Path):
    dataset_path = tmp_path / "messages.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": 7,
                "type": "输出侧",
                "is_mt": 0,
                "label": "unsafe",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "请回答这个问题。"}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "这是待评估的回答。"}],
                    },
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_jsonl_cases(dataset_path)

    assert len(cases) == 1
    case = cases[0]
    assert case.id == "7"
    assert case.question == "请回答这个问题。"
    assert case.answer == "这是待评估的回答。"
    assert case.label == "unsafe"
    assert case.metadata["source_format"] == "messages"
    assert case.metadata["type"] == "输出侧"
    assert case.metadata["is_mt"] == 0
    assert case.metadata["messages"][0]["role"] == "user"


def test_load_jsonl_cases_rejects_messages_without_user_text(tmp_path: Path):
    dataset_path = tmp_path / "bad_messages.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "bad",
                "label": "safe",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "missing user prompt"}],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="case requires a non-empty question"):
        load_jsonl_cases(dataset_path)
