from pathlib import Path

from safeguard_harness.core import SafetyCase
from safeguard_harness.providers import (
    BinaryModelOutput,
    ClassifierHeadApiProvider,
    PromptBinaryApiProvider,
    build_binary_provider,
    load_provider_config,
)


def test_binary_model_output_accepts_numeric_and_string_labels():
    unsafe = BinaryModelOutput.from_payload({"label": 1, "confidence": 0.91})
    safe = BinaryModelOutput.from_payload({"prediction": "safe", "score": 0.82})
    label_only = BinaryModelOutput.from_payload({"label": 1})

    assert unsafe.label == 1
    assert unsafe.confidence == 0.91
    assert safe.label == 0
    assert safe.confidence == 0.82
    assert label_only.label == 1
    assert label_only.confidence is None


def test_prompt_binary_api_provider_sends_prompt_and_parses_prediction(monkeypatch):
    monkeypatch.setenv("PROMPT_KEY", "secret")
    calls = []

    def transport(request):
        calls.append(request)
        return {"prediction": 1, "confidence": 0.73, "reason": "risk"}

    provider = PromptBinaryApiProvider(
        base_url="https://model.example/prompt",
        api_key_env="PROMPT_KEY",
        timeout_seconds=12,
        transport=transport,
    )

    output = provider.classify_prompt("Judge this")

    assert output.label == 1
    assert output.confidence == 0.73
    assert calls[0]["url"] == "https://model.example/prompt"
    assert calls[0]["headers"]["Authorization"] == "Bearer secret"
    assert calls[0]["json"]["prompt"] == "Judge this"
    assert calls[0]["timeout_seconds"] == 12


def test_classifier_head_api_provider_sends_case_payload_and_parses_confidence(monkeypatch):
    monkeypatch.setenv("HEAD_KEY", "head-secret")
    calls = []

    def transport(request):
        calls.append(request)
        return {"label": 0, "confidence": 0.88, "logits": [2.1, -1.4]}

    provider = ClassifierHeadApiProvider(
        base_url="https://model.example/head",
        api_key_env="HEAD_KEY",
        transport=transport,
    )
    case = SafetyCase(id="c1", question="hello", answer="world")

    output = provider.classify_case(case)

    assert output.label == 0
    assert output.confidence == 0.88
    assert calls[0]["json"]["id"] == "c1"
    assert calls[0]["json"]["question"] == "hello"
    assert calls[0]["headers"]["Authorization"] == "Bearer head-secret"


def test_build_binary_provider_from_yaml_config(tmp_path: Path):
    provider_path = tmp_path / "prompt_binary_api.yaml"
    provider_path.write_text(
        """
type: mock_prompt_binary
default_label: 1
default_confidence: 0.64
""",
        encoding="utf-8",
    )

    config = load_provider_config(provider_path)
    provider = build_binary_provider(config)
    output = provider.classify_prompt("anything")

    assert output.label == 1
    assert output.confidence == 0.64
