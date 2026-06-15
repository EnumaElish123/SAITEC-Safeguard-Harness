from pathlib import Path

from safeguard_harness.config import load_pipeline
from safeguard_harness.core import SafetyCase
from safeguard_harness.methods import BinaryModelMethod


def test_binary_model_method_maps_prompt_output_to_method_result(tmp_path: Path):
    provider_path = tmp_path / "provider.yaml"
    prompt_path = tmp_path / "prompt.txt"
    pipeline_path = tmp_path / "pipeline.yaml"
    provider_path.write_text(
        """
type: mock_prompt_binary
default_label: 1
default_confidence: 0.77
""",
        encoding="utf-8",
    )
    prompt_path.write_text("Judge: {question}", encoding="utf-8")
    pipeline_path.write_text(
        f"""
runner: static
methods:
  prompt_binary:
    type: prompt_binary_model
    provider_config: {provider_path.as_posix()}
    prompt_template_path: {prompt_path.as_posix()}
steps:
  - id: prompt_binary
    method: prompt_binary
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    method = pipeline.methods["prompt_binary"]
    decision = pipeline.judge(SafetyCase(id="c1", question="demo"))

    assert isinstance(method, BinaryModelMethod)
    assert decision.label == "unsafe"
    assert decision.trace.steps[0].result.confidence == 0.77
    assert decision.trace.steps[0].result.metadata["provider_kind"] == "prompt_binary"


def test_classifier_head_method_uses_confidence_as_unsafe_score_for_unsafe_label(tmp_path: Path):
    provider_path = tmp_path / "provider.yaml"
    pipeline_path = tmp_path / "pipeline.yaml"
    provider_path.write_text(
        """
type: mock_classifier_head
default_label: 1
default_confidence: 0.92
""",
        encoding="utf-8",
    )
    pipeline_path.write_text(
        f"""
runner: static
methods:
  head:
    type: classifier_head_model
    provider_config: {provider_path.as_posix()}
steps:
  - id: head
    method: head
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="demo"))
    result = decision.trace.steps[0].result

    assert decision.label == "unsafe"
    assert result.unsafe_score == 0.92
    assert result.confidence == 0.92
    assert result.metadata["provider_kind"] == "classifier_head"

