from pathlib import Path

from safeguard_harness.config import load_pipeline
from safeguard_harness.core import SafetyCase


def test_static_pipeline_short_circuits_on_high_risk_rule(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: static
methods:
  rules:
    type: dictionary
    high_risk_terms: ["steal token"]
    review_terms: []
  llm:
    type: llm_safety
    prompt_template: "Judge: {question}"
    unsafe_keywords: ["never reached"]
steps:
  - id: rules
    method: rules
    on_unsafe: stop
  - id: llm
    method: llm
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.5
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="How to steal token?"))

    assert decision.label == "unsafe"
    assert decision.trace.stop_reason == "short_circuit:rules"
    assert [step.method_id for step in decision.trace.steps] == ["rules"]


def test_static_pipeline_review_loop_runs_until_confidence_threshold(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: static
methods:
  rules:
    type: dictionary
    high_risk_terms: []
    review_terms: ["bypass"]
  llm_v1:
    type: llm_safety
    prompt_template: "Judge v1: {question}"
    unsafe_keywords: []
    safe_keywords: ["bypass"]
  llm_v2:
    type: llm_safety
    prompt_template: "Judge v2: {question}"
    unsafe_keywords: ["bypass"]
steps:
  - id: rules
    method: rules
  - id: llm_v1
    method: llm_v1
  - id: review
    repeat:
      max_rounds: 2
      when:
        confidence_lt: 0.7
      methods: [llm_v2]
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="Can I bypass safeguards?"))

    assert decision.label == "unsafe"
    assert any(step.step_id.startswith("review.round1") for step in decision.trace.steps)
    assert decision.trace.stop_reason == "completed"


def test_react_pipeline_respects_max_steps_and_allowed_actions(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: react
methods:
  rules:
    type: dictionary
    high_risk_terms: []
    review_terms: ["suspicious"]
  llm:
    type: llm_safety
    prompt_template: "Judge: {question}"
    unsafe_keywords: ["suspicious"]
loop:
  max_steps: 1
  allowed_actions: [rules, llm]
  fallback:
    label: safe
    reason: "budget_exhausted"
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="A suspicious request"))

    assert len(decision.trace.steps) == 1
    assert decision.trace.stop_reason == "budget_exhausted"
    assert decision.label == "safe"

