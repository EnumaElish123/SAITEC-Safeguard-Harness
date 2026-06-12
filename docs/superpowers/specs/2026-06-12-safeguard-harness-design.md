# Safeguard Harness Design

## Goal

Build a Python harness for safety-classification experiments and deployment. The first usable version must work without real models, real dictionaries, or real multimodal probes by exposing stable interfaces, placeholder adapters, empty dictionary schemas, runnable sample data, and deterministic mock providers.

## Operating Modes

The system has two modes:

- Experiment mode: run a configured pipeline over a validation dataset, generate predictions, metrics, traces, and error slices, then manually adjust dictionaries, prompts, thresholds, and pipeline configuration before rerunning.
- Deployment mode: load a fixed pipeline and judge one case or a JSONL batch through a CLI or Python API.

## Core Data Model

Every input is normalized into a `SafetyCase` with `id`, `question`, optional `answer`, `modality`, optional attachments, optional label, and metadata. Every judge method returns a `MethodResult` containing a label, score, confidence, evidence, and metadata. Pipeline execution returns a `Decision` plus a `RunTrace` that records each step, observation, and stop reason.

The first-stage label set is binary: `safe` and `unsafe`. The schema leaves room for later `risk_type`, `attack_type`, and fine-grained category fields.

## Method Interfaces

All judge methods implement one interface:

```python
class JudgeMethod:
    def judge(self, case: SafetyCase, context: RunContext) -> MethodResult:
        ...
```

The first implementation includes these method families:

- Rule dictionary method: reads high-risk and review-risk dictionary entries. High-risk hits can short-circuit to unsafe. Review-risk hits emit risk evidence for downstream methods. Fuzzy matching is represented by a `FuzzyMatcher` interface and a default exact/substring matcher.
- LLM safety judge method: renders a configurable prompt template and calls an LLM provider adapter. Until real models exist, a mock provider returns deterministic outputs based on config.
- Aligned refusal probe method: wraps only the question, calls an aligned model provider, and treats refusal-style responses as unsafe evidence.
- Multimodal probe method: runs only for non-text or attachment-bearing cases and returns a neutral skipped result until a real probe adapter is installed.

Different prompt templates or provider settings are different method instances in configuration.

## Pipeline and Loop Control

Two runners are supported:

- Static runner: deterministic steps, conditions, short-circuiting, aggregation, retry/review loops, and max-round limits. This is the default for deployment.
- ReAct runner: bounded action-observation loop over registered methods. It supports `max_steps`, `max_llm_calls`, `allowed_actions`, confidence stop conditions, required evidence, and fallback decisions. It is intended for experiment analysis, not default production submission.

Loop control is a first-class concept. Any loop must have an explicit budget, stop condition, and fallback behavior. Deployment loops must be deterministic and reproducible.

## Configuration

Configuration is YAML. The first version supports:

- pipeline definitions under `configs/pipelines/`
- provider definitions under `configs/providers/`
- dictionary paths under `configs/dictionaries/`
- prompt templates under `configs/prompts/`
- dataset definitions under `configs/datasets/`

The runner snapshots the resolved pipeline config into each output directory so experiment results can be reproduced.

## Dataset and Evaluation

Datasets are JSONL. Each row contains at least an `id`, `question`, and optional `label`.

Evaluation writes:

- `predictions.jsonl`
- `metrics.json`
- `report.md`
- `errors_false_positive.jsonl`
- `errors_false_negative.jsonl`
- `config_snapshot.yaml`

Metrics include accuracy, precision, recall, F1, confusion matrix, per-modality slices, latency, and method-call counts.

## CLI

The package exposes:

```powershell
python -m safeguard_harness evaluate --pipeline configs/pipelines/experiment_v1.yaml --dataset data/examples/sample_eval.jsonl --output outputs/runs/demo
python -m safeguard_harness predict --pipeline configs/pipelines/prod_v1.yaml --input data/examples/sample_eval.jsonl --output outputs/submission.jsonl
python -m safeguard_harness judge --pipeline configs/pipelines/prod_v1.yaml --question "example question"
```

## Initial Non-Goals

- No real competition dictionaries are included.
- No real model credentials are required.
- No automatic prompt optimization or automatic training loop is included.
- No server API is required in the first version; CLI and Python API are enough.

