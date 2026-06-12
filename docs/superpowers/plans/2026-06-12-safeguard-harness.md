# Safeguard Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Python safety harness framework with configurable methods, bounded loops, dataset evaluation, deployment CLI, starter configs, and documentation.

**Architecture:** The harness uses small native Python modules: typed dataclasses for core records, a method registry for interchangeable judges, YAML config loading for pipelines, deterministic runners for static and ReAct-style execution, and JSONL-based evaluation outputs. Real dictionaries and models are represented by stable interfaces plus deterministic starter adapters.

**Tech Stack:** Python 3.11, pytest, PyYAML, argparse, dataclasses, pathlib, json/jsonl files.

---

## File Structure

- `pyproject.toml`: package metadata, dependencies, pytest configuration.
- `src/safeguard_harness/core.py`: labels, cases, method results, decisions, traces, and run context.
- `src/safeguard_harness/datasets.py`: JSONL loading and writing helpers.
- `src/safeguard_harness/config.py`: YAML loading and method/pipeline construction.
- `src/safeguard_harness/methods.py`: method registry and initial method implementations.
- `src/safeguard_harness/orchestration.py`: static runner, ReAct runner, aggregation, loop handling.
- `src/safeguard_harness/evaluation.py`: metrics, reports, prediction output, error slices.
- `src/safeguard_harness/cli.py` and `src/safeguard_harness/__main__.py`: CLI commands.
- `configs/`: starter YAML configs.
- `dictionaries/`: empty but schema-valid dictionary files.
- `data/examples/`: sample validation dataset.
- `tests/`: behavior tests written before production implementation.

### Task 1: Core Schemas

**Files:**
- Test: `tests/test_core.py`
- Create: `src/safeguard_harness/core.py`

- [ ] Write failing tests for `SafetyCase`, `MethodResult`, `Decision`, and trace serialization.
- [ ] Run `python -m pytest tests/test_core.py -q` and confirm imports fail.
- [ ] Implement the dataclasses and serialization helpers.
- [ ] Re-run the test and confirm it passes.

### Task 2: Methods

**Files:**
- Test: `tests/test_methods.py`
- Create: `src/safeguard_harness/methods.py`

- [ ] Write failing tests for high-risk dictionary hits, review-risk hits, LLM mock prompt calls, refusal detection, and multimodal skip behavior.
- [ ] Run `python -m pytest tests/test_methods.py -q` and confirm imports fail.
- [ ] Implement method registry, matcher interface, dictionary method, mock LLM safety judge, refusal probe, and multimodal probe.
- [ ] Re-run the test and confirm it passes.

### Task 3: Pipeline Runners and Loops

**Files:**
- Test: `tests/test_pipeline.py`
- Create: `src/safeguard_harness/orchestration.py`
- Create: `src/safeguard_harness/config.py`

- [ ] Write failing tests for static short-circuit, weighted aggregation, low-confidence review loop, and bounded ReAct action loop.
- [ ] Run `python -m pytest tests/test_pipeline.py -q` and confirm imports fail.
- [ ] Implement YAML construction, static runner, review-loop support, ReAct loop budgets, and final decision aggregation.
- [ ] Re-run the test and confirm it passes.

### Task 4: Evaluation and CLI

**Files:**
- Test: `tests/test_evaluation_cli.py`
- Create: `src/safeguard_harness/datasets.py`
- Create: `src/safeguard_harness/evaluation.py`
- Create: `src/safeguard_harness/cli.py`
- Create: `src/safeguard_harness/__main__.py`

- [ ] Write failing tests for JSONL loading, metric generation, output files, and CLI judge/predict/evaluate commands.
- [ ] Run `python -m pytest tests/test_evaluation_cli.py -q` and confirm imports fail.
- [ ] Implement datasets, evaluator, CLI commands, and module entrypoint.
- [ ] Re-run the test and confirm it passes.

### Task 5: Starter Assets and Documentation

**Files:**
- Create: `configs/pipelines/experiment_v1.yaml`
- Create: `configs/pipelines/prod_v1.yaml`
- Create: `configs/providers/mock.yaml`
- Create: `configs/datasets/sample_eval.yaml`
- Create: `configs/prompts/safety_prompt_v1.txt`
- Create: `configs/prompts/refusal_probe_v1.txt`
- Create: `dictionaries/high_risk_terms.yaml`
- Create: `dictionaries/review_terms.yaml`
- Create: `data/examples/sample_eval.jsonl`
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] Add starter configs and empty dictionary schemas.
- [ ] Add sample data demonstrating safe, unsafe, and review-risk paths.
- [ ] Update README with install, config, experiment, deployment, and extension instructions.
- [ ] Update AGENTS latest progress with this implementation.
- [ ] Run `python -m pytest -q`.
- [ ] Run a sample evaluate command and inspect output files.

