# Model Provider Interfaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class model provider integration points for prompt-based binary classifiers and classifier-head binary classifiers with confidence.

**Architecture:** Keep transport/model details in `providers.py`, keep safety-method conversion in `methods.py`, and keep concrete runtime selection in YAML under `configs/providers/` and `configs/pipelines/`. Model files are kept out of Git through `.gitignore` and are referenced by environment variables or local paths.

**Tech Stack:** Python 3.11, dataclasses, urllib request hooks, PyYAML, pytest.

---

### Task 1: Provider Interfaces

**Files:**
- Create: `src/safeguard_harness/providers.py`
- Test: `tests/test_providers.py`

- [ ] Write failing tests for `BinaryModelOutput`, prompt API parsing, classifier-head API parsing, and provider config loading.
- [ ] Implement provider classes with injectable transports for tests.
- [ ] Verify `python -m pytest tests/test_providers.py -q` passes.

### Task 2: Method and Config Integration

**Files:**
- Modify: `src/safeguard_harness/methods.py`
- Modify: `src/safeguard_harness/config.py`
- Test: `tests/test_model_methods_config.py`

- [ ] Write failing tests for `prompt_binary_model` and `classifier_head_model` method construction from YAML.
- [ ] Implement method wrappers that convert `BinaryModelOutput` into `MethodResult`.
- [ ] Verify focused tests pass.

### Task 3: Starter Configs and Documentation

**Files:**
- Modify: `.gitignore`
- Add: `configs/providers/prompt_binary_api.yaml`
- Add: `configs/providers/classifier_head_api.yaml`
- Add: `configs/providers/local_classifier_head.yaml`
- Add: `configs/pipelines/model_interfaces_v1.yaml`
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] Add provider config examples without secrets.
- [ ] Add `models/` to `.gitignore`.
- [ ] Document where interfaces, provider configs, prompt templates, pipelines, and model files live.
- [ ] Run all tests and a sample CLI command.

