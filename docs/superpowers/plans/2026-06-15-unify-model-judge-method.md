# Unify Model Judge Method Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the split `LlmSafetyJudgeMethod` and `BinaryModelMethod` implementations with one `ModelJudgeMethod`.

**Architecture:** Keep non-model methods unchanged. `ModelJudgeMethod` owns model input mode (`prompt` or `case`) and output parser (`text_safety` or `binary`). Existing YAML types remain compatible by mapping `llm_safety`, `prompt_binary_model`, and `classifier_head_model` to the unified class.

**Tech Stack:** Python 3.11, dataclasses, pytest, PyYAML.

---

### Task 1: Red Tests

**Files:**
- Modify: `tests/test_methods.py`
- Modify: `tests/test_model_methods_config.py`
- Modify: `tests/test_providers.py`

- [ ] Assert direct model method usage goes through `ModelJudgeMethod`.
- [ ] Assert all three model YAML types construct `ModelJudgeMethod`.
- [ ] Assert binary providers can return only a label and rely on `default_confidence`.

### Task 2: Refactor Implementation

**Files:**
- Modify: `src/safeguard_harness/methods.py`
- Modify: `src/safeguard_harness/config.py`
- Modify: `src/safeguard_harness/providers.py`

- [ ] Add `ModelJudgeMethod` with `input_mode` and `output_parser`.
- [ ] Route text LLM, prompt-binary, and classifier-head paths through it.
- [ ] Remove separate `LlmSafetyJudgeMethod` and `BinaryModelMethod` implementations.
- [ ] Allow binary provider payloads without confidence.

### Task 3: Docs and Config

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] Document that model methods are unified.
- [ ] Update latest progress.
- [ ] Run full tests and sample CLI commands.

