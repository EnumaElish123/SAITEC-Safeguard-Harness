from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from safeguard_harness.internal_llm import InternalLlmJudge
from safeguard_harness.methods import (
    DEFAULT_REFUSAL_MARKERS,
    DictionaryRuleMethod,
    ImageProbeReviewMethod,
    ModelJudgeMethod,
    MockLlmProvider,
    MultimodalProbeMethod,
    ProgressiveRuleClassifierMethod,
    ProgressiveRuleDocument,
    RegexRuleMethod,
    RefusalProbeMethod,
    coerce_terms,
)
from safeguard_harness.orchestration import ReactPipeline, StaticPipeline
from safeguard_harness.providers import (
    MockPromptBinaryProvider,
    build_binary_provider,
    build_multimodal_provider,
    build_text_generation_provider,
    load_provider_config,
)


def load_pipeline(path: str | Path) -> StaticPipeline | ReactPipeline:
    config_path = Path(path)
    raw_config = load_yaml(config_path)
    semantic_fallback = build_base_llm_judge(raw_config.get("base_llm"), config_path.parent)
    methods = {
        method_id: build_method(method_id, method_config, config_path.parent, semantic_fallback)
        for method_id, method_config in (raw_config.get("methods") or {}).items()
    }
    runner = raw_config.get("runner", "static")
    common = {
        "runner": runner,
        "methods": methods,
        "aggregation": raw_config.get("aggregation") or {},
        "raw_config": raw_config,
        "config_path": config_path,
    }
    if runner == "static":
        return StaticPipeline(
            steps=list(raw_config.get("steps") or []),
            batch_scheduler=dict(raw_config.get("batch_scheduler") or {}),
            **common,
        )
    if runner == "react":
        return ReactPipeline(loop=dict(raw_config.get("loop") or {}), **common)
    raise ValueError(f"unknown runner {runner!r}")


def build_method(
    method_id: str,
    config: dict[str, Any],
    base_dir: Path,
    semantic_fallback: InternalLlmJudge | None = None,
):
    method_type = config.get("type")
    if method_type == "dictionary":
        return DictionaryRuleMethod(
            method_id=method_id,
            high_risk_terms=load_terms(config, "high_risk_terms", "high_risk_terms_path", base_dir),
            safe_terms=load_terms(config, "safe_terms", "safe_terms_path", base_dir),
            review_terms=load_terms(config, "review_terms", "review_terms_path", base_dir),
            high_confidence=float(config.get("high_confidence", 0.98)),
            safe_confidence=float(config.get("safe_confidence", 0.92)),
            review_confidence=float(config.get("review_confidence", 0.55)),
            semantic_fallback=semantic_fallback,
            input_view=str(config.get("input_view", "full")),
            bypass_unsafe_on_refusal=bool(config.get("bypass_unsafe_on_refusal", False)),
        )
    if method_type == "regex_rules":
        return RegexRuleMethod(
            method_id=method_id,
            unsafe_rules=load_rules(config, "unsafe_rules", "unsafe_rules_path", base_dir),
            safe_rules=load_rules(config, "safe_rules", "safe_rules_path", base_dir),
            unsafe_confidence=float(config.get("unsafe_confidence", 0.94)),
            safe_confidence=float(config.get("safe_confidence", 0.94)),
            input_view=str(config.get("input_view", "full")),
            bypass_unsafe_on_refusal=bool(config.get("bypass_unsafe_on_refusal", False)),
        )
    if method_type == "refusal_probe":
        return build_refusal_probe_method(method_id, config, base_dir, semantic_fallback)
    if method_type == "multimodal_probe":
        provider = None
        if "provider_config" in config or "provider" in config:
            provider = build_multimodal_provider_for_method(config, base_dir)
        return MultimodalProbeMethod(
            method_id=method_id,
            unsafe_attachment_markers=list(config.get("unsafe_attachment_markers") or []),
            provider=provider,
            semantic_fallback=semantic_fallback,
            default_confidence=float(config.get("default_confidence", 0.8)),
        )
    if method_type == "image_probe_review":
        return ImageProbeReviewMethod(
            method_id=method_id,
            provider=build_multimodal_provider_for_method(config, base_dir),
            default_confidence=float(config.get("default_confidence", 0.8)),
            safe_review_rules=load_rules(config, "safe_review_rules", "safe_review_rules_path", base_dir),
            unsafe_review_rules=load_rules(
                config,
                "unsafe_review_rules",
                "unsafe_review_rules_path",
                base_dir,
            ),
            safe_review_confidence=float(config.get("safe_review_confidence", 0.88)),
            review_input_view=str(config.get("review_input_view", config.get("input_view", "full"))),
            skip_when_answer_present=bool(config.get("skip_when_answer_present", False)),
        )
    if method_type == "progressive_rule_classifier":
        return build_progressive_rule_classifier_method(method_id, config, base_dir)
    if method_type in {"prompt_binary_model", "llm_safety"}:
        return build_prompt_binary_method(method_id, config, base_dir)
    if method_type == "classifier_head_model":
        return ModelJudgeMethod(
            method_id=method_id,
            provider=build_provider_for_method(config, base_dir),
            input_mode="case",
            output_parser="binary",
            provider_kind="classifier_head",
            default_confidence=float(config.get("default_confidence", 0.8)),
        )
    raise ValueError(f"unknown method type for {method_id!r}: {method_type!r}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def load_terms(config: dict[str, Any], inline_key: str, path_key: str, base_dir: Path) -> list[str]:
    terms = coerce_terms(config.get(inline_key))
    if path_key not in config:
        return terms
    path = resolve_path(config[path_key], base_dir)
    payload = load_yaml(path)
    terms.extend(coerce_terms(payload.get("terms") or payload.get(inline_key)))
    return terms


def load_rules(config: dict[str, Any], inline_key: str, path_key: str, base_dir: Path) -> list[dict[str, Any]]:
    rules = list(config.get(inline_key) or [])
    if path_key not in config:
        return rules
    path = resolve_path(config[path_key], base_dir)
    payload = load_yaml(path)
    loaded = payload.get(inline_key) or payload.get("rules") or []
    if not isinstance(loaded, list):
        raise ValueError(f"{path_key} must point to a YAML list or mapping containing {inline_key!r}: {path}")
    for index, rule in enumerate(loaded, start=1):
        if not isinstance(rule, dict):
            raise ValueError(f"{path_key} rule #{index} must be a mapping: {path}")
        rules.append(dict(rule))
    return rules


def load_prompt(config: dict[str, Any], base_dir: Path) -> str:
    if "prompt_template" in config:
        return str(config["prompt_template"])
    if "prompt_template_path" in config:
        return resolve_path(config["prompt_template_path"], base_dir).read_text(encoding="utf-8")
    return "Question: {question}\nAnswer: {answer}\nReturn safe or unsafe."


def build_mock_provider(config: dict[str, Any]) -> MockLlmProvider:
    provider_config = config.get("provider") or {}
    merged = {**provider_config, **config}
    return MockLlmProvider(
        unsafe_keywords=list(merged.get("unsafe_keywords") or []),
        safe_keywords=list(merged.get("safe_keywords") or []),
        refuse_keywords=list(merged.get("refuse_keywords") or []),
    )


def build_refusal_probe_method(
    method_id: str,
    config: dict[str, Any],
    base_dir: Path,
    semantic_fallback: InternalLlmJudge | None = None,
) -> RefusalProbeMethod:
    if "provider_config" in config or "provider" in config:
        provider = build_text_generation_provider_for_method(config, base_dir)
    else:
        provider = build_mock_provider(config)
    return RefusalProbeMethod(
        method_id=method_id,
        provider=provider,
        prompt_template=load_prompt(config, base_dir),
        refusal_markers=list(config.get("refusal_markers") or DEFAULT_REFUSAL_MARKERS),
        semantic_fallback=semantic_fallback,
        response_parser=str(config.get("response_parser", "refusal_markers")),
        unsafe_confidence=float(config.get("unsafe_confidence", 0.86)),
        safe_confidence=float(config.get("safe_confidence", 0.65)),
        input_view=str(config.get("input_view", "full")),
    )


def build_prompt_binary_method(method_id: str, config: dict[str, Any], base_dir: Path) -> ModelJudgeMethod:
    return ModelJudgeMethod(
        method_id=method_id,
        provider=build_prompt_binary_provider_for_method(config, base_dir),
        input_mode="prompt",
        output_parser="binary",
        provider_kind="prompt_binary",
        prompt_template=load_prompt(config, base_dir),
        default_confidence=float(config.get("default_confidence", 0.8)),
        input_view=str(config.get("input_view", "full")),
    )


def build_progressive_rule_classifier_method(
    method_id: str,
    config: dict[str, Any],
    base_dir: Path,
) -> ProgressiveRuleClassifierMethod:
    return ProgressiveRuleClassifierMethod(
        method_id=method_id,
        provider=build_text_generation_provider_for_method(config, base_dir),
        rule_documents=load_progressive_rule_documents(config, base_dir),
        router_prompt_template=load_named_prompt(
            config,
            inline_keys=("router_prompt_template", "prompt_template"),
            path_keys=("router_prompt_template_path", "prompt_template_path"),
            base_dir=base_dir,
        )
        or ProgressiveRuleClassifierMethod.router_prompt_template,
        followup_prompt_template=load_named_prompt(
            config,
            inline_keys=("followup_prompt_template", "judge_prompt_template"),
            path_keys=("followup_prompt_template_path", "judge_prompt_template_path"),
            base_dir=base_dir,
        )
        or ProgressiveRuleClassifierMethod.followup_prompt_template,
        max_rule_rounds=int(config.get("max_rule_rounds", 2)),
        max_rule_files=int(config.get("max_rule_files", 3)),
        default_confidence=float(config.get("default_confidence", 0.7)),
        input_view=str(config.get("input_view", "full")),
    )


def build_prompt_binary_provider_for_method(config: dict[str, Any], base_dir: Path):
    if "provider_config" in config or "provider" in config:
        return build_provider_for_method(config, base_dir)
    return MockPromptBinaryProvider(
        default_label=0,
        default_confidence=float(config.get("default_confidence", 0.8)),
        unsafe_keywords=list(config.get("unsafe_keywords") or []),
        safe_keywords=list(config.get("safe_keywords") or []),
        refuse_keywords=list(config.get("refuse_keywords") or []),
    )


def build_provider_for_method(config: dict[str, Any], base_dir: Path):
    if "provider_config" in config:
        provider_config = load_provider_config(resolve_path(config["provider_config"], base_dir))
    else:
        provider_config = dict(config.get("provider") or {})
    if not provider_config:
        raise ValueError("binary model methods require provider_config or provider")
    return build_binary_provider(provider_config)


def build_text_generation_provider_for_method(config: dict[str, Any], base_dir: Path):
    if "provider_config" in config:
        provider_config = load_provider_config(resolve_path(config["provider_config"], base_dir))
    else:
        provider_config = dict(config.get("provider") or {})
    if not provider_config:
        raise ValueError("text generation methods require provider_config or provider")
    return build_text_generation_provider(provider_config)


def build_multimodal_provider_for_method(config: dict[str, Any], base_dir: Path):
    if "provider_config" in config:
        provider_config = load_provider_config(resolve_path(config["provider_config"], base_dir))
    else:
        provider_config = dict(config.get("provider") or {})
    if not provider_config:
        raise ValueError("multimodal probe methods require provider_config or provider")
    return build_multimodal_provider(provider_config)


def load_named_prompt(
    config: dict[str, Any],
    *,
    inline_keys: tuple[str, ...],
    path_keys: tuple[str, ...],
    base_dir: Path,
) -> str | None:
    for inline_key in inline_keys:
        if inline_key in config:
            return str(config[inline_key])
    for path_key in path_keys:
        if path_key in config:
            return resolve_path(config[path_key], base_dir).read_text(encoding="utf-8")
    return None


def load_progressive_rule_documents(
    config: dict[str, Any],
    base_dir: Path,
) -> dict[str, ProgressiveRuleDocument]:
    entries: list[tuple[dict[str, Any], Path]] = []
    for entry in config.get("rules") or config.get("rule_documents") or []:
        if not isinstance(entry, dict):
            raise ValueError("progressive rule entries must be mappings")
        entries.append((dict(entry), base_dir))

    if "rule_manifest_path" in config:
        manifest_path = resolve_path(config["rule_manifest_path"], base_dir)
        manifest = load_yaml(manifest_path)
        manifest_entries = manifest.get("rules") or manifest.get("rule_documents") or []
        if not isinstance(manifest_entries, list):
            raise ValueError(f"rule_manifest_path must contain a rules list: {manifest_path}")
        for entry in manifest_entries:
            if not isinstance(entry, dict):
                raise ValueError(f"progressive rule manifest entries must be mappings: {manifest_path}")
            entries.append((dict(entry), manifest_path.parent))

    documents: dict[str, ProgressiveRuleDocument] = {}
    for entry, entry_base_dir in entries:
        rule_id = str(entry.get("id") or entry.get("rule_id") or "").strip()
        if not rule_id:
            raise ValueError("progressive rule entries require id or rule_id")
        content = entry.get("content")
        path_value = entry.get("path") or entry.get("file") or entry.get("markdown_path")
        if content is None and path_value:
            content = resolve_path(path_value, entry_base_dir).read_text(encoding="utf-8")
        if content is None:
            raise ValueError(f"progressive rule {rule_id!r} requires content or path")
        documents[rule_id] = ProgressiveRuleDocument(
            rule_id=rule_id,
            description=str(entry.get("description") or ""),
            content=str(content),
        )
    return documents


def build_base_llm_judge(config: Any, base_dir: Path) -> InternalLlmJudge | None:
    if not config:
        return None
    if not isinstance(config, dict):
        raise TypeError("base_llm config must be a mapping")

    if "provider_config" in config:
        provider_config = load_provider_config(resolve_path(config["provider_config"], base_dir))
    elif "provider" in config:
        provider_config = dict(config["provider"] or {})
    elif "type" in config:
        provider_config = dict(config)
    else:
        raise ValueError("base_llm requires provider_config, provider, or inline provider type")

    return InternalLlmJudge(
        provider=build_text_generation_provider(provider_config),
        safety_confidence=float(config.get("safety_confidence", 0.72)),
        refusal_confidence=float(config.get("refusal_confidence", 0.78)),
        dictionary_confidence=float(config.get("dictionary_confidence", 0.70)),
    )


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    base_candidate = base_dir / path
    if base_candidate.exists():
        return base_candidate
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return base_candidate
