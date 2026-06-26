from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from safeguard_harness.runtimes.devices import (
    coerce_torch_dtype,
    import_torch,
    patch_broken_triton_namespace,
    resolve_torch_device,
)

SAFETY_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe)", re.IGNORECASE)


@dataclass
class MergedSafeGuardRuntime:
    model: Any
    processor: Any
    device: str


def parse_safety_label(text: str | None) -> str | None:
    if text is None:
        return None
    match = SAFETY_PATTERN.search(text)
    if not match:
        return None
    return "Unsafe" if match.group(1).casefold() == "unsafe" else "Safe"


def load_merged_safeguard(
    model_path: str,
    device: str = "auto",
    torch_dtype: str = "bfloat16",
    device_map: Any | None = None,
    max_memory: dict[Any, Any] | None = None,
    offload_folder: str | None = None,
    disable_torch_compile: bool = False,
    patch_torch_distributed_tensor: bool = False,
) -> MergedSafeGuardRuntime:
    patch_broken_triton_namespace()
    torch = import_torch()
    original_torch_compile = _apply_transformers_load_compat(
        torch,
        disable_torch_compile=disable_torch_compile,
        patch_torch_distributed_tensor=patch_torch_distributed_tensor,
    )
    resolved_device = resolve_torch_device(device)
    dtype = coerce_torch_dtype(torch, torch_dtype)

    try:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "merged SafeGuard local inference requires transformers. Install the local-model dependencies."
            ) from exc

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory
        if offload_folder is not None:
            from pathlib import Path

            Path(offload_folder).mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = offload_folder
        model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    finally:
        if original_torch_compile is not None:
            torch.compile = original_torch_compile

    runtime_device = _model_input_device(model, fallback=resolved_device) if device_map is not None else resolved_device
    if device_map is None:
        model.to(resolved_device)
    model.eval()

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.do_sample = False
        generation_config.top_k = None
        generation_config.top_p = None
        generation_config.temperature = None

    return MergedSafeGuardRuntime(model=model, processor=processor, device=str(runtime_device))


def _model_input_device(model: Any, fallback: str) -> str:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for device in hf_device_map.values():
            normalized = _normalize_device_name(device)
            if normalized not in {"cpu", "disk", "meta"}:
                return normalized
        for device in hf_device_map.values():
            normalized = _normalize_device_name(device)
            if normalized not in {"disk", "meta"}:
                return normalized
    model_device = getattr(model, "device", None)
    if model_device is not None:
        normalized = str(model_device)
        if normalized not in {"meta", "disk"}:
            return normalized
    try:
        return str(next(model.parameters()).device)
    except (AttributeError, StopIteration):
        return fallback


def _normalize_device_name(device: Any) -> str:
    if isinstance(device, int):
        return f"cuda:{device}"
    normalized = str(device)
    if normalized.isdigit():
        return f"cuda:{normalized}"
    return normalized


def _apply_transformers_load_compat(
    torch: Any,
    *,
    disable_torch_compile: bool,
    patch_torch_distributed_tensor: bool,
) -> Any | None:
    original_torch_compile = None
    if patch_torch_distributed_tensor:
        _patch_torch_distributed_tensor_namespace()
    if disable_torch_compile and hasattr(torch, "compile"):
        original_torch_compile = torch.compile

        def identity_compile(model: Any = None, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if model is None:
                return lambda wrapped: wrapped
            return model

        torch.compile = identity_compile
    return original_torch_compile


def _patch_torch_distributed_tensor_namespace() -> None:
    try:
        import torch.distributed._tensor as source_module
        import torch.distributed.tensor as target_module
    except (ImportError, AttributeError):
        return

    for name in ("DTensor", "Placement", "Replicate", "Shard", "distribute_module"):
        if not hasattr(target_module, name) and hasattr(source_module, name):
            setattr(target_module, name, getattr(source_module, name))


def build_chat_input(runtime: MergedSafeGuardRuntime, prompt: str) -> dict[str, Any]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = runtime.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = runtime.processor(text=[text], images=None, videos=None, return_tensors="pt")
    return {key: value.to(runtime.device) for key, value in inputs.items()}


def infer_safety(
    runtime: MergedSafeGuardRuntime,
    prompt: str,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    torch = import_torch()
    inputs = build_chat_input(runtime, prompt)
    with torch.inference_mode():
        generated = runtime.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated[:, input_len:]
    output_text = runtime.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return {
        "prediction_text": output_text,
        "prediction_label": parse_safety_label(output_text),
    }
