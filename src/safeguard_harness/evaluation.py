from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from safeguard_harness.core import SAFE, UNSAFE, Decision, SafetyCase
from safeguard_harness.datasets import write_jsonl
from safeguard_harness.orchestration import Pipeline


@dataclass
class EvaluationSummary:
    metrics: dict[str, Any]
    predictions: list[dict[str, Any]]
    output_dir: Path


def evaluate_dataset(
    pipeline: Pipeline,
    cases: Iterable[SafetyCase],
    output_dir: str | Path,
    *,
    config_snapshot: dict[str, Any] | None = None,
) -> EvaluationSummary:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    decisions: list[tuple[SafetyCase, Decision]] = []

    for case in cases:
        decision = pipeline.judge(case)
        decisions.append((case, decision))
        predictions.append({"case": case.to_dict(), "decision": decision.to_dict()})

    metrics = compute_metrics(decisions)
    write_jsonl(output / "predictions.jsonl", predictions)
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(output / "errors_false_positive.jsonl", _false_positives(decisions))
    write_jsonl(output / "errors_false_negative.jsonl", _false_negatives(decisions))
    (output / "report.md").write_text(render_report(metrics), encoding="utf-8")
    if config_snapshot is not None:
        (output / "config_snapshot.yaml").write_text(
            yaml.safe_dump(config_snapshot, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return EvaluationSummary(metrics=metrics, predictions=predictions, output_dir=output)


def compute_metrics(decisions: list[tuple[SafetyCase, Decision]]) -> dict[str, Any]:
    labeled = [(case, decision) for case, decision in decisions if case.label in {SAFE, UNSAFE}]
    tp = sum(1 for case, decision in labeled if case.label == UNSAFE and decision.label == UNSAFE)
    tn = sum(1 for case, decision in labeled if case.label == SAFE and decision.label == SAFE)
    fp = sum(1 for case, decision in labeled if case.label == SAFE and decision.label == UNSAFE)
    fn = sum(1 for case, decision in labeled if case.label == UNSAFE and decision.label == SAFE)
    total = len(labeled)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "total": total,
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        },
        "by_modality": _metrics_by_modality(labeled),
        "method_calls": sum(len(decision.trace.steps) if decision.trace else 0 for _, decision in decisions),
    }


def render_report(metrics: dict[str, Any]) -> str:
    confusion = metrics["confusion_matrix"]
    return "\n".join(
        [
            "# Safeguard Harness Evaluation Report",
            "",
            f"- Total labeled cases: {metrics['total']}",
            f"- Accuracy: {metrics['accuracy']:.4f}",
            f"- Precision: {metrics['precision']:.4f}",
            f"- Recall: {metrics['recall']:.4f}",
            f"- F1: {metrics['f1']:.4f}",
            f"- TP/TN/FP/FN: {confusion['true_positive']}/{confusion['true_negative']}/{confusion['false_positive']}/{confusion['false_negative']}",
            f"- Method calls: {metrics['method_calls']}",
            "",
        ]
    )


def _false_positives(decisions: list[tuple[SafetyCase, Decision]]) -> list[dict[str, Any]]:
    return [
        {"case": case.to_dict(), "decision": decision.to_dict()}
        for case, decision in decisions
        if case.label == SAFE and decision.label == UNSAFE
    ]


def _false_negatives(decisions: list[tuple[SafetyCase, Decision]]) -> list[dict[str, Any]]:
    return [
        {"case": case.to_dict(), "decision": decision.to_dict()}
        for case, decision in decisions
        if case.label == UNSAFE and decision.label == SAFE
    ]


def _metrics_by_modality(labeled: list[tuple[SafetyCase, Decision]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[tuple[SafetyCase, Decision]]] = {}
    for case, decision in labeled:
        grouped.setdefault(case.modality, []).append((case, decision))
    return {
        modality: {
            "total": len(items),
            "accuracy": _safe_div(
                sum(1 for case, decision in items if case.label == decision.label),
                len(items),
            ),
        }
        for modality, items in grouped.items()
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0

