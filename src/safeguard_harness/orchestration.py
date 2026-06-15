from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from safeguard_harness.core import SAFE, UNSAFE, Decision, MethodResult, RunContext, RunTrace, SafetyCase, TraceStep
from safeguard_harness.methods import JudgeMethod


@dataclass
class Pipeline:
    runner: str
    methods: dict[str, JudgeMethod]
    aggregation: dict[str, Any] = field(default_factory=dict)
    raw_config: dict[str, Any] = field(default_factory=dict)

    def judge(self, case: SafetyCase) -> Decision:
        raise NotImplementedError

    def aggregate(self, case_id: str, trace: RunTrace) -> Decision:
        threshold = float(self.aggregation.get("unsafe_threshold", 0.5))
        results = [step.result for step in trace.steps if not step.result.skipped]
        if not results:
            return Decision(
                case_id=case_id,
                label=SAFE,
                unsafe_score=0.0,
                confidence=0.0,
                reasons=["no method results"],
                trace=trace,
            )

        unsafe_score = max(result.unsafe_score for result in results)
        unsafe_results = [result for result in results if result.label == UNSAFE]
        if unsafe_results:
            confidence = max(result.confidence for result in unsafe_results)
        else:
            confidence = sum(result.confidence for result in results) / len(results)
        label = UNSAFE if unsafe_score >= threshold else SAFE
        reasons = [
            evidence
            for result in results
            for evidence in result.evidence
            if result.label == UNSAFE or result.unsafe_score > 0
        ]
        if not reasons:
            reasons = ["no unsafe evidence above threshold"]
        return Decision(
            case_id=case_id,
            label=label,
            unsafe_score=unsafe_score,
            confidence=confidence,
            reasons=reasons,
            trace=trace,
        )


@dataclass
class StaticPipeline(Pipeline):
    steps: list[dict[str, Any]] = field(default_factory=list)

    def judge(self, case: SafetyCase) -> Decision:
        trace = RunTrace(case_id=case.id, runner="static")
        context = RunContext()
        for step in self.steps:
            if "method" in step:
                result = self._run_method(step["method"], case, context)
                trace.add_step(
                    TraceStep(
                        step_id=str(step.get("id") or step["method"]),
                        method_id=step["method"],
                        result=result,
                    )
                )
                if result.label == UNSAFE and step.get("on_unsafe") == "stop":
                    trace.stop_reason = f"short_circuit:{step.get('id') or step['method']}"
                    return Decision(
                        case_id=case.id,
                        label=UNSAFE,
                        unsafe_score=result.unsafe_score,
                        confidence=result.confidence,
                        reasons=list(result.evidence),
                        trace=trace,
                    )
                continue

            if "repeat" in step:
                self._run_repeat(step_id=str(step.get("id") or "repeat"), spec=step["repeat"], case=case, context=context, trace=trace)

        trace.stop_reason = trace.stop_reason or "completed"
        return self.aggregate(case.id, trace)

    def _run_repeat(
        self,
        *,
        step_id: str,
        spec: dict[str, Any],
        case: SafetyCase,
        context: RunContext,
        trace: RunTrace,
    ) -> None:
        max_rounds = int(spec.get("max_rounds", 1))
        method_ids = list(spec.get("methods") or [])
        for round_index in range(1, max_rounds + 1):
            decision = self.aggregate(case.id, trace)
            if not _should_repeat(decision, spec.get("when") or {}):
                return
            for method_id in method_ids:
                result = self._run_method(method_id, case, context)
                trace.add_step(
                    TraceStep(
                        step_id=f"{step_id}.round{round_index}.{method_id}",
                        method_id=method_id,
                        result=result,
                    )
                )

    def _run_method(self, method_id: str, case: SafetyCase, context: RunContext) -> MethodResult:
        try:
            method = self.methods[method_id]
        except KeyError as exc:
            raise KeyError(f"unknown method {method_id!r}") from exc
        return method.judge(case, context)


@dataclass
class ReactPipeline(Pipeline):
    loop: dict[str, Any] = field(default_factory=dict)

    def judge(self, case: SafetyCase) -> Decision:
        trace = RunTrace(case_id=case.id, runner="react")
        context = RunContext()
        max_steps = int(self.loop.get("max_steps", 4))
        max_llm_calls = self.loop.get("max_llm_calls")
        allowed_actions = list(self.loop.get("allowed_actions") or self.methods.keys())
        stop_when = self.loop.get("stop_when") or {}

        for action_index, method_id in enumerate(allowed_actions[:max_steps], start=1):
            if max_llm_calls is not None and context.llm_calls >= int(max_llm_calls):
                trace.stop_reason = "llm_budget_exhausted"
                break
            result = self.methods[method_id].judge(case, context)
            trace.add_step(
                TraceStep(
                    step_id=f"react.step{action_index}.{method_id}",
                    method_id=method_id,
                    result=result,
                    metadata={"action": method_id},
                )
            )
            decision = self.aggregate(case.id, trace)
            if _stop_condition_met(decision, stop_when):
                trace.stop_reason = "stop_condition_met"
                return decision

        if not trace.stop_reason:
            trace.stop_reason = "budget_exhausted"
        if "fallback" in self.loop and trace.stop_reason in {"budget_exhausted", "llm_budget_exhausted"}:
            fallback = self.loop["fallback"]
            return Decision(
                case_id=case.id,
                label=str(fallback.get("label", SAFE)),
                unsafe_score=1.0 if fallback.get("label") == UNSAFE else 0.0,
                confidence=0.0,
                reasons=[str(fallback.get("reason", trace.stop_reason))],
                trace=trace,
            )
        return self.aggregate(case.id, trace)


def _should_repeat(decision: Decision, when: dict[str, Any]) -> bool:
    if "confidence_lt" in when and not decision.confidence < float(when["confidence_lt"]):
        return False
    if "unsafe_score_lt" in when and not decision.unsafe_score < float(when["unsafe_score_lt"]):
        return False
    return bool(when)


def _stop_condition_met(decision: Decision, stop_when: dict[str, Any]) -> bool:
    if not stop_when:
        return False
    if "confidence_gte" in stop_when and decision.confidence >= float(stop_when["confidence_gte"]):
        return True
    if "unsafe_score_gte" in stop_when and decision.unsafe_score >= float(stop_when["unsafe_score_gte"]):
        return True
    return False

