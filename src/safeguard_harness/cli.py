from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from safeguard_harness.config import load_pipeline
from safeguard_harness.core import SafetyCase
from safeguard_harness.datasets import load_jsonl_cases, write_jsonl
from safeguard_harness.evaluation import evaluate_dataset


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="safeguard-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    judge = subparsers.add_parser("judge", help="Judge one question through a configured pipeline.")
    judge.add_argument("--pipeline", required=True)
    judge.add_argument("--question", required=True)
    judge.add_argument("--answer")
    judge.add_argument("--id", default="case")
    judge.set_defaults(func=cmd_judge)

    predict = subparsers.add_parser("predict", help="Run batch prediction over a JSONL file.")
    predict.add_argument("--pipeline", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.set_defaults(func=cmd_predict)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a pipeline on labeled JSONL data.")
    evaluate.add_argument("--pipeline", required=True)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(func=cmd_evaluate)

    return parser


def cmd_judge(args: argparse.Namespace) -> int:
    pipeline = load_pipeline(args.pipeline)
    case = SafetyCase(id=args.id, question=args.question, answer=args.answer)
    decision = pipeline.judge(case)
    print(json.dumps(decision.to_dict(), ensure_ascii=False))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    pipeline = load_pipeline(args.pipeline)
    cases = load_jsonl_cases(args.input)
    rows = []
    for case in cases:
        decision = pipeline.judge(case)
        rows.append({"case_id": case.id, **decision.to_dict()})
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} predictions to {Path(args.output)}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    pipeline = load_pipeline(args.pipeline)
    cases = load_jsonl_cases(args.dataset)
    summary = evaluate_dataset(
        pipeline,
        cases,
        args.output,
        config_snapshot=pipeline.raw_config,
    )
    print(json.dumps({"accuracy": summary.metrics["accuracy"], "metrics": summary.metrics}, ensure_ascii=False))
    return 0

