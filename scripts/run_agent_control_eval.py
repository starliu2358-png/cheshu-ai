"""Run the model-free Vehicle Book Agent control-plane evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from cockpit_agent.vehicle_book.agent_evaluation import (
    DEFAULT_AGENT_CONTROL_EVAL_PATH,
    AgentControlEvaluationError,
    evaluate_agent_controls,
    load_agent_control_eval_cases,
)

DEFAULT_OUTPUT_DIR = Path("outputs/eval")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic Vehicle Book Agent control evaluation."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_AGENT_CONTROL_EVAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = evaluate_agent_controls(load_agent_control_eval_cases(args.dataset))
        case_rows = [
            {
                "case_id": result.case.case_id,
                "scenario": result.case.scenario,
                "category": result.case.category,
                "critical": result.case.critical,
                "passed": result.passed,
                "details": dict(result.details),
            }
            for result in summary.results
        ]
        summary_payload = {
            "dataset_path": str(args.dataset),
            "total_cases": summary.total_cases,
            "passed_cases": summary.passed_cases,
            "pass_rate": summary.pass_rate,
            "critical_failures": summary.critical_failures,
            "failed_case_ids": list(summary.failed_case_ids),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "agent_control_case_results.json").write_text(
            json.dumps(case_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "agent_control_summary.json").write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Agent Control Cases: {summary.total_cases}")
        print(f"Pass Rate: {summary.pass_rate:.2%}")
        print(f"Critical Failures: {summary.critical_failures}")
        print(
            "Failed Case IDs: "
            + (" | ".join(summary.failed_case_ids) if summary.failed_case_ids else "None")
        )
        return 0 if not summary.failed_case_ids else 1
    except (OSError, ValueError, AgentControlEvaluationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
