from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cockpit_agent.vehicle_book.agent_evaluation import (
    AgentControlEvalCase,
    AgentControlEvaluationError,
    evaluate_agent_controls,
    load_agent_control_eval_cases,
)


def test_agent_control_dataset_is_strict() -> None:
    case = AgentControlEvalCase.from_dict(
        {
            "case_id": "case",
            "scenario": "single_search",
            "category": "tool_loop",
            "critical": True,
        }
    )
    assert case.scenario == "single_search"
    with pytest.raises(AgentControlEvaluationError, match="unsupported scenario"):
        AgentControlEvalCase.from_dict(
            {
                "case_id": "bad",
                "scenario": "unknown",
                "category": "tool_loop",
                "critical": True,
            }
        )


def test_all_agent_control_cases_pass_without_model_or_network() -> None:
    summary = evaluate_agent_controls(load_agent_control_eval_cases())

    assert summary.total_cases == 10
    assert summary.pass_rate == 1.0
    assert summary.critical_failures == 0
    assert summary.failed_case_ids == ()


def test_agent_control_eval_cli_writes_repeatable_reports(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_agent_control_eval.py",
            "--output-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(
        (tmp_path / "agent_control_summary.json").read_text(encoding="utf-8")
    )
    assert summary["pass_rate"] == 1.0
    assert summary["failed_case_ids"] == []
