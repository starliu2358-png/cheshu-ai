from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_release_gate


def _summary(decision: str = "PASS") -> dict[str, object]:
    return {
        "decision": decision,
        "gates": [
            {
                "name": "live_api_acceptance",
                "status": "NOT_RUN",
                "hard_or_soft": "soft",
                "threshold": "explicit live run required",
                "actual": "NOT_RUN",
                "sample_count": 0,
                "reason": "not run",
                "evidence_path": "",
            }
        ],
    }


def test_default_cli_is_offline_and_writes_stable_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fake_command(arguments: list[str]) -> int:
        commands.append(arguments)
        return 0

    monkeypatch.setattr(run_release_gate, "_run_command", fake_command)
    monkeypatch.setattr(run_release_gate, "measure_retrieval_p95_ms", lambda: 7.5)
    monkeypatch.setattr(run_release_gate, "evaluate_release_gate", lambda *args, **kwargs: _summary())
    monkeypatch.setattr(run_release_gate, "render_markdown_report", lambda value: "# report\n")

    exit_code = run_release_gate.main(
        ["--candidate-dir", str(tmp_path / "candidate"), "--baseline-dir", str(tmp_path / "baseline"), "--output-dir", str(tmp_path / "out"), "--json-only"]
    )

    assert exit_code == 0
    assert any("scripts/run_answer_eval.py" in command for command in commands)
    assert all("live" not in command for command in commands)
    assert (tmp_path / "out/release_gate_summary.json").is_file()
    assert (tmp_path / "out/release_gate_report.md").read_text(encoding="utf-8") == "# report\n"


@pytest.mark.parametrize("decision", ["FAIL", "BLOCKED"])
def test_cli_returns_nonzero_for_non_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: str
) -> None:
    monkeypatch.setattr(run_release_gate, "_run_command", lambda arguments: 0)
    monkeypatch.setattr(run_release_gate, "measure_retrieval_p95_ms", lambda: 1.0)
    monkeypatch.setattr(run_release_gate, "evaluate_release_gate", lambda *args, **kwargs: _summary(decision))
    monkeypatch.setattr(run_release_gate, "render_markdown_report", lambda value: "# report\n")

    assert run_release_gate.main(["--output-dir", str(tmp_path), "--json-only"]) == 1


def test_initialize_baseline_requires_force_when_present(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()

    assert run_release_gate.main(["--baseline-dir", str(baseline), "--initialize-baseline"]) == 1


def test_json_report_contains_no_environment_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_release_gate, "_run_command", lambda arguments: 0)
    monkeypatch.setattr(run_release_gate, "measure_retrieval_p95_ms", lambda: 1.0)
    monkeypatch.setattr(run_release_gate, "evaluate_release_gate", lambda *args, **kwargs: _summary())
    monkeypatch.setattr(run_release_gate, "render_markdown_report", lambda value: "# report\n")

    run_release_gate.main(["--output-dir", str(tmp_path), "--json-only"])

    payload = json.loads((tmp_path / "release_gate_summary.json").read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert "API_KEY" not in serialized
    assert ".env" not in serialized
