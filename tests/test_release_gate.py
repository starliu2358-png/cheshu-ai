from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from cockpit_agent.vehicle_book import release_gate


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _candidate(tmp_path: Path) -> Path:
    candidate = tmp_path / "candidate"
    candidate.mkdir(parents=True)
    _write(
        candidate / "retrieval_summary.json",
        {
            "top_k": 3,
            "recall_at_3": 1.0,
            "cross_vehicle_error_rate": 0.0,
            "cross_version_error_rate": 0.0,
        },
    )
    _write(
        candidate / "retrieval_case_results.json",
        [
            {
                "case_id": "positive",
                "vehicle_id": "alpha_aster_x1_max_v1",
                "critical": True,
                "expected_no_answer": False,
                "recall_at_3": 1.0,
                "no_answer_false_positive": None,
            },
            {
                "case_id": "fallback",
                "vehicle_id": "alpha_aster_x1_max_v1",
                "critical": False,
                "expected_no_answer": True,
                "recall_at_3": None,
                "no_answer_false_positive": False,
            },
        ],
    )
    _write(
        candidate / "answer_summary.json",
        {"top_k": 4, "citation_accuracy": 1.0, "cited_cases": 1},
    )
    _write(
        candidate / "answer_case_results.json",
        [
            {
                "case_id": "positive",
                "vehicle_id": "alpha_aster_x1_max_v1",
                "cited_chunk_ids": ["oem_alpha_aster_x1_v1_adas:acc"],
                "expected_fallback": False,
                "fallback_correct": True,
                "critical_error": False,
            },
            {
                "case_id": "fallback",
                "vehicle_id": "alpha_aster_x1_max_v1",
                "cited_chunk_ids": [],
                "expected_fallback": True,
                "fallback_correct": True,
                "critical_error": False,
            },
        ],
    )
    return candidate


def _baseline(candidate: Path, tmp_path: Path) -> Path:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    for name in release_gate.BASELINE_REPORTS:
        shutil.copyfile(candidate / name, baseline / name)
    _write(
        baseline / "baseline_manifest.json",
        release_gate.create_baseline_manifest(candidate, pytest_passed=True),
    )
    return baseline


def _result(candidate: Path, baseline: Path) -> dict[str, Any]:
    return release_gate.evaluate_release_gate(candidate, baseline, pytest_passed=True, latency_p95_ms=12.0)


def _gate(result: dict[str, Any], name: str) -> dict[str, Any]:
    return next(gate for gate in result["gates"] if gate["name"] == name)


def test_all_hard_gates_pass_and_soft_live_is_not_run(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    result = _result(candidate, _baseline(candidate, tmp_path))

    assert result["decision"] == "PASS"
    assert _gate(result, "p95_retrieval_latency")["status"] == "PASS"
    assert _gate(result, "live_api_acceptance")["status"] == "NOT_RUN"
    assert "DEEPSEEK" not in release_gate.render_markdown_report(result)


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("schema_metadata_validation", lambda value: value["schema_errors"].append("invalid")),
        ("duplicate_active_ids", lambda value: value["duplicate_document_ids"].append("duplicate")),
        ("duplicate_active_ids", lambda value: value["duplicate_chunk_ids"].append("duplicate")),
        ("broken_citation_anchors", lambda value: value["broken_anchors"].append("broken")),
    ],
)
def test_validation_hard_gates_fail_individually(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, mutate: Any
) -> None:
    candidate = _candidate(tmp_path)
    baseline = _baseline(candidate, tmp_path)
    original = release_gate._registry_scan

    def broken(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
        scan, chunks = original(rows)
        mutate(scan)
        return scan, chunks

    monkeypatch.setattr(release_gate, "_registry_scan", broken)
    result = _result(candidate, baseline)

    assert result["decision"] == "FAIL"
    assert _gate(result, name)["status"] == "FAIL"


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("cross_vehicle_error_rate", 0.1),
        ("cross_version_error_rate", 0.1),
        ("recall_at_3", 0.89),
    ],
)
def test_retrieval_hard_gates_fail(tmp_path: Path, target: str, value: float) -> None:
    candidate = _candidate(tmp_path)
    baseline = _baseline(candidate, tmp_path)
    summary = json.loads((candidate / "retrieval_summary.json").read_text(encoding="utf-8"))
    summary[target] = value
    _write(candidate / "retrieval_summary.json", summary)

    result = _result(candidate, baseline)

    assert result["decision"] == "FAIL"


@pytest.mark.parametrize(
    ("summary_key", "row_update", "gate_name"),
    [
        ("citation_accuracy", None, "citation_correctness"),
        (None, {"fallback_correct": False}, "fallback_no_answer_accuracy"),
        (None, {"critical_error": True}, "critical_error"),
    ],
)
def test_answer_hard_gates_fail(
    tmp_path: Path, summary_key: str | None, row_update: dict[str, bool] | None, gate_name: str
) -> None:
    candidate = _candidate(tmp_path)
    baseline = _baseline(candidate, tmp_path)
    if summary_key:
        summary = json.loads((candidate / "answer_summary.json").read_text(encoding="utf-8"))
        summary[summary_key] = 0.94
        _write(candidate / "answer_summary.json", summary)
    else:
        rows = json.loads((candidate / "answer_case_results.json").read_text(encoding="utf-8"))
        rows[-1 if gate_name == "fallback_no_answer_accuracy" else 0].update(row_update or {})
        _write(candidate / "answer_case_results.json", rows)

    result = _result(candidate, baseline)

    assert result["decision"] == "FAIL"
    assert _gate(result, gate_name)["status"] == "FAIL"


def test_pytest_failure_and_candidate_regression_fail(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    baseline = _baseline(candidate, tmp_path)
    result = release_gate.evaluate_release_gate(candidate, baseline, pytest_passed=False)

    assert result["decision"] == "FAIL"
    assert _gate(result, "pytest")["status"] == "FAIL"

    summary = json.loads((candidate / "answer_summary.json").read_text(encoding="utf-8"))
    summary["citation_accuracy"] = 0.96
    _write(candidate / "answer_summary.json", summary)
    result = _result(candidate, baseline)
    assert _gate(result, "candidate_vs_baseline")["status"] == "FAIL"


def test_missing_reports_or_baseline_blocks(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    (candidate / "answer_summary.json").unlink()
    assert release_gate.evaluate_release_gate(candidate, tmp_path / "missing", pytest_passed=True)["decision"] == "BLOCKED"

    candidate = _candidate(tmp_path / "second")
    assert release_gate.evaluate_release_gate(candidate, tmp_path / "missing", pytest_passed=True)["decision"] == "BLOCKED"


@pytest.mark.parametrize(
    "manifest_update",
    [
        {"report_format_version": "other"},
        {"top_k": {"retrieval": 4, "answer": 4}},
        {"datasets": {"retrieval": {"path": "other", "checksum": "other"}}},
    ],
)
def test_incompatible_baseline_is_not_comparable_and_blocks(
    tmp_path: Path, manifest_update: dict[str, Any]
) -> None:
    candidate = _candidate(tmp_path)
    baseline = _baseline(candidate, tmp_path)
    manifest = json.loads((baseline / "baseline_manifest.json").read_text(encoding="utf-8"))
    manifest.update(manifest_update)
    _write(baseline / "baseline_manifest.json", manifest)

    result = _result(candidate, baseline)

    assert result["decision"] == "BLOCKED"
    assert _gate(result, "candidate_vs_baseline")["status"] == "NOT_COMPARABLE"


def test_no_answer_accuracy_does_not_include_positive_cases(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    baseline = _baseline(candidate, tmp_path)
    rows = json.loads((candidate / "answer_case_results.json").read_text(encoding="utf-8"))
    rows[0]["fallback_correct"] = False
    _write(candidate / "answer_case_results.json", rows)

    result = _result(candidate, baseline)

    gate = _gate(result, "fallback_no_answer_accuracy")
    assert gate["actual"] == 1.0
    assert gate["sample_count"] == 1
    assert gate["status"] == "PASS"


def test_latency_review_does_not_block_a_pass(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    result = release_gate.evaluate_release_gate(candidate, _baseline(candidate, tmp_path), pytest_passed=True, latency_p95_ms=501)

    assert result["decision"] == "PASS"
    assert _gate(result, "p95_retrieval_latency")["status"] == "REVIEW"
