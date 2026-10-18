import hashlib
import json
from pathlib import Path

from cockpit_agent.vehicle_book.dashboard import load_evaluation_dashboard


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _reports(tmp_path: Path, baseline_status: str = "PASS", reason: str = "No regression.") -> tuple[Path, Path, Path]:
    retrieval = tmp_path / "retrieval.json"
    answer = tmp_path / "answer.json"
    gate = tmp_path / "gate.json"
    _write(retrieval, {"top_k": 3, "total_cases": 7, "recall_at_3": 0.8, "mrr": 0.7, "cross_vehicle_error_rate": 0.1, "cross_version_error_rate": 0.2, "failed_case_ids": ["r-1"]})
    _write(answer, {"mode": "replay", "total_cases": 7, "answer_accuracy": 0.6, "citation_accuracy": 0.7, "fallback_accuracy": 0.8, "grounded_answer_rate": 0.9, "critical_error_rate": 0.1, "failed_case_ids": ["a-1"], "runtime_error_case_ids": ["a-runtime"], "category_metrics": {"comfort": {"total_cases": 3, "positive_cases": 2, "answer_accuracy": 0.5, "citation_accuracy": 0.6, "fallback_accuracy": 1.0, "grounded_answer_rate": 0.5, "critical_error_rate": 0.0}}})
    _write(gate, {"decision": "FAIL", "gates": [
        {"name": "hard_pass", "category": "validation", "hard_or_soft": "hard", "threshold": "1", "sample_count": 7, "status": "PASS", "reason": "ok", "evidence_path": "outputs/eval"},
        {"name": "hard_fail", "category": "answer", "hard_or_soft": "hard", "threshold": "0", "sample_count": 7, "status": "FAIL", "reason": "failed", "evidence_path": "outputs/eval"},
        {"name": "candidate_vs_baseline", "category": "comparison", "hard_or_soft": "hard", "threshold": "no regression", "sample_count": 7, "status": baseline_status, "reason": reason, "evidence_path": "data/baselines/test"},
        {"name": "p95", "category": "performance", "hard_or_soft": "soft", "threshold": "500", "sample_count": 7, "status": "REVIEW", "reason": "slow", "evidence_path": "outputs/eval"},
        {"name": "live_api_acceptance", "category": "live", "hard_or_soft": "soft", "threshold": "run", "sample_count": 0, "status": "NOT_RUN", "reason": "not run", "evidence_path": "outputs/eval"}
    ], "comparison_metrics": {"retrieval_slices": {"model:X": {"sample_count": 4, "positive_sample_count": 3, "recall_at_3": 0.75}}}})
    return retrieval, answer, gate


def test_dashboard_normalizes_existing_report_values_and_slices(tmp_path: Path) -> None:
    retrieval, answer, gate = _reports(tmp_path)

    view = load_evaluation_dashboard(retrieval_report_path=retrieval, answer_report_path=answer, release_gate_report_path=gate)

    assert view.overall_decision == "FAIL"
    assert {item.status for item in view.hard_gates} >= {"PASS", "FAIL"}
    assert {item.status for item in view.soft_gates} == {"REVIEW", "NOT_RUN"}
    assert view.retrieval_metrics[0].value == 0.8
    assert view.retrieval_metrics[0].sample_count == 7
    assert view.answer_metrics[0].value == 0.6
    assert view.answer_mode == "replay"
    assert view.live_status == "NOT_RUN"
    assert view.failed_case_ids == ("a-1", "a-runtime", "r-1")
    assert [(item.name, item.sample_count) for item in view.slices] == [("comfort", 3), ("model:X", 4)]
    assert view.baseline.status == "PASS"
    assert view.baseline.availability == "AVAILABLE"


def test_dashboard_handles_missing_reports_without_implying_success(tmp_path: Path) -> None:
    view = load_evaluation_dashboard(retrieval_report_path=tmp_path / "missing-r.json", answer_report_path=tmp_path / "missing-a.json", release_gate_report_path=tmp_path / "missing-g.json")

    assert view.overall_decision == "NOT_RUN"
    assert view.retrieval.status == "NOT_RUN"
    assert "报告缺失" in view.answer.message
    assert view.answer_mode == "NOT_RUN"
    assert view.live_status == "NOT_RUN"
    assert view.baseline.availability == "NOT_RUN"


def test_dashboard_distinguishes_missing_and_not_comparable_baselines(tmp_path: Path) -> None:
    retrieval, answer, gate = _reports(tmp_path, baseline_status="BLOCKED", reason="Baseline manifest is missing.")
    missing = load_evaluation_dashboard(retrieval_report_path=retrieval, answer_report_path=answer, release_gate_report_path=gate)
    assert missing.baseline.availability == "MISSING"
    assert missing.baseline.status == "BLOCKED"

    _reports(tmp_path, baseline_status="NOT_COMPARABLE", reason="top_k differs")
    incomparable = load_evaluation_dashboard(retrieval_report_path=retrieval, answer_report_path=answer, release_gate_report_path=gate)
    assert incomparable.baseline.availability == "NOT_COMPARABLE"
    assert incomparable.baseline.status == "NOT_COMPARABLE"


def test_dashboard_only_reads_input_reports(tmp_path: Path) -> None:
    retrieval, answer, gate = _reports(tmp_path)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (retrieval, answer, gate)}

    load_evaluation_dashboard(retrieval_report_path=retrieval, answer_report_path=answer, release_gate_report_path=gate)

    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before} == before
