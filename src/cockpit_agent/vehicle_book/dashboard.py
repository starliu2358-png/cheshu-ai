"""Read-only normalization of existing evaluation artifacts for the UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import REPOSITORY_ROOT

DEFAULT_EVAL_DIR = REPOSITORY_ROOT / "outputs" / "eval"
DEFAULT_RETRIEVAL_REPORT_PATH = DEFAULT_EVAL_DIR / "retrieval_summary.json"
DEFAULT_ANSWER_REPORT_PATH = DEFAULT_EVAL_DIR / "answer_summary.json"
DEFAULT_RELEASE_GATE_REPORT_PATH = (
    REPOSITORY_ROOT / "outputs" / "release_gate" / "release_gate_summary.json"
)
KNOWN_STATUSES = frozenset({"PASS", "FAIL", "BLOCKED", "REVIEW", "NOT_RUN", "NOT_COMPARABLE"})


@dataclass(frozen=True, slots=True)
class DashboardReport:
    status: str
    source_path: str
    generated_at: str
    message: str


@dataclass(frozen=True, slots=True)
class DashboardGate:
    name: str
    category: str
    threshold: str
    sample_count: str
    status: str
    reason: str
    evidence_path: str


@dataclass(frozen=True, slots=True)
class DashboardMetric:
    name: str
    value: float | int | str | None
    sample_count: int | None = None


@dataclass(frozen=True, slots=True)
class DashboardSlice:
    name: str
    source: str
    sample_count: int | None
    positive_sample_count: int | None
    recall_at_3: float | None
    answer_accuracy: float | None
    citation_accuracy: float | None
    fallback_accuracy: float | None
    grounded_answer_rate: float | None
    critical_error_rate: float | None


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    status: str
    availability: str
    reason: str
    source_path: str


@dataclass(frozen=True, slots=True)
class EvaluationDashboardView:
    release_gate: DashboardReport
    retrieval: DashboardReport
    answer: DashboardReport
    overall_decision: str
    hard_gates: tuple[DashboardGate, ...]
    soft_gates: tuple[DashboardGate, ...]
    retrieval_metrics: tuple[DashboardMetric, ...]
    answer_metrics: tuple[DashboardMetric, ...]
    slices: tuple[DashboardSlice, ...]
    failed_case_ids: tuple[str, ...]
    answer_mode: str
    live_status: str
    baseline: BaselineComparison


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.name


def _text(value: Any, fallback: str = "—") -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _load_report(path: Path, label: str) -> tuple[DashboardReport, dict[str, Any] | None]:
    source_path = _display_path(path)
    if not path.is_file():
        return (
            DashboardReport("NOT_RUN", source_path, "报告未提供生成时间", f"未运行：{label}报告缺失。"),
            None,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (
            DashboardReport("BLOCKED", source_path, "报告未提供生成时间", f"{label}报告无法读取：{exc.__class__.__name__}。"),
            None,
        )
    if not isinstance(payload, dict):
        return (
            DashboardReport("BLOCKED", source_path, "报告未提供生成时间", f"{label}报告格式无效。"),
            None,
        )
    generated_at = _text(payload.get("generated_at"), "报告未提供生成时间")
    return DashboardReport("AVAILABLE", source_path, generated_at, f"已读取{label}报告。"), payload


def _gate_view(gate: dict[str, Any]) -> DashboardGate:
    sample_count = gate.get("sample_count")
    return DashboardGate(
        name=_text(gate.get("name")),
        category=_text(gate.get("category")),
        threshold=_text(gate.get("threshold")),
        sample_count=str(sample_count) if isinstance(sample_count, (int, str)) else "—",
        status=_text(gate.get("status"), "BLOCKED"),
        reason=_text(gate.get("reason")),
        evidence_path=_text(gate.get("evidence_path")),
    )


def _baseline(gates: tuple[DashboardGate, ...], report_available: bool) -> BaselineComparison:
    if not report_available:
        return BaselineComparison("NOT_RUN", "NOT_RUN", "未运行：Release Gate 报告缺失。", "—")
    gate = next((item for item in gates if item.name == "candidate_vs_baseline"), None)
    if gate is None:
        return BaselineComparison("NOT_RUN", "NOT_RUN", "报告未包含 Candidate vs Baseline 门禁。", "—")
    reason_lower = gate.reason.lower()
    if "missing" in reason_lower:
        availability = "MISSING"
    elif gate.status == "NOT_COMPARABLE":
        availability = "NOT_COMPARABLE"
    else:
        availability = "AVAILABLE"
    return BaselineComparison(gate.status, availability, gate.reason, gate.evidence_path)


def _failed_ids(payload: dict[str, Any] | None, *fields: str) -> set[str]:
    if payload is None:
        return set()
    return {
        value
        for field in fields
        for value in payload.get(field, [])
        if isinstance(payload.get(field), list) and isinstance(value, str) and value
    }


def load_evaluation_dashboard(
    *,
    retrieval_report_path: Path = DEFAULT_RETRIEVAL_REPORT_PATH,
    answer_report_path: Path = DEFAULT_ANSWER_REPORT_PATH,
    release_gate_report_path: Path = DEFAULT_RELEASE_GATE_REPORT_PATH,
) -> EvaluationDashboardView:
    """Load existing JSON reports without evaluating, writing, networking, or model calls."""
    retrieval_report, retrieval = _load_report(retrieval_report_path, "Retrieval Eval")
    answer_report, answer = _load_report(answer_report_path, "Answer Eval")
    gate_report, release_gate = _load_report(release_gate_report_path, "Release Gate")

    raw_gates = release_gate.get("gates", []) if release_gate else []
    gates = tuple(
        sorted(
            (_gate_view(item) for item in raw_gates if isinstance(item, dict)),
            key=lambda item: (item.name, item.category),
        )
    )
    hard_gates = tuple(item for item in gates if item.category != "—" and any(
        isinstance(raw, dict) and _text(raw.get("name")) == item.name and raw.get("hard_or_soft") == "hard"
        for raw in raw_gates
    ))
    soft_gates = tuple(item for item in gates if item not in hard_gates)
    decision = _text(release_gate.get("decision"), "BLOCKED") if release_gate else gate_report.status
    if decision not in KNOWN_STATUSES:
        decision = "BLOCKED"

    retrieval_metrics = tuple(
        DashboardMetric(name, _number(retrieval.get(key)) if retrieval else None, _integer(retrieval.get("total_cases")) if retrieval else None)
        for name, key in (
            ("Recall@K", "recall_at_3"), ("MRR", "mrr"),
            ("跨车型错误率", "cross_vehicle_error_rate"),
            ("跨版本错误率", "cross_version_error_rate"),
        )
    )
    answer_metrics = tuple(
        DashboardMetric(name, _number(answer.get(key)) if answer else None, _integer(answer.get("total_cases")) if answer else None)
        for name, key in (
            ("Answer Accuracy", "answer_accuracy"), ("Citation Accuracy", "citation_accuracy"),
            ("Fallback Accuracy", "fallback_accuracy"), ("Grounded Answer Rate", "grounded_answer_rate"),
            ("Critical Error Rate", "critical_error_rate"),
        )
    )
    slices: list[DashboardSlice] = []
    if answer and isinstance(answer.get("category_metrics"), dict):
        for name, metric in sorted(answer["category_metrics"].items()):
            if isinstance(metric, dict):
                slices.append(DashboardSlice(name, "Answer Category", _integer(metric.get("total_cases")), _integer(metric.get("positive_cases")), None, _number(metric.get("answer_accuracy")), _number(metric.get("citation_accuracy")), _number(metric.get("fallback_accuracy")), _number(metric.get("grounded_answer_rate")), _number(metric.get("critical_error_rate"))))
    comparison_metrics = release_gate.get("comparison_metrics", {}) if release_gate else {}
    if isinstance(comparison_metrics, dict) and isinstance(comparison_metrics.get("retrieval_slices"), dict):
        for name, metric in sorted(comparison_metrics["retrieval_slices"].items()):
            if isinstance(metric, dict):
                slices.append(DashboardSlice(name, "Release Gate Slice", _integer(metric.get("sample_count")), _integer(metric.get("positive_sample_count")), _number(metric.get("recall_at_3")), None, None, _number(metric.get("fallback_accuracy")), None, None))

    return EvaluationDashboardView(
        release_gate=gate_report,
        retrieval=retrieval_report,
        answer=answer_report,
        overall_decision=decision,
        hard_gates=hard_gates,
        soft_gates=soft_gates,
        retrieval_metrics=retrieval_metrics,
        answer_metrics=answer_metrics,
        slices=tuple(slices),
        failed_case_ids=tuple(sorted(_failed_ids(retrieval, "failed_case_ids") | _failed_ids(answer, "failed_case_ids", "runtime_error_case_ids"))),
        answer_mode=_text(answer.get("mode"), "NOT_RUN") if answer else "NOT_RUN",
        live_status=next((gate.status for gate in gates if gate.name == "live_api_acceptance"), "NOT_RUN"),
        baseline=_baseline(gates, release_gate is not None),
    )
