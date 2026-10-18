"""Local, deterministic release gating for Vehicle Book evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluation import load_retrieval_eval_cases
from .ingest import KnowledgeIngestor
from .registry import (
    DEFAULT_DOCUMENTS_PATH,
    DEFAULT_VEHICLES_PATH,
    REPOSITORY_ROOT,
    DocumentRegistry,
    RegistryError,
    VehicleRegistry,
)
from .retriever import VehicleBookRetriever

REPORT_FORMAT_VERSION = "vehicle-book-release-gate-v1"
BASELINE_REPORTS = (
    "retrieval_summary.json",
    "retrieval_case_results.json",
    "answer_summary.json",
    "answer_case_results.json",
)
REQUIRED_CANDIDATE_REPORTS = BASELINE_REPORTS
HARD = "hard"
SOFT = "soft"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, (dict, list)):
        raise TypeError(f"{path} must contain a JSON object or array")
    return payload


def _gate(
    name: str,
    category: str,
    hard_or_soft: str,
    threshold: str,
    actual: Any,
    sample_count: int | Mapping[str, int],
    status: str,
    reason: str,
    evidence_path: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "category": category,
        "hard_or_soft": hard_or_soft,
        "threshold": threshold,
        "actual": actual,
        "sample_count": sample_count,
        "status": status,
        "reason": reason,
        "evidence_path": evidence_path,
    }


def _dataset_context() -> dict[str, dict[str, str]]:
    paths = {
        "retrieval": REPOSITORY_ROOT / "data/eval/vehicle_book_retrieval_eval.jsonl",
        "answer": REPOSITORY_ROOT / "data/eval/vehicle_book_answer_eval.jsonl",
    }
    return {
        name: {
            "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
            "checksum": _sha256(path),
        }
        for name, path in paths.items()
    }


def _registry_checksum() -> str:
    digest = hashlib.sha256()
    for path in (DEFAULT_VEHICLES_PATH, DEFAULT_DOCUMENTS_PATH):
        digest.update(path.relative_to(REPOSITORY_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    documents = DocumentRegistry()
    for document in sorted(documents._documents.values(), key=lambda item: item.document_id):
        source = documents.repository_root / document.source_path
        digest.update(document.source_path.as_posix().encode("utf-8"))
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _required_files(directory: Path, names: Sequence[str]) -> list[str]:
    return [name for name in names if not (directory / name).is_file()]


def _load_candidate(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    missing = _required_files(directory, REQUIRED_CANDIDATE_REPORTS)
    if missing:
        raise FileNotFoundError("missing candidate reports: " + ", ".join(missing))
    retrieval_summary = _json(directory / "retrieval_summary.json")
    retrieval_cases = _json(directory / "retrieval_case_results.json")
    answer_summary = _json(directory / "answer_summary.json")
    answer_cases = _json(directory / "answer_case_results.json")
    if not isinstance(retrieval_summary, dict) or not isinstance(answer_summary, dict):
        raise TypeError("candidate summaries must be JSON objects")
    if not isinstance(retrieval_cases, list) or not isinstance(answer_cases, list):
        raise TypeError("candidate case results must be JSON arrays")
    return retrieval_summary, retrieval_cases, answer_summary, answer_cases


def _registry_scan(answer_cases: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    try:
        registry = VehicleRegistry()
        documents = DocumentRegistry(vehicle_registry=registry)
        ingestor = KnowledgeIngestor(vehicle_registry=registry, document_registry=documents)
        document_ids = [item.document_id for item in documents._documents.values() if item.status == "published"]
        chunks: dict[str, Any] = {}
        duplicates: list[str] = []
        for profile in registry.list_profiles(include_pre_release=True):
            for chunk in ingestor.ingest(profile.vehicle_id):
                if chunk.chunk_id in chunks:
                    duplicates.append(chunk.chunk_id)
                chunks[chunk.chunk_id] = chunk
    except (OSError, RegistryError, ValueError) as exc:
        return (
            {
                "profile_count": 0,
                "published_document_count": 0,
                "published_chunk_count": 0,
                "schema_errors": [str(exc)],
                "duplicate_document_ids": [],
                "duplicate_chunk_ids": [],
                "broken_anchors": [],
            },
            [],
        )

    broken: list[str] = []
    for row in answer_cases:
        case_id = str(row.get("case_id", "<unknown>"))
        vehicle_id = row.get("vehicle_id")
        cited_ids = row.get("cited_chunk_ids", [])
        if not isinstance(cited_ids, list):
            broken.append(f"{case_id}: cited_chunk_ids is not an array")
            continue
        for chunk_id in cited_ids:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                broken.append(f"{case_id}: unknown citation anchor {chunk_id}")
            elif chunk.metadata.vehicle_id != vehicle_id:
                broken.append(f"{case_id}: citation anchor belongs to another vehicle")

    return (
        {
            "profile_count": len(registry.list_profiles(include_pre_release=True)),
            "published_document_count": len(document_ids),
            "published_chunk_count": len(chunks),
            "schema_errors": [],
            "duplicate_document_ids": sorted(
                key for key, count in Counter(document_ids).items() if count > 1
            ),
            "duplicate_chunk_ids": sorted(set(duplicates)),
            "broken_anchors": broken,
        },
        sorted(chunks),
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _slice_metrics(retrieval_cases: Sequence[Mapping[str, Any]], answer_cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        registry = VehicleRegistry()
    except RegistryError:
        return {}
    groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in retrieval_cases:
        profile = registry.resolve(str(row["vehicle_id"]))
        groups[f"oem:{profile.oem}"].append(row)
        groups[f"model:{profile.model}"].append(row)
        groups[f"software_version:{profile.software_version}"].append(row)
        if bool(row.get("critical")):
            groups["risk:high_or_critical"].append(row)
        if bool(row.get("expected_no_answer")):
            groups["no-answer"].append(row)

    fallback_by_case = {
        str(row.get("case_id")): bool(row.get("fallback_correct"))
        for row in answer_cases
        if bool(row.get("expected_fallback"))
    }
    metrics: dict[str, Any] = {}
    for name, rows in sorted(groups.items()):
        positive = [row for row in rows if not bool(row.get("expected_no_answer"))]
        if positive:
            recall = _mean([float(row["recall_at_3"]) for row in positive])
            metrics[name] = {
                "sample_count": len(rows),
                "positive_sample_count": len(positive),
                "recall_at_3": recall,
            }
        else:
            false_positive = _mean(
                [float(bool(row.get("no_answer_false_positive"))) for row in rows]
            )
            matched = [fallback_by_case.get(str(row.get("case_id"))) for row in rows]
            values = [float(value) for value in matched if value is not None]
            metrics[name] = {
                "sample_count": len(rows),
                "positive_sample_count": 0,
                "recall_at_3": None,
                "no_answer_false_positive_rate": false_positive,
                "fallback_accuracy": _mean(values),
            }
    return metrics


def _comparison_metrics(
    retrieval_summary: Mapping[str, Any],
    answer_summary: Mapping[str, Any],
    answer_cases: Sequence[Mapping[str, Any]],
    slices: Mapping[str, Any],
    scan: Mapping[str, Any],
    pytest_passed: bool | None,
) -> dict[str, Any]:
    fallback_rows = [row for row in answer_cases if bool(row.get("expected_fallback"))]
    fallback = _mean([float(bool(row.get("fallback_correct"))) for row in fallback_rows])
    return {
        "schema_validation_pass_rate": 1.0 if not scan["schema_errors"] else 0.0,
        "duplicate_active_document_id_count": len(scan["duplicate_document_ids"]),
        "duplicate_active_chunk_id_count": len(scan["duplicate_chunk_ids"]),
        "broken_citation_anchor_count": len(scan["broken_anchors"]),
        "wrong_vehicle_retrieval_rate": retrieval_summary.get("cross_vehicle_error_rate"),
        "wrong_version_retrieval_rate": retrieval_summary.get("cross_version_error_rate"),
        "retrieval_recall_at_3": retrieval_summary.get("recall_at_3"),
        "retrieval_slices": slices,
        "citation_correctness": answer_summary.get("citation_accuracy"),
        "fallback_no_answer_accuracy": fallback,
        "fallback_no_answer_sample_count": len(fallback_rows),
        "critical_error_count": sum(bool(row.get("critical_error")) for row in answer_cases),
        "pytest_pass_rate": 1.0 if pytest_passed else 0.0 if pytest_passed is not None else None,
    }


def _compatible(manifest: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if manifest.get("report_format_version") != REPORT_FORMAT_VERSION:
        reasons.append("report format version differs")
    if manifest.get("datasets") != candidate["datasets"]:
        reasons.append("dataset path or checksum differs")
    if manifest.get("top_k") != candidate["top_k"]:
        reasons.append("top_k differs")
    if manifest.get("registry_checksum") != candidate["registry_checksum"]:
        reasons.append("registry or document-source checksum differs")
    return reasons


def _regressions(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    lower_is_bad = ("retrieval_recall_at_3", "citation_correctness", "fallback_no_answer_accuracy", "pytest_pass_rate")
    higher_is_bad = (
        "duplicate_active_document_id_count",
        "duplicate_active_chunk_id_count",
        "broken_citation_anchor_count",
        "wrong_vehicle_retrieval_rate",
        "wrong_version_retrieval_rate",
        "critical_error_count",
    )
    for key in (*lower_is_bad, *higher_is_bad):
        if key not in baseline or candidate.get(key) is None or baseline.get(key) is None:
            reasons.append(f"baseline metric missing: {key}")
        elif (key in lower_is_bad and candidate[key] < baseline[key]) or (
            key in higher_is_bad and candidate[key] > baseline[key]
        ):
            reasons.append(f"hard metric regressed: {key}")
    baseline_slices = baseline.get("retrieval_slices")
    candidate_slices = candidate.get("retrieval_slices")
    if not isinstance(baseline_slices, Mapping) or not isinstance(candidate_slices, Mapping):
        reasons.append("baseline retrieval slices missing")
    else:
        for name, baseline_slice in baseline_slices.items():
            candidate_slice = candidate_slices.get(name)
            if not isinstance(candidate_slice, Mapping):
                reasons.append(f"baseline slice missing: {name}")
            elif baseline_slice.get("positive_sample_count") != candidate_slice.get("positive_sample_count"):
                reasons.append(f"slice sample count differs: {name}")
            elif baseline_slice.get("recall_at_3") is not None and (
                candidate_slice.get("recall_at_3") is None
                or candidate_slice["recall_at_3"] < baseline_slice["recall_at_3"]
            ):
                reasons.append(f"hard metric regressed: retrieval slice {name}")
    return reasons


def evaluate_release_gate(
    candidate_dir: Path | str,
    baseline_dir: Path | str | None,
    *,
    pytest_passed: bool | None,
    latency_p95_ms: float | None = None,
    live_status: str = "NOT_RUN",
    live_reason: str = "Live API acceptance was not explicitly executed.",
    require_baseline: bool = True,
) -> dict[str, Any]:
    """Evaluate deterministic release gates from existing report artifacts."""
    candidate_path = Path(candidate_dir)
    gates: list[dict[str, Any]] = []
    try:
        retrieval_summary, retrieval_cases, answer_summary, answer_cases = _load_candidate(candidate_path)
    except (FileNotFoundError, TypeError, ValueError, KeyError, OSError) as exc:
        gates.append(_gate("candidate_reports", "input", HARD, "complete and readable", None, 0, "BLOCKED", str(exc), str(candidate_path)))
        return {"report_format_version": REPORT_FORMAT_VERSION, "decision": "BLOCKED", "gates": gates, "comparison_metrics": {}}
    scan, _ = _registry_scan(answer_cases)
    slices = _slice_metrics(retrieval_cases, answer_cases)

    metrics = _comparison_metrics(retrieval_summary, answer_summary, answer_cases, slices, scan, pytest_passed)
    evidence = str(candidate_path)
    gates.extend(
        [
            _gate("schema_metadata_validation", "validation", HARD, "100%", metrics["schema_validation_pass_rate"], scan["published_document_count"], "PASS" if not scan["schema_errors"] else "FAIL", "Registry and all published sources loaded." if not scan["schema_errors"] else "; ".join(scan["schema_errors"]), evidence),
            _gate("duplicate_active_ids", "validation", HARD, "0 document_id and 0 chunk_id", {"document_id": metrics["duplicate_active_document_id_count"], "chunk_id": metrics["duplicate_active_chunk_id_count"]}, scan["published_chunk_count"], "PASS" if not scan["duplicate_document_ids"] and not scan["duplicate_chunk_ids"] else "FAIL", "No duplicate active IDs." if not scan["duplicate_document_ids"] and not scan["duplicate_chunk_ids"] else "Duplicate active IDs found.", evidence),
            _gate("broken_citation_anchors", "validation", HARD, "0", metrics["broken_citation_anchor_count"], len(answer_cases), "PASS" if not scan["broken_anchors"] else "FAIL", "All cited chunk IDs resolve to the exact vehicle." if not scan["broken_anchors"] else "; ".join(scan["broken_anchors"]), evidence),
            _gate("wrong_vehicle_wrong_version_retrieval", "retrieval", HARD, "0%", {"wrong_vehicle": metrics["wrong_vehicle_retrieval_rate"], "wrong_version": metrics["wrong_version_retrieval_rate"]}, len(retrieval_cases), "PASS" if metrics["wrong_vehicle_retrieval_rate"] == 0 and metrics["wrong_version_retrieval_rate"] == 0 else "FAIL", "Rates are derived from retrieval evaluation.", evidence),
        ]
    )
    failing_slices = [name for name, value in slices.items() if value["recall_at_3"] is not None and value["recall_at_3"] < 0.8]
    recall_ok = metrics["retrieval_recall_at_3"] is not None and metrics["retrieval_recall_at_3"] >= 0.9 and not failing_slices
    gates.append(_gate("retrieval_recall_at_3", "retrieval", HARD, "overall >= 90%; each positive required slice >= 80%", {"overall": metrics["retrieval_recall_at_3"], "slices": slices}, {name: value["sample_count"] for name, value in slices.items()}, "PASS" if recall_ok else "FAIL", "All required positive slices meet threshold." if recall_ok else "Failed slices: " + ", ".join(failing_slices), evidence))
    fallback_ok = metrics["fallback_no_answer_accuracy"] is not None and metrics["fallback_no_answer_accuracy"] >= 0.9
    gates.extend(
        [
            _gate("citation_correctness", "answer", HARD, ">= 95%", metrics["citation_correctness"], int(answer_summary.get("cited_cases", 0)), "PASS" if metrics["citation_correctness"] is not None and metrics["citation_correctness"] >= 0.95 else "FAIL", "Answer evaluation citation accuracy.", evidence),
            _gate("fallback_no_answer_accuracy", "answer", HARD, ">= 90%; expected_fallback cases only", metrics["fallback_no_answer_accuracy"], metrics["fallback_no_answer_sample_count"], "PASS" if fallback_ok else "FAIL", "Computed only from expected_fallback=true rows.", evidence),
            _gate("critical_error", "answer", HARD, "0", metrics["critical_error_count"], len(answer_cases), "PASS" if metrics["critical_error_count"] == 0 else "FAIL", "Critical errors are counted from case results.", evidence),
            _gate("pytest", "test", HARD, "100%", metrics["pytest_pass_rate"], 1, "PASS" if pytest_passed is True else "FAIL" if pytest_passed is False else "BLOCKED", "pytest result supplied by CLI." if pytest_passed is not None else "pytest was not executed.", evidence),
        ]
    )

    context_error: str | None = None
    try:
        registry_checksum = _registry_checksum()
    except (OSError, RegistryError, ValueError, TypeError) as exc:
        registry_checksum = None
        context_error = str(exc)
    candidate_context = {
        "datasets": _dataset_context(),
        "registry_checksum": registry_checksum,
        "top_k": {"retrieval": retrieval_summary.get("top_k"), "answer": answer_summary.get("top_k")},
    }
    baseline_path = Path(baseline_dir) if baseline_dir is not None else None
    if baseline_path is None or not (baseline_path / "baseline_manifest.json").is_file():
        status = "BLOCKED" if require_baseline else "NOT_RUN"
        gates.append(_gate("candidate_vs_baseline", "comparison", HARD, "no hard metric regression", None, 0, status, "Baseline manifest is missing." if require_baseline else "Baseline comparison omitted during initialization.", str(baseline_path) if baseline_path else ""))
    else:
        missing = _required_files(baseline_path, BASELINE_REPORTS)
        manifest = _json(baseline_path / "baseline_manifest.json")
        if not isinstance(manifest, dict) or missing:
            gates.append(_gate("candidate_vs_baseline", "comparison", HARD, "no hard metric regression", None, 0, "BLOCKED", "Baseline reports missing: " + ", ".join(missing), str(baseline_path)))
        else:
            try:
                _load_candidate(baseline_path)
            except (FileNotFoundError, TypeError, ValueError, OSError) as exc:
                gates.append(_gate("candidate_vs_baseline", "comparison", HARD, "no hard metric regression", None, 0, "BLOCKED", f"Invalid baseline reports: {exc}", str(baseline_path)))
                compatibility = None
            else:
                compatibility = _compatible(manifest, candidate_context)
            baseline_metrics = manifest.get("hard_metrics")
            if context_error is not None:
                gates.append(_gate("candidate_vs_baseline", "comparison", HARD, "no hard metric regression", None, 0, "FAIL", f"Candidate registry context is invalid: {context_error}", str(baseline_path)))
            elif compatibility is None:
                pass
            elif compatibility or not isinstance(baseline_metrics, Mapping):
                reason = "; ".join(compatibility or ["baseline hard_metrics missing"])
                gates.append(_gate("candidate_vs_baseline", "comparison", HARD, "no hard metric regression", None, 0, "NOT_COMPARABLE", reason, str(baseline_path)))
            else:
                regressions = _regressions(metrics, baseline_metrics)
                gates.append(_gate("candidate_vs_baseline", "comparison", HARD, "no hard metric regression", {"baseline_id": manifest.get("baseline_id"), "regressions": regressions}, len(metrics), "PASS" if not regressions else "FAIL", "No hard metric regression." if not regressions else "; ".join(regressions), str(baseline_path)))

    latency_status = "NOT_RUN" if latency_p95_ms is None else "PASS" if latency_p95_ms <= 500 else "REVIEW"
    gates.append(_gate("p95_retrieval_latency", "performance", SOFT, "<= 500 ms", latency_p95_ms, len(retrieval_cases) if latency_p95_ms is not None else 0, latency_status, "Latency was not measured." if latency_p95_ms is None else "Local retrieval P95 measurement.", evidence))
    gates.append(_gate("live_api_acceptance", "live", SOFT, "explicit live run required", live_status, 0, live_status, live_reason, evidence))
    hard_statuses = [gate["status"] for gate in gates if gate["hard_or_soft"] == HARD]
    decision = "BLOCKED" if "BLOCKED" in hard_statuses or "NOT_COMPARABLE" in hard_statuses else "FAIL" if "FAIL" in hard_statuses else "PASS"
    return {
        "report_format_version": REPORT_FORMAT_VERSION,
        "decision": decision,
        "candidate_dir": str(candidate_path),
        "baseline_dir": str(baseline_path) if baseline_path else None,
        "candidate_context": candidate_context,
        "comparison_metrics": metrics,
        "gates": gates,
    }


def measure_retrieval_p95_ms() -> float:
    """Measure local retrieval latency without changing retriever behavior."""
    retriever = VehicleBookRetriever()
    durations: list[float] = []
    for case in load_retrieval_eval_cases():
        start = time.perf_counter()
        retriever.retrieve(case.vehicle_id, case.question, top_k=3, include_pre_release=case.include_pre_release)
        durations.append((time.perf_counter() - start) * 1000)
    return sorted(durations)[max(0, math.ceil(len(durations) * 0.95) - 1)]


def create_baseline_manifest(candidate_dir: Path | str, *, pytest_passed: bool) -> dict[str, Any]:
    """Create the immutable metadata stored alongside copied baseline reports."""
    result = evaluate_release_gate(candidate_dir, None, pytest_passed=pytest_passed, require_baseline=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False).stdout.strip() or "unknown"
    return {
        "baseline_id": "vehicle_book_release_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision,
        "datasets": result["candidate_context"]["datasets"],
        "registry_checksum": result["candidate_context"]["registry_checksum"],
        "top_k": result["candidate_context"]["top_k"],
        "report_format_version": REPORT_FORMAT_VERSION,
        "hard_metrics": result["comparison_metrics"],
    }


def render_markdown_report(summary: Mapping[str, Any]) -> str:
    lines = ["# Vehicle Book Release Gate", "", f"**Decision: {summary['decision']}**", "", "| Gate | Type | Threshold | Actual | Samples | Status |", "| --- | --- | --- | --- | ---: | --- |"]
    for gate in summary["gates"]:
        actual = json.dumps(gate["actual"], ensure_ascii=False, sort_keys=True)
        samples = gate["sample_count"] if isinstance(gate["sample_count"], int) else json.dumps(gate["sample_count"], ensure_ascii=False, sort_keys=True)
        lines.append(f"| {gate['name']} | {gate['hard_or_soft']} | {gate['threshold']} | {actual} | {samples} | {gate['status']} |")
    lines.extend(["", "## Reasons", ""])
    for gate in summary["gates"]:
        lines.append(f"- `{gate['name']}`: {gate['reason']} ({gate['evidence_path']})")
    return "\n".join(lines) + "\n"
