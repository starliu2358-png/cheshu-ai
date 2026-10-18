"""Run Vehicle Book answer evaluation in live or offline replay mode."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cockpit_agent.vehicle_book.answer_evaluation import (
    AnswerCaseResult,
    AnswerEvalSummary,
    AnswerEvaluationError,
    evaluate_answer_results,
    load_answer_eval_cases,
)
from cockpit_agent.vehicle_book.answerer import DeepSeekAnswerGenerator, GroundedAnswerer
from cockpit_agent.vehicle_book.schemas import (
    AnswerResult,
    Citation,
    DocumentMetadata,
    KnowledgeChunk,
    RetrievedChunk,
)
from cockpit_agent.vehicle_book.service import VehicleBookService

DEFAULT_DATASET = Path("data/eval/vehicle_book_answer_eval.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/eval")
DEFAULT_RAW_RESULTS = DEFAULT_OUTPUT_DIR / "answer_raw_results.json"
DEFAULT_TOP_K = 4

RAW_RESULTS_JSON = "answer_raw_results.json"
CASE_RESULTS_JSON = "answer_case_results.json"
CASE_RESULTS_CSV = "answer_case_results.csv"
SUMMARY_JSON = "answer_summary.json"

CASE_FIELDS = (
    "case_id",
    "vehicle_id",
    "question",
    "category",
    "critical",
    "expected_fallback",
    "answer",
    "is_fallback",
    "fallback_reason",
    "required_citation_chunk_ids",
    "allowed_citation_chunk_ids",
    "cited_chunk_ids",
    "citation_vehicle_ids",
    "citation_document_versions",
    "answer_correct",
    "citation_correct",
    "fallback_correct",
    "grounded",
    "critical_error",
    "cross_vehicle_answer_error",
    "cross_version_answer_error",
    "missing_keyword_groups",
    "unsupported_keyword_groups",
    "matched_forbidden_phrases",
    "missing_required_citation_chunk_ids",
    "unexpected_citation_chunk_ids",
    "runtime_error",
)
LIST_FIELDS = frozenset(
    {
        "required_citation_chunk_ids",
        "allowed_citation_chunk_ids",
        "cited_chunk_ids",
        "citation_vehicle_ids",
        "citation_document_versions",
        "missing_keyword_groups",
        "unsupported_keyword_groups",
        "matched_forbidden_phrases",
        "missing_required_citation_chunk_ids",
        "unexpected_citation_chunk_ids",
    }
)


class RawResultsError(ValueError):
    """Raised when canonical saved answer results cannot be replayed."""


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Vehicle Book Answer Eval using live generation or offline replay."
    )
    parser.add_argument(
        "--mode",
        choices=("live", "replay"),
        default="replay",
        help="live calls the configured generator; replay is fully offline (default: replay)",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--raw-results",
        type=Path,
        default=DEFAULT_RAW_RESULTS,
        help="canonical raw AnswerResult JSON file for writing (live) or reading (replay)",
    )
    parser.add_argument("--top-k", type=_positive_int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="evaluate only this case ID; may be supplied more than once",
    )
    parser.add_argument(
        "--max-cases",
        type=_positive_int,
        help="evaluate only the first N selected cases, in dataset order",
    )
    return parser


def _require_mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RawResultsError(f"{description} must be a JSON object")
    return value


def _require_text(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RawResultsError(f"{description} must be a non-empty string")
    return value.strip()


def _require_number(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RawResultsError(f"{description} must be a number")
    return float(value)


def _raw_result_payload(result: AnswerResult) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "vehicle_id": result.vehicle_id,
        "fallback_reason": result.fallback_reason,
        "citations": [
            {
                "chunk_id": citation.chunk_id,
                "document_id": citation.document_id,
                "vehicle_id": citation.vehicle_id,
                "title": citation.title,
                "document_version": citation.document_version,
                "section": citation.section,
                "source_path": str(citation.source_path),
                "rank": citation.rank,
                "score": citation.score,
            }
            for citation in result.citations
        ],
        "evidence": [
            {
                "chunk_id": item.chunk.chunk_id,
                "section": item.chunk.section,
                "text": item.chunk.text,
                "metadata": {
                    "document_id": item.chunk.metadata.document_id,
                    "vehicle_id": item.chunk.metadata.vehicle_id,
                    "title": item.chunk.metadata.title,
                    "content_type": item.chunk.metadata.content_type,
                    "document_version": item.chunk.metadata.document_version,
                    "source_path": str(item.chunk.metadata.source_path),
                    "status": item.chunk.metadata.status,
                    "locale": item.chunk.metadata.locale,
                    "supersedes": item.chunk.metadata.supersedes,
                },
                "score": item.score,
                "rank": item.rank,
            }
            for item in result.evidence
        ],
    }


def _citation_from_payload(value: object, index: int) -> Citation:
    item = _require_mapping(value, f"citations[{index}]")
    return Citation(
        chunk_id=_require_text(item.get("chunk_id"), f"citations[{index}].chunk_id"),
        document_id=_require_text(item.get("document_id"), f"citations[{index}].document_id"),
        vehicle_id=_require_text(item.get("vehicle_id"), f"citations[{index}].vehicle_id"),
        title=_require_text(item.get("title"), f"citations[{index}].title"),
        document_version=_require_text(
            item.get("document_version"), f"citations[{index}].document_version"
        ),
        section=_require_text(item.get("section"), f"citations[{index}].section"),
        source_path=Path(_require_text(item.get("source_path"), f"citations[{index}].source_path")),
        rank=int(_require_number(item.get("rank"), f"citations[{index}].rank")),
        score=_require_number(item.get("score"), f"citations[{index}].score"),
    )


def _evidence_from_payload(value: object, index: int) -> RetrievedChunk:
    item = _require_mapping(value, f"evidence[{index}]")
    metadata_data = dict(_require_mapping(item.get("metadata"), f"evidence[{index}].metadata"))
    try:
        metadata = DocumentMetadata.from_dict(metadata_data)
    except ValueError as exc:
        raise RawResultsError(f"invalid evidence[{index}].metadata: {exc}") from exc
    chunk = KnowledgeChunk(
        chunk_id=_require_text(item.get("chunk_id"), f"evidence[{index}].chunk_id"),
        section=_require_text(item.get("section"), f"evidence[{index}].section"),
        text=_require_text(item.get("text"), f"evidence[{index}].text"),
        metadata=metadata,
    )
    return RetrievedChunk(
        chunk=chunk,
        score=_require_number(item.get("score"), f"evidence[{index}].score"),
        rank=int(_require_number(item.get("rank"), f"evidence[{index}].rank")),
    )


def _answer_result_from_payload(value: object) -> AnswerResult:
    data = _require_mapping(value, "answer_result")
    citations_data = data.get("citations")
    evidence_data = data.get("evidence")
    if not isinstance(citations_data, list) or not isinstance(evidence_data, list):
        raise RawResultsError("answer_result.citations and answer_result.evidence must be arrays")
    fallback_reason = data.get("fallback_reason")
    if fallback_reason is not None and not isinstance(fallback_reason, str):
        raise RawResultsError("answer_result.fallback_reason must be a string or null")
    return AnswerResult(
        answer=_require_text(data.get("answer"), "answer_result.answer"),
        vehicle_id=_require_text(data.get("vehicle_id"), "answer_result.vehicle_id"),
        citations=tuple(_citation_from_payload(item, index) for index, item in enumerate(citations_data)),
        evidence=tuple(_evidence_from_payload(item, index) for index, item in enumerate(evidence_data)),
        fallback_reason=fallback_reason,
    )


def _runtime_result(vehicle_id: str) -> AnswerResult:
    """Keep a live runtime failure visible to the shared deterministic evaluator."""
    return AnswerResult(answer="", vehicle_id=vehicle_id, fallback_reason="runtime_error")


def _load_raw_results(
    path: Path, case_vehicle_ids: Mapping[str, str]
) -> tuple[dict[str, AnswerResult], dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RawResultsError(f"raw results file does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RawResultsError(f"cannot read valid JSON raw results from {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise RawResultsError("raw results must be a JSON array")

    results: dict[str, AnswerResult] = {}
    runtime_errors: dict[str, str] = {}
    for index, entry in enumerate(payload):
        data = _require_mapping(entry, f"raw result[{index}]")
        case_id = _require_text(data.get("case_id"), f"raw result[{index}].case_id")
        if case_id in results:
            raise RawResultsError(f"duplicate raw result case_id: {case_id}")
        if case_id not in case_vehicle_ids:
            raise RawResultsError(f"raw result references unknown dataset case_id: {case_id}")
        runtime_error = data.get("runtime_error")
        answer_payload = data.get("answer_result")
        if runtime_error is not None:
            if not isinstance(runtime_error, str) or not runtime_error.strip() or answer_payload is not None:
                raise RawResultsError(
                    f"raw result {case_id} must have either answer_result or runtime_error"
                )
            results[case_id] = _runtime_result(case_vehicle_ids[case_id])
            runtime_errors[case_id] = runtime_error.strip()
        elif answer_payload is None:
            raise RawResultsError(f"raw result {case_id} is missing answer_result")
        else:
            results[case_id] = _answer_result_from_payload(answer_payload)
    return results, runtime_errors


def _select_cases(cases: Sequence[Any], case_ids: Sequence[str], max_cases: int | None) -> list[Any]:
    requested = set(case_ids)
    known_ids = {case.case_id for case in cases}
    unknown_ids = requested - known_ids
    if unknown_ids:
        raise AnswerEvaluationError("unknown case_id: " + ", ".join(sorted(unknown_ids)))
    selected = [case for case in cases if not requested or case.case_id in requested]
    return selected[:max_cases] if max_cases is not None else selected


def _case_row(result: AnswerCaseResult, runtime_error: str | None) -> dict[str, Any]:
    answer_result = result.answer_result
    return {
        "case_id": result.case.case_id,
        "vehicle_id": result.case.vehicle_id,
        "question": result.case.question,
        "category": result.case.category,
        "critical": result.case.critical,
        "expected_fallback": result.case.expected_fallback,
        "answer": answer_result.answer,
        "is_fallback": answer_result.is_fallback,
        "fallback_reason": answer_result.fallback_reason,
        "required_citation_chunk_ids": list(result.case.required_citation_chunk_ids),
        "allowed_citation_chunk_ids": list(result.case.allowed_citation_chunk_ids),
        "cited_chunk_ids": [citation.chunk_id for citation in answer_result.citations],
        "citation_vehicle_ids": [citation.vehicle_id for citation in answer_result.citations],
        "citation_document_versions": [
            citation.document_version for citation in answer_result.citations
        ],
        "answer_correct": result.answer_correct,
        "citation_correct": result.citation_correct,
        "fallback_correct": result.fallback_correct,
        "grounded": result.grounded,
        "critical_error": result.critical_error,
        "cross_vehicle_answer_error": result.cross_vehicle_answer_error,
        "cross_version_answer_error": result.cross_version_answer_error,
        "missing_keyword_groups": [list(group) for group in result.missing_keyword_groups],
        "unsupported_keyword_groups": [list(group) for group in result.unsupported_keyword_groups],
        "matched_forbidden_phrases": list(result.matched_forbidden_phrases),
        "missing_required_citation_chunk_ids": list(
            result.missing_required_citation_chunk_ids
        ),
        "unexpected_citation_chunk_ids": list(result.unexpected_citation_chunk_ids),
        "runtime_error": runtime_error,
    }


def _summary_payload(
    summary: AnswerEvalSummary,
    *,
    dataset: Path,
    raw_results: Path,
    mode: str,
    top_k: int,
    runtime_error_case_ids: Sequence[str],
) -> dict[str, Any]:
    def category_payload(category_summary: AnswerEvalSummary) -> dict[str, Any]:
        return {
            "total_cases": category_summary.total_cases,
            "positive_cases": category_summary.positive_cases,
            "fallback_cases": category_summary.fallback_cases,
            "critical_cases": category_summary.critical_cases,
            "cited_cases": category_summary.cited_cases,
            "answer_accuracy": category_summary.answer_accuracy,
            "citation_accuracy": category_summary.citation_accuracy,
            "fallback_accuracy": category_summary.fallback_accuracy,
            "grounded_answer_rate": category_summary.grounded_answer_rate,
            "critical_error_rate": category_summary.critical_error_rate,
            "cross_vehicle_answer_error_rate": (
                category_summary.cross_vehicle_answer_error_rate
            ),
            "cross_version_answer_error_rate": (
                category_summary.cross_version_answer_error_rate
            ),
            "cross_vehicle_answer_error_cases": (
                category_summary.cross_vehicle_answer_error_cases
            ),
            "cross_version_answer_error_cases": (
                category_summary.cross_version_answer_error_cases
            ),
            "failed_case_ids": list(category_summary.failed_case_ids),
        }

    return {
        "dataset_path": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "raw_results_path": str(raw_results),
        "mode": mode,
        "retrieval_mode": (
            os.getenv("CHESHU_RETRIEVAL_MODE", "bm25") if mode == "live" else None
        ),
        "top_k": top_k,
        "total_cases": summary.total_cases,
        "positive_cases": summary.positive_cases,
        "fallback_cases": summary.fallback_cases,
        "critical_cases": summary.critical_cases,
        "cited_cases": summary.cited_cases,
        "answer_accuracy": summary.answer_accuracy,
        "citation_accuracy": summary.citation_accuracy,
        "fallback_accuracy": summary.fallback_accuracy,
        "grounded_answer_rate": summary.grounded_answer_rate,
        "critical_error_rate": summary.critical_error_rate,
        "cross_vehicle_answer_error_rate": summary.cross_vehicle_answer_error_rate,
        "cross_version_answer_error_rate": summary.cross_version_answer_error_rate,
        "cross_vehicle_answer_error_cases": summary.cross_vehicle_answer_error_cases,
        "cross_version_answer_error_cases": summary.cross_version_answer_error_cases,
        "failed_case_ids": list(summary.failed_case_ids),
        "runtime_error_case_ids": list(runtime_error_case_ids),
        "category_metrics": {
            category: category_payload(category_summary)
            for category, category_summary in summary.category_metrics.items()
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _csv_value(field: str, value: Any) -> Any:
    if field in LIST_FIELDS:
        if value and isinstance(value[0], list):
            return "|".join("/".join(item) for item in value)
        return "|".join(str(item) for item in value)
    return "" if value is None else value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=CASE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(field, row[field]) for field in CASE_FIELDS})


def _format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def _print_summary(summary: AnswerEvalSummary, runtime_error_case_ids: Sequence[str]) -> None:
    print(f"Total Cases: {summary.total_cases}")
    print(f"Positive Cases: {summary.positive_cases}")
    print(f"Fallback Cases: {summary.fallback_cases}")
    print(f"Critical Cases: {summary.critical_cases}")
    print(f"Answer Accuracy: {_format_rate(summary.answer_accuracy)}")
    print(f"Citation Accuracy: {_format_rate(summary.citation_accuracy)}")
    print(f"Fallback Accuracy: {_format_rate(summary.fallback_accuracy)}")
    print(f"Grounded Answer Rate: {_format_rate(summary.grounded_answer_rate)}")
    print(f"Critical Error Rate: {_format_rate(summary.critical_error_rate)}")
    print("Cross-Vehicle Answer Error Rate: " + _format_rate(summary.cross_vehicle_answer_error_rate))
    print("Cross-Version Answer Error Rate: " + _format_rate(summary.cross_version_answer_error_rate))
    print(f"Failed Case IDs: {' | '.join(summary.failed_case_ids) or 'None'}")
    print(f"Runtime Error Case IDs: {' | '.join(runtime_error_case_ids) or 'None'}")


def _run_live(cases: Sequence[Any], top_k: int) -> tuple[dict[str, AnswerResult], dict[str, str], list[dict[str, Any]]]:
    generator = DeepSeekAnswerGenerator.from_env()
    service = VehicleBookService(answerer=GroundedAnswerer(generator))
    results: dict[str, AnswerResult] = {}
    runtime_errors: dict[str, str] = {}
    raw_rows: list[dict[str, Any]] = []
    for case in cases:
        try:
            result = service.answer(
                case.question,
                case.vehicle_id,
                top_k=top_k,
                include_pre_release=case.include_pre_release,
            )
        except Exception as exc:  # noqa: BLE001 - per-case live errors must not abort evaluation.
            message = f"{type(exc).__name__}: {exc}"
            results[case.case_id] = _runtime_result(case.vehicle_id)
            runtime_errors[case.case_id] = message
            raw_rows.append(
                {"case_id": case.case_id, "answer_result": None, "runtime_error": message}
            )
        else:
            results[case.case_id] = result
            raw_rows.append(
                {
                    "case_id": case.case_id,
                    "answer_result": _raw_result_payload(result),
                    "runtime_error": None,
                }
            )
    return results, runtime_errors, raw_rows


def _run(args: argparse.Namespace) -> None:
    cases = load_answer_eval_cases(args.dataset)
    selected = _select_cases(cases, args.case_id, args.max_cases)
    if not selected:
        raise AnswerEvaluationError("no evaluation cases selected")
    vehicle_ids = {case.case_id: case.vehicle_id for case in cases}

    if args.mode == "live":
        results, runtime_errors, raw_rows = _run_live(selected, args.top_k)
    else:
        replay_results, runtime_errors = _load_raw_results(args.raw_results, vehicle_ids)
        missing = [case.case_id for case in selected if case.case_id not in replay_results]
        if missing:
            raise RawResultsError("raw results missing selected case_id: " + ", ".join(missing))
        results = {case.case_id: replay_results[case.case_id] for case in selected}
        runtime_errors = {
            case.case_id: runtime_errors[case.case_id]
            for case in selected
            if case.case_id in runtime_errors
        }
        raw_rows = []

    summary = evaluate_answer_results(selected, results)
    runtime_error_case_ids = [case.case_id for case in selected if case.case_id in runtime_errors]
    rows = [_case_row(result, runtime_errors.get(result.case.case_id)) for result in summary.results]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "live":
        _write_json(args.raw_results, raw_rows)
    elif args.raw_results.resolve() != (args.output_dir / RAW_RESULTS_JSON).resolve():
        # Keep replay's canonical source immutable, while still materializing the requested report set.
        _write_json(args.output_dir / RAW_RESULTS_JSON, json.loads(args.raw_results.read_text(encoding="utf-8")))
    _write_json(args.output_dir / CASE_RESULTS_JSON, rows)
    _write_csv(args.output_dir / CASE_RESULTS_CSV, rows)
    _write_json(
        args.output_dir / SUMMARY_JSON,
        _summary_payload(
            summary,
            dataset=args.dataset,
            raw_results=args.raw_results,
            mode=args.mode,
            top_k=args.top_k,
            runtime_error_case_ids=runtime_error_case_ids,
        ),
    )
    _print_summary(summary, runtime_error_case_ids)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        _run(args)
    except (OSError, RawResultsError, AnswerEvaluationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI boundary should remain concise for users.
        print(f"error: answer evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
