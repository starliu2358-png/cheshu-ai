"""Run deterministic Vehicle Book retrieval evaluation and write reports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cockpit_agent.vehicle_book.evaluation import (
    RetrievalCaseResult,
    RetrievalEvalSummary,
    RetrievalEvaluationError,
    RetrievalEvaluator,
    load_retrieval_eval_cases,
)
from cockpit_agent.vehicle_book.retriever import VehicleBookRetriever
from cockpit_agent.vehicle_book.schemas import RetrievedChunk

DEFAULT_DATASET = Path("data/eval/vehicle_book_retrieval_eval.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/eval")
DEFAULT_TOP_K = 3

CASE_RESULTS_JSON = "retrieval_case_results.json"
CASE_RESULTS_CSV = "retrieval_case_results.csv"
SUMMARY_JSON = "retrieval_summary.json"

CASE_FIELDS = (
    "case_id",
    "vehicle_id",
    "question",
    "category",
    "critical",
    "expected_no_answer",
    "relevant_chunk_ids",
    "retrieved_chunk_ids",
    "retrieved_vehicle_ids",
    "retrieved_document_versions",
    "scores",
    "hit_at_1",
    "hit_at_3",
    "recall_at_3",
    "reciprocal_rank",
    "cross_vehicle_errors",
    "cross_version_errors",
    "no_answer_false_positive",
)

LIST_FIELDS = frozenset(
    {
        "relevant_chunk_ids",
        "retrieved_chunk_ids",
        "retrieved_vehicle_ids",
        "retrieved_document_versions",
        "scores",
    }
)


@dataclass(frozen=True, slots=True)
class _RetrievalCall:
    vehicle_id: str
    query: str
    results: tuple[RetrievedChunk, ...]


class _RecordingRetriever:
    """Delegate retrieval unchanged while retaining evidence for report details."""

    def __init__(self, retriever: VehicleBookRetriever) -> None:
        self._retriever = retriever
        self.calls: list[_RetrievalCall] = []

    def retrieve(
        self,
        vehicle_id: str,
        query: str,
        top_k: int = 2,
        *,
        include_pre_release: bool = False,
    ) -> Sequence[RetrievedChunk]:
        results = self._retriever.retrieve(
            vehicle_id,
            query,
            top_k=top_k,
            include_pre_release=include_pre_release,
        )
        self.calls.append(_RetrievalCall(vehicle_id, query, results))
        return results


def _evaluation_top_k(value: str) -> int:
    try:
        top_k = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top-k must be an integer") from exc
    if top_k <= 0:
        raise argparse.ArgumentTypeError("top-k must be greater than zero")
    if top_k < 3:
        raise argparse.ArgumentTypeError(
            "top-k must be at least 3 because this evaluation reports Hit@3 and Recall@3"
        )
    return top_k


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic, model-free Vehicle Book retrieval evaluation."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"UTF-8 JSONL evaluation dataset (default: {DEFAULT_DATASET.as_posix()})",
    )
    parser.add_argument(
        "--top-k",
        type=_evaluation_top_k,
        default=DEFAULT_TOP_K,
        help=f"number of retrieval results per case, minimum 3 (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"report output directory (default: {DEFAULT_OUTPUT_DIR.as_posix()})",
    )
    return parser


def _case_row(result: RetrievalCaseResult, call: _RetrievalCall) -> dict[str, Any]:
    if call.vehicle_id != result.case.vehicle_id or call.query != result.case.question:
        raise RuntimeError(f"retrieval record mismatch for case {result.case_id}")
    if tuple(item.chunk.chunk_id for item in call.results) != result.retrieved_chunk_ids:
        raise RuntimeError(f"retrieval result mismatch for case {result.case_id}")

    return {
        "case_id": result.case_id,
        "vehicle_id": result.case.vehicle_id,
        "question": result.case.question,
        "category": result.category,
        "critical": result.case.critical,
        "expected_no_answer": result.case.expected_no_answer,
        "relevant_chunk_ids": list(result.case.relevant_chunk_ids),
        "retrieved_chunk_ids": list(result.retrieved_chunk_ids),
        "retrieved_vehicle_ids": [
            item.chunk.metadata.vehicle_id for item in call.results
        ],
        "retrieved_document_versions": [
            item.chunk.metadata.document_version for item in call.results
        ],
        "scores": [item.score for item in call.results],
        "hit_at_1": result.hit_at_1,
        "hit_at_3": result.hit_at_3,
        "recall_at_3": result.recall_at_3,
        "reciprocal_rank": result.reciprocal_rank,
        "cross_vehicle_errors": result.cross_vehicle_errors,
        "cross_version_errors": result.cross_version_errors,
        "no_answer_false_positive": result.no_answer_false_positive,
    }


def _is_failed(result: RetrievalCaseResult) -> bool:
    isolation_failure = result.cross_vehicle_errors > 0 or result.cross_version_errors > 0
    if result.case.expected_no_answer:
        return isolation_failure or result.no_answer_false_positive is True
    return isolation_failure or result.hit_at_3 is not True


def _failed_case_ids(results: Sequence[RetrievalCaseResult]) -> list[str]:
    return [result.case_id for result in results if _is_failed(result)]


def _summary_payload(
    summary: RetrievalEvalSummary,
    dataset: Path,
    top_k: int,
    failed_case_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "dataset_path": str(dataset),
        "top_k": top_k,
        "total_cases": summary.case_count,
        "positive_cases": summary.positive_case_count,
        "no_answer_cases": summary.no_answer_case_count,
        "critical_positive_cases": summary.critical_positive_case_count,
        "hit_at_1": summary.hit_at_1,
        "hit_at_3": summary.hit_at_3,
        "critical_hit_at_3": summary.critical_hit_at_3,
        "recall_at_3": summary.recall_at_3,
        "mrr": summary.mrr,
        "cross_vehicle_error_rate": summary.cross_vehicle_error_rate,
        "cross_version_error_rate": summary.cross_version_error_rate,
        "no_answer_false_positive_rate": summary.no_answer_false_positive_rate,
        "failed_case_ids": list(failed_case_ids),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _csv_value(field_name: str, value: Any) -> Any:
    if field_name in LIST_FIELDS:
        return "|".join(str(item) for item in value)
    if value is None:
        return ""
    return value


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=CASE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field_name: _csv_value(field_name, row[field_name]) for field_name in CASE_FIELDS}
            )


def _format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def _print_summary(summary: RetrievalEvalSummary, failed_case_ids: Sequence[str]) -> None:
    print(f"Total Cases: {summary.case_count}")
    print(f"Positive Cases: {summary.positive_case_count}")
    print(f"No-answer Cases: {summary.no_answer_case_count}")
    print(f"Critical Positive Cases: {summary.critical_positive_case_count}")
    print(f"Hit@1: {_format_rate(summary.hit_at_1)}")
    print(f"Hit@3: {_format_rate(summary.hit_at_3)}")
    print(f"Critical Hit@3: {_format_rate(summary.critical_hit_at_3)}")
    print(f"Recall@3: {_format_rate(summary.recall_at_3)}")
    print(f"MRR: {_format_rate(summary.mrr)}")
    print(f"Cross-Vehicle Error Rate: {_format_rate(summary.cross_vehicle_error_rate)}")
    print(f"Cross-Version Error Rate: {_format_rate(summary.cross_version_error_rate)}")
    print(
        "No-Answer False Positive Rate: "
        f"{_format_rate(summary.no_answer_false_positive_rate)}"
    )
    print(f"Failed Case IDs: {' | '.join(failed_case_ids) if failed_case_ids else 'None'}")


def _run(dataset: Path, top_k: int, output_dir: Path) -> None:
    cases = load_retrieval_eval_cases(dataset)
    recording_retriever = _RecordingRetriever(VehicleBookRetriever())
    summary = RetrievalEvaluator(recording_retriever, top_k=top_k).evaluate(cases)

    if len(recording_retriever.calls) != len(summary.results):
        raise RuntimeError("retrieval call count does not match evaluation result count")
    rows = [
        _case_row(result, call)
        for result, call in zip(summary.results, recording_retriever.calls, strict=True)
    ]
    failed_case_ids = _failed_case_ids(summary.results)
    summary_payload = _summary_payload(summary, dataset, top_k, failed_case_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / CASE_RESULTS_JSON, rows)
    _write_csv(output_dir / CASE_RESULTS_CSV, rows)
    _write_json(output_dir / SUMMARY_JSON, summary_payload)
    _print_summary(summary, failed_case_ids)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        _run(args.dataset, args.top_k, args.output_dir)
    except (OSError, RetrievalEvaluationError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI boundary must not expose a traceback.
        print(f"error: evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
