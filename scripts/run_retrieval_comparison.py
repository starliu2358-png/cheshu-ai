"""Compare BM25, vector, and RRF hybrid retrieval on one evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cockpit_agent.vehicle_book.evaluation import (
    RetrievalEvalSummary,
    RetrievalEvaluationError,
    RetrievalEvaluator,
    load_retrieval_eval_cases,
)
from cockpit_agent.vehicle_book.hybrid_retriever import (
    HybridVehicleBookRetriever,
    VectorVehicleBookRetriever,
)
from cockpit_agent.vehicle_book.retriever import VehicleBookRetriever

DEFAULT_DATASET = Path("data/eval/vehicle_book_retrieval_eval.jsonl")
DEFAULT_SUPPLEMENT = Path("data/eval/vehicle_book_retrieval_v12_supplement.jsonl")
DEFAULT_OUTPUT = Path("outputs/eval/retrieval_comparison.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--supplementary-dataset",
        type=Path,
        default=DEFAULT_SUPPLEMENT,
        help="additional V1.2 synonym, abbreviation, follow-up, and no-answer cases",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def _metrics(
    summary: RetrievalEvalSummary,
    elapsed: float,
    cold_start_elapsed: float,
    latencies: Sequence[float],
) -> dict[str, object]:
    ordered_latencies = sorted(latencies)
    p95_index = max(0, math.ceil(0.95 * len(ordered_latencies)) - 1)
    return {
        "case_count": summary.case_count,
        "hit_at_1": summary.hit_at_1,
        "hit_at_3": summary.hit_at_3,
        "recall_at_3": summary.recall_at_3,
        "mrr": summary.mrr,
        "cross_vehicle_error_rate": summary.cross_vehicle_error_rate,
        "cross_version_error_rate": summary.cross_version_error_rate,
        "no_answer_false_positive_rate": summary.no_answer_false_positive_rate,
        "cold_start_latency_ms": round(cold_start_elapsed * 1000, 2),
        "total_latency_ms": round(elapsed * 1000, 2),
        "mean_latency_ms": round(elapsed * 1000 / summary.case_count, 2),
        "p95_latency_ms": round(ordered_latencies[p95_index] * 1000, 2),
        "missed_case_ids": [
            result.case_id
            for result in summary.results
            if result.case.expected_no_answer is False and result.hit_at_3 is False
        ],
        "false_positive_case_ids": [
            result.case_id
            for result in summary.results
            if result.no_answer_false_positive is True
        ],
    }


class TimedRetriever:
    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever
        self.latencies: list[float] = []

    def retrieve(self, vehicle_id, query, top_k=2, *, include_pre_release=False):
        started_at = time.perf_counter()
        try:
            return self.retriever.retrieve(
                vehicle_id,
                query,
                top_k=top_k,
                include_pre_release=include_pre_release,
            )
        finally:
            self.latencies.append(time.perf_counter() - started_at)


def _release_gate(results: dict[str, dict[str, object]]) -> dict[str, object]:
    baseline = results["bm25"]
    hybrid = results["hybrid_rrf"]
    checks = {
        "recall_at_3_not_below_bm25": hybrid["recall_at_3"] >= baseline["recall_at_3"],
        "hit_at_1_at_least_90_percent": hybrid["hit_at_1"] >= 0.90,
        "mrr_at_least_90_percent": hybrid["mrr"] >= 0.90,
        "cross_vehicle_errors_zero": hybrid["cross_vehicle_error_rate"] == 0,
        "cross_version_errors_zero": hybrid["cross_version_error_rate"] == 0,
        "no_answer_false_positives_not_above_bm25": (
            hybrid["no_answer_false_positive_rate"]
            <= baseline["no_answer_false_positive_rate"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.top_k < 3:
        print("error: --top-k must be at least 3", file=sys.stderr)
        return 1
    try:
        cases = load_retrieval_eval_cases(args.dataset) + load_retrieval_eval_cases(
            args.supplementary_dataset
        )

        def hybrid() -> HybridVehicleBookRetriever:
            lexical = VehicleBookRetriever()
            return HybridVehicleBookRetriever(
                lexical=lexical,
                vector=VectorVehicleBookRetriever(
                    vehicle_registry=lexical.vehicle_registry,
                    ingestor=lexical.ingestor,
                ),
            )

        retriever_factories = {
            "bm25": VehicleBookRetriever,
            "vector": VectorVehicleBookRetriever,
            "hybrid_rrf": hybrid,
        }
        payload: dict[str, Any] = {
            "dataset_path": str(args.dataset),
            "supplementary_dataset_path": str(args.supplementary_dataset),
            "dataset_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (args.dataset, args.supplementary_dataset)
            },
            "top_k": args.top_k,
            "embedding_model": "BAAI/bge-small-zh-v1.5",
            "retrievers": {},
        }
        results = payload["retrievers"]
        warmup_cases = tuple(dict.fromkeys(case.vehicle_id for case in cases))
        for name, retriever_factory in retriever_factories.items():
            retriever = retriever_factory()
            cold_started_at = time.perf_counter()
            for vehicle_id in warmup_cases:
                case = next(item for item in cases if item.vehicle_id == vehicle_id)
                retriever.retrieve(
                    case.vehicle_id,
                    case.question,
                    top_k=args.top_k,
                    include_pre_release=case.include_pre_release,
                )
            cold_start_elapsed = time.perf_counter() - cold_started_at
            timed = TimedRetriever(retriever)
            started_at = time.perf_counter()
            summary = RetrievalEvaluator(timed, top_k=args.top_k).evaluate(cases)
            results[name] = _metrics(
                summary,
                time.perf_counter() - started_at,
                cold_start_elapsed,
                timed.latencies,
            )
            print(f"{name}: {json.dumps(results[name], ensure_ascii=False)}")
        payload["retrieval_gate"] = _release_gate(results)
        reranker_triggered = (
            results["hybrid_rrf"]["hit_at_1"] < 0.90
            or results["hybrid_rrf"]["mrr"] < 0.90
        )
        payload["reranker"] = {
            "triggered": reranker_triggered,
            "status": "evaluation_required" if reranker_triggered else "not_enabled",
            "reason": (
                "Hybrid Hit@1 or MRR is below 90%."
                if reranker_triggered
                else "Hybrid Hit@1 and MRR exceed the predefined trigger."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not payload["retrieval_gate"]["passed"]:
            print("error: hybrid retrieval release gate failed", file=sys.stderr)
            return 1
    except (OSError, RetrievalEvaluationError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
