"""Combine V1.2 retrieval comparison and live answer evidence into a release gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DEFAULT_COMPARISON = Path("outputs/eval/retrieval_comparison.json")
DEFAULT_ANSWER = Path("outputs/eval/hybrid_answer_full/answer_summary.json")
DEFAULT_BASELINE = Path("data/baselines/vehicle_book_release_v1/answer_summary.json")
DEFAULT_OUTPUT = Path("outputs/eval/v12_gate.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--answer-summary", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-p95-ms", type=float, default=100.0)
    return parser


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        comparison = _read(args.comparison)
        answer = _read(args.answer_summary)
        baseline = _read(args.baseline_summary)
        hybrid = comparison["retrievers"]["hybrid_rrf"]
        checks = {
            "retrieval_gate": comparison["retrieval_gate"]["passed"] is True,
            "retrieval_datasets_unchanged": all(
                hashlib.sha256(Path(path).read_bytes()).hexdigest() == checksum
                for path, checksum in comparison["dataset_sha256"].items()
            ),
            "reranker_not_triggered": comparison["reranker"]["triggered"] is False,
            "hybrid_p95_under_target": hybrid["p95_latency_ms"] <= args.max_p95_ms,
            "answer_evaluation_is_live": answer["mode"] == "live",
            "answer_evaluation_uses_hybrid": answer["retrieval_mode"] == "hybrid",
            "answer_dataset_unchanged": (
                hashlib.sha256(Path(answer["dataset_path"]).read_bytes()).hexdigest()
                == answer["dataset_sha256"]
            ),
            "answer_case_count_preserved": answer["total_cases"] >= baseline["total_cases"],
            "answer_accuracy_preserved": (
                answer["answer_accuracy"] >= baseline["answer_accuracy"]
            ),
            "citation_accuracy_preserved": (
                answer["citation_accuracy"] >= baseline["citation_accuracy"]
            ),
            "fallback_accuracy_preserved": (
                answer["fallback_accuracy"] >= baseline["fallback_accuracy"]
            ),
            "no_cross_vehicle_answers": answer["cross_vehicle_answer_error_rate"] == 0,
            "no_cross_version_answers": answer["cross_version_answer_error_rate"] == 0,
            "no_failed_answer_cases": not answer["failed_case_ids"],
            "no_runtime_error_cases": not answer["runtime_error_case_ids"],
        }
        payload = {
            "decision": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "comparison_report": str(args.comparison),
            "answer_report": str(args.answer_summary),
            "baseline_report": str(args.baseline_summary),
            "max_p95_ms": args.max_p95_ms,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["decision"] == "PASS" else 1
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
