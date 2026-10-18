from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from cockpit_agent.vehicle_book.evaluation import (
    RetrievalEvalCase,
    RetrievalEvaluationError,
    evaluate_retrieval,
    load_retrieval_eval_cases,
)
from cockpit_agent.vehicle_book.ingest import KnowledgeIngestor
from cockpit_agent.vehicle_book.schemas import RetrievedChunk


def test_case_parser_is_strict_about_fields_and_boolean_types() -> None:
    payload = {
        "case_id": "case",
        "vehicle_id": "vehicle",
        "question": "question",
        "relevant_chunk_ids": ["chunk"],
        "include_pre_release": False,
        "category": "category",
        "critical": False,
        "expected_no_answer": False,
    }

    assert RetrievalEvalCase.from_dict(payload).relevant_chunk_ids == ("chunk",)

    with pytest.raises(RetrievalEvaluationError, match="unknown fields"):
        RetrievalEvalCase.from_dict({**payload, "extra": True})
    with pytest.raises(RetrievalEvaluationError, match="critical must be a boolean"):
        RetrievalEvalCase.from_dict({**payload, "critical": 0})


def test_loader_rejects_duplicate_case_ids_before_evaluation(tmp_path: Path) -> None:
    source = Path("data/eval/vehicle_book_retrieval_eval.jsonl").read_text(encoding="utf-8")
    first_line = source.splitlines()[0]
    path = tmp_path / "duplicate.jsonl"
    path.write_text(first_line + "\n" + first_line + "\n", encoding="utf-8")

    with pytest.raises(RetrievalEvaluationError, match="duplicate case_id"):
        load_retrieval_eval_cases(path)


def test_loader_rejects_gold_chunk_from_another_exact_profile(tmp_path: Path) -> None:
    payload = json.loads(
        Path("data/eval/vehicle_book_retrieval_eval.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    payload["relevant_chunk_ids"] = ["oem_alpha_aster_x1_v2_adas:lane-centering-control"]
    path = tmp_path / "wrong-profile.jsonl"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(RetrievalEvaluationError, match="exact vehicle profile"):
        load_retrieval_eval_cases(path)


class _FakeRetriever:
    def __init__(self, by_question: dict[str, tuple[RetrievedChunk, ...]]) -> None:
        self.by_question = by_question
        self.calls: list[tuple[str, str, int, bool]] = []

    def retrieve(
        self,
        vehicle_id: str,
        query: str,
        top_k: int = 2,
        *,
        include_pre_release: bool = False,
    ) -> tuple[RetrievedChunk, ...]:
        self.calls.append((vehicle_id, query, top_k, include_pre_release))
        return self.by_question.get(query, ())


def test_metrics_keep_no_answer_cases_out_of_positive_denominators() -> None:
    cases = load_retrieval_eval_cases()
    positive = cases[0]
    no_answer = next(case for case in cases if case.expected_no_answer)
    no_answer = replace(no_answer, critical=True)
    ingestor = KnowledgeIngestor()
    chunks = {
        vehicle_id: ingestor.ingest(vehicle_id)
        for vehicle_id in (
            "alpha_aster_x1_max_v1",
            "alpha_aster_x1_max_v2",
            "alpha_aster_x2_max_v1",
        )
    }
    relevant = next(
        chunk
        for chunk in chunks[positive.vehicle_id]
        if chunk.chunk_id == positive.relevant_chunk_ids[0]
    )
    returned = (
        RetrievedChunk(chunks["alpha_aster_x1_max_v2"][0], 3.0, 1),
        RetrievedChunk(relevant, 2.0, 2),
        RetrievedChunk(chunks["alpha_aster_x2_max_v1"][0], 1.0, 3),
    )
    retriever = _FakeRetriever({positive.question: returned})

    summary = evaluate_retrieval((positive, no_answer), retriever)

    assert summary.positive_case_count == 1
    assert summary.no_answer_case_count == 1
    assert summary.critical_positive_case_count == 1
    assert summary.hit_at_1 == 0.0
    assert summary.hit_at_3 == 1.0
    assert summary.critical_hit_at_3 == 1.0
    assert summary.recall_at_3 == 1.0
    assert summary.mrr == 0.5
    assert summary.cross_vehicle_error_rate == pytest.approx(1 / 3)
    assert summary.cross_version_error_rate == pytest.approx(1 / 3)
    assert summary.no_answer_false_positive_rate == 0.0
    assert summary.results[1].hit_at_3 is None
    assert len(retriever.calls) == 2


def test_cli_summary_json_contains_critical_hit_at_three(tmp_path: Path) -> None:
    output_dir = tmp_path / "eval"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_retrieval_eval.py",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(
        (output_dir / "retrieval_summary.json").read_text(encoding="utf-8")
    )

    assert summary["critical_positive_cases"] == 10
    assert summary["critical_hit_at_3"] == 1.0
