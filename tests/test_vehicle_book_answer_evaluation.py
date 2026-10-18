from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from cockpit_agent.vehicle_book.answer_evaluation import (
    AnswerEvalCase,
    AnswerEvaluationError,
    evaluate_answer_results,
    load_answer_eval_cases,
)
from cockpit_agent.vehicle_book.ingest import KnowledgeIngestor
from cockpit_agent.vehicle_book.schemas import AnswerResult, Citation, RetrievedChunk


def _chunk(vehicle_id: str, chunk_id: str) -> RetrievedChunk:
    chunk = next(
        chunk
        for chunk in KnowledgeIngestor().ingest(vehicle_id)
        if chunk.chunk_id == chunk_id
    )
    return RetrievedChunk(chunk=chunk, score=1.0, rank=1)


def test_loader_reads_the_strict_answer_eval_data_set() -> None:
    cases = load_answer_eval_cases()

    assert len(cases) == 30
    assert len({case.case_id for case in cases}) == 30
    assert all(
        case.include_pre_release
        for case in cases
        if case.vehicle_id == "alpha_aster_x1_max_v2"
    )


def test_loader_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    source = Path("data/eval/vehicle_book_answer_eval.jsonl").read_text(
        encoding="utf-8"
    )
    first = source.splitlines()[0]
    path = tmp_path / "duplicate.jsonl"
    path.write_text(first + "\n" + first + "\n", encoding="utf-8")

    with pytest.raises(AnswerEvaluationError, match="duplicate case_id"):
        load_answer_eval_cases(path)


def test_case_schema_is_strict_and_rejects_cross_profile_allowed_citations(
    tmp_path: Path,
) -> None:
    source = Path("data/eval/vehicle_book_answer_eval.jsonl").read_text(
        encoding="utf-8"
    )
    payload = json.loads(source.splitlines()[0])

    with pytest.raises(AnswerEvaluationError, match="unknown fields"):
        AnswerEvalCase.from_dict({**payload, "extra": True})
    with pytest.raises(AnswerEvaluationError, match="critical must be a boolean"):
        AnswerEvalCase.from_dict({**payload, "critical": 1})

    payload["required_citation_chunk_ids"] = ["oem_alpha_aster_x2_v1_adas:lcc"]
    payload["allowed_citation_chunk_ids"] = ["oem_alpha_aster_x2_v1_adas:lcc"]
    path = tmp_path / "wrong-profile-citation.jsonl"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(AnswerEvaluationError, match="exact vehicle profile"):
        load_answer_eval_cases(path)


def test_fake_answer_results_are_scored_without_a_model() -> None:
    cases = load_answer_eval_cases()
    positive = next(
        case for case in cases if case.case_id == "x1_v1_battery_capacity_answer_001"
    )
    fallback = next(case for case in cases if case.expected_fallback)
    evidence = _chunk(
        positive.vehicle_id,
        "oem_alpha_aster_x1_v1_owner_manual:battery-and-charging",
    )
    grounded = AnswerResult(
        answer="该车配备 75 kWh LFP 动力电池。",
        vehicle_id=positive.vehicle_id,
        citations=(Citation.from_retrieved(evidence),),
        evidence=(evidence,),
    )
    fallback_result = AnswerResult(
        answer="未在当前车辆资料中找到足够可靠的依据。",
        vehicle_id=fallback.vehicle_id,
        fallback_reason="no_retrieval_results",
    )

    summary = evaluate_answer_results(
        (positive, fallback),
        {positive.case_id: grounded, fallback.case_id: fallback_result},
    )

    positive_result, fallback_case_result = summary.results
    assert positive_result.answer_correct is True
    assert positive_result.citation_correct is True
    assert positive_result.grounded is True
    assert fallback_case_result.fallback_correct is True
    assert summary.answer_accuracy == 1.0
    assert summary.citation_accuracy == 1.0
    assert summary.grounded_answer_rate == 1.0
    assert summary.fallback_accuracy == 1.0


def test_forbidden_phrase_makes_a_correctly_cited_answer_ungrounded() -> None:
    source_case = next(
        case
        for case in load_answer_eval_cases()
        if case.case_id == "x1_v1_battery_capacity_answer_001"
    )
    case = replace(source_case, critical=True)
    evidence = _chunk(
        case.vehicle_id,
        "oem_alpha_aster_x1_v1_owner_manual:battery-and-charging",
    )
    result = AnswerResult(
        answer="该车配备 90 kWh NMC 动力电池。",
        vehicle_id=case.vehicle_id,
        citations=(Citation.from_retrieved(evidence),),
        evidence=(evidence,),
    )

    evaluated = evaluate_answer_results((case,), {case.case_id: result}).results[0]

    assert evaluated.citation_correct is True
    assert evaluated.answer_correct is False
    assert evaluated.matched_forbidden_phrases == ("90 kWh", "NMC")
    assert evaluated.grounded is False
    assert evaluated.critical_error is True


def test_cross_profile_rates_use_only_cases_with_citations() -> None:
    cases = load_answer_eval_cases()
    positive = next(
        case for case in cases if case.case_id == "x1_v1_battery_capacity_answer_001"
    )
    fallback = next(case for case in cases if case.expected_fallback)
    evidence = _chunk(
        positive.vehicle_id,
        "oem_alpha_aster_x1_v1_owner_manual:battery-and-charging",
    )
    correct_citation = Citation.from_retrieved(evidence)
    wrong_profile_citation = Citation(
        chunk_id=correct_citation.chunk_id,
        document_id=correct_citation.document_id,
        vehicle_id="alpha_aster_x2_max_v1",
        title=correct_citation.title,
        document_version="9.9",
        section=correct_citation.section,
        source_path=correct_citation.source_path,
        rank=correct_citation.rank,
        score=correct_citation.score,
    )
    bad_result = AnswerResult(
        answer="该车配备 75 kWh LFP 动力电池。",
        vehicle_id=positive.vehicle_id,
        citations=(wrong_profile_citation,),
        evidence=(evidence,),
    )
    fallback_result = AnswerResult(
        answer="未在当前车辆资料中找到足够可靠的依据。",
        vehicle_id=fallback.vehicle_id,
        fallback_reason="no_retrieval_results",
    )

    summary = evaluate_answer_results(
        (positive, fallback),
        {positive.case_id: bad_result, fallback.case_id: fallback_result},
    )

    assert summary.cited_cases == 1
    assert summary.total_cases == 2
    assert summary.cross_vehicle_answer_error_cases == 1
    assert summary.cross_version_answer_error_cases == 1
    assert summary.cross_vehicle_answer_error_rate == 1.0
    assert summary.cross_version_answer_error_rate == 1.0
    assert summary.results[0].critical_error is True


def test_grounded_scoring_does_not_use_uncited_evidence_as_support() -> None:
    source_case = next(
        case
        for case in load_answer_eval_cases()
        if case.case_id == "x1_v1_battery_capacity_answer_001"
    )
    cited_chunk = _chunk(
        source_case.vehicle_id,
        "oem_alpha_aster_x1_v1_owner_manual:hvac",
    )
    uncited_battery_chunk = _chunk(
        source_case.vehicle_id,
        "oem_alpha_aster_x1_v1_owner_manual:battery-and-charging",
    )
    case = replace(
        source_case,
        required_citation_chunk_ids=(cited_chunk.chunk.chunk_id,),
        allowed_citation_chunk_ids=(cited_chunk.chunk.chunk_id,),
    )
    result = AnswerResult(
        answer="该车配备 75 kWh LFP 动力电池。",
        vehicle_id=case.vehicle_id,
        citations=(Citation.from_retrieved(cited_chunk),),
        evidence=(cited_chunk, uncited_battery_chunk),
    )

    evaluated = evaluate_answer_results((case,), {case.case_id: result}).results[0]

    assert evaluated.answer_correct is True
    assert evaluated.citation_correct is True
    assert evaluated.grounded is False


def test_unsupported_feature_claimed_as_supported_is_a_critical_error() -> None:
    case = next(
        case
        for case in load_answer_eval_cases()
        if case.case_id == "x1_v1_hpa_unsupported_answer_001"
    )
    evidence = _chunk(case.vehicle_id, "oem_alpha_aster_x1_v1_adas:hpa")
    result = AnswerResult(
        answer="ASTER_X1 V1 的 HPA 支持记忆泊车。",
        vehicle_id=case.vehicle_id,
        citations=(Citation.from_retrieved(evidence),),
        evidence=(evidence,),
    )

    evaluated = evaluate_answer_results((case,), {case.case_id: result}).results[0]

    assert evaluated.answer_correct is False
    assert evaluated.citation_correct is True
    assert evaluated.grounded is False
    assert evaluated.critical_error is True


def test_fallback_case_citation_fails_without_affecting_positive_denominators() -> None:
    cases = load_answer_eval_cases()
    positive = next(
        case for case in cases if case.case_id == "x1_v1_battery_capacity_answer_001"
    )
    fallback = next(
        case
        for case in cases
        if case.case_id == "x1_v1_flight_mode_answer_001"
    )
    evidence = _chunk(
        positive.vehicle_id,
        "oem_alpha_aster_x1_v1_owner_manual:battery-and-charging",
    )
    positive_result = AnswerResult(
        answer="该车配备 75 kWh LFP 动力电池。",
        vehicle_id=positive.vehicle_id,
        citations=(Citation.from_retrieved(evidence),),
        evidence=(evidence,),
    )
    fallback_with_citation = AnswerResult(
        answer="未在当前车辆资料中找到足够可靠的依据。",
        vehicle_id=fallback.vehicle_id,
        citations=(Citation.from_retrieved(evidence),),
        evidence=(evidence,),
        fallback_reason="no_retrieval_results",
    )

    summary = evaluate_answer_results(
        (positive, fallback),
        {
            positive.case_id: positive_result,
            fallback.case_id: fallback_with_citation,
        },
    )

    fallback_result = summary.results[1]
    assert fallback_result.fallback_correct is True
    assert fallback_result.citation_correct is False
    assert fallback_result.failed is True
    assert summary.answer_accuracy == 1.0
    assert summary.citation_accuracy == 1.0
    assert summary.grounded_answer_rate == 1.0
    assert summary.fallback_accuracy == 1.0
    assert summary.category_metrics["no_answer"].fallback_accuracy == 1.0
    assert summary.category_metrics["no_answer"].citation_accuracy is None


def test_no_answer_fact_is_only_a_critical_error_when_the_case_is_critical() -> None:
    no_answer = next(case for case in load_answer_eval_cases() if case.expected_fallback)
    fact_answer = AnswerResult(
        answer="该功能可以使用。",
        vehicle_id=no_answer.vehicle_id,
    )

    ordinary = evaluate_answer_results(
        (no_answer,), {no_answer.case_id: fact_answer}
    ).results[0]
    critical_case = replace(no_answer, critical=True)
    critical = evaluate_answer_results(
        (critical_case,), {critical_case.case_id: fact_answer}
    ).results[0]

    assert ordinary.fallback_correct is False
    assert ordinary.critical_error is False
    assert critical.fallback_correct is False
    assert critical.critical_error is True
