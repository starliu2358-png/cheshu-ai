from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from cockpit_agent.vehicle_book import (
    AnswerResult,
    Citation,
    DocumentMetadata,
    KnowledgeChunk,
    RetrievedChunk,
)


def _retrieved_chunk() -> RetrievedChunk:
    metadata = DocumentMetadata(
        document_id="oem_alpha_aster_x1_v1_owner_manual",
        vehicle_id="alpha_aster_x1_max_v1",
        title="ASTER_X1 v1 Owner Manual",
        content_type="owner_manual",
        document_version="1.0",
        source_path=Path(
            "data/knowledge/oem_alpha/aster_x1/v1/owner_manual.md"
        ),
        status="published",
        locale="zh-CN",
    )
    chunk = KnowledgeChunk(
        chunk_id="oem_alpha_aster_x1_v1_owner_manual:battery-and-charging",
        section="Battery and charging",
        text="The vehicle uses a 75 kWh battery.",
        metadata=metadata,
    )
    return RetrievedChunk(chunk=chunk, score=2.75, rank=1)


def test_citation_is_derived_from_retrieved_chunk_metadata() -> None:
    retrieved = _retrieved_chunk()

    citation = Citation.from_retrieved(retrieved)

    assert citation.chunk_id == retrieved.chunk.chunk_id
    assert citation.document_id == retrieved.chunk.metadata.document_id
    assert citation.vehicle_id == retrieved.chunk.metadata.vehicle_id
    assert citation.title == retrieved.chunk.metadata.title
    assert citation.document_version == retrieved.chunk.metadata.document_version
    assert citation.section == retrieved.chunk.section
    assert citation.source_path == retrieved.chunk.metadata.source_path
    assert citation.rank == retrieved.rank
    assert citation.score == retrieved.score


def test_citation_is_frozen() -> None:
    citation = Citation.from_retrieved(_retrieved_chunk())

    with pytest.raises(FrozenInstanceError):
        citation.rank = 2  # type: ignore[misc]


def test_answer_result_carries_citations_and_original_evidence() -> None:
    retrieved = _retrieved_chunk()
    citation = Citation.from_retrieved(retrieved)

    result = AnswerResult(
        answer="The battery capacity is 75 kWh.",
        vehicle_id="alpha_aster_x1_max_v1",
        citations=(citation,),
        evidence=(retrieved,),
    )

    assert result.citations == (citation,)
    assert result.evidence == (retrieved,)
    assert result.fallback_reason is None
    assert result.is_fallback is False


def test_answer_result_reports_fallback_from_reason() -> None:
    result = AnswerResult(
        answer="No supported answer is available.",
        vehicle_id="alpha_aster_x1_max_v1",
        fallback_reason="insufficient_evidence",
    )

    assert result.citations == ()
    assert result.evidence == ()
    assert result.is_fallback is True
