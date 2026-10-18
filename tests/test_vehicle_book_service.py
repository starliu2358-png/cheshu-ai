from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from cockpit_agent.vehicle_book.answerer import (
    AnswerDraft,
    GroundedAnswerer,
)
from cockpit_agent.vehicle_book.registry import RegistryError, VehicleRegistry
from cockpit_agent.vehicle_book.schemas import (
    AnswerResult,
    DocumentMetadata,
    KnowledgeChunk,
    RetrievedChunk,
)
from cockpit_agent.vehicle_book.service import FALLBACK_ANSWER, VehicleBookService


@dataclass
class FakeRetriever:
    evidence: tuple[RetrievedChunk, ...]
    calls: list[tuple[str, str, int, bool]] = field(default_factory=list)

    def retrieve(
        self,
        vehicle_id: str,
        query: str,
        top_k: int = 2,
        *,
        include_pre_release: bool = False,
    ) -> tuple[RetrievedChunk, ...]:
        self.calls.append((vehicle_id, query, top_k, include_pre_release))
        return self.evidence


@dataclass
class FakeGenerator:
    response: str
    calls: int = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return self.response


@dataclass
class FakeDraftAnswerer:
    draft: AnswerDraft

    def generate_draft(
        self,
        vehicle_id: str,
        question: str,
        evidence: tuple[RetrievedChunk, ...],
    ) -> AnswerDraft:
        return self.draft


def _evidence() -> tuple[RetrievedChunk, ...]:
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
        text="The vehicle has a 75 kWh battery.",
        metadata=metadata,
    )
    return (RetrievedChunk(chunk=chunk, score=2.5, rank=1),)


def _model_response(
    *,
    cited_chunk_ids: list[str] | None = None,
    insufficient_evidence: bool = False,
) -> str:
    return json.dumps(
        {
            "answer": "The battery capacity is 75 kWh.",
            "cited_chunk_ids": (
                cited_chunk_ids
                if cited_chunk_ids is not None
                else [_evidence()[0].chunk.chunk_id]
            ),
            "insufficient_evidence": insufficient_evidence,
        }
    )


def _service(
    retriever: FakeRetriever,
    generator: FakeGenerator,
) -> VehicleBookService:
    return VehicleBookService(
        vehicle_registry=VehicleRegistry(),
        retriever=retriever,
        answerer=GroundedAnswerer(generator),
    )


def _real_retrieval_service(response: str) -> VehicleBookService:
    return VehicleBookService(
        vehicle_registry=VehicleRegistry(),
        answerer=GroundedAnswerer(FakeGenerator(response)),
    )


def _assert_citation_matches_real_evidence(
    result: AnswerResult,
    expected_chunk_id: str,
) -> None:
    citations = result.citations
    evidence = result.evidence
    assert len(citations) == 1
    matched = next(
        item for item in evidence if item.chunk.chunk_id == expected_chunk_id
    )
    citation = citations[0]
    assert citation.chunk_id == matched.chunk.chunk_id
    assert citation.document_id == matched.chunk.metadata.document_id
    assert citation.vehicle_id == matched.chunk.metadata.vehicle_id
    assert citation.title == matched.chunk.metadata.title
    assert citation.document_version == matched.chunk.metadata.document_version
    assert citation.section == matched.chunk.section
    assert citation.source_path == matched.chunk.metadata.source_path
    assert citation.rank == matched.rank
    assert citation.score == matched.score


def test_grounded_x1_v1_answer_has_only_real_v1_citation() -> None:
    chunk_id = "oem_alpha_aster_x1_v1_owner_manual:battery-and-charging"
    response = _model_response(cited_chunk_ids=[chunk_id])

    result = _real_retrieval_service(response).answer(
        "battery charging capacity",
        "alpha_aster_x1_max_v1",
        top_k=10,
    )

    assert result.is_fallback is False
    assert result.vehicle_id == "alpha_aster_x1_max_v1"
    assert all(
        item.chunk.metadata.vehicle_id == "alpha_aster_x1_max_v1"
        for item in result.evidence
    )
    _assert_citation_matches_real_evidence(result, chunk_id)


def test_grounded_x1_v2_answer_requires_and_accepts_pre_release_access() -> None:
    chunk_id = "oem_alpha_aster_x1_v2_adas:hpa"
    response = _model_response(cited_chunk_ids=[chunk_id])

    result = _real_retrieval_service(response).answer(
        "HPA memorized parking route",
        "alpha_aster_x1_max_v2",
        top_k=10,
        include_pre_release=True,
    )

    assert result.is_fallback is False
    assert result.vehicle_id == "alpha_aster_x1_max_v2"
    assert all(
        item.chunk.metadata.vehicle_id == "alpha_aster_x1_max_v2"
        for item in result.evidence
    )
    _assert_citation_matches_real_evidence(result, chunk_id)


def test_grounded_x2_charging_answer_has_only_real_x2_citation() -> None:
    chunk_id = "oem_alpha_aster_x2_v1_owner_manual:battery-and-charging"
    response = _model_response(cited_chunk_ids=[chunk_id])

    result = _real_retrieval_service(response).answer(
        "maximum DC battery charging power",
        "alpha_aster_x2_max_v1",
        top_k=10,
    )

    assert result.is_fallback is False
    assert result.vehicle_id == "alpha_aster_x2_max_v1"
    assert all(
        item.chunk.metadata.vehicle_id == "alpha_aster_x2_max_v1"
        for item in result.evidence
    )
    _assert_citation_matches_real_evidence(result, chunk_id)


def test_service_builds_citations_from_exact_retrieved_objects() -> None:
    evidence = _evidence()
    retriever = FakeRetriever(evidence)
    generator = FakeGenerator(_model_response())

    result = _service(retriever, generator).answer(
        "What is the battery capacity?",
        "alpha_aster_x1_max_v1",
        top_k=3,
    )

    assert result.is_fallback is False
    assert result.vehicle_id == "alpha_aster_x1_max_v1"
    assert result.evidence is evidence
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == evidence[0].chunk.chunk_id
    assert result.citations[0].source_path == evidence[0].chunk.metadata.source_path
    assert result.citations[0].rank == evidence[0].rank
    assert result.citations[0].score == evidence[0].score
    assert retriever.calls == [
        (
            "alpha_aster_x1_max_v1",
            "What is the battery capacity?",
            3,
            False,
        )
    ]


def test_service_omits_redundant_configuration_citation() -> None:
    detailed = _evidence()[0]
    configuration = RetrievedChunk(
        chunk=KnowledgeChunk(
            chunk_id="oem_alpha_aster_x1_v1_configuration:summary",
            section="Configuration",
            text="Battery capacity: 75 kWh.",
            metadata=DocumentMetadata(
                document_id="oem_alpha_aster_x1_v1_configuration",
                vehicle_id="alpha_aster_x1_max_v1",
                title="ASTER_X1 v1 Configuration",
                content_type="configuration",
                document_version="1.0",
                source_path=Path(
                    "data/knowledge/oem_alpha/aster_x1/v1/configuration.md"
                ),
                status="published",
                locale="zh-CN",
            ),
        ),
        score=2.0,
        rank=2,
    )
    result = _service(
        FakeRetriever((detailed, configuration)),
        FakeGenerator(
            _model_response(
                cited_chunk_ids=[
                    detailed.chunk.chunk_id,
                    configuration.chunk.chunk_id,
                ]
            )
        ),
    ).answer("What is the battery capacity?", "alpha_aster_x1_max_v1")

    assert [citation.chunk_id for citation in result.citations] == [detailed.chunk.chunk_id]


def test_no_results_returns_fallback_without_calling_generator() -> None:
    retriever = FakeRetriever(())
    generator = FakeGenerator(_model_response())

    result = _service(retriever, generator).answer(
        "Unknown feature?",
        "alpha_aster_x1_max_v1",
    )

    assert result.answer == FALLBACK_ANSWER
    assert result.fallback_reason == "no_retrieval_results"
    assert result.is_fallback is True
    assert generator.calls == 0


def test_model_declared_insufficient_evidence_returns_fallback() -> None:
    generator = FakeGenerator(
        _model_response(cited_chunk_ids=[], insufficient_evidence=True)
    )

    result = _service(FakeRetriever(_evidence()), generator).answer(
        "What color is the car?",
        "alpha_aster_x1_max_v1",
    )

    assert result.answer == FALLBACK_ANSWER
    assert result.fallback_reason == "insufficient_evidence"
    assert result.citations == ()


@pytest.mark.parametrize(
    ("response", "expected_reason"),
    [
        ("not json", "malformed_model_output"),
        (_model_response(cited_chunk_ids=[]), "missing_citation_ids"),
        (
            _model_response(cited_chunk_ids=["invented:chunk"]),
            "unknown_citation_ids",
        ),
    ],
)
def test_invalid_model_output_returns_deterministic_fallback(
    response: str,
    expected_reason: str,
) -> None:
    result = _service(
        FakeRetriever(_evidence()),
        FakeGenerator(response),
    ).answer(
        "What is the battery capacity?",
        "alpha_aster_x1_max_v1",
    )

    assert result.answer == "这次没能整理出可靠的回答，请稍后再试。"
    assert result.fallback_reason == expected_reason
    assert result.citations == ()
    assert result.evidence == _evidence()


def test_service_revalidates_ids_from_injected_answerer() -> None:
    answerer = FakeDraftAnswerer(
        AnswerDraft(
            answer="Invented answer",
            cited_chunk_ids=("invented:chunk",),
            insufficient_evidence=False,
        )
    )
    service = VehicleBookService(
        vehicle_registry=VehicleRegistry(),
        retriever=FakeRetriever(_evidence()),
        answerer=answerer,
    )

    result = service.answer(
        "What is the battery capacity?",
        "alpha_aster_x1_max_v1",
    )

    assert result.is_fallback is True
    assert result.fallback_reason == "unknown_citation_ids"


def test_registry_errors_are_not_converted_to_fallbacks() -> None:
    generator = FakeGenerator(_model_response())
    service = _service(FakeRetriever(_evidence()), generator)

    with pytest.raises(RegistryError, match="Unknown vehicle_id 'unknown'"):
        service.answer("Question", "unknown")

    with pytest.raises(RegistryError, match="include_pre_release=True"):
        service.answer("Question", "alpha_aster_x1_max_v2")

    assert generator.calls == 0
