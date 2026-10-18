import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from cockpit_agent.vehicle_book.answerer import (
    AnswerDraftError,
    GroundedAnswerer,
)
from cockpit_agent.vehicle_book.schemas import (
    DocumentMetadata,
    KnowledgeChunk,
    RetrievedChunk,
)


@dataclass
class FakeGenerator:
    response: str
    prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


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


def _other_profile_evidence() -> RetrievedChunk:
    metadata = DocumentMetadata(
        document_id="oem_alpha_aster_x2_v1_owner_manual",
        vehicle_id="alpha_aster_x2_max_v1",
        title="ASTER_X2 v1 Owner Manual",
        content_type="owner_manual",
        document_version="1.0",
        source_path=Path(
            "data/knowledge/oem_alpha/aster_x2/v1/owner_manual.md"
        ),
        status="published",
        locale="zh-CN",
    )
    chunk = KnowledgeChunk(
        chunk_id="oem_alpha_aster_x2_v1_owner_manual:battery-and-charging",
        section="Battery and charging",
        text="OTHER_PROFILE_SENTINEL: The vehicle has a 90 kWh battery.",
        metadata=metadata,
    )
    return RetrievedChunk(chunk=chunk, score=2.0, rank=1)


def _response(**overrides: object) -> str:
    payload: dict[str, object] = {
        "answer": "The battery capacity is 75 kWh.",
        "cited_chunk_ids": [
            "oem_alpha_aster_x1_v1_owner_manual:battery-and-charging"
        ],
        "insufficient_evidence": False,
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_grounded_answerer_returns_validated_json_draft() -> None:
    generator = FakeGenerator(_response())
    evidence = _evidence()
    other_profile = _other_profile_evidence()

    draft = GroundedAnswerer(generator).generate_draft(
        "alpha_aster_x1_max_v1",
        "What is the battery capacity?",
        evidence,
    )

    assert draft.answer == "The battery capacity is 75 kWh."
    assert draft.cited_chunk_ids == (
        "oem_alpha_aster_x1_v1_owner_manual:battery-and-charging",
    )
    assert draft.insufficient_evidence is False
    assert generator.prompt is not None
    assert 'Requested vehicle_id: alpha_aster_x1_max_v1' in generator.prompt
    assert evidence[0].chunk.chunk_id in generator.prompt
    assert "The vehicle has a 75 kWh battery." in generator.prompt
    assert other_profile.chunk.metadata.vehicle_id not in generator.prompt
    assert other_profile.chunk.chunk_id not in generator.prompt
    assert "OTHER_PROFILE_SENTINEL" not in generator.prompt
    assert "source_path" not in generator.prompt
    assert "Answer only what the question asks." in generator.prompt
    assert "Cover each attribute explicitly requested by the question." in generator.prompt
    assert "Treat the question as a strict scope boundary." in generator.prompt
    assert "Use the smallest sufficient set of citations" in generator.prompt


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        json.dumps([]),
        json.dumps({"answer": "unsupported shape"}),
        _response(extra="not allowed"),
        _response(answer=75),
        _response(cited_chunk_ids="chunk-id"),
        _response(cited_chunk_ids=[75]),
        _response(insufficient_evidence="false"),
    ],
)
def test_invalid_json_or_schema_raises_safe_error(response: str) -> None:
    with pytest.raises(AnswerDraftError):
        GroundedAnswerer(FakeGenerator(response)).generate_draft(
            "alpha_aster_x1_max_v1",
            "What is the battery capacity?",
            _evidence(),
        )


def test_unknown_citation_id_is_rejected() -> None:
    response = _response(cited_chunk_ids=["invented-document:invented-section"])

    with pytest.raises(AnswerDraftError, match="not supplied as evidence"):
        GroundedAnswerer(FakeGenerator(response)).generate_draft(
            "alpha_aster_x1_max_v1",
            "What is the battery capacity?",
            _evidence(),
        )


def test_factual_answer_requires_a_citation() -> None:
    with pytest.raises(AnswerDraftError, match="at least one citation"):
        GroundedAnswerer(
            FakeGenerator(_response(cited_chunk_ids=[]))
        ).generate_draft(
            "alpha_aster_x1_max_v1",
            "What is the battery capacity?",
            _evidence(),
        )


def test_insufficient_evidence_draft_must_not_claim_citations() -> None:
    response = _response(
        answer="The supplied evidence is insufficient.",
        cited_chunk_ids=[],
        insufficient_evidence=True,
    )

    draft = GroundedAnswerer(FakeGenerator(response)).generate_draft(
        "alpha_aster_x1_max_v1",
        "What color is the vehicle?",
        _evidence(),
    )

    assert draft.insufficient_evidence is True
    assert draft.cited_chunk_ids == ()


def test_cross_vehicle_evidence_is_rejected_before_generation() -> None:
    generator = FakeGenerator(_response())

    with pytest.raises(AnswerDraftError, match="different vehicle"):
        GroundedAnswerer(generator).generate_draft(
            "alpha_aster_x2_max_v1",
            "What is the battery capacity?",
            _evidence(),
        )

    assert generator.prompt is None
