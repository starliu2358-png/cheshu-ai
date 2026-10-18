from pathlib import Path

import pytest

from cockpit_agent.vehicle_book.registry import RegistryError
from cockpit_agent.vehicle_book.retriever import VehicleBookRetriever
from cockpit_agent.vehicle_book.schemas import DocumentMetadata, RetrievedChunk


def _retrieve_all(
    vehicle_id: str, *, include_pre_release: bool = False
) -> tuple[RetrievedChunk, ...]:
    return VehicleBookRetriever().retrieve(
        vehicle_id,
        "HVAC battery charging seat LCC ACC AEB HPA warning",
        top_k=100,
        include_pre_release=include_pre_release,
    )


def test_aster_x1_v1_never_returns_v2_chunks() -> None:
    results = _retrieve_all("alpha_aster_x1_max_v1")

    assert results
    assert {result.chunk.metadata.vehicle_id for result in results} == {
        "alpha_aster_x1_max_v1"
    }
    assert all("aster_x1_v1" in result.chunk.metadata.document_id for result in results)
    assert all("aster_x1_v2" not in result.chunk.metadata.document_id for result in results)
    assert all("/aster_x1/v1/" in result.chunk.metadata.source_path.as_posix() for result in results)


def test_aster_x1_v2_never_returns_v1_chunks_when_enabled() -> None:
    results = _retrieve_all(
        "alpha_aster_x1_max_v2",
        include_pre_release=True,
    )

    assert results
    assert {result.chunk.metadata.vehicle_id for result in results} == {
        "alpha_aster_x1_max_v2"
    }
    assert all("aster_x1_v2" in result.chunk.metadata.document_id for result in results)
    assert all("aster_x1_v1" not in result.chunk.metadata.document_id for result in results)
    assert all("/aster_x1/v2/" in result.chunk.metadata.source_path.as_posix() for result in results)


def test_aster_x1_never_returns_aster_x2_chunks() -> None:
    results = _retrieve_all("alpha_aster_x1_max_v1")

    assert results
    assert all("aster_x2" not in result.chunk.metadata.document_id for result in results)
    assert all("/aster_x2/" not in result.chunk.metadata.source_path.as_posix() for result in results)


def test_pre_release_is_denied_before_ingestion_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = VehicleBookRetriever()
    ingestion_calls: list[str] = []

    def record_ingestion(vehicle_id: str) -> tuple[()]:
        ingestion_calls.append(vehicle_id)
        return ()

    monkeypatch.setattr(retriever.ingestor, "ingest", record_ingestion)

    with pytest.raises(RegistryError, match="include_pre_release=True"):
        retriever.retrieve("alpha_aster_x1_max_v2", "HPA")

    assert ingestion_calls == []


def test_unknown_vehicle_id_fails_clearly() -> None:
    with pytest.raises(
        RegistryError,
        match="Unknown vehicle_id 'unknown_vehicle'",
    ):
        VehicleBookRetriever().retrieve("unknown_vehicle", "ACC")


def test_results_expose_rank_score_section_metadata_and_source_path() -> None:
    results = VehicleBookRetriever().retrieve(
        "alpha_aster_x2_max_v1",
        "maximum DC battery charging power",
        top_k=3,
    )

    assert results
    for expected_rank, result in enumerate(results, start=1):
        assert isinstance(result, RetrievedChunk)
        assert result.rank == expected_rank
        assert isinstance(result.score, float)
        assert result.score > 0
        assert result.chunk.section
        assert isinstance(result.chunk.metadata, DocumentMetadata)
        assert result.chunk.metadata.vehicle_id == "alpha_aster_x2_max_v1"
        assert isinstance(result.chunk.metadata.source_path, Path)
        assert not result.chunk.metadata.source_path.is_absolute()


def test_repeated_retrieval_has_identical_order_and_chunk_ids() -> None:
    retriever = VehicleBookRetriever()

    first = retriever.retrieve(
        "alpha_aster_x1_max_v1",
        "battery charging HVAC seat",
        top_k=8,
    )
    second = retriever.retrieve(
        "alpha_aster_x1_max_v1",
        "battery charging HVAC seat",
        top_k=8,
    )

    assert [result.chunk.chunk_id for result in first] == [
        result.chunk.chunk_id for result in second
    ]
    assert [result.rank for result in first] == [result.rank for result in second]
    assert [result.score for result in first] == [result.score for result in second]
