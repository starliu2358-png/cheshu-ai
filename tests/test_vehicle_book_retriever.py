from pathlib import Path

import pytest

from cockpit_agent.vehicle_book.registry import RegistryError
from cockpit_agent.vehicle_book.retriever import VehicleBookRetriever
from cockpit_agent.vehicle_book.schemas import RetrievedChunk


@pytest.mark.parametrize(
    ("vehicle_id", "expected_document_prefix"),
    [
        ("alpha_aster_x1_max_v1", "oem_alpha_aster_x1_v1_"),
        ("alpha_aster_x2_max_v1", "oem_alpha_aster_x2_v1_"),
    ],
)
def test_released_profiles_retrieve_only_exact_vehicle_chunks(
    vehicle_id: str, expected_document_prefix: str
) -> None:
    results = VehicleBookRetriever().retrieve(
        vehicle_id, "battery charging HVAC LCC seat", top_k=20
    )

    assert results
    assert all(isinstance(result, RetrievedChunk) for result in results)
    assert all(result.chunk.metadata.vehicle_id == vehicle_id for result in results)
    assert all(
        result.chunk.metadata.document_id.startswith(expected_document_prefix)
        for result in results
    )
    assert all(result.chunk.metadata.status == "published" for result in results)
    assert all(
        isinstance(result.chunk.metadata.source_path, Path)
        and not result.chunk.metadata.source_path.is_absolute()
        for result in results
    )
    assert [result.rank for result in results] == list(range(1, len(results) + 1))
    assert all(result.score > 0 for result in results)


def test_pre_release_profile_requires_explicit_opt_in() -> None:
    retriever = VehicleBookRetriever()

    with pytest.raises(RegistryError, match="include_pre_release=True"):
        retriever.retrieve("alpha_aster_x1_max_v2", "HPA")

    results = retriever.retrieve(
        "alpha_aster_x1_max_v2",
        "HPA memorized route",
        include_pre_release=True,
    )

    assert results
    assert {
        result.chunk.metadata.vehicle_id for result in results
    } == {"alpha_aster_x1_max_v2"}
    assert all(
        result.chunk.metadata.document_id.startswith("oem_alpha_aster_x1_v2_")
        for result in results
    )


def test_unknown_vehicle_fails_with_registry_error() -> None:
    with pytest.raises(RegistryError, match="Unknown vehicle_id 'missing_vehicle'"):
        VehicleBookRetriever().retrieve("missing_vehicle", "ACC")


@pytest.mark.parametrize("top_k", [0, -1])
def test_top_k_must_be_positive(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        VehicleBookRetriever().retrieve(
            "alpha_aster_x1_max_v1", "ACC", top_k=top_k
        )


def test_blank_query_returns_no_results() -> None:
    assert VehicleBookRetriever().retrieve(
        "alpha_aster_x1_max_v1", "  \n\t"
    ) == ()


@pytest.mark.parametrize(
    ("vehicle_id", "question", "expected_chunk_id", "include_pre_release"),
    [
        (
            "alpha_aster_x1_max_v1",
            "ASTER_X1 V1 在 30km/h 时是否可以开启 LCC？",
            "oem_alpha_aster_x1_v1_adas:lane-centering-control",
            False,
        ),
        (
            "alpha_aster_x1_max_v1",
            "开启 LCC 对左右车道线有什么要求？",
            "oem_alpha_aster_x1_v1_adas:lane-centering-control",
            False,
        ),
        (
            "alpha_aster_x1_max_v1",
            "ASTER_X1 V1 是否支持 HPA 记忆泊车？",
            "oem_alpha_aster_x1_v1_adas:hpa",
            False,
        ),
        (
            "alpha_aster_x1_max_v2",
            "ASTER_X1 V2 在 30km/h 时是否可以开启 LCC？",
            "oem_alpha_aster_x1_v2_adas:lane-centering-control",
            True,
        ),
        (
            "alpha_aster_x1_max_v2",
            "ASTER_X1 V2 是否支持 HPA 记忆泊车？",
            "oem_alpha_aster_x1_v2_adas:hpa",
            True,
        ),
    ],
)
def test_exact_lcc_and_hpa_terms_retrieve_matching_section_in_top_three(
    vehicle_id: str,
    question: str,
    expected_chunk_id: str,
    include_pre_release: bool,
) -> None:
    results = VehicleBookRetriever().retrieve(
        vehicle_id,
        question,
        top_k=3,
        include_pre_release=include_pre_release,
    )

    assert expected_chunk_id in [result.chunk.chunk_id for result in results]


def test_repeated_query_tokens_do_not_receive_repeated_bm25_weight() -> None:
    retriever = VehicleBookRetriever()

    single = retriever.retrieve("alpha_aster_x1_max_v1", "HPA", top_k=10)
    repeated = retriever.retrieve(
        "alpha_aster_x1_max_v1",
        "HPA HPA HPA HPA HPA",
        top_k=10,
    )

    assert [result.chunk.chunk_id for result in repeated] == [
        result.chunk.chunk_id for result in single
    ]
    assert [result.score for result in repeated] == [result.score for result in single]


@pytest.mark.parametrize(
    ("vehicle_id", "question", "include_pre_release"),
    [
        ("alpha_aster_x1_max_v1", "飞行模式应该如何开启？", False),
        ("alpha_aster_x1_max_v2", "如何让车辆自动驾驶去火星？", True),
        ("alpha_aster_x2_max_v1", "水下驾驶模式应该如何启动？", False),
    ],
)
def test_unknown_questions_return_no_evidence(
    vehicle_id: str,
    question: str,
    include_pre_release: bool,
) -> None:
    assert VehicleBookRetriever().retrieve(
        vehicle_id,
        question,
        top_k=3,
        include_pre_release=include_pre_release,
    ) == ()
