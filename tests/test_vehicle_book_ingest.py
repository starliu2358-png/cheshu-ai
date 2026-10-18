import pytest

from cockpit_agent.vehicle_book.ingest import KnowledgeIngestor
from cockpit_agent.vehicle_book.registry import RegistryError


@pytest.mark.parametrize(
    ("vehicle_id", "expected_document_ids"),
    [
        (
            "alpha_aster_x1_max_v1",
            {
                "oem_alpha_aster_x1_v1_owner_manual",
                "oem_alpha_aster_x1_v1_adas",
                "oem_alpha_aster_x1_v1_configuration",
            },
        ),
        (
            "alpha_aster_x1_max_v2",
            {
                "oem_alpha_aster_x1_v2_owner_manual",
                "oem_alpha_aster_x1_v2_adas",
                "oem_alpha_aster_x1_v2_configuration",
            },
        ),
        (
            "alpha_aster_x2_max_v1",
            {
                "oem_alpha_aster_x2_v1_owner_manual",
                "oem_alpha_aster_x2_v1_adas",
                "oem_alpha_aster_x2_v1_configuration",
            },
        ),
    ],
)
def test_each_vehicle_ingests_only_its_three_registered_documents(
    vehicle_id: str, expected_document_ids: set[str]
) -> None:
    chunks = KnowledgeIngestor().ingest(vehicle_id)

    assert chunks
    assert {chunk.metadata.document_id for chunk in chunks} == expected_document_ids
    assert {chunk.metadata.vehicle_id for chunk in chunks} == {vehicle_id}
    assert {chunk.metadata.status for chunk in chunks} == {"published"}


def test_unknown_vehicle_uses_existing_registry_error() -> None:
    with pytest.raises(RegistryError, match="Unknown vehicle_id 'missing_vehicle'"):
        KnowledgeIngestor().ingest("missing_vehicle")
