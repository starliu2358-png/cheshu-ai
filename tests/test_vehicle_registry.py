from pathlib import Path

import pytest

from cockpit_agent.vehicle_book import (
    DocumentRegistry,
    RegistryError,
    VehicleRegistry,
)


def test_known_vehicle_id_resolves_complete_profile() -> None:
    profile = VehicleRegistry().resolve("alpha_aster_x1_max_v1")

    assert profile.oem == "OEM_ALPHA"
    assert profile.model == "ASTER_X1"
    assert profile.trim == "MAX"
    assert profile.software_version == "1.0"
    assert profile.release_status == "released"


def test_unknown_vehicle_id_has_clear_error() -> None:
    with pytest.raises(RegistryError, match="Unknown vehicle_id 'missing_vehicle'"):
        VehicleRegistry().resolve("missing_vehicle")


def test_x1_versions_resolve_to_different_software_and_release_status() -> None:
    registry = VehicleRegistry()
    v1 = registry.resolve("alpha_aster_x1_max_v1")
    v2 = registry.resolve("alpha_aster_x1_max_v2")

    assert v1.software_version == "1.0"
    assert v2.software_version == "2.0"
    assert v1.release_status == "released"
    assert v2.release_status == "pre_release"


def test_pre_release_profiles_are_hidden_by_default() -> None:
    registry = VehicleRegistry()

    assert {item.vehicle_id for item in registry.list_profiles()} == {
        "alpha_aster_x1_max_v1",
        "alpha_aster_x2_max_v1",
        "model3-rwd-cn-current",
        "model3-lr-rwd-cn-current",
        "model3-lr-awd-cn-current",
        "model3-performance-awd-cn-current",
    }
    assert len(registry.list_profiles(include_pre_release=True)) == 7


def test_document_registry_links_three_existing_sources_per_vehicle() -> None:
    registry = DocumentRegistry()
    documents = registry.for_vehicle("alpha_aster_x1_max_v2")

    assert {item.content_type for item in documents} == {
        "owner_manual",
        "adas",
        "configuration",
    }
    assert all((registry.repository_root / item.source_path).is_file() for item in documents)


def test_document_registry_rejects_unknown_vehicle_reference(tmp_path: Path) -> None:
    registry_path = tmp_path / "documents.json"
    registry_path.write_text(
        """[
          {
            "document_id": "bad_document",
            "vehicle_id": "missing_vehicle",
            "title": "Bad document",
            "content_type": "owner_manual",
            "document_version": "1.0",
            "source_path": "data/knowledge/missing.md",
            "status": "published",
            "locale": "zh-CN"
          }
        ]""",
        encoding="utf-8",
    )

    with pytest.raises(RegistryError, match="Unknown vehicle_id 'missing_vehicle'"):
        DocumentRegistry(path=registry_path)
