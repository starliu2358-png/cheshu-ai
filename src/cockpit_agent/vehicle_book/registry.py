from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

from .schemas import DocumentMetadata, VehicleProfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VEHICLES_PATH = REPOSITORY_ROOT / "data" / "registry" / "vehicles.json"
DEFAULT_DOCUMENTS_PATH = REPOSITORY_ROOT / "data" / "registry" / "documents.json"


class RegistryError(ValueError):
    """Raised when registry data is invalid or cannot be resolved safely."""


def _load_records(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RegistryError(f"Registry file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RegistryError(f"Registry file is not valid JSON: {path}: {exc}") from exc

    if not isinstance(payload, list):
        raise RegistryError(f"Registry root must be a JSON array: {path}")
    if not all(isinstance(item, dict) for item in payload):
        raise RegistryError(f"Every registry entry must be a JSON object: {path}")
    return payload


T = TypeVar("T")


def _index_unique(records: Iterable[T], field_name: str) -> dict[str, T]:
    index: dict[str, T] = {}
    for record in records:
        key = getattr(record, field_name)
        if key in index:
            raise RegistryError(f"Duplicate {field_name}: {key}")
        index[key] = record
    return index


class VehicleRegistry:
    def __init__(self, path: Path = DEFAULT_VEHICLES_PATH) -> None:
        try:
            profiles = [VehicleProfile.from_dict(item) for item in _load_records(path)]
        except ValueError as exc:
            raise RegistryError(f"Invalid vehicle registry {path}: {exc}") from exc
        self.path = path
        self._profiles = _index_unique(profiles, "vehicle_id")

    def resolve(self, vehicle_id: str) -> VehicleProfile:
        try:
            return self._profiles[vehicle_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._profiles))
            raise RegistryError(
                f"Unknown vehicle_id '{vehicle_id}'. Available vehicle IDs: {available}"
            ) from exc

    def list_profiles(self, include_pre_release: bool = False) -> list[VehicleProfile]:
        return [
            profile
            for profile in self._profiles.values()
            if include_pre_release or profile.release_status == "released"
        ]


class DocumentRegistry:
    def __init__(
        self,
        path: Path = DEFAULT_DOCUMENTS_PATH,
        vehicle_registry: VehicleRegistry | None = None,
        repository_root: Path = REPOSITORY_ROOT,
    ) -> None:
        self.path = path
        self.repository_root = repository_root
        self.vehicle_registry = vehicle_registry or VehicleRegistry()
        try:
            documents = [DocumentMetadata.from_dict(item) for item in _load_records(path)]
        except ValueError as exc:
            raise RegistryError(f"Invalid document registry {path}: {exc}") from exc

        for document in documents:
            self.vehicle_registry.resolve(document.vehicle_id)
            source = self.repository_root / document.source_path
            if not source.is_file():
                raise RegistryError(
                    f"Document source does not exist for {document.document_id}: "
                    f"{document.source_path}"
                )
        self._documents = _index_unique(documents, "document_id")

    def resolve(self, document_id: str) -> DocumentMetadata:
        try:
            return self._documents[document_id]
        except KeyError as exc:
            raise RegistryError(f"Unknown document_id '{document_id}'") from exc

    def for_vehicle(
        self, vehicle_id: str, status: str | None = "published"
    ) -> list[DocumentMetadata]:
        self.vehicle_registry.resolve(vehicle_id)
        return [
            document
            for document in self._documents.values()
            if document.vehicle_id == vehicle_id
            and (status is None or document.status == status)
        ]
