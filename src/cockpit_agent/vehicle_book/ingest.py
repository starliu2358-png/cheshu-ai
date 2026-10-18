from __future__ import annotations

from pathlib import Path

from .chunker import chunk_markdown
from .registry import DocumentRegistry, RegistryError, VehicleRegistry
from .schemas import DocumentMetadata, KnowledgeChunk


class KnowledgeIngestor:
    """Load registered documents for one exact vehicle and chunk them in memory."""

    def __init__(
        self,
        vehicle_registry: VehicleRegistry | None = None,
        document_registry: DocumentRegistry | None = None,
    ) -> None:
        if document_registry is None:
            self.vehicle_registry = vehicle_registry or VehicleRegistry()
            self.document_registry = DocumentRegistry(
                vehicle_registry=self.vehicle_registry
            )
        else:
            self.document_registry = document_registry
            self.vehicle_registry = vehicle_registry or document_registry.vehicle_registry

        self.repository_root = self.document_registry.repository_root.resolve()

    def _resolve_source(self, document: DocumentMetadata) -> Path:
        source = (self.repository_root / document.source_path).resolve()
        try:
            source.relative_to(self.repository_root)
        except ValueError as exc:
            raise RegistryError(
                f"Document source points outside the repository root for "
                f"{document.document_id}: {document.source_path}"
            ) from exc
        return source

    def ingest(self, vehicle_id: str) -> tuple[KnowledgeChunk, ...]:
        """Load published chunks belonging only to the requested vehicle profile."""
        profile = self.vehicle_registry.resolve(vehicle_id)
        documents = self.document_registry.for_vehicle(
            profile.vehicle_id, status="published"
        )

        chunks: list[KnowledgeChunk] = []
        for document in documents:
            source = self._resolve_source(document)
            markdown = source.read_text(encoding="utf-8")
            chunks.extend(chunk_markdown(document, markdown))

        return tuple(chunks)
