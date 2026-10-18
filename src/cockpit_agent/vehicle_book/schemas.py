from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


def _required_text(data: dict[str, Any], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class VehicleProfile:
    vehicle_id: str
    oem: str
    model: str
    trim: str
    software_version: str | None
    release_status: str
    market: str | None = None
    display_name: str | None = None

    VALID_RELEASE_STATUSES: ClassVar[frozenset[str]] = frozenset({"released", "pre_release"})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VehicleProfile:
        software_version = data.get("software_version")
        if software_version is not None and (
            not isinstance(software_version, str) or not software_version.strip()
        ):
            raise ValueError("software_version must be a non-empty string when provided")
        market = data.get("market")
        if market is not None and (not isinstance(market, str) or not market.strip()):
            raise ValueError("market must be a non-empty string when provided")
        display_name = data.get("display_name")
        if display_name is not None and (
            not isinstance(display_name, str) or not display_name.strip()
        ):
            raise ValueError("display_name must be a non-empty string when provided")
        profile = cls(
            vehicle_id=_required_text(data, "vehicle_id"),
            oem=_required_text(data, "oem"),
            model=_required_text(data, "model"),
            trim=_required_text(data, "trim"),
            software_version=software_version.strip() if software_version else None,
            release_status=_required_text(data, "release_status"),
            market=market.strip() if market else None,
            display_name=display_name.strip() if display_name else None,
        )
        if profile.release_status not in cls.VALID_RELEASE_STATUSES:
            raise ValueError(
                "release_status must be one of: " + ", ".join(sorted(cls.VALID_RELEASE_STATUSES))
            )
        return profile


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    document_id: str
    vehicle_id: str
    title: str
    content_type: str
    document_version: str
    source_path: Path
    status: str
    locale: str = "zh-CN"
    supersedes: str | None = None
    source_url: str | None = None
    market: str | None = None
    retrieved_at: str | None = None

    VALID_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {"draft", "review", "published", "retired"}
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentMetadata:
        supersedes = data.get("supersedes")
        if supersedes is not None and (not isinstance(supersedes, str) or not supersedes.strip()):
            raise ValueError("supersedes must be a non-empty string when provided")

        source_url = data.get("source_url")
        if source_url is not None and (
            not isinstance(source_url, str)
            or not source_url.startswith(("https://", "http://"))
        ):
            raise ValueError("source_url must be an http(s) URL when provided")
        market = data.get("market")
        if market is not None and (not isinstance(market, str) or not market.strip()):
            raise ValueError("market must be a non-empty string when provided")
        retrieved_at = data.get("retrieved_at")
        if retrieved_at is not None and (
            not isinstance(retrieved_at, str) or not retrieved_at.strip()
        ):
            raise ValueError("retrieved_at must be a non-empty string when provided")

        metadata = cls(
            document_id=_required_text(data, "document_id"),
            vehicle_id=_required_text(data, "vehicle_id"),
            title=_required_text(data, "title"),
            content_type=_required_text(data, "content_type"),
            document_version=_required_text(data, "document_version"),
            source_path=Path(_required_text(data, "source_path")),
            status=_required_text(data, "status"),
            locale=_required_text(data, "locale"),
            supersedes=supersedes.strip() if supersedes else None,
            source_url=source_url,
            market=market.strip() if market else None,
            retrieved_at=retrieved_at.strip() if retrieved_at else None,
        )
        if metadata.status not in cls.VALID_STATUSES:
            raise ValueError("status must be one of: " + ", ".join(sorted(cls.VALID_STATUSES)))
        if metadata.source_path.is_absolute():
            raise ValueError("source_path must be relative to the repository root")
        return metadata


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    chunk_id: str
    section: str
    text: str
    metadata: DocumentMetadata


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: KnowledgeChunk
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class Citation:
    chunk_id: str
    document_id: str
    vehicle_id: str
    title: str
    document_version: str
    section: str
    source_path: Path
    rank: int
    score: float
    source_url: str | None = None
    market: str | None = None
    retrieved_at: str | None = None

    @classmethod
    def from_retrieved(cls, retrieved: RetrievedChunk) -> Citation:
        """Build citation provenance from an actual retrieval result."""
        chunk = retrieved.chunk
        metadata = chunk.metadata
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=metadata.document_id,
            vehicle_id=metadata.vehicle_id,
            title=metadata.title,
            document_version=metadata.document_version,
            section=chunk.section,
            source_path=metadata.source_path,
            rank=retrieved.rank,
            score=retrieved.score,
            source_url=metadata.source_url,
            market=metadata.market,
            retrieved_at=metadata.retrieved_at,
        )


@dataclass(frozen=True, slots=True)
class AnswerDraft:
    answer: str
    cited_chunk_ids: tuple[str, ...]
    insufficient_evidence: bool


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str
    vehicle_id: str
    citations: tuple[Citation, ...] = field(default_factory=tuple)
    evidence: tuple[RetrievedChunk, ...] = field(default_factory=tuple)
    fallback_reason: str | None = None
    boundary_result: str | None = None
    clarification_options: tuple[str, ...] = field(default_factory=tuple)
    answer_source: str = "official"

    @property
    def is_fallback(self) -> bool:
        return self.fallback_reason is not None
