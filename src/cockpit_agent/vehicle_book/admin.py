"""Read-only view models for the local Knowledge Admin demonstration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import REPOSITORY_ROOT, DocumentRegistry, VehicleRegistry

DEFAULT_RELEASE_GATE_REPORT_PATH = (
    REPOSITORY_ROOT / "outputs" / "release_gate" / "release_gate_summary.json"
)
_RELEASE_GATE_STATUSES = frozenset({"PASS", "FAIL", "BLOCKED"})


@dataclass(frozen=True, slots=True)
class AdminDocumentView:
    """Safe, display-only document metadata for one vehicle profile."""

    document_id: str
    title: str
    content_type: str
    document_version: str
    status: str
    locale: str
    source_path: str


@dataclass(frozen=True, slots=True)
class AdminProfileView:
    """Safe, display-only vehicle profile metadata and its applicable documents."""

    vehicle_id: str
    oem: str
    model: str
    trim: str
    software_version: str
    release_status: str
    documents: tuple[AdminDocumentView, ...]

    @property
    def label(self) -> str:
        return f"{self.oem} {self.model} {self.trim} · 软件 {self.software_version} ({self.release_status})"


@dataclass(frozen=True, slots=True)
class AdminGateView:
    """A deliberately narrow, safe subset of one existing gate result."""

    name: str
    category: str
    hard_or_soft: str
    threshold: str
    status: str
    reason: str
    evidence_path: str


@dataclass(frozen=True, slots=True)
class ReleaseGateView:
    """A read-only summary of a previously generated release-gate report."""

    status: str
    report_path: str
    message: str
    gates: tuple[AdminGateView, ...]


@dataclass(frozen=True, slots=True)
class KnowledgeAdminView:
    """Stable, typed data required by the read-only Knowledge Admin tab."""

    profiles: tuple[AdminProfileView, ...]
    release_gate: ReleaseGateView


def _text(value: Any, fallback: str = "—") -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _report_display_path(path: Path) -> str:
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.name


def _load_release_gate_view(report_path: Path) -> ReleaseGateView:
    display_path = _report_display_path(report_path)
    if not report_path.is_file():
        return ReleaseGateView(
            status="NOT_RUN",
            report_path=display_path,
            message="未运行：未找到 Release Gate 摘要报告。",
            gates=(),
        )

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ReleaseGateView(
            status="BLOCKED",
            report_path=display_path,
            message=f"Release Gate 报告无法读取：{exc.__class__.__name__}。",
            gates=(),
        )

    if not isinstance(payload, dict):
        return ReleaseGateView(
            status="BLOCKED",
            report_path=display_path,
            message="Release Gate 报告格式无效。",
            gates=(),
        )

    status = payload.get("decision")
    if status not in _RELEASE_GATE_STATUSES:
        return ReleaseGateView(
            status="BLOCKED",
            report_path=display_path,
            message="Release Gate 报告缺少有效总体状态。",
            gates=(),
        )

    gates_payload = payload.get("gates", [])
    gates: list[AdminGateView] = []
    if isinstance(gates_payload, list):
        for gate in gates_payload:
            if not isinstance(gate, dict):
                continue
            gates.append(
                AdminGateView(
                    name=_text(gate.get("name")),
                    category=_text(gate.get("category")),
                    hard_or_soft=_text(gate.get("hard_or_soft")),
                    threshold=_text(gate.get("threshold")),
                    status=_text(gate.get("status")),
                    reason=_text(gate.get("reason")),
                    evidence_path=_text(gate.get("evidence_path")),
                )
            )
    return ReleaseGateView(
        status=status,
        report_path=display_path,
        message="已读取现有 Release Gate 摘要；打开页面不会重新运行门禁。",
        gates=tuple(sorted(gates, key=lambda gate: (gate.name, gate.category))),
    )


def load_knowledge_admin_view(
    *,
    vehicle_registry: VehicleRegistry | None = None,
    document_registry: DocumentRegistry | None = None,
    release_gate_report_path: Path = DEFAULT_RELEASE_GATE_REPORT_PATH,
) -> KnowledgeAdminView:
    """Construct the local admin view without writing files or running a gate/model."""
    vehicles = vehicle_registry or VehicleRegistry()
    documents = document_registry or DocumentRegistry(vehicle_registry=vehicles)
    profiles: list[AdminProfileView] = []
    for profile in sorted(
        vehicles.list_profiles(include_pre_release=True), key=lambda item: item.vehicle_id
    ):
        applicable_documents = tuple(
            AdminDocumentView(
                document_id=document.document_id,
                title=document.title,
                content_type=document.content_type,
                document_version=document.document_version,
                status=document.status,
                locale=document.locale,
                source_path=document.source_path.as_posix(),
            )
            for document in sorted(
                documents.for_vehicle(profile.vehicle_id, status=None),
                key=lambda item: (item.content_type, item.document_version, item.document_id),
            )
        )
        profiles.append(
            AdminProfileView(
                vehicle_id=profile.vehicle_id,
                oem=profile.oem,
                model=profile.model,
                trim=profile.trim,
                software_version=profile.software_version,
                release_status=profile.release_status,
                documents=applicable_documents,
            )
        )
    return KnowledgeAdminView(
        profiles=tuple(profiles),
        release_gate=_load_release_gate_view(release_gate_report_path),
    )
