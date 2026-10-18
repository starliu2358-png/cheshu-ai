import hashlib
import json
from pathlib import Path

from cockpit_agent.vehicle_book.admin import load_knowledge_admin_view
from cockpit_agent.vehicle_book.registry import DEFAULT_DOCUMENTS_PATH, DEFAULT_VEHICLES_PATH


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_admin_profiles_are_stable_and_documents_are_profile_scoped() -> None:
    view = load_knowledge_admin_view()

    assert [profile.vehicle_id for profile in view.profiles] == sorted(
        profile.vehicle_id for profile in view.profiles
    )
    assert [document.content_type for document in view.profiles[0].documents] == [
        "adas",
        "configuration",
        "owner_manual",
    ]
    document_ids_by_profile = [
        {document.document_id for document in profile.documents} for profile in view.profiles
    ]
    for profile in view.profiles:
        assert profile.documents
        assert all(".env" not in document.source_path for document in profile.documents)
    assert not document_ids_by_profile[0] & document_ids_by_profile[1]
    assert not document_ids_by_profile[0] & document_ids_by_profile[2]
    assert not document_ids_by_profile[1] & document_ids_by_profile[2]


def test_admin_reads_existing_gate_status_without_running_gate(tmp_path: Path) -> None:
    report = tmp_path / "release_gate_summary.json"
    report.write_text(
        json.dumps(
            {
                "decision": "FAIL",
                "api_key": "should-not-be-exposed",
                "prompt": "a complete prompt should not be exposed",
                "gates": [
                    {
                        "name": "pytest",
                        "category": "test",
                        "hard_or_soft": "hard",
                        "threshold": "100%",
                        "status": "FAIL",
                        "reason": "pytest failed",
                        "evidence_path": "outputs/eval",
                        "actual": "should-not-be-exposed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    view = load_knowledge_admin_view(release_gate_report_path=report)
    rendered = repr(view)

    assert view.release_gate.status == "FAIL"
    assert view.release_gate.gates[0].status == "FAIL"
    assert "should-not-be-exposed" not in rendered
    assert "complete prompt" not in rendered


def test_admin_marks_missing_gate_report_as_not_run(tmp_path: Path) -> None:
    view = load_knowledge_admin_view(
        release_gate_report_path=tmp_path / "release_gate_summary.json"
    )

    assert view.release_gate.status == "NOT_RUN"
    assert "未运行" in view.release_gate.message
    assert not view.release_gate.gates


def test_admin_loading_does_not_write_registry_or_report(tmp_path: Path) -> None:
    report = tmp_path / "release_gate_summary.json"
    report.write_text('{"decision": "PASS", "gates": []}', encoding="utf-8")
    before = {path: _sha256(path) for path in (DEFAULT_VEHICLES_PATH, DEFAULT_DOCUMENTS_PATH, report)}

    load_knowledge_admin_view(release_gate_report_path=report)

    assert {path: _sha256(path) for path in before} == before
