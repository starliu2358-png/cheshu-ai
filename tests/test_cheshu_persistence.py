from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from cockpit_agent.api import FeedbackStore, create_app
from cockpit_agent.vehicle_book.answerer import EvidenceAnswerer
from cockpit_agent.vehicle_book.service import VehicleBookService
from cockpit_agent.vehicle_book.sqlite_session import SQLiteSessionStore

TRIM = "model3-rwd-cn-current"


def _client(path: Path) -> tuple[TestClient, SQLiteSessionStore]:
    store = SQLiteSessionStore(path / "conversations.sqlite3")
    app = create_app(
        conversation_store=store,
        feedback_store=FeedbackStore(path / "feedback.jsonl"),
        service_factory=lambda: VehicleBookService(answerer=EvidenceAnswerer()),
        execution_mode="rag",
    )
    return TestClient(app), store


def test_conversation_metadata_history_and_context_survive_restart(tmp_path: Path) -> None:
    first_client, first_store = _client(tmp_path)
    created = first_client.post("/api/v1/conversations", json={"trimId": TRIM})
    assert created.status_code == 201
    conversation_id = created.json()["conversationId"]

    renamed = first_client.patch(
        f"/api/v1/conversations/{conversation_id}",
        json={"title": "哨兵模式 USB", "pinned": True},
    )
    assert renamed.status_code == 200
    answer = first_client.post(
        "/api/v1/answers",
        json={
            "question": "哨兵模式怎么使用？",
            "trimId": TRIM,
            "conversationId": conversation_id,
            "turnId": "client-turn-1",
        },
    ).json()
    feedback = first_client.post(
        f"/api/v1/answers/{answer['answerId']}/feedback", json={"value": "up"}
    )
    assert feedback.status_code == 200
    first_store.close()

    restarted_client, restarted_store = _client(tmp_path)
    listed = restarted_client.get("/api/v1/conversations").json()
    assert listed[0]["conversationId"] == conversation_id
    assert listed[0]["title"] == "哨兵模式 USB"
    assert listed[0]["pinned"] is True
    detail = restarted_client.get(f"/api/v1/conversations/{conversation_id}").json()
    assert detail["turns"][0]["id"] == "client-turn-1"
    assert detail["turns"][0]["answer"] == answer
    assert detail["turns"][0]["feedback"] == "up"
    assert detail["turns"][0]["status"] == "completed"
    assert (
        restarted_client.post(
            f"/api/v1/answers/{answer['answerId']}/feedback", json={"value": "down"}
        ).status_code
        == 200
    )
    assert (
        restarted_client.get(f"/api/v1/conversations/{conversation_id}").json()["turns"][0][
            "feedback"
        ]
        == "down"
    )
    assert (
        restarted_client.post(
            "/api/v1/answers",
            json={
                "question": "哨兵模式怎么使用？",
                "trimId": TRIM,
                "conversationId": conversation_id,
                "turnId": "client-turn-1",
            },
        ).json()
        == answer
    )
    assert (
        len(restarted_client.get(f"/api/v1/conversations/{conversation_id}").json()["turns"]) == 1
    )

    follow_up = restarted_client.post(
        "/api/v1/answers",
        json={"question": "不插U盘呢？", "trimId": TRIM, "conversationId": conversation_id},
    )
    assert follow_up.status_code == 200
    assert follow_up.json()["conversationId"] == conversation_id
    assert "USB" in follow_up.json()["answer"]
    assert (
        restarted_client.patch(
            f"/api/v1/conversations/{conversation_id}", json={"pinned": False}
        ).json()["pinned"]
        is False
    )
    assert restarted_client.delete(f"/api/v1/conversations/{conversation_id}").status_code == 204
    restarted_store.close()

    final_client, final_store = _client(tmp_path)
    assert final_client.get("/api/v1/conversations").json() == []
    assert final_client.get(f"/api/v1/conversations/{conversation_id}").status_code == 404
    assert (
        final_client.post(
            "/api/v1/answers",
            json={"question": "不插U盘呢？", "trimId": TRIM, "conversationId": conversation_id},
        ).status_code
        == 404
    )
    final_store.close()


def test_empty_chat_metadata_is_persistent_and_vehicle_is_fixed(tmp_path: Path) -> None:
    client, store = _client(tmp_path)
    created = client.post("/api/v1/conversations", json={"trimId": TRIM}).json()
    conversation_id = created["conversationId"]
    assert (
        client.patch(
            f"/api/v1/conversations/{conversation_id}", json={"title": "出发前检查"}
        ).status_code
        == 200
    )
    assert (
        client.patch(f"/api/v1/conversations/{conversation_id}", json={"title": " "}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/answers",
            json={
                "question": "如何打开充电口？",
                "trimId": "model3-lr-rwd-cn-current",
                "conversationId": conversation_id,
            },
        ).status_code
        == 409
    )
    store.close()

    restarted_client, restarted_store = _client(tmp_path)
    assert (
        restarted_client.get(f"/api/v1/conversations/{conversation_id}").json()["title"]
        == "出发前检查"
    )
    restarted_store.close()


def test_in_flight_turn_is_visible_and_interrupted_after_restart(tmp_path: Path) -> None:
    client, store = _client(tmp_path)
    conversation_id = client.post("/api/v1/conversations", json={"trimId": TRIM}).json()[
        "conversationId"
    ]
    store.begin_turn(conversation_id, "pending-1", "如何打开充电口？")
    pending = client.get(f"/api/v1/conversations/{conversation_id}").json()["turns"]
    assert pending[0]["status"] == "processing"
    assert pending[0]["answer"] is None
    store.close()

    restarted_client, restarted_store = _client(tmp_path)
    interrupted = restarted_client.get(f"/api/v1/conversations/{conversation_id}").json()["turns"]
    assert interrupted[0]["status"] == "interrupted"
    restarted_store.close()
