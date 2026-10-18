from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cockpit_agent.api import FeedbackStore, create_app
from cockpit_agent.vehicle_book.answerer import EvidenceAnswerer
from cockpit_agent.vehicle_book.schemas import AnswerResult, Citation
from cockpit_agent.vehicle_book.service import VehicleBookService
from cockpit_agent.vehicle_book.sqlite_session import SQLiteSessionStore


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    return TestClient(
        create_app(
            feedback_store=FeedbackStore(tmp_path / "feedback.jsonl"),
            conversation_store=SQLiteSessionStore(tmp_path / "conversations.sqlite3"),
            service_factory=lambda: VehicleBookService(answerer=EvidenceAnswerer()),
            execution_mode="rag",
        )
    )


def test_lists_only_four_released_tesla_cn_trims(tmp_path: Path, monkeypatch) -> None:
    response = _client(tmp_path, monkeypatch).get("/api/v1/vehicles")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 4
    assert {item["trimId"] for item in payload} == {
        "model3-rwd-cn-current",
        "model3-lr-rwd-cn-current",
        "model3-lr-awd-cn-current",
        "model3-performance-awd-cn-current",
    }
    assert all(item["model"] == "Model 3" for item in payload)


def test_answers_charging_question_without_model_key(tmp_path: Path, monkeypatch) -> None:
    response = _client(tmp_path, monkeypatch).post(
        "/api/v1/answers",
        json={
            "question": "如何打开充电口？",
            "trimId": "model3-rwd-cn-current",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["boundaryResult"] is None
    assert payload["fallbackReason"] is None
    assert "打开充电端口" in payload["answer"]
    assert "切勿强行打开" in payload["answer"]
    assert payload["speechText"] == payload["answer"]
    assert payload["citations"] == [
        {
            "id": "tesla_model3_rwd_cn_charging_2024plus",
            "title": "Tesla 中国 Model 3 车主手册",
            "section": "打开充电端口",
            "url": "https://www.tesla.cn/ownersmanual/model3/zh_cn/"
            "GUID-BEE08D47-0CE0-4BDD-83F2-9854FB3D578F.html",
            "retrievedAt": "2026-10-18",
        }
    ]


def test_zero_retrieval_is_unconfirmed_not_unsupported(tmp_path: Path, monkeypatch) -> None:
    response = _client(tmp_path, monkeypatch).post(
        "/api/v1/answers",
        json={
            "question": "月球基地咖啡机怎么校准？",
            "trimId": "model3-rwd-cn-current",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fallbackReason"] == "no_retrieval_results"
    assert payload["boundaryResult"] is None
    assert "还不能确认" in payload["answer"]
    assert payload["citations"] == []


def test_rejects_market_and_software_fields(tmp_path: Path, monkeypatch) -> None:
    response = _client(tmp_path, monkeypatch).post(
        "/api/v1/answers",
        json={
            "question": "如何打开充电口？",
            "trimId": "model3-rwd-cn-current",
            "market": "US",
            "softwareVersion": "2026.1",
        },
    )

    assert response.status_code == 422


def test_rejects_unknown_trim(tmp_path: Path, monkeypatch) -> None:
    response = _client(tmp_path, monkeypatch).post(
        "/api/v1/answers",
        json={"question": "如何打开充电口？", "trimId": "model-y-us"},
    )

    assert response.status_code == 404


def test_feedback_is_idempotent_and_switchable(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    answer = client.post(
        "/api/v1/answers",
        json={
            "question": "如何打开充电口？",
            "trimId": "model3-rwd-cn-current",
        },
    ).json()
    endpoint = f"/api/v1/answers/{answer['answerId']}/feedback"

    first = client.post(endpoint, json={"value": "up"})
    repeated = client.post(endpoint, json={"value": "up"})
    switched = client.post(endpoint, json={"value": "down"})

    assert first.status_code == repeated.status_code == switched.status_code == 200
    assert first.json()["value"] == repeated.json()["value"] == "up"
    assert first.json()["recordedAt"] == repeated.json()["recordedAt"]
    assert switched.json()["value"] == "down"
    assert len((tmp_path / "feedback.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_feedback_rejects_unknown_answer(tmp_path: Path, monkeypatch) -> None:
    response = _client(tmp_path, monkeypatch).post(
        "/api/v1/answers/missing/feedback", json={"value": "up"}
    )

    assert response.status_code == 404


def test_answers_sentry_mode_with_official_manual(tmp_path: Path, monkeypatch) -> None:
    response = _client(tmp_path, monkeypatch).post(
        "/api/v1/answers",
        json={
            "question": "哨兵模式怎么用？",
            "trimId": "model3-lr-awd-cn-current",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fallbackReason"] is None
    assert payload["boundaryResult"] is None
    assert "控制” > “安全” > “哨兵模式" in payload["answer"]
    assert "USB" in payload["answer"]
    assert payload["citations"][0]["url"].endswith(
        "GUID-56703182-8191-4DAE-AF07-2FDC0EB64663.html"
    )


def test_answers_configuration_difference_and_acceleration(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    difference = client.post(
        "/api/v1/answers",
        json={
            "question": "不同配置有什么差异？",
            "trimId": "model3-rwd-cn-current",
        },
    ).json()
    acceleration = client.post(
        "/api/v1/answers",
        json={
            "question": "加速情况",
            "trimId": "model3-performance-awd-cn-current",
        },
    ).json()

    assert difference["fallbackReason"] is None
    assert all(
        value in difference["answer"]
        for value in ("6.1 秒", "5.2 秒", "3.8 秒", "3.1 秒")
    )
    assert difference["citations"][0]["url"] == "https://www.tesla.cn/model3"
    assert "Performance 高性能全轮驱动版为 3.1 秒" in acceleration["answer"]
    assert acceleration["citations"][0]["url"] == "https://www.tesla.cn/model3"


def test_answers_driver_assistance_and_does_not_call_it_autonomous_driving(
    tmp_path: Path, monkeypatch
) -> None:
    response = _client(tmp_path, monkeypatch).post(
        "/api/v1/answers",
        json={
            "question": "有哪些自动驾驶功能？",
            "trimId": "model3-rwd-cn-current",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fallbackReason"] is None
    assert "不能让 Model 3 实现自动驾驶" in payload["answer"]
    assert "主动巡航控制" in payload["answer"]
    assert payload["citations"][0]["url"].endswith(
        "GUID-101D1BF5-52D2-469A-A57D-E7230BBEE94B.html"
    )


def test_lcc_question_is_unconfirmed_and_explains_official_term(
    tmp_path: Path, monkeypatch
) -> None:
    response = _client(tmp_path, monkeypatch).post(
        "/api/v1/answers",
        json={
            "question": "是否具备LCC功能？",
            "trimId": "model3-rwd-cn-current",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fallbackReason"] is None
    assert payload["boundaryResult"] == "unconfirmed"
    assert "辅助转向" in payload["answer"]
    assert "还不能确定" in payload["answer"]


def test_active_cruise_availability_is_explicitly_supported(
    tmp_path: Path, monkeypatch
) -> None:
    response = _client(tmp_path, monkeypatch).post(
        "/api/v1/answers",
        json={
            "question": "是否具备主动巡航控制？",
            "trimId": "model3-lr-rwd-cn-current",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fallbackReason"] is None
    assert payload["boundaryResult"] == "supported"
    assert "所有 Tesla 车辆均配备主动巡航控制" in payload["answer"]


@pytest.mark.parametrize(
    ("question", "expected_text", "expected_section"),
    [
        ("如何打开前备箱？", "手机应用程序", "打开前备箱"),
        ("儿童安全座椅怎么安装？", "儿童约束系统", "座椅和安全约束系统"),
        ("如何连接手机蓝牙？", "控制” > “蓝牙", "连接手机蓝牙"),
        ("防撞辅助包括什么？", "自动紧急制动", "主动安全功能"),
        ("温度控制怎么使用？", "自动模式", "温度控制"),
        ("导航如何规划充电站？", "超级充电站", "导航和娱乐"),
        ("如何预约充电？", "设置定时充电", "定时充电、出发时间和续航建议"),
        ("软件怎么更新？", "控制” > “软件", "软件和日常维护"),
    ],
)
def test_answers_expanded_manual_topics(
    tmp_path: Path,
    monkeypatch,
    question: str,
    expected_text: str,
    expected_section: str,
) -> None:
    response = _client(tmp_path, monkeypatch).post(
        "/api/v1/answers",
        json={"question": question, "trimId": "model3-rwd-cn-current"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fallbackReason"] is None
    assert expected_text in payload["answer"]
    assert payload["citations"][0]["section"] == expected_section
    assert payload["citations"][0]["url"] == (
        "https://www.tesla.cn/ownersmanual/model3/zh_cn/Owners_Manual.pdf"
    )


def test_distinguishes_normal_and_no_power_front_trunk_procedures(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    normal = client.post(
        "/api/v1/answers",
        json={"question": "如何打开前备箱？", "trimId": "model3-rwd-cn-current"},
    ).json()
    no_power = client.post(
        "/api/v1/answers",
        json={"question": "没电时怎么打开前备箱？", "trimId": "model3-rwd-cn-current"},
    ).json()

    assert normal["citations"][0]["section"] == "打开前备箱"
    assert no_power["citations"][0]["section"] == "没电时打开前备箱"
    assert "外接低压电源" in no_power["answer"]


def test_natural_follow_up_reuses_context_and_conversation_id(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    first = client.post(
        "/api/v1/answers",
        json={"question": "哨兵模式怎么使用？", "trimId": "model3-rwd-cn-current"},
    ).json()
    follow_up = client.post(
        "/api/v1/answers",
        json={
            "question": "不插U盘呢？",
            "trimId": "model3-rwd-cn-current",
            "conversationId": first["conversationId"],
        },
    )

    assert follow_up.status_code == 200
    payload = follow_up.json()
    assert payload["conversationId"] == first["conversationId"]
    assert payload["fallbackReason"] is None
    assert "USB" in payload["answer"]
    assert "哨兵模式" in payload["citations"][0]["section"]


def test_conversation_rejects_trim_switch_and_can_be_deleted(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    first = client.post(
        "/api/v1/answers",
        json={"question": "如何充电？", "trimId": "model3-rwd-cn-current"},
    ).json()
    changed = client.post(
        "/api/v1/answers",
        json={
            "question": "那这个版本呢？",
            "trimId": "model3-lr-rwd-cn-current",
            "conversationId": first["conversationId"],
        },
    )
    deleted = client.delete(f"/api/v1/conversations/{first['conversationId']}")

    assert changed.status_code == 409
    assert deleted.status_code == 204


def test_systematic_three_state_boundary_eval(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    eval_path = Path("data/eval/cheshu_boundary_eval.jsonl")
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        response = client.post(
            "/api/v1/answers",
            json={"question": case["question"], "trimId": "model3-rwd-cn-current"},
        )
        assert response.status_code == 200, case["id"]
        assert response.json()["boundaryResult"] == case["expected"], case["id"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("续航差异", "830 公里"),
        ("充电功率有什么差异？", "250 千瓦"),
        ("音响配置有什么差异？", "15 个扬声器"),
        ("定时充电怎么设置？", "计划出发时间"),
    ],
)
def test_answers_finer_v1_coverage(
    tmp_path: Path, monkeypatch, question: str, expected: str
) -> None:
    payload = _client(tmp_path, monkeypatch).post(
        "/api/v1/answers",
        json={"question": question, "trimId": "model3-rwd-cn-current"},
    ).json()
    assert payload["fallbackReason"] is None
    assert expected in payload["answer"]


def test_general_answer_is_saved_and_unconfirmed_preflight_reaches_agent(tmp_path: Path) -> None:
    class GeneralAgent:
        async def answer(self, question, vehicle_id, **kwargs):
            return AnswerResult(answer="一般而言，这是一类驾驶辅助功能。",
                                vehicle_id=vehicle_id, answer_source="general",
                                boundary_result="unconfirmed")

    store = SQLiteSessionStore(tmp_path / "general.sqlite3")
    client = TestClient(create_app(
        execution_mode="agent", agent_service_factory=GeneralAgent,
        conversation_store=store, feedback_store=FeedbackStore(tmp_path / "feedback.jsonl"),
    ))
    payload = client.post("/api/v1/answers", json={
        "question": "NOA 是什么？", "trimId": "model3-rwd-cn-current",
    }).json()
    assert payload["answerSource"] == "general"
    assert payload["citations"] == []
    assert payload["fallbackReason"] is None
    assert payload["speechText"] == payload["answer"]
    assert store.snapshot(payload["conversationId"]).recent_turns[-1].answer_source == "general"


def test_agent_mode_keeps_public_answer_schema(tmp_path: Path, monkeypatch) -> None:
    class FakeAgentService:
        async def answer(
            self,
            question: str,
            vehicle_id: str,
            *,
            context: str | None = None,
            conversation_id: str | None = None,
        ) -> AnswerResult:
            assert question == "如何打开充电口？"
            assert vehicle_id == "model3-rwd-cn-current"
            assert conversation_id
            return AnswerResult(
                answer="通过 Agent Tool 找到的回答。",
                vehicle_id=vehicle_id,
                citations=(
                    Citation(
                        chunk_id="charging:open",
                        document_id="tesla_model3_rwd_cn_charging_2024plus",
                        vehicle_id=vehicle_id,
                        title="Tesla 中国 Model 3 车主手册",
                        document_version="2024+",
                        section="打开充电端口",
                        source_path=Path("data/charging.md"),
                        rank=1,
                        score=1.0,
                        source_url="https://www.tesla.cn/ownersmanual/model3/zh_cn/test.html",
                        market="CN",
                        retrieved_at="2026-10-09",
                    ),
                ),
            )

    client = TestClient(
        create_app(
            feedback_store=FeedbackStore(tmp_path / "feedback.jsonl"),
            conversation_store=SQLiteSessionStore(tmp_path / "conversations.sqlite3"),
            execution_mode="agent",
            agent_service_factory=FakeAgentService,
        )
    )

    payload = client.post(
        "/api/v1/answers",
        json={
            "question": "如何打开充电口？",
            "trimId": "model3-rwd-cn-current",
        },
    ).json()

    assert set(payload) == {
        "answerId",
        "conversationId",
        "answer",
        "boundaryResult",
        "citations",
        "speechText",
        "fallbackReason",
        "clarificationOptions",
        "answerSource",
    }
    assert payload["answer"] == "通过 Agent Tool 找到的回答。"
    assert payload["citations"][0]["section"] == "打开充电端口"


def test_agent_mode_keeps_deterministic_scope_boundary(tmp_path: Path) -> None:
    class UnexpectedAgentService:
        async def answer(self, *args, **kwargs):
            raise AssertionError("Out-of-scope requests must not enter the Agent loop")

    client = TestClient(
        create_app(
            feedback_store=FeedbackStore(tmp_path / "feedback.jsonl"),
            conversation_store=SQLiteSessionStore(tmp_path / "conversations.sqlite3"),
            execution_mode="agent",
            agent_service_factory=UnexpectedAgentService,
        )
    )
    payload = client.post(
        "/api/v1/answers",
        json={
            "question": "现在车速是多少？",
            "trimId": "model3-rwd-cn-current",
        },
    ).json()

    assert payload["boundaryResult"] == "unsupported"
    assert "无法查看车辆的实时" in payload["answer"]


def test_clarification_choices_are_returned_and_follow_up_stays_in_session(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str | None]] = []

    class FakeAgentService:
        async def answer(self, question, vehicle_id, *, context, conversation_id):
            calls.append((question, context))
            if len(calls) == 1:
                return AnswerResult(
                    answer="你想了解哪项车辆功能？",
                    vehicle_id=vehicle_id,
                    fallback_reason="clarification_required",
                    clarification_options=(
                        "哨兵模式录像如何保存？",
                        "USB 存储设备有什么要求？",
                    ),
                )
            return AnswerResult(
                answer="这次回答依据手册检索结果。",
                vehicle_id=vehicle_id,
                fallback_reason="insufficient_evidence",
            )

    client = TestClient(
        create_app(
            feedback_store=FeedbackStore(tmp_path / "feedback.jsonl"),
            conversation_store=SQLiteSessionStore(tmp_path / "conversations.sqlite3"),
            execution_mode="agent",
            agent_service_factory=FakeAgentService,
        )
    )
    first = client.post(
        "/api/v1/answers",
        json={"question": "沙发JFK数据飞轮书法家", "trimId": "model3-rwd-cn-current"},
    ).json()
    second = client.post(
        "/api/v1/answers",
        json={
            "question": first["clarificationOptions"][0],
            "trimId": "model3-rwd-cn-current",
            "conversationId": first["conversationId"],
        },
    ).json()

    assert first["fallbackReason"] == "clarification_required"
    assert len(first["clarificationOptions"]) == 2
    assert second["conversationId"] == first["conversationId"]
    assert "pending_clarification" in (calls[1][1] or "")
