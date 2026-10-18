from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from cockpit_agent.vehicle_book.agent_loop import (
    DEEPSEEK_MODEL_SETTINGS,
    MAX_RETRIEVAL_CALLS,
    AgentAnswer,
    AgentContext,
    AgentOutputValidationError,
    AgentRunState,
    AgentToolRuntime,
    VehicleBookAgentService,
    search_vehicle_book,
)
from cockpit_agent.vehicle_book.registry import VehicleRegistry
from cockpit_agent.vehicle_book.schemas import (
    DocumentMetadata,
    KnowledgeChunk,
    RetrievedChunk,
)


def _retrieved(vehicle_id: str, chunk_id: str, content_type: str = "owner_manual") -> RetrievedChunk:
    metadata = DocumentMetadata(
        document_id=f"{vehicle_id}-{content_type}",
        vehicle_id=vehicle_id,
        title="测试官方资料",
        content_type=content_type,
        document_version="1",
        source_path=Path("data/test.md"),
        status="published",
        market="CN",
        source_url="https://www.tesla.cn/test",
    )
    return RetrievedChunk(
        chunk=KnowledgeChunk(
            chunk_id=chunk_id,
            section="测试章节",
            text="这是可信测试证据。",
            metadata=metadata,
        ),
        score=1.0,
        rank=1,
    )


@dataclass
class FakeRetriever:
    calls: list[tuple[str, str]] = field(default_factory=list)

    def retrieve(
        self,
        vehicle_id: str,
        query: str,
        top_k: int = 4,
        *,
        include_pre_release: bool = False,
    ) -> tuple[RetrievedChunk, ...]:
        self.calls.append((vehicle_id, query))
        return (_retrieved(vehicle_id, f"chunk-{len(self.calls)}"),)


def _context(retriever: FakeRetriever | None = None) -> AgentContext:
    registry = VehicleRegistry()
    return AgentContext(
        profile=registry.resolve("model3-rwd-cn-current"),
        retriever=retriever or FakeRetriever(),
        run_state=AgentRunState(),
    )


def test_search_tool_is_profile_bound_and_returns_evidence_only() -> None:
    retriever = FakeRetriever()
    context = _context(retriever)

    result = AgentToolRuntime(context).search("如何打开充电口？")

    assert result["ok"] is True
    assert set(result["results"][0]) == {
        "chunk_id",
        "document_id",
        "title",
        "section",
        "text",
        "vehicle_id",
        "document_version",
        "market",
        "score",
        "rank",
        "source_url",
        "evidence_role",
    }
    assert retriever.calls == [("model3-rwd-cn-current", "如何打开充电口？")]
    assert "answer" not in result
    assert context.run_state.tool_calls[0].tool == "search_vehicle_book"
    assert context.run_state.tool_calls[0].result_chunk_ids == ("chunk-1",)


def test_model_visible_search_schema_exposes_only_query() -> None:
    assert search_vehicle_book.name == "search_vehicle_book"
    assert set(search_vehicle_book.params_json_schema["properties"]) == {"query"}
    assert search_vehicle_book.params_json_schema["additionalProperties"] is False


def test_deepseek_agent_disables_thinking_for_chat_completions_adapter(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    service = VehicleBookAgentService()
    profile = service.vehicle_registry.resolve("model3-rwd-cn-current")

    agent = service._agent(profile)

    assert agent.model_settings == DEEPSEEK_MODEL_SETTINGS
    assert agent.model_settings.extra_body == {"thinking": {"type": "disabled"}}
    assert agent.model_settings.extra_args is None
    assert agent.output_type is None


def test_forced_evidence_retry_requires_first_tool_call(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    service = VehicleBookAgentService(retriever=FakeRetriever())
    profile = service.vehicle_registry.resolve("model3-rwd-cn-current")

    agent = service._agent(profile, require_search=True)

    assert agent.model_settings.tool_choice == "required"
    assert agent.reset_tool_choice is True
    assert service._agent(profile).model_settings.tool_choice is None


def test_agent_answer_parses_json_object_transport() -> None:
    parsed = VehicleBookAgentService._parse_agent_answer(
        '{"answer":"资料不足","cited_chunk_ids":[],"boundary_result":"unconfirmed",'
        '"insufficient_evidence":true,"clarification_question":null,'
        '"stop_reason":"insufficient_evidence"}'
    )

    assert parsed.insufficient_evidence is True
    assert parsed.stop_reason == "insufficient_evidence"


def test_duplicate_query_is_rejected_without_retrieval() -> None:
    retriever = FakeRetriever()
    context = _context(retriever)
    runtime = AgentToolRuntime(context)

    first = runtime.search("LCC 功能")
    duplicate = runtime.search("  lcc   功能 ")

    assert first["ok"] is True
    assert duplicate["error"] == "duplicate_query"
    assert len(retriever.calls) == 1
    assert context.run_state.retrieval_calls == 1
    assert context.run_state.tool_calls[-1].error == "duplicate_query"


def test_fourth_distinct_query_is_rejected_before_execution() -> None:
    retriever = FakeRetriever()
    context = _context(retriever)
    runtime = AgentToolRuntime(context)

    for index in range(MAX_RETRIEVAL_CALLS):
        assert runtime.search(f"query {index}")["ok"] is True
    denied = runtime.search("query 4")

    assert denied["error"] == "retrieval_limit"
    assert context.run_state.stop_reason == "retrieval_limit"
    assert len(retriever.calls) == MAX_RETRIEVAL_CALLS


def test_final_answer_must_cite_current_run_evidence() -> None:
    context = _context()
    with pytest.raises(AgentOutputValidationError, match="requires citations"):
        VehicleBookAgentService._validated_result(
            AgentAnswer(answer="确定性事实", cited_chunk_ids=[]),
            context,
        )


def test_availability_claim_requires_capability_evidence() -> None:
    context = _context()
    manual = _retrieved(context.profile.vehicle_id, "manual", "owner_manual")
    context.run_state.evidence_ledger["manual"] = manual
    context.run_state.evidence_roles["manual"] = "operation"

    with pytest.raises(AgentOutputValidationError, match="capability evidence"):
        VehicleBookAgentService._validated_result(
            AgentAnswer(
                answer="该配置支持此功能。",
                cited_chunk_ids=["manual"],
                boundary_result="supported",
            ),
            context,
        )


def test_validated_answer_preserves_only_ledger_citations() -> None:
    context = _context()
    configuration = _retrieved(
        context.profile.vehicle_id, "configuration", "configuration"
    )
    context.run_state.evidence_ledger["configuration"] = configuration
    context.run_state.evidence_roles["configuration"] = "capability"

    result = VehicleBookAgentService._validated_result(
        AgentAnswer(
            answer="该配置支持此功能。",
            cited_chunk_ids=["configuration"],
            boundary_result="supported",
        ),
        context,
    )

    assert result.boundary_result == "supported"
    assert result.citations[0].chunk_id == "configuration"
    assert result.vehicle_id == context.profile.vehicle_id


def test_clarification_returns_choices_without_search_or_citations() -> None:
    context = _context()
    result = VehicleBookAgentService._validated_result(
        AgentAnswer(
            answer="请说明想了解的车辆功能。",
            clarification_question="你想了解哪项车辆功能？",
            clarification_options=["哨兵模式的录像如何保存？", "USB 存储设备有什么要求？"],
            stop_reason="clarification_required",
        ),
        context,
    )

    assert result.fallback_reason == "clarification_required"
    assert result.clarification_options == (
        "哨兵模式的录像如何保存？",
        "USB 存储设备有什么要求？",
    )
    assert result.citations == ()
    assert context.run_state.retrieval_calls == 0


def test_unrelated_input_is_not_forced_into_previous_feature() -> None:
    context = _context()
    context.conversation_context = {
        "current_feature": "哨兵模式",
        "follow_up_focus": None,
        "pending_clarification": None,
    }

    result = VehicleBookAgentService._validated_result(
        AgentAnswer(
            answer="请说明问题。",
            clarification_question="你想了解哨兵模式的哪方面？",
            clarification_options=["哨兵模式怎么开？", "哨兵模式怎么录像？"],
            stop_reason="clarification_required",
        ),
        context,
    )

    assert "新的车辆问题" in result.answer
    assert any("另一项车辆功能" in choice for choice in result.clarification_options)


def test_generic_clarification_always_offers_a_new_topic() -> None:
    context = _context()
    context.conversation_context = {
        "current_feature": "哨兵模式",
        "follow_up_focus": None,
        "pending_clarification": None,
    }

    result = VehicleBookAgentService._validated_result(
        AgentAnswer(
            answer="请说明问题。",
            clarification_question="您想了解 Model 3 的哪个方面？",
            clarification_options=["哨兵模式如何保存录像？", "行车记录仪如何设置？"],
        ),
        context,
    )

    assert result.clarification_options[-1] == "我想问另一项车辆功能，具体问题是……"


def test_follow_up_cannot_repeat_previous_answer_verbatim() -> None:
    context = _context()
    context.conversation_context = {
        "follow_up_focus": "USB 设备",
        "previous_answer": "哨兵模式可以使用 USB 设备保存录像。",
    }
    item = _retrieved(context.profile.vehicle_id, "manual")
    context.run_state.evidence_ledger["manual"] = item
    context.run_state.evidence_roles["manual"] = "operation"

    with pytest.raises(AgentOutputValidationError, match="repeats"):
        VehicleBookAgentService._validated_result(
            AgentAnswer(
                answer="哨兵模式可以使用 USB 设备保存录像。",
                cited_chunk_ids=["manual"],
            ),
            context,
        )


def test_invalid_context_does_not_crash_agent_setup() -> None:
    assert VehicleBookAgentService._conversation_context("not json") == {}
    assert VehicleBookAgentService._conversation_context("[]") == {}


def test_explicit_clarification_reply_cannot_repeat_topic_picker() -> None:
    context = _context()
    context.current_question = "我想继续问充电，具体问题是几个充电口"
    options = ["我想继续问充电，具体问题是……", "我想问另一项车辆功能，具体问题是……"]
    context.conversation_context = {
        "pending_clarification": "你想继续问充电，还是问另一项车辆功能？",
        "clarification_options": options,
    }
    with pytest.raises(AgentOutputValidationError, match="Repeated prior clarification"):
        VehicleBookAgentService._validated_result(
            AgentAnswer(answer="请选择", clarification_question="你想继续问哪一项？",
                        clarification_options=options), context,
        )
    # A genuinely new ambiguity may still be clarified.
    result = VehicleBookAgentService._validated_result(
        AgentAnswer(answer="你是问盖板还是插枪接口？",
                    clarification_question="你是问盖板还是插枪接口？",
                    clarification_options=["盖板数量", "插枪接口数量"]), context,
    )
    assert result.fallback_reason == "clarification_required"


def test_repair_receives_current_question_not_just_old_clarification(monkeypatch) -> None:
    import asyncio
    import json
    from types import SimpleNamespace
    from cockpit_agent.vehicle_book import agent_loop

    context = _context()
    context.current_question = "具体问题是几个充电口"
    context.conversation_context = {"pending_clarification": "想问哪项功能？"}
    service = VehicleBookAgentService(retriever=FakeRetriever())
    monkeypatch.setattr(service, "_model", lambda: "test-model")

    async def run(agent, prompt, **kwargs):
        assert json.loads(prompt)["current_question"] == "具体问题是几个充电口"
        return SimpleNamespace(final_output=AgentAnswer(
            answer="暂时无法确认", insufficient_evidence=True,
            boundary_result="unconfirmed", stop_reason="insufficient_evidence",
        ), raw_responses=[object()])

    monkeypatch.setattr(agent_loop.Runner, "run", run)
    result = asyncio.run(service._repair({}, context, ValueError("invalid")))
    assert result.insufficient_evidence
    assert context.run_state.model_turns == 1


def test_specific_low_voltage_clarification_is_not_replaced_by_topic_picker() -> None:
    context = _context()
    context.current_question = "低压充电怎么操作？"
    context.conversation_context = {"current_feature": "充电"}
    question = "你是想给低压电池补电，还是用家用电源给车辆充电？"
    options = ["给低压电池补电", "用家用电源给车辆充电"]
    result = VehicleBookAgentService._validated_result(
        AgentAnswer(answer=question, clarification_question=question,
                    clarification_options=options), context,
    )
    assert result.answer == question
    assert result.clarification_options == tuple(options)


def test_general_answer_has_no_official_provenance() -> None:
    result = VehicleBookAgentService._validated_result(
        AgentAnswer(answer="低温通常会影响电池充电速度。", answer_source="general",
                    boundary_result="unconfirmed"), _context(),
    )
    assert result.answer_source == "general"
    assert result.citations == ()
    assert result.fallback_reason is None


@pytest.mark.parametrize("extra", [{"cited_chunk_ids": ["fake"]}, {"boundary_result": "supported"}])
def test_general_answer_cannot_claim_official_confirmation(extra) -> None:
    with pytest.raises(AgentOutputValidationError, match="General knowledge"):
        VehicleBookAgentService._validated_result(
            AgentAnswer(answer="推测", answer_source="general", **extra), _context(),
        )


def test_insufficient_evidence_triggers_bounded_supplement(monkeypatch, tmp_path) -> None:
    import asyncio
    from types import SimpleNamespace
    from cockpit_agent.vehicle_book import agent_loop

    service = VehicleBookAgentService(retriever=FakeRetriever(), log_path=tmp_path / "runs.jsonl")
    monkeypatch.setattr(service, "_model", lambda: "test-model")
    calls = []

    async def run(agent, prompt, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            AgentToolRuntime(kwargs["context"]).search("冬天充电慢")
            answer = AgentAnswer(answer="资料不足", insufficient_evidence=True)
        else:
            assert kwargs["max_turns"] == 1
            answer = AgentAnswer(answer="低温一般会降低充电速度。", answer_source="general")
        return SimpleNamespace(final_output=answer, raw_responses=[object()])

    monkeypatch.setattr(agent_loop.Runner, "run", run)
    result = asyncio.run(service.answer("冬天充电慢？", "model3-rwd-cn-current"))
    assert len(calls) == 2
    assert result.answer_source == "general"
    assert "低温" in result.answer


def test_general_source_survives_sqlite_reload_and_summary(tmp_path) -> None:
    from cockpit_agent.vehicle_book.sqlite_session import SQLiteSessionStore
    from cockpit_agent.vehicle_book.session import _summary_line
    path = tmp_path / "sessions.sqlite3"
    store = SQLiteSessionStore(path)
    cid = store.resolve(None, "model3-rwd-cn-current")
    store.append(cid, "冬天充电慢？", "一般是低温影响。", answer_source="general")
    restored = SQLiteSessionStore(path).snapshot(cid).recent_turns[0]
    assert restored.answer_source == "general"
    assert "来源=general" in _summary_line(restored)


def test_supplement_cannot_use_seventh_model_turn(monkeypatch, tmp_path) -> None:
    import asyncio
    from types import SimpleNamespace
    from cockpit_agent.vehicle_book import agent_loop
    service = VehicleBookAgentService(retriever=FakeRetriever(), log_path=tmp_path / "runs.jsonl")
    monkeypatch.setattr(service, "_model", lambda: "test-model")
    calls = []

    async def run(agent, prompt, **kwargs):
        calls.append(1)
        AgentToolRuntime(kwargs["context"]).search("充电")
        return SimpleNamespace(final_output=AgentAnswer(answer="不足", insufficient_evidence=True),
                               raw_responses=[object()] * 6)

    monkeypatch.setattr(agent_loop.Runner, "run", run)
    result = asyncio.run(service.answer("充电？", "model3-rwd-cn-current"))
    assert len(calls) == 1
    assert result.fallback_reason == "model_turn_limit"


def test_clock_has_seconds_weekday_and_explicit_timezone(monkeypatch):
    from datetime import datetime
    from cockpit_agent.vehicle_book import agent_loop
    class FixedClock:
        @staticmethod
        def now(tz):
            return datetime(2026, 10, 17, 18, 7, 23, tzinfo=tz)
    monkeypatch.setattr(agent_loop, "datetime", FixedClock)
    assert agent_loop.current_time_payload() == {
        "datetime": "2026-10-17T18:07:23+08:00",
        "timezone": "Asia/Shanghai", "weekday": "星期六",
    }
    service = VehicleBookAgentService(retriever=FakeRetriever())
    monkeypatch.setattr(service, "_model", lambda: "test-model")
    agent = service._agent(service.vehicle_registry.resolve("model3-rwd-cn-current"))
    assert "18:07:23" in agent.instructions
    assert {tool.name for tool in agent.tools} == {"search_vehicle_book", "get_current_time"}


def test_no_evidence_does_not_force_general_questions_into_manual(monkeypatch, tmp_path):
    import asyncio
    from types import SimpleNamespace
    from cockpit_agent.vehicle_book import agent_loop
    retriever = FakeRetriever()
    service = VehicleBookAgentService(retriever=retriever, log_path=tmp_path / "runs.jsonl")
    monkeypatch.setattr(service, "_model", lambda: "test-model")
    calls = []
    async def run(agent, prompt, **kwargs):
        calls.append(agent)
        answer = (AgentAnswer(answer="不足", insufficient_evidence=True) if len(calls) == 1
                  else AgentAnswer(answer="现在是北京时间18点。", answer_source="general"))
        return SimpleNamespace(final_output=answer, raw_responses=[object()])
    monkeypatch.setattr(agent_loop.Runner, "run", run)
    result = asyncio.run(service.answer("现在几点？", "model3-rwd-cn-current"))
    assert result.answer_source == "general"
    assert retriever.calls == []
    assert len(calls) == 2
    assert calls[1].tools == []


def test_json_fence_is_transport_not_evidence_failure():
    result = VehicleBookAgentService._parse_agent_answer(
        '```json\n{"answer":"你好", "answer_source":"general"}\n```'
    )
    assert result.answer == "你好"
    with pytest.raises(Exception):
        VehicleBookAgentService._parse_agent_answer('prefix {"answer":"hello"}')


def test_validation_diagnostic_does_not_log_answer_text():
    from pydantic import ValidationError
    from cockpit_agent.vehicle_book.agent_loop import _validation_code
    try:
        AgentAnswer.model_validate({"answer": "private text", "answer_source": "wrong"})
    except ValidationError as error:
        code = _validation_code(error)
        assert "answer_source:literal_error" in code
        assert "private text" not in code
