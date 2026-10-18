import asyncio
import json
from types import SimpleNamespace

import pytest

from cockpit_agent.vehicle_book import agent_loop as loop
from test_vehicle_book_agent_loop import FakeRetriever, _context


def grounded(text="这是可信测试证据。", source="official", supplement=""):
    return loop.AgentAnswer(answer=text + ("\n\n" + supplement if supplement else ""),
        answer_source=source, cited_chunk_ids=["chunk-1"],
        grounded_segments=[loop.GroundedSegment(text=text, cited_chunk_ids=["chunk-1"])],
        general_supplement=supplement)


@pytest.mark.parametrize("fence", ["```JSON\r\n{}\r\n```", "```json  \n{}\n```", "```\n{}\n```"])
def test_harmless_transport(fence):
    raw = json.dumps({"answer": "你好", "answer_source": "general"})
    assert loop.VehicleBookAgentService._parse_agent_answer(fence.format(raw)).answer == "你好"


@pytest.mark.parametrize("raw", ['{"answer":', 'prefix {"answer":"你好"}', '{"answer":"你好","invented":1}'])
def test_no_arbitrary_json_recovery(raw):
    with pytest.raises(ValueError):
        loop.VehicleBookAgentService._parse_agent_answer(raw)


def test_mixed_segment_contract():
    ctx = _context()
    loop.AgentToolRuntime(ctx).search("测试")
    answer = grounded(source="mixed", supplement="一般来说，可以先检查设置。")
    result = loop.VehicleBookAgentService._validated_result(answer, ctx)
    assert result.answer_source == "mixed"
    assert len(result.citations) == 1
    with pytest.raises(loop.AgentOutputValidationError, match="segments"):
        loop.VehicleBookAgentService._validated_result(answer.model_copy(update={"answer": "不匹配"}), ctx)


def setup_service(monkeypatch, tmp_path, outputs):
    service = loop.VehicleBookAgentService(retriever=FakeRetriever(), log_path=tmp_path / "log.jsonl")
    monkeypatch.setattr(service, "_model", lambda: "test-model")
    calls = []
    async def run(agent, prompt, **kwargs):
        calls.append((agent, prompt, kwargs))
        # Exercise real shared hooks, including failed/in-flight calls.
        await kwargs["hooks"].on_llm_start(SimpleNamespace(context=kwargs["context"]), agent, None, [])
        if len(calls) == 1:
            loop.AgentToolRuntime(kwargs["context"]).search("测试")
            assert kwargs["max_turns"] == 4
        else:
            assert not agent.tools
            assert kwargs["max_turns"] == 1
        return SimpleNamespace(final_output=outputs[len(calls)-1], raw_responses=[object()])
    monkeypatch.setattr(loop.Runner, "run", run)
    return service, calls


def test_schema_repair_then_review(monkeypatch, tmp_path):
    service, calls = setup_service(monkeypatch, tmp_path, ["invalid", grounded(), grounded()])
    result = asyncio.run(service.answer("测试", "model3-rwd-cn-current"))
    assert result.fallback_reason is None
    assert len(calls) == 3
    assert loop.OUTPUT_CONTRACT in calls[0][0].instructions
    assert loop.OUTPUT_CONTRACT in calls[1][0].instructions
    assert json.loads(calls[2][1])["supplied_evidence"][0]["chunk_id"] == "chunk-1"


def test_review_removes_unsupported_claim_not_just_relabels(monkeypatch, tmp_path):
    candidate = loop.AgentAnswer(answer="这车一定更省电。", cited_chunk_ids=["chunk-1"])
    safe = loop.AgentAnswer(answer="单凭续航不能判断能耗，还需要比较同工况数据。", answer_source="general")
    service, calls = setup_service(monkeypatch, tmp_path, [candidate, safe])
    result = asyncio.run(service.answer("是不是更省电？", "model3-rwd-cn-current"))
    assert result.answer == safe.answer
    assert result.citations == ()
    assert len(calls) == 2


def test_invalid_review_only_one_correction(monkeypatch, tmp_path):
    service, calls = setup_service(monkeypatch, tmp_path, [grounded(), "bad", "bad"])
    result = asyncio.run(service.answer("测试", "model3-rwd-cn-current"))
    assert result.fallback_reason == "invalid_final_output"
    assert len(calls) == 3
    audit = json.loads((tmp_path / "log.jsonl").read_text(encoding="utf-8"))
    assert audit["model_turns"] == 3
    assert audit["repair_attempts"] == 1


def test_budget_blocks_seventh_call_and_expired_deadline():
    ctx = _context()
    wrapper = SimpleNamespace(context=ctx)
    ctx.run_state.model_turns = 6
    with pytest.raises(loop.MaxTurnsExceeded):
        asyncio.run(loop.BudgetHooks().on_llm_start(wrapper, None, None, []))
    ctx.run_state.started_at -= 16
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(loop.BudgetHooks().on_llm_start(wrapper, None, None, []))


def test_review_timeout_never_releases_unreviewed_answer(monkeypatch, tmp_path):
    service = loop.VehicleBookAgentService(retriever=FakeRetriever(), log_path=tmp_path / "log.jsonl")
    monkeypatch.setattr(service, "_model", lambda: "test-model")
    monkeypatch.setattr(loop, "RUN_TIMEOUT_SECONDS", 0.05)
    calls = []
    async def run(agent, prompt, **kwargs):
        calls.append(agent)
        await kwargs["hooks"].on_llm_start(SimpleNamespace(context=kwargs["context"]), agent, None, [])
        if len(calls) == 1:
            loop.AgentToolRuntime(kwargs["context"]).search("测试")
            return SimpleNamespace(final_output=grounded(), raw_responses=[object()])
        await asyncio.sleep(1)
        raise AssertionError("review must be cancelled")
    monkeypatch.setattr(loop.Runner, "run", run)
    result = asyncio.run(service.answer("测试", "model3-rwd-cn-current"))
    assert result.fallback_reason == "timeout"
    assert result.citations == ()
    assert "可信测试证据" not in result.answer
    log = json.loads((tmp_path / "log.jsonl").read_text(encoding="utf-8"))
    assert log["model_turns"] == 2


def test_review_cannot_introduce_unseen_citation(monkeypatch, tmp_path):
    wrong = grounded().model_copy(update={"cited_chunk_ids": ["foreign"]})
    service, calls = setup_service(monkeypatch, tmp_path, [grounded(), wrong, wrong])
    result = asyncio.run(service.answer("测试", "model3-rwd-cn-current"))
    assert result.citations == ()
    assert result.fallback_reason == "invalid_final_output"


def test_reviewed_operation_survives_overconfident_boundary(monkeypatch, tmp_path):
    candidate = grounded().model_copy(update={"boundary_result": "supported"})
    service, _ = setup_service(monkeypatch, tmp_path, [candidate, candidate])
    result = asyncio.run(service.answer("如何操作？", "model3-rwd-cn-current"))
    assert result.fallback_reason is None
    assert result.boundary_result == "unconfirmed"
    assert result.citations


def test_review_composes_canonical_segments(monkeypatch, tmp_path):
    reviewed = grounded(source="mixed", supplement="一般来说，先确认需求再选择。")
    reviewed = reviewed.model_copy(update={"answer": "不同排版的冗余全文"})
    service, _ = setup_service(monkeypatch, tmp_path, [grounded(), reviewed])
    result = asyncio.run(service.answer("测试", "model3-rwd-cn-current"))
    assert result.answer_source == "mixed"
    assert result.answer == "这是可信测试证据。\n\n一般来说，先确认需求再选择。"


def test_cross_vehicle_evidence_rejected_before_review(monkeypatch, tmp_path):
    from dataclasses import replace
    from test_vehicle_book_agent_loop import _retrieved
    ctx = _context()
    item = _retrieved("model3-lr-awd-cn-current", "chunk-1")
    ctx.run_state.evidence_ledger["chunk-1"] = item
    ctx.run_state.evidence_roles["chunk-1"] = "capability"
    with pytest.raises(loop.AgentOutputValidationError, match="bound profile"):
        loop.VehicleBookAgentService._validated_result(grounded(), ctx, draft=True)
    ctx.run_state.evidence_ledger["chunk-1"] = replace(item, chunk=replace(item.chunk,
        metadata=replace(item.chunk.metadata, vehicle_id=ctx.profile.vehicle_id, market="US")))
    with pytest.raises(loop.AgentOutputValidationError, match="bound profile"):
        loop.VehicleBookAgentService._validated_result(grounded(), ctx, draft=True)


def test_mixed_history_survives_restart(tmp_path):
    from cockpit_agent.vehicle_book.sqlite_session import SQLiteSessionStore
    path = tmp_path / "sessions.db"
    store = SQLiteSessionStore(path)
    cid = store.resolve(None, "model3-rwd-cn-current")
    store.append(cid, "问题", "官方段落。\n\n一般来说……", answer_source="mixed")
    restored = SQLiteSessionStore(path).snapshot(cid)
    assert restored.recent_turns[0].answer_source == "mixed"


def test_normal_charging_cannot_retrieve_emergency_procedure_via_rewritten_query():
    from dataclasses import replace
    from test_vehicle_book_agent_loop import _retrieved
    ctx = _context()
    item = _retrieved(ctx.profile.vehicle_id, "emergency")
    item = replace(item, chunk=replace(item.chunk, section="异常情况下手动释放充电电缆"))
    ctx.retriever = SimpleNamespace(retrieve=lambda *a, **kw: (item,))
    ctx.current_question = "怎么停止充电？"
    assert loop.AgentToolRuntime(ctx).search("应急释放拉索")["results"] == []
    assert ctx.run_state.evidence_ledger == {}
    ctx.current_question = "常规方法都失败了，如何手动释放？"
    assert loop.AgentToolRuntime(ctx).search("手动释放充电电缆")["results"]


def test_charging_source_is_shared_and_emergency_is_separate():
    from cockpit_agent.vehicle_book.ingest import KnowledgeIngestor
    ingestor = KnowledgeIngestor()
    for profile in ingestor.vehicle_registry.list_profiles():
        if profile.oem != "Tesla":
            continue
        chunks = ingestor.ingest(profile.vehicle_id)
        normal = next(c for c in chunks if c.section == "停止充电与正常拔出连接器")
        emergency = next(c for c in chunks if c.section == "异常情况下手动释放充电电缆")
        assert normal.metadata.retrieved_at == "2026-10-18"
        assert "向下拉" not in normal.text
        assert "禁止操作" in emergency.text
        assert "不得一边拉索一边拔" in emergency.text
