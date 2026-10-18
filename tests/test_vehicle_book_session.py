from __future__ import annotations

import json

import pytest

from cockpit_agent.vehicle_book.session import InMemorySessionStore


def test_session_binds_vehicle_and_deleted_id_cannot_be_reused() -> None:
    store = InMemorySessionStore()
    conversation_id = store.resolve(None, "model3-rwd-cn-current")

    assert store.resolve(conversation_id, "model3-rwd-cn-current") == conversation_id
    with pytest.raises(ValueError, match="不能切换车型"):
        store.resolve(conversation_id, "model3-lr-rwd-cn-current")

    store.delete(conversation_id)
    with pytest.raises(KeyError, match="未找到该对话"):
        store.resolve(conversation_id, "model3-rwd-cn-current")


def test_session_context_is_structured_for_follow_up() -> None:
    store = InMemorySessionStore()
    conversation_id = store.resolve(None, "model3-rwd-cn-current")
    store.append(
        conversation_id,
        "哨兵模式怎么使用？",
        "需要启用哨兵模式。",
        cited_chunk_ids=("sentry:usage",),
        boundary_result="supported",
    )

    context = json.loads(store.context_for(conversation_id, "不插 U 盘呢？") or "{}")

    assert context["vehicle_id"] == "model3-rwd-cn-current"
    assert context["current_feature"] == "哨兵模式"
    assert context["confirmed_entities"]["feature"] == "哨兵模式"
    assert context["recent_turns"][0]["cited_chunk_ids"] == ["sentry:usage"]


def test_session_deterministically_compresses_older_turns() -> None:
    store = InMemorySessionStore(
        recent_turn_limit=3,
        compression_turn_threshold=6,
        compression_character_threshold=10_000,
    )
    conversation_id = store.resolve(None, "model3-rwd-cn-current")
    for index in range(7):
        store.append(
            conversation_id,
            f"第 {index} 个充电问题",
            f"第 {index} 个结论",
            cited_chunk_ids=(f"chunk-{index}",),
        )

    state = store.snapshot(conversation_id)

    assert state.total_turns == 7
    assert [turn.question for turn in state.recent_turns] == [
        "第 4 个充电问题",
        "第 5 个充电问题",
        "第 6 个充电问题",
    ]
    assert "chunk-0" in (state.memory_summary or "")
    assert "chunk-3" in (state.memory_summary or "")


def test_unresolved_question_is_kept_as_structured_state() -> None:
    store = InMemorySessionStore()
    conversation_id = store.resolve(None, "model3-rwd-cn-current")
    store.append(
        conversation_id,
        "它支持吗？",
        "当前资料不足。",
        unresolved_question="它支持吗？",
    )

    state = store.snapshot(conversation_id)
    assert state.unresolved_question == "它支持吗？"


def test_usb_follow_up_has_standalone_search_query_and_new_focus() -> None:
    store = InMemorySessionStore()
    conversation_id = store.resolve(None, "model3-rwd-cn-current")
    store.append(
        conversation_id,
        "哨兵模式怎么保存录像？",
        "哨兵模式录像可保存到 USB 设备。",
        cited_chunk_ids=("sentry:recording",),
    )

    context = json.loads(store.context_for(conversation_id, "USB 设备") or "{}")

    assert context["standalone_query"] == "哨兵模式录像对 USB 存储设备有什么要求？"
    assert context["follow_up_focus"] == "USB 设备"
    assert context["previous_answer"] == "哨兵模式录像可保存到 USB 设备。"


def test_unrelated_gibberish_has_context_but_no_forced_rewrite() -> None:
    store = InMemorySessionStore()
    conversation_id = store.resolve(None, "model3-rwd-cn-current")
    store.append(conversation_id, "哨兵模式怎么使用？", "先开启哨兵模式。")

    assert store.context_for(conversation_id, "沙发JFK数据飞轮书法家") is None
    context = json.loads(
        store.context_for(
            conversation_id, "沙发JFK数据飞轮书法家", include_unrelated=True
        ) or "{}"
    )
    assert context["current_feature"] == "哨兵模式"
    assert context["standalone_query"] is None
    assert context["follow_up_focus"] is None


def test_clarification_is_kept_until_user_replies() -> None:
    store = InMemorySessionStore()
    conversation_id = store.resolve(None, "model3-rwd-cn-current")
    store.append(
        conversation_id,
        "沙发JFK数据飞轮书法家",
        "你想了解哪项车辆功能？",
        clarification_question="你想了解哪项车辆功能？",
        clarification_options=("我想了解哨兵模式录像如何保存", "我想了解 USB 设备要求"),
    )

    context = json.loads(
        store.context_for(conversation_id, "我想了解哨兵模式录像如何保存") or "{}"
    )
    assert context["pending_clarification"] == "你想了解哪项车辆功能？"
    assert context["clarification_options"][0] == "我想了解哨兵模式录像如何保存"
    assert context["standalone_query"] is None

    store.append(conversation_id, "我想了解哨兵模式录像如何保存", "可以保存到 USB 设备。")
    assert store.snapshot(conversation_id).pending_clarification is None


@pytest.mark.parametrize("question", ["那空调怎么开呢？", "换个话题，今天星期几？", "那蓝牙怎么连接呢？"])
def test_explicit_new_topic_beats_follow_up_markers_and_pending(question):
    store = InMemorySessionStore()
    cid = store.resolve(None, "model3-rwd-cn-current")
    store.append(cid, "哨兵模式怎么使用？", "你想问什么？", clarification_question="想问录像吗？")
    # Establish an earlier actual topic, as happens in a real multi-turn conversation.
    store.append(cid, "哨兵模式怎么用？", "说明")
    store.append(cid, "它？", "请补充", clarification_question="想问录像吗？")
    context = json.loads(store.context_for(cid, question, include_unrelated=True))
    assert context["context_relation"] == "new_or_unclear"
    assert context["standalone_query"] is None


def test_usb_detail_is_not_replaced_and_new_topic_persists():
    store = InMemorySessionStore()
    cid = store.resolve(None, "model3-rwd-cn-current")
    store.append(cid, "哨兵模式如何录像？", "保存到 USB 设备。")
    query = json.loads(store.context_for(cid, "USB 怎么格式化呢？"))["standalone_query"]
    assert "格式化" in query and "哨兵模式" in query
    store.append(cid, "那蓝牙如何配对呢？", "连接手机")
    assert store.snapshot(cid).current_feature == "蓝牙"
    assert "蓝牙" in json.loads(store.context_for(cid, "它连接不上呢？"))["standalone_query"]
    store.append(cid, "换个话题，今天星期几？", "星期日。哨兵模式可以稍后再聊。")
    assert store.snapshot(cid).current_feature is None


def test_steering_follow_up_and_source_request_are_completed():
    store = InMemorySessionStore()
    cid = store.resolve(None, "model3-rwd-cn-current")
    store.append(cid, "自动辅助转向有哪些限制？", "需保持监督。")
    for question in ["那下雨天呢？", "如果车道线看不清呢？", "刚才是哪个章节？"]:
        query = json.loads(store.context_for(cid, question))["standalone_query"]
        assert "自动辅助转向" in query
        assert question in query


def test_explanation_request_is_not_vehicle_control():
    from cockpit_agent.vehicle_book.service import VehicleBookService
    assert VehicleBookService.preflight_boundary("现在怎么打开空调？", "model3-rwd-cn-current") is None
    assert VehicleBookService.preflight_boundary("帮我直接打开空调", "model3-rwd-cn-current").boundary_result == "unsupported"


def test_comparison_subject_does_not_revert_to_bound_trim():
    store = InMemorySessionStore()
    cid = store.resolve(None, "model3-rwd-cn-current")
    store.append(cid, "后轮驱动版和长续航后驱续航有何区别？", "分别634和830公里。")
    store.append(cid, "那长续航全驱呢？", "18英寸753公里。")
    context = json.loads(store.context_for(cid, "它的音响有几个扬声器？", include_unrelated=True))
    assert context["referenced_trim"] == "长续航全轮驱动版"
    assert context["vehicle_id"] == "model3-rwd-cn-current"
    store.append(cid, "它的音响有几个扬声器？", "15个。")
    store.append(cid, "换个话题，蓝牙怎么连接？", "在控制中配对。")
    assert "discussed_trim" not in store.snapshot(cid).confirmed_entities


def test_acceptance_suite_has_fixed_five_groups_and_four_turn_dialogues():
    import runpy
    from pathlib import Path
    from collections import Counter
    cases = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/run_core_acceptance.py"))["CASES"]
    assert Counter(case[1] for case in cases) == {
        "function": 5, "retrieval": 4, "comparison": 4, "boundary": 5, "multi": 6, "basic": 4,
    }
    assert all(len(case[2]) == 4 for case in cases if case[1] == "multi")
