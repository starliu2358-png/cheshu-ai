from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Protocol
from uuid import uuid4

RECENT_TURN_LIMIT = 3
COMPRESSION_TURN_THRESHOLD = 6
COMPRESSION_CHARACTER_THRESHOLD = 3000

FOLLOW_UP_MARKERS = (
    "它",
    "这个",
    "该功能",
    "那",
    "呢",
    "不插",
    "那么",
    "具体",
    "有什么区别",
    "前者",
    "后者",
    "入门配置",
    "后排",
    "刚才", "出处", "来源", "章节", "原文",
)

FEATURE_KEYWORDS = (
    "自动辅助转向", "辅助转向", "主动巡航", "自动紧急制动",
    "蓝牙", "赛道模式", "导航", "软件更新",
    "充电",
    "哨兵模式",
    "前备箱",
    "后备箱",
    "座椅",
    "空调",
    "续航",
    "音响",
    "辅助驾驶",
    "自动驾驶",
    "车灯",
    "雨刷",
    "钥匙",
    "轮胎",
    "摄像头",
)


@dataclass(slots=True)
class SessionTurn:
    question: str
    answer: str
    cited_chunk_ids: tuple[str, ...] = ()
    boundary_result: str | None = None
    answer_source: str = "official"


@dataclass(slots=True)
class SessionState:
    conversation_id: str
    vehicle_id: str
    current_topic: str | None = None
    current_feature: str | None = None
    confirmed_entities: dict[str, str] = field(default_factory=dict)
    unresolved_question: str | None = None
    pending_clarification: str | None = None
    clarification_options: tuple[str, ...] = ()
    recent_turns: list[SessionTurn] = field(default_factory=list)
    memory_summary: str | None = None
    total_turns: int = 0


class SessionStore(Protocol):
    def resolve(self, conversation_id: str | None, vehicle_id: str) -> str: ...

    def context_for(
        self, conversation_id: str, question: str, *, include_unrelated: bool = False
    ) -> str | None: ...

    def append(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        *,
        cited_chunk_ids: tuple[str, ...] = (),
        boundary_result: str | None = None,
        answer_source: str = "official",
        unresolved_question: str | None = None,
        clarification_question: str | None = None,
        clarification_options: tuple[str, ...] = (),
    ) -> None: ...

    def delete(self, conversation_id: str) -> None: ...


def _feature_from_text(text: str) -> str | None:
    return next((keyword for keyword in FEATURE_KEYWORDS if keyword in text), None)


def _mentioned_trims(text: str) -> set[str]:
    """Resolve only explicit names of the four supported trims; never change binding."""
    remaining = text.lower()
    found = set()
    for canonical, aliases in (
        ("Performance 高性能全轮驱动版", ("performance", "高性能全轮驱动版", "高性能版")),
        ("长续航全轮驱动版", ("长续航全轮驱动版", "长续航全轮驱动", "长续航全驱", "lr awd")),
        ("长续航后轮驱动版", ("长续航后轮驱动版", "长续航后轮驱动", "长续航后驱", "lr rwd")),
        ("后轮驱动版", ("后轮驱动版", "后驱版", "rwd")),
    ):
        for alias in aliases:
            if alias in remaining:
                found.add(canonical)
                remaining = remaining.replace(alias, "")
    return found


def _summary_line(turn: SessionTurn) -> str:
    conclusion = " ".join(turn.answer.split())[:240]
    citations = ",".join(turn.cited_chunk_ids) or "无"
    return f"用户问题={turn.question[:120]}；来源={turn.answer_source}；结论={conclusion}；引用={citations}"


class InMemorySessionStore:
    """Thread-safe structured session state with deterministic compression."""

    def __init__(
        self,
        *,
        recent_turn_limit: int = RECENT_TURN_LIMIT,
        compression_turn_threshold: int = COMPRESSION_TURN_THRESHOLD,
        compression_character_threshold: int = COMPRESSION_CHARACTER_THRESHOLD,
    ) -> None:
        self.recent_turn_limit = recent_turn_limit
        self.compression_turn_threshold = compression_turn_threshold
        self.compression_character_threshold = compression_character_threshold
        self._lock = Lock()
        self._items: dict[str, SessionState] = {}

    def resolve(self, conversation_id: str | None, vehicle_id: str) -> str:
        with self._lock:
            if conversation_id is None:
                conversation_id = str(uuid4())
                self._items[conversation_id] = SessionState(
                    conversation_id=conversation_id,
                    vehicle_id=vehicle_id,
                    confirmed_entities={"vehicle_id": vehicle_id},
                )
                return conversation_id
            state = self._items.get(conversation_id)
            if state is None:
                raise KeyError("未找到该对话，请新建对话后重试。")
            if state.vehicle_id != vehicle_id:
                raise ValueError("同一对话不能切换车型配置，请新建对话。")
            return conversation_id

    @staticmethod
    def _is_follow_up(question: str, state: SessionState) -> bool:
        if any(marker in question for marker in ("换个话题", "换一个话题", "不聊这个", "先不说这个")):
            return False
        feature = _feature_from_text(question)
        if feature and state.current_feature and feature != state.current_feature:
            return False
        if state.pending_clarification:
            return True
        if not state.recent_turns:
            return False
        if any(marker in question for marker in FOLLOW_UP_MARKERS):
            return True
        normalized = "".join(question.lower().split()).strip("？?。！!，, ")
        previous = state.recent_turns[-1]
        previous_text = "".join((previous.question + previous.answer).lower().split())
        return 2 <= len(normalized) <= 36 and normalized in previous_text

    @staticmethod
    def _standalone_query(question: str, state: SessionState) -> str:
        feature = state.current_feature
        if not feature:
            return question
        if feature == "哨兵模式" and any(token in question.lower() for token in ("usb", "u盘")):
            if question.strip("？?。 ") == "USB 设备":
                return "哨兵模式录像对 USB 存储设备有什么要求？"
            return f"哨兵模式录像的 USB 存储设备：{question}"
        if question.startswith("它"):
            return feature + question[1:]
        return f"关于{feature}，{question}"

    def context_for(
        self, conversation_id: str, question: str, *, include_unrelated: bool = False
    ) -> str | None:
        with self._lock:
            state = self._items[conversation_id]
            if not state.recent_turns and not state.memory_summary:
                return None
            follows_prior = self._is_follow_up(question, state)
            if not follows_prior and not include_unrelated:
                return None
            follow_up = follows_prior and not state.pending_clarification
            explicit_feature = _feature_from_text(question)
            mentioned_trims = _mentioned_trims(question)
            referenced_trim = next(iter(mentioned_trims)) if len(mentioned_trims) == 1 else None
            if not mentioned_trims and any(word in question for word in ("它", "这个配置", "这款", "那款")):
                referenced_trim = state.confirmed_entities.get("discussed_trim")
            payload = {
                "referenced_trim": referenced_trim,
                "context_relation": "follow_up" if follows_prior else "new_or_unclear",
                "explicit_current_feature": explicit_feature,
                "current_question": question,
                "clarification_reply": question if state.pending_clarification else None,
                "vehicle_id": state.vehicle_id,
                "current_topic": state.current_topic,
                "current_feature": state.current_feature,
                "confirmed_entities": state.confirmed_entities,
                "unresolved_question": state.unresolved_question,
                "pending_clarification": state.pending_clarification,
                "clarification_options": state.clarification_options,
                "follow_up_focus": question if follow_up else None,
                "standalone_query": (
                    self._standalone_query(question, state) if follow_up else None
                ),
                "previous_answer": state.recent_turns[-1].answer,
                "memory_summary": state.memory_summary,
                "recent_turns": [asdict(turn) for turn in state.recent_turns],
            }
            return json.dumps(payload, ensure_ascii=False)

    def append(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        *,
        cited_chunk_ids: tuple[str, ...] = (),
        boundary_result: str | None = None,
        answer_source: str = "official",
        unresolved_question: str | None = None,
        clarification_question: str | None = None,
        clarification_options: tuple[str, ...] = (),
    ) -> None:
        with self._lock:
            state = self._items[conversation_id]
            if not clarification_question:
                mentioned_trims = _mentioned_trims(question)
                if len(mentioned_trims) == 1:
                    state.confirmed_entities["discussed_trim"] = next(iter(mentioned_trims))
                elif len(mentioned_trims) > 1 or any(word in question for word in ("换个话题", "不聊这个")):
                    state.confirmed_entities.pop("discussed_trim", None)
                feature = _feature_from_text(question)
                if feature:
                    state.current_feature = feature
                    state.current_topic = feature
                    state.confirmed_entities["feature"] = feature
                elif not self._is_follow_up(question, state):
                    # Model-generated answer text must not resurrect a stale user topic.
                    state.current_feature = None
                    state.current_topic = question[:120]
                    state.confirmed_entities.pop("feature", None)
            state.unresolved_question = unresolved_question
            state.pending_clarification = clarification_question
            state.clarification_options = clarification_options if clarification_question else ()
            state.recent_turns.append(
                SessionTurn(
                    question=question,
                    answer=answer,
                    cited_chunk_ids=cited_chunk_ids,
                    boundary_result=boundary_result,
                    answer_source=answer_source,
                )
            )
            state.total_turns += 1
            self._compress_if_needed(state)

    def _compress_if_needed(self, state: SessionState) -> None:
        raw_size = sum(
            len(turn.question) + len(turn.answer) for turn in state.recent_turns
        )
        if (
            len(state.recent_turns) <= self.compression_turn_threshold
            and raw_size <= self.compression_character_threshold
        ):
            return
        older = state.recent_turns[: -self.recent_turn_limit]
        retained = state.recent_turns[-self.recent_turn_limit :]
        summary_parts = [state.memory_summary] if state.memory_summary else []
        summary_parts.extend(_summary_line(turn) for turn in older)
        state.memory_summary = "\n".join(summary_parts)[-self.compression_character_threshold :]
        state.recent_turns = retained

    def delete(self, conversation_id: str) -> None:
        with self._lock:
            self._items.pop(conversation_id, None)

    def snapshot(self, conversation_id: str) -> SessionState:
        """Return a detached state snapshot for tests and diagnostics."""

        with self._lock:
            state = self._items[conversation_id]
            return SessionState(
                conversation_id=state.conversation_id,
                vehicle_id=state.vehicle_id,
                current_topic=state.current_topic,
                current_feature=state.current_feature,
                confirmed_entities=dict(state.confirmed_entities),
                unresolved_question=state.unresolved_question,
                pending_clarification=state.pending_clarification,
                clarification_options=state.clarification_options,
                recent_turns=[SessionTurn(**asdict(turn)) for turn in state.recent_turns],
                memory_summary=state.memory_summary,
                total_turns=state.total_turns,
            )
