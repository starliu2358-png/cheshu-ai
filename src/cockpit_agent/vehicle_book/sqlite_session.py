"""Durable, single-user conversation metadata, history, and Agent context."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from .session import InMemorySessionStore, SessionState, SessionTurn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteSessionStore(InMemorySessionStore):
    """Keep the existing context behavior while persisting every mutation."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._persistence_lock = RLock()
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=10)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                vehicle_id TEXT NOT NULL,
                title TEXT,
                pinned INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                state_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS turns (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                question TEXT NOT NULL,
                answer_id TEXT NOT NULL UNIQUE,
                answer_json TEXT NOT NULL,
                feedback TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_turns_conversation
                ON turns(conversation_id, created_at);
            CREATE TABLE IF NOT EXISTS pending_turns (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                question TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('processing', 'interrupted')),
                created_at TEXT NOT NULL
            );
            """
        )
        with self._db:
            self._db.execute(
                "UPDATE pending_turns SET status = 'interrupted' WHERE status = 'processing'"
            )
        for row in self._db.execute("SELECT id, state_json FROM conversations"):
            payload = json.loads(row["state_json"])
            payload["clarification_options"] = tuple(payload.get("clarification_options", ()))
            payload["recent_turns"] = [
                SessionTurn(
                    question=turn["question"],
                    answer=turn["answer"],
                    cited_chunk_ids=tuple(turn.get("cited_chunk_ids", ())),
                    boundary_result=turn.get("boundary_result"),
                    answer_source=turn.get("answer_source", "official"),
                )
                for turn in payload.get("recent_turns", [])
            ]
            self._items[row["id"]] = SessionState(**payload)

    def _write_state(self, state: SessionState) -> None:
        timestamp = _now()
        self._db.execute(
            """INSERT INTO conversations
               (id, vehicle_id, created_at, updated_at, state_json)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 updated_at = excluded.updated_at,
                 state_json = excluded.state_json""",
            (
                state.conversation_id,
                state.vehicle_id,
                timestamp,
                timestamp,
                json.dumps(asdict(state), ensure_ascii=False),
            ),
        )

    def _save_state(self, conversation_id: str) -> None:
        with self._db:
            self._write_state(self.snapshot(conversation_id))

    def resolve(self, conversation_id: str | None, vehicle_id: str) -> str:
        with self._persistence_lock:
            resolved = super().resolve(conversation_id, vehicle_id)
            if conversation_id is None:
                self._save_state(resolved)
            return resolved

    def append(self, conversation_id: str, question: str, answer: str, **kwargs: Any) -> None:
        with self._persistence_lock:
            super().append(conversation_id, question, answer, **kwargs)
            self._save_state(conversation_id)

    def delete(self, conversation_id: str) -> None:
        with self._persistence_lock:
            super().delete(conversation_id)
            with self._db:
                self._db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def update_metadata(
        self, conversation_id: str, *, title: str | None = None, pinned: bool | None = None
    ) -> dict[str, Any]:
        with self._persistence_lock:
            if title is None and pinned is None:
                raise ValueError("至少需要修改一个字段。")
            if title is not None:
                title = title.strip()
                if not 1 <= len(title) <= 60:
                    raise ValueError("对话名称须为 1–60 个字符。")
            row = self._db.execute(
                "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if row is None:
                raise KeyError(conversation_id)
            assignments = ["updated_at = ?"]
            values: list[Any] = [_now()]
            if title is not None:
                assignments.append("title = ?")
                values.append(title)
            if pinned is not None:
                assignments.append("pinned = ?")
                values.append(int(pinned))
            values.append(conversation_id)
            with self._db:
                self._db.execute(
                    f"UPDATE conversations SET {', '.join(assignments)} WHERE id = ?", values
                )
            return self.get_conversation(conversation_id, include_turns=False)

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._persistence_lock:
            rows = self._db.execute(
                """SELECT id, vehicle_id, title, pinned, created_at, updated_at
                   FROM conversations ORDER BY pinned DESC, updated_at DESC"""
            ).fetchall()
            return [self._summary(row) for row in rows]

    @staticmethod
    def _summary(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "conversationId": row["id"],
            "trimId": row["vehicle_id"],
            "title": row["title"],
            "pinned": bool(row["pinned"]),
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    def get_conversation(
        self, conversation_id: str, *, include_turns: bool = True
    ) -> dict[str, Any]:
        with self._persistence_lock:
            row = self._db.execute(
                """SELECT id, vehicle_id, title, pinned, created_at, updated_at
                   FROM conversations WHERE id = ?""",
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(conversation_id)
            result = self._summary(row)
            if include_turns:
                turns = self._db.execute(
                    """SELECT id, question, answer_json, feedback, created_at FROM turns
                       WHERE conversation_id = ? ORDER BY created_at, rowid""",
                    (conversation_id,),
                ).fetchall()
                complete = [
                    {
                        "id": turn["id"],
                        "question": turn["question"],
                        "answer": json.loads(turn["answer_json"]),
                        "feedback": turn["feedback"],
                        "status": "completed",
                        "createdAt": turn["created_at"],
                    }
                    for turn in turns
                ]
                pending = self._db.execute(
                    """SELECT id, question, status, created_at FROM pending_turns
                       WHERE conversation_id = ? ORDER BY created_at, rowid""",
                    (conversation_id,),
                ).fetchall()
                complete.extend(
                    {
                        "id": turn["id"],
                        "question": turn["question"],
                        "answer": None,
                        "feedback": None,
                        "status": turn["status"],
                        "createdAt": turn["created_at"],
                    }
                    for turn in pending
                )
                result["turns"] = sorted(complete, key=lambda turn: (turn["createdAt"], turn["id"]))
            return result

    def begin_turn(self, conversation_id: str, turn_id: str, question: str) -> None:
        with self._persistence_lock, self._db:
            row = self._db.execute(
                "SELECT conversation_id, question, status FROM pending_turns WHERE id = ?",
                (turn_id,),
            ).fetchone()
            if row:
                if (
                    row["conversation_id"] != conversation_id
                    or row["question"] != question
                    or row["status"] == "processing"
                ):
                    raise ValueError("该问题正在处理或请求 ID 已被其他问题使用。")
                self._db.execute(
                    "UPDATE pending_turns SET status = 'processing' WHERE id = ?", (turn_id,)
                )
            else:
                self._db.execute(
                    """INSERT INTO pending_turns
                       (id, conversation_id, question, status, created_at)
                       VALUES (?, ?, ?, 'processing', ?)""",
                    (turn_id, conversation_id, question, _now()),
                )

    def interrupt_turn(self, turn_id: str) -> None:
        with self._persistence_lock, self._db:
            self._db.execute(
                "UPDATE pending_turns SET status = 'interrupted' WHERE id = ?", (turn_id,)
            )

    def append_completed_answer(
        self,
        conversation_id: str,
        turn_id: str,
        question: str,
        answer: dict[str, Any],
        **context_fields: Any,
    ) -> None:
        with self._persistence_lock:
            before = self.snapshot(conversation_id)
            try:
                with self._db:
                    super().append(conversation_id, question, answer["answer"], **context_fields)
                    self._write_state(self.snapshot(conversation_id))
                    pending = self._db.execute(
                        "SELECT created_at FROM pending_turns WHERE id = ?", (turn_id,)
                    ).fetchone()
                    self._db.execute(
                        """INSERT INTO turns
                           (id, conversation_id, question, answer_id, answer_json, created_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            turn_id,
                            conversation_id,
                            question,
                            answer["answerId"],
                            json.dumps(answer, ensure_ascii=False),
                            pending["created_at"] if pending else _now(),
                        ),
                    )
                    self._db.execute("DELETE FROM pending_turns WHERE id = ?", (turn_id,))
            except Exception:
                with self._lock:
                    self._items[conversation_id] = before
                raise

    def has_answer(self, answer_id: str) -> bool:
        with self._persistence_lock:
            return (
                self._db.execute("SELECT 1 FROM turns WHERE answer_id = ?", (answer_id,)).fetchone()
                is not None
            )

    def answer_for_turn(
        self, conversation_id: str, turn_id: str, question: str
    ) -> dict[str, Any] | None:
        with self._persistence_lock:
            row = self._db.execute(
                "SELECT question, answer_json FROM turns WHERE conversation_id = ? AND id = ?",
                (conversation_id, turn_id),
            ).fetchone()
            if row and row["question"] != question:
                raise ValueError("同一请求 ID 不能用于不同问题。")
            return json.loads(row["answer_json"]) if row else None

    def record_feedback(self, answer_id: str, value: str) -> None:
        with self._persistence_lock, self._db:
            self._db.execute(
                "UPDATE turns SET feedback = ? WHERE answer_id = ?", (value, answer_id)
            )

    def create_conversation(self, vehicle_id: str) -> dict[str, Any]:
        conversation_id = self.resolve(None, vehicle_id)
        return self.get_conversation(conversation_id, include_turns=False)

    def close(self) -> None:
        with self._persistence_lock:
            self._db.close()
