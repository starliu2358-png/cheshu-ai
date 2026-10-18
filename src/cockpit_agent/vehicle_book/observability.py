from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .registry import REPOSITORY_ROOT

DEFAULT_AGENT_LOG_PATH = (
    REPOSITORY_ROOT / "outputs" / "logs" / "vehicle_book_agent.jsonl"
)
_LOG_LOCK = Lock()


@dataclass(slots=True)
class ToolCallLog:
    tool: str
    query: str
    duration_ms: int
    result_chunk_ids: tuple[str, ...] = ()
    error: str | None = None


@dataclass(slots=True)
class AgentRunLog:
    run_id: str
    vehicle_id: str
    conversation_id: str | None
    model_turns: int
    retrieval_calls: int
    repair_attempts: int
    stop_reason: str
    duration_ms: int
    tool_calls: list[ToolCallLog] = field(default_factory=list)
    evidence_count: int = 0
    model_reported_insufficient: bool = False
    answer_source: str | None = None
    validation_errors: list[str] = field(default_factory=list)
    review_attempts: int = 0
    reviewed_segments: list[dict[str, Any]] = field(default_factory=list)
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class AgentRunLogger:
    """Append audit-safe run metadata without prompts, answers, or chain of thought."""

    def __init__(self, path: Path | None = None) -> None:
        configured = os.getenv("CHESHU_AGENT_LOG_PATH")
        self.path = path or (Path(configured) if configured else DEFAULT_AGENT_LOG_PATH)

    def record(self, event: AgentRunLog) -> None:
        payload: dict[str, Any] = asdict(event)
        try:
            with _LOG_LOCK:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            # Observability must never change the user-facing answer path.
            return
