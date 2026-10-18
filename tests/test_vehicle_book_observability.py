from __future__ import annotations

import json

from cockpit_agent.vehicle_book.observability import (
    AgentRunLog,
    AgentRunLogger,
    ToolCallLog,
)


def test_agent_run_log_contains_audit_metadata_only(tmp_path) -> None:
    path = tmp_path / "agent.jsonl"
    logger = AgentRunLogger(path)
    logger.record(
        AgentRunLog(
            run_id="run-1",
            vehicle_id="model3-rwd-cn-current",
            conversation_id="conversation-1",
            model_turns=2,
            retrieval_calls=1,
            repair_attempts=0,
            stop_reason="completed",
            duration_ms=42,
            tool_calls=[
                ToolCallLog(
                    tool="search_vehicle_book",
                    query="充电口怎么打开",
                    duration_ms=3,
                    result_chunk_ids=("charging:open",),
                )
            ],
        )
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stop_reason"] == "completed"
    assert payload["tool_calls"][0]["result_chunk_ids"] == ["charging:open"]
    assert "answer" not in payload
    assert "reasoning" not in payload
