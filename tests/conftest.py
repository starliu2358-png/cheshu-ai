from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_local_retrieval_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep repository .env rollout settings out of deterministic unit tests."""
    monkeypatch.setenv("CHESHU_RETRIEVAL_MODE", "bm25")
