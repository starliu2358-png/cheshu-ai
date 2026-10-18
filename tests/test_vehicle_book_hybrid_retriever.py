from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cockpit_agent.vehicle_book.hybrid_retriever import (
    FaissIndexStore,
    HybridVehicleBookRetriever,
    VectorVehicleBookRetriever,
    build_retriever_from_env,
    knowledge_checksum,
)
from cockpit_agent.vehicle_book.registry import RegistryError, VehicleRegistry
from cockpit_agent.vehicle_book.retriever import VehicleBookRetriever
from cockpit_agent.vehicle_book.schemas import (
    DocumentMetadata,
    KnowledgeChunk,
    RetrievedChunk,
)


class FakeEmbedder:
    model_id = "test/embedding"

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        if "充电" in text or "补能" in text:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "哨兵" in text or "监控" in text:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def encode_documents(self, texts):
        return np.stack([self._vector(text) for text in texts])

    def encode_query(self, text):
        return self._vector(text)


class MemoryIndexStore:
    def __init__(self) -> None:
        self.builds = 0

    def get_or_build(self, *, vehicle_id, model_id, chunks, embeddings):
        self.builds += 1
        return embeddings()

    def search(self, index, query, top_k):
        scores = index @ query.reshape(-1, 1)
        positions = np.argsort(-scores[:, 0])[:top_k]
        return scores[positions, 0], positions


def _chunk(chunk_id: str, vehicle_id: str = "vehicle-a") -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        section=chunk_id,
        text=chunk_id,
        metadata=DocumentMetadata(
            document_id=f"doc-{chunk_id}",
            vehicle_id=vehicle_id,
            title="Test",
            content_type="owner_manual",
            document_version="v1",
            source_path=Path("data/test.md"),
            status="published",
            market="CN",
        ),
    )


class StaticRetriever:
    def __init__(self, chunks: list[KnowledgeChunk], score: float = 1.0) -> None:
        self.chunks = chunks
        self.score = score

    def retrieve(self, vehicle_id, query, top_k=2, *, include_pre_release=False):
        return tuple(
            RetrievedChunk(chunk=chunk, score=self.score, rank=rank)
            for rank, chunk in enumerate(self.chunks[:top_k], start=1)
        )


def test_vector_retrieval_recalls_semantic_charging_query() -> None:
    store = MemoryIndexStore()
    retriever = VectorVehicleBookRetriever(
        embedder=FakeEmbedder(),
        index_store=store,
        min_score=0.8,
    )

    results = retriever.retrieve("model3-rwd-cn-current", "补能口怎么开启？", top_k=2)

    assert results
    assert results[0].chunk.chunk_id == ("tesla_model3_rwd_cn_charging_2024plus:打开充电端口")
    assert all(item.chunk.metadata.vehicle_id == "model3-rwd-cn-current" for item in results)


def test_vector_retrieval_preserves_pre_release_gate() -> None:
    retriever = VectorVehicleBookRetriever(embedder=FakeEmbedder(), index_store=MemoryIndexStore())

    with pytest.raises(RegistryError, match="include_pre_release=True"):
        retriever.retrieve("alpha_aster_x1_max_v2", "LCC")


def test_both_retrievers_filter_wrong_market_and_version() -> None:
    registry = VehicleRegistry()
    source = _chunk("充电接口", "model3-rwd-cn-current")
    wrong_market = KnowledgeChunk(
        chunk_id=source.chunk_id,
        section=source.section,
        text=source.text,
        metadata=DocumentMetadata(
            document_id=source.metadata.document_id,
            vehicle_id=source.metadata.vehicle_id,
            title=source.metadata.title,
            content_type=source.metadata.content_type,
            document_version=source.metadata.document_version,
            source_path=source.metadata.source_path,
            status="published",
            market="US",
        ),
    )
    alpha = _chunk("LCC", "alpha_aster_x1_max_v1")
    wrong_version = replace(alpha, metadata=replace(alpha.metadata, document_version="2.0"))

    class Ingestor:
        vehicle_registry = registry

        def __init__(self, chunk):
            self.chunk = chunk

        def ingest(self, vehicle_id):
            return (self.chunk,)

    for vehicle_id, query, chunk in (
        ("model3-rwd-cn-current", "充电接口", wrong_market),
        ("alpha_aster_x1_max_v1", "LCC", wrong_version),
    ):
        ingestor = Ingestor(chunk)
        assert VehicleBookRetriever(
            vehicle_registry=registry, ingestor=ingestor
        ).retrieve(vehicle_id, query) == ()
        assert VectorVehicleBookRetriever(
            vehicle_registry=registry,
            ingestor=ingestor,
            embedder=FakeEmbedder(),
            index_store=MemoryIndexStore(),
        ).retrieve(vehicle_id, query) == ()


def test_rrf_rewards_chunks_returned_by_both_retrievers() -> None:
    shared = _chunk("shared")
    lexical_only = _chunk("lexical")
    vector_only = _chunk("vector")
    retriever = HybridVehicleBookRetriever(
        lexical=StaticRetriever([lexical_only, shared]),  # type: ignore[arg-type]
        vector=StaticRetriever([shared, vector_only]),  # type: ignore[arg-type]
        rrf_k=60,
        candidate_multiplier=2,
    )

    results = retriever.retrieve("vehicle-a", "query", top_k=3)

    assert [item.chunk.chunk_id for item in results] == [
        "shared",
        "lexical",
        "vector",
    ]
    assert [item.rank for item in results] == [1, 2, 3]


def test_semantic_only_results_require_stronger_confidence() -> None:
    chunk = _chunk("semantic")
    weak = HybridVehicleBookRetriever(
        lexical=StaticRetriever([]),  # type: ignore[arg-type]
        vector=StaticRetriever([chunk], score=0.49),  # type: ignore[arg-type]
    )
    strong = HybridVehicleBookRetriever(
        lexical=StaticRetriever([]),  # type: ignore[arg-type]
        vector=StaticRetriever([chunk], score=0.75),  # type: ignore[arg-type]
    )

    assert weak.retrieve("vehicle-a", "query") == ()
    assert [item.chunk.chunk_id for item in strong.retrieve("vehicle-a", "query")] == [
        "semantic"
    ]


def test_reranker_cannot_replace_candidate_evidence() -> None:
    original = _chunk("original")

    class ForgingReranker:
        def rerank(self, query, candidates, top_k):
            forged = replace(
                candidates[0],
                chunk=replace(
                    original,
                    metadata=replace(original.metadata, vehicle_id="other-vehicle"),
                ),
            )
            return (forged,)

    retriever = HybridVehicleBookRetriever(
        lexical=StaticRetriever([original]),  # type: ignore[arg-type]
        vector=StaticRetriever([]),  # type: ignore[arg-type]
        reranker=ForgingReranker(),
    )

    result = retriever.retrieve("vehicle-a", "query")
    assert result[0].chunk.metadata.vehicle_id == "vehicle-a"


def test_knowledge_checksum_changes_with_content_and_is_order_independent() -> None:
    first = _chunk("first")
    second = _chunk("second")
    changed = KnowledgeChunk(
        chunk_id=second.chunk_id,
        section=second.section,
        text="changed",
        metadata=second.metadata,
    )

    assert knowledge_checksum([first, second]) == knowledge_checksum([second, first])
    assert knowledge_checksum([first, second]) != knowledge_checksum([first, changed])


def test_faiss_index_reuses_valid_file_and_rebuilds_changed_or_corrupt_data(
    tmp_path: Path,
) -> None:
    chunks = [_chunk("one"), _chunk("two")]
    calls = 0

    def embeddings() -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def build(store: FaissIndexStore, contents: list[KnowledgeChunk]):
        return store.get_or_build(
            vehicle_id="vehicle-a",
            model_id="test/embedding",
            chunks=contents,
            embeddings=embeddings,
        )

    index = build(FaissIndexStore(tmp_path), chunks)
    assert index.ntotal == 2
    assert calls == 1
    assert build(FaissIndexStore(tmp_path), chunks).ntotal == 2
    assert calls == 1

    changed = KnowledgeChunk(
        chunk_id=chunks[1].chunk_id,
        section=chunks[1].section,
        text="new content",
        metadata=chunks[1].metadata,
    )
    assert build(FaissIndexStore(tmp_path), [chunks[0], changed]).ntotal == 2
    assert calls == 2

    index_path = next(tmp_path.glob("*.faiss"))
    index_path.write_bytes(b"damaged index")
    assert build(FaissIndexStore(tmp_path), [chunks[0], changed]).ntotal == 2
    assert calls == 3


def test_retriever_factory_defaults_to_bm25(monkeypatch) -> None:
    monkeypatch.delenv("CHESHU_RETRIEVAL_MODE", raising=False)
    assert isinstance(build_retriever_from_env(), VehicleBookRetriever)


def test_retriever_factory_rejects_unknown_mode(monkeypatch) -> None:
    monkeypatch.setenv("CHESHU_RETRIEVAL_MODE", "unknown")
    with pytest.raises(ValueError, match="bm25.*hybrid"):
        build_retriever_from_env()
