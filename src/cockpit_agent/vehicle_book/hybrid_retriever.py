from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

import numpy as np

from .ingest import KnowledgeIngestor
from .registry import REPOSITORY_ROOT, RegistryError, VehicleRegistry
from .retriever import VehicleBookRetriever
from .schemas import KnowledgeChunk, RetrievedChunk

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_INDEX_DIR = REPOSITORY_ROOT / "outputs" / "indexes" / "vehicle_book"
DEFAULT_VECTOR_MIN_SCORE = 0.65
DEFAULT_SEMANTIC_ONLY_MIN_SCORE = 0.65
DEFAULT_VECTOR_THRESHOLDS_PATH = REPOSITORY_ROOT / "data" / "registry" / "vector_thresholds.json"
DEFAULT_RRF_K = 60
DEFAULT_VECTOR_RRF_WEIGHT = 0.35
INDEX_FORMAT_VERSION = 1
QUERY_ALIASES = {
    "autopilot": "辅助驾驶",
    "bluetooth": "手机 蓝牙 连接",
    "pairing": "配对 连接",
    "补能": "充电",
    "掀开": "打开",
    "口盖": "充电接口盖板",
    "前舱盖": "前备箱",
    "掀开": "打开",
}


class VectorDependencyError(RuntimeError):
    """Raised when V1.2 vector dependencies are not installed."""


class EmbeddingBackend(Protocol):
    @property
    def model_id(self) -> str: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class CandidateReranker(Protocol):
    """Optional V1.2 extension point; disabled until an evaluation gate requires it."""

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> Sequence[RetrievedChunk]: ...


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding backend returned a zero-length vector")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


class BgeEmbeddingBackend:
    """Lazy local Sentence Transformers wrapper for BGE embeddings."""

    QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, model_id: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self._model_id = model_id
        self._model: Any | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise VectorDependencyError(
                    "hybrid retrieval requires the 'retrieval' optional dependencies"
                ) from exc
            self._model = SentenceTransformer(self._model_id)
        return self._model

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        encoder = getattr(model, "encode_document", model.encode)
        return _normalize_rows(
            encoder(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

    def encode_query(self, text: str) -> np.ndarray:
        model = self._load()
        values = model.encode(
            [self.QUERY_PREFIX + text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _normalize_rows(values)[0]


@dataclass(frozen=True, slots=True)
class VectorIndexMetadata:
    format_version: int
    vehicle_id: str
    model_id: str
    knowledge_checksum: str
    dimension: int
    chunk_ids: tuple[str, ...]
    document_versions: tuple[str, ...]
    index_sha256: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VectorIndexMetadata:
        return cls(
            format_version=int(payload["format_version"]),
            vehicle_id=str(payload["vehicle_id"]),
            model_id=str(payload["model_id"]),
            knowledge_checksum=str(payload["knowledge_checksum"]),
            dimension=int(payload["dimension"]),
            chunk_ids=tuple(str(value) for value in payload["chunk_ids"]),
            document_versions=tuple(str(value) for value in payload["document_versions"]),
            index_sha256=str(payload["index_sha256"]),
        )


def knowledge_checksum(chunks: Sequence[KnowledgeChunk]) -> str:
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda item: item.chunk_id):
        metadata = chunk.metadata
        payload = {
            "chunk_id": chunk.chunk_id,
            "section": chunk.section,
            "text": chunk.text,
            "document_id": metadata.document_id,
            "document_version": metadata.document_version,
            "vehicle_id": metadata.vehicle_id,
            "market": metadata.market,
            "status": metadata.status,
        }
        digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "index"


def load_vector_thresholds(path: Path = DEFAULT_VECTOR_THRESHOLDS_PATH) -> dict[str, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        raise TypeError("vector threshold registry must be a JSON object")
    thresholds: dict[str, float] = {}
    for vehicle_id, value in payload.items():
        if not isinstance(vehicle_id, str) or not vehicle_id.strip():
            raise ValueError("vector threshold vehicle IDs must be non-empty strings")
        if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
            raise ValueError("vector thresholds must be numbers between 0 and 1")
        thresholds[vehicle_id] = float(value)
    return thresholds


def expand_lexical_query(query: str) -> str:
    normalized = query.lower()
    additions = [canonical for alias, canonical in QUERY_ALIASES.items() if alias in normalized]
    return " ".join((query, *dict.fromkeys(additions)))


class FaissIndexStore:
    """Build, validate, persist, and search exact inner-product indexes."""

    def __init__(self, index_dir: Path = DEFAULT_INDEX_DIR) -> None:
        self.index_dir = index_dir
        self._lock = RLock()
        self._cache: dict[tuple[str, str, str], Any] = {}

    @staticmethod
    def _faiss() -> Any:
        try:
            import faiss
        except ImportError as exc:
            raise VectorDependencyError(
                "hybrid retrieval requires the 'retrieval' optional dependencies"
            ) from exc
        return faiss

    def _paths(self, vehicle_id: str, model_id: str) -> tuple[Path, Path]:
        stem = f"{_safe_name(vehicle_id)}--{hashlib.sha256(model_id.encode()).hexdigest()[:12]}"
        return self.index_dir / f"{stem}.faiss", self.index_dir / f"{stem}.json"

    def _load_valid(
        self,
        index_path: Path,
        metadata_path: Path,
        expected: VectorIndexMetadata,
    ) -> Any | None:
        if not index_path.is_file() or not metadata_path.is_file():
            return None
        try:
            actual = VectorIndexMetadata.from_dict(
                json.loads(metadata_path.read_text(encoding="utf-8"))
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return None
        comparable = (
            "format_version",
            "vehicle_id",
            "model_id",
            "knowledge_checksum",
            "chunk_ids",
            "document_versions",
        )
        if any(getattr(actual, field) != getattr(expected, field) for field in comparable):
            return None
        if _file_sha256(index_path) != actual.index_sha256:
            return None
        try:
            index = self._faiss().read_index(str(index_path))
        except RuntimeError:
            return None
        if (
            actual.dimension <= 0
            or index.d != actual.dimension
            or index.ntotal != len(actual.chunk_ids)
        ):
            return None
        return index

    def get_or_build(
        self,
        *,
        vehicle_id: str,
        model_id: str,
        chunks: Sequence[KnowledgeChunk],
        embeddings: Callable[[], np.ndarray],
    ) -> Any:
        checksum = knowledge_checksum(chunks)
        cache_key = (vehicle_id, model_id, checksum)
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
            index_path, metadata_path = self._paths(vehicle_id, model_id)
            expected = VectorIndexMetadata(
                format_version=INDEX_FORMAT_VERSION,
                vehicle_id=vehicle_id,
                model_id=model_id,
                knowledge_checksum=checksum,
                dimension=0,
                chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
                document_versions=tuple(chunk.metadata.document_version for chunk in chunks),
                index_sha256="",
            )
            index = self._load_valid(index_path, metadata_path, expected)
            if index is None:
                vectors = _normalize_rows(embeddings())
                if len(vectors) != len(chunks):
                    raise ValueError("embedding count must match chunk count")
                faiss = self._faiss()
                dimension = int(vectors.shape[1])
                index = faiss.IndexFlatIP(dimension)
                index.add(vectors)
                self.index_dir.mkdir(parents=True, exist_ok=True)
                temporary_index = index_path.with_suffix(".faiss.tmp")
                temporary_metadata = metadata_path.with_suffix(".json.tmp")
                faiss.write_index(index, str(temporary_index))
                stored = VectorIndexMetadata(
                    **{
                        **asdict(expected),
                        "dimension": dimension,
                        "index_sha256": _file_sha256(temporary_index),
                    }
                )
                temporary_metadata.write_text(
                    json.dumps(asdict(stored), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                temporary_index.replace(index_path)
                temporary_metadata.replace(metadata_path)
            self._cache[cache_key] = index
            return index

    def search(self, index: Any, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        scores, positions = index.search(_normalize_rows(query), top_k)
        return scores[0], positions[0]


class VectorVehicleBookRetriever:
    """Profile-isolated semantic retrieval backed by BGE and FAISS."""

    def __init__(
        self,
        *,
        vehicle_registry: VehicleRegistry | None = None,
        ingestor: KnowledgeIngestor | None = None,
        embedder: EmbeddingBackend | None = None,
        index_store: FaissIndexStore | None = None,
        min_score: float | None = None,
        score_thresholds: Mapping[str, float] | None = None,
    ) -> None:
        self.vehicle_registry = vehicle_registry or VehicleRegistry()
        self.ingestor = ingestor or KnowledgeIngestor(vehicle_registry=self.vehicle_registry)
        self.embedder = embedder or BgeEmbeddingBackend(
            os.getenv("CHESHU_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        )
        self.index_store = index_store or FaissIndexStore(
            Path(os.getenv("CHESHU_VECTOR_INDEX_DIR", str(DEFAULT_INDEX_DIR)))
        )
        self.min_score = DEFAULT_VECTOR_MIN_SCORE if min_score is None else min_score
        self.score_thresholds = dict(
            load_vector_thresholds()
            if score_thresholds is None and min_score is None
            else score_thresholds or {}
        )

    def _eligible_chunks(
        self, vehicle_id: str, include_pre_release: bool
    ) -> tuple[KnowledgeChunk, ...]:
        profile = self.vehicle_registry.resolve(vehicle_id)
        if profile.release_status == "pre_release" and not include_pre_release:
            raise RegistryError(
                f"Vehicle '{profile.vehicle_id}' is pre_release; "
                "pass include_pre_release=True to retrieve its documents"
            )
        return tuple(
            chunk
            for chunk in self.ingestor.ingest(profile.vehicle_id)
            if chunk.metadata.vehicle_id == profile.vehicle_id
            and chunk.metadata.status == "published"
            and (profile.market is None or chunk.metadata.market == profile.market)
            and (
                profile.software_version is None
                or chunk.metadata.document_version == profile.software_version
            )
        )

    @staticmethod
    def _document_text(chunk: KnowledgeChunk) -> str:
        return f"{chunk.metadata.title}\n{chunk.section}\n{chunk.text}"

    def retrieve(
        self,
        vehicle_id: str,
        query: str,
        top_k: int = 2,
        *,
        include_pre_release: bool = False,
    ) -> tuple[RetrievedChunk, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip():
            return ()
        chunks = self._eligible_chunks(vehicle_id, include_pre_release)
        if not chunks:
            return ()
        index = self._index(vehicle_id, chunks)
        scores, positions = self.index_store.search(
            index,
            self.embedder.encode_query(query),
            min(top_k, len(chunks)),
        )
        results: list[RetrievedChunk] = []
        for score, position in zip(scores, positions, strict=True):
            threshold = self.score_thresholds.get(vehicle_id, self.min_score)
            if position < 0 or float(score) < threshold:
                continue
            chunk = chunks[int(position)]
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(score),
                    rank=len(results) + 1,
                )
            )
        return tuple(results)

    def _index(self, vehicle_id: str, chunks: Sequence[KnowledgeChunk]) -> Any:
        return self.index_store.get_or_build(
            vehicle_id=vehicle_id,
            model_id=self.embedder.model_id,
            chunks=chunks,
            embeddings=lambda: self.embedder.encode_documents(
                [self._document_text(chunk) for chunk in chunks]
            ),
        )

    def build_index(self, vehicle_id: str, *, include_pre_release: bool = False) -> str:
        chunks = self._eligible_chunks(vehicle_id, include_pre_release)
        if not chunks:
            raise ValueError(f"no published chunks available for vehicle '{vehicle_id}'")
        self._index(vehicle_id, chunks)
        return knowledge_checksum(chunks)


class HybridVehicleBookRetriever:
    """Fuse BM25 and semantic rankings with Reciprocal Rank Fusion."""

    def __init__(
        self,
        *,
        lexical: VehicleBookRetriever | None = None,
        vector: VectorVehicleBookRetriever | None = None,
        reranker: CandidateReranker | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        vector_weight: float = DEFAULT_VECTOR_RRF_WEIGHT,
        semantic_only_min_score: float = DEFAULT_SEMANTIC_ONLY_MIN_SCORE,
        candidate_multiplier: int = 4,
    ) -> None:
        if rrf_k <= 0 or vector_weight <= 0 or candidate_multiplier <= 0:
            raise ValueError("rrf_k, vector_weight, and candidate_multiplier must be positive")
        if not 0 <= semantic_only_min_score <= 1:
            raise ValueError("semantic_only_min_score must be between 0 and 1")
        self.lexical = lexical or VehicleBookRetriever()
        self.vector = vector or VectorVehicleBookRetriever(
            vehicle_registry=self.lexical.vehicle_registry,
            ingestor=self.lexical.ingestor,
        )
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.semantic_only_min_score = semantic_only_min_score
        self.candidate_multiplier = candidate_multiplier

    def retrieve(
        self,
        vehicle_id: str,
        query: str,
        top_k: int = 2,
        *,
        include_pre_release: bool = False,
    ) -> tuple[RetrievedChunk, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        candidate_k = max(top_k * self.candidate_multiplier, top_k)
        rankings = (
            self.lexical.retrieve(
                vehicle_id,
                expand_lexical_query(query),
                top_k=candidate_k,
                include_pre_release=include_pre_release,
            ),
            self.vector.retrieve(
                vehicle_id,
                query,
                top_k=candidate_k,
                include_pre_release=include_pre_release,
            ),
        )
        if not rankings[0]:
            rankings = (
                rankings[0],
                tuple(
                    item
                    for item in rankings[1]
                    if item.score >= self.semantic_only_min_score
                ),
            )
        chunks: dict[str, KnowledgeChunk] = {}
        fused_scores: dict[str, float] = {}
        for weight, ranking in zip((1.0, self.vector_weight), rankings, strict=True):
            for item in ranking:
                chunk_id = item.chunk.chunk_id
                chunks[chunk_id] = item.chunk
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + weight / (
                    self.rrf_k + item.rank
                )
        ordered = sorted(
            fused_scores,
            key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
        )[:candidate_k]
        candidates = tuple(
            RetrievedChunk(
                chunk=chunks[chunk_id],
                score=fused_scores[chunk_id],
                rank=rank,
            )
            for rank, chunk_id in enumerate(ordered, start=1)
        )
        if self.reranker is None:
            return candidates[:top_k]
        reranked = tuple(self.reranker.rerank(query, candidates, top_k))
        candidate_by_id = {item.chunk.chunk_id: item for item in candidates}
        returned_ids = [item.chunk.chunk_id for item in reranked]
        if (
            len(reranked) > top_k
            or len(returned_ids) != len(set(returned_ids))
            or any(chunk_id not in candidate_by_id for chunk_id in returned_ids)
        ):
            raise ValueError("reranker returned invalid candidates")
        return tuple(
            RetrievedChunk(
                chunk=candidate_by_id[chunk_id].chunk,
                score=candidate_by_id[chunk_id].score,
                rank=rank,
            )
            for rank, chunk_id in enumerate(returned_ids, start=1)
        )


def build_retriever_from_env(
    *, vehicle_registry: VehicleRegistry | None = None
) -> VehicleBookRetriever | HybridVehicleBookRetriever:
    mode = os.getenv("CHESHU_RETRIEVAL_MODE", "bm25").strip().lower()
    registry = vehicle_registry or VehicleRegistry()
    lexical = VehicleBookRetriever(vehicle_registry=registry)
    if mode == "bm25":
        return lexical
    if mode != "hybrid":
        raise ValueError("CHESHU_RETRIEVAL_MODE must be 'bm25' or 'hybrid'")
    vector = VectorVehicleBookRetriever(
        vehicle_registry=registry,
        ingestor=lexical.ingestor,
        min_score=(
            float(os.environ["CHESHU_VECTOR_MIN_SCORE"])
            if "CHESHU_VECTOR_MIN_SCORE" in os.environ
            else None
        ),
    )
    return HybridVehicleBookRetriever(
        lexical=lexical,
        vector=vector,
        rrf_k=int(os.getenv("CHESHU_RRF_K", str(DEFAULT_RRF_K))),
        vector_weight=float(os.getenv("CHESHU_RRF_VECTOR_WEIGHT", str(DEFAULT_VECTOR_RRF_WEIGHT))),
        semantic_only_min_score=float(
            os.getenv(
                "CHESHU_SEMANTIC_ONLY_MIN_SCORE",
                str(DEFAULT_SEMANTIC_ONLY_MIN_SCORE),
            )
        ),
    )
