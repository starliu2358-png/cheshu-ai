"""Deterministic, model-free evaluation for Vehicle Book retrieval.

The evaluator deliberately operates on ``RetrievedChunk`` objects rather than
answers.  It can therefore be run in CI without credentials or network access.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

from .ingest import KnowledgeIngestor
from .registry import REPOSITORY_ROOT, RegistryError, VehicleRegistry
from .schemas import RetrievedChunk, VehicleProfile

DEFAULT_RETRIEVAL_EVAL_PATH = (
    REPOSITORY_ROOT / "data" / "eval" / "vehicle_book_retrieval_eval.jsonl"
)


class RetrievalEvaluationError(ValueError):
    """Raised for an invalid retrieval evaluation set or result."""


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrievalEvaluationError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class RetrievalEvalCase:
    case_id: str
    vehicle_id: str
    question: str
    relevant_chunk_ids: tuple[str, ...]
    include_pre_release: bool
    category: str
    critical: bool
    expected_no_answer: bool

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "case_id",
            "vehicle_id",
            "question",
            "relevant_chunk_ids",
            "include_pre_release",
            "category",
            "critical",
            "expected_no_answer",
        }
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetrievalEvalCase:
        """Parse one case, rejecting missing, additional, or loosely typed fields."""
        if not isinstance(data, Mapping):
            raise RetrievalEvaluationError("evaluation case must be a JSON object")

        keys = set(data)
        if not all(isinstance(key, str) for key in keys):
            raise RetrievalEvaluationError("evaluation case field names must be strings")
        missing = cls._FIELDS - keys
        extra = keys - cls._FIELDS
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append("missing fields: " + ", ".join(sorted(missing)))
            if extra:
                details.append("unknown fields: " + ", ".join(sorted(extra)))
            raise RetrievalEvaluationError("; ".join(details))

        relevant = data["relevant_chunk_ids"]
        if not isinstance(relevant, list):
            raise RetrievalEvaluationError("relevant_chunk_ids must be a JSON array")
        relevant_ids = tuple(
            _non_empty_string(value, f"relevant_chunk_ids[{index}]")
            for index, value in enumerate(relevant)
        )
        if len(relevant_ids) != len(set(relevant_ids)):
            raise RetrievalEvaluationError("relevant_chunk_ids must not contain duplicates")

        for name in ("include_pre_release", "critical", "expected_no_answer"):
            if type(data[name]) is not bool:
                raise RetrievalEvaluationError(f"{name} must be a boolean")

        expected_no_answer = data["expected_no_answer"]
        if expected_no_answer and relevant_ids:
            raise RetrievalEvaluationError(
                "expected_no_answer cases must have no relevant_chunk_ids"
            )
        if not expected_no_answer and not relevant_ids:
            raise RetrievalEvaluationError(
                "positive cases must contain at least one relevant_chunk_id"
            )

        return cls(
            case_id=_non_empty_string(data["case_id"], "case_id"),
            vehicle_id=_non_empty_string(data["vehicle_id"], "vehicle_id"),
            question=_non_empty_string(data["question"], "question"),
            relevant_chunk_ids=relevant_ids,
            include_pre_release=data["include_pre_release"],
            category=_non_empty_string(data["category"], "category"),
            critical=data["critical"],
            expected_no_answer=expected_no_answer,
        )


@dataclass(frozen=True, slots=True)
class RetrievalCaseResult:
    case: RetrievalEvalCase
    returned_chunk_ids: tuple[str, ...]
    hit_at_1: bool | None
    hit_at_3: bool | None
    recall_at_3: float | None
    reciprocal_rank: float | None
    cross_vehicle_errors: int
    cross_version_errors: int
    no_answer_false_positive: bool | None

    @property
    def case_id(self) -> str:
        return self.case.case_id

    @property
    def category(self) -> str:
        return self.case.category

    @property
    def returned_count(self) -> int:
        return len(self.returned_chunk_ids)

    @property
    def retrieved_chunk_ids(self) -> tuple[str, ...]:
        return self.returned_chunk_ids

    @property
    def cross_vehicle_error_rate(self) -> float:
        return self.cross_vehicle_errors / self.returned_count if self.returned_count else 0.0

    @property
    def cross_version_error_rate(self) -> float:
        return self.cross_version_errors / self.returned_count if self.returned_count else 0.0


@dataclass(frozen=True, slots=True)
class RetrievalEvalSummary:
    case_count: int
    positive_case_count: int
    no_answer_case_count: int
    critical_positive_case_count: int
    returned_result_count: int
    hit_at_1: float | None
    hit_at_3: float | None
    critical_hit_at_3: float | None
    recall_at_3: float | None
    mrr: float | None
    cross_vehicle_error_rate: float | None
    cross_version_error_rate: float | None
    no_answer_false_positive_rate: float | None
    results: tuple[RetrievalCaseResult, ...] = field(default_factory=tuple)
    category_summaries: Mapping[str, RetrievalEvalSummary] = field(default_factory=dict)

    @property
    def mean_reciprocal_rank(self) -> float | None:
        return self.mrr


class RetrievalEvaluatorRetriever(Protocol):
    def retrieve(
        self,
        vehicle_id: str,
        query: str,
        top_k: int = 2,
        *,
        include_pre_release: bool = False,
    ) -> Sequence[RetrievedChunk]: ...


def load_retrieval_eval_cases(
    path: Path | str = DEFAULT_RETRIEVAL_EVAL_PATH,
    *,
    vehicle_registry: VehicleRegistry | None = None,
    ingestor: KnowledgeIngestor | None = None,
) -> tuple[RetrievalEvalCase, ...]:
    """Load and fully validate a UTF-8 JSONL retrieval evaluation set."""
    eval_path = Path(path)
    cases: list[RetrievalEvalCase] = []
    seen_ids: set[str] = set()
    try:
        with eval_path.open("r", encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    raise RetrievalEvaluationError(
                        f"{eval_path}:{line_number}: blank lines are not allowed"
                    )
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise RetrievalEvaluationError(
                        f"{eval_path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                try:
                    case = RetrievalEvalCase.from_dict(payload)
                except RetrievalEvaluationError as exc:
                    raise RetrievalEvaluationError(f"{eval_path}:{line_number}: {exc}") from exc
                if case.case_id in seen_ids:
                    raise RetrievalEvaluationError(
                        f"{eval_path}:{line_number}: duplicate case_id: {case.case_id}"
                    )
                seen_ids.add(case.case_id)
                cases.append(case)
    except (OSError, UnicodeError) as exc:
        raise RetrievalEvaluationError(f"cannot read evaluation file {eval_path}: {exc}") from exc

    if not cases:
        raise RetrievalEvaluationError(f"evaluation file contains no cases: {eval_path}")

    registry, chunk_ingestor = _dependencies(vehicle_registry, ingestor)
    validate_relevant_chunks(cases, vehicle_registry=registry, ingestor=chunk_ingestor)
    return tuple(cases)


def _dependencies(
    vehicle_registry: VehicleRegistry | None,
    ingestor: KnowledgeIngestor | None,
) -> tuple[VehicleRegistry, KnowledgeIngestor]:
    if ingestor is not None:
        registry = vehicle_registry or ingestor.vehicle_registry
        return registry, ingestor
    registry = vehicle_registry or VehicleRegistry()
    return registry, KnowledgeIngestor(vehicle_registry=registry)


def validate_relevant_chunks(
    cases: Iterable[RetrievalEvalCase],
    *,
    vehicle_registry: VehicleRegistry | None = None,
    ingestor: KnowledgeIngestor | None = None,
) -> None:
    """Ensure every gold chunk exists in the published corpus of the exact profile."""
    registry, chunk_ingestor = _dependencies(vehicle_registry, ingestor)
    chunks_by_vehicle: dict[str, dict[str, str]] = {}

    for case in cases:
        try:
            profile = registry.resolve(case.vehicle_id)
            if profile.release_status == "pre_release" and not case.include_pre_release:
                raise RetrievalEvaluationError(
                    "pre_release vehicle requires include_pre_release=true"
                )
            if case.vehicle_id not in chunks_by_vehicle:
                chunks = chunk_ingestor.ingest(profile.vehicle_id)
                index: dict[str, str] = {}
                for chunk in chunks:
                    if chunk.chunk_id in index:
                        raise RetrievalEvaluationError(
                            f"duplicate chunk_id in profile {profile.vehicle_id}: {chunk.chunk_id}"
                        )
                    index[chunk.chunk_id] = chunk.metadata.vehicle_id
                chunks_by_vehicle[case.vehicle_id] = index
        except (RegistryError, OSError, UnicodeError) as exc:
            raise RetrievalEvaluationError(f"case {case.case_id}: {exc}") from exc

        available = chunks_by_vehicle[case.vehicle_id]
        for chunk_id in case.relevant_chunk_ids:
            if chunk_id not in available:
                raise RetrievalEvaluationError(
                    f"case {case.case_id}: relevant chunk {chunk_id!r} does not exist "
                    f"in exact vehicle profile {case.vehicle_id!r}"
                )
            if available[chunk_id] != case.vehicle_id:
                raise RetrievalEvaluationError(
                    f"case {case.case_id}: relevant chunk {chunk_id!r} belongs to "
                    f"{available[chunk_id]!r}, not {case.vehicle_id!r}"
                )


def _isolation_errors(
    result: RetrievedChunk,
    expected: VehicleProfile,
    registry: VehicleRegistry,
) -> tuple[bool, bool]:
    metadata = result.chunk.metadata
    try:
        actual = registry.resolve(metadata.vehicle_id)
    except RegistryError:
        return True, False

    cross_vehicle = metadata.vehicle_id != expected.vehicle_id and (
        actual.oem != expected.oem
        or actual.model != expected.model
        or actual.trim != expected.trim
        or actual.software_version == expected.software_version
    )
    cross_version = not cross_vehicle and (
        actual.software_version != expected.software_version
        or (
            expected.software_version is not None
            and metadata.document_version != expected.software_version
        )
    )
    return cross_vehicle, cross_version


def evaluate_retrieval(
    cases: Iterable[RetrievalEvalCase],
    retriever: RetrievalEvaluatorRetriever,
    *,
    top_k: int = 3,
    vehicle_registry: VehicleRegistry | None = None,
    ingestor: KnowledgeIngestor | None = None,
) -> RetrievalEvalSummary:
    """Evaluate ``retriever`` once per case, preserving input and result order."""
    if type(top_k) is not int or top_k < 3:
        raise RetrievalEvaluationError("top_k must be an integer greater than or equal to 3")
    case_tuple = tuple(cases)
    if len({case.case_id for case in case_tuple}) != len(case_tuple):
        raise RetrievalEvaluationError("duplicate case_id in evaluation cases")

    registry, chunk_ingestor = _dependencies(vehicle_registry, ingestor)
    validate_relevant_chunks(case_tuple, vehicle_registry=registry, ingestor=chunk_ingestor)

    evaluated: list[RetrievalCaseResult] = []
    for case in case_tuple:
        profile = registry.resolve(case.vehicle_id)
        returned = tuple(
            retriever.retrieve(
                case.vehicle_id,
                case.question,
                top_k=top_k,
                include_pre_release=case.include_pre_release,
            )
        )
        if not all(isinstance(item, RetrievedChunk) for item in returned):
            raise RetrievalEvaluationError(
                f"case {case.case_id}: retriever must return RetrievedChunk values"
            )
        returned_ids = tuple(item.chunk.chunk_id for item in returned)
        top_three = returned_ids[:3]
        relevant = set(case.relevant_chunk_ids)
        first_relevant_rank = next(
            (rank for rank, chunk_id in enumerate(returned_ids, start=1) if chunk_id in relevant),
            None,
        )
        cross_vehicle = 0
        cross_version = 0
        for item in returned:
            vehicle_error, version_error = _isolation_errors(item, profile, registry)
            cross_vehicle += int(vehicle_error)
            cross_version += int(version_error)

        if case.expected_no_answer:
            hit_1 = hit_3 = recall_3 = reciprocal_rank = None
            false_positive: bool | None = bool(returned)
        else:
            hit_1 = bool(returned_ids[:1] and returned_ids[0] in relevant)
            hit_3 = any(chunk_id in relevant for chunk_id in top_three)
            recall_3 = len(relevant.intersection(top_three)) / len(relevant)
            reciprocal_rank = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
            false_positive = None

        evaluated.append(
            RetrievalCaseResult(
                case=case,
                returned_chunk_ids=returned_ids,
                hit_at_1=hit_1,
                hit_at_3=hit_3,
                recall_at_3=recall_3,
                reciprocal_rank=reciprocal_rank,
                cross_vehicle_errors=cross_vehicle,
                cross_version_errors=cross_version,
                no_answer_false_positive=false_positive,
            )
        )

    return summarize_retrieval_results(evaluated)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize(
    results: Sequence[RetrievalCaseResult],
    *,
    include_categories: bool,
) -> RetrievalEvalSummary:
    positive = [result for result in results if not result.case.expected_no_answer]
    no_answer = [result for result in results if result.case.expected_no_answer]
    critical_positive = [result for result in positive if result.case.critical]
    returned_count = sum(result.returned_count for result in results)
    cross_vehicle_count = sum(result.cross_vehicle_errors for result in results)
    cross_version_count = sum(result.cross_version_errors for result in results)

    categories: dict[str, RetrievalEvalSummary] = {}
    if include_categories:
        grouped: defaultdict[str, list[RetrievalCaseResult]] = defaultdict(list)
        for result in results:
            grouped[result.category].append(result)
        categories = {
            category: _summarize(grouped[category], include_categories=False)
            for category in sorted(grouped)
        }

    return RetrievalEvalSummary(
        case_count=len(results),
        positive_case_count=len(positive),
        no_answer_case_count=len(no_answer),
        critical_positive_case_count=len(critical_positive),
        returned_result_count=returned_count,
        hit_at_1=_mean([float(result.hit_at_1) for result in positive]),
        hit_at_3=_mean([float(result.hit_at_3) for result in positive]),
        critical_hit_at_3=_mean(
            [float(result.hit_at_3) for result in critical_positive]
        ),
        recall_at_3=_mean(
            [result.recall_at_3 for result in positive if result.recall_at_3 is not None]
        ),
        mrr=_mean(
            [result.reciprocal_rank for result in positive if result.reciprocal_rank is not None]
        ),
        cross_vehicle_error_rate=(cross_vehicle_count / returned_count if returned_count else 0.0),
        cross_version_error_rate=(cross_version_count / returned_count if returned_count else 0.0),
        no_answer_false_positive_rate=_mean(
            [float(result.no_answer_false_positive) for result in no_answer]
        ),
        results=tuple(results),
        category_summaries=categories,
    )


def summarize_retrieval_results(
    results: Iterable[RetrievalCaseResult],
) -> RetrievalEvalSummary:
    """Aggregate case results overall and by category."""
    return _summarize(tuple(results), include_categories=True)


class RetrievalEvaluator:
    """Convenience wrapper retaining the injected dependencies."""

    def __init__(
        self,
        retriever: RetrievalEvaluatorRetriever,
        *,
        vehicle_registry: VehicleRegistry | None = None,
        ingestor: KnowledgeIngestor | None = None,
        top_k: int = 3,
    ) -> None:
        self.retriever = retriever
        self.vehicle_registry, self.ingestor = _dependencies(vehicle_registry, ingestor)
        self.top_k = top_k

    def evaluate(self, cases: Iterable[RetrievalEvalCase]) -> RetrievalEvalSummary:
        return evaluate_retrieval(
            cases,
            self.retriever,
            top_k=self.top_k,
            vehicle_registry=self.vehicle_registry,
            ingestor=self.ingestor,
        )
