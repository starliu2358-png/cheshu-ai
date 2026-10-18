"""Deterministic, replayable evaluation for Vehicle Book answers."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .ingest import KnowledgeIngestor
from .registry import REPOSITORY_ROOT, RegistryError, VehicleRegistry
from .schemas import AnswerResult, RetrievedChunk, VehicleProfile

DEFAULT_ANSWER_EVAL_PATH = (
    REPOSITORY_ROOT / "data" / "eval" / "vehicle_book_answer_eval.jsonl"
)
VALID_FALLBACK_REASONS = frozenset(
    {
        "no_retrieval_results",
        "invalid_retrieval_evidence",
        "insufficient_evidence",
        "missing_citation_ids",
        "unknown_citation_ids",
        "malformed_model_output",
    }
)


class AnswerEvaluationError(ValueError):
    """Raised when answer-evaluation input or injected results are invalid."""


def _non_empty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnswerEvaluationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _text_list(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AnswerEvaluationError(f"{field_name} must be a JSON array")
    items = tuple(
        _non_empty_text(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(items) != len(set(items)):
        raise AnswerEvaluationError(f"{field_name} must not contain duplicates")
    return items


def _keyword_groups(value: object) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list):
        raise AnswerEvaluationError("required_keyword_groups must be a JSON array")
    groups: list[tuple[str, ...]] = []
    for group_index, group in enumerate(value):
        if not isinstance(group, list) or not group:
            raise AnswerEvaluationError(
                f"required_keyword_groups[{group_index}] must be a non-empty JSON array"
            )
        phrases = tuple(
            _non_empty_text(
                phrase,
                f"required_keyword_groups[{group_index}][{phrase_index}]",
            )
            for phrase_index, phrase in enumerate(group)
        )
        if len(phrases) != len(set(phrases)):
            raise AnswerEvaluationError(
                f"required_keyword_groups[{group_index}] must not contain duplicates"
            )
        groups.append(phrases)
    return tuple(groups)


@dataclass(frozen=True, slots=True)
class AnswerEvalCase:
    case_id: str
    vehicle_id: str
    question: str
    include_pre_release: bool
    required_keyword_groups: tuple[tuple[str, ...], ...]
    forbidden_phrases: tuple[str, ...]
    required_citation_chunk_ids: tuple[str, ...]
    allowed_citation_chunk_ids: tuple[str, ...]
    expected_fallback: bool
    allowed_fallback_reasons: tuple[str, ...]
    category: str
    critical: bool

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "case_id",
            "vehicle_id",
            "question",
            "include_pre_release",
            "required_keyword_groups",
            "forbidden_phrases",
            "required_citation_chunk_ids",
            "allowed_citation_chunk_ids",
            "expected_fallback",
            "allowed_fallback_reasons",
            "category",
            "critical",
        }
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnswerEvalCase:
        if not isinstance(data, Mapping):
            raise AnswerEvaluationError("evaluation case must be a JSON object")
        keys = set(data)
        if not all(isinstance(key, str) for key in keys):
            raise AnswerEvaluationError("evaluation case field names must be strings")
        missing = cls._FIELDS - keys
        extra = keys - cls._FIELDS
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append("missing fields: " + ", ".join(sorted(missing)))
            if extra:
                details.append("unknown fields: " + ", ".join(sorted(extra)))
            raise AnswerEvaluationError("; ".join(details))

        for field_name in ("include_pre_release", "expected_fallback", "critical"):
            if type(data[field_name]) is not bool:
                raise AnswerEvaluationError(f"{field_name} must be a boolean")

        required_ids = _text_list(
            data["required_citation_chunk_ids"], "required_citation_chunk_ids"
        )
        allowed_ids = _text_list(
            data["allowed_citation_chunk_ids"], "allowed_citation_chunk_ids"
        )
        fallback_reasons = _text_list(
            data["allowed_fallback_reasons"], "allowed_fallback_reasons"
        )
        unknown_reasons = set(fallback_reasons) - VALID_FALLBACK_REASONS
        if unknown_reasons:
            raise AnswerEvaluationError(
                "allowed_fallback_reasons contains unknown reasons: "
                + ", ".join(sorted(unknown_reasons))
            )

        keyword_groups = _keyword_groups(data["required_keyword_groups"])
        forbidden_phrases = _text_list(data["forbidden_phrases"], "forbidden_phrases")
        expected_fallback = data["expected_fallback"]
        if not set(required_ids).issubset(allowed_ids):
            raise AnswerEvaluationError(
                "required_citation_chunk_ids must be a subset of "
                "allowed_citation_chunk_ids"
            )
        if expected_fallback:
            if keyword_groups or forbidden_phrases or required_ids or allowed_ids:
                raise AnswerEvaluationError(
                    "fallback cases must not define answer phrases or citation chunk IDs"
                )
            if not fallback_reasons:
                raise AnswerEvaluationError(
                    "fallback cases must define allowed_fallback_reasons"
                )
        elif not keyword_groups or not required_ids or fallback_reasons:
            raise AnswerEvaluationError(
                "positive cases require keyword groups and required citations, "
                "and must not define fallback reasons"
            )

        return cls(
            case_id=_non_empty_text(data["case_id"], "case_id"),
            vehicle_id=_non_empty_text(data["vehicle_id"], "vehicle_id"),
            question=_non_empty_text(data["question"], "question"),
            include_pre_release=data["include_pre_release"],
            required_keyword_groups=keyword_groups,
            forbidden_phrases=forbidden_phrases,
            required_citation_chunk_ids=required_ids,
            allowed_citation_chunk_ids=allowed_ids,
            expected_fallback=expected_fallback,
            allowed_fallback_reasons=fallback_reasons,
            category=_non_empty_text(data["category"], "category"),
            critical=data["critical"],
        )


@dataclass(frozen=True, slots=True)
class AnswerCaseResult:
    case: AnswerEvalCase
    answer_result: AnswerResult
    answer_correct: bool | None
    citation_correct: bool
    fallback_correct: bool
    grounded: bool | None
    critical_error: bool
    cross_vehicle_answer_error: bool
    cross_version_answer_error: bool
    missing_keyword_groups: tuple[tuple[str, ...], ...]
    unsupported_keyword_groups: tuple[tuple[str, ...], ...]
    matched_forbidden_phrases: tuple[str, ...]
    missing_required_citation_chunk_ids: tuple[str, ...]
    unexpected_citation_chunk_ids: tuple[str, ...]

    @property
    def case_id(self) -> str:
        return self.case.case_id

    @property
    def category(self) -> str:
        return self.case.category

    @property
    def has_citations(self) -> bool:
        return bool(self.answer_result.citations)

    @property
    def failed(self) -> bool:
        if self.case.expected_fallback:
            return (
                not self.fallback_correct
                or not self.citation_correct
                or self.cross_vehicle_answer_error
                or self.cross_version_answer_error
            )
        return (
            self.answer_correct is not True
            or not self.citation_correct
            or not self.fallback_correct
            or self.grounded is not True
            or self.cross_vehicle_answer_error
            or self.cross_version_answer_error
        )


@dataclass(frozen=True, slots=True)
class AnswerEvalSummary:
    total_cases: int
    positive_cases: int
    fallback_cases: int
    critical_cases: int
    cited_cases: int
    answer_accuracy: float | None
    citation_accuracy: float | None
    fallback_accuracy: float | None
    grounded_answer_rate: float | None
    critical_error_rate: float | None
    cross_vehicle_answer_error_rate: float | None
    cross_version_answer_error_rate: float | None
    cross_vehicle_answer_error_cases: int
    cross_version_answer_error_cases: int
    failed_case_ids: tuple[str, ...]
    results: tuple[AnswerCaseResult, ...] = field(default_factory=tuple)
    category_metrics: Mapping[str, AnswerEvalSummary] = field(default_factory=dict)

    @property
    def cited_case_count(self) -> int:
        return self.cited_cases

    @property
    def cross_vehicle_answer_error_case_count(self) -> int:
        return self.cross_vehicle_answer_error_cases

    @property
    def cross_version_answer_error_case_count(self) -> int:
        return self.cross_version_answer_error_cases


def _dependencies(
    vehicle_registry: VehicleRegistry | None,
    ingestor: KnowledgeIngestor | None,
) -> tuple[VehicleRegistry, KnowledgeIngestor]:
    if ingestor is not None:
        return vehicle_registry or ingestor.vehicle_registry, ingestor
    registry = vehicle_registry or VehicleRegistry()
    return registry, KnowledgeIngestor(vehicle_registry=registry)


def validate_answer_eval_cases(
    cases: Iterable[AnswerEvalCase],
    *,
    vehicle_registry: VehicleRegistry | None = None,
    ingestor: KnowledgeIngestor | None = None,
) -> None:
    """Validate vehicle policy and citation IDs against the exact profile corpus."""
    registry, chunk_ingestor = _dependencies(vehicle_registry, ingestor)
    known_chunks: dict[str, set[str]] = {}
    for case in cases:
        if not set(case.required_citation_chunk_ids).issubset(
            case.allowed_citation_chunk_ids
        ):
            raise AnswerEvaluationError(
                f"case {case.case_id}: required_citation_chunk_ids must be a subset "
                "of allowed_citation_chunk_ids"
            )
        if set(case.allowed_fallback_reasons) - VALID_FALLBACK_REASONS:
            raise AnswerEvaluationError(
                f"case {case.case_id}: allowed_fallback_reasons contains unknown reasons"
            )
        if case.expected_fallback:
            if (
                case.required_keyword_groups
                or case.forbidden_phrases
                or case.required_citation_chunk_ids
                or case.allowed_citation_chunk_ids
                or not case.allowed_fallback_reasons
            ):
                raise AnswerEvaluationError(
                    f"case {case.case_id}: fallback case labels are inconsistent"
                )
        elif (
            not case.required_keyword_groups
            or not case.required_citation_chunk_ids
            or case.allowed_fallback_reasons
        ):
            raise AnswerEvaluationError(
                f"case {case.case_id}: positive case labels are inconsistent"
            )
        try:
            profile = registry.resolve(case.vehicle_id)
            if profile.release_status == "pre_release" and not case.include_pre_release:
                raise AnswerEvaluationError(
                    "pre_release vehicle requires include_pre_release=true"
                )
            if case.vehicle_id not in known_chunks:
                chunks = chunk_ingestor.ingest(profile.vehicle_id)
                chunk_ids = [chunk.chunk_id for chunk in chunks]
                if len(chunk_ids) != len(set(chunk_ids)):
                    raise AnswerEvaluationError(
                        f"duplicate chunk_id in profile {profile.vehicle_id}"
                    )
                known_chunks[case.vehicle_id] = set(chunk_ids)
        except (RegistryError, OSError, UnicodeError) as exc:
            raise AnswerEvaluationError(f"case {case.case_id}: {exc}") from exc

        unknown_ids = set(case.allowed_citation_chunk_ids) - known_chunks[case.vehicle_id]
        if unknown_ids:
            raise AnswerEvaluationError(
                f"case {case.case_id}: allowed citation chunks do not belong to exact "
                f"vehicle profile {case.vehicle_id}: {', '.join(sorted(unknown_ids))}"
            )


def load_answer_eval_cases(
    path: Path | str = DEFAULT_ANSWER_EVAL_PATH,
    *,
    vehicle_registry: VehicleRegistry | None = None,
    ingestor: KnowledgeIngestor | None = None,
) -> tuple[AnswerEvalCase, ...]:
    """Load a strict UTF-8 JSONL Answer Eval data set."""
    eval_path = Path(path)
    cases: list[AnswerEvalCase] = []
    case_ids: set[str] = set()
    try:
        with eval_path.open("r", encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    raise AnswerEvaluationError(
                        f"{eval_path}:{line_number}: blank lines are not allowed"
                    )
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise AnswerEvaluationError(
                        f"{eval_path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                try:
                    case = AnswerEvalCase.from_dict(payload)
                except AnswerEvaluationError as exc:
                    raise AnswerEvaluationError(
                        f"{eval_path}:{line_number}: {exc}"
                    ) from exc
                if case.case_id in case_ids:
                    raise AnswerEvaluationError(
                        f"{eval_path}:{line_number}: duplicate case_id: {case.case_id}"
                    )
                case_ids.add(case.case_id)
                cases.append(case)
    except (OSError, UnicodeError) as exc:
        raise AnswerEvaluationError(f"cannot read evaluation file {eval_path}: {exc}") from exc

    if not cases:
        raise AnswerEvaluationError(f"evaluation file contains no cases: {eval_path}")
    registry, chunk_ingestor = _dependencies(vehicle_registry, ingestor)
    validate_answer_eval_cases(cases, vehicle_registry=registry, ingestor=chunk_ingestor)
    return tuple(cases)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _matching_phrases(groups: Sequence[tuple[str, ...]], text: str) -> tuple[tuple[str, ...], ...]:
    normalized = _normalize(text)
    return tuple(
        group
        for group in groups
        if not any(_normalize(phrase) in normalized for phrase in group)
    )


def _citation_evidence(
    answer_result: AnswerResult,
) -> dict[str, RetrievedChunk]:
    return {item.chunk.chunk_id: item for item in answer_result.evidence}


def _cross_profile_errors(
    answer_result: AnswerResult,
    profile: VehicleProfile,
    case: AnswerEvalCase,
) -> tuple[bool, bool]:
    cross_vehicle = False
    cross_version = False
    for citation in answer_result.citations:
        cross_vehicle = cross_vehicle or citation.vehicle_id != case.vehicle_id
        cross_version = cross_version or citation.document_version != profile.software_version
    return cross_vehicle, cross_version


def _score_case(
    case: AnswerEvalCase,
    answer_result: AnswerResult,
    profile: VehicleProfile,
) -> AnswerCaseResult:
    citation_ids = tuple(citation.chunk_id for citation in answer_result.citations)
    allowed_ids = set(case.allowed_citation_chunk_ids)
    unexpected_ids = tuple(chunk_id for chunk_id in citation_ids if chunk_id not in allowed_ids)
    missing_required_ids = tuple(
        chunk_id for chunk_id in case.required_citation_chunk_ids if chunk_id not in citation_ids
    )
    cross_vehicle, cross_version = _cross_profile_errors(answer_result, profile, case)

    if case.expected_fallback:
        fallback_correct = (
            answer_result.is_fallback
            and answer_result.fallback_reason in case.allowed_fallback_reasons
        )
        citation_correct = not answer_result.citations
        answer_correct: bool | None = None
        grounded: bool | None = None
        missing_groups: tuple[tuple[str, ...], ...] = ()
        unsupported_groups: tuple[tuple[str, ...], ...] = ()
        forbidden = ()
    else:
        fallback_correct = not answer_result.is_fallback
        missing_groups = _matching_phrases(case.required_keyword_groups, answer_result.answer)
        normalized_answer = _normalize(answer_result.answer)
        forbidden = tuple(
            phrase
            for phrase in case.forbidden_phrases
            if _normalize(phrase) in normalized_answer
        )
        answer_correct = not answer_result.is_fallback and not missing_groups and not forbidden
        citation_correct = (
            bool(answer_result.citations)
            and not unexpected_ids
            and not missing_required_ids
            and not cross_vehicle
            and not cross_version
        )
        evidence_by_id = _citation_evidence(answer_result)
        cited_text = "\n".join(
            evidence_by_id[citation.chunk_id].chunk.text
            for citation in answer_result.citations
            if citation.chunk_id in evidence_by_id
        )
        unsupported_groups = _matching_phrases(case.required_keyword_groups, cited_text)
        grounded = (
            answer_correct
            and citation_correct
            and not unsupported_groups
            and len(evidence_by_id) >= len(answer_result.citations)
        )

    critical_error = (
        cross_vehicle
        or cross_version
        or (
            case.critical
            and (
                (not fallback_correct)
                or (answer_correct is False)
                or not citation_correct
            )
        )
    )
    return AnswerCaseResult(
        case=case,
        answer_result=answer_result,
        answer_correct=answer_correct,
        citation_correct=citation_correct,
        fallback_correct=fallback_correct,
        grounded=grounded,
        critical_error=critical_error,
        cross_vehicle_answer_error=cross_vehicle,
        cross_version_answer_error=cross_version,
        missing_keyword_groups=missing_groups,
        unsupported_keyword_groups=unsupported_groups,
        matched_forbidden_phrases=forbidden,
        missing_required_citation_chunk_ids=missing_required_ids,
        unexpected_citation_chunk_ids=unexpected_ids,
    )


def evaluate_answer_results(
    cases: Iterable[AnswerEvalCase],
    results_by_case_id: Mapping[str, AnswerResult],
    *,
    vehicle_registry: VehicleRegistry | None = None,
    ingestor: KnowledgeIngestor | None = None,
) -> AnswerEvalSummary:
    """Score injected ``AnswerResult`` objects without invoking a model or service."""
    case_tuple = tuple(cases)
    if len({case.case_id for case in case_tuple}) != len(case_tuple):
        raise AnswerEvaluationError("duplicate case_id in evaluation cases")
    registry, chunk_ingestor = _dependencies(vehicle_registry, ingestor)
    validate_answer_eval_cases(
        case_tuple, vehicle_registry=registry, ingestor=chunk_ingestor
    )

    scored: list[AnswerCaseResult] = []
    for case in case_tuple:
        try:
            answer_result = results_by_case_id[case.case_id]
        except KeyError as exc:
            raise AnswerEvaluationError(f"missing AnswerResult for case {case.case_id}") from exc
        if not isinstance(answer_result, AnswerResult):
            raise AnswerEvaluationError(
                f"case {case.case_id}: mapped value must be an AnswerResult"
            )
        scored.append(_score_case(case, answer_result, registry.resolve(case.vehicle_id)))
    return summarize_answer_results(scored)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize(
    results: Sequence[AnswerCaseResult], *, include_categories: bool
) -> AnswerEvalSummary:
    positive = [result for result in results if not result.case.expected_fallback]
    fallback = [result for result in results if result.case.expected_fallback]
    critical = [result for result in results if result.case.critical]
    cited = [result for result in results if result.has_citations]
    categories: dict[str, AnswerEvalSummary] = {}
    if include_categories:
        grouped: defaultdict[str, list[AnswerCaseResult]] = defaultdict(list)
        for result in results:
            grouped[result.category].append(result)
        categories = {
            category: _summarize(grouped[category], include_categories=False)
            for category in sorted(grouped)
        }

    return AnswerEvalSummary(
        total_cases=len(results),
        positive_cases=len(positive),
        fallback_cases=len(fallback),
        critical_cases=len(critical),
        cited_cases=len(cited),
        answer_accuracy=_mean(
            [float(result.answer_correct) for result in positive if result.answer_correct is not None]
        ),
        citation_accuracy=_mean([float(result.citation_correct) for result in positive]),
        fallback_accuracy=_mean([float(result.fallback_correct) for result in results]),
        grounded_answer_rate=_mean(
            [float(result.grounded) for result in positive if result.grounded is not None]
        ),
        critical_error_rate=_mean([float(result.critical_error) for result in critical]),
        cross_vehicle_answer_error_rate=_mean(
            [float(result.cross_vehicle_answer_error) for result in cited]
        ),
        cross_version_answer_error_rate=_mean(
            [float(result.cross_version_answer_error) for result in cited]
        ),
        cross_vehicle_answer_error_cases=sum(
            result.cross_vehicle_answer_error for result in cited
        ),
        cross_version_answer_error_cases=sum(
            result.cross_version_answer_error for result in cited
        ),
        failed_case_ids=tuple(result.case_id for result in results if result.failed),
        results=tuple(results),
        category_metrics=categories,
    )


def summarize_answer_results(
    results: Iterable[AnswerCaseResult],
) -> AnswerEvalSummary:
    """Aggregate deterministic case scores overall and by category."""
    return _summarize(tuple(results), include_categories=True)
