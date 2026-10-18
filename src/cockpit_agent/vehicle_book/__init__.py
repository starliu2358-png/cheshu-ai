"""Vehicle-aware knowledge models and local registries."""

from .evaluation import (
    RetrievalCaseResult,
    RetrievalEvalCase,
    RetrievalEvalSummary,
    RetrievalEvaluationError,
    RetrievalEvaluator,
    evaluate_retrieval,
    load_retrieval_eval_cases,
    summarize_retrieval_results,
    validate_relevant_chunks,
)
from .registry import DocumentRegistry, RegistryError, VehicleRegistry
from .schemas import (
    AnswerDraft,
    AnswerResult,
    Citation,
    DocumentMetadata,
    KnowledgeChunk,
    RetrievedChunk,
    VehicleProfile,
)

__all__ = [
    "AnswerDraft",
    "AnswerResult",
    "Citation",
    "DocumentMetadata",
    "DocumentRegistry",
    "KnowledgeChunk",
    "RegistryError",
    "RetrievalCaseResult",
    "RetrievalEvalCase",
    "RetrievalEvalSummary",
    "RetrievalEvaluationError",
    "RetrievalEvaluator",
    "RetrievedChunk",
    "VehicleProfile",
    "VehicleRegistry",
    "evaluate_retrieval",
    "load_retrieval_eval_cases",
    "summarize_retrieval_results",
    "validate_relevant_chunks",
]
