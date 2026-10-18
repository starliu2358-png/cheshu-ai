from __future__ import annotations

from typing import Protocol

from .answerer import (
    AnswerDraft,
    AnswerDraftError,
    DeepSeekAnswerGenerator,
    EvidenceAnswerer,
    GroundedAnswerer,
)
from .hybrid_retriever import build_retriever_from_env
from .registry import RegistryError, VehicleRegistry
from .schemas import AnswerResult, Citation, RetrievedChunk

FALLBACK_ANSWER = "这点我暂时没在当前车型的资料中找到明确说明，还不能确认。"


def fallback_answer(reason: str) -> str:
    """Keep unavailable evidence distinct from a failed request."""
    if reason == "timeout":
        return "这次查询花的时间有些长，暂时没能完成。请稍后再试。"
    if reason in {"model_turn_limit", "retrieval_limit"}:
        return "这次查询暂时还没能确认答案。你可以补充具体功能或使用场景，我再帮你查。"
    if reason in {"no_retrieval_results", "insufficient_evidence"}:
        return FALLBACK_ANSWER
    return "这次没能整理出可靠的回答，请稍后再试。"


class EvidenceRetriever(Protocol):
    def retrieve(
        self,
        vehicle_id: str,
        query: str,
        top_k: int = 2,
        *,
        include_pre_release: bool = False,
    ) -> tuple[RetrievedChunk, ...]: ...


class DraftAnswerer(Protocol):
    def generate_draft(
        self,
        vehicle_id: str,
        question: str,
        evidence: tuple[RetrievedChunk, ...],
    ) -> AnswerDraft: ...


class VehicleBookService:
    """Orchestrate exact-profile retrieval and grounded answer construction."""

    def __init__(
        self,
        vehicle_registry: VehicleRegistry | None = None,
        retriever: EvidenceRetriever | None = None,
        answerer: DraftAnswerer | None = None,
    ) -> None:
        self.vehicle_registry = vehicle_registry or VehicleRegistry()
        self.retriever = retriever or build_retriever_from_env(
            vehicle_registry=self.vehicle_registry,
        )
        self.answerer = answerer or self._default_answerer()

    @staticmethod
    def _default_answerer() -> DraftAnswerer:
        try:
            return GroundedAnswerer(DeepSeekAnswerGenerator.from_env())
        except ValueError as exc:
            if "DEEPSEEK_API_KEY" not in str(exc):
                raise
            return EvidenceAnswerer()

    def answer(
        self,
        question: str,
        vehicle_id: str,
        top_k: int = 4,
        *,
        include_pre_release: bool = False,
        context: str | None = None,
    ) -> AnswerResult:
        profile = self.vehicle_registry.resolve(vehicle_id)
        if profile.release_status == "pre_release" and not include_pre_release:
            raise RegistryError(
                f"Vehicle '{profile.vehicle_id}' is pre_release; "
                "pass include_pre_release=True to retrieve its documents"
            )

        boundary_result = self.preflight_boundary(question, profile.vehicle_id)
        if boundary_result is not None:
            return boundary_result

        contextual_question = (
            f"以下是同一段对话的最近上下文：\n{context}\n当前问题：{question}"
            if context
            else question
        )
        evidence = self.retriever.retrieve(
            profile.vehicle_id,
            contextual_question,
            top_k=top_k,
            include_pre_release=include_pre_release,
        )
        if not evidence:
            return self._fallback(
                profile.vehicle_id,
                evidence,
                reason="no_retrieval_results",
            )

        if any(
            result.chunk.metadata.vehicle_id != profile.vehicle_id
            for result in evidence
        ):
            return self._fallback(
                profile.vehicle_id,
                (),
                reason="invalid_retrieval_evidence",
            )

        try:
            draft = self.answerer.generate_draft(
                profile.vehicle_id,
                contextual_question,
                evidence,
            )
        except AnswerDraftError as exc:
            return self._fallback(
                profile.vehicle_id,
                evidence,
                reason=self._draft_error_reason(exc),
            )

        if draft.insufficient_evidence:
            return self._fallback(
                profile.vehicle_id,
                evidence,
                reason="insufficient_evidence",
            )

        evidence_by_id = {
            result.chunk.chunk_id: result
            for result in evidence
        }
        if not draft.cited_chunk_ids:
            return self._fallback(
                profile.vehicle_id,
                evidence,
                reason="missing_citation_ids",
            )

        try:
            matched = tuple(
                evidence_by_id[chunk_id]
                for chunk_id in draft.cited_chunk_ids
            )
        except KeyError:
            return self._fallback(
                profile.vehicle_id,
                evidence,
                reason="unknown_citation_ids",
            )

        citations = tuple(
            Citation.from_retrieved(result)
            for result in self._preferred_citation_evidence(matched)
        )
        return AnswerResult(
            answer=draft.answer,
            vehicle_id=profile.vehicle_id,
            citations=citations,
            evidence=evidence,
            boundary_result=self._classify_boundary(contextual_question),
        )

    @classmethod
    def preflight_boundary(
        cls, question: str, vehicle_id: str
    ) -> AnswerResult | None:
        """Apply deterministic product boundaries before RAG or Agent execution."""
        return cls._scope_boundary(question, vehicle_id) or cls._availability_boundary(
            question, vehicle_id
        )

    @staticmethod
    def _fallback(
        vehicle_id: str,
        evidence: tuple[RetrievedChunk, ...],
        *,
        reason: str,
    ) -> AnswerResult:
        return AnswerResult(
            answer=fallback_answer(reason),
            vehicle_id=vehicle_id,
            citations=(),
            evidence=evidence,
            fallback_reason=reason,
        )

    @staticmethod
    def _preferred_citation_evidence(
        cited_evidence: tuple[RetrievedChunk, ...],
    ) -> tuple[RetrievedChunk, ...]:
        """Prefer detailed source material over a redundant configuration summary.

        Configuration chunks help retrieval and answer generation, but a detailed
        manual or ADAS section is the more precise provenance when both are cited.
        Keep configuration evidence when it is the only source the draft selected.
        """
        detailed = tuple(
            result
            for result in cited_evidence
            if result.chunk.metadata.content_type != "configuration"
        )
        return detailed or cited_evidence

    @staticmethod
    def _classify_boundary(question: str) -> str | None:
        """Label only explicit availability questions with safely known outcomes."""
        availability_markers = ("是否", "有没有", "有无", "具备", "支持", "能否")
        if not any(marker in question for marker in availability_markers):
            return None

        normalized = question.upper()
        if any(term in normalized for term in ("LCC", "NOA", "FSD")) or any(
            term in question for term in ("辅助转向", "自动辅助变道", "智能召唤")
        ):
            return "unconfirmed"
        if "主动巡航" in question or "自动紧急制动" in question:
            return "supported"
        return None

    @staticmethod
    def _scope_boundary(question: str, vehicle_id: str) -> AnswerResult | None:
        if any(marker in question for marker in ("怎么", "如何", "步骤", "方法", "哪里查看", "在哪看")):
            # Asking how to operate/check is not a request to actuate/read telemetry.
            return None
        realtime_terms = ("现在电量", "当前电量", "车辆在哪", "当前位置", "现在车速", "当前车速", "车门关好")
        control_verbs = ("帮我", "替我", "直接", "现在")
        control_targets = ("打开空调", "关闭空调", "开窗", "关窗", "解锁", "锁车", "打开后备箱", "座椅加热")
        if any(term in question for term in realtime_terms):
            answer = "我目前无法查看车辆的实时电量、位置、车速或车门状态。你可以在车辆触摸屏或 Tesla 官方应用中查看。"
        elif any(verb in question for verb in control_verbs) and any(
            target in question for target in control_targets
        ):
            answer = "我不能直接替你操作车辆，不过可以帮你查找手册里的操作方法。"
        else:
            return None
        return AnswerResult(
            answer=answer,
            vehicle_id=vehicle_id,
            citations=(),
            evidence=(),
            boundary_result="unsupported",
        )

    @staticmethod
    def _availability_boundary(question: str, vehicle_id: str) -> AnswerResult | None:
        availability_markers = ("是否", "有没有", "有无", "具备", "支持", "能否")
        normalized = question.upper()
        uncertain = any(term in normalized for term in ("LCC", "NOA", "FSD")) or any(
            term in question for term in ("辅助转向", "自动辅助变道", "智能召唤")
        )
        if not uncertain or not any(marker in question for marker in availability_markers):
            return None
        detail = (
            "LCC 在 Tesla 官方资料中通常对应辅助转向；"
            if "LCC" in normalized
            else ""
        )
        return AnswerResult(
            answer=f"{detail}这项功能是否可用，还需要结合车辆配置、选装和软件情况确认。你可以查看车辆触摸屏和 Tesla 官方账户中的配置；仅凭当前选择的车型，我还不能确定。",
            vehicle_id=vehicle_id,
            citations=(),
            evidence=(),
            boundary_result="unconfirmed",
        )

    @staticmethod
    def _draft_error_reason(error: AnswerDraftError) -> str:
        message = str(error)
        if "not supplied as evidence" in message:
            return "unknown_citation_ids"
        if "at least one citation" in message:
            return "missing_citation_ids"
        return "malformed_model_output"
