from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from .vehicle_book.agent_loop import VehicleBookAgentService
from .vehicle_book.hybrid_retriever import HybridVehicleBookRetriever
from .vehicle_book.registry import REPOSITORY_ROOT, RegistryError, VehicleRegistry
from .vehicle_book.service import VehicleBookService
from .vehicle_book.sqlite_session import SQLiteSessionStore

TESLA_CN_TRIM_IDS = (
    "model3-rwd-cn-current",
    "model3-lr-rwd-cn-current",
    "model3-lr-awd-cn-current",
    "model3-performance-awd-cn-current",
)
DEFAULT_FEEDBACK_PATH = REPOSITORY_ROOT / "outputs" / "feedback" / "cheshu_feedback.jsonl"
DEFAULT_SESSION_PATH = REPOSITORY_ROOT / "outputs" / "state" / "cheshu.sqlite3"


class VehicleOption(BaseModel):
    trimId: str
    displayName: str
    model: str


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    trimId: str = Field(min_length=1)
    conversationId: str | None = Field(default=None, min_length=1)
    turnId: str | None = Field(default=None, min_length=1, max_length=100)


class ConversationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trimId: str = Field(min_length=1)


class ConversationUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    pinned: bool | None = None


class CitationResponse(BaseModel):
    id: str
    title: str
    section: str
    url: str
    retrievedAt: str


class AnswerResponse(BaseModel):
    answerSource: str = "official"
    answerId: str
    conversationId: str
    answer: str
    boundaryResult: Literal["supported", "unsupported", "unconfirmed"] | None = None
    citations: list[CitationResponse]
    speechText: str
    fallbackReason: str | None = None
    clarificationOptions: list[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Literal["up", "down"]


class FeedbackResponse(BaseModel):
    answerId: str
    value: Literal["up", "down"]
    recordedAt: str


class FeedbackStore:
    """Append local feedback events while keeping the latest value in memory."""

    def __init__(self, path: Path = DEFAULT_FEEDBACK_PATH) -> None:
        self.path = path
        self._lock = Lock()
        self._latest: dict[str, FeedbackResponse] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.is_file():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                event = FeedbackResponse.model_validate_json(line)
            except ValueError:
                continue
            self._latest[event.answerId] = event

    def record(self, answer_id: str, value: Literal["up", "down"]) -> FeedbackResponse:
        with self._lock:
            current = self._latest.get(answer_id)
            if current is not None and current.value == value:
                return current
            event = FeedbackResponse(
                answerId=answer_id,
                value=value,
                recordedAt=datetime.now(timezone.utc).isoformat(),
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(event.model_dump(), ensure_ascii=False) + "\n")
            self._latest[answer_id] = event
        return event


ConversationStore = SQLiteSessionStore


def create_app(
    *,
    service_factory: Callable[[], VehicleBookService] = VehicleBookService,
    vehicle_registry: VehicleRegistry | None = None,
    feedback_store: FeedbackStore | None = None,
    conversation_store: SQLiteSessionStore | None = None,
    agent_service_factory: Callable[[], VehicleBookAgentService] = VehicleBookAgentService,
    execution_mode: str | None = None,
) -> FastAPI:
    registry = vehicle_registry or VehicleRegistry()
    feedback = feedback_store or FeedbackStore()
    database_path = Path(os.getenv("CHESHU_DB_PATH", str(DEFAULT_SESSION_PATH)))
    if not database_path.is_absolute():
        database_path = REPOSITORY_ROOT / database_path
    conversations = conversation_store or SQLiteSessionStore(database_path)
    selected_mode = (execution_mode or os.getenv("CHESHU_EXECUTION_MODE", "rag")).strip().lower()
    if selected_mode not in {"rag", "agent"}:
        raise ValueError("CHESHU_EXECUTION_MODE must be 'rag' or 'agent'")
    rag_service = service_factory() if selected_mode == "rag" else None
    agent_service = agent_service_factory() if selected_mode == "agent" else None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        active_service = agent_service if selected_mode == "agent" else rag_service
        retriever = getattr(active_service, "retriever", None)
        if isinstance(retriever, HybridVehicleBookRetriever):
            for vehicle_id in TESLA_CN_TRIM_IDS:
                await asyncio.to_thread(retriever.vector.build_index, vehicle_id)
            await asyncio.to_thread(
                retriever.vector.retrieve,
                TESLA_CN_TRIM_IDS[0],
                "充电",
                1,
            )
        yield

    app = FastAPI(title="车书 AI API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "market": "CN"}

    @app.get("/api/v1/vehicles", response_model=list[VehicleOption])
    def list_vehicles() -> list[VehicleOption]:
        options: list[VehicleOption] = []
        for trim_id in TESLA_CN_TRIM_IDS:
            profile = registry.resolve(trim_id)
            if profile.market != "CN" or profile.oem != "Tesla":
                continue
            options.append(
                VehicleOption(
                    trimId=profile.vehicle_id,
                    displayName=profile.display_name or profile.trim,
                    model=profile.model,
                )
            )
        return options

    @app.get("/api/v1/conversations")
    def list_conversations() -> list[dict]:
        return conversations.list_conversations()

    @app.post("/api/v1/conversations", status_code=201)
    def create_conversation(request: ConversationCreateRequest) -> dict:
        if request.trimId not in TESLA_CN_TRIM_IDS:
            raise HTTPException(status_code=404, detail="未找到该中国大陆车型配置。")
        try:
            profile = registry.resolve(request.trimId)
        except RegistryError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if profile.market != "CN" or profile.oem != "Tesla":
            raise HTTPException(status_code=404, detail="未找到该中国大陆车型配置。")
        return conversations.create_conversation(request.trimId)

    @app.get("/api/v1/conversations/{conversation_id}")
    def get_conversation(conversation_id: str) -> dict:
        try:
            return conversations.get_conversation(conversation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="未找到该对话。") from exc

    @app.patch("/api/v1/conversations/{conversation_id}")
    def update_conversation(
        conversation_id: str, request: ConversationUpdateRequest
    ) -> dict:
        try:
            return conversations.update_metadata(
                conversation_id, title=request.title, pinned=request.pinned
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="未找到该对话。") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/answers", response_model=AnswerResponse)
    async def answer(request: AnswerRequest) -> AnswerResponse:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="问题不能为空。")
        if request.trimId not in TESLA_CN_TRIM_IDS:
            raise HTTPException(status_code=404, detail="未找到该中国大陆车型配置。")
        turn_id = request.turnId or str(uuid4())
        pending_started = False
        try:
            profile = registry.resolve(request.trimId)
            if profile.market != "CN" or profile.oem != "Tesla":
                raise HTTPException(status_code=404, detail="未找到该中国大陆车型配置。")
            try:
                conversation_id = conversations.resolve(
                    request.conversationId, profile.vehicle_id
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            if request.turnId:
                try:
                    saved = conversations.answer_for_turn(
                        conversation_id, request.turnId, question
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                if saved is not None:
                    return AnswerResponse.model_validate(saved)
            try:
                conversations.begin_turn(conversation_id, turn_id, question)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            pending_started = True
            context = conversations.context_for(
                conversation_id,
                question,
                include_unrelated=selected_mode == "agent",
            )
            boundary_result = VehicleBookService.preflight_boundary(
                question, profile.vehicle_id
            )
            if boundary_result is not None and not (
                selected_mode == "agent" and boundary_result.boundary_result == "unconfirmed"
            ):
                result = boundary_result
            elif selected_mode == "agent":
                assert agent_service is not None
                result = await agent_service.answer(
                    question,
                    profile.vehicle_id,
                    context=context,
                    conversation_id=conversation_id,
                )
            else:
                assert rag_service is not None
                result = rag_service.answer(
                    question, profile.vehicle_id, context=context
                )
        except RegistryError as exc:
            if pending_started:
                conversations.interrupt_turn(turn_id)
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except BaseException:
            if pending_started:
                conversations.interrupt_turn(turn_id)
            raise

        answer_id = str(uuid4())
        citations = [
            CitationResponse(
                id=citation.document_id,
                title=citation.title,
                section=citation.section,
                url=citation.source_url or "",
                retrievedAt=citation.retrieved_at or "",
            )
            for citation in result.citations
            if citation.source_url
            and citation.market == "CN"
            and citation.source_url.startswith((
                "https://www.tesla.cn/", "https://service.tesla.cn/"
            ))
        ]
        response = AnswerResponse(
            answerId=answer_id,
            conversationId=conversation_id,
            answer=result.answer,
            answerSource=result.answer_source,
            boundaryResult=result.boundary_result,
            citations=citations,
            speechText=result.answer,
            fallbackReason=result.fallback_reason,
            clarificationOptions=list(result.clarification_options),
        )
        try:
            conversations.append_completed_answer(
                conversation_id,
                turn_id,
                question,
                response.model_dump(),
                cited_chunk_ids=tuple(citation.chunk_id for citation in result.citations),
                boundary_result=result.boundary_result,
                answer_source=result.answer_source,
                unresolved_question=(
                    question
                    if result.fallback_reason
                    in {"clarification_required", "insufficient_evidence"}
                    else None
                ),
                clarification_question=(
                    result.answer if result.fallback_reason == "clarification_required" else None
                ),
                clarification_options=result.clarification_options,
            )
        except BaseException:
            conversations.interrupt_turn(turn_id)
            raise
        return response

    @app.delete("/api/v1/conversations/{conversation_id}", status_code=204)
    def delete_conversation(conversation_id: str) -> None:
        conversations.delete(conversation_id)

    @app.post(
        "/api/v1/answers/{answer_id}/feedback",
        response_model=FeedbackResponse,
    )
    def submit_feedback(answer_id: str, request: FeedbackRequest) -> FeedbackResponse:
        if not conversations.has_answer(answer_id):
            raise HTTPException(status_code=404, detail="未找到该回答。")
        recorded = feedback.record(answer_id, request.value)
        conversations.record_feedback(answer_id, recorded.value)
        return recorded

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("cockpit_agent.api:app", host="127.0.0.1", port=8000, reload=True)
