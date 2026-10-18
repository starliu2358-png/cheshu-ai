from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

from dotenv import load_dotenv
from openai import OpenAI

from .schemas import RetrievedChunk
from .response_style import RESPONSE_STYLE


class AnswerGenerator(Protocol):
    """Isolate text generation from grounding and response validation."""

    def generate(self, prompt: str) -> str:
        """Return one raw model response for the supplied prompt."""
        ...


class DeepSeekAnswerGenerator:
    """OpenAI-compatible generator configured with the project's DeepSeek settings."""

    def __init__(self, client: OpenAI, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

    @classmethod
    def from_env(cls) -> DeepSeekAnswerGenerator:
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

        if not api_key:
            raise ValueError(
                "未找到 DEEPSEEK_API_KEY，请在项目根目录的 .env 文件中配置。"
            )

        return cls(
            client=OpenAI(api_key=api_key, base_url=base_url),
            model_name=model_name,
        )

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise AnswerDraftError("Generator returned an empty response")
        return content


class AnswerDraftError(ValueError):
    """Raised when evidence or a generated answer draft cannot be trusted safely."""


@dataclass(frozen=True, slots=True)
class AnswerDraft:
    answer: str
    cited_chunk_ids: tuple[str, ...]
    insufficient_evidence: bool


class EvidenceAnswerer:
    """Provide a deterministic, citation-backed answer when no model key is configured."""

    def generate_draft(
        self,
        vehicle_id: str,
        question: str,
        evidence: tuple[RetrievedChunk, ...],
    ) -> AnswerDraft:
        if not vehicle_id.strip() or not question.strip():
            raise ValueError("vehicle_id and question must be non-empty")
        if not evidence:
            raise AnswerDraftError("At least one evidence chunk is required")
        selected = evidence[0]
        if selected.chunk.metadata.vehicle_id != vehicle_id:
            raise AnswerDraftError("Evidence contains a chunk for a different vehicle")

        lines = selected.chunk.text.splitlines()
        answer = "\n".join(re.sub(r"^#{1,6}\s+", "", line) for line in lines).strip()
        return AnswerDraft(
            answer=answer,
            cited_chunk_ids=(selected.chunk.chunk_id,),
            insufficient_evidence=False,
        )


class GroundedAnswerer:
    """Create and validate an evidence-only answer draft."""

    _OUTPUT_KEYS = frozenset(
        {"answer", "cited_chunk_ids", "insufficient_evidence"}
    )

    def __init__(self, generator: AnswerGenerator) -> None:
        self.generator = generator

    def generate_draft(
        self,
        vehicle_id: str,
        question: str,
        evidence: tuple[RetrievedChunk, ...],
    ) -> AnswerDraft:
        if not vehicle_id.strip():
            raise ValueError("vehicle_id must be non-empty")
        if not question.strip():
            raise ValueError("question must be non-empty")
        if not evidence:
            raise AnswerDraftError("At least one evidence chunk is required")

        for result in evidence:
            if result.chunk.metadata.vehicle_id != vehicle_id:
                raise AnswerDraftError(
                    "Evidence contains a chunk for a different vehicle"
                )

        prompt = self._build_prompt(vehicle_id, question, evidence)
        raw_response = self.generator.generate(prompt)
        return self._parse_and_validate(raw_response, evidence)

    @staticmethod
    def _build_prompt(
        vehicle_id: str,
        question: str,
        evidence: tuple[RetrievedChunk, ...],
    ) -> str:
        evidence_payload = [
            {
                "chunk_id": result.chunk.chunk_id,
                "vehicle_id": result.chunk.metadata.vehicle_id,
                "document_version": result.chunk.metadata.document_version,
                "section": result.chunk.section,
                "text": result.chunk.text,
            }
            for result in evidence
        ]
        contract = {
            "answer": "string",
            "cited_chunk_ids": ["string"],
            "insufficient_evidence": False,
        }
        return (
            "Answer the question using only the supplied evidence.\n"
            "Do not use outside knowledge. Do not invent document names, paths, "
            "features, limits, or procedures.\n"
            "Answer only what the question asks. Do not add related features, "
            "conditions, procedures, or version comparisons unless they are needed "
            "to answer the question.\n"
            "Cover each attribute explicitly requested by the question. When the "
            "question asks about applicable objects, trims, seats, levels, limits, "
            "or ranges, state the applicable object together with its value; do not "
            "replace that scope with only a number. Omit adjacent capabilities that "
            "the question did not ask about.\n"
            "Treat the question as a strict scope boundary. For a question about "
            "levels, ranges, or limits, include the applicable scope and requested "
            "value, but do not add automatic modes or other adjacent capabilities "
            "unless the question explicitly asks about them. Preserve measurement "
            "units exactly as written in the supporting evidence.\n"
            "Citations are claim-level evidence, not a list of all retrieved context. "
            "Cite only the supplied chunks that directly support facts stated in the "
            "answer. Use the smallest sufficient set of citations; do not cite a "
            "chunk merely because it identifies the vehicle or is generally related.\n"
            "Every factual answer must cite one or more supplied chunk_id values.\n"
            "If the evidence is insufficient, set insufficient_evidence to true, "
            "use an empty cited_chunk_ids list, and do not guess.\n"
            "Return one JSON object only, with exactly these keys and types:\n"
            f"{json.dumps(contract, ensure_ascii=False)}\n"
            f"{RESPONSE_STYLE}\n"
            f"Requested vehicle_id: {vehicle_id}\n"
            f"Question: {question.strip()}\n"
            "Evidence:\n"
            f"{json.dumps(evidence_payload, ensure_ascii=False)}"
        )

    @classmethod
    def _parse_and_validate(
        cls,
        raw_response: str,
        evidence: tuple[RetrievedChunk, ...],
    ) -> AnswerDraft:
        try:
            payload: Any = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError) as exc:
            raise AnswerDraftError("Generator response is not valid JSON") from exc

        if not isinstance(payload, dict) or set(payload) != cls._OUTPUT_KEYS:
            raise AnswerDraftError(
                "Generator response must contain exactly answer, cited_chunk_ids, "
                "and insufficient_evidence"
            )

        answer = payload["answer"]
        cited_chunk_ids = payload["cited_chunk_ids"]
        insufficient_evidence = payload["insufficient_evidence"]

        if not isinstance(answer, str):
            raise AnswerDraftError("answer must be a string")
        if not isinstance(insufficient_evidence, bool):
            raise AnswerDraftError("insufficient_evidence must be a boolean")
        if not isinstance(cited_chunk_ids, list) or not all(
            isinstance(chunk_id, str) and chunk_id.strip()
            for chunk_id in cited_chunk_ids
        ):
            raise AnswerDraftError(
                "cited_chunk_ids must be a list of non-empty strings"
            )
        if len(cited_chunk_ids) != len(set(cited_chunk_ids)):
            raise AnswerDraftError("cited_chunk_ids must not contain duplicates")

        available_ids = {result.chunk.chunk_id for result in evidence}
        unknown_ids = set(cited_chunk_ids) - available_ids
        if unknown_ids:
            raise AnswerDraftError(
                "Generator cited a chunk that was not supplied as evidence"
            )

        if insufficient_evidence:
            if cited_chunk_ids:
                raise AnswerDraftError(
                    "An insufficient-evidence draft cannot contain citations"
                )
        elif not answer.strip() or not cited_chunk_ids:
            raise AnswerDraftError(
                "A factual answer requires non-empty text and at least one citation"
            )

        return AnswerDraft(
            answer=answer.strip(),
            cited_chunk_ids=tuple(cited_chunk_ids),
            insufficient_evidence=insufficient_evidence,
        )
