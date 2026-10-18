from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from .ingest import KnowledgeIngestor
from .registry import RegistryError, VehicleRegistry
from .schemas import KnowledgeChunk, RetrievedChunk, VehicleProfile

LEXICAL_PATTERN = re.compile(
    r"[A-Za-z0-9_]+(?:[./-][A-Za-z0-9_]+)*|[\u4e00-\u9fff]+"
)
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_]+(?:[./-][A-Za-z0-9_]+)*")
CHINESE_PATTERN = re.compile(r"^[\u4e00-\u9fff]+$")

CHINESE_NGRAM_SIZES = (2, 3)
MIN_WEIGHTED_QUERY_COVERAGE = 0.09
MIN_MATCHED_QUERY_FEATURES = 2
SECTION_MATCH_BOOST = 1.5

# These words are useful prose but weak evidence on their own. Acronyms, model
# identifiers, and numeric identifiers are recognized structurally instead.
LATIN_QUERY_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "can",
        "for",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "which",
        "with",
    }
)


class VehicleBookRetriever:
    """Vehicle-aware BM25 retrieval over registered local knowledge."""

    def __init__(
        self,
        vehicle_registry: VehicleRegistry | None = None,
        ingestor: KnowledgeIngestor | None = None,
    ) -> None:
        if ingestor is None:
            self.vehicle_registry = vehicle_registry or VehicleRegistry()
            self.ingestor = KnowledgeIngestor(vehicle_registry=self.vehicle_registry)
        else:
            self.ingestor = ingestor
            self.vehicle_registry = vehicle_registry or ingestor.vehicle_registry

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Return normalized Latin identifiers and overlapping Chinese n-grams."""
        normalized = unicodedata.normalize("NFKC", text)
        tokens: list[str] = []
        for match in LEXICAL_PATTERN.finditer(normalized):
            value = match.group(0)
            if CHINESE_PATTERN.fullmatch(value):
                for size in CHINESE_NGRAM_SIZES:
                    tokens.extend(
                        value[index : index + size]
                        for index in range(len(value) - size + 1)
                    )
            else:
                tokens.append(value.lower())
        return tokens

    @staticmethod
    def _significant_identifiers(text: str) -> frozenset[str]:
        """Find acronyms and structured identifiers without a domain allow-list."""
        normalized = unicodedata.normalize("NFKC", text)
        identifiers: set[str] = set()
        for match in IDENTIFIER_PATTERN.finditer(normalized):
            value = match.group(0)
            if (
                any(character.isdigit() or character in "_./-" for character in value)
                or (value.isupper() and 2 <= len(value) <= 12)
            ):
                identifiers.add(value.lower())
        return frozenset(identifiers)

    @classmethod
    def _profile_context_tokens(cls, profile: VehicleProfile) -> frozenset[str]:
        model = profile.model
        software_version = profile.software_version
        values = [value for value in (model, software_version, profile.oem, profile.trim) if value]
        version_numbers = re.findall(r"\d+", software_version or "")
        if version_numbers:
            values.append(f"v{version_numbers[0]}")
            values.append(f"v{software_version}")
        return frozenset(token for value in values for token in cls._tokenize(value))

    @classmethod
    def _section_features(cls, section: str) -> frozenset[str]:
        features = set(cls._tokenize(section))
        latin_words = [
            word.lower()
            for word in re.findall(r"[A-Za-z]+", unicodedata.normalize("NFKC", section))
            if word.lower() not in LATIN_QUERY_STOP_WORDS
        ]
        if len(latin_words) >= 2:
            features.add("".join(word[0] for word in latin_words))
        return frozenset(features)

    @staticmethod
    def _inverse_document_frequency(
        token: str,
        document_frequency: Counter[str],
        document_count: int,
    ) -> float:
        frequency_in_corpus = document_frequency[token]
        return math.log(
            1
            + (document_count - frequency_in_corpus + 0.5)
            / (frequency_in_corpus + 0.5)
        )

    @staticmethod
    def _bm25_score(
        query_tokens: list[str],
        document_tokens: list[str],
        document_frequency: Counter[str],
        document_count: int,
        average_length: float,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        term_frequency = Counter(document_tokens)
        score = 0.0

        for token in query_tokens:
            frequency = term_frequency[token]
            if frequency == 0:
                continue

            inverse_document_frequency = VehicleBookRetriever._inverse_document_frequency(
                token,
                document_frequency,
                document_count,
            )
            denominator = frequency + k1 * (
                1 - b + b * len(document_tokens) / max(average_length, 1)
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1)
            ) / denominator

        return score

    @classmethod
    def _has_sufficient_evidence(
        cls,
        query_tokens: tuple[str, ...],
        significant_identifiers: frozenset[str],
        document_tokens: list[str],
        document_frequency: Counter[str],
        document_count: int,
    ) -> bool:
        document_token_set = set(document_tokens)
        if significant_identifiers.intersection(document_token_set):
            return True

        meaningful_tokens = tuple(
            token
            for token in query_tokens
            if CHINESE_PATTERN.fullmatch(token)
            or (len(token) >= 3 and token not in LATIN_QUERY_STOP_WORDS)
        )
        matched_tokens = [
            token for token in meaningful_tokens if token in document_token_set
        ]
        if len(matched_tokens) < MIN_MATCHED_QUERY_FEATURES:
            return False

        total_weight = sum(
            cls._inverse_document_frequency(
                token, document_frequency, document_count
            )
            for token in meaningful_tokens
        )
        matched_weight = sum(
            cls._inverse_document_frequency(
                token, document_frequency, document_count
            )
            for token in matched_tokens
        )
        return total_weight > 0 and matched_weight / total_weight >= MIN_WEIGHTED_QUERY_COVERAGE

    def retrieve(
        self,
        vehicle_id: str,
        query: str,
        top_k: int = 2,
        *,
        include_pre_release: bool = False,
    ) -> tuple[RetrievedChunk, ...]:
        """Resolve, authorize, load, and rank chunks for one exact vehicle."""
        profile = self.vehicle_registry.resolve(vehicle_id)

        if profile.release_status == "pre_release" and not include_pre_release:
            raise RegistryError(
                f"Vehicle '{profile.vehicle_id}' is pre_release; "
                "pass include_pre_release=True to retrieve its documents"
            )

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        context_tokens = self._profile_context_tokens(profile)
        query_tokens = tuple(
            dict.fromkeys(
                token for token in self._tokenize(query) if token not in context_tokens
            )
        )
        if not query_tokens:
            return ()

        loaded_chunks = self.ingestor.ingest(profile.vehicle_id)
        eligible_chunks = tuple(
            chunk
            for chunk in loaded_chunks
            if chunk.metadata.vehicle_id == profile.vehicle_id
            and chunk.metadata.status == "published"
            and (profile.market is None or chunk.metadata.market == profile.market)
            and (
                profile.software_version is None
                or chunk.metadata.document_version == profile.software_version
            )
        )
        if not eligible_chunks:
            return ()

        tokenized_chunks = [self._tokenize(chunk.text) for chunk in eligible_chunks]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized_chunks:
            document_frequency.update(set(tokens))

        document_count = len(tokenized_chunks)
        average_length = sum(map(len, tokenized_chunks)) / document_count
        significant_identifiers = self._significant_identifiers(query) - context_tokens
        scored: list[tuple[float, KnowledgeChunk]] = []
        for chunk, tokens in zip(eligible_chunks, tokenized_chunks, strict=True):
            if not self._has_sufficient_evidence(
                query_tokens,
                significant_identifiers,
                tokens,
                document_frequency,
                document_count,
            ):
                continue
            score = self._bm25_score(
                list(query_tokens),
                tokens,
                document_frequency,
                document_count,
                average_length,
            )
            section_matches = self._section_features(chunk.section).intersection(
                query_tokens
            )
            score += SECTION_MATCH_BOOST * sum(
                self._inverse_document_frequency(
                    token, document_frequency, document_count
                )
                for token in section_matches
            )
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return tuple(
            RetrievedChunk(chunk=chunk, score=score, rank=rank)
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        )

    def search(
        self,
        vehicle_id: str,
        query: str,
        top_k: int = 2,
        *,
        include_pre_release: bool = False,
    ) -> tuple[RetrievedChunk, ...]:
        """Alias for retrieve, matching the legacy retriever's search terminology."""
        return self.retrieve(
            vehicle_id,
            query,
            top_k,
            include_pre_release=include_pre_release,
        )
