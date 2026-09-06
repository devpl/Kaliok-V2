from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from kaliok.storage.models import NormalizedContentUnit


@dataclass(frozen=True)
class CandidateOccurrence:
    candidate_type: str
    raw_value: str
    normalized_value: str | None
    payload: dict[str, Any]
    confidence: float | None
    start_offset: int
    end_offset: int
    exact_text: str


class CandidateDetector(Protocol):
    key: str
    version: str | None

    @property
    def configuration(self) -> dict[str, Any]: ...

    def detect(self, unit: NormalizedContentUnit) -> tuple[CandidateOccurrence, ...]: ...


@dataclass(frozen=True)
class LexicalTerm:
    value: str
    candidate_type: str
    normalized_value: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None


class LexicalCandidateDetector:
    key = "lexical_dictionary"
    version = "1"

    def __init__(
        self,
        terms: list[LexicalTerm] | tuple[LexicalTerm, ...],
        *,
        case_sensitive: bool = False,
        boundary_policy: str = "unicode_word",
        dictionary_identity: dict[str, Any] | None = None,
    ) -> None:
        if not terms:
            raise ValueError("Au moins un terme lexical est requis.")
        if any(not term.value.strip() for term in terms):
            raise ValueError("La valeur d'un terme lexical est requise.")
        if boundary_policy not in {"unicode_word", "substring"}:
            raise ValueError(
                "boundary_policy doit être 'unicode_word' ou 'substring'."
            )
        self._terms = tuple(terms)
        self._case_sensitive = case_sensitive
        self._boundary_policy = boundary_policy
        self.dictionary_identity = dict(dictionary_identity) if dictionary_identity else None

    @property
    def configuration(self) -> dict[str, Any]:
        return {
            "case_sensitive": self._case_sensitive,
            "boundary_policy": self._boundary_policy,
            "terms": [{
                "value": term.value,
                "candidate_type": term.candidate_type,
                "normalized_value": term.normalized_value,
                "confidence": term.confidence,
                "payload": dict(term.payload),
            } for term in self._terms],
        }

    def detect(self, unit: NormalizedContentUnit) -> tuple[CandidateOccurrence, ...]:
        flags = 0 if self._case_sensitive else re.IGNORECASE
        occurrences: list[CandidateOccurrence] = []
        for term in self._terms:
            pattern = re.escape(term.value)
            if self._boundary_policy == "unicode_word":
                if re.match(r"\w", term.value[0], re.UNICODE):
                    pattern = rf"(?<!\w){pattern}"
                if re.match(r"\w", term.value[-1], re.UNICODE):
                    pattern = rf"{pattern}(?!\w)"
            for match in re.finditer(pattern, unit.content, flags):
                exact_text = match.group(0)
                occurrences.append(
                    CandidateOccurrence(
                        candidate_type=term.candidate_type,
                        raw_value=exact_text,
                        normalized_value=(
                            term.normalized_value
                            if term.normalized_value is not None
                            else term.value.strip()
                        ),
                        payload=dict(term.payload),
                        confidence=term.confidence,
                        start_offset=match.start(),
                        end_offset=match.end(),
                        exact_text=exact_text,
                    )
                )
        return tuple(sorted(occurrences, key=lambda item: (item.start_offset, item.end_offset)))
