from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from uuid import UUID


Identifier = UUID | str
Metadata = Mapping[str, Any]


@dataclass(frozen=True)
class Provenance:
    document_id: Identifier | None = None
    document_version_id: Identifier | None = None
    processing_run_id: Identifier | None = None
    page: int | None = None
    source_ids: tuple[Identifier, ...] = ()
    representation: str | None = None
    embedding_model: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedDocument:
    content: Any
    provenance: Provenance = field(default_factory=Provenance)
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalUnit:
    unit_id: Identifier
    text: str
    provenance: Provenance = field(default_factory=Provenance)
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingRecord:
    unit: RetrievalUnit
    vector: Sequence[float]
    model: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    unit: RetrievalUnit
    score: float | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class RankedCandidate:
    candidate: Candidate
    rank: int
    score: float | None = None
    metadata: Metadata = field(default_factory=dict)

    @property
    def unit(self) -> RetrievalUnit:
        return self.candidate.unit


@dataclass(frozen=True)
class ContextBundle:
    question: str
    text: str
    candidates: tuple[RankedCandidate, ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class RagAnswer:
    text: str
    context: ContextBundle
    metadata: Metadata = field(default_factory=dict)
