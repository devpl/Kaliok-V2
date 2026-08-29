from __future__ import annotations

from collections.abc import Callable, Sequence
from uuid import UUID

from sqlmodel import Session, select

from kaliok.embeddings.ollama import EMBEDDING_MODEL
from kaliok.embeddings.service import SimilarChunk, search_similar_chunks
from kaliok.rag.types import Candidate, EmbeddingRecord, Provenance, RetrievalUnit
from kaliok.storage.models import (
    ChunkEmbedding,
    DocumentChunk,
    DocumentVersion,
    EmbeddingModel,
    NormalizedContentUnit,
)


NORMALIZED_CHUNKING_STRATEGY = "normalized-content-unit"
NORMALIZED_CHUNKING_VERSION = "1"


def _active_embedding_model(
    session: Session,
    model_name: str,
) -> EmbeddingModel | None:
    return session.exec(
        select(EmbeddingModel).where(
            EmbeddingModel.provider == "ollama",
            EmbeddingModel.model_name == model_name,
            EmbeddingModel.dimensions == 1024,
            EmbeddingModel.is_active.is_(True),
        )
    ).first()


def _get_or_create_embedding_model(session: Session) -> EmbeddingModel:
    model = session.exec(
        select(EmbeddingModel).where(
            EmbeddingModel.provider == "ollama",
            EmbeddingModel.model_name == EMBEDDING_MODEL,
            EmbeddingModel.dimensions == 1024,
        )
    ).first()
    if model is not None:
        return model
    model = EmbeddingModel(
        provider="ollama",
        model_name=EMBEDDING_MODEL,
        dimensions=1024,
        distance_metric="cosine",
        is_active=True,
    )
    session.add(model)
    session.flush()
    return model


def normalized_version_is_indexed(
    session: Session,
    document_version_id: UUID,
    *,
    model_name: str = EMBEDDING_MODEL,
) -> bool:
    model = _active_embedding_model(session, model_name)
    if model is None:
        return False
    units = list(
        session.exec(
            select(NormalizedContentUnit).where(
                NormalizedContentUnit.document_version_id
                == document_version_id
            )
        ).all()
    )
    if not units:
        return False
    chunks = list(
        session.exec(
            select(DocumentChunk).where(
                DocumentChunk.document_version_id == document_version_id,
                DocumentChunk.chunking_strategy
                == NORMALIZED_CHUNKING_STRATEGY,
                DocumentChunk.chunking_version == NORMALIZED_CHUNKING_VERSION,
            )
        ).all()
    )
    if {chunk.id for chunk in chunks} != {unit.id for unit in units}:
        return False
    embeddings = list(
        session.exec(
            select(ChunkEmbedding).where(
                ChunkEmbedding.chunk_id.in_([chunk.id for chunk in chunks]),
                ChunkEmbedding.embedding_model_id == model.id,
            )
        ).all()
    )
    return {item.chunk_id for item in embeddings} == {
        chunk.id for chunk in chunks
    }


class PostgresVectorIndexStore:
    """Persist one retrieval chunk per normalized content unit."""

    def __init__(
        self,
        session: Session,
        *,
        model_name: str = EMBEDDING_MODEL,
    ) -> None:
        if model_name != EMBEDDING_MODEL:
            raise ValueError(
                "Seul le modèle d'embedding Kaliok actif peut être persisté."
            )
        self._session = session
        self._model_name = model_name

    def write(self, records: Sequence[EmbeddingRecord]) -> None:
        if not records:
            return
        version_ids = {
            record.unit.provenance.document_version_id for record in records
        }
        if None in version_ids or len(version_ids) != 1:
            raise ValueError("Une seule DocumentVersion est attendue par écriture.")
        if any(record.model != self._model_name for record in records):
            raise ValueError("Modèle d'embedding incohérent avec l'index.")
        model = _get_or_create_embedding_model(self._session)
        for expected_index, record in enumerate(records):
            unit_id = self._uuid(record.unit.unit_id)
            unit_index = record.unit.metadata.get("unit_index")
            if unit_index != expected_index:
                raise ValueError("Les unités doivent être ordonnées depuis zéro.")
            source_unit = self._session.get(NormalizedContentUnit, unit_id)
            if source_unit is None:
                raise ValueError(f"NormalizedContentUnit inconnue : {unit_id}.")
            if source_unit.document_version_id not in version_ids:
                raise ValueError("Unité rattachée à une autre DocumentVersion.")
            chunk = self._session.get(DocumentChunk, unit_id)
            if chunk is None:
                chunk = DocumentChunk(
                    id=unit_id,
                    document_version_id=source_unit.document_version_id,
                    parent_chunk_id=source_unit.parent_unit_id,
                    chunk_index=source_unit.unit_index,
                    content=record.unit.text,
                    char_count=len(record.unit.text),
                    page_start=None,
                    page_end=None,
                    chunking_strategy=NORMALIZED_CHUNKING_STRATEGY,
                    chunking_version=NORMALIZED_CHUNKING_VERSION,
                )
                self._session.add(chunk)
                self._session.flush()
            else:
                self._validate_existing(chunk, source_unit, record)
            key = (chunk.id, model.id)
            if self._session.get(ChunkEmbedding, key) is None:
                self._session.add(
                    ChunkEmbedding(
                        chunk_id=chunk.id,
                        embedding_model_id=model.id,
                        embedding=list(record.vector),
                    )
                )
        self._session.flush()

    @staticmethod
    def _uuid(value: object) -> UUID:
        if isinstance(value, UUID):
            return value
        try:
            return UUID(str(value))
        except (TypeError, ValueError) as error:
            raise ValueError("RetrievalUnit.unit_id doit être un UUID.") from error

    @staticmethod
    def _validate_existing(
        chunk: DocumentChunk,
        source_unit: NormalizedContentUnit,
        record: EmbeddingRecord,
    ) -> None:
        if (
            chunk.document_version_id != source_unit.document_version_id
            or chunk.chunk_index != source_unit.unit_index
            or chunk.content != record.unit.text
            or chunk.chunking_strategy != NORMALIZED_CHUNKING_STRATEGY
            or chunk.chunking_version != NORMALIZED_CHUNKING_VERSION
        ):
            raise ValueError("Chunk existant incohérent avec le contenu normalisé.")


class PostgresVectorRetriever:
    def __init__(
        self,
        session: Session,
        document_version_id: UUID,
        *,
        model_name: str = EMBEDDING_MODEL,
        search: Callable[..., list[SimilarChunk]] = search_similar_chunks,
    ) -> None:
        self._session = session
        self._document_version_id = document_version_id
        self._model_name = model_name
        self._search = search

    def retrieve(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
    ) -> tuple[Candidate, ...]:
        model = _active_embedding_model(self._session, self._model_name)
        if model is None:
            raise ValueError(
                f"Modèle d'embedding actif introuvable : {self._model_name}."
            )
        results = self._search(
            query_embedding=list(query_embedding),
            embedding_model_id=model.id,
            document_version_id=self._document_version_id,
            limit=top_k,
        )
        version = self._session.get(
            DocumentVersion, self._document_version_id
        )
        if version is None:
            raise ValueError(
                f"DocumentVersion inconnue : {self._document_version_id}."
            )
        candidates: list[Candidate] = []
        for result in results:
            chunk = self._session.get(DocumentChunk, result.chunk_id)
            if (
                chunk is None
                or chunk.document_version_id != self._document_version_id
                or chunk.chunking_strategy != NORMALIZED_CHUNKING_STRATEGY
                or chunk.chunking_version != NORMALIZED_CHUNKING_VERSION
            ):
                continue
            source_unit = self._session.get(
                NormalizedContentUnit, result.chunk_id
            )
            if source_unit is None:
                continue
            provenance = Provenance(
                document_id=version.document_id,
                document_version_id=source_unit.document_version_id,
                source_ids=(source_unit.id,),
                representation="normalized_content_unit",
                embedding_model=self._model_name,
                metadata={
                    "normalized_content_unit_id": source_unit.id,
                    "source_unit_id": source_unit.source_unit_id,
                    "unit_index": source_unit.unit_index,
                    "content_type": source_unit.content_type,
                    "parent_unit_id": source_unit.parent_unit_id,
                },
            )
            candidates.append(
                Candidate(
                    unit=RetrievalUnit(
                        unit_id=result.chunk_id,
                        text=result.content,
                        provenance=provenance,
                        metadata=provenance.metadata,
                    ),
                    score=1.0 - result.distance,
                    metadata={"distance": result.distance},
                )
            )
        return tuple(candidates)
