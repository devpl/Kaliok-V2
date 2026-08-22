from __future__ import annotations

import hashlib
from pathlib import Path

from sqlmodel import Session, select

from kaliok.documents.cleaning import clean_document
from kaliok.documents.reader import read_document
from kaliok.documents.semantic_chunking import (
    chunk_document_semantically,
)
from kaliok.embeddings.ollama import (
    EMBEDDING_MODEL,
    embed_text,
)
from kaliok.paths import TEST_DOCUMENTS
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    ChunkEmbedding,
    Document,
    DocumentChunk,
    DocumentVersion,
    EmbeddingModel,
    Source,
)


PDF_PATH = TEST_DOCUMENTS / "RIDEAU.pdf"

EMBEDDING_DIMENSIONS = 1024
CHUNKING_STRATEGY = "llamaindex-semantic-cleaning"
CHUNKING_VERSION = "1"


def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


def get_or_create_source(
    session: Session,
) -> Source:
    source = session.exec(
        select(Source).where(
            Source.name == "test_documents",
            Source.source_type == "local_directory",
        )
    ).first()

    if source is not None:
        return source

    source = Source(
        name="test_documents",
        source_type="local_directory",
        external_reference=str(TEST_DOCUMENTS),
    )

    session.add(source)
    session.flush()

    return source


def get_or_create_document(
    session: Session,
    source: Source,
    path: Path,
) -> Document:
    document = session.exec(
        select(Document).where(
            Document.source_id == source.id,
            Document.external_id == path.name,
        )
    ).first()

    if document is not None:
        return document

    document = Document(
        source_id=source.id,
        external_id=path.name,
        title=path.stem,
        document_family="test_document",
        status="active",
        language="fr",
    )

    session.add(document)
    session.flush()

    return document


def find_existing_version(
    session: Session,
    document: Document,
    file_hash: str,
) -> DocumentVersion | None:
    return session.exec(
        select(DocumentVersion).where(
            DocumentVersion.document_id == document.id,
            DocumentVersion.file_hash == file_hash,
        )
    ).first()


def get_next_version_number(
    session: Session,
    document: Document,
) -> int:
    versions = session.exec(
        select(DocumentVersion).where(
            DocumentVersion.document_id == document.id
        )
    ).all()

    if not versions:
        return 1

    return max(
        version.version_number
        for version in versions
    ) + 1


def get_or_create_embedding_model(
    session: Session,
) -> EmbeddingModel:
    embedding_model = session.exec(
        select(EmbeddingModel).where(
            EmbeddingModel.provider == "ollama",
            EmbeddingModel.model_name == EMBEDDING_MODEL,
            EmbeddingModel.dimensions == EMBEDDING_DIMENSIONS,
        )
    ).first()

    if embedding_model is not None:
        return embedding_model

    embedding_model = EmbeddingModel(
        provider="ollama",
        model_name=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        distance_metric="cosine",
        is_active=True,
    )

    session.add(embedding_model)
    session.flush()

    return embedding_model


def main() -> None:
    print(f"Document : {PDF_PATH}")

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Document introuvable : {PDF_PATH}"
        )

    file_hash = calculate_sha256(PDF_PATH)

    print(f"SHA-256  : {file_hash}")

    # ---------------------------------------------------------
    # Lecture et nettoyage
    # ---------------------------------------------------------

    print("\n1. Lecture du document...")

    document_content = read_document(PDF_PATH)

    print(
        f"   Pages      : {document_content.page_count}"
    )
    print(
        f"   Blocs      : {len(document_content.blocks)}"
    )
    print(
        f"   Caractères : {len(document_content.text)}"
    )

    print("\n2. Nettoyage...")

    cleaned_document = clean_document(
        document_content
    )

    print(
        f"   Caractères nettoyés : "
        f"{len(cleaned_document.text)}"
    )

    # ---------------------------------------------------------
    # Chunking sémantique
    # ---------------------------------------------------------

    print("\n3. Chunking sémantique...")

    chunks = chunk_document_semantically(
        cleaned_document,
        breakpoint_percentile_threshold=95,
        buffer_size=1,
    )

    print(
        f"   Chunks : {len(chunks)}"
    )

    if not chunks:
        raise RuntimeError(
            "Aucun chunk généré."
        )

    # ---------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------

    print("\n4. Génération des embeddings...")

    embeddings: list[list[float]] = []

    for position, chunk in enumerate(
        chunks,
        start=1,
    ):
        print(
            f"   Embedding {position}/{len(chunks)}",
            end="\r",
        )

        embedding = embed_text(
            chunk.text
        )

        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                "Dimension inattendue : "
                f"{len(embedding)}"
            )

        embeddings.append(
            embedding
        )

    print()

    # ---------------------------------------------------------
    # Stockage PostgreSQL
    # ---------------------------------------------------------

    print("\n5. Enregistrement PostgreSQL...")

    engine = create_database_engine()

    with Session(engine) as session:
        source = get_or_create_source(
            session
        )

        document = get_or_create_document(
            session,
            source,
            PDF_PATH,
        )

        existing_version = find_existing_version(
            session,
            document,
            file_hash,
        )

        if existing_version is not None:
            print()
            print(
                "Cette version du document est déjà "
                "présente en base."
            )
            print(
                f"DocumentVersion : "
                f"{existing_version.id}"
            )
            print()
            print(
                "Aucun doublon n'a été créé."
            )

            session.rollback()
            return

        embedding_model = (
            get_or_create_embedding_model(
                session
            )
        )

        version_number = (
            get_next_version_number(
                session,
                document,
            )
        )

        version = DocumentVersion(
            document_id=document.id,
            version_number=version_number,
            filename=PDF_PATH.name,
            mime_type="application/pdf",
            file_hash=file_hash,
            file_size=PDF_PATH.stat().st_size,
            storage_uri=str(PDF_PATH),
            page_count=cleaned_document.page_count,
            document_type="pdf",
            version_status="active",
            processing_status="completed",
            readability_status="readable",
            is_current=True,
        )

        session.add(version)
        session.flush()

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):
            stored_chunk = DocumentChunk(
                document_version_id=version.id,
                chunk_index=chunk.index,
                content=chunk.text,
                char_count=len(chunk.text),
                page_start=chunk.page,
                page_end=chunk.page,
                chunking_strategy=CHUNKING_STRATEGY,
                chunking_version=CHUNKING_VERSION,
            )

            session.add(stored_chunk)
            session.flush()

            stored_embedding = ChunkEmbedding(
                chunk_id=stored_chunk.id,
                embedding_model_id=embedding_model.id,
                embedding=embedding,
            )

            session.add(
                stored_embedding
            )

        session.commit()

        version_id = version.id
        document_id = document.id
        embedding_model_id = embedding_model.id

    # ---------------------------------------------------------
    # Résultat
    # ---------------------------------------------------------

    print()
    print("Indexation terminée.")
    print()
    print(
        f"Document ID          : {document_id}"
    )
    print(
        f"DocumentVersion ID   : {version_id}"
    )
    print(
        f"EmbeddingModel ID    : {embedding_model_id}"
    )
    print(
        f"Chunks enregistrés   : {len(chunks)}"
    )
    print(
        f"Embeddings enregistrés : {len(embeddings)}"
    )


if __name__ == "__main__":
    main()
