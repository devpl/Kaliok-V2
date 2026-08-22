from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text

from kaliok.documents.chunking import chunk_document
from kaliok.documents.reader import read_document
from kaliok.embeddings.ollama import EMBEDDING_MODEL, embed_text
from kaliok.embeddings.service import search_similar_chunks
from kaliok.paths import TEST_DOCUMENTS
from kaliok.storage.database import create_database_engine


EMBEDDING_DIMENSIONS = 1024

DEFAULT_PDF = TEST_DOCUMENTS / "RIDEAU.pdf"
DEFAULT_QUESTION = "De quoi parle principalement ce document ?"


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def get_or_create_embedding_model(connection) -> tuple[UUID, bool]:
    row = connection.execute(
        text(
            """
            SELECT id
            FROM embedding_models
            WHERE provider = 'ollama'
              AND model_name = :model_name
              AND dimensions = :dimensions
            ORDER BY created_at
            LIMIT 1
            """
        ),
        {
            "model_name": EMBEDDING_MODEL,
            "dimensions": EMBEDDING_DIMENSIONS,
        },
    ).first()

    if row is not None:
        return row.id, False

    model_id = uuid4()

    connection.execute(
        text(
            """
            INSERT INTO embedding_models (
                id,
                provider,
                model_name,
                dimensions,
                distance_metric,
                configuration,
                is_active,
                created_at
            )
            VALUES (
                :id,
                'ollama',
                :model_name,
                :dimensions,
                'cosine',
                '{}'::jsonb,
                true,
                now()
            )
            """
        ),
        {
            "id": model_id,
            "model_name": EMBEDDING_MODEL,
            "dimensions": EMBEDDING_DIMENSIONS,
        },
    )

    return model_id, True


def store_document(
    pdf_path: Path,
    chunks,
    embeddings: list[list[float]],
    page_count: int,
) -> tuple[UUID, UUID, UUID, UUID, bool]:
    engine = create_database_engine()

    source_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()

    insert_embedding_query = text(
        """
        INSERT INTO chunk_embeddings (
            chunk_id,
            embedding_model_id,
            embedding,
            created_at
        )
        VALUES (
            :chunk_id,
            :embedding_model_id,
            :embedding,
            now()
        )
        """
    ).bindparams(
        bindparam(
            "embedding",
            type_=Vector(EMBEDDING_DIMENSIONS),
        )
    )

    with engine.begin() as connection:
        model_id, model_created = get_or_create_embedding_model(
            connection
        )

        connection.execute(
            text(
                """
                INSERT INTO sources (
                    id,
                    name,
                    source_type,
                    external_reference,
                    configuration,
                    is_active,
                    created_at,
                    updated_at
                )
                VALUES (
                    :id,
                    :name,
                    'manual_test',
                    :external_reference,
                    '{}'::jsonb,
                    true,
                    now(),
                    now()
                )
                """
            ),
            {
                "id": source_id,
                "name": "document-vector-search-check",
                "external_reference": str(pdf_path),
            },
        )

        connection.execute(
            text(
                """
                INSERT INTO documents (
                    id,
                    source_id,
                    title,
                    status,
                    language,
                    metadata,
                    created_at,
                    updated_at
                )
                VALUES (
                    :id,
                    :source_id,
                    :title,
                    'active',
                    'fr',
                    '{}'::jsonb,
                    now(),
                    now()
                )
                """
            ),
            {
                "id": document_id,
                "source_id": source_id,
                "title": pdf_path.name,
            },
        )

        connection.execute(
            text(
                """
                INSERT INTO document_versions (
                    id,
                    document_id,
                    version_number,
                    filename,
                    mime_type,
                    file_hash,
                    file_size,
                    storage_uri,
                    page_count,
                    version_status,
                    processing_status,
                    readability_status,
                    is_current,
                    metadata,
                    created_at,
                    processed_at
                )
                VALUES (
                    :id,
                    :document_id,
                    1,
                    :filename,
                    'application/pdf',
                    :file_hash,
                    :file_size,
                    :storage_uri,
                    :page_count,
                    'published',
                    'completed',
                    'readable',
                    true,
                    '{}'::jsonb,
                    now(),
                    now()
                )
                """
            ),
            {
                "id": version_id,
                "document_id": document_id,
                "filename": pdf_path.name,
                "file_hash": calculate_sha256(pdf_path),
                "file_size": pdf_path.stat().st_size,
                "storage_uri": pdf_path.resolve().as_uri(),
                "page_count": page_count,
            },
        )

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = uuid4()

            connection.execute(
                text(
                    """
                    INSERT INTO document_chunks (
                        id,
                        document_version_id,
                        chunk_index,
                        content,
                        char_count,
                        page_start,
                        page_end,
                        chunking_strategy,
                        chunking_version,
                        metadata,
                        created_at
                    )
                    VALUES (
                        :id,
                        :document_version_id,
                        :chunk_index,
                        :content,
                        :char_count,
                        :page_start,
                        :page_end,
                        'kaliok',
                        '1',
                        '{}'::jsonb,
                        now()
                    )
                    """
                ),
                {
                    "id": chunk_id,
                    "document_version_id": version_id,
                    "chunk_index": chunk.index,
                    "content": chunk.text,
                    "char_count": len(chunk.text),
                    "page_start": chunk.page,
                    "page_end": chunk.page,
                },
            )

            connection.execute(
                insert_embedding_query,
                {
                    "chunk_id": chunk_id,
                    "embedding_model_id": model_id,
                    "embedding": embedding,
                },
            )

    return (
        source_id,
        document_id,
        version_id,
        model_id,
        model_created,
    )


def cleanup(
    source_id: UUID,
    document_id: UUID,
    version_id: UUID,
    model_id: UUID,
    model_created: bool,
) -> None:
    engine = create_database_engine()

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                DELETE FROM chunk_embeddings
                WHERE chunk_id IN (
                    SELECT id
                    FROM document_chunks
                    WHERE document_version_id = :version_id
                )
                """
            ),
            {
                "version_id": version_id,
            },
        )

        connection.execute(
            text(
                """
                DELETE FROM document_chunks
                WHERE document_version_id = :version_id
                """
            ),
            {
                "version_id": version_id,
            },
        )

        connection.execute(
            text(
                """
                DELETE FROM document_versions
                WHERE id = :version_id
                """
            ),
            {
                "version_id": version_id,
            },
        )

        connection.execute(
            text(
                """
                DELETE FROM documents
                WHERE id = :document_id
                """
            ),
            {
                "document_id": document_id,
            },
        )

        connection.execute(
            text(
                """
                DELETE FROM sources
                WHERE id = :source_id
                """
            ),
            {
                "source_id": source_id,
            },
        )

        if model_created:
            connection.execute(
                text(
                    """
                    DELETE FROM embedding_models
                    WHERE id = :model_id
                    """
                ),
                {
                    "model_id": model_id,
                },
            )


def main() -> None:
    pdf_path = (
        Path(sys.argv[1])
        if len(sys.argv) >= 2
        else DEFAULT_PDF
    )

    question = (
        sys.argv[2]
        if len(sys.argv) >= 3
        else DEFAULT_QUESTION
    )

    pdf_path = pdf_path.resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Document introuvable : {pdf_path}"
        )

    print(f"Document : {pdf_path}")
    print(f"Modèle embedding : {EMBEDDING_MODEL}")

    print("\n1. Lecture du document...")

    document_content = read_document(pdf_path)

    print(
        f"   Pages lues : {document_content.page_count}"
    )
    print(
        f"   Blocs extraits : {len(document_content.blocks)}"
    )
    print(
        f"   Caractères extraits : {len(document_content.text)}"
    )

    print("\n2. Chunking...")

    chunks = chunk_document(document_content)

    if not chunks:
        raise RuntimeError(
            "Le chunking n'a produit aucun chunk."
        )

    print(f"   Chunks produits : {len(chunks)}")

    print("\n3. Génération des embeddings...")

    embeddings = []

    for position, chunk in enumerate(
        chunks,
        start=1,
    ):
        print(
            f"   Embedding {position}/{len(chunks)}",
            end="\r",
        )

        embedding = embed_text(chunk.text)

        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                f"Dimension inattendue : {len(embedding)}"
            )

        embeddings.append(embedding)

    print()
    print(
        f"   {len(embeddings)} embeddings générés "
        f"en {EMBEDDING_DIMENSIONS} dimensions."
    )

    ids = None

    try:
        print("\n4. Enregistrement PostgreSQL...")

        ids = store_document(
            pdf_path=pdf_path,
            chunks=chunks,
            embeddings=embeddings,
            page_count=document_content.page_count,
        )

        (
            source_id,
            document_id,
            version_id,
            model_id,
            model_created,
        ) = ids

        print(f"   Document ID : {document_id}")
        print(f"   Version ID  : {version_id}")

        print("\n5. Question :")
        print(f"   {question}")

        print("\n6. Embedding de la question...")

        query_embedding = embed_text(question)

        print("\n7. Recherche pgvector...")

        results = search_similar_chunks(
            query_embedding=query_embedding,
            embedding_model_id=model_id,
            document_version_id=version_id,
            limit=5,
        )

        print("\nRésultats :")

        for position, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"\n--- Résultat {position} "
                f"| distance={result.distance:.4f} ---"
            )

            print(result.content)

    finally:
        if ids is not None:
            print(
                "\n8. Nettoyage des données de test..."
            )

            cleanup(
                source_id=source_id,
                document_id=document_id,
                version_id=version_id,
                model_id=model_id,
                model_created=model_created,
            )

            print("   Nettoyage terminé.")


if __name__ == "__main__":
    main()
