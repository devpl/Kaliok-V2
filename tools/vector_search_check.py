from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text

from kaliok.embeddings.ollama import EMBEDDING_MODEL, embed_text
from kaliok.embeddings.service import search_similar_chunks
from kaliok.storage.database import create_database_engine


EMBEDDING_DIMENSIONS = 1024


def main():
    engine = create_database_engine()

    samples = [
        "Le contrat prend effet le 5 août 2026.",
        "La facture doit être réglée sous trente jours.",
        "Le bien immobilier est situé à Évreux.",
    ]

    query = "Quelle est la date de début du contrat ?"

    embeddings = [embed_text(sample) for sample in samples]
    query_embedding = embed_text(query)

    print(f"Dimension embedding : {len(query_embedding)}")

    with engine.begin() as connection:
        source_id = uuid4()
        document_id = uuid4()
        version_id = uuid4()
        model_id = uuid4()

        connection.execute(
            text(
                """
                INSERT INTO sources (
                    id,
                    name,
                    source_type,
                    configuration,
                    is_active,
                    created_at,
                    updated_at
                )
                VALUES (
                    :id,
                    'test-vector-search',
                    'test',
                    '{}'::jsonb,
                    true,
                    now(),
                    now()
                )
                """
            ),
            {
                "id": source_id,
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
                    metadata,
                    created_at,
                    updated_at
                )
                VALUES (
                    :id,
                    :source_id,
                    'Test vector search',
                    'active',
                    '{}'::jsonb,
                    now(),
                    now()
                )
                """
            ),
            {
                "id": document_id,
                "source_id": source_id,
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
                    file_hash,
                    storage_uri,
                    version_status,
                    processing_status,
                    readability_status,
                    is_current,
                    metadata,
                    created_at
                )
                VALUES (
                    :id,
                    :document_id,
                    1,
                    'test-vector-search.txt',
                    'test-vector-search',
                    'memory://test-vector-search',
                    'published',
                    'completed',
                    'readable',
                    true,
                    '{}'::jsonb,
                    now()
                )
                """
            ),
            {
                "id": version_id,
                "document_id": document_id,
            },
        )

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

        for index, (sample, embedding) in enumerate(
            zip(samples, embeddings)
        ):
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
                        chunking_strategy,
                        metadata,
                        created_at
                    )
                    VALUES (
                        :id,
                        :document_version_id,
                        :chunk_index,
                        :content,
                        :char_count,
                        'test',
                        '{}'::jsonb,
                        now()
                    )
                    """
                ),
                {
                    "id": chunk_id,
                    "document_version_id": version_id,
                    "chunk_index": index,
                    "content": sample,
                    "char_count": len(sample),
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

    results = search_similar_chunks(
        query_embedding=query_embedding,
        embedding_model_id=model_id,
        limit=3,
    )

    print("\nQuestion :")
    print(query)

    print("\nRésultats :")
    for result in results:
        print(
            f"- distance={result.distance:.4f} | "
            f"{result.content}"
        )


if __name__ == "__main__":
    main()
