from sqlalchemy import text

from kaliok.storage.database import create_database_engine


def main():
    engine = create_database_engine()

    with engine.connect() as connection:
        tables = connection.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN (
                      'documents',
                      'document_versions',
                      'document_chunks',
                      'chunk_embeddings',
                      'embedding_models'
                  )
                ORDER BY table_name
                """
            )
        ).scalars().all()

        columns = connection.execute(
            text(
                """
                SELECT
                    a.attname AS column_name,
                    pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type
                FROM pg_catalog.pg_attribute a
                JOIN pg_catalog.pg_class c
                    ON c.oid = a.attrelid
                JOIN pg_catalog.pg_namespace n
                    ON n.oid = c.relnamespace
                WHERE n.nspname = 'public'
                  AND c.relname = 'chunk_embeddings'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                ORDER BY a.attnum
                """
            )
        ).all()

        indexes = connection.execute(
            text(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                  AND tablename = 'chunk_embeddings'
                ORDER BY indexname
                """
            )
        ).all()

    print("Tables kaliok :")
    for table in tables:
        print(f"  - {table}")

    print("\nColonnes chunk_embeddings :")
    for column_name, data_type in columns:
        print(f"  - {column_name}: {data_type}")

    print("\nIndex chunk_embeddings :")
    for index_name, index_def in indexes:
        print(f"  - {index_name}")
        print(f"    {index_def}")


if __name__ == "__main__":
    main()
