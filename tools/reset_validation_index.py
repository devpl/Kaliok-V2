from __future__ import annotations

import json

from sqlmodel import Session, delete, select

from kaliok.paths import TEST_DOCUMENTS
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    ChunkEmbedding,
    Document,
    DocumentChunk,
    DocumentVersion,
)


VALIDATION_DIR = TEST_DOCUMENTS / "validation"
DATASET_PATH = VALIDATION_DIR / "qa_validation.json"


def main() -> None:
    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    filenames = {
        document["file"]
        for document in dataset["documents"]
    }

    engine = create_database_engine()

    with Session(engine) as session:
        documents = session.exec(
            select(Document).where(
                Document.external_id.in_(filenames)
            )
        ).all()

        if not documents:
            print(
                "Aucun document de validation indexé."
            )
            return

        document_ids = [
            document.id
            for document in documents
        ]

        versions = session.exec(
            select(DocumentVersion).where(
                DocumentVersion.document_id.in_(
                    document_ids
                )
            )
        ).all()

        version_ids = [
            version.id
            for version in versions
        ]

        chunks = session.exec(
            select(DocumentChunk).where(
                DocumentChunk.document_version_id.in_(
                    version_ids
                )
            )
        ).all()

        chunk_ids = [
            chunk.id
            for chunk in chunks
        ]

        if chunk_ids:
            session.exec(
                delete(ChunkEmbedding).where(
                    ChunkEmbedding.chunk_id.in_(
                        chunk_ids
                    )
                )
            )

            session.exec(
                delete(DocumentChunk).where(
                    DocumentChunk.id.in_(
                        chunk_ids
                    )
                )
            )

        if version_ids:
            session.exec(
                delete(DocumentVersion).where(
                    DocumentVersion.id.in_(
                        version_ids
                    )
                )
            )

        session.commit()

    print(
        "Index de validation supprimé."
    )
    print(
        f"Documents concernés : {len(documents)}"
    )
    print(
        f"Versions supprimées : {len(version_ids)}"
    )
    print(
        f"Chunks supprimés    : {len(chunk_ids)}"
    )
    print()
    print(
        "Les objets Document sont conservés."
    )


if __name__ == "__main__":
    main()
