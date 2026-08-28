from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlmodel import Session, select

from kaliok.documents.cleaning import clean_document
from kaliok.documents.reader import read_document
from kaliok.documents.semantic_chunking import (
    chunk_document_semantically,
)
from kaliok.embeddings.ollama import (
    EMBEDDING_MODEL,
    embed_texts,
)
from kaliok.hashing import calculate_sha256
from kaliok.paths import TEST_DOCUMENTS
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    ChunkContentBlock,
    ChunkEmbedding,
    ContentBlock,
    Document,
    DocumentChunk,
    DocumentVersion,
    EmbeddingModel,
    Page,
    ProcessingRun,
    Source,
    utc_now,
)


EMBEDDING_DIMENSIONS = 1024
CHUNKING_STRATEGY = "llamaindex-semantic-cleaning"
CHUNKING_VERSION = "1"
PROCESS_TYPE = "document_extraction"
PERCEPTION_ENGINE = "kaliok-reader"
PERCEPTION_VERSION = "3"


@dataclass
class IndexDocumentResult:
    document_id: UUID
    document_version_id: UUID
    embedding_model_id: UUID
    chunk_count: int
    already_indexed: bool


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

    return (
        max(
            version.version_number
            for version in versions
        )
        + 1
    )


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


def count_chunks(
    session: Session,
    document_version_id: UUID,
) -> int:
    chunks = session.exec(
        select(DocumentChunk).where(
            DocumentChunk.document_version_id
            == document_version_id
        )
    ).all()

    return len(chunks)


def count_pages(
    session: Session,
    document_version_id: UUID,
) -> int:
    pages = session.exec(
        select(Page).where(
            Page.document_version_id
            == document_version_id
        )
    ).all()

    return len(pages)


def get_perception_storage_state(
    session: Session,
    version: DocumentVersion,
) -> str:
    pages = session.exec(
        select(Page).where(
            Page.document_version_id
            == version.id
        )
    ).all()

    if not pages:
        return "missing"

    if (
        version.page_count is None
        or len(pages) != version.page_count
    ):
        return "partial"

    page_metadata_complete = all(
        page.width is not None
        and page.height is not None
        and page.readability_score is not None
        and page.readability_status
        not in {
            None,
            "unknown",
        }
        and page.perception_mode
        not in {
            None,
            "unknown",
        }
        for page in pages
    )

    if not page_metadata_complete:
        return "page_metadata_missing"

    runs = session.exec(
        select(ProcessingRun).where(
            ProcessingRun.document_version_id
            == version.id,
            ProcessingRun.process_type
            == PROCESS_TYPE,
            ProcessingRun.status
            == "completed",
        )
    ).all()

    current_runs = [
        run
        for run in runs
        if run.engine == PERCEPTION_ENGINE
        and run.engine_version
        == PERCEPTION_VERSION
    ]

    if not current_runs:
        return "perception_stale"

    current_run_ids = {
        run.id
        for run in current_runs
    }

    if not all(
        page.perception_processing_run_id
        in current_run_ids
        for page in pages
    ):
        return "perception_stale"

    current_blocks = session.exec(
        select(ContentBlock).where(
            ContentBlock.processing_run_id.in_(
                current_run_ids
            )
        )
    ).all()

    if not current_blocks:
        return "perception_stale"

    page_ids = {
        page.id
        for page in pages
    }

    if not all(
        block.page_id in page_ids
        for block in current_blocks
    ):
        return "perception_stale"

    pages_with_current_blocks = {
        block.page_id
        for block in current_blocks
    }

    if not page_ids.issubset(
        pages_with_current_blocks
    ):
        return "perception_stale"

    return "complete"


def get_index_storage_state(
    session: Session,
    version: DocumentVersion,
    embedding_model: EmbeddingModel,
) -> str:
    chunks = session.exec(
        select(DocumentChunk).where(
            DocumentChunk.document_version_id
            == version.id
        )
    ).all()

    if not chunks:
        return "chunks_missing"

    chunk_indices = sorted(
        chunk.chunk_index
        for chunk in chunks
    )

    if chunk_indices != list(range(len(chunks))):
        return "chunks_incomplete"

    chunk_ids = {
        chunk.id
        for chunk in chunks
    }

    embeddings = session.exec(
        select(ChunkEmbedding).where(
            ChunkEmbedding.chunk_id.in_(chunk_ids),
            ChunkEmbedding.embedding_model_id
            == embedding_model.id,
        )
    ).all()

    embedded_chunk_ids = {
        embedding.chunk_id
        for embedding in embeddings
    }

    if embedded_chunk_ids != chunk_ids:
        return "embeddings_incomplete"

    pages = session.exec(
        select(Page).where(
            Page.document_version_id
            == version.id
        )
    ).all()

    current_run_by_page_id = {
        page.id: page.perception_processing_run_id
        for page in pages
    }

    blocks = session.exec(
        select(ContentBlock).where(
            ContentBlock.page_id.in_(
                current_run_by_page_id
            )
        )
    ).all()

    current_block_ids = {
        block.id
        for block in blocks
        if block.processing_run_id
        == current_run_by_page_id.get(block.page_id)
    }

    links = session.exec(
        select(ChunkContentBlock).where(
            ChunkContentBlock.chunk_id.in_(
                chunk_ids
            )
        )
    ).all()

    linked_chunk_ids = {
        link.chunk_id
        for link in links
        if link.content_block_id
        in current_block_ids
    }

    if linked_chunk_ids != chunk_ids:
        return "content_block_links_incomplete"

    return "complete"


def _validated_index_chunk_count(
    session: Session,
    version: DocumentVersion,
    embedding_model: EmbeddingModel,
) -> int:
    perception_state = get_perception_storage_state(
        session,
        version,
    )

    if perception_state != "complete":
        raise RuntimeError(
            "Perception incomplète : "
            f"{perception_state}."
        )

    index_state = get_index_storage_state(
        session,
        version,
        embedding_model,
    )

    if index_state != "complete":
        raise RuntimeError(
            "Index incomplet : "
            f"{index_state}."
        )

    return count_chunks(
        session,
        version.id,
    )


def print_duration(
    label: str,
    duration: float,
) -> None:
    print(
        f"  {label:<22}: {duration:>8.2f} s"
    )


def _blocks_by_page(
    document_content,
) -> dict[int, list[tuple[int, object]]]:
    grouped: dict[
        int,
        list[tuple[int, object]],
    ] = defaultdict(list)

    for block_index, block in enumerate(
        document_content.blocks
    ):
        grouped[block.page].append(
            (
                block_index,
                block,
            )
        )

    return dict(grouped)


def _page_perception_mode(
    blocks: list[tuple[int, object]],
) -> str:
    has_native_text = any(
        block.extraction_method == "native"
        and bool(block.text.strip())
        for _, block in blocks
    )

    has_ocr_text = any(
        block.extraction_method == "ocr"
        and bool(block.text.strip())
        for _, block in blocks
    )

    if has_native_text and has_ocr_text:
        return "mixed"

    if has_native_text:
        return "native"

    if has_ocr_text:
        return "ocr"

    return "none"


def _document_is_readable(
    document_content,
) -> bool:
    return any(
        bool(block.text.strip())
        for block in document_content.blocks
    )


def _page_info_by_number(
    document_content,
) -> dict[int, object]:
    return {
        page.page: page
        for page in document_content.pages
    }


def _apply_document_page_to_storage(
    stored_page: Page,
    document_page,
    processing_run_id: UUID | None,
) -> None:
    stored_page.width = document_page.width
    stored_page.height = document_page.height

    stored_page.has_native_text = (
        document_page.native_text_length > 0
    )
    stored_page.native_text_length = (
        document_page.native_text_length
    )

    stored_page.readability_status = (
        document_page.readability_status
    )
    stored_page.readability_score = (
        document_page.readability_score
    )
    stored_page.readability_reason = (
        document_page.readability_reason
    )

    stored_page.perception_mode = (
        document_page.perception_mode
    )

    stored_page.ocr_required = (
        document_page.ocr_required
    )
    stored_page.ocr_performed = (
        document_page.ocr_performed
    )
    stored_page.ocr_reason = (
        document_page.ocr_reason
    )
    stored_page.ocr_engine = (
        document_page.ocr_engine
    )
    stored_page.ocr_confidence_mean = (
        document_page.ocr_confidence_mean
    )

    if document_page.ocr_performed:
        stored_page.ocr_processing_run_id = (
            processing_run_id
        )
    else:
        stored_page.ocr_processing_run_id = None


def _enrich_existing_page_metadata(
    session: Session,
    version: DocumentVersion,
    document_content,
) -> tuple[
    ProcessingRun,
    dict[int, ContentBlock],
]:
    return _store_new_perception_on_existing_pages(
        session,
        version,
        document_content,
    )


def _store_perception(
    session: Session,
    version: DocumentVersion,
    document_content,
) -> tuple[
    ProcessingRun,
    dict[int, ContentBlock],
]:
    processing_run = ProcessingRun(
        document_version_id=version.id,
        process_type=PROCESS_TYPE,
        status="completed",
        engine=PERCEPTION_ENGINE,
        engine_version=PERCEPTION_VERSION,
        completed_at=utc_now(),
    )

    session.add(processing_run)
    session.flush()

    grouped = _blocks_by_page(
        document_content
    )

    page_info = _page_info_by_number(
        document_content
    )

    stored_blocks_by_source_index: dict[
        int,
        ContentBlock,
    ] = {}

    for page_number in range(
        1,
        document_content.page_count + 1,
    ):
        page_blocks = grouped.get(
            page_number,
            [],
        )

        document_page = page_info.get(
            page_number
        )

        if document_page is None:
            raise RuntimeError(
                "Informations de perception "
                f"absentes pour la page {page_number}."
            )

        stored_page = Page(
            document_version_id=version.id,
            page_number=page_number,
            page_status="active",
        )

        _apply_document_page_to_storage(
            stored_page,
            document_page,
            processing_run.id,
        )
        stored_page.perception_processing_run_id = (
            processing_run.id
        )

        session.add(stored_page)
        session.flush()

        for reading_order, (
            source_block_index,
            block,
        ) in enumerate(
            page_blocks
        ):
            stored_block = ContentBlock(
                page_id=stored_page.id,
                processing_run_id=(
                    processing_run.id
                ),
                block_index=source_block_index,
                reading_order=reading_order,
                block_type="text",
                content=block.text,
                extraction_method=(
                    block.extraction_method
                ),
                extraction_engine=(
                    block.extraction_engine
                ),
                extraction_engine_version=(
                    block.extraction_engine_version
                ),
                confidence=block.confidence,
                bbox_x=block.bbox_x,
                bbox_y=block.bbox_y,
                bbox_width=block.bbox_width,
                bbox_height=block.bbox_height,
                coordinate_system=(
                    block.coordinate_system
                ),
            )

            session.add(stored_block)
            session.flush()

            stored_blocks_by_source_index[
                source_block_index
            ] = stored_block

    return (
        processing_run,
        stored_blocks_by_source_index,
    )



def _store_new_perception_on_existing_pages(
    session: Session,
    version: DocumentVersion,
    document_content,
) -> tuple[
    ProcessingRun,
    dict[int, ContentBlock],
]:
    page_info = _page_info_by_number(
        document_content
    )
    grouped = _blocks_by_page(
        document_content
    )

    stored_pages = session.exec(
        select(Page).where(
            Page.document_version_id
            == version.id
        )
    ).all()

    if len(stored_pages) != document_content.page_count:
        raise RuntimeError(
            "Nouvelle perception impossible : "
            "le nombre de pages stockées ne "
            "correspond pas au document."
        )

    stored_pages_by_number = {
        page.page_number: page
        for page in stored_pages
    }

    processing_run = ProcessingRun(
        document_version_id=version.id,
        process_type=PROCESS_TYPE,
        status="completed",
        engine=PERCEPTION_ENGINE,
        engine_version=PERCEPTION_VERSION,
        completed_at=utc_now(),
    )

    session.add(processing_run)
    session.flush()

    stored_blocks_by_source_index: dict[
        int,
        ContentBlock,
    ] = {}

    for page_number in range(
        1,
        document_content.page_count + 1,
    ):
        stored_page = stored_pages_by_number.get(
            page_number
        )
        document_page = page_info.get(
            page_number
        )

        if stored_page is None or document_page is None:
            raise RuntimeError(
                "Nouvelle perception impossible : "
                f"page {page_number} introuvable."
            )

        _apply_document_page_to_storage(
            stored_page,
            document_page,
            processing_run.id,
        )
        stored_page.perception_processing_run_id = (
            processing_run.id
        )
        session.add(stored_page)

        for reading_order, (
            source_block_index,
            block,
        ) in enumerate(
            grouped.get(
                page_number,
                [],
            )
        ):
            stored_block = ContentBlock(
                page_id=stored_page.id,
                processing_run_id=(
                    processing_run.id
                ),
                block_index=source_block_index,
                reading_order=reading_order,
                block_type="text",
                content=block.text,
                extraction_method=(
                    block.extraction_method
                ),
                extraction_engine=(
                    block.extraction_engine
                ),
                extraction_engine_version=(
                    block.extraction_engine_version
                ),
                confidence=block.confidence,
                bbox_x=block.bbox_x,
                bbox_y=block.bbox_y,
                bbox_width=block.bbox_width,
                bbox_height=block.bbox_height,
                coordinate_system=(
                    block.coordinate_system
                ),
            )

            session.add(stored_block)
            session.flush()

            stored_blocks_by_source_index[
                source_block_index
            ] = stored_block

    return (
        processing_run,
        stored_blocks_by_source_index,
    )

def _get_existing_chunks_by_index(
    session: Session,
    document_version_id: UUID,
) -> dict[int, DocumentChunk]:
    stored_chunks = session.exec(
        select(DocumentChunk).where(
            DocumentChunk.document_version_id
            == document_version_id
        )
    ).all()

    return {
        chunk.chunk_index: chunk
        for chunk in stored_chunks
    }


def _link_existing_chunks_to_blocks(
    session: Session,
    chunks,
    stored_chunks_by_index: dict[
        int,
        DocumentChunk,
    ],
    stored_blocks_by_source_index: dict[
        int,
        ContentBlock,
    ],
) -> None:
    if len(stored_chunks_by_index) != len(chunks):
        raise RuntimeError(
            "Backfill impossible sans ambiguïté : "
            "le nombre de chunks stockés diffère "
            "du chunking actuel."
        )

    for chunk in chunks:
        stored_chunk = (
            stored_chunks_by_index.get(
                chunk.index
            )
        )

        if stored_chunk is None:
            raise RuntimeError(
                "Backfill impossible : "
                f"chunk_index={chunk.index} absent."
            )

        if stored_chunk.content != chunk.text:
            raise RuntimeError(
                "Backfill impossible sans ambiguïté : "
                f"le contenu du chunk {chunk.index} "
                "diffère du chunking actuel."
            )

        stored_block = (
            stored_blocks_by_source_index.get(
                chunk.source_block_index
            )
        )

        if stored_block is None:
            raise RuntimeError(
                "Bloc source introuvable pour "
                f"le chunk {chunk.index}."
            )

        existing_links = session.exec(
            select(ChunkContentBlock).where(
                ChunkContentBlock.chunk_id
                == stored_chunk.id
            )
        ).all()

        for existing_link in existing_links:
            session.delete(existing_link)

        session.flush()

        session.add(
            ChunkContentBlock(
                chunk_id=stored_chunk.id,
                content_block_id=stored_block.id,
                block_order=0,
            )
        )


def index_document(
    path: str | Path,
    *,
    verbose: bool = True,
) -> IndexDocumentResult:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document introuvable : {path}"
        )

    total_start = time.perf_counter()

    if verbose:
        print()
        print(f"Document : {path.name}")

    file_hash = calculate_sha256(path)
    engine = create_database_engine()

    existing_version_id: UUID | None = None
    perception_backfill = False
    page_metadata_backfill = False
    perception_refresh = False

    # ---------------------------------------------------------
    # Vérification avant traitement coûteux
    # ---------------------------------------------------------

    with Session(engine) as session:
        source = get_or_create_source(session)

        document = get_or_create_document(
            session,
            source,
            path,
        )

        embedding_model = (
            get_or_create_embedding_model(
                session
            )
        )

        existing_version = (
            find_existing_version(
                session,
                document,
                file_hash,
            )
        )

        if existing_version is not None:
            perception_state = (
                get_perception_storage_state(
                    session,
                    existing_version,
                )
            )

            if perception_state == "complete":
                chunk_count = (
                    _validated_index_chunk_count(
                        session,
                        existing_version,
                        embedding_model,
                    )
                )
                session.commit()

                if verbose:
                    print("  Déjà indexé.")
                    print(
                        "  DocumentVersion : "
                        f"{existing_version.id}"
                    )
                    print(
                        f"  Chunks          : "
                        f"{chunk_count}"
                    )
                    print_duration(
                        "TOTAL",
                        time.perf_counter()
                        - total_start,
                    )

                return IndexDocumentResult(
                    document_id=document.id,
                    document_version_id=(
                        existing_version.id
                    ),
                    embedding_model_id=(
                        embedding_model.id
                    ),
                    chunk_count=chunk_count,
                    already_indexed=True,
                )

            if perception_state == "partial":
                raise RuntimeError(
                    "État de perception partiel en base : "
                    "arrêt pour éviter tout doublon."
                )

            existing_version_id = (
                existing_version.id
            )

            perception_backfill = (
                perception_state == "missing"
            )

            page_metadata_backfill = (
                perception_state
                == "page_metadata_missing"
            )

            perception_refresh = (
                perception_state
                == "perception_stale"
            )

            if verbose:
                if perception_backfill:
                    print(
                        "  Version existante sans "
                        "perception relationnelle : "
                        "backfill complet."
                    )
                elif page_metadata_backfill:
                    print(
                        "  Métadonnées de page "
                        "incomplètes : enrichissement."
                    )
                elif perception_refresh:
                    print(
                        "  Perception ancienne : "
                        "nouvelle génération de blocs."
                    )

        session.commit()

    # ---------------------------------------------------------
    # Lecture
    # ---------------------------------------------------------

    if verbose:
        print("  Lecture...")

    start = time.perf_counter()

    document_content = read_document(path)

    lecture_duration = (
        time.perf_counter() - start
    )

    if page_metadata_backfill:
        start = time.perf_counter()

        cleaned_document = clean_document(
            document_content
        )

        chunks = chunk_document_semantically(
            cleaned_document,
            breakpoint_percentile_threshold=95,
            buffer_size=1,
        )

        if not chunks:
            raise RuntimeError(
                "Aucun chunk généré pour "
                f"{path.name}."
            )

        with Session(engine) as session:
            source = get_or_create_source(session)

            document = get_or_create_document(
                session,
                source,
                path,
            )

            embedding_model = (
                get_or_create_embedding_model(
                    session
                )
            )

            existing_version = (
                find_existing_version(
                    session,
                    document,
                    file_hash,
                )
            )

            if (
                existing_version is None
                or existing_version.id
                != existing_version_id
            ):
                raise RuntimeError(
                    "La version à enrichir "
                    "n'est plus disponible."
                )

            current_state = (
                get_perception_storage_state(
                    session,
                    existing_version,
                )
            )

            if current_state == "complete":
                chunk_count = (
                    _validated_index_chunk_count(
                        session,
                        existing_version,
                        embedding_model,
                    )
                )
                session.rollback()

                return IndexDocumentResult(
                    document_id=document.id,
                    document_version_id=(
                        existing_version.id
                    ),
                    embedding_model_id=(
                        embedding_model.id
                    ),
                    chunk_count=chunk_count,
                    already_indexed=True,
                )

            if current_state != "page_metadata_missing":
                raise RuntimeError(
                    "État de perception inattendu "
                    "pendant l'enrichissement."
                )

            stored_chunks_by_index = (
                _get_existing_chunks_by_index(
                    session,
                    existing_version.id,
                )
            )

            (
                _,
                stored_blocks_by_source_index,
            ) = _enrich_existing_page_metadata(
                session,
                existing_version,
                document_content,
            )

            _link_existing_chunks_to_blocks(
                session,
                chunks,
                stored_chunks_by_index,
                stored_blocks_by_source_index,
            )

            session.commit()

            chunk_count = count_chunks(
                session,
                existing_version.id,
            )

            result = IndexDocumentResult(
                document_id=document.id,
                document_version_id=(
                    existing_version.id
                ),
                embedding_model_id=(
                    embedding_model.id
                ),
                chunk_count=chunk_count,
                already_indexed=False,
            )

        database_duration = (
            time.perf_counter() - start
        )

        total_duration = (
            time.perf_counter() - total_start
        )

        if verbose:
            print(
                "  Métadonnées de page enrichies."
            )
            print()
            print_duration(
                "Lecture",
                lecture_duration,
            )
            print_duration(
                "PostgreSQL",
                database_duration,
            )
            print_duration(
                "TOTAL",
                total_duration,
            )

        return result

    # ---------------------------------------------------------
    # Nettoyage
    # ---------------------------------------------------------

    if verbose:
        print("  Nettoyage...")

    start = time.perf_counter()

    cleaned_document = clean_document(
        document_content
    )

    cleaning_duration = (
        time.perf_counter() - start
    )

    # ---------------------------------------------------------
    # Chunking sémantique
    # ---------------------------------------------------------

    if verbose:
        print("  Chunking sémantique...")

    start = time.perf_counter()

    chunks = chunk_document_semantically(
        cleaned_document,
        breakpoint_percentile_threshold=95,
        buffer_size=1,
    )

    chunking_duration = (
        time.perf_counter() - start
    )

    if not chunks:
        raise RuntimeError(
            "Aucun chunk généré pour "
            f"{path.name}."
        )

    if verbose:
        print(
            f"  Chunks : {len(chunks)}"
        )

    # ---------------------------------------------------------
    # Embeddings par lot
    # ---------------------------------------------------------

    embeddings: list[list[float]] = []
    embedding_duration = 0.0

    if not perception_backfill and not perception_refresh:
        if verbose:
            print(
                "  Génération des embeddings "
                "par lot..."
            )

        start = time.perf_counter()

        embeddings = embed_texts(
            [
                chunk.text
                for chunk in chunks
            ]
        )

        embedding_duration = (
            time.perf_counter() - start
        )

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Nombre d'embeddings inattendu : "
                f"{len(embeddings)} pour "
                f"{len(chunks)} chunks."
            )
    elif verbose:
        print(
            "  Embeddings existants : "
            "aucune régénération."
        )

    # ---------------------------------------------------------
    # PostgreSQL
    # ---------------------------------------------------------

    start = time.perf_counter()

    with Session(engine) as session:
        source = get_or_create_source(session)

        document = get_or_create_document(
            session,
            source,
            path,
        )

        embedding_model = (
            get_or_create_embedding_model(
                session
            )
        )

        existing_version = (
            find_existing_version(
                session,
                document,
                file_hash,
            )
        )

        if perception_refresh:
            if (
                existing_version is None
                or existing_version.id
                != existing_version_id
            ):
                raise RuntimeError(
                    "La version à rafraîchir "
                    "n'est plus disponible."
                )

            current_state = (
                get_perception_storage_state(
                    session,
                    existing_version,
                )
            )

            if current_state == "complete":
                chunk_count = (
                    _validated_index_chunk_count(
                        session,
                        existing_version,
                        embedding_model,
                    )
                )
                session.rollback()

                return IndexDocumentResult(
                    document_id=document.id,
                    document_version_id=(
                        existing_version.id
                    ),
                    embedding_model_id=(
                        embedding_model.id
                    ),
                    chunk_count=chunk_count,
                    already_indexed=True,
                )

            if current_state != "perception_stale":
                raise RuntimeError(
                    "État de perception inattendu "
                    "pendant le rafraîchissement."
                )

            version = existing_version

            stored_chunks_by_index = (
                _get_existing_chunks_by_index(
                    session,
                    version.id,
                )
            )

            # Validation stricte avant toute écriture.
            # Si la perception v3 modifie le chunking,
            # on arrête sans remplacer l'historique.
            if len(stored_chunks_by_index) != len(chunks):
                raise RuntimeError(
                    "Perception v3 non appliquée : "
                    "le nombre de chunks a changé. "
                    "Une nouvelle version de chunking "
                    "sera nécessaire."
                )

            for chunk in chunks:
                stored_chunk = (
                    stored_chunks_by_index.get(
                        chunk.index
                    )
                )

                if (
                    stored_chunk is None
                    or stored_chunk.content
                    != chunk.text
                ):
                    raise RuntimeError(
                        "Perception v3 non appliquée : "
                        f"le chunk {chunk.index} a changé. "
                        "Aucune donnée existante n'a "
                        "été remplacée."
                    )

            (
                _,
                stored_blocks_by_source_index,
            ) = _store_new_perception_on_existing_pages(
                session,
                version,
                document_content,
            )

            _link_existing_chunks_to_blocks(
                session,
                chunks,
                stored_chunks_by_index,
                stored_blocks_by_source_index,
            )

            version.processing_status = "completed"
            version.readability_status = (
                "readable"
                if _document_is_readable(
                    document_content
                )
                else "unreadable"
            )
            version.processed_at = utc_now()

            session.add(version)
            session.commit()

            result = IndexDocumentResult(
                document_id=document.id,
                document_version_id=version.id,
                embedding_model_id=(
                    embedding_model.id
                ),
                chunk_count=len(chunks),
                already_indexed=False,
            )

        elif perception_backfill:
            if (
                existing_version is None
                or existing_version.id
                != existing_version_id
            ):
                raise RuntimeError(
                    "La version à backfiller "
                    "n'est plus disponible."
                )

            current_state = (
                get_perception_storage_state(
                    session,
                    existing_version,
                )
            )

            if current_state == "complete":
                chunk_count = (
                    _validated_index_chunk_count(
                        session,
                        existing_version,
                        embedding_model,
                    )
                )

                session.rollback()

                return IndexDocumentResult(
                    document_id=document.id,
                    document_version_id=(
                        existing_version.id
                    ),
                    embedding_model_id=(
                        embedding_model.id
                    ),
                    chunk_count=chunk_count,
                    already_indexed=True,
                )

            if current_state != "missing":
                raise RuntimeError(
                    "État de perception inattendu "
                    "pendant le backfill complet."
                )

            version = existing_version

            (
                _,
                stored_blocks_by_source_index,
            ) = _store_perception(
                session,
                version,
                document_content,
            )

            stored_chunks_by_index = (
                _get_existing_chunks_by_index(
                    session,
                    version.id,
                )
            )

            _link_existing_chunks_to_blocks(
                session,
                chunks,
                stored_chunks_by_index,
                stored_blocks_by_source_index,
            )

            version.processing_status = (
                "completed"
            )
            version.readability_status = (
                "readable"
                if _document_is_readable(
                    document_content
                )
                else "unreadable"
            )
            version.processed_at = utc_now()

            session.add(version)
            session.commit()

            result = IndexDocumentResult(
                document_id=document.id,
                document_version_id=version.id,
                embedding_model_id=(
                    embedding_model.id
                ),
                chunk_count=len(chunks),
                already_indexed=False,
            )

        else:
            if existing_version is not None:
                chunk_count = (
                    _validated_index_chunk_count(
                        session,
                        existing_version,
                        embedding_model,
                    )
                )

                session.rollback()

                return IndexDocumentResult(
                    document_id=document.id,
                    document_version_id=(
                        existing_version.id
                    ),
                    embedding_model_id=(
                        embedding_model.id
                    ),
                    chunk_count=chunk_count,
                    already_indexed=True,
                )

            versions = session.exec(
                select(DocumentVersion).where(
                    DocumentVersion.document_id
                    == document.id
                )
            ).all()

            for previous_version in versions:
                previous_version.is_current = False
                session.add(
                    previous_version
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
                filename=path.name,
                mime_type="application/pdf",
                file_hash=file_hash,
                file_size=path.stat().st_size,
                storage_uri=str(path),
                page_count=(
                    document_content.page_count
                ),
                document_type="pdf",
                version_status="active",
                processing_status="completed",
                readability_status=(
                    "readable"
                    if _document_is_readable(
                        document_content
                    )
                    else "unreadable"
                ),
                is_current=True,
                processed_at=utc_now(),
            )

            session.add(version)
            session.flush()

            (
                _,
                stored_blocks_by_source_index,
            ) = _store_perception(
                session,
                version,
                document_content,
            )

            for chunk, embedding in zip(
                chunks,
                embeddings,
            ):
                stored_chunk = DocumentChunk(
                    document_version_id=(
                        version.id
                    ),
                    chunk_index=chunk.index,
                    content=chunk.text,
                    char_count=len(chunk.text),
                    page_start=chunk.page,
                    page_end=chunk.page,
                    chunking_strategy=(
                        CHUNKING_STRATEGY
                    ),
                    chunking_version=(
                        CHUNKING_VERSION
                    ),
                )

                session.add(stored_chunk)
                session.flush()

                stored_block = (
                    stored_blocks_by_source_index.get(
                        chunk.source_block_index
                    )
                )

                if stored_block is None:
                    raise RuntimeError(
                        "Bloc source introuvable "
                        "pour le chunk "
                        f"{chunk.index}."
                    )

                session.add(
                    ChunkContentBlock(
                        chunk_id=stored_chunk.id,
                        content_block_id=(
                            stored_block.id
                        ),
                        block_order=0,
                    )
                )

                session.add(
                    ChunkEmbedding(
                        chunk_id=stored_chunk.id,
                        embedding_model_id=(
                            embedding_model.id
                        ),
                        embedding=embedding,
                    )
                )

            session.commit()

            result = IndexDocumentResult(
                document_id=document.id,
                document_version_id=version.id,
                embedding_model_id=(
                    embedding_model.id
                ),
                chunk_count=len(chunks),
                already_indexed=False,
            )

    database_duration = (
        time.perf_counter() - start
    )

    total_duration = (
        time.perf_counter() - total_start
    )

    if verbose:
        print("  Indexation terminée.")
        print()
        print_duration(
            "Lecture",
            lecture_duration,
        )
        print_duration(
            "Nettoyage",
            cleaning_duration,
        )
        print_duration(
            "Chunking sémantique",
            chunking_duration,
        )
        print_duration(
            "Embeddings batch",
            embedding_duration,
        )
        print_duration(
            "PostgreSQL",
            database_duration,
        )
        print_duration(
            "TOTAL",
            total_duration,
        )

    return result
