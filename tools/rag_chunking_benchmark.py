from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from kaliok.documents.chunking import chunk_document
from kaliok.documents.cleaning import clean_document
from kaliok.documents.llama_chunking import (
    chunk_document_with_llamaindex,
)
from kaliok.documents.reader import read_document
from kaliok.documents.semantic_chunking import (
    chunk_document_semantically,
)
from kaliok.embeddings.ollama import embed_text
from kaliok.paths import TEST_DOCUMENTS


PDF_PATH = TEST_DOCUMENTS / "RIDEAU.pdf"

QUESTIONS = [
    "Quelle est la durée de fabrication ?",
    "Quel est le prix TTC ?",
    "Quelles sont les conditions de paiement ?",
    "Quelle est la dimension de la véranda ?",
    "Quelle est la garantie indiquée ?",
]


@dataclass
class RankedChunk:
    text: str
    page: int
    index: int
    distance: float


def cosine_distance(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    norm_a = sqrt(
        sum(a * a for a in vector_a)
    )

    norm_b = sqrt(
        sum(b * b for b in vector_b)
    )

    if norm_a == 0 or norm_b == 0:
        raise ValueError(
            "Impossible de calculer une distance cosinus "
            "avec un vecteur nul."
        )

    cosine_similarity = dot_product / (norm_a * norm_b)

    return 1.0 - cosine_similarity


def embed_chunks(chunks) -> list[list[float]]:
    embeddings: list[list[float]] = []

    for position, chunk in enumerate(
        chunks,
        start=1,
    ):
        print(
            f"   Embedding {position}/{len(chunks)}",
            end="\r",
        )

        embeddings.append(
            embed_text(chunk.text)
        )

    print()

    return embeddings


def search_chunks(
    chunks,
    chunk_embeddings: list[list[float]],
    query_embedding: list[float],
    limit: int = 3,
) -> list[RankedChunk]:
    results: list[RankedChunk] = []

    for chunk, embedding in zip(
        chunks,
        chunk_embeddings,
    ):
        distance = cosine_distance(
            query_embedding,
            embedding,
        )

        results.append(
            RankedChunk(
                text=chunk.text,
                page=chunk.page,
                index=chunk.index,
                distance=distance,
            )
        )

    results.sort(
        key=lambda result: result.distance
    )

    return results[:limit]


def print_results(
    title: str,
    results: list[RankedChunk],
) -> None:
    print(f"\n{title}")
    print("-" * len(title))

    for position, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n#{position} "
            f"| page {result.page} "
            f"| chunk {result.index} "
            f"| distance={result.distance:.4f}"
        )

        print(result.text)


def main() -> None:
    print(f"Document : {PDF_PATH}")

    document = read_document(PDF_PATH)
    cleaned_document = clean_document(document)

    print(
        f"Pages      : {document.page_count}"
    )
    print(
        f"Blocs      : {len(document.blocks)}"
    )
    print(
        f"Caractères bruts    : {len(document.text)}"
    )
    print(
        f"Caractères nettoyés : {len(cleaned_document.text)}"
    )

    # ---------------------------------------------------------
    # 1. kaliok brut
    # ---------------------------------------------------------

    print("\n1. Chunking kaliok brut...")

    kaliok_chunks = chunk_document(
        document,
        max_chars=1000,
        overlap_chars=0,
    )

    print(
        f"   {len(kaliok_chunks)} chunks"
    )

    # ---------------------------------------------------------
    # 2. LlamaIndex SentenceSplitter brut
    # ---------------------------------------------------------

    print("\n2. Chunking LlamaIndex brut...")

    llama_raw_chunks = chunk_document_with_llamaindex(
        document,
        chunk_size=256,
        chunk_overlap=32,
    )

    print(
        f"   {len(llama_raw_chunks)} chunks"
    )

    # ---------------------------------------------------------
    # 3. LlamaIndex SentenceSplitter + cleaning
    # ---------------------------------------------------------

    print("\n3. Chunking LlamaIndex + cleaning...")

    llama_clean_chunks = chunk_document_with_llamaindex(
        cleaned_document,
        chunk_size=256,
        chunk_overlap=32,
    )

    print(
        f"   {len(llama_clean_chunks)} chunks"
    )

    # ---------------------------------------------------------
    # 4. SemanticSplitter + cleaning
    # ---------------------------------------------------------

    print("\n4. Chunking SemanticSplitter + cleaning...")

    semantic_chunks = chunk_document_semantically(
        cleaned_document,
        breakpoint_percentile_threshold=95,
        buffer_size=1,
    )

    print(
        f"   {len(semantic_chunks)} chunks"
    )

    # ---------------------------------------------------------
    # 5. Embeddings
    # ---------------------------------------------------------

    print(
        "\n5. Embeddings kaliok brut..."
    )

    kaliok_embeddings = embed_chunks(
        kaliok_chunks
    )

    print(
        "\n6. Embeddings LlamaIndex brut..."
    )

    llama_raw_embeddings = embed_chunks(
        llama_raw_chunks
    )

    print(
        "\n7. Embeddings LlamaIndex + cleaning..."
    )

    llama_clean_embeddings = embed_chunks(
        llama_clean_chunks
    )

    print(
        "\n8. Embeddings SemanticSplitter + cleaning..."
    )

    semantic_embeddings = embed_chunks(
        semantic_chunks
    )

    # ---------------------------------------------------------
    # 6. Benchmark
    # ---------------------------------------------------------

    print(
        "\n\n========================================"
    )
    print("BENCHMARK DE RECHERCHE")
    print(
        "========================================"
    )

    for question_number, question in enumerate(
        QUESTIONS,
        start=1,
    ):
        print(
            "\n\n"
            "========================================"
        )
        print(
            f"QUESTION {question_number}"
        )
        print(
            "========================================"
        )
        print(question)

        query_embedding = embed_text(
            question
        )

        kaliok_results = search_chunks(
            chunks=kaliok_chunks,
            chunk_embeddings=kaliok_embeddings,
            query_embedding=query_embedding,
            limit=3,
        )

        llama_raw_results = search_chunks(
            chunks=llama_raw_chunks,
            chunk_embeddings=llama_raw_embeddings,
            query_embedding=query_embedding,
            limit=3,
        )

        llama_clean_results = search_chunks(
            chunks=llama_clean_chunks,
            chunk_embeddings=llama_clean_embeddings,
            query_embedding=query_embedding,
            limit=3,
        )

        semantic_results = search_chunks(
            chunks=semantic_chunks,
            chunk_embeddings=semantic_embeddings,
            query_embedding=query_embedding,
            limit=3,
        )

        print_results(
            "KALIOK BRUT",
            kaliok_results,
        )

        print_results(
            "LLAMAINDEX BRUT",
            llama_raw_results,
        )

        print_results(
            "LLAMAINDEX + CLEANING",
            llama_clean_results,
        )

        print_results(
            "SEMANTIC SPLITTER + CLEANING",
            semantic_results,
        )

    print(
        "\n\n========================================"
    )
    print("FIN DU BENCHMARK")
    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
