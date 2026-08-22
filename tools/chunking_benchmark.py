from statistics import mean

from kaliok.documents.chunking import chunk_document
from kaliok.documents.llama_chunking import (
    chunk_document_with_llamaindex,
)
from kaliok.documents.reader import read_document
from kaliok.paths import TEST_DOCUMENTS


PDF_PATH = TEST_DOCUMENTS / "RIDEAU.pdf"


def print_stats(
    name: str,
    chunks: list[str],
) -> None:
    sizes = [len(chunk) for chunk in chunks]

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Nombre de chunks : {len(chunks)}")

    if sizes:
        print(f"Taille minimale   : {min(sizes)}")
        print(f"Taille maximale   : {max(sizes)}")
        print(f"Taille moyenne    : {mean(sizes):.1f}")


def main() -> None:
    print(f"Document : {PDF_PATH}")

    document = read_document(PDF_PATH)

    print(f"Pages     : {document.page_count}")
    print(f"Blocs     : {len(document.blocks)}")
    print(f"Caractères: {len(document.text)}")

    # ---------------------------------------------------------
    # 1. Chunker kaliok actuel
    # ---------------------------------------------------------

    kaliok_chunks = chunk_document(
        document,
        max_chars=1000,
        overlap_chars=0,
    )

    kaliok_texts = [
        chunk.text
        for chunk in kaliok_chunks
    ]

    # ---------------------------------------------------------
    # 2. LlamaIndex, mais page par page
    # ---------------------------------------------------------

    llama_chunks = chunk_document_with_llamaindex(
        document,
        chunk_size=256,
        chunk_overlap=32,
    )

    llama_texts = [
        chunk.text
        for chunk in llama_chunks
    ]

    # ---------------------------------------------------------
    # Statistiques
    # ---------------------------------------------------------

    print_stats(
        "Chunker kaliok",
        kaliok_texts,
    )

    print_stats(
        "LlamaIndex SentenceSplitter par page",
        llama_texts,
    )

    # ---------------------------------------------------------
    # Aperçu kaliok
    # ---------------------------------------------------------

    print("\n\n=== PREMIERS CHUNKS KALIOK ===")

    for chunk in kaliok_chunks[:5]:
        print(
            f"\n--- kaliok chunk {chunk.index} "
            f"| page {chunk.page} "
            f"| {len(chunk.text)} caractères ---"
        )
        print(chunk.text)

    # ---------------------------------------------------------
    # Aperçu LlamaIndex
    # ---------------------------------------------------------

    print(
        "\n\n=== PREMIERS CHUNKS LLAMAINDEX "
        "PAR PAGE ==="
    )

    for chunk in llama_chunks[:5]:
        print(
            f"\n--- LlamaIndex chunk {chunk.index} "
            f"| page {chunk.page} "
            f"| {len(chunk.text)} caractères ---"
        )
        print(chunk.text)

    # ---------------------------------------------------------
    # Distribution par page
    # ---------------------------------------------------------

    print("\n\n=== DISTRIBUTION LLAMAINDEX PAR PAGE ===")

    pages: dict[int, int] = {}

    for chunk in llama_chunks:
        pages[chunk.page] = pages.get(chunk.page, 0) + 1

    for page, count in sorted(pages.items()):
        print(
            f"Page {page}: {count} chunk(s)"
        )


if __name__ == "__main__":
    main()
