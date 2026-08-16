from kaliok.documents.chunking import chunk_document
from kaliok.documents.models import DocumentContent, TextBlock
from kaliok.documents.chunking import split_long_text

def test_chunk_document_by_sentences():
    document = DocumentContent(
        source="test.pdf",
        page_count=1,
        blocks=[
            TextBlock(
                text=(
                    "Première phrase assez longue. "
                    "Deuxième phrase un peu plus longue. "
                    "Troisième phrase."
                ),
                page=1,
            ),
        ],
    )

    chunks = chunk_document(
        document,
        max_chars=55,
    )

    assert len(chunks) == 2

    assert chunks[0].text == "Première phrase assez longue."
    assert chunks[0].page == 1
    assert chunks[0].index == 0

    assert chunks[1].text == (
        "Deuxième phrase un peu plus longue. "
        "Troisième phrase."
    )
    assert chunks[1].page == 1
    assert chunks[1].index == 1





def test_split_long_text_without_cutting_words():
    text = "Alpha bravo charlie delta echo foxtrot"

    parts = split_long_text(
        text,
        max_chars=12,
    )

    assert parts == [
        "Alpha bravo",
        "charlie",
        "delta echo",
        "foxtrot",
    ]

    assert all(len(part) <= 12 for part in parts)

def test_chunk_document_with_overlap():
        document = DocumentContent(
            source="test.pdf",
            page_count=1,
            blocks=[
                TextBlock(
                    text=(
                        "Première phrase assez longue. "
                        "Deuxième phrase un peu plus longue. "
                        "Troisième phrase."
                    ),
                    page=1,
                ),
            ],
        )

        chunks = chunk_document(
            document,
            max_chars=55,
            overlap_chars=10,
        )

        assert len(chunks) == 2

        assert chunks[0].text == "Première phrase assez longue."

        assert chunks[1].text.endswith(
            "Deuxième phrase un peu plus longue. Troisième phrase."
        )

        assert "z longue." in chunks[1].text
