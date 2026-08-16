from kaliok.documents.chunking import chunk_document
from kaliok.documents.models import DocumentContent, TextBlock


def test_chunk_document():
    document = DocumentContent(
        source="test.pdf",
        page_count=2,
        blocks=[
            TextBlock(
                text="ABCDEFGHIJ",
                page=1,
            ),
            TextBlock(
                text="KLMNOP",
                page=2,
            ),
        ],
    )

    chunks = chunk_document(
        document,
        max_chars=4,
    )

    assert len(chunks) == 5

    assert chunks[0].text == "ABCD"
    assert chunks[0].page == 1
    assert chunks[0].index == 0

    assert chunks[1].text == "EFGH"
    assert chunks[1].page == 1
    assert chunks[1].index == 1

    assert chunks[2].text == "IJ"
    assert chunks[2].page == 1

    assert chunks[3].text == "KLMN"
    assert chunks[3].page == 2

    assert chunks[4].text == "OP"
    assert chunks[4].page == 2
    assert chunks[4].index == 4
