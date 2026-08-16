from kaliok.documents.models import DocumentContent, TextBlock


def test_document_content_text():
    document = DocumentContent(
        source="test.pdf",
        page_count=2,
        blocks=[
            TextBlock(
                text="Texte page 1",
                page=1,
            ),
            TextBlock(
                text="Texte page 2",
                page=2,
                confidence=0.98,
            ),
        ],
    )

    assert document.source == "test.pdf"
    assert document.page_count == 2
    assert len(document.blocks) == 2

    assert document.blocks[0].page == 1
    assert document.blocks[0].confidence is None

    assert document.blocks[1].page == 2
    assert document.blocks[1].confidence == 0.98

    assert document.text == "Texte page 1\nTexte page 2"
