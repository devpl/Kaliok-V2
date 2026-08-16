import pytest

from kaliok.documents.reader import read_document
from kaliok.paths import TEST_DOCUMENTS


def test_read_native_pdf():
    pdf_file = TEST_DOCUMENTS / "RIDEAU.pdf"

    document = read_document(pdf_file)

    assert document.source == "RIDEAU.pdf"
    assert document.page_count == 11
    assert len(document.text) > 20

    assert all(
        block.confidence is None
        for block in document.blocks
    )


@pytest.mark.slow
def test_read_scanned_pdf():
    pdf_file = (
        TEST_DOCUMENTS
        / "lilas"
        / "doc040826-04082026160521.pdf"
    )

    document = read_document(pdf_file)

    assert document.source == "doc040826-04082026160521.pdf"
    assert document.page_count == 4
    assert len(document.text) > 20

    assert any(
        block.confidence is not None
        for block in document.blocks
    )


@pytest.mark.slow
def test_read_mixed_pdf():
    pdf_file = TEST_DOCUMENTS / "mixed_test.pdf"

    document = read_document(pdf_file)

    assert document.page_count == 2

    page_1_blocks = [
        block
        for block in document.blocks
        if block.page == 1
    ]

    page_2_blocks = [
        block
        for block in document.blocks
        if block.page == 2
    ]

    assert page_1_blocks
    assert page_2_blocks

    assert all(
        block.confidence is None
        for block in page_1_blocks
    )

    assert any(
        block.confidence is not None
        for block in page_2_blocks
    )
