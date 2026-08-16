import pytest

from kaliok.documents.rapidocr_reader import read_pdf_with_rapidocr
from kaliok.paths import TEST_DOCUMENTS


@pytest.mark.slow
def test_read_pdf_with_rapidocr():
    pdf_file = (
        TEST_DOCUMENTS
        / "lilas"
        / "doc040826-04082026160521.pdf"
    )

    document = read_pdf_with_rapidocr(pdf_file)

    assert document.source == "doc040826-04082026160521.pdf"
    assert document.page_count == 4
    assert len(document.blocks) > 0
    assert len(document.text) > 0

    assert document.blocks[0].page == 1
    assert document.blocks[0].confidence is not None
