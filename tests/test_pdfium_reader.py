from kaliok.documents.pdfium_reader import read_pdf_with_pdfium
from kaliok.paths import TEST_DOCUMENTS


def test_read_pdf_with_pdfium():
    pdf_file = TEST_DOCUMENTS / "RIDEAU.pdf"

    document = read_pdf_with_pdfium(pdf_file)

    assert document.source == "RIDEAU.pdf"
    assert document.page_count == 11
    assert len(document.blocks) > 0
    assert len(document.text) > 0

    assert document.blocks[0].page == 1
    assert document.blocks[0].confidence is None
