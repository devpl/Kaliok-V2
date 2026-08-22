from time import perf_counter

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from kaliok.paths import TEST_DOCUMENTS


PDF_FILE = TEST_DOCUMENTS / "RIDEAU.pdf"


def main():
    print(f"PDF : {PDF_FILE}")

    pipeline_options = PdfPipelineOptions()

    # On désactive volontairement l'OCR
    # et la reconnaissance de structure des tableaux
    # pour isoler le comportement du pipeline PDF Docling.
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )

    start = perf_counter()

    result = converter.convert(PDF_FILE)

    elapsed = perf_counter() - start

    print()
    print(f"Temps : {elapsed:.3f} s")
    print()

    print("=== TEXTE ===")
    print(result.document.export_to_text())


if __name__ == "__main__":
    main()