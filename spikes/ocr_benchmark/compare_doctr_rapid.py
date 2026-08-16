from pathlib import Path
from time import perf_counter

import pypdfium2 as pdfium

from kaliok.ocr.doctr_engine import DocTrEngine
from kaliok.ocr.rapidocr_engine import RapidOcrEngine
from kaliok.paths import TEST_DOCUMENTS


PDF_FILE = TEST_DOCUMENTS / "lilas" / "doc040826-04082026160521.pdf"
PAGE_NUMBER = 1
SCALE = 1.0

OUTPUT_DIR = TEST_DOCUMENTS / "ocr_compare"


def run_engine(engine, image):
    start = perf_counter()
    results = engine.recognize(image)
    elapsed = perf_counter() - start

    text = " ".join(result.text for result in results)

    return text, elapsed


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf = pdfium.PdfDocument(PDF_FILE)

    try:
        page = pdf[PAGE_NUMBER - 1]
        bitmap = page.render(scale=SCALE)
        image = bitmap.to_pil()

        print(f"PDF   : {PDF_FILE.name}")
        print(f"Page  : {PAGE_NUMBER}")
        print(f"Scale : {SCALE}")
        print()

        print("Chargement docTR...")
        doctr_engine = DocTrEngine()

        print("Chargement RapidOCR...")
        rapid_engine = RapidOcrEngine()

        doctr_text, doctr_time = run_engine(doctr_engine, image)
        rapid_text, rapid_time = run_engine(rapid_engine, image)

        base_name = f"{PDF_FILE.stem}_page_{PAGE_NUMBER}"

        doctr_file = OUTPUT_DIR / f"{base_name}_doctr.txt"
        rapid_file = OUTPUT_DIR / f"{base_name}_rapidocr.txt"

        doctr_file.write_text(doctr_text, encoding="utf-8")
        rapid_file.write_text(rapid_text, encoding="utf-8")

        print()
        print(f"docTR     : {doctr_time:.3f} s")
        print(f"RapidOCR  : {rapid_time:.3f} s")

        print()
        print(f"docTR     -> {doctr_file}")
        print(f"RapidOCR  -> {rapid_file}")

    finally:
        pdf.close()


if __name__ == "__main__":
    main()
