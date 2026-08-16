from pathlib import Path
from time import perf_counter

import pypdfium2 as pdfium

from kaliok.ocr.doctr_engine import DocTrEngine
from kaliok.paths import TEST_DOCUMENTS


PDF_FILES = [
    TEST_DOCUMENTS / "lilas" / "doc040826-04082026160521.pdf",
    TEST_DOCUMENTS / "lilas" / "doc310726-31072026140505.pdf",
    TEST_DOCUMENTS / "lilas" / "doc240726-24072026104404.pdf",
]

SCALE = 1.0


def process_pdf(pdf_file: Path, engine: DocTrEngine) -> tuple[int, float]:
    pdf = pdfium.PdfDocument(pdf_file)

    try:
        page_count = len(pdf)
        pdf_start = perf_counter()

        print()
        print("=" * 80)
        print(f"PDF : {pdf_file.name}")
        print(f"Pages : {page_count}")
        print("=" * 80)

        for page_index in range(page_count):
            page_number = page_index + 1
            page = pdf[page_index]

            bitmap = page.render(scale=SCALE)
            image = bitmap.to_pil()

            start = perf_counter()
            results = engine.recognize(image)
            elapsed = perf_counter() - start

            print(
                f"Page {page_number:>2}/{page_count} : "
                f"{elapsed:.3f} s - "
                f"{len(results)} mots"
            )

        total_time = perf_counter() - pdf_start

        print()
        print(f"Temps PDF : {total_time:.3f} s")
        print(f"Moyenne   : {total_time / page_count:.3f} s/page")

        return page_count, total_time

    finally:
        pdf.close()


def main():
    print("Benchmark docTR")
    print(f"Scale : {SCALE}")

    print()
    print("Chargement du moteur...")

    load_start = perf_counter()
    engine = DocTrEngine()
    load_time = perf_counter() - load_start

    print(f"Moteur chargé en {load_time:.3f} s")

    total_pages = 0
    total_ocr_time = 0.0

    benchmark_start = perf_counter()

    for pdf_file in PDF_FILES:
        page_count, pdf_time = process_pdf(pdf_file, engine)

        total_pages += page_count
        total_ocr_time += pdf_time

    benchmark_time = perf_counter() - benchmark_start

    print()
    print("=" * 80)
    print("RÉSULTAT GLOBAL")
    print("=" * 80)
    print(f"Documents           : {len(PDF_FILES)}")
    print(f"Pages               : {total_pages}")
    print(f"Chargement moteur   : {load_time:.3f} s")
    print(f"Temps OCR total     : {total_ocr_time:.3f} s")
    print(f"Moyenne OCR         : {total_ocr_time / total_pages:.3f} s/page")
    print(f"Temps benchmark     : {benchmark_time:.3f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()
