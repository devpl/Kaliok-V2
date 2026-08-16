import argparse
from pathlib import Path
from time import perf_counter

import pypdfium2 as pdfium
from rapidocr import RapidOCR

from kaliok.paths import TEST_DOCUMENTS


def ocr_image(engine, image_path: Path):
    start = perf_counter()
    result = engine(str(image_path))
    elapsed = perf_counter() - start

    for text, score in zip(result.txts, result.scores):
        print(f"{score:.3f} | {text}")

    return elapsed


def process_pdf(engine, pdf_path: Path, scale: float):
    pdf = pdfium.PdfDocument(pdf_path)

    total_time = 0.0

    for page_index in range(len(pdf)):
        page_number = page_index + 1
        page = pdf[page_index]

        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()

        output_dir = TEST_DOCUMENTS / "rendered"
        output_dir.mkdir(parents=True, exist_ok=True)

        image_path = (
            output_dir
            / f"{pdf_path.stem}_page_{page_number}_scale_{scale:g}.png"
        )

        image.save(image_path)

        print()
        print("=" * 70)
        print(f"PAGE {page_number}")
        print("=" * 70)

        elapsed = ocr_image(engine, image_path)
        total_time += elapsed

        print()
        print(f"Temps page {page_number} : {elapsed:.3f} s")

    print()
    print("=" * 70)
    print(f"Nombre de pages : {len(pdf)}")
    print(f"Temps OCR total : {total_time:.3f} s")
    print(f"Moyenne/page    : {total_time / len(pdf):.3f} s")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RapidOCR sur une image ou un PDF"
    )

    parser.add_argument(
        "file",
        help="Image ou PDF à analyser",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Facteur de rendu PDFium pour les PDF",
    )

    args = parser.parse_args()

    file_path = Path(args.file)

    engine = RapidOCR()

    if file_path.suffix.lower() == ".pdf":
        process_pdf(engine, file_path, args.scale)
    else:
        elapsed = ocr_image(engine, file_path)
        print()
        print(f"Temps OCR : {elapsed:.3f} s")


if __name__ == "__main__":
    main()
