import argparse
from pathlib import Path
from time import perf_counter

import pypdfium2 as pdfium
from paddleocr import PaddleOCR

TEST_DOCUMENTS = Path("/data")


def ocr_image(engine, image_path: Path):
    start = perf_counter()
    results = engine.predict(str(image_path))
    elapsed = perf_counter() - start

    for result in results:
        texts = result["rec_texts"]
        scores = result["rec_scores"]

        for text, score in zip(texts, scores):
            print(f"{score:.3f} | {text}")

    return elapsed


def process_pdf(engine, pdf_path: Path, scale: float):
    pdf = pdfium.PdfDocument(pdf_path)

    total_time = 0.0

    output_dir = TEST_DOCUMENTS / "rendered"
    output_dir.mkdir(parents=True, exist_ok=True)

    for page_index in range(len(pdf)):
        page_number = page_index + 1
        page = pdf[page_index]

        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()

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
        description="Benchmark PaddleOCR sur une image ou un PDF"
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

    engine = PaddleOCR(
        lang="fr",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    if file_path.suffix.lower() == ".pdf":
        process_pdf(engine, file_path, args.scale)
    else:
        elapsed = ocr_image(engine, file_path)
        print()
        print(f"Temps OCR : {elapsed:.3f} s")


if __name__ == "__main__":
    main()
