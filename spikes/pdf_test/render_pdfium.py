import argparse

import pypdfium2 as pdfium

from kaliok.paths import TEST_DOCUMENTS


def main():
    parser = argparse.ArgumentParser(
        description="Rendu d'une page PDF avec PDFium"
    )

    parser.add_argument(
        "scale",
        type=float,
        help="Facteur de rendu PDFium, par exemple 1, 2 ou 3",
    )

    parser.add_argument(
        "--pdf",
        default="COMPROMIS_VENTE_SCAN.pdf",
        help="Nom du PDF dans test_documents",
    )

    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Numéro de page à rendre, à partir de 1",
    )

    args = parser.parse_args()

    pdf_file = TEST_DOCUMENTS / args.pdf
    output_dir = TEST_DOCUMENTS / "rendered"

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir
        / f"{pdf_file.stem}_page_{args.page}_scale_{args.scale:g}.png"
    )

    pdf = pdfium.PdfDocument(pdf_file)

    page_index = args.page - 1
    page = pdf[page_index]

    bitmap = page.render(scale=args.scale)
    image = bitmap.to_pil()
    image.save(output_file)

    print(f"PDF       : {pdf_file.name}")
    print(f"Page      : {args.page}")
    print(f"Scale     : {args.scale}")
    print(f"Dimensions: {image.width} x {image.height}")
    print(f"Image     : {output_file}")


if __name__ == "__main__":
    main()
