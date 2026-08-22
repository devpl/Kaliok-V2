from pathlib import Path

from kaliok.documents.cleaning import clean_document
from kaliok.documents.reader import read_document
from kaliok.paths import TEST_DOCUMENTS


VALIDATION_DIR = TEST_DOCUMENTS / "validation"

PREVIEW_CHARS = 500


def main() -> None:
    pdf_files = sorted(
        path
        for path in VALIDATION_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        print(
            f"Aucun PDF trouvé dans : {VALIDATION_DIR}"
        )
        return

    print(
        f"{len(pdf_files)} document(s) trouvé(s) dans :"
    )
    print(
        VALIDATION_DIR
    )

    for document_number, pdf_path in enumerate(
        pdf_files,
        start=1,
    ):
        print()
        print("=" * 80)
        print(
            f"DOCUMENT {document_number}/{len(pdf_files)}"
        )
        print("=" * 80)
        print(
            f"Fichier : {pdf_path.name}"
        )

        try:
            document = read_document(
                pdf_path
            )
        except Exception as exc:
            print(
                f"ERREUR EXTRACTION : {exc}"
            )
            continue

        cleaned_document = clean_document(
            document
        )

        print(
            f"Pages              : {document.page_count}"
        )
        print(
            f"Blocs              : {len(document.blocks)}"
        )
        print(
            f"Caractères bruts   : {len(document.text)}"
        )
        print(
            f"Caractères nettoyés: {len(cleaned_document.text)}"
        )

        print()
        print(
            "APERÇU PAR PAGE"
        )
        print(
            "-" * 80
        )

        for block in cleaned_document.blocks:
            preview = (
                block.text
                .replace("\r", " ")
                .replace("\n", " ")
                .strip()
            )

            if len(preview) > PREVIEW_CHARS:
                preview = (
                    preview[:PREVIEW_CHARS]
                    + "..."
                )

            print()
            print(
                f"[Page {block.page}]"
            )

            if block.confidence is not None:
                print(
                    "Confiance OCR : "
                    f"{block.confidence:.4f}"
                )

            if preview:
                print(
                    preview
                )
            else:
                print(
                    "<aucun texte>"
                )


if __name__ == "__main__":
    main()
