from kaliok.documents.cleaning import clean_document, find_repeated_lines
from kaliok.documents.reader import read_document
from kaliok.paths import TEST_DOCUMENTS


PDF_PATH = TEST_DOCUMENTS / "RIDEAU.pdf"


def main() -> None:
    print(f"Document : {PDF_PATH}")

    document = read_document(PDF_PATH)

    repeated_lines = find_repeated_lines(document)

    print("\nLignes répétées détectées :")
    print("-" * 40)

    for line in sorted(repeated_lines):
        print(f"- {line}")

    cleaned_document = clean_document(document)

    print("\n\n========================================")
    print("COMPARAISON AVANT / APRÈS")
    print("========================================")

    for original_block, cleaned_block in zip(
        document.blocks,
        cleaned_document.blocks,
    ):
        print(
            f"\n\n========== PAGE {original_block.page} =========="
        )

        print("\n--- AVANT ---")
        print(original_block.text)

        print("\n--- APRÈS ---")
        print(cleaned_block.text)


if __name__ == "__main__":
    main()
