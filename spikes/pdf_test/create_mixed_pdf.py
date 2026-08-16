from pypdf import PdfReader, PdfWriter

from kaliok.paths import TEST_DOCUMENTS


def main():
    native_pdf = TEST_DOCUMENTS / "RIDEAU.pdf"
    scanned_pdf = TEST_DOCUMENTS / "lilas" / "doc040826-04082026160521.pdf"

    output_pdf = TEST_DOCUMENTS / "mixed_test.pdf"

    native_reader = PdfReader(native_pdf)
    scanned_reader = PdfReader(scanned_pdf)

    writer = PdfWriter()

    # Page 1 : PDF natif
    writer.add_page(native_reader.pages[0])

    # Page 2 : PDF scanné
    writer.add_page(scanned_reader.pages[0])

    with output_pdf.open("wb") as f:
        writer.write(f)

    print(f"PDF mixte créé : {output_pdf}")


if __name__ == "__main__":
    main()
