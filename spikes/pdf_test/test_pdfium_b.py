from pathlib import Path
from time import perf_counter

import pypdfium2 as pdfium

import hashlib


from kaliok.paths import TEST_DOCUMENTS

PDF_FILE = TEST_DOCUMENTS / "COMPROMIS_VENTE_SCAN.pdf"

start = perf_counter()

pdf = pdfium.PdfDocument(PDF_FILE)

print(f"Nombre de pages : {len(pdf)}")
print()



for index in range(len(pdf)):
    page = pdf[index]
    textpage = page.get_textpage()
    text = textpage.get_text_range()

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    print(
        f"Page {index + 1}: "
        f"{len(text)} caractères | "
        f"SHA256: {digest}"
    )

elapsed = perf_counter() - start

print(f"Temps total : {elapsed:.3f} s")
