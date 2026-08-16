from pathlib import Path
from time import perf_counter

import pypdfium2 as pdfium

from kaliok.paths import TEST_DOCUMENTS

PDF_FILE = TEST_DOCUMENTS / "RIDEAU.pdf"


start = perf_counter()

pdf = pdfium.PdfDocument(PDF_FILE)

print(f"Nombre de pages : {len(pdf)}")
print()

for index in range(len(pdf)):
    page = pdf[index]
    textpage = page.get_textpage()
    text = textpage.get_text_range()

    print(f"--- PAGE {index + 1} ---")
    print(text[:2000])
    print()

elapsed = perf_counter() - start

print(f"Temps total : {elapsed:.3f} s")
