import fitz  # pymupdf
from pathlib import Path

pdf_path = Path("paper.pdf")
out_dir = Path("page_previews")
out_dir.mkdir(exist_ok=True)

doc = fitz.open(str(pdf_path))
print(f"Total pages: {len(doc)}")

for i, page in enumerate(doc):
    mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
    pix = page.get_pixmap(matrix=mat)
    out_path = out_dir / f"page_{i+1:02d}.png"
    pix.save(str(out_path))
    print(f"Saved {out_path}")

doc.close()
print("Done.")
