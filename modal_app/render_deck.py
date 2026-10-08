"""
Render presentation/deck.pptx to PNGs in a Modal CPU container (LibreOffice + PyMuPDF),
for visual QA when no local renderer is available.

    modal run modal_app/render_deck.py     # -> presentation/_render/slide-NN.png
    modal run modal_app/render_deck.py --deck presentation/v2/deck_v2.pptx --out presentation/v2/_render
"""
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libreoffice-impress", "fonts-crosextra-carlito", "fonts-crosextra-caladea",
                 "fonts-liberation", "fonts-dejavu")
    .pip_install("pymupdf")
)
app = modal.App("autokernel-render")


@app.function(image=image, cpu=4, memory=8192, timeout=600)
def render(pptx: bytes, dpi: int = 80) -> list:
    import subprocess
    import tempfile

    import pymupdf
    with tempfile.TemporaryDirectory() as td:
        src = pathlib.Path(td) / "deck.pptx"
        src.write_bytes(pptx)
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", td, str(src)],
                       check=True, capture_output=True, timeout=300)
        doc = pymupdf.open(str(pathlib.Path(td) / "deck.pdf"))
        return [p.get_pixmap(dpi=dpi).tobytes("png") for p in doc]


@app.local_entrypoint()
def main(dpi: int = 80, deck: str = "presentation/deck.pptx", out: str = "presentation/_render"):
    out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("slide-*.png"):
        old.unlink()
    pages = render.remote((ROOT / deck).read_bytes(), dpi)
    for i, png in enumerate(pages, 1):
        (out / f"slide-{i:02d}.png").write_bytes(png)
    print(f"rendered {len(pages)} slides to {out}")
