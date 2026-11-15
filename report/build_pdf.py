"""Build a PDF from a Markdown report: Markdown -> styled HTML -> headless Chrome print-to-PDF.
Usage: uv run --with markdown python report/build_pdf.py report/paper.md report/paper.pdf
Images are referenced relative to the Markdown file. A4 page, 11 pt body, figures scaled to the text width.
"""
import subprocess, sys
from pathlib import Path
import markdown

src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
html_body = markdown.markdown(src.read_text(), extensions=["tables", "sane_lists", "attr_list"])
CSS = """
@page { size: A4; margin: 18mm 18mm 18mm 18mm; }
body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.38; color: #111; }
h1 { font-size: 17pt; margin: 0 0 4pt 0; line-height: 1.2; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt 0; }
h3 { font-size: 11pt; margin: 10pt 0 3pt 0; }
h1, h2, h3 { break-after: avoid; page-break-after: avoid; }
p:has(img) + p { break-before: avoid; page-break-before: avoid; }
p { margin: 0 0 6pt 0; text-align: left; }
ul, ol { margin: 0 0 6pt 0; padding-left: 18pt; }
li { margin-bottom: 2pt; }
img { width: 100%; display: block; margin: 6pt auto 2pt auto; page-break-inside: avoid; }
p:has(img) { page-break-inside: avoid; margin-bottom: 0; }
em { color: #333; }
p > em:only-child { display: block; font-size: 9pt; line-height: 1.3; margin-bottom: 8pt; }
table { border-collapse: collapse; font-size: 9pt; margin: 4pt 0 8pt 0; }
th, td { border-bottom: 1px solid #ccc; padding: 2pt 6pt; text-align: left; }
code { font-size: 9pt; }
.pagebreak { page-break-before: always; }
"""
# Google Docs sets Old Standard TT at "1.15" spacing 15.6 pt apart (CSS 1.418) and at single spacing about 13.6 pt (1.233)
LH = "1.233" if "--single" in sys.argv else "1.418"
if "--apart" in sys.argv:  # preview in the look of Apart's Google Doc template: US Letter, 1-inch margins, Old Standard TT 11 pt
    CSS = CSS.replace("@page { size: A4; margin: 18mm 18mm 18mm 18mm; }", "@page { size: Letter; margin: 1in; }")
    CSS = CSS.replace("font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.38;",
                      "font-family: 'Old Standard TT', serif; font-size: 11pt; line-height: %s;" % LH)
    CSS = CSS.replace("h2 { font-size: 12.5pt;", "h2 { font-size: 14pt;").replace("h3 { font-size: 11pt;", "h3 { font-size: 12pt;")
    CSS = "@import url('https://fonts.googleapis.com/css2?family=Old+Standard+TT:ital,wght@0,400;0,700;1,400&display=block');" + CSS
title = next((l[2:].strip() for l in src.read_text().splitlines() if l.startswith("# ")), src.stem)
html = f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{CSS}</style></head><body>{html_body}</body></html>"
tmp = src.with_suffix(".build.html")
tmp.write_text(html)
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}",
                tmp.as_uri()], check=True, capture_output=True)
tmp.unlink()
print("wrote", out)
