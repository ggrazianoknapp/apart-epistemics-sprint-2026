"""Build a Word file on Apart's official report template from the Markdown paper, for upload to Google Docs.

Usage: uv run -q --with python-docx --with markdown --with beautifulsoup4 \
           python report/build_docx.py report/paper.md report/paper.docx

Starts from reference/apart_report_template.docx (Apart's Google Doc template, exported to Word): keeps its title
block, author grid, "With Apart Research" line, abstract box and sprint footnote, removes the guidance text, and adds
the paper's sections in the template's font (Old Standard TT). "<!-- GG: ... -->" notes become yellow-highlighted
[GG: ...] text so they are easy to find and delete in Google Docs.
"""
import re, sys, copy
from pathlib import Path
import markdown
from bs4 import BeautifulSoup, NavigableString, Tag
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.opc.constants import RELATIONSHIP_TYPE as RT

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "reference" / "apart_report_template.docx"
FONT = "Old Standard TT"
TEXT_W = 6.5  # inches of text width: US Letter with the template's 1-inch margins
NOTE = re.compile(r"⟦(.*?)⟧", re.S)
URL = re.compile(r"(https?://[^\s<>()]+[^\s<>().,;:])")

src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
md = src.read_text()

# ---- notes and the parts that go into the title block ----
md = re.sub(r"<!--\s*(GG:.*?)\s*-->", lambda m: "⟦" + " ".join(m.group(1).split()) + "⟧", md, flags=re.S)
md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
title = re.search(r"^# (.+)$", md, re.M).group(1).strip()
head, rest = md.split("\n## Abstract", 1)
abstract_md, body_md = rest.split("\n## 1.", 1)
body_md = "## 1." + body_md
head_notes = NOTE.findall(head)  # e.g. the author-line note

doc = Document(str(TPL))
body = doc.element.body

# ---- fonts and styles: the whole document in the template's serif ----
rpr_default = doc.styles.element.find(qn("w:docDefaults")).find(qn("w:rPrDefault")).find(qn("w:rPr"))
fonts = rpr_default.find(qn("w:rFonts"))
for a in ["w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"]:
    fonts.set(qn(a), FONT)
# single line spacing for body text (the template's 1.15 runs the paper well past the 8-page limit in this font)
doc.styles["Normal"].paragraph_format.line_spacing = 1.0
for name, size, before, after in [("Heading 2", 14, 14, 4), ("Heading 3", 12, 9, 3)]:
    st = doc.styles[name]
    st.font.size, st.font.bold, st.font.color.rgb = Pt(size), True, RGBColor(0, 0, 0)
    st.paragraph_format.space_before, st.paragraph_format.space_after = Pt(before), Pt(after)


# ---- inline text ----
def set_rpr(run_el, bold=False, italic=False, code=False, size=None, highlight=False, sup=False, sub=False, link=False):
    rPr = run_el.find(qn("w:rPr"))
    if rPr is None:
        rPr = OxmlElement("w:rPr"); run_el.insert(0, rPr)
    def add(tag, **attrs):
        e = OxmlElement(tag)
        for k, v in attrs.items():
            e.set(qn(k), v)
        rPr.append(e)
    if code:
        add("w:rFonts", **{"w:ascii": "Courier New", "w:hAnsi": "Courier New", "w:cs": "Courier New"})
    if bold:
        add("w:b")
    if italic:
        add("w:i")
    if link:
        add("w:color", **{"w:val": "1155CC"}); add("w:u", **{"w:val": "single"})
    if size or code:
        add("w:sz", **{"w:val": str(int(2 * (size or 9)))})
    if highlight:
        add("w:highlight", **{"w:val": "yellow"})
    if sup or sub:
        add("w:vertAlign", **{"w:val": "superscript" if sup else "subscript"})


def text_run(text, fmt):
    text = text.replace("\u2212", "\u2013")
    r = OxmlElement("w:r"); set_rpr(r, **fmt)
    t = OxmlElement("w:t"); t.text = text; t.set(qn("xml:space"), "preserve"); r.append(t)
    return r


def add_text(par, text, fmt):
    for i, piece in enumerate(NOTE.split(text)):  # odd pieces are GG notes
        if not piece:
            continue
        if i % 2:
            par._p.append(text_run(f"[{piece}]", {**fmt, "highlight": True, "bold": True}))
            continue
        for j, chunk in enumerate(URL.split(piece)):  # odd chunks are bare URLs
            if chunk:
                (add_link(par, chunk, chunk, fmt) if j % 2 else par._p.append(text_run(chunk, fmt)))


def add_link(par, url, text, fmt):
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), par.part.relate_to(url, RT.HYPERLINK, is_external=True))
    h.append(text_run(text, {**fmt, "link": True})); par._p.append(h)


def inline(par, node, fmt=None):
    fmt = fmt or {}
    for ch in node.children:
        if isinstance(ch, NavigableString):
            add_text(par, re.sub(r"\s+", " ", str(ch)), fmt)
        elif ch.name in ("strong", "b"):
            inline(par, ch, {**fmt, "bold": True})
        elif ch.name in ("em", "i"):
            inline(par, ch, {**fmt, "italic": True})
        elif ch.name == "code":
            par._p.append(text_run(ch.get_text(), {**fmt, "code": True}))
        elif ch.name == "sup":
            inline(par, ch, {**fmt, "sup": True})
        elif ch.name == "sub":
            inline(par, ch, {**fmt, "sub": True})
        elif ch.name == "a":
            add_link(par, ch.get("href", ""), ch.get_text(), fmt)
        elif ch.name == "br":
            par.add_run().add_break()
        else:
            inline(par, ch, fmt)


def trim(par):
    """Drop leading/trailing spaces left by Markdown line breaks."""
    ts = par._p.findall(".//" + qn("w:t"))
    if ts:
        ts[0].text = ts[0].text.lstrip(); ts[-1].text = ts[-1].text.rstrip()


# ---- title block ----
for t in body.iter(qn("w:t")):
    if t.text == "PROJECT TITLE":
        t.text = title
title_tbl = doc.tables[0]
# the author grid is the nested table whose own cells say "Author name 1..6"; one author goes in the middle cell
authors = next(tb for tb in title_tbl._tbl.iter(qn("w:tbl"))
               if any("Author name 1" in "".join(t.text or "" for par in tc.findall(qn("w:p")) for t in par.iter(qn("w:t")))
                      for tr in tb.findall(qn("w:tr")) for tc in tr.findall(qn("w:tc"))))
author_names = re.findall(r"\*\*(.+?)\*\*", head.split("# " + title, 1)[1]) or ["Author"]
_m = re.search(r"\*\*\s*·\s*([^·\n]+?)\s*·", head.split("# " + title, 1)[1])  # "**Name** · Affiliation · with Apart Research"
AFFIL = _m.group(1) if _m else None
rows = authors.findall(qn("w:tr"))
for k, tc in enumerate(rows[0].findall(qn("w:tc"))):
    for t in tc.iter(qn("w:t")):
        if (t.text or "").startswith("Author name"):
            t.text = author_names[0] if k == 1 else ""
        elif t.text == "Affiliation":
            if k == 1 and AFFIL:
                t.text = AFFIL
            elif k == 1:
                set_rpr(t.getparent(), highlight=True)  # placeholder until the author line gives an affiliation
            else:
                t.text = ""
for tr in rows[1:]:
    authors.remove(tr)
# keep only the middle cell, spanning the grid's full width, so Google Docs does not squeeze the name
tcs = rows[0].findall(qn("w:tc"))
full = sum(int(float(g.get(qn("w:w")))) for g in authors.find(qn("w:tblGrid")).findall(qn("w:gridCol")))
for k in (0, 2):
    rows[0].remove(tcs[k])
grid = authors.find(qn("w:tblGrid"))
for g in list(grid):
    grid.remove(g)
g = OxmlElement("w:gridCol"); g.set(qn("w:w"), str(full)); grid.append(g)
tcPr = tcs[1].find(qn("w:tcPr"))
tcW = tcPr.find(qn("w:tcW"))
if tcW is None:
    tcW = OxmlElement("w:tcW"); tcPr.insert(0, tcW)
tcW.set(qn("w:w"), str(full)); tcW.set(qn("w:type"), "dxa")
abs_p = next(p for p in title_tbl._tbl.iter(qn("w:p")) if "Summarize your project" in "".join(x.text or "" for x in p.iter(qn("w:t"))))
from docx.text.paragraph import Paragraph
abs_paras = [b for b in BeautifulSoup(markdown.markdown(abstract_md, extensions=["smarty"]), "html.parser").find_all("p")]
anchor = abs_p
for k, bp in enumerate(abs_paras):
    p = abs_p if k == 0 else copy.deepcopy(abs_p)
    if k:
        anchor.addnext(p); anchor = p
    for r in list(p):
        if r.tag != qn("w:pPr"):
            p.remove(r)
    par = Paragraph(p, title_tbl._parent)
    inline(par, bp); trim(par)
    par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

# ---- remove the guidance box and guidance text ----
children = list(body)
for el in children[3:]:
    if el.tag != qn("w:sectPr"):
        body.remove(el)

# ---- numbering for lists (one restartable list instance per Markdown list) ----
numbering = doc.part.numbering_part.element
def abstract_of(num_id):
    n = next(x for x in numbering.findall(qn("w:num")) if x.get(qn("w:numId")) == str(num_id))
    return int(n.find(qn("w:abstractNumId")).get(qn("w:val")))
ABS_DECIMAL, ABS_BULLET = abstract_of(5), abstract_of(3)
def new_list(ordered):
    num = numbering.add_num(ABS_DECIMAL if ordered else ABS_BULLET)
    num.add_lvlOverride(ilvl=0).add_startOverride(1)
    return num.numId


def para(style=None):
    p = doc.add_paragraph(style=style)
    if style is None:
        p.paragraph_format.space_after = Pt(6)
    return p


def justify(p):
    """Justify body text; lines that carry a URL stay left-aligned so the spacing does not stretch."""
    if "http" not in p.text:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


# ---- body ----
soup = BeautifulSoup(markdown.markdown(body_md, extensions=["tables", "sane_lists", "attr_list", "smarty"]), "html.parser")
last_was_img = False
section = ""
for el in soup.children:
    if not isinstance(el, Tag):
        continue
    if el.name in ("h1", "h2", "h3", "h4"):
        p = para("Heading 2" if el.name in ("h1", "h2") else "Heading 3"); inline(p, el); trim(p)
        section = el.get_text().strip()
        if el.get_text().startswith("Appendix A"):  # appendices start on a new page, as in the PDF build
            p.paragraph_format.page_break_before = True
        last_was_img = False
    elif el.name == "p" and el.find("img"):
        img = el.find("img")
        w = re.search(r"width:\s*(\d+)%", img.get("style", ""))
        pct = int(w.group(1)) / 100 if w else 1.0
        p = para(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True; p.paragraph_format.space_after = Pt(2)
        p.add_run().add_picture(str((src.parent / img["src"]).resolve()), width=Inches(TEXT_W * pct))
        last_was_img = True
    elif el.name == "p" and len(el.contents) == 1 and getattr(el.contents[0], "name", None) == "em":
        p = para(); inline(p, el.contents[0], {"size": 9.5}); trim(p); justify(p)  # caption or small note
        if p.text.startswith("Table "):
            p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_after = Pt(10 if last_was_img else 6)
        last_was_img = False
    elif el.name == "p":
        p = para(); inline(p, el); trim(p); justify(p); last_was_img = False
    elif el.name == "ul" and section == "References":
        for li in el.find_all("li", recursive=False):
            p = para(); inline(p, li, {"size": 9.5}); trim(p)
            p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(0.3), Inches(-0.3)
            p.paragraph_format.space_after = Pt(3)
        last_was_img = False
    elif el.name in ("ul", "ol"):
        nid = new_list(el.name == "ol")
        items = el.find_all("li", recursive=False)
        for k, li in enumerate(items):
            p = para()
            numPr = p._p.get_or_add_pPr().get_or_add_numPr()
            numPr.get_or_add_ilvl().val = 0; numPr.get_or_add_numId().val = nid
            p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(0.4), Inches(-0.25)
            p.paragraph_format.space_after = Pt(2 if k < len(items) - 1 else 6)
            inline(p, li.p if li.p else li); trim(p); justify(p)
        last_was_img = False
    elif el.name == "table":
        rows = el.find_all("tr")
        ncol = max(len(r.find_all(["th", "td"])) for r in rows)
        t = doc.add_table(rows=len(rows), cols=ncol)
        tblPr = t._tbl.tblPr
        t.alignment = WD_TABLE_ALIGNMENT.CENTER  # centred, spanning the text width
        tblW = tblPr.find(qn("w:tblW"))
        if tblW is None:
            tblW = OxmlElement("w:tblW"); tblPr.append(tblW)
        tblW.set(qn("w:w"), "5000"); tblW.set(qn("w:type"), "pct")
        borders = OxmlElement("w:tblBorders")
        for side in ["top", "bottom", "insideH"]:
            b = OxmlElement(f"w:{side}")
            for k, v in {"w:val": "single", "w:sz": "4", "w:space": "0", "w:color": "BBBBBB"}.items():
                b.set(qn(k), v)
            borders.append(b)
        tblPr.append(borders)
        t.autofit = False
        wts = [1.0] * ncol
        for r in rows:
            for j, c in enumerate(r.find_all(["th", "td"])):
                n = len(c.get_text().strip())
                wts[j] = max(wts[j], min(n, 46) * (0.55 if c.name == "th" else 1.0) + (7 if "[" in c.get_text() else 3))
                wts[j] = max(wts[j], max((len(w) for w in c.get_text().split()), default=0) + 3.5)  # longest word fits
        # no word may break inside a cell: each column is at least as wide as its longest word (8.5 pt type)
        longest = [0] * ncol
        for r in rows:
            for j, c in enumerate(r.find_all(["th", "td"])):
                longest[j] = max(longest[j], max((len(w) for w in c.get_text().split()), default=0))
        width = [TEXT_W * w / sum(wts) for w in wts]
        floor = [0.068 * n + 0.17 for n in longest]
        for _ in range(4):
            short = [j for j in range(ncol) if width[j] < floor[j]]
            if not short:
                break
            need = sum(floor[j] - width[j] for j in short)
            free = [j for j in range(ncol) if width[j] > floor[j] + 0.01]
            room = sum(width[j] - floor[j] for j in free)
            for j in short:
                width[j] = floor[j]
            for j in free:
                width[j] -= need * (width[j] - floor[j]) / room if room > need else (width[j] - floor[j])
        wts = width
        twips = [int(TEXT_W * 1440 * w / sum(wts)) for w in wts]
        hdr = OxmlElement("w:tblHeader"); hdr.set(qn("w:val"), "true")  # the header row repeats when a table breaks
        t.rows[0]._tr.get_or_add_trPr().append(hdr)
        for j in range(ncol):
            for i in range(len(rows)):
                t.cell(i, j).width = Inches(TEXT_W * wts[j] / sum(wts))
        grid = t._tbl.tblGrid  # Google Docs reads column widths from the grid, with a fixed layout
        for g, w in zip(grid.findall(qn("w:gridCol")), twips):
            g.set(qn("w:w"), str(w))
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
        tblW.set(qn("w:w"), str(sum(twips))); tblW.set(qn("w:type"), "dxa")
        for i, r in enumerate(rows):
            for j, c in enumerate(r.find_all(["th", "td"])):
                cp = t.cell(i, j).paragraphs[0]
                inline(cp, c, {"size": 8.5, "bold": c.name == "th"}); trim(cp)
                cp.paragraph_format.space_after = Pt(0)
                if len(rows) <= 20 and i < len(rows) - 1:
                    cp.paragraph_format.keep_with_next = True  # short tables stay on one page, under their caption
                if "right" in (c.get("style") or "") or c.get("align") == "right":
                    cp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        para().paragraph_format.space_after = Pt(2)
        last_was_img = False
    elif el.name == "div" and "pagebreak" in (el.get("class") or []):
        pass  # page breaks are set on the appendix heading instead
    elif el.name == "hr":
        pass
    else:
        p = para(); inline(p, el); trim(p); last_was_img = False

# page numbers, centred in the footer
foot = doc.sections[0].footer
fp = foot.paragraphs[0] if foot.paragraphs else foot.add_paragraph()
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
for kind, text in [("begin", None), (None, "PAGE"), ("end", None)]:
    r = OxmlElement("w:r")
    if kind:
        fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), kind); r.append(fc)
    else:
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = text; r.append(it)
    fp._p.append(r)
doc.core_properties.title = title
doc.save(str(out))
print("wrote", out, "| GG notes in the author line:", head_notes)
