"""Convert the Task 4 report markdown into a .docx (no Word heading styles)."""
import re, sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

src, dst, _, lang = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
if lang == "th":
    FONT, TITLE, H2, H3, BODY, TABLE = "TH Sarabun New", 24, 18, 16, 16, 14
else:
    FONT, TITLE, H2, H3, BODY, TABLE = "Calibri", 18, 14, 12, 11, 10
MONO = "Consolas"
MONO_SIZE = 11 if lang == "th" else 9.5

def set_font(run_or_style_font, name, size, bold=None, element=None):
    f = run_or_style_font
    f.name = name
    f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    rpr = element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts"); rpr.append(fonts)
    for key in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(key), name)
    szcs = rpr.find(qn("w:szCs"))
    if szcs is None:
        szcs = OxmlElement("w:szCs"); rpr.append(szcs)
    szcs.set(qn("w:val"), str(int(size * 2)))
    if bold:
        if rpr.find(qn("w:bCs")) is None:
            rpr.append(OxmlElement("w:bCs"))

doc = Document()
for s in doc.sections:
    s.top_margin = s.bottom_margin = Cm(2.54); s.left_margin = s.right_margin = Cm(2.54)
for name in ("Normal", "List Bullet", "List Number"):
    st = doc.styles[name]
    set_font(st.font, FONT, BODY, element=st.element)
    st.paragraph_format.space_after = Pt(2)
doc.styles["Normal"].paragraph_format.space_after = Pt(4)

TOKEN = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")
def clean(text):
    return text.replace("\u2212", "-").replace("\u2014", ", ")

def add_runs(par, text, size=BODY, bold=False):
    for part in TOKEN.split(clean(text)):
        if not part:
            continue
        if part.startswith("**"):
            add_runs(par, part[2:-2], size, True)
        elif part.startswith("`"):
            r = par.add_run(part[1:-1]); set_font(r.font, MONO, MONO_SIZE if size >= BODY else MONO_SIZE - 1, bold, r._element)
        elif part.startswith("["):
            add_runs(par, re.match(r"\[([^\]]+)\]", part).group(1), size, bold)
        else:
            r = par.add_run(part); set_font(r.font, FONT, size, bold, r._element)

def heading(text, size, center=False, before=10):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before); p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_runs(p, text, size, True)

def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    align = [":" in c and c.strip().endswith(":") for c in cells[1]]
    body = [cells[0]] + cells[2:]
    t = doc.add_table(rows=len(body), cols=len(body[0]))
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(body):
        for j, value in enumerate(row):
            p = t.cell(i, j).paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            if align[j] and i > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            add_runs(p, value, TABLE, bold=(i == 0))
    doc.add_paragraph().paragraph_format.space_after = Pt(0)

lines = src.read_text(encoding="utf-8").splitlines()
i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith("|"):
        block = []
        while i < len(lines) and lines[i].startswith("|"):
            block.append(lines[i]); i += 1
        table(block); continue
    if line.startswith("# "):
        heading(line[2:], TITLE, center=True, before=0)
    elif line.startswith("## "):
        heading(line[3:], H2, before=14)
    elif line.startswith("### "):
        heading(line[4:], H3, before=10)
    elif line.startswith("- "):
        add_runs(doc.add_paragraph(style="List Bullet"), line[2:])
    elif re.match(r"\d+\. ", line):
        add_runs(doc.add_paragraph(style="List Number"), line.split(". ", 1)[1])
    elif line.strip() and not ("task4-report." in line and ".md" in line):
        add_runs(doc.add_paragraph(), line)
    i += 1
doc.save(dst)
print("wrote", dst)
