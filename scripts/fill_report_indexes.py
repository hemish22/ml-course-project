"""Write page numbers into the report's table of contents and lists of figures / tables.

python-docx cannot paginate, so the build leaves the three index fields empty and Word shows
placeholder numbers until fields are updated. This step reads the page on which every heading
and caption landed in a rendered PDF of the same document and stores complete entries as the
fields' cached result, so the Word file opens with correct numbers and no update prompt.

Usage:
    python scripts/fill_report_indexes.py report/ML_Project_Report.docx report/ML_Project_Report.pdf
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

LISTS = {  # style of the listed paragraph -> (index, entry style)
    "Front Heading": ("toc", "toc 1"),
    "Chapter Heading": ("toc", "toc 1"),
    "Section Heading": ("toc", "toc 2"),
    "Figure Caption": ("lof", "table of figures"),
    "Table Caption": ("lot", "table of figures"),
}


def _norm(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().lower()
    return re.sub(r"([-–]) ", r"\1", text)  # undo line breaks after hyphens / dashes


def page_texts(pdf: Path) -> list[tuple[str, str | None]]:
    """(normalized body text, footer page label) of every PDF page."""
    pages = []
    for page in fitz.open(str(pdf)):
        footer_top = page.rect.height - 60
        body, label = [], None
        for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
            if y0 >= footer_top and re.fullmatch(r"[0-9ivxlc]+", text.strip()):
                label = text.strip()
            else:
                body.append(text)
        pages.append((_norm(" ".join(body)), label))
    return pages


def locate(entries: list[tuple[str, str, str]], pages: list[tuple[str, str | None]]) -> list[str]:
    """Page label of each entry; entries are found in document order (monotonic page pointer)."""
    labels, pointer = [], 1
    for _, _, text in entries:
        key = _norm(text)
        for i in range(pointer, len(pages)):
            body, label = pages[i]
            hit = next((m for m in re.finditer(re.escape(key), body)
                        if not body[m.end():m.end() + 3].lstrip().startswith("..")), None)
            if hit is not None:
                labels.append(label or "")
                pointer = i
                break
        else:
            raise ValueError(f"Could not find '{text}' in the PDF")
    return labels


def _entry_runs(paragraph, text: str, page: str) -> None:
    paragraph.add_run(text)
    paragraph.add_run("\t")
    paragraph.add_run(page)


def _fld(run, kind: str) -> bool:
    char = run.find(qn("w:fldChar"))
    return char is not None and char.get(qn("w:fldCharType")) == kind


def _reset_field(paragraph) -> None:
    """Collapse a previously filled index field back into its first paragraph (makes filling repeatable)."""
    if any(_fld(r, "end") for r in paragraph._p.iter(qn("w:r"))):
        return
    sibling = paragraph._p.getnext()
    while sibling is not None:
        nxt = sibling.getnext()
        runs = list(sibling.iter(qn("w:r")))
        end = next((r for r in runs if _fld(r, "end")), None)
        sibling.getparent().remove(sibling)
        if end is not None:
            paragraph._p.append(end)
            return
        sibling = nxt
    raise ValueError("Unterminated index field")


def fill(docx_path: Path, pdf_path: Path) -> None:
    doc = Document(str(docx_path))
    entries = [(LISTS[p.style.name][0], LISTS[p.style.name][1], p.text) for p in doc.paragraphs
               if p.style is not None and p.style.name in LISTS and p.text.strip()]
    pages = locate(entries, page_texts(pdf_path))
    by_index: dict[str, list[tuple[str, str, str]]] = {"toc": [], "lof": [], "lot": []}
    for (index, style, text), page in zip(entries, pages):
        by_index[index].append((style, text, page))

    for paragraph in list(doc.paragraphs):
        instr = "".join(t.text for t in paragraph._p.iter(qn("w:instrText")))
        if "TOC" not in instr:
            continue
        index = "lof" if "Figure Caption" in instr else "lot" if "Table Caption" in instr else "toc"
        items = by_index[index]
        if not items:
            continue
        _reset_field(paragraph)
        runs = list(paragraph._p.iter(qn("w:r")))
        separate = next(i for i, r in enumerate(runs)
                        if r.find(qn("w:fldChar")) is not None
                        and r.find(qn("w:fldChar")).get(qn("w:fldCharType")) == "separate")
        end_run = runs[-1]
        for r in runs[separate + 1:]:
            r.getparent().remove(r)
        paragraph.style = doc.styles[items[0][0]]
        paragraph.paragraph_format.alignment = None
        _entry_runs(paragraph, items[0][1], items[0][2])
        last = paragraph
        for style, text, page in items[1:]:
            new = copy.deepcopy(last._p)
            for child in list(new):
                if child.tag != qn("w:pPr"):
                    new.remove(child)
            last._p.addnext(new)
            last = Paragraph(new, paragraph._parent)
            last.style = doc.styles[style]
            _entry_runs(last, text, page)
        last._p.append(end_run)

    settings = doc.settings.element
    for flag in settings.findall(qn("w:updateFields")):
        settings.remove(flag)
    doc.save(str(docx_path))
    print(f"Filled {len(entries)} index entries in {docx_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("docx", type=Path)
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    fill(args.docx, args.pdf)


if __name__ == "__main__":
    main()
