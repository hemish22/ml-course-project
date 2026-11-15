"""Small python-docx helpers that apply the project-report formatting guidelines.

Guidelines: A4; Times New Roman; chapter headings 16 pt bold uppercase centred;
section 14 pt bold; subsection 12 pt bold; body 12 pt justified, 1.5 line spacing,
6 pt after; margins left 1.5", others 1"; captions 11 pt bold (figure below, table
above); table text 11 pt; code Consolas 10 pt single spaced; chapter-wise numbering.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable, Sequence

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

FONT = "Times New Roman"


def _set_font(style_or_run, name: str = FONT, size: float | None = None, bold: bool | None = None) -> None:
    font = style_or_run.font
    font.name = name
    if size is not None:
        font.size = Pt(size)
    if bold is not None:
        font.bold = bold
    font.color.rgb = RGBColor(0, 0, 0)
    element = style_or_run.element if hasattr(style_or_run, "element") else style_or_run._element
    rpr = element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), name)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if fonts.get(qn(attr)) is not None:
            del fonts.attrib[qn(attr)]


def _field(paragraph, instruction: str, placeholder: str = "") -> None:
    """Insert a complex field (TOC, PAGE, ...) into ``paragraph``."""
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    begin.set(qn("w:dirty"), "true")
    run._r.append(begin)
    run = paragraph.add_run()
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" {instruction} "
    run._r.append(instr)
    run = paragraph.add_run()
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    run._r.append(sep)
    paragraph.add_run(placeholder)
    run = paragraph.add_run()
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(end)


class Report:
    """Append guideline-formatted content to a copy of the template's cover page."""

    def __init__(self, template: Path, keep_elements: int) -> None:
        self.doc = Document(str(template))
        body = self.doc.element.body
        children = list(body.iterchildren())
        for element in children[keep_elements:]:
            if element.tag != qn("w:sectPr"):
                body.remove(element)
        self.chapter = 0
        self.figure = 0
        self.table = 0
        self.eq_no = 0
        self._fresh_section = True
        self._styles()

    # ------------------------------------------------------------------ styles
    def _styles(self) -> None:
        styles = self.doc.styles
        names = [s.name for s in styles]

        def get(name: str):
            return styles[name] if name in names else styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)

        # Dedicated styles: the template's Normal/Heading styles stay untouched so the cover page keeps its layout.
        normal = get("Report Body")
        normal.base_style = None
        _set_font(normal, size=12, bold=False)
        pf = normal.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.space_after = Pt(6)
        pf.space_before = Pt(0)
        pf.left_indent = pf.first_line_indent = Inches(0)
        for name, size, align, level in (("Chapter Heading", 16, WD_ALIGN_PARAGRAPH.CENTER, 0),
                                         ("Front Heading", 16, WD_ALIGN_PARAGRAPH.CENTER, 0),
                                         ("Section Heading", 14, WD_ALIGN_PARAGRAPH.LEFT, 1),
                                         ("Subsection Heading", 12, WD_ALIGN_PARAGRAPH.LEFT, 2)):
            style = get(name)
            style.base_style = normal
            _set_font(style, size=size, bold=True)
            style.font.all_caps = level == 0
            style.paragraph_format.alignment = align
            style.paragraph_format.space_before = Pt(12 if level else 6)
            style.paragraph_format.space_after = Pt(6 if level else 18)
            style.paragraph_format.keep_with_next = True
            ppr = style.element.get_or_add_pPr()
            outline = OxmlElement("w:outlineLvl")
            outline.set(qn("w:val"), str(level))
            ppr.append(outline)
        for name in ("Figure Caption", "Table Caption"):
            style = get(name)
            style.base_style = normal
            _set_font(style, size=11, bold=True)
            style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            style.paragraph_format.space_before = Pt(4)
            style.paragraph_format.space_after = Pt(10)
            style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            style.paragraph_format.keep_with_next = name == "Table Caption"
        if True:
            code = get("Code Block")
            code.base_style = normal
            _set_font(code, name="Consolas", size=10)
            code.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
            code.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            code.paragraph_format.space_after = Pt(0)
        if True:
            tt = get("Table Text")
            tt.base_style = normal
            _set_font(tt, size=11)
            tt.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
            tt.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            tt.paragraph_format.space_after = Pt(0)
        for name in ("List Bullet", "List Number"):
            if name in names:
                _set_font(styles[name], size=12)
                styles[name].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                styles[name].paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
                styles[name].paragraph_format.space_after = Pt(3)

    def _p(self, style: str = "Report Body"):
        return self.doc.add_paragraph(style=style)

    # ---------------------------------------------------------------- sections
    def new_section(self, page_format: str, start: int = 1, footer_page_number: bool = True) -> None:
        """Start a new page section with its own page-number format (lowerRoman / decimal).

        The first call reuses the template's trailing section, which already starts after the cover.
        """
        if self._fresh_section:
            self._fresh_section = False
            section = self.doc.sections[-1]
            section.start_type = WD_SECTION.NEW_PAGE
        else:
            section = self.doc.add_section(WD_SECTION.NEW_PAGE)
        section.page_width, section.page_height = Inches(8.27), Inches(11.69)
        section.left_margin = Inches(1.5)
        section.right_margin = section.top_margin = section.bottom_margin = Inches(1)
        section.footer_distance = Inches(0.5)
        sectpr = section._sectPr
        for old in sectpr.findall(qn("w:pgNumType")):
            sectpr.remove(old)
        pg = OxmlElement("w:pgNumType")
        pg.set(qn("w:fmt"), page_format)
        pg.set(qn("w:start"), str(start))
        sectpr.append(pg)
        section.footer.is_linked_to_previous = False
        footer = section.footer
        for p in footer.paragraphs[1:]:
            p._element.getparent().remove(p._element)
        p = footer.paragraphs[0]
        for r in list(p.runs):
            r._r.getparent().remove(r._r)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if footer_page_number:
            _field(p, "PAGE", "1")
            for r in p.runs:
                _set_font(r, size=11)

    def page_break(self) -> None:
        self._p().add_run().add_break(WD_BREAK.PAGE)

    # ------------------------------------------------------------------ blocks
    def title(self, text: str, size: float = 16, space_after: float = 12, toc: bool = False) -> None:
        """Unnumbered centred heading (ABSTRACT, REFERENCES, ...); optionally listed in the TOC."""
        p = self._p("Front Heading" if toc else "Report Body")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(space_after)
        run = p.add_run(text.upper())
        _set_font(run, size=size, bold=True)

    def chapter_heading(self, title: str, first: bool = False) -> None:
        self.chapter += 1
        self.figure = self.table = self.eq_no = 0
        if not first:
            self.page_break()
        label = self._p()
        label.alignment = WD_ALIGN_PARAGRAPH.CENTER
        label.paragraph_format.space_after = Pt(6)
        label.paragraph_format.keep_with_next = True
        _set_font(label.add_run(f"CHAPTER {self.chapter}"), size=16, bold=True)
        self._p("Chapter Heading").add_run(f"{self.chapter}. {title}")

    def h2(self, text: str) -> None:
        self._p("Section Heading").add_run(text)

    def h3(self, text: str) -> None:
        self._p("Subsection Heading").add_run(text)

    def para(self, text: str, *, bold_prefix: str | None = None, italic: bool = False, align=None) -> None:
        p = self._p()
        if align is not None:
            p.alignment = align
        if bold_prefix:
            run = p.add_run(bold_prefix)
            run.bold = True
        self._rich(p, text, italic=italic)

    @staticmethod
    def _rich(p, text: str, italic: bool = False) -> None:
        """Very small markup: **bold** segments."""
        parts = text.split("**")
        for i, part in enumerate(parts):
            if not part:
                continue
            for k, piece in enumerate(part.split("`")):
                if not piece:
                    continue
                run = p.add_run(piece)
                run.bold = i % 2 == 1
                run.italic = italic
                if k % 2 == 1:
                    _set_font(run, name="Consolas", size=10.5)
                    run.bold = i % 2 == 1

    def bullets(self, items: Iterable[str], numbered: bool = False) -> None:
        for item in items:
            p = self.doc.add_paragraph(style="List Number" if numbered else "List Bullet")
            self._rich(p, item)

    def equation(self, expression: str, where: str | None = None) -> str:
        """Centred equation with a right-aligned chapter-wise number; returns the number."""
        self.eq_no += 1
        number = f"({self.chapter}.{self.eq_no})"
        p = self._p()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        width = self.doc.sections[-1].page_width - self.doc.sections[-1].left_margin - self.doc.sections[-1].right_margin
        stops = p.paragraph_format.tab_stops
        stops.add_tab_stop(int(width / 2), WD_TAB_ALIGNMENT.CENTER)
        stops.add_tab_stop(int(width), WD_TAB_ALIGNMENT.RIGHT)
        run = p.add_run("\t" + expression)
        _set_font(run, name="Cambria Math", size=12)
        run.italic = False
        p.add_run("\t" + number)
        if where:
            w = self._p()
            self._rich(w, where)
        return number

    def figure_block(self, path: Path, caption: str, width_in: float = 6.0, interpretation: str | None = None) -> str:
        self.figure += 1
        label = f"Figure {self.chapter}.{self.figure}"
        p = self._p()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        p.add_run().add_picture(str(path), width=Inches(min(width_in, 5.75)))
        self._p("Figure Caption").add_run(f"{label}: {caption}")
        if interpretation:
            self.para(interpretation, bold_prefix="Interpretation: ")
        return label

    def next_figure_label(self) -> str:
        return f"Figure {self.chapter}.{self.figure + 1}"

    def next_table_label(self) -> str:
        return f"Table {self.chapter}.{self.table + 1}"

    def table_block(self, caption: str, header: Sequence[str], rows: Sequence[Sequence[str]],
                    widths: Sequence[float] | None = None, interpretation: str | None = None,
                    bold_rows: Sequence[int] = ()) -> str:
        self.table += 1
        label = f"Table {self.chapter}.{self.table}"
        self._p("Table Caption").add_run(f"{label}: {caption}")
        table = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        table.style = self.doc.styles["Table Grid"] if "Table Grid" in [s.name for s in self.doc.styles] else None
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, text in enumerate(header):
            self._cell(table.cell(0, j), text, bold=True, shade=True)
        for i, row in enumerate(rows, start=1):
            for j, text in enumerate(row):
                self._cell(table.cell(i, j), str(text), bold=(i - 1) in bold_rows)
        # Repeat header row across pages.
        trpr = table.rows[0]._tr.get_or_add_trPr()
        header_flag = OxmlElement("w:tblHeader")
        header_flag.set(qn("w:val"), "true")
        trpr.append(header_flag)
        text_width = 5.75
        widths = list(widths) if widths else [text_width / len(header)] * len(header)
        scale = min(1.0, text_width / sum(widths))
        widths = [w * scale for w in widths]
        table.autofit = False
        tblpr = table._tbl.tblPr
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "fixed")
        tblpr.append(layout)
        margins = OxmlElement("w:tblCellMar")
        for side in ("left", "right"):
            m = OxmlElement(f"w:{side}")
            m.set(qn("w:w"), "60")
            m.set(qn("w:type"), "dxa")
            margins.append(m)
        tblpr.append(margins)
        for col, w in zip(table._tbl.tblGrid.findall(qn("w:gridCol")), widths):
            col.set(qn("w:w"), str(int(w * 1440)))
        for row in table.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Inches(w)
        spacer = self._p()
        spacer.paragraph_format.space_after = Pt(2)
        if interpretation:
            self.para(interpretation, bold_prefix="Interpretation: ")
        return label

    @staticmethod
    def _cell(cell, text: str, bold: bool = False, shade: bool = False) -> None:
        cell.text = ""
        p = cell.paragraphs[0]
        p.style = "Table Text"
        run = p.add_run(text)
        run.bold = bold
        if shade:
            tcpr = cell._tc.get_or_add_tcPr()
            fill = OxmlElement("w:shd")
            fill.set(qn("w:val"), "clear")
            fill.set(qn("w:color"), "auto")
            fill.set(qn("w:fill"), "D9E2F3")
            tcpr.append(fill)

    def code(self, text: str) -> None:
        for line in text.rstrip("\n").split("\n"):
            p = self._p("Code Block")
            p.add_run(line.replace("\t", "    ") if line else " ")

    def toc(self, instruction: str, placeholder: str) -> None:
        p = self._p()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        _field(p, instruction, placeholder)

    def finish(self, path: Path) -> None:
        settings = self.doc.settings.element
        update = settings.find(qn("w:updateFields"))
        if update is None:
            update = OxmlElement("w:updateFields")
            settings.append(update)
        update.set(qn("w:val"), "true")
        self.doc.save(str(path))


def clone(element):
    return copy.deepcopy(element)
