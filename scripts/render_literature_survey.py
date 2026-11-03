"""Render the editable survey Markdown as the submission PDF."""

from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate
import reportlab

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "LITERATURE_SURVEY.md"
OUTPUT = ROOT / "litsurvey.pdf"
LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
CODE_PATTERN = re.compile(r"`([^`]+)`")
BOLD_PATTERN = re.compile(r"\*\*([^*]+)\*\*")
ITALIC_PATTERN = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")


def _register_fonts() -> None:
    font_dir = Path(reportlab.__file__).resolve().parent / "fonts"
    for name in ("Vera", "VeraBd", "VeraIt", "VeraBI"):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / f"{name}.ttf")))
    pdfmetrics.registerFontFamily(
        "Vera", normal="Vera", bold="VeraBd", italic="VeraIt", boldItalic="VeraBI"
    )


def _inline_markup(text: str) -> str:
    fragments: list[str] = []
    position = 0
    for match in LINK_PATTERN.finditer(text):
        fragments.append(escape(text[position : match.start()]))
        label = escape(match.group(1))
        target = escape(match.group(2), quote=True)
        fragments.append(f'<link href="{target}">{label}</link>')
        position = match.end()
    fragments.append(escape(text[position:]))
    markup = "".join(fragments)
    markup = CODE_PATTERN.sub(r'<font name="Courier">\1</font>', markup)
    markup = BOLD_PATTERN.sub(r"<b>\1</b>", markup)
    return ITALIC_PATTERN.sub(r"<i>\1</i>", markup)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "SurveyTitle",
            parent=base["Title"],
            fontName="VeraBd",
            fontSize=19,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#15324b"),
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "SurveyHeading1",
            parent=base["Heading1"],
            fontName="VeraBd",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#15324b"),
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "SurveyHeading2",
            parent=base["Heading2"],
            fontName="VeraBd",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#254f70"),
            spaceBefore=9,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "SurveyBody",
            parent=base["BodyText"],
            fontName="Vera",
            fontSize=9.5,
            leading=13.5,
            spaceAfter=7,
            splitLongWords=True,
        ),
        "list": ParagraphStyle(
            "SurveyList",
            parent=base["BodyText"],
            fontName="Vera",
            fontSize=9.5,
            leading=13.5,
            leftIndent=16,
            firstLineIndent=-12,
            spaceAfter=7,
            splitLongWords=True,
        ),
    }


def _story(markdown: str, styles: dict[str, ParagraphStyle]) -> list[object]:
    story: list[object] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        text = " ".join(line.strip() for line in paragraph_lines)
        list_match = re.match(r"^(\d+\.|-)\s+(.*)$", text)
        if list_match:
            marker, content = list_match.groups()
            story.append(
                Paragraph(_inline_markup(content), styles["list"], bulletText=marker)
            )
        else:
            story.append(Paragraph(_inline_markup(text), styles["body"]))
        paragraph_lines.clear()

    for line in markdown.splitlines():
        if not line.strip():
            flush_paragraph()
            continue
        if re.match(r"^(\d+\.|-)\s+", line):
            flush_paragraph()
            paragraph_lines.append(line)
            flush_paragraph()
            continue
        heading = re.match(r"^(#{1,2})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            style = styles["title"] if level == 1 else styles["h1"]
            story.append(
                KeepTogether([Paragraph(_inline_markup(heading.group(2)), style)])
            )
        elif line.startswith("### "):
            flush_paragraph()
            story.append(
                KeepTogether([Paragraph(_inline_markup(line[4:]), styles["h2"])])
            )
        else:
            paragraph_lines.append(line)
    flush_paragraph()
    return story


def _draw_page_number(canvas: object, document: object) -> None:
    canvas.saveState()
    canvas.setFont("Vera", 8)
    canvas.setFillColor(colors.HexColor("#52616b"))
    canvas.drawCentredString(letter[0] / 2, 30, str(document.page))
    canvas.restoreState()


def main() -> None:
    _register_fonts()
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=58,
        leftMargin=58,
        topMargin=54,
        bottomMargin=48,
        title="Reinforcement Learning for a Smart Thermostat Energy Saver",
        author="CSCE 642 course project",
    )
    document.build(
        _story(SOURCE.read_text(encoding="utf-8"), _styles()),
        onFirstPage=_draw_page_number,
        onLaterPages=_draw_page_number,
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
