"""Build the synthetic fixture corpus shared by every knowledge-base test (handoff KWP1).

Everything here is invented: the vendors (Coastline Home Inspections, Tidewater
Plumbing & Rooter, Bayside Electric Co., Harborline), the people, the address,
the amounts and every reference number. No house data (rule K2).

Run it to regenerate ``tests/knowledge/fixtures/files/``::

    uv run python -m tests.knowledge.fixtures.build_fixture_corpus

The output is byte-for-byte reproducible for the locked ``reportlab`` and
``python-docx`` versions (``invariant=1`` PDFs, fixed DOCX timestamps); a test
rebuilds it into a temp dir and compares. :mod:`tests.knowledge.fixtures.corpus`
exposes the paths and the ground truth that later work packages test against.

Files:

- ``inspection.pdf``: 12 pages. A DocuSign-style envelope line and an
  inspector line on top of every page, the property address and "Page N of 12"
  at the bottom, a summary list, three sections of numbered findings with
  severity and recommendation, a photo page with **no text layer** (p. 9), a
  hyphenated standards page, an invoice page and a certification page.
- ``estimate.pdf``: 2 pages with letter-spaced headings ("D ES C R I PT I O N O F").
- ``invoice.pdf``: 1 page.
- ``bundle.pdf``: the three above concatenated (15 pages), for segmentation.
- ``heat_pump_guide.docx``: a title and three top-level headings.
- ``irrigation_notes.html``: a title and three second-level sections, plus
  script and style that must not leak into the text.
- ``garage_door.md``: a title and three ``##`` sections, with a fenced block.
- ``pool_log.txt``: about 7,000 characters with no headings.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

FILES_DIR = Path(__file__).resolve().parent / "files"

# ------------------------------------------------------------------ content
ENVELOPE_LINE = "Docusign Envelope ID: 5E7C1A90-3B2D-4C6E-9F10-A1B2C3D4E5F6"
INSPECTOR_LINE = "Coastline Home Inspections | Inspector: Jordan Avery Reyes | Report CHI-2025-0314"
ADDRESS = "742 Driftwood Way, Port Alder, CA 90000"
INSPECTION_PAGES = 12
INSPECTION_VENDOR = "Coastline Home Inspections"
INSPECTION_DATE = "2025-03-14"
ESTIMATE_VENDOR = "Tidewater Plumbing & Rooter, Inc."
ESTIMATE_DATE = "2025-03-20"
INVOICE_VENDOR = "Bayside Electric Co."
INVOICE_DATE = "2025-04-02"
PHOTO_PAGE = 9

#: The recommendation line that repeats on exactly two pages (pp. 3 and 4), at
#: the bottom of each: cleaning must keep it.
REPEATED_LINE = "Recommendation: Recommend a qualified contractor evaluate and repair."

#: Letter-spaced headings exactly as PDFium extracts them, and their repair.
LETTER_SPACED = {
    "E S T I M A T E": "ESTIMATE",
    "D ES C R I PT I O N O F W O R K": "DESCRIPTION OF WORK",
    "T E R M S A N D C O N D I T I O N S": "TERMS AND CONDITIONS",
}


@dataclass(frozen=True)
class Finding:
    id: str
    area: str
    component: str
    title: str
    severity: str
    description: str
    recommendation: str
    page: int

    @property
    def heading(self) -> str:
        return f"{self.id} {self.area} – {self.component}: {self.title}"


SECTIONS = {
    1: ("ROOF AND EXTERIOR", (3, 4)),
    2: ("ELECTRICAL", (5, 6)),
    3: ("PLUMBING", (7, 8)),
}

FINDINGS: tuple[Finding, ...] = (
    Finding("1.1.1", "Roof", "Covering", "Cracked Tiles", "Repair",
            "Four cracked concrete tiles were observed on the south slope above the garage. "
            "Cracked tiles can let water reach the underlayment and the roof sheathing.",
            "Recommendation: Recommend a licensed roofing contractor replace the cracked tiles.", 3),
    Finding("1.1.2", "Roof", "Flashing", "Loose Step Flashing", "Repair",
            "Step flashing at the chimney is loose and the sealant along its top edge has "
            "failed. Staining on the attic sheathing below suggests past leaks.",
            REPEATED_LINE, 3),
    Finding("1.2.1", "Exterior", "Gutters", "Downspout Discharges at Foundation", "Maintenance",
            "The rear left downspout discharges next to the foundation, and the soil there "
            "slopes toward the house.",
            "Recommendation: Recommend extending the downspout at least 4 feet from the house.", 4),
    Finding("1.3.1", "Exterior", "Siding", "Wood Rot at Fascia Trim", "Repair",
            "Soft, deteriorated wood was found at the fascia trim near the back door. The "
            "paint has peeled and the wood is exposed.",
            REPEATED_LINE, 4),
    Finding("2.1.1", "Electrical", "Service Panel", "Double-Tapped Breaker", "Safety Hazard",
            "Two conductors are connected to a single-pole 20-amp breaker in the main panel. "
            "This breaker is not rated for two conductors.",
            "Recommendation: Recommend a licensed electrician correct the double tap.", 5),
    Finding("2.2.1", "Electrical", "Receptacles", "No GFCI Protection in Kitchen",
            "Safety Hazard",
            "Kitchen counter receptacles to the right of the sink are not GFCI protected.",
            "Recommendation: Recommend a licensed electrician install GFCI protection.", 5),
    Finding("2.3.1", "Electrical", "Smoke Alarms", "Missing in Hallway", "Safety Hazard",
            "No smoke alarm was found in the hallway outside the bedrooms.",
            "Recommendation: Recommend installing a smoke alarm in the hallway.", 6),
    Finding("2.3.2", "Electrical", "Lighting", "Inoperable Garage Light", "Monitor",
            "The garage ceiling light did not respond to the switch. The bulb may be "
            "burned out.",
            "Recommendation: Recommend replacing the bulb and testing the fixture again.", 6),
    Finding("3.1.1", "Plumbing", "Water Heater", "Age", "Monitor",
            "The 50-gallon gas water heater was manufactured in 2012 and is 13 years old, "
            "near the end of its typical service life.",
            "Recommendation: Recommend budgeting for replacement.", 7),
    Finding("3.1.2", "Plumbing", "Water Heater", "Valve Leak", "Repair",
            "The temperature and pressure relief valve is dripping, and mineral deposits "
            "have formed at the discharge pipe.",
            "Recommendation: Recommend a licensed plumber replace the relief valve.", 7),
    Finding("3.1.3", "Plumbing", "Water Heater", "No Expansion Tank", "Maintenance",
            "No thermal expansion tank is installed on the cold water supply line.",
            "Recommendation: Recommend a licensed plumber install a thermal expansion tank.", 7),
    Finding("3.2.1", "Plumbing", "Supply Lines", "Galvanized Piping", "Monitor",
            "Galvanized steel supply piping was observed in the crawlspace. Reduced water "
            "pressure was noted at the upstairs shower.",
            "Recommendation: Recommend a licensed plumber evaluate the supply piping.", 8),
    Finding("3.3.1", "Plumbing", "Main Shutoff", "Location", "Maintenance",
            "The main water shutoff valve is located at the front of the house, left of the "
            "garage door.",
            "Recommendation: Recommend labeling the valve so it can be found quickly.", 8),
)  # fmt: skip

INVOICE_PAGE_LINES = (
    "Coastline Home Inspections",
    "Invoice number: CHI-INV-20250314",
    "Invoice date: March 14, 2025",
    "Bill to: Morgan Example",
    f"Service address: {ADDRESS}",
    "",
    "Home inspection, single-family (1,850 sq ft): $475.00",
    "Sewer lateral video scope: $150.00",
    "Total due: $625.00",
    "",
    "Paid in full on March 14, 2025. Balance due: $0.00",
    "Thank you for choosing Coastline Home Inspections.",
)

STANDARDS_LINES = (
    "This inspection was performed in accordance with the Standards of Practice of the",
    "Pacific Association of Home Inspectors (a fictional association). The inspection is a",
    "visual, non-invasive examination of the readily accessible systems and components of the",
    "home. The inspector is not required to deter-",
    "mine the remaining life of any system or component, to move furniture or stored items,",
    "or to enter any area that is unsafe or not readily accessible.",
    "",
    "The report describes the condition of the house on the day of the inspection. Conditions",
    "change over time, and items that were working on the day of the inspection may fail",
    "later. The client should have every item marked Repair or Safety Hazard evaluated by a",
    "licensed profes-",
    "sional before the end of the contingency period.",
)


# ------------------------------------------------------------------ drawing
PAGE_W, PAGE_H = 612, 792
LEFT, RIGHT = 60, 552
BODY_FONT, BOLD_FONT = "Helvetica", "Helvetica-Bold"


class _Page:
    """A cursor over one reportlab page."""

    def __init__(self, canvas, top: float = 720) -> None:
        self.c = canvas
        self.y = top

    def line(self, text: str, font: str = BODY_FONT, size: float = 10, lead: float = 14) -> None:
        if text:
            self.c.setFont(font, size)
            self.c.drawString(LEFT, self.y, text)
        self.y -= lead

    def para(self, text: str, font: str = BODY_FONT, size: float = 10, lead: float = 13) -> None:
        from reportlab.lib.utils import simpleSplit

        for part in simpleSplit(text, font, size, RIGHT - LEFT):
            self.line(part, font, size, lead)

    def gap(self, amount: float = 8) -> None:
        self.y -= amount

    def spaced(self, spaced: str, size: float = 13) -> None:
        """Draw a heading glyph group by glyph group, as a letter-spaced PDF does."""
        from reportlab.pdfbase.pdfmetrics import stringWidth

        self.c.setFont(BOLD_FONT, size)
        x = LEFT
        for group in spaced.split(" "):
            self.c.drawString(x, self.y, group)
            x += stringWidth(group, BOLD_FONT, size) + 6
        self.y -= size + 8


def _inspection_frame(canvas, number: int) -> None:
    canvas.setFont(BODY_FONT, 7)
    canvas.drawString(LEFT, 772, ENVELOPE_LINE)
    canvas.drawString(LEFT, 760, INSPECTOR_LINE)
    canvas.drawString(LEFT, 40, ADDRESS)
    canvas.drawRightString(RIGHT, 28, f"Page {number} of {INSPECTION_PAGES}")


def _photo_page(canvas) -> None:
    """A page that is only a picture: no text at all, so no text layer."""
    from PIL import Image, ImageDraw
    from reportlab.lib.utils import ImageReader

    image = Image.new("RGB", (160, 120))
    pixels = image.load()
    for x in range(160):
        for y in range(120):
            pixels[x, y] = (40 + x // 2, 90 + y, 150 - x // 3)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 30, 70, 90), fill=(170, 60, 40))
    draw.polygon([(90, 90), (120, 40), (150, 90)], fill=(90, 90, 90))
    canvas.drawImage(ImageReader(image), 96, 250, width=420, height=315)


def _findings_page(canvas, number: int, section: int, first_page: bool) -> None:
    title, _ = SECTIONS[section]
    page = _Page(canvas)
    if first_page:
        page.line(f"{section}. {title}", BOLD_FONT, 14, 22)
    else:
        page.line(f"{section}. {title} (continued)", BOLD_FONT, 12, 20)
    for finding in (f for f in FINDINGS if f.page == number):
        page.line(finding.heading, BOLD_FONT, 11, 15)
        page.line(f"Severity: {finding.severity}", BODY_FONT, 10, 14)
        page.para(finding.description)
        page.line(finding.recommendation)
        page.gap(10)


def draw_inspection(canvas) -> None:
    for number in range(1, INSPECTION_PAGES + 1):
        if number == PHOTO_PAGE:
            _photo_page(canvas)
            canvas.showPage()
            continue
        _inspection_frame(canvas, number)
        page = _Page(canvas)
        if number == 1:
            page.line("HOME INSPECTION REPORT", BOLD_FONT, 20, 30)
            for text in (
                f"Property: {ADDRESS}",
                "Client: Morgan Example",
                "Inspection date: March 14, 2025",
                "Inspector: Jordan Avery Reyes, License HI-000000",
                "Prepared by Coastline Home Inspections",
                "Report number: CHI-2025-0314",
                "",
                "Weather: Clear, 64 degrees F. The house was vacant and the utilities were on.",
                "Year built: 1987. Approximate size: 1,850 sq ft. Foundation: raised, with crawlspace.",
            ):
                page.line(text, lead=16)
        elif number == 2:
            page.line("SUMMARY", BOLD_FONT, 14, 22)
            page.para(
                "This summary lists the items that need attention. It is not the whole report; "
                "read every section, including the limitations."
            )
            page.gap()
            for finding in FINDINGS:
                page.line(f"{finding.heading} ({finding.severity})", lead=15)
        elif number in (3, 5, 7):
            _findings_page(canvas, number, (number - 1) // 2, first_page=True)
        elif number in (4, 6, 8):
            _findings_page(canvas, number, (number - 2) // 2, first_page=False)
        elif number == 10:
            page.line("STANDARDS OF PRACTICE AND LIMITATIONS", BOLD_FONT, 13, 20)
            for text in STANDARDS_LINES:
                page.line(text, lead=13)
        elif number == 11:
            page.line("INVOICE", BOLD_FONT, 16, 24)
            for text in INVOICE_PAGE_LINES:
                page.line(text, lead=15)
        elif number == 12:
            page.line("INSPECTOR CERTIFICATION", BOLD_FONT, 13, 20)
            page.para(
                "I certify that I personally inspected the property named in this report and "
                "that I have no interest in the property or in any repairs. The report was "
                "completed on March 15, 2025."
            )
            page.gap()
            page.line("Jordan Avery Reyes, Coastline Home Inspections")
        canvas.showPage()


ESTIMATE_LETTERHEAD = (ESTIMATE_VENDOR, "License C36-000000 | (555) 010-0100 | Port Alder, CA")


def draw_estimate(canvas) -> None:
    # page 1
    page = _Page(canvas, top=740)
    page.line(ESTIMATE_LETTERHEAD[0], BOLD_FONT, 15, 18)
    page.line(ESTIMATE_LETTERHEAD[1], BODY_FONT, 8, 26)
    page.spaced("E S T I M A T E", 16)
    for text in (
        "Estimate No. TPR-58213",
        "Date: March 20, 2025",
        "Customer: Morgan Example",
        f"Job site: {ADDRESS}",
    ):
        page.line(text)
    page.gap(10)
    page.spaced("D ES C R I PT I O N O F W O R K")
    for text in (
        "1. Replace the water heater temperature and pressure relief valve: $185.00",
        "2. Install a 2-gallon thermal expansion tank on the cold water line: $420.00",
        "",
        "Subtotal: $605.00",
        "Tax: $0.00",
        "Total: $605.00",
    ):
        page.line(text)
    canvas.setFont(BODY_FONT, 8)
    canvas.drawRightString(RIGHT, 28, "Page 1 of 2")
    canvas.showPage()
    # page 2
    page = _Page(canvas, top=740)
    page.line(ESTIMATE_LETTERHEAD[0], BOLD_FONT, 15, 18)
    page.line(ESTIMATE_LETTERHEAD[1], BODY_FONT, 8, 26)
    page.spaced("T E R M S A N D C O N D I T I O N S")
    page.para(
        "This estimate is valid for 30 days. A 50% deposit is due before work begins and the "
        "balance is due on completion. All work carries a one-year labor warranty. Permits, "
        "if required, are included in the price."
    )
    page.gap(20)
    page.line("Accepted by: ______________________    Date: ____________")
    canvas.setFont(BODY_FONT, 8)
    canvas.drawRightString(RIGHT, 28, "Page 2 of 2")
    canvas.showPage()


def draw_invoice(canvas) -> None:
    page = _Page(canvas, top=740)
    page.line(INVOICE_VENDOR, BOLD_FONT, 15, 18)
    page.line("License C10-000000 | (555) 010-0199", BODY_FONT, 8, 26)
    page.line("INVOICE", BOLD_FONT, 16, 24)
    for text in (
        "Invoice # BE-10442",
        "Date: April 2, 2025",
        "Customer: Morgan Example",
        f"Service address: {ADDRESS}",
    ):
        page.line(text)
    page.gap()
    page.para(
        "Work performed: Separated the double-tapped conductors in the main panel and "
        "installed a new 20-amp breaker. Installed a GFCI receptacle at the kitchen counter."
    )
    page.gap()
    for text in ("Labor (2.0 hours): $240.00", "Materials: $100.00", "Total: $340.00"):
        page.line(text)
    canvas.setFont(BODY_FONT, 8)
    canvas.drawRightString(RIGHT, 28, "Page 1 of 1")
    canvas.showPage()


def _pdf(path: Path, *drawers) -> None:
    from reportlab.pdfgen import canvas as rl_canvas

    canvas = rl_canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H), invariant=1)
    canvas.setAuthor("HAL 9000 fixture builder")
    canvas.setTitle(path.stem)
    canvas.setCreator("tests/knowledge/fixtures/build_fixture_corpus.py")
    for draw in drawers:
        draw(canvas)
    canvas.save()


# --------------------------------------------------------------- text files
DOCX_TITLE = "Harborline HP-3000 Heat Pump: Owner's Guide"
DOCX_HEADINGS = ("Filter Maintenance", "Thermostat Settings", "Warranty Registration")


def _fixed_zip(data: bytes) -> bytes:
    """Rewrite a zip with fixed timestamps and order, so the DOCX is reproducible."""
    source = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            fixed = zipfile.ZipInfo(info.filename, date_time=(2025, 1, 1, 0, 0, 0))
            fixed.compress_type = zipfile.ZIP_DEFLATED
            fixed.external_attr = 0o644 << 16
            target.writestr(fixed, source.read(info.filename))
    return out.getvalue()


def write_docx(path: Path) -> None:
    import docx

    document = docx.Document()
    moment = datetime(2025, 1, 1, 0, 0, 0)
    props = document.core_properties
    props.author = "HAL 9000 fixture builder"
    props.last_modified_by = "HAL 9000 fixture builder"
    props.title = DOCX_TITLE
    props.created = moment
    props.modified = moment
    props.revision = 1

    document.add_paragraph(DOCX_TITLE, style="Title")
    document.add_paragraph("Keep this guide with the unit. Model HP-3000, serial HL-000000-00.")

    document.add_heading(DOCX_HEADINGS[0], level=1)
    document.add_paragraph(
        "Replace the 16 x 25 x 1 inch MERV 8 air filter every 90 days, or every 60 days when "
        "pets live in the house. A clogged filter reduces airflow and can freeze the coil."
    )
    document.add_heading("Checking the filter", level=2)
    document.add_paragraph(
        "Slide the filter out of the return grille in the hallway ceiling. Hold it up to a "
        "light; if you cannot see light through it, replace it."
    )

    document.add_heading(DOCX_HEADINGS[1], level=1)
    document.add_paragraph("The factory schedule uses these setpoints:")
    table = document.add_table(rows=3, cols=2)
    for row, (mode, value) in zip(
        table.rows, (("Mode", "Setpoint"), ("Heat", "68 F"), ("Cool", "76 F"))
    ):
        row.cells[0].text = mode
        row.cells[1].text = value
    document.add_paragraph("Hold the MODE button for five seconds to switch to emergency heat.")

    document.add_heading(DOCX_HEADINGS[2], level=1)
    document.add_paragraph(
        "The compressor carries a 10-year limited warranty when the unit is registered within "
        "60 days of installation. Other parts carry a 5-year limited warranty. Registration "
        "number: HL-WR-000123."
    )

    buffer = io.BytesIO()
    document.save(buffer)
    path.write_bytes(_fixed_zip(buffer.getvalue()))


HTML_TITLE = "Irrigation Controller Notes"
HTML_SECTIONS = ("Zones", "Watering Schedule", "Winterizing")

HTML_SOURCE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{HTML_TITLE}</title>
<style>body {{ font-family: sans-serif; }} .secret {{ display: none; }}</style>
<script>window.trackingPixel = "do-not-index";</script>
</head>
<body>
<h1>{HTML_TITLE}</h1>
<p>Notes for the <b>RainWise 8</b> controller in the garage (a fictional model).</p>
<h2>{HTML_SECTIONS[0]}</h2>
<table>
<tr><th>Zone</th><th>Area</th><th>Heads</th></tr>
<tr><td>1</td><td>Front lawn</td><td>Rotors</td></tr>
<tr><td>2</td><td>Side yard drip</td><td>Drip emitters</td></tr>
<tr><td>3</td><td>Back lawn</td><td>Spray heads</td></tr>
</table>
<h2>{HTML_SECTIONS[1]}</h2>
<ul>
<li>Lawns run Monday, Wednesday and Friday at 5:00 AM for 12 minutes.</li>
<li>Drip runs Tuesday and Saturday for 30 minutes.</li>
</ul>
<p>Rain delay: press <em>RAIN</em> twice for a 48-hour pause.</p>
<script>console.log("not text");</script>
<h2>{HTML_SECTIONS[2]}</h2>
<p>Shut the backflow valve in November, then blow out the lines at 50 psi.<br>
Restart in March.</p>
<noscript>Enable JavaScript.</noscript>
</body>
</html>
"""

MARKDOWN_SECTIONS = ("Opener", "Remote Codes", "Maintenance")
MARKDOWN_SOURCE = f"""# Garage Door Opener

A LiftRight 1/2 HP belt-drive opener (a fictional model), installed in 2019.

## {MARKDOWN_SECTIONS[0]}

The opener is on a dedicated 15-amp circuit, breaker 14 in the garage subpanel.
The travel limits were last adjusted in 2024.

## {MARKDOWN_SECTIONS[1]}

To pair a new remote, press the LEARN button on the motor head once, then press
the remote button within 30 seconds.

```
# this line is inside a code block, not a heading
LEARN -> remote button -> light blinks twice
```

## {MARKDOWN_SECTIONS[2]}

Lubricate the rail and rollers every six months with white lithium grease.
Test the auto-reverse with a 2x4 laid flat under the door every month.
"""

_POOL_SENTENCES = (
    "Checked the pool pump basket and removed leaves",
    "Free chlorine read {c} ppm and pH read {p}",
    "Backwashed the sand filter until the sight glass ran clear",
    "Brushed the walls and the steps, then vacuumed the deep end",
    "Topped up the water level by about {w} inches",
    "The heater was off; the cover stayed on overnight",
    "Added {s} pounds of salt to the chlorinator",
    "Cleaned the skimmer weir and checked the O-ring for cracks",
)


def pool_log_text(target: int = 7000) -> str:
    """Heading-less prose, about ``target`` characters, deterministic."""
    paragraphs: list[str] = []
    day = 0
    while sum(len(p) + 2 for p in paragraphs) < target:
        day += 1
        values = {
            "c": 1 + day % 3,
            "p": f"7.{2 + day % 5}",
            "w": 1 + day % 2,
            "s": 10 * (1 + day % 4),
        }
        sentences = [
            _POOL_SENTENCES[(day + k) % len(_POOL_SENTENCES)].format(**values) + "."
            for k in range(4)
        ]
        paragraphs.append(f"Visit {day}. " + " ".join(sentences))
    return "\n\n".join(paragraphs) + "\n"


# --------------------------------------------------------------------- main
FILE_NAMES = (
    "inspection.pdf",
    "estimate.pdf",
    "invoice.pdf",
    "bundle.pdf",
    "heat_pump_guide.docx",
    "irrigation_notes.html",
    "garage_door.md",
    "pool_log.txt",
)


def build(out_dir: Path = FILES_DIR) -> list[Path]:
    """Write every fixture file into ``out_dir``; returns the paths in :data:`FILE_NAMES` order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _pdf(out_dir / "inspection.pdf", draw_inspection)
    _pdf(out_dir / "estimate.pdf", draw_estimate)
    _pdf(out_dir / "invoice.pdf", draw_invoice)
    _pdf(out_dir / "bundle.pdf", draw_inspection, draw_estimate, draw_invoice)
    write_docx(out_dir / "heat_pump_guide.docx")
    (out_dir / "irrigation_notes.html").write_text(HTML_SOURCE, encoding="utf-8")
    (out_dir / "garage_door.md").write_text(MARKDOWN_SOURCE, encoding="utf-8")
    (out_dir / "pool_log.txt").write_text(pool_log_text(), encoding="utf-8")
    return [out_dir / name for name in FILE_NAMES]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - a developer command
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=FILES_DIR, help="output directory")
    args = parser.parse_args(argv)
    for path in build(args.out):
        print(f"{path.stat().st_size:>8}  {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
