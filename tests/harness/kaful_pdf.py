"""Synthetic PDFs for ingest tests, built with PyMuPDF (no manuals needed).

`write_manual` writes one page of each kind the classifier knows, with printed
"Page N of M" footers. The page order is fixed and documented in PAGE_KINDS.
"""

from pathlib import Path

import fitz  # type: ignore[import-untyped]

PAGE_KINDS = ("text", "table_ruled", "table_unruled", "figure_image", "figure_curves", "blank")
WIDTH, HEIGHT = 595.0, 842.0
BODY = ("The booster pump P1 raises supply pressure. Replace the inlet cartridge filter F1 every "
        "2000 operating hours or when the differential pressure exceeds 0.8 bar. ")


def _footer(page: fitz.Page, number: int, total: int) -> None:
    page.insert_text((260, HEIGHT - 20), f"Page {number} of {total}", fontsize=8)


def _text(page: fitz.Page) -> None:
    page.insert_textbox(fitz.Rect(60, 80, WIDTH - 60, HEIGHT - 120), BODY * 6, fontsize=10)


def _ruled_table(page: fitz.Page) -> None:
    rows = [("Parameter", "Value", "Unit"), ("Rated flow", "6,0", "m3/h"), ("Inlet pressure", "1,5", "bar"),
            ("Motor power", "1,5", "kW")]
    x, y, w, h = 80.0, 120.0, 140.0, 24.0
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = fitz.Rect(x + c * w, y + r * h, x + (c + 1) * w, y + (r + 1) * h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 4, rect.y1 - 8), cell, fontsize=9)


def _unruled_table(page: fitz.Page) -> None:
    labels = ["Water Quantity [tick]", "Cake Thickness [mm]", "Tamping [kg]", "Pre-infusion [s]", "Milk temp [°C]",
              "Foam [%]", "Grind time [s]", "Steam [s]", "Hot water [ml]", "Bypass [ml]"]
    y = 90.0
    for i, label in enumerate(labels):
        for text in (label, f"{10 + i * 5},0", "-"):
            page.insert_text((80, y), text, fontsize=9)
            y += 14


def _image(page: fitz.Page) -> None:
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 64, 64), False)
    pixmap.set_rect(pixmap.irect, (40, 120, 200))
    page.insert_image(fitz.Rect(80, 300, 400, 600), pixmap=pixmap)
    page.insert_text((80, 120), "Figure 3: touch screen, countdown shows 10 s.", fontsize=10)


def _curves(page: fitz.Page) -> None:
    shape = page.new_shape()
    for i in range(200):
        shape.draw_bezier((80, 400 + i % 50), (150, 300), (250, 500), (400, 380 + i % 40))
    shape.finish(color=(0, 0, 0), width=0.3)
    shape.commit()
    page.insert_text((80, 120), "Figure 4: torque-speed characteristic.", fontsize=10)


BUILDERS = {"text": _text, "table_ruled": _ruled_table, "table_unruled": _unruled_table,
            "figure_image": _image, "figure_curves": _curves, "blank": lambda _p: None}


def write_manual(path: Path, kinds: tuple[str, ...] = PAGE_KINDS, footers: bool = True,
                 toc: bool = True) -> Path:
    """Write a synthetic manual with one page per entry of `kinds`."""
    doc = fitz.open()
    for number, kind in enumerate(kinds, start=1):
        page = doc.new_page(width=WIDTH, height=HEIGHT)
        BUILDERS[kind](page)
        if footers and kind != "blank":
            _footer(page, number, len(kinds))
    if toc:
        entries = [[1, "Overview", 1], [1, "Data", 2], [2, "Settings", 3], [1, "Figures", 4]]
        doc.set_toc([e for e in entries if e[2] <= len(kinds)])
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path, garbage=4, deflate=True, no_new_id=True)
    return path
