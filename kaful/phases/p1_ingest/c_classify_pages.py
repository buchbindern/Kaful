"""Page classifier: decides which pages need the LLM layer, from PyMuPDF signals only.

Rules apply in order; the first match wins. Thresholds live in `[ingest.classifier]`.

    blank          little text, no image, few drawings      -> not chunked
    scanned        little text, mostly image                -> LLM (OCR)
    garbled        broken text layer                        -> LLM
    table_unruled  many short / numeric lines, no ruled table -> LLM (column-major cell soup)
    figure         a raster image of real size, or many curves -> LLM
    table_ruled    find_tables found a good table           -> PyMuPDF markdown, no LLM
    text           everything else                          -> raw text only

Ruled-table markdown is added to every page that has a good ruled table, whatever its
class. Bump CLASSIFIER_VERSION whenever a rule (not a threshold) changes; thresholds
are recorded in the run manifest's config snapshot.
"""

from kaful.artifacts.document_index import PageClass, PageSignals
from kaful.config import ClassifierSettings

CLASSIFIER_VERSION = "classifier-v1"


def classify_page(signals: PageSignals, settings: ClassifierSettings) -> PageClass:
    s, c = signals, settings
    if s.chars < c.blank_max_chars and s.image_frac < c.blank_max_image_frac and s.drawings < c.blank_max_drawings:
        return "blank"
    if s.chars < c.scanned_max_chars and s.image_frac >= c.scanned_min_image_frac:
        return "scanned"
    if s.garble_ratio > c.garbled_min_ratio:
        return "garbled"
    if (s.ruled_tables == 0 and s.lines >= c.unruled_min_lines and s.short_line_frac >= c.unruled_min_short_frac
            and s.numeric_line_frac >= c.unruled_min_numeric_frac):
        return "table_unruled"
    if s.image_frac >= c.figure_min_image_frac or s.curves >= c.figure_min_curves:
        return "figure"
    if s.ruled_tables > 0:
        return "table_ruled"
    return "text"
