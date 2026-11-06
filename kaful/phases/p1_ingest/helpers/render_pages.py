"""Page images for the LLM layer: PNG, scaled so the long edge fits the model's limit.

Rendering is deterministic for a pinned PyMuPDF version, so the request hash (and with
it the response cache) is stable across runs.
"""

import fitz  # type: ignore[import-untyped]

from kaful.config import IngestSettings

POINTS_PER_INCH = 72.0


def page_png(page: fitz.Page, settings: IngestSettings) -> bytes:
    """The page as PNG at `render_max_dpi`, or smaller if its long edge would exceed the limit."""
    long_edge_pt = max(page.rect.width, page.rect.height)
    zoom = min(settings.render_max_dpi / POINTS_PER_INCH, settings.render_long_edge_px / long_edge_pt)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return bytes(pixmap.tobytes("png"))
