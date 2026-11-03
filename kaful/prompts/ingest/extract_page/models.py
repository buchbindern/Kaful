"""Output model for ingest.extract_page: tables, figures and (for unreadable text layers) OCR text.

The page's raw text layer is kept separately and is what evidence quotes are checked
against; this output only adds what the text layer cannot carry.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TableRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cells: list[str] = Field(description="Cell texts in column order, exactly as printed; '' for an empty cell.")


class Table(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(description="The table's caption or title as printed; null if none.")
    source: Literal["text_layer", "image"] = Field(
        description="text_layer: the cell texts appear in the provided text layer; image: they are only in the image.")
    header: list[str] = Field(description="Column headings as printed (joined if they span several lines).")
    rows: list[TableRow]


class FigureValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, description="What the value is, as labelled in the figure.")
    value: str = Field(min_length=1, description="The value exactly as shown (keep decimal commas).")
    unit: str | None = Field(description="Unit as shown; null if none is shown.")
    how: Literal["printed", "read_from_curve"] = Field(
        description="printed: shown as text in the figure; read_from_curve: estimated from a plotted line.")


class Figure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["screenshot", "photo", "diagram", "chart", "drawing", "symbol", "other"]
    caption: str | None = Field(description="The figure's caption as printed; null if none.")
    description: str = Field(min_length=1, description="One to three factual sentences on what the figure shows.")
    text_in_image: list[str] = Field(description="Labels and text visible in the figure, exactly as shown.")
    values: list[FigureValue]


class PageExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tables: list[Table]
    figures: list[Figure]
    ocr_text: str | None = Field(
        description="Full page transcription, only when the text layer is empty or unreadable; otherwise null.")
