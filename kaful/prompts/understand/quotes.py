"""Evidence quote shared by the understand prompts' output models.

The model cites the document and page it copied from; code then checks the quote
against that page's raw text (hard rule 4). No output model has a confidence field
(hard rule 15): confidence is computed from agreement and evidence.
"""

from pydantic import BaseModel, ConfigDict, Field

ID_PATTERN = r"^[a-z][a-z0-9_]*$"


class Quote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(pattern=ID_PATTERN, description="The document id from the [doc_id | section] header.")
    page: int = Field(ge=1, description="The page number from the nearest '--- page N' marker above the quote.")
    quote: str = Field(min_length=1, description="Text copied exactly from that page.")


def duplicates(names: list[str]) -> list[str]:
    return sorted({n for n in names if names.count(n) > 1})
