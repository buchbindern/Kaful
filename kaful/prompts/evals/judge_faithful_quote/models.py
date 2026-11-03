"""Output model for the faithfulness judge. A label and a reason; no confidence (hard rule 15)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rationale: str = Field(min_length=1, description="One or two sentences naming the source text relied on.")
    verdict: Literal["supported", "partial", "unsupported"]
