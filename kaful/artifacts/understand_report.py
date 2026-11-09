"""Phase 2 (understand) report: how the MachineModel and SensorContract were derived.

Saved data, not console output: the context the prompts saw, each item's agreement
across the N samples and the class of each of its evidence quotes, every downgrade
(hard rule 4), and every reference or unit that was dropped or failed a check.
"""

from operator import attrgetter
from typing import Annotated, Literal

from pydantic import Field

from kaful.artifacts.base import (
    ArtifactBase,
    EvidenceClass,
    Fraction,
    NonEmptyStr,
    SourceKind,
    StrictModel,
    sorted_unique,
)

ItemKind = Literal["boundary", "channel", "component", "fault_mode", "flow_path"]


class ContextSelection(StrictModel):
    """What the prompts were given: the whole index, or the chunks retrieval picked."""

    mode: Literal["whole", "retrieval"]
    chunk_ids: tuple[NonEmptyStr, ...] = Field(description="In the order they appear in the prompt.")
    estimated_tokens: int = Field(ge=0)
    queries: tuple[NonEmptyStr, ...] = Field(default=(), description="Retrieval queries, in the order run.")
    documents: tuple[NonEmptyStr, ...] = Field(default=(), description=(
        "doc_ids the context was drawn from, sorted (2026-11-08: machine documents only for components "
        "and channels)."))
    context_sha256: NonEmptyStr = Field(description="sha256 of the rendered context text.")


class ItemCheck(StrictModel):
    kind: ItemKind
    id: NonEmptyStr
    agreement: Fraction = Field(description="Share of the N samples that produced this item.")
    evidence: tuple[EvidenceClass, ...] = Field(description="Class of each evidence quote, in quote order.")
    source: SourceKind
    downgraded_from: SourceKind | None = None
    confidence: Fraction
    found_by: Literal["components", "fault_modes"] | None = Field(default=None, description=(
        "For a component: the step that proposed it, 'fault_modes' for a carrier (M7b, approved 2026-11-09). "
        "None for other kinds, and in reports written before M7b."))


class UnderstandReport(ArtifactBase):
    kind: Literal["understand_report"] = "understand_report"
    context: ContextSelection
    contexts: dict[str, ContextSelection] = Field(default={}, description=(
        "Each prompt's own context by prompt id (2026-11-08); `context` is the components prompt's."))
    n_samples: int = Field(ge=1)
    prompts: tuple[NonEmptyStr, ...] = Field(description="'<prompt id> v<version> <model>' per prompt, sorted.")
    items: Annotated[tuple[ItemCheck, ...], sorted_unique(attrgetter("kind", "id"), "items")]
    merges: tuple[NonEmptyStr, ...] = Field(default=(), description=(
        "Component ids merged across samples: 'kept <- other (reason), ...' (sorted)."))
    dropped: tuple[NonEmptyStr, ...] = Field(default=(), description="References removed, with the reason (sorted).")
    unit_problems: tuple[NonEmptyStr, ...] = Field(default=(), description="Units pint cannot parse (sorted).")
    notes: tuple[NonEmptyStr, ...] = Field(default=(), description="Other decisions code made (sorted).")

    def downgraded(self) -> tuple[ItemCheck, ...]:
        return tuple(i for i in self.items if i.downgraded_from is not None)
