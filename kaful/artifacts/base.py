"""Shared building blocks for every artifact.

Conventions enforced here:
- models are frozen and reject unknown fields;
- floats must be finite;
- every collection is a tuple that validators require to be sorted and unique,
  so no artifact depends on set or dict iteration order (hard rule 14);
- each artifact carries `schema_version` and a `kind` discriminator.
"""

from collections.abc import Callable, Hashable, Sequence
from operator import attrgetter
from typing import Annotated, Any, Literal, TypeVar, get_args

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from kaful.paths import RelPath

# `deployment`: declared by the user in machine.toml (e.g. an added accelerometer), not
# found in any document; its ref must start with "machine.toml".
SourceKind = Literal[
    "manual", "standard", "handbook", "literature", "calibrated", "data", "deployment", "llm_judgement", "default"
]
SOURCE_KINDS: tuple[SourceKind, ...] = get_args(SourceKind)

# in_layer: found only in an LLM transcription layer of the cited page (a table or figure
# read from the page image), not in its raw text, so rule 4 does not accept it as `manual`.
EvidenceClass = Literal["unverified", "verbatim", "near", "in_layer", "unsupported"]
DEPLOYMENT_REF_PREFIX = "machine.toml"

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
NonEmptyStr = Annotated[str, Field(min_length=1)]

T = TypeVar("T")


class StrictModel(BaseModel):
    """Base for all Kaful models: frozen, no extra fields, finite floats."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def sorted_unique(key: Callable[[Any], Hashable], what: str = "items") -> AfterValidator:
    """Validator for a tuple that must be strictly increasing by `key`."""

    def check(items: Sequence[T]) -> Sequence[T]:
        keys = [key(item) for item in items]
        for before, after in zip(keys, keys[1:]):
            if before == after:
                raise ValueError(f"duplicate {what}: {before!r}")
            if before > after:  # type: ignore[operator]
                raise ValueError(f"{what} must be sorted: {before!r} comes before {after!r}")
        return items

    return AfterValidator(check)


def by_attr(name: str) -> AfterValidator:
    """`sorted_unique` keyed on one attribute; the error names the attribute."""
    return sorted_unique(attrgetter(name), what=name)


SORTED_STRS = sorted_unique(lambda s: s, what="names")


class Evidence(StrictModel):
    """A quote from a document page that supports a value or claim."""

    doc_id: NonEmptyStr
    page: int = Field(ge=1, description="Physical 1-based page index in the document.")
    page_label: str | None = Field(default=None, description="Printed page label, e.g. '4-12'.")
    quote: NonEmptyStr
    verification: EvidenceClass = "unverified"


class Provenanced(StrictModel):
    """Mixin for anything that carries a number or claim with provenance (hard rule 3).

    `confidence` is computed by code, never taken from LLM output (hard rule 15). It is
    derived from measurable signals: agreement across repeated extractions, the
    verification class of the evidence quotes, and the source kind. Prompt output
    models must not contain a confidence field.

    `downgraded_from` is set by code when a claimed source fails a check, e.g. a
    `manual` value whose quote is not found on the cited page (hard rule 4).
    """

    source: SourceKind
    ref: NonEmptyStr = Field(description="Citation, e.g. 'manual.pdf p.72' or 'ISO 281:2007 §6'.")
    confidence: Fraction
    evidence: tuple[Evidence, ...] = ()
    downgraded_from: SourceKind | None = None

    @model_validator(mode="after")
    def _check_provenance(self) -> "Provenanced":
        if self.source == "manual" and not self.evidence:
            raise ValueError("source 'manual' requires at least one evidence quote")
        if self.downgraded_from == self.source:
            raise ValueError("downgraded_from must differ from source")
        if self.source == "deployment" and not self.ref.startswith(DEPLOYMENT_REF_PREFIX):
            raise ValueError(f"source 'deployment' must cite {DEPLOYMENT_REF_PREFIX!r}, got {self.ref!r}")
        return self


class ArtifactRef(StrictModel):
    """Pointer to a saved artifact: where it is and the hash of its bytes."""

    kind: Identifier
    run_id: NonEmptyStr
    path: RelPath
    sha256: Sha256


class ArtifactBase(StrictModel):
    """Fields every phase artifact carries. Subclasses add `kind: Literal[...]`."""

    schema_version: Literal[1] = 1
    run_id: NonEmptyStr
    machine_id: Identifier
    index_hash: str | None = Field(
        default=None, description="Hash of the DocumentIndex this artifact was derived from."
    )
    inputs: Annotated[
        tuple[ArtifactRef, ...], sorted_unique(attrgetter("kind", "path"), "inputs")
    ] = ()
