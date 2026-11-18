"""The figure spec (`docs/figures/figures.toml`): the lineages, the figures and the style.

It is committed config and, with the saved artifacts it names, the only input to `kaful figures build`. Lineages are
named by their evaluate run, whose manifest names every upstream artifact (nothing is looked up by date or "latest").
"""

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from kaful.artifacts.base import Identifier, NonEmptyStr, StrictModel

FigureId = Annotated[str, Field(pattern=r"^F[0-9]+$")]
Color = Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")]
UnitRule = Literal["median_abs_log_error", "median_abs_log_error_all_issues"]
COLOR_KEYS = ("truth", "particle_filter", "open_loop", "ukf", "target", "observation", "cone")


class Style(StrictModel):
    """Sizes in inches and points; colours by meaning, the same in every figure."""

    single_width_in: float = Field(gt=0.0)
    double_width_in: float = Field(gt=0.0)
    panel_height_in: float = Field(gt=0.0)
    dpi: int = Field(ge=50)
    font_size: float = Field(gt=0.0)
    font_family: NonEmptyStr
    svg_hashsalt: NonEmptyStr
    band_alpha: float = Field(gt=0.0, le=1.0)
    synthetic_label: NonEmptyStr
    colors: dict[str, Color]

    @model_validator(mode="after")
    def _check_colors(self) -> "Style":
        missing = sorted(set(COLOR_KEYS) - set(self.colors))
        if missing:
            raise ValueError(f"style.colors lacks {missing}")
        return self


class LineageSpec(StrictModel):
    machine: Identifier
    evaluate_run: NonEmptyStr
    label: NonEmptyStr = Field(description="How figures name the machine, e.g. 'CNC spindle'.")


class FigureBase(StrictModel):
    id: FigureId
    slug: Identifier = Field(description="File stem after the id, e.g. F3_<slug>.svg.")
    caption: NonEmptyStr
    width: Literal["single", "double"] = "double"


class UnitChoice(StrictModel):
    """The unit shown: chosen by `unit_rule` over every unit, unless `unit` names one (recorded either way)."""

    unit_rule: UnitRule = "median_abs_log_error"
    unit: Identifier | None = None


class RulFanSpec(FigureBase, UnitChoice):
    kind: Literal["rul_fan"]
    lineage: Identifier
    quantity: NonEmptyStr = Field(description="An event, or 'first_failure(<a>, <b>)' as calibration names it.")
    estimators: tuple[Identifier, ...] = Field(min_length=1, description="The first is the one the rule ranks.")


class TrackingSpec(FigureBase, UnitChoice):
    kind: Literal["tracking"]
    lineage: Identifier
    quantity: NonEmptyStr = Field(description="Ranks units for the rule, as in rul_fan.")
    channel: Identifier
    bindings: tuple[Identifier, ...] = Field(min_length=1)
    estimators: tuple[Identifier, ...] = Field(min_length=1)


class CalibrationSpec(FigureBase):
    kind: Literal["calibration"]
    lineages: tuple[Identifier, ...] = Field(min_length=1)
    estimators: tuple[Identifier, ...] = Field(min_length=1, description="The first is the judged (primary) one.")


FigureSpec = Annotated[RulFanSpec | TrackingSpec | CalibrationSpec, Field(discriminator="kind")]


class FiguresSpec(StrictModel):
    style: Style
    lineages: dict[Identifier, LineageSpec]
    labels: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict, description=(
        "Display names for quantities and bindings (as calibration names them); others are shown cleaned up."))
    figures: tuple[FigureSpec, ...]

    @model_validator(mode="after")
    def _check(self) -> "FiguresSpec":
        ids = [f.id for f in self.figures]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate figure ids: {sorted(ids)}")
        for f in self.figures:
            names = f.lineages if isinstance(f, CalibrationSpec) else (f.lineage,)
            unknown = sorted(set(names) - set(self.lineages))
            if unknown:
                raise ValueError(f"{f.id}: unknown lineages {unknown}")
        return self

    def stem(self, figure: FigureBase) -> str:
        return f"{figure.id}_{figure.slug}"


def load_spec(path: Path) -> FiguresSpec:
    with path.open("rb") as fh:
        return FiguresSpec.model_validate(tomllib.load(fh))
