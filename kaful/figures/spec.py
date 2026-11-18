"""The figure spec (`docs/figures/figures.toml`): the lineages, the figures and the style.

It is committed config and, with the saved artifacts it names, the only input to `kaful figures build`. Lineages are
named by their evaluate run, whose manifest names every upstream artifact (nothing is looked up by date or "latest").
"""

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from kaful.artifacts.base import Identifier, NonEmptyStr, StrictModel

FigureId = Annotated[str, Field(pattern=r"^F[0-9]+[a-z]?$")]  # F4t: a variant of F4
Color = Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")]
UnitRule = Literal["median_abs_log_error", "median_abs_log_error_all_issues"]
COLOR_KEYS = ("truth", "particle_filter", "open_loop", "ukf", "target", "observation", "cone", "legacy", "cost_high",
              "with_maintenance", "without_maintenance")


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
    source_colors: dict[str, Color] = Field(default_factory=dict, description="Provenance source -> colour (F6).")

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


class FailureTimeSpec(FigureBase, UnitChoice):
    kind: Literal["failure_time"]
    lineage: Identifier
    quantity: NonEmptyStr = Field(description="An event, or 'first_failure(<a>, <b>)' as calibration names it.")
    estimators: tuple[Identifier, ...] = Field(min_length=1, description="The first is the one the rule ranks.")


class TrackingSpec(FigureBase, UnitChoice):
    kind: Literal["tracking"]
    lineage: Identifier
    quantity: NonEmptyStr = Field(description="Ranks units for the rule, as in failure_time.")
    channel: Identifier
    bindings: tuple[Identifier, ...] = Field(min_length=1)
    estimators: tuple[Identifier, ...] = Field(min_length=1)


class CalibrationSpec(FigureBase):
    kind: Literal["calibration"]
    lineages: tuple[Identifier, ...] = Field(min_length=1)
    estimators: tuple[Identifier, ...] = Field(min_length=1, description="The first is the judged (primary) one.")
    observed_only: bool = Field(default=False, description="Leave out the circular section (a talk variant).")
    x_min: float | None = Field(default=None, ge=0.0, lt=1.0, description=(
        "Where the coverage axis starts; None: the lowest CI bound, rounded down. A CI below it is drawn to the "
        "edge with an arrowhead; the CSV keeps its value."))


class CostSpec(FigureBase):
    kind: Literal["cost"]
    lineages: tuple[Identifier, ...] = Field(min_length=1)


class ProvenanceSpec(FigureBase):
    kind: Literal["provenance"]
    lineages: tuple[Identifier, ...] = Field(min_length=1)
    sources: tuple[Identifier, ...] = Field(min_length=1, description="Provenance sources, in stacking order.")


class SharpnessSpec(FigureBase):
    kind: Literal["sharpness"]
    lineages: tuple[Identifier, ...] = Field(min_length=1)
    estimators: tuple[Identifier, ...] = Field(min_length=1)
    widths_panel: bool = Field(default=True, description="False: the life-contraction panel only (a talk variant).")


class MaintenancePart(StrictModel):
    lineage: Identifier
    binding: Identifier
    bands: tuple[Literal["nominal", "documented_maintenance"], ...] = Field(
        default=("nominal", "documented_maintenance"), description="Profiles drawn with their 5-95% band.")


class MaintenanceSpec(FigureBase):
    kind: Literal["maintenance"]
    parts: tuple[MaintenancePart, ...] = Field(min_length=1)
    draws: int = Field(ge=1, description="Prior draws per profile for the 5-95% band (synthesize's draw_unit).")
    seed: int = Field(description="SeedSequence seed for the draws.")
    steps_per_shortest: int = Field(ge=10, description="dt = min(median life, maintenance intervals) / this.")
    lives_shown: float = Field(gt=0.0, description="The x axis spans this many median lives (capped at horizon).")


class WhatIfSpec(FigureBase):
    kind: Literal["whatif"]
    lineage: Identifier
    file: NonEmptyStr = Field(description="The what-if report, relative to the machine directory.")


class FailureGridSpec(FigureBase, UnitChoice):
    kind: Literal["failure_grid"]
    lineage: Identifier
    quantity: NonEmptyStr
    estimators: tuple[Identifier, ...] = Field(min_length=1)
    columns: int = Field(ge=1)


class PipelineSpec(FigureBase):
    kind: Literal["pipeline"]
    lineage: Identifier


FigureSpec = Annotated[FailureTimeSpec | TrackingSpec | CalibrationSpec | CostSpec | ProvenanceSpec | SharpnessSpec
                       | MaintenanceSpec | WhatIfSpec | FailureGridSpec | PipelineSpec, Field(discriminator="kind")]


class ReportSpec(StrictModel):
    """The demo report (kaful/figures/report.py), written after the figures."""

    file: NonEmptyStr = Field(pattern=r"^[a-z0-9_]+\.html$")
    title: NonEmptyStr
    subtitle: NonEmptyStr
    lineages: tuple[Identifier, ...] = Field(min_length=1, description="Summary, provenance and verification rows.")
    scope_lineage: Identifier
    selector_figure: FigureId = Field(description="A failure_time figure; the report offers it for every unit.")
    quote_max_chars: int = Field(ge=40, le=300, description=(
        "Manual quotes are cut to one sentence of at most this many characters: the page is committed."))


class FiguresSpec(StrictModel):
    style: Style
    report: ReportSpec | None = None
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
            names = lineage_names(f)
            unknown = sorted(set(names) - set(self.lineages))
            if unknown:
                raise ValueError(f"{f.id}: unknown lineages {unknown}")
        if self.report is not None:
            unknown = sorted({*self.report.lineages, self.report.scope_lineage} - set(self.lineages))
            if unknown or self.report.selector_figure not in ids:
                raise ValueError(f"report: unknown lineages {unknown} or figure {self.report.selector_figure}")
        return self

    def stem(self, figure: FigureBase) -> str:
        return f"{figure.id}_{figure.slug}"


def lineage_names(f: "FigureSpec") -> tuple[str, ...]:
    if isinstance(f, MaintenanceSpec):
        return tuple(p.lineage for p in f.parts)
    if isinstance(f, (CalibrationSpec, CostSpec, ProvenanceSpec, SharpnessSpec)):
        return f.lineages
    return (f.lineage,)


def load_spec(path: Path) -> FiguresSpec:
    with path.open("rb") as fh:
        return FiguresSpec.model_validate(tomllib.load(fh))
