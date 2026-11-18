"""What every figure module returns to the build: the figure, its plotted numbers, and how its unit was chosen.

Also the shared context (spec style, settings, the lineage opener) and display names for quantities.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from matplotlib.figure import Figure

from kaful.config import KafulSettings
from kaful.figures.sources import Inputs, LoadedLineage, open_lineage
from kaful.figures.spec import FiguresSpec, Style
from kaful.figures.style import Cell
from kaful.figures.forecasts import FIRST_FAILURE

FAILURE_SUFFIX = "_failure"


@dataclass(frozen=True)
class Drawn:
    figure: Figure
    columns: tuple[str, ...]
    rows: list[dict[str, Cell]]
    selection: str | None = None
    notes: tuple[str, ...] = ()


@dataclass
class Context:
    spec: FiguresSpec
    settings: KafulSettings
    inputs: Inputs = field(default_factory=Inputs)

    @property
    def style(self) -> Style:
        return self.spec.style

    def label(self, name: str) -> str:
        """A quantity's or binding's display name: the spec's [labels], else cleaned up."""
        return self.spec.labels.get(name) or pretty_quantity(name)

    def lineage(self, name: str) -> LoadedLineage:
        return open_lineage(name, self.spec.lineages[name], self.settings, self.inputs)


Draw = Callable[..., Drawn]


def pretty_quantity(quantity: str) -> str:
    m = FIRST_FAILURE.match(quantity)
    if m:
        return "first failure of " + " / ".join(b.strip().replace("_", " ") for b in m.group("bindings").split(","))
    return quantity.removesuffix(FAILURE_SUFFIX).replace("_", " ")


def pretty_estimator(estimator: str) -> str:
    return {"particle_filter": "particle filter", "open_loop": "open-loop", "ukf": "UKF"}.get(estimator, estimator)
