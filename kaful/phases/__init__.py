"""Pipeline phases, their order, and phase selection for `--from` / `--only`.

Each subpackage holds one phase. A phase has one job, a typed input and a typed
output; see CLAUDE.md. No phase is implemented yet, so `PHASE_IMPLEMENTATIONS` is
empty and the CLI records every selected phase as `not_implemented`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, get_args

if TYPE_CHECKING:
    from kaful.artifacts.manifest import PhaseRecord
    from kaful.runs import RunContext

PhaseName = Literal[
    "ingest",
    "understand",
    "scope",
    "parameterize",
    "build",
    "verify",
    "synthesize",
    "estimate",
    "evaluate",
]
PHASE_ORDER: tuple[PhaseName, ...] = get_args(PhaseName)
# Optional phases are never required upstream; `synthesize` is for the no-data case only.
OPTIONAL_PHASES: frozenset[PhaseName] = frozenset({"synthesize"})


class Phase(Protocol):
    """One pipeline phase."""

    name: PhaseName

    def run(self, ctx: RunContext) -> PhaseRecord:
        """Read upstream artifacts from `ctx`, save this phase's artifact, return its record."""
        ...


PHASE_IMPLEMENTATIONS: Mapping[PhaseName, Phase] = {}


class PhaseSelectionError(ValueError):
    """`--from` / `--only` do not describe a valid phase selection."""


def select_phases(from_phase: PhaseName | None, only: Sequence[PhaseName]) -> tuple[PhaseName, ...]:
    """Phases to run, in pipeline order.

    No options selects every phase. `--from P` selects P and everything after it.
    `--only` selects the listed phases, which must be contiguous in pipeline order
    (optional phases may be left out of the middle).
    """
    if from_phase is not None and only:
        raise PhaseSelectionError("use either --from or --only, not both")
    if from_phase is not None:
        return PHASE_ORDER[PHASE_ORDER.index(from_phase) :]
    if not only:
        return PHASE_ORDER
    wanted = set(only)
    selected = tuple(p for p in PHASE_ORDER if p in wanted)
    first, last = PHASE_ORDER.index(selected[0]), PHASE_ORDER.index(selected[-1])
    gaps = [p for p in PHASE_ORDER[first:last] if p not in wanted and p not in OPTIONAL_PHASES]
    if gaps:
        raise PhaseSelectionError(f"--only phases must be contiguous; missing {gaps}")
    return selected


def upstream_phases(selected: Sequence[PhaseName]) -> tuple[PhaseName, ...]:
    """Required phases before the first selected one; their artifacts come from a base run."""
    first = PHASE_ORDER.index(selected[0])
    return tuple(p for p in PHASE_ORDER[:first] if p not in OPTIONAL_PHASES)
