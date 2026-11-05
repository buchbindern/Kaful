"""Implemented phases, by name. Kept apart from kaful.phases to avoid import cycles
(phase modules import kaful.runs, which imports kaful.phases)."""

from collections.abc import Mapping

from kaful.phases import Phase, PhaseName
from kaful.phases.p1_ingest.phase import IngestPhase
from kaful.phases.understand.phase import UnderstandPhase

PHASE_IMPLEMENTATIONS: Mapping[PhaseName, Phase] = {"ingest": IngestPhase(), "understand": UnderstandPhase()}
