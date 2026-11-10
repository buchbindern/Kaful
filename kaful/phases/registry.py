"""Implemented phases, by name. Kept apart from kaful.phases to avoid import cycles
(phase modules import kaful.runs, which imports kaful.phases)."""

from collections.abc import Mapping

from kaful.phases import Phase, PhaseName
from kaful.phases.p1_ingest.phase import IngestPhase
from kaful.phases.p2_understand.phase import UnderstandPhase
from kaful.phases.p3_scope.phase import ScopePhase
from kaful.phases.p4_parameterize.phase import ParameterizePhase

PHASE_IMPLEMENTATIONS: Mapping[PhaseName, Phase] = {
    "ingest": IngestPhase(), "understand": UnderstandPhase(), "scope": ScopePhase(),
    "parameterize": ParameterizePhase()}
