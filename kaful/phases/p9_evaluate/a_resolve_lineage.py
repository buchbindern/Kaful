"""Step a: the lineage of the evaluated run, i.e. per phase its status and the run that produced its artifacts.

A reused phase record carries its producing run's artifacts, so the evaluated run's manifest alone names every
artifact; nothing is looked up by date or by "latest".
"""

from dataclasses import dataclass

from kaful.artifacts.base import ArtifactRef
from kaful.artifacts.evaluation import PhaseLineage
from kaful.artifacts.manifest import RunManifest
from kaful.phases import PHASE_ORDER, PhaseName

REACHED = ("passed", "reused")


@dataclass(frozen=True)
class Lineage:
    run_id: str
    machine_id: str
    phases: tuple[PhaseLineage, ...]

    def get(self, phase: PhaseName) -> PhaseLineage:
        return next(p for p in self.phases if p.phase == phase)

    def reached(self, phase: PhaseName) -> bool:
        return self.get(phase).status in REACHED

    def refs(self, phase: PhaseName, kind: str) -> tuple[ArtifactRef, ...]:
        return tuple(r for r in self.get(phase).artifacts if r.kind == kind)

    def ref(self, phase: PhaseName, kind: str) -> ArtifactRef | None:
        found = self.refs(phase, kind)
        return found[0] if len(found) == 1 else None

    def why_not(self, phase: PhaseName) -> str:
        """Why a phase gives no artifact: the first required phase up to it that did not pass."""
        for p in PHASE_ORDER[: PHASE_ORDER.index(phase) + 1]:
            rec = self.get(p)
            if rec.status not in REACHED and p != "synthesize":
                fail = f": {rec.failures[0]}" if rec.failures else ""
                return f"{p} {rec.status} in run {self.run_id}{fail}"
        return f"{phase} gave no artifact"


def resolve(manifest: RunManifest) -> Lineage:
    """Every phase before evaluate, as the manifest records it."""
    out = []
    for rec in manifest.phases:
        if rec.phase == "evaluate":
            continue
        producers = sorted({a.run_id for a in rec.artifacts})
        if len(producers) > 1:
            raise ValueError(f"{rec.phase}: artifacts from several runs {producers}; a phase has one producing run")
        out.append(PhaseLineage(phase=rec.phase, status=rec.status, produced_by=producers[0] if producers else None,
                                artifacts=rec.artifacts, failures=rec.failures))
    return Lineage(run_id=manifest.run_id, machine_id=manifest.machine_id, phases=tuple(out))
