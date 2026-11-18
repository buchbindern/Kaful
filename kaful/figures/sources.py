"""Opening a lineage for figures, and recording every file a figure reads (path relative to the repo, sha256).

Used by every figure's extract step. A lineage is named by its evaluate run: its manifest names each phase's
artifacts, and its evaluation report is read from the same manifest.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from kaful.artifacts.base import ArtifactBase, ArtifactRef
from kaful.artifacts.evaluation import EvaluationReport
from kaful.artifacts.forecast import Forecast, ForecastSet
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.synthetic import SyntheticDataset
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.config import REPO_ROOT, KafulSettings
from kaful.figures.spec import LineageSpec
from kaful.paths import resolve_in
from kaful.phases.p9_evaluate.a_resolve_lineage import Lineage, resolve
from kaful.runs import load_artifact, read_manifest, runs_dir, sha256_file

M = TypeVar("M", bound=ArtifactBase)


@dataclass
class Inputs:
    """Files read for one figure: path relative to `root` (the repo) -> sha256, sorted on output. A file outside
    the root is an error: the manifest never holds an absolute path."""

    root: Path = REPO_ROOT
    files: dict[str, str] = field(default_factory=dict)

    def add(self, path: Path) -> Path:
        rel = path.resolve().relative_to(self.root.resolve()).as_posix()
        if rel not in self.files:
            self.files[rel] = sha256_file(path)
        return path

    def sorted(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self.files.items()))


@dataclass(frozen=True)
class LoadedLineage:
    name: str
    spec: LineageSpec
    machine_dir: Path
    lineage: Lineage
    report: EvaluationReport
    inputs: Inputs

    def load(self, ref: ArtifactRef, model: type[M]) -> M:
        self.inputs.add(resolve_in(self.machine_dir, ref.path))
        return load_artifact(self.machine_dir, ref, model)  # checks the ref's sha256

    def file(self, rel: str) -> Path:
        """A sidecar file named relative to the machine directory, recorded."""
        return self.inputs.add(resolve_in(self.machine_dir, rel))

    def _ref(self, *candidates: tuple[str, str]) -> ArtifactRef:
        for phase, kind in candidates:
            ref = self.lineage.ref(phase, kind)  # type: ignore[arg-type]
            if ref is not None:
                return ref
        raise ValueError(f"{self.name}: lineage {self.lineage.run_id} has no {candidates[0][1]}")

    def twin(self) -> TwinDefinition:
        return self.load(self._ref(("verify", "twin_definition"), ("build", "twin_definition")), TwinDefinition)

    def forecast_set(self) -> ForecastSet:
        return self.load(self._ref(("estimate", "forecast_set")), ForecastSet)

    def dataset(self) -> SyntheticDataset:
        return self.load(self._ref(("synthesize", "synthetic_dataset")), SyntheticDataset)

    def contract(self) -> SensorContract:
        return self.load(self._ref(("understand", "sensor_contract")), SensorContract)

    def forecasts(self, estimators: tuple[str, ...]) -> list[Forecast]:
        fs = self.forecast_set()
        out = []
        for e in sorted(fs.forecasts, key=lambda e: (e.source, e.estimator)):
            if e.estimator in estimators:
                ref = ArtifactRef(kind="forecast", run_id=fs.run_id, path=e.path, sha256=e.sha256)
                out.append(self.load(ref, Forecast))
        return out

    @property
    def time_unit(self) -> str:
        if self.report.time_unit is None:
            raise ValueError(f"{self.name}: the evaluation report has no time unit")
        return self.report.time_unit


def open_lineage(name: str, spec: LineageSpec, settings: KafulSettings, inputs: Inputs) -> LoadedLineage:
    machine_dir = settings.machine_dir(spec.machine)
    run_dir = runs_dir(machine_dir) / spec.evaluate_run
    manifest = read_manifest(inputs.add(run_dir / "manifest.json").parent)
    rec = next((p for p in manifest.phases if p.phase == "evaluate"), None)
    refs = [a for a in rec.artifacts if a.kind == "evaluation_report"] if rec is not None else []
    if rec is None or rec.status not in ("passed", "reused") or len(refs) != 1:
        raise ValueError(f"{name}: run {spec.evaluate_run} has no passed evaluation report")
    inputs.add(resolve_in(machine_dir, refs[0].path))
    report = load_artifact(machine_dir, refs[0], EvaluationReport)
    return LoadedLineage(name=name, spec=spec, machine_dir=machine_dir, lineage=resolve(manifest), report=report,
                         inputs=inputs)
