"""The synthesize phase (optional; the no-data case): the verified twin + ParamSet + SensorContract -> a
SyntheticDataset and one synthetic_truth per unit (M12). No LLM call.

A pipeline sanity check, not validation: the data comes from the twin's own ParamSet, so scoring the twin on it is
partly circular. Every dataset and truth file says so.

Steps, in the order `synthesize` runs them for each unit:
  a_draw_units      the unit's true parameters from the priors (only those the twin reads, sorted; life_nominal
                    derived from k), and its observation window, from its own seed streams
  b_simulate_truth  the verified twin under the stored profile (documented maintenance), with the true values and
                    no noise, on verify's step
  c_check_units     verify's dynamic checks (kaful.dynamics) on the unit's truth; a failure fails the phase
  d_observe         the observation models (kaful.observation) on the sampling grid and at threshold events, up to
                    the window end; the true events and states
  e_write           observations CSV (SensorContract channels only), true states CSV, the truth artifact
Units run in index order; everything is sorted, so the files do not depend on hash order (rule 14).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kaful.artifacts.base import ArtifactBase, ArtifactRef
from kaful.artifacts.manifest import PhaseRecord
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.synthetic import SyntheticDataset, SyntheticTruth, SyntheticUnit
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.backends import load_backend
from kaful.config import KafulSettings
from kaful.phases import PhaseName
from kaful.phases.p7_synthesize.a_draw_units import draw_unit
from kaful.phases.p7_synthesize.b_simulate_truth import shortest_median_life, simulate_unit
from kaful.phases.p7_synthesize.c_check_units import check_unit
from kaful.phases.p7_synthesize.d_observe import observe
from kaful.phases.p7_synthesize.e_write import SaveTruth, write_unit
from kaful.runs import RunContext, load_artifact, now_utc

log = logging.getLogger("kaful.synthesize")

FLAG_GRID = "linear channels sampled every shortest median life / synthesize.samples_per_shortest_life (no documented period)"
FLAG_NO_CLOCK = "no usage clock: timestamps = synthesize.timestamp_origin + t, assuming 100% duty"
FLAG_NOISELESS = "truth is noiseless given the true parameters (no process noise; M13's estimators add their own)"
FLAG_NOMINAL = "operating conditions held at nominal (undocumented inputs, f = 1): units differ by parameters only"


@dataclass(frozen=True)
class Synthesized:
    dataset: SyntheticDataset | None
    failures: tuple[str, ...]


def _flags(twin: TwinDefinition, contract: SensorContract) -> tuple[str, ...]:
    flags = {FLAG_GRID, FLAG_NOISELESS}
    if contract.usage_clock is None:
        flags.add(FLAG_NO_CLOCK)
    if any(i.provenance == "undocumented" for i in twin.external_inputs):
        flags.add(FLAG_NOMINAL)
    return tuple(sorted(flags))


def synthesize(base: dict[str, Any], twin: TwinDefinition, ps: ParamSet, contract: SensorContract,
               settings: KafulSettings, machine_dir: Path, phase_dir: Path, save_truth: SaveTruth) -> Synthesized:
    s = settings.synthesize
    profile = next((p for p in twin.profiles if p.name == s.profile), None)
    if profile is None:
        return Synthesized(None, (f"the twin has no stored profile {s.profile!r}",))
    draws = [draw_unit(ps, twin, s, i) for i in range(s.n_units)]
    backend = load_backend(settings.build.backend)
    built = backend.build(twin, ps, contract)
    shortest, horizon = shortest_median_life(twin, ps), profile.segments[-1].end
    dt, interval = shortest / settings.verify.steps_per_shortest_life, shortest / s.samples_per_shortest_life
    units: list[SyntheticUnit] = []
    failures: list[str] = []
    for i, draw in enumerate(draws):
        result = simulate_unit(backend, built, profile, draw.true_params, dt, s.seed)
        failures += check_unit(draw.unit_id, twin, result, profile, draw.true_params, settings.verify.damage_tolerance)
        t_end = draw.window_fraction * horizon
        observed = observe(twin, result, draw.true_params, contract, twin.time_unit, interval, t_end,
                           s.timestamp_origin, draw.noise)
        truth = SyntheticTruth(**base, unit_id=draw.unit_id, seed=s.seed, spawn_key=i, true_params=draw.true_params,
                               derived_params=draw.derived_params, horizon=horizon, t_end=t_end,
                               events=observed.events, profile=profile)
        units.append(write_unit(machine_dir, phase_dir, contract, observed, truth, save_truth))
        log.info("unit synthesized", extra={"fields": {"unit": draw.unit_id, "rows": units[-1].n_rows,
                                                       "in_window": sum(e.in_window for e in observed.events)}})
    channels = sorted({o.channel for o in twin.observations} | ({contract.usage_clock} if contract.usage_clock else set()))
    dataset = SyntheticDataset(
        **base, seed=s.seed, time_unit=twin.time_unit, profile=profile, horizon=horizon, sample_interval=interval,
        timestamp_origin=None if contract.usage_clock else s.timestamp_origin, channels=tuple(channels),
        excluded_components=tuple(sorted({t.component for t in ps.templates} - set(twin.components))),
        flags=_flags(twin, contract), units=tuple(units))
    return Synthesized(dataset, tuple(failures))


class SynthesizePhase:
    name: PhaseName = "synthesize"

    def run(self, ctx: RunContext) -> PhaseRecord:
        started = now_utc()
        try:
            verified = {r.kind: r for r in ctx.manifest.phase("verify").artifacts}
            params = {r.kind: r for r in ctx.manifest.phase("parameterize").artifacts}
            understand = {r.kind: r for r in ctx.manifest.phase("understand").artifacts}
            twin = self._load(ctx, verified, "twin_definition", TwinDefinition)
            ps = self._load(ctx, params, "param_set", ParamSet)
            contract = self._load(ctx, understand, "sensor_contract", SensorContract)
        except FileNotFoundError as exc:
            log.error("synthesize cannot start", extra={"fields": {"error": str(exc)}})
            return PhaseRecord(phase=self.name, status="failed", started_at=started, finished_at=now_utc(),
                               failures=(str(exc),))
        inputs = tuple(sorted((verified["twin_definition"], params["param_set"], understand["sensor_contract"]),
                              key=lambda r: (r.kind, r.path)))
        base = {"run_id": ctx.manifest.run_id, "machine_id": ctx.machine.machine_id, "index_hash": twin.index_hash,
                "inputs": inputs}

        def save_truth(truth: SyntheticTruth, name: str) -> ArtifactRef:
            return ctx.save_artifact(self.name, truth, name)

        done = synthesize(base, twin, ps, contract, ctx.settings, ctx.machine_dir, ctx.run_dir / self.name, save_truth)
        if done.dataset is None or done.failures:
            return PhaseRecord(phase=self.name, status="failed", started_at=started, finished_at=now_utc(),
                               failures=done.failures)
        ref = ctx.save_artifact(self.name, done.dataset)
        return PhaseRecord(phase=self.name, status="passed", started_at=started, finished_at=now_utc(),
                           artifacts=(ref,), notes=(done.dataset.disclaimer,
                                                    f"{len(done.dataset.units)} units, seed {done.dataset.seed}"))

    @staticmethod
    def _load(ctx: RunContext, refs: dict[str, ArtifactRef], kind: str, model: type[ArtifactBase]) -> Any:
        if kind not in refs:
            raise FileNotFoundError(f"run {ctx.manifest.run_id} has no {kind}")
        return load_artifact(ctx.machine_dir, refs[kind], model)
