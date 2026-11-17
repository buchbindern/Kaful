"""Factories for kaful artifacts: one minimal and one rich instance per kind.

Dict fields in the rich instances are built by iterating sets, so their insertion
order varies with PYTHONHASHSEED; the determinism test relies on that. Running
this module prints every rich artifact as canonical JSON.
"""

from collections.abc import Callable
from datetime import UTC, datetime

from kaful.artifacts import (
    DocumentIndex,
    EvaluationReport,
    Forecast,
    MachineModel,
    ParamSet,
    Scope,
    SensorContract,
    SyntheticDataset,
    SyntheticTruth,
    TwinDefinition,
    VerificationReport,
)
from kaful.artifacts.base import ArtifactBase, ArtifactRef, Evidence
from kaful.artifacts.understand_report import ContextSelection, ItemCheck, UnderstandReport
from kaful.artifacts.document_index import (
    Chunk,
    DocumentRecord,
    IndexFingerprint,
    PageLayer,
    PageSignals,
    PageText,
    chunk_id_for,
    index_hash_for,
    text_sha256,
)
from kaful.artifacts.distributions import Delta, Empirical, Lognormal, Normal, TruncatedNormal, Uniform
from kaful.artifacts.evaluation import (
    Cost,
    EvidenceCount,
    FileRef,
    Grounding,
    NotReached,
    PhaseCost,
    PhaseLineage,
    PromptUse,
    Prompts,
    RunScore,
)
from kaful.artifacts.forecast import (
    EventForecast,
    ForecastEntry,
    ForecastSet,
    GroupRecord,
    HealthSummary,
    Issue,
    OperatingProfile,
    ParamSummary,
    ProfileSegment,
    ScheduledEvent,
    SkippedEstimator,
)
from kaful.artifacts.machine_model import Component, FaultMode, FlowPath
from kaful.artifacts.manifest import CodeVersion, InputFile, PhaseRecord, RunManifest
from kaful.artifacts.param_set import LifeStatement, Parameter
from kaful.artifacts.scope import ScopeDecision
from kaful.artifacts.sensor_contract import SensorChannel
from kaful.artifacts.synthetic import EventTruth, SyntheticUnit, UnitFile
from kaful.artifacts.twin_definition import (
    Connection,
    EventDef,
    ExternalInput,
    GeneratedCode,
    ObservationBinding,
    TemplateBinding,
)
from kaful.artifacts.verification import CheckResult, Quarantine
from kaful.phases import PHASE_ORDER

RUN_ID = "20261102T120000Z-abc123"
MACHINE = "tiny_pump"
SHA = "a" * 64
SHA_B = "b" * 64
BASE = {"run_id": RUN_ID, "machine_id": MACHINE}
EVIDENCE = Evidence(doc_id="manual", page=3, page_label="1-3", quote="Replace the seal every 2000 h.")
MANUAL = {"source": "manual", "ref": "manual.pdf p.3", "confidence": 0.8, "evidence": (EVIDENCE,)}
JUDGED = {"source": "llm_judgement", "ref": "judgement", "confidence": 0.3}
PERIOD_EVIDENCE = Evidence(doc_id="manual", page=4, quote="All channels are logged once per second.")


def _from_set(keys: set[str], value: Callable[[str], float]) -> dict[str, float]:
    return {k: value(k) for k in keys}


FINGERPRINT = IndexFingerprint(classifier_version="v1", chunker_version="v1", extraction=None,
                               embedding_model="text-embedding-3-large", embedding_dimensions=3072)
SIGNALS = PageSignals(chars=40, lines=2, image_frac=0.0, drawings=0, curves=0, short_line_frac=0.0,
                      numeric_line_frac=0.0, garble_ratio=0.0, ruled_tables=0)


def document_index() -> DocumentIndex:
    text = "pages one and two"
    chunk = Chunk(chunk_id=chunk_id_for(SHA, 1, 2, "v1"), doc_id="manual", page_start=1, page_end=2,
                  label_start="1-1", label_end="1-2", text=text, text_sha256=text_sha256(text), chunker_version="v1")
    table = PageLayer(kind="table_md", text="| a | b |", producer="pymupdf find_tables")
    return DocumentIndex(
        **BASE,
        index_hash=index_hash_for((chunk,), FINGERPRINT),
        documents=(DocumentRecord(doc_id="manual", path="manual/pump.pdf", kind="manual", sha256=SHA, n_pages=3),),
        pages=tuple(PageText(doc_id="manual", page=p, label=f"1-{p}", label_source="printed", text=f"page {p}",
                             page_class="table_ruled" if p == 2 else "text", signals=SIGNALS,
                             layers=(table,) if p == 2 else ()) for p in (1, 2, 3)),
        chunks=(chunk,),
        fingerprint=FINGERPRINT,
    )


def sensor_contract() -> SensorContract:
    return SensorContract(
        **BASE,
        usage_clock="run_hours",
        channels=(
            SensorChannel(name="flow_rate", description="Outlet flow", role="observation",
                          dtype="float", unit="L/min", range=(0.0, 40.0), sampling="periodic", sample_period_s=1.0,
                          sample_period_evidence=PERIOD_EVIDENCE, **MANUAL),
            SensorChannel(name="maintenance", description="Maintenance action", role="event",
                          dtype="category", categories=("replace_seal", "service"), sampling="event_driven", **MANUAL),
            SensorChannel(name="run_hours", description="Operating hours", role="usage_clock",
                          dtype="float", unit="h", range=(0.0, 1e6), sampling="periodic", sample_period_s=60.0,
                          sample_period_evidence=PERIOD_EVIDENCE, **MANUAL),
            SensorChannel(name="suction_temp", description="Inlet temperature", role="operating_input",
                          dtype="float", unit="degC", range=(-10.0, 80.0), sampling="unknown", **JUDGED),
        ),
    )


def machine_model() -> MachineModel:
    return MachineModel(
        **BASE,
        n_extractions=3,
        components=(
            Component(id="impeller", name="Impeller", description="Rotating part", category="rotor", **MANUAL),
            Component(id="seal", name="Shaft seal", description="Mechanical seal", category="seal", **MANUAL),
        ),
        flow_paths=(FlowPath(id="shaft_drive", from_component="impeller", to_component="seal",
                             medium="mechanical", **JUDGED),),
        fault_modes=(FaultMode(id="seal_wear", component="seal", name="Seal face wear", mechanism="wear",
                               effect="leakage", driven_by=("suction_temp",),
                               observable_by=("flow_rate",), **MANUAL),),
    )


CONTEXT = ContextSelection(mode="whole", chunk_ids=("c1",), estimated_tokens=900, context_sha256="d" * 64)


def understand_report() -> UnderstandReport:
    return UnderstandReport(
        **BASE, context=CONTEXT, n_samples=3, prompts=("understand.components v1 claude-haiku-5-5",),
        contexts={"understand.components": (CONTEXT,), "understand.fault_modes": (CONTEXT, CONTEXT)},
        items=(ItemCheck(kind="channel", id="flow_rate", agreement=1.0, evidence=("verbatim",), source="manual",
                         confidence=1.0, parts_found=1),
               ItemCheck(kind="component", id="seal", agreement=2 / 3, evidence=("in_layer",), source="llm_judgement",
                         downgraded_from="manual", confidence=1 / 3, parts_found=2)),
        dropped=("fault mode x: unknown component 'y'",), notes=("n",),
    )


def scope() -> Scope:
    return Scope(**BASE, decisions=(
        ScopeDecision(component="impeller", decision="connection_only", reason="no wear data", decided_by="llm"),
        ScopeDecision(component="seal", decision="full_model", reason="documented wear part", decided_by="rule"),
    ))


def param_set() -> ParamSet:
    life = LifeStatement(id="seal_interval", component="seal", kind="service_interval", value=2000.0,
                         unit="h", bound="lower", **MANUAL)
    return ParamSet(
        **BASE,
        time_unit="h",
        life_statements=(life,),
        parameters=(
            Parameter(name="flow_noise", component="machine", role="noise", unit="L/min",
                      prior=Uniform(low=0.1, high=0.5), **JUDGED),
            Parameter(name="nominal_temp", component="machine", role="nominal_input", unit="degC",
                      prior=Normal(mean=25.0, sd=5.0), **JUDGED),
            Parameter(name="initial_wear", component="seal", role="initial_state", unit="mm",
                      prior=Delta(value=0.0), source="default", ref="new seal", confidence=1.0),
            Parameter(name="temp_exponent", component="seal", role="exponent", unit="1",
                      prior=TruncatedNormal(mean=1.0, sd=0.3, low=0.0, high=3.0), **JUDGED),
            Parameter(name="wear_limit", component="seal", role="threshold", unit="mm",
                      prior=Delta(value=1.5), **MANUAL),
            Parameter(name="wear_rate", component="seal", role="rate", unit="mm/h",
                      prior=Lognormal(median=7.5e-4, sigma_log=0.5),
                      posterior=Empirical(samples=(7e-4, 8e-4, 9e-4)),
                      source="calibrated", ref="seal_interval", confidence=0.6),
        ),
    )


def twin_definition() -> TwinDefinition:
    return TwinDefinition(
        **BASE,
        time_unit="h",
        components=("impeller", "seal"),
        bindings=(TemplateBinding(
            id="impeller_wear", component="impeller", fault_mode=None, template="accumulation", template_version="1",
            damage_state="impeller_wear_mm", damage_unit="mm", threshold_param="impeller.wear_limit",
            params={"k": "impeller.k"}, justification="erosion"),
            TemplateBinding(
            id="seal_wear", component="seal", fault_mode="seal_wear", template="archard",
            template_version="1", damage_state="seal_wear_mm", damage_unit="mm",
            threshold_param="seal.wear_limit",
            params={k: f"seal.{k}" for k in {"wear_rate", "temp_exponent"}},
            conditions={"temperature": "suction_temp"}, justification="sliding wear",
            escape_hatch=GeneratedCode(path="outputs/runs/x/build/seal.py", sha256=SHA,
                                       entrypoint="rate", reason="nonstandard"),
        )),
        connections=(Connection(id="impeller_to_seal", from_binding="impeller_wear", to_binding="seal_wear",
                                form="hardness_passthrough", form_version="1",
                                params={"residual_fraction": "impeller.residual_fraction"}, basis="declared",
                                justification="eroded impeller debris wears the seal", **MANUAL),),
        external_inputs=(ExternalInput(name="suction_temp", channel="suction_temp",
                                       nominal_param="machine.nominal_temp"),),
        events=(EventDef(name="seal_failure", binding="seal_wear"),),
        observations=(ObservationBinding(
            id="seal_wear__flow", binding="seal_wear", channel="flow", form="linear", form_version="1",
            params={k: f"seal.flow_{k}" for k in {"y_healthy", "y_limit", "noise_sd"}}, basis="declared",
            ref="machine.toml [[observation]] seal flow", justification="a worn seal leaks, so flow drops"),),
    )


def verification_report() -> VerificationReport:
    return VerificationReport(**BASE, quarantined=(
        Quarantine(component="impeller", checks=("health_bounds",), reason="health_bounds: max 1.2"),), results=(
        CheckResult(check="health_bounds", target="impeller", status="fail",
                    measured=_from_set({"min", "max"}, lambda k: 1.2 if k == "max" else 0.0)),
        CheckResult(check="health_bounds", target="seal", status="pass", limits={"max": 1.0}),
    ), recheck=(CheckResult(check="health_bounds", target="seal", status="pass", limits={"max": 1.0}),))


def operating_profile() -> OperatingProfile:
    return OperatingProfile(
        name="nominal",
        segments=(ProfileSegment(start=0.0, end=100.0, values={"suction_temp": 25.0}),
                  ProfileSegment(start=100.0, end=500.0, values={"suction_temp": 40.0})),
        events=(ScheduledEvent(t=250.0, channel="maintenance", value="service"),),
    )


def _unit_file(name: str) -> UnitFile:
    return UnitFile(path=f"outputs/runs/x/synthesize/{name}", sha256=SHA)


def synthetic_dataset() -> SyntheticDataset:
    return SyntheticDataset(
        **BASE, seed=7, time_unit="h", profile=operating_profile(), horizon=500.0, sample_interval=5.0,
        timestamp_origin=datetime(2026, 1, 1, tzinfo=UTC), channels=("flow",), excluded_components=("impeller",),
        flags=("sampling grid chosen by synthesize.samples_per_shortest_life",),
        units=(SyntheticUnit(unit_id="unit_01", spawn_key=0, observations=_unit_file("observations/unit_01.csv"),
                             truth=_unit_file("truth/unit_01.json"),
                             truth_states=_unit_file("truth/unit_01.states.csv"), t_end=200.0, n_rows=41),))


def synthetic_truth() -> SyntheticTruth:
    return SyntheticTruth(
        **BASE, unit_id="unit_01", seed=7, spawn_key=0, horizon=500.0, t_end=200.0, profile=operating_profile(),
        true_params=dict(sorted(_from_set({"seal.wear_rate", "seal.temp_exponent"}, lambda k: float(len(k))).items())),
        derived_params={"seal.life_nominal": 2000.0},
        events=(EventTruth(event="seal_failure", first_time=180.0, limit_times=(180.0, 420.0), in_window=True),))


def forecast() -> Forecast:
    q = {"p05": 0.001, "p50": 0.002, "p95": 0.004}
    return Forecast(
        **BASE, source="unit_01", estimator="particle_filter", seed=7, spawn_key=1, horizon=3000.0,
        process_noise_cv=0.1, n_particles=4, profile=operating_profile(), ignored_profile_channels=("speed",),
        groups=(GroupRecord(bindings=("seal",), estimator="particle_filter", updates=3, resamples=1,
                            degenerate_updates=0, min_ess=2.5, channels_used={"leak_flow": 3},
                            reports={"seal_alarm": 1}),),
        issues=(Issue(t_issue=100.0, events=(EventForecast(
                    event="seal_failure", binding="seal", n_samples=4, n_hits=3, n_censored=1, n_already_triggered=0,
                    p_event_before_horizon=0.75, percentiles={"p05": 1710.0, "p50": 1900.0, "p95": None}),),
                      params=(ParamSummary(param_id="seal.k", role="rate", prior=q, posterior=q),),
                      health=(HealthSummary(binding="seal", quantiles={"p05": 0.8, "p50": 0.9, "p95": 0.95}),),
                      samples="outputs/runs/x/estimate/samples/unit_01.particle_filter.0.csv", samples_sha256=SHA),),
        trace="outputs/runs/x/estimate/traces/unit_01.particle_filter.csv", trace_sha256=SHA,
        flags=("process noise cv rho = 0.1 is an estimator assumption",))


def forecast_set() -> ForecastSet:
    return ForecastSet(
        **BASE, time_unit="h", settings={"n_particles": 4, "process_noise_cv": 0.1, "methods": ["particle_filter"]},
        forecasts=(ForecastEntry(source="unit_01", estimator="particle_filter",
                                 path="outputs/runs/x/estimate/unit_01.particle_filter.json", sha256=SHA),),
        skipped=(SkippedEstimator(source="unit_01", estimator="ukf", reason="no group has a linear channel"),),
        flags=("rho is an estimator assumption",), disclaimer="synthetic")


def evaluation_report() -> EvaluationReport:
    ref = ArtifactRef(kind="param_set", run_id=RUN_ID, path="outputs/runs/x/parameterize/param_set.json", sha256=SHA)
    return EvaluationReport(
        **BASE,
        evaluated_run=RUN_ID,
        complete=False,
        time_unit="h",
        lineage=tuple(PhaseLineage(phase=p, status="reused" if p == "parameterize" else "skipped",
                                   produced_by=RUN_ID if p == "parameterize" else None,
                                   artifacts=(ref,) if p == "parameterize" else ()) for p in PHASE_ORDER),
        grounding=Grounding(
            evidence=(EvidenceCount(kind="param_set", quotes=_from_set(
                {"unverified", "verbatim", "near", "in_layer", "unsupported"}, lambda k: 1.0 if k == "near" else 0.0),
                provenanced=3, downgraded={"manual -> default": 1}),),
            params_by_source={"manual": 2, "calibrated": 1}),
        prompts=Prompts(uses=(PromptUse(prompt_id="understand.components", version=5, phase="understand",
                                        models=("claude-haiku-5-5",), calls=3, lock="locked", locked_version=5,
                                        run_scores=(RunScore(result="evals/results/x.json", sample=0,
                                                             metrics=_from_set({"parses", "recall"},
                                                                               lambda k: 1.0)),)),)),
        cost=Cost(by_phase=(PhaseCost(phase="ingest", produced_by=RUN_ID, calls=2, billed_usd=0.2, unbilled_usd=0.0),),
                  ledgers=(FileRef(path="outputs/runs/x/ledger.jsonl", sha256=SHA),), development_billed_usd=1.5,
                  development_runs=4, runs_total=6),
        not_reached=(NotReached(section="twin_quality", reason="verify was never reached"),),
        inputs=(ArtifactRef(kind="forecast", run_id=RUN_ID, path="outputs/runs/x/estimate/forecast.json",
                            sha256=SHA_B),),
    )


def run_manifest() -> RunManifest:
    return RunManifest(
        run_id=RUN_ID, machine_id=MACHINE, created_at=datetime(2026, 11, 2, 12, tzinfo=UTC),
        code=CodeVersion(git_commit="0" * 40, git_dirty=False, kaful_version="0.1.0"),
        argv=("run", MACHINE), budget_usd=10.0,
        config={"sandbox": _from_set({"timeout_s", "cpu_limit_s"}, lambda k: 1.0)},
        config_hash=SHA, inputs=(InputFile(path="machine.toml", sha256=SHA),),
        python="3.11.9", phases=tuple(PhaseRecord(phase=p) for p in PHASE_ORDER),
    )


RICH: dict[str, Callable[[], ArtifactBase]] = {
    "document_index": document_index,
    "machine_model": machine_model,
    "sensor_contract": sensor_contract,
    "understand_report": understand_report,
    "scope": scope,
    "param_set": param_set,
    "twin_definition": twin_definition,
    "verification_report": verification_report,
    "synthetic_dataset": synthetic_dataset,
    "synthetic_truth": synthetic_truth,
    "forecast": forecast,
    "forecast_set": forecast_set,
    "evaluation_report": evaluation_report,
}

MINIMAL: dict[str, Callable[[], ArtifactBase]] = {
    "document_index": lambda: DocumentIndex(**BASE, index_hash=index_hash_for((), FINGERPRINT), documents=(),
                                            pages=(), chunks=(), fingerprint=FINGERPRINT),
    "machine_model": lambda: MachineModel(**BASE, components=(), n_extractions=1),
    "sensor_contract": lambda: SensorContract(**BASE, channels=()),
    "understand_report": lambda: UnderstandReport(**BASE, context=CONTEXT, n_samples=1, prompts=(), items=()),
    "scope": lambda: Scope(**BASE, decisions=()),
    "param_set": lambda: ParamSet(**BASE, time_unit="h", parameters=()),
    "twin_definition": lambda: TwinDefinition(**BASE, time_unit="h", components=(), bindings=()),
    "verification_report": lambda: VerificationReport(**BASE, results=()),
    "synthetic_dataset": lambda: SyntheticDataset(**BASE, seed=0, time_unit="h", profile=operating_profile(),
                                                  horizon=1.0, sample_interval=1.0, channels=(), units=()),
    "synthetic_truth": lambda: SyntheticTruth(**BASE, unit_id="unit_00", seed=0, spawn_key=0, true_params={},
                                              horizon=1.0, t_end=1.0, events=(), profile=operating_profile()),
    "forecast": lambda: Forecast(**BASE, source="s", estimator="open_loop", seed=0, spawn_key=0, horizon=1.0,
                                 process_noise_cv=0.1, n_particles=1, profile=operating_profile(), groups=(),
                                 issues=(Issue(t_issue=0.0, events=(), samples="s.csv", samples_sha256=SHA),),
                                 trace="t.csv", trace_sha256=SHA),
    "forecast_set": lambda: ForecastSet(**BASE, time_unit="h", settings={}, forecasts=()),
    "evaluation_report": lambda: EvaluationReport(
        **BASE, evaluated_run=RUN_ID, complete=False, lineage=(), prompts=Prompts(uses=()),
        cost=Cost(by_phase=(), ledgers=(), development_billed_usd=0.0, development_runs=0, runs_total=0),
        not_reached=(NotReached(section="grounding", reason="understand failed"),
                     NotReached(section="twin_quality", reason="verify was never reached"))),
}


if __name__ == "__main__":
    import sys

    from kaful.runs import dump_json

    for kind in sorted(RICH):
        sys.stdout.write(dump_json(RICH[kind]()))
    sys.stdout.write(dump_json(run_manifest()))
