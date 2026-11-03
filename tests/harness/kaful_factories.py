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
    TwinDefinition,
    VerificationReport,
)
from kaful.artifacts.base import ArtifactBase, ArtifactRef, Evidence
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
from kaful.artifacts.evaluation import CalibrationMetrics, PromptScore, SensitivityEntry
from kaful.artifacts.forecast import EventForecast, OperatingProfile, ProfileSegment, ScheduledEvent
from kaful.artifacts.machine_model import Component, FaultMode, FlowPath
from kaful.artifacts.manifest import CodeVersion, InputFile, PhaseRecord, RunManifest
from kaful.artifacts.param_set import LifeStatement, Parameter
from kaful.artifacts.scope import ScopeDecision
from kaful.artifacts.sensor_contract import SensorChannel
from kaful.artifacts.synthetic import SyntheticUnit
from kaful.artifacts.twin_definition import (
    Connection,
    EventDef,
    ExternalInput,
    GeneratedCode,
    TemplateBinding,
)
from kaful.artifacts.verification import CheckResult
from kaful.phases import PHASE_ORDER

RUN_ID = "20261102T120000Z-abc123"
MACHINE = "tiny_pump"
SHA = "a" * 64
SHA_B = "b" * 64
BASE = {"run_id": RUN_ID, "machine_id": MACHINE}
EVIDENCE = Evidence(doc_id="manual", page=3, page_label="1-3", quote="Replace the seal every 2000 h.")
MANUAL = {"source": "manual", "ref": "manual.pdf p.3", "confidence": 0.8, "evidence": (EVIDENCE,)}
JUDGED = {"source": "llm_judgement", "ref": "judgement", "confidence": 0.3}


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
                          dtype="float", unit="L/min", range=(0.0, 40.0), sample_period_s=1.0, **MANUAL),
            SensorChannel(name="maintenance", description="Maintenance action", role="event",
                          dtype="category", categories=("replace_seal", "service"), **MANUAL),
            SensorChannel(name="run_hours", description="Operating hours", role="usage_clock",
                          dtype="float", unit="h", range=(0.0, 1e6), sample_period_s=60.0, **MANUAL),
            SensorChannel(name="suction_temp", description="Inlet temperature", role="operating_input",
                          dtype="float", unit="degC", range=(-10.0, 80.0), sample_period_s=1.0, **JUDGED),
        ),
    )


def machine_model() -> MachineModel:
    return MachineModel(
        **BASE,
        time_unit="h",
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
        components=("seal",),
        bindings=(TemplateBinding(
            id="seal_wear", component="seal", fault_mode="seal_wear", template="archard",
            template_version="1", damage_state="seal_wear_mm", damage_unit="mm",
            threshold_param="seal.wear_limit",
            params={k: f"seal.{k}" for k in {"wear_rate", "temp_exponent"}},
            conditions={"temperature": "suction_temp"}, justification="sliding wear",
            escape_hatch=GeneratedCode(path="outputs/runs/x/build/seal.py", sha256=SHA,
                                       entrypoint="rate", reason="nonstandard"),
        ),),
        connections=(Connection(id="c1", source_port="impeller.torque", target_port="seal.torque",
                                source_unit="N*m", target_unit="N*m", match="exact", score=1.0),),
        external_inputs=(ExternalInput(name="suction_temp", channel="suction_temp",
                                       nominal_param="machine.nominal_temp"),),
        events=(EventDef(name="seal_failure", binding="seal_wear"),),
    )


def verification_report() -> VerificationReport:
    return VerificationReport(**BASE, quarantined=("impeller",), results=(
        CheckResult(check="health_bounds", target="impeller", status="fail",
                    measured=_from_set({"min", "max"}, lambda k: 1.2 if k == "max" else 0.0)),
        CheckResult(check="health_bounds", target="seal", status="pass", limits={"max": 1.0}),
    ))


def operating_profile() -> OperatingProfile:
    return OperatingProfile(
        name="nominal",
        segments=(ProfileSegment(start=0.0, end=100.0, values={"suction_temp": 25.0}),
                  ProfileSegment(start=100.0, end=500.0, values={"suction_temp": 40.0})),
        events=(ScheduledEvent(t=250.0, channel="maintenance", value="service"),),
    )


def synthetic_dataset() -> SyntheticDataset:
    return SyntheticDataset(**BASE, seed=7, profile=operating_profile(), units=(SyntheticUnit(
        unit_id="unit_01", observations="outputs/runs/x/synthesize/unit_01.parquet",
        truth_states="outputs/runs/x/synthesize/unit_01_truth.parquet",
        true_failure_times={"seal_failure": 1800.0},
        true_params=_from_set({"seal.wear_rate", "seal.temp_exponent"}, lambda k: float(len(k))),
    ),))


def forecast() -> Forecast:
    return Forecast(**BASE, source="unit_01", estimator="particle_filter", seed=7, t_issue=100.0,
                    profile=operating_profile(), events=(EventForecast(
                        event="seal_failure", n_samples=4, toe_samples=(1700.0, 1900.0, 2100.0),
                        n_censored=1, horizon=3000.0, p_event_before_horizon=0.75,
                        percentiles={"p05": 1710.0, "p50": 1900.0, "p95": None},
                        already_triggered_fraction=0.0),))


def evaluation_report() -> EvaluationReport:
    return EvaluationReport(
        **BASE,
        grounding={"verbatim": 5, "near": 2, "unsupported": 1},
        params_by_source={"manual": 2, "calibrated": 1},
        prompt_scores=(PromptScore(prompt_id="extract_components", version="1", metric="f1", value=0.8),),
        verification_pass_rate=0.5,
        calibration=CalibrationMetrics(coverage_90=0.9, rmse_median=120.0),
        sensitivity=(SensitivityEntry(param_id="seal.wear_rate", share=0.7),),
        cost_by_phase=_from_set({"ingest", "understand"}, lambda k: 0.1),
        disclaimer="Rates are checked for consistency with documentation, not accuracy.",
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
    "scope": scope,
    "param_set": param_set,
    "twin_definition": twin_definition,
    "verification_report": verification_report,
    "synthetic_dataset": synthetic_dataset,
    "forecast": forecast,
    "evaluation_report": evaluation_report,
}

MINIMAL: dict[str, Callable[[], ArtifactBase]] = {
    "document_index": lambda: DocumentIndex(**BASE, index_hash=index_hash_for((), FINGERPRINT), documents=(),
                                            pages=(), chunks=(), fingerprint=FINGERPRINT),
    "machine_model": lambda: MachineModel(**BASE, time_unit="h", components=(), n_extractions=1),
    "sensor_contract": lambda: SensorContract(**BASE, channels=()),
    "scope": lambda: Scope(**BASE, decisions=()),
    "param_set": lambda: ParamSet(**BASE, time_unit="h", parameters=()),
    "twin_definition": lambda: TwinDefinition(**BASE, time_unit="h", components=(), bindings=()),
    "verification_report": lambda: VerificationReport(**BASE, results=()),
    "synthetic_dataset": lambda: SyntheticDataset(**BASE, seed=0, profile=operating_profile(), units=()),
    "forecast": lambda: Forecast(**BASE, source="s", estimator="open_loop", seed=0, t_issue=0.0,
                                 profile=operating_profile(), events=()),
    "evaluation_report": lambda: EvaluationReport(**BASE, grounding={}, params_by_source={},
                                                  verification_pass_rate=1.0, cost_by_phase={},
                                                  disclaimer="consistency, not accuracy"),
}


if __name__ == "__main__":
    import sys

    from kaful.runs import dump_json

    for kind in sorted(RICH):
        sys.stdout.write(dump_json(RICH[kind]()))
    sys.stdout.write(dump_json(run_manifest()))
