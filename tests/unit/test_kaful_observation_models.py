"""Observation models through parameterize (step f), build (step d) and the static gates (M12), on the small
machine of the scope, parameterize and build tests: the shaft seal's wear is observable by `leak_flow`."""

from typing import Any

import pytest
from pydantic import ValidationError

from kaful.artifacts.distributions import Lognormal
from kaful.artifacts.sensor_contract import SensorChannel
from kaful.config import DeclaredObservation, DeclaredObservationChannel, DeclaredValue, KafulSettings
from kaful.phases.p3_scope.phase import build_scope
from kaful.phases.p4_parameterize.f_observation_priors import FLAG_DOWNGRADED, FLAG_JUDGED, FLAG_NOISE, FLAG_TRANSCRIBED
from kaful.phases.p4_parameterize.phase import build_param_set
from kaful.phases.p5_build.phase import build_twin
from kaful.phases.p6_verify.d_quarantine import reduce
from kaful.artifacts.verification import Quarantine
from kaful.twin_checks import observation_problems
from tests.harness.kaful_factories import BASE
from tests.unit.test_kaful_parameterize import _index, _judge_all
from tests.unit.test_kaful_scope import _channel, _contract, _machine, _model

SETTINGS = KafulSettings()
LEAK_PAGE = "Normal leakage is below 2 L/min. Replace the seal when the leakage exceeds 6 L/min."


def _value(**kw: Any) -> DeclaredValue:
    return DeclaredValue(**{"source": "deployment", "ref": "test", **kw})


def _leak(y_limit: DeclaredValue | None = None, y_healthy: DeclaredValue | None = None,
          noise: DeclaredValue | None = None) -> dict[str, Any]:
    return {"observation": (DeclaredObservation(component="shaft_seal", channel="leak_flow", form="linear",
                                                y_limit=y_limit or _value(low=4.0, high=8.0), reason="a worn seal leaks"),),
            "observation_channel": (DeclaredObservationChannel(channel="leak_flow",
                                                               y_healthy=y_healthy or _value(low=0.5, high=2.0),
                                                               noise_sd=noise),)}


def _alarm(dtype: str = "bool") -> SensorChannel:
    categories = ("leak_warning", "other") if dtype == "category" else None
    return SensorChannel(name="seal_alarm", description="Seal alarm", role="event", dtype=dtype,  # type: ignore[arg-type]
                         categories=categories, sampling="event_driven", source="manual", ref="manual p1",
                         confidence=0.9, evidence=_channel("x", "x").evidence)


def _index_with_leak_page() -> Any:
    index = _index(True)
    pages = tuple(p.model_copy(update={"text": p.text + "\n" + LEAK_PAGE}) if (p.doc_id, p.page) == ("manual", 1)
                  else p for p in index.pages)
    return index.model_copy(update={"pages": pages})


def _run(machine: Any, contract: Any = None) -> tuple[Any, Any, tuple[str, ...], tuple[str, ...]]:
    contract, index = contract or _contract(_alarm()), _index_with_leak_page()
    scope = build_scope({**BASE}, index, _model(), contract, machine, SETTINGS)[0]
    built = build_param_set({**BASE}, index, _model(), scope, contract, machine, SETTINGS, _judge_all)
    twin, failures = build_twin({**BASE}, built.param_set, scope, contract, machine, SETTINGS)
    return twin, built.param_set, built.failures, failures


def test_a_linear_observation_gets_its_three_priors_and_a_binding_with_its_scope_link() -> None:
    twin, ps, p_fail, b_fail = _run(_machine(**_leak()))
    assert p_fail == () and b_fail == ()
    limit, healthy, noise = (ps.get(i) for i in ("shaft_seal.leak_flow_y_limit", "machine.leak_flow_y_healthy",
                                                  "machine.leak_flow_noise_sd"))
    assert (limit.role, healthy.role, noise.role) == ("observation", "observation", "noise")
    assert {limit.unit, healthy.unit, noise.unit} == {"L/min"}
    assert isinstance(noise.prior, Lognormal) and noise.source == "default"
    assert noise.prior.median == pytest.approx(0.05 * (limit.prior.median - healthy.prior.median))
    assert limit.prior.quantile(0.05) == pytest.approx(4.0) and limit.prior.quantile(0.95) == pytest.approx(8.0)
    [obs] = twin.observations
    assert (obs.id, obs.binding, obs.channel, obs.form, obs.basis, obs.ref) == (
        "shaft_seal_on_leak_flow", "shaft_seal", "leak_flow", "linear", "observable_by", "seal_wear")
    assert obs.params == {"y_healthy": "machine.leak_flow_y_healthy", "y_limit": "shaft_seal.leak_flow_y_limit",
                          "noise_sd": "machine.leak_flow_noise_sd"}
    assert FLAG_NOISE in next(t for t in ps.templates if t.component == "shaft_seal").flags


def test_a_declared_noise_replaces_the_default() -> None:
    _, ps, failures, _ = _run(_machine(**_leak(noise=_value(value=0.1))))
    assert failures == () and ps.get("machine.leak_flow_noise_sd").source == "deployment"
    assert ps.get("machine.leak_flow_noise_sd").ref.startswith("machine.toml")


def test_a_manual_value_keeps_its_source_only_when_its_quote_is_on_the_page() -> None:
    found = _value(value=6.0, source="manual", ref="manual p1", doc_id="manual", page=1,
                   quote="Replace the seal when the leakage exceeds 6 L/min.")
    _, ps, failures, _ = _run(_machine(**_leak(y_limit=found)))
    limit = ps.get("shaft_seal.leak_flow_y_limit")
    assert failures == () and limit.source == "manual" and limit.evidence[0].verification == "verbatim"
    assert limit.evidence[0].page_label == "1"
    missing = found.model_copy(update={"quote": "Replace the seal when the leakage reaches 9 L/min."})
    _, ps, failures, _ = _run(_machine(**_leak(y_limit=missing)))
    limit = ps.get("shaft_seal.leak_flow_y_limit")
    assert failures == () and (limit.source, limit.downgraded_from) == ("deployment", "manual")
    assert FLAG_DOWNGRADED in next(t for t in ps.templates if t.component == "shaft_seal").flags


def test_a_transcribed_literature_value_keeps_its_source_and_is_flagged() -> None:
    fig = _value(low=4.0, high=8.0, source="literature", ref="catalogue p2 figure", transcribed=True)
    _, ps, failures, _ = _run(_machine(**_leak(y_limit=fig)))
    assert failures == () and ps.get("shaft_seal.leak_flow_y_limit").source == "literature"
    assert FLAG_TRANSCRIBED in next(t for t in ps.templates if t.component == "shaft_seal").flags


def test_a_judged_value_is_flagged_and_scores_below_a_documented_one() -> None:
    judged = _value(low=4.0, high=8.0, source="llm_judgement", ref="judgement")
    _, ps, failures, _ = _run(_machine(**_leak(y_limit=judged)))
    _, documented, _, _ = _run(_machine(**_leak(y_limit=_value(low=4.0, high=8.0, source="literature", ref="x"))))
    pid = "shaft_seal.leak_flow_y_limit"
    assert failures == () and ps.get(pid).confidence == pytest.approx(0.3 * documented.get(pid).confidence)
    assert FLAG_JUDGED in next(t for t in ps.templates if t.component == "shaft_seal").flags


def test_a_cited_page_outside_the_index_fails_the_gate() -> None:
    gone = _value(value=6.0, source="literature", ref="x", doc_id="manual", page=99, quote="q")
    _, _, failures, _ = _run(_machine(**_leak(y_limit=gone)))
    assert any("p99 is not in the document index" in f for f in failures)


def test_a_y_limit_that_overlaps_the_healthy_reading_fails_the_gate() -> None:
    _, _, failures, _ = _run(_machine(**_leak(y_limit=_value(low=1.5, high=3.0))))
    assert any("direction is undetermined" in f for f in failures)


def test_a_channel_outside_the_contract_fails_the_gate() -> None:
    ghost = {"observation": (DeclaredObservation(component="shaft_seal", channel="ghost", form="threshold_event",
                                                 reason="r"),)}
    _, _, failures, _ = _run(_machine(**ghost))
    assert failures == ("machine.toml [[observation]] shaft_seal -> ghost: channel not in the SensorContract",)


def test_a_threshold_event_needs_no_parameter_and_records_a_declared_basis() -> None:
    alarm = {"observation": (DeclaredObservation(component="shaft_seal", channel="seal_alarm", form="threshold_event",
                                                 reason="the controller alarms at the leak limit"),)}
    twin, ps, p_fail, b_fail = _run(_machine(**alarm))
    assert p_fail == () and b_fail == ()
    [obs] = twin.observations
    assert (obs.form, obs.params, obs.basis, obs.event_value) == ("threshold_event", {}, "declared", None)
    assert obs.ref == "machine.toml [[observation]] shaft_seal -> seal_alarm"
    assert not [x for x in ps.parameters if "seal_alarm" in x.name]


def test_an_observation_of_a_component_without_a_full_model_is_not_bound() -> None:
    valve = {"observation": (DeclaredObservation(component="valve", channel="seal_alarm", form="threshold_event",
                                                 reason="r"),)}
    twin, _, p_fail, b_fail = _run(_machine(**valve))
    assert p_fail == () and b_fail == () and twin.observations == ()


@pytest.mark.parametrize(("change", "message"), [
    ({"form": "linear"}, "needs a numeric channel"),
    ({"event_value": "leak_warning"}, "a bool channel reports True"),
])
def test_the_form_must_suit_the_channel(change: dict[str, Any], message: str) -> None:
    alarm = {"observation": (DeclaredObservation(component="shaft_seal", channel="seal_alarm", form="threshold_event",
                                                 reason="r"),)}
    twin, ps, _, _ = _run(_machine(**alarm))
    bad = twin.model_copy(update={"observations": (twin.observations[0].model_copy(update=change),)})
    assert any(message in p.message for p in observation_problems(bad, ps, _contract(_alarm())))


def test_a_category_channel_needs_one_of_its_categories() -> None:
    alarm = {"observation": (DeclaredObservation(component="shaft_seal", channel="seal_alarm", form="threshold_event",
                                                 event_value="leak_warning", reason="r"),)}
    contract = _contract(_alarm("category"))
    twin, ps, _, failures = _run(_machine(**alarm), contract)
    assert failures == () and twin.observations[0].event_value == "leak_warning"
    bad = twin.model_copy(update={"observations": (twin.observations[0].model_copy(update={"event_value": "x"}),)})
    assert [p.check for p in observation_problems(bad, ps, contract)] == ["observation_form"]


def test_units_slots_and_shared_channel_parameters_are_gated() -> None:
    twin, ps, _, _ = _run(_machine(**_leak()))
    obs = twin.observations[0]
    wrong_unit = ps.model_copy(update={"parameters": tuple(
        x.model_copy(update={"unit": "m3/h"}) if x.param_id == "shaft_seal.leak_flow_y_limit" else x
        for x in ps.parameters)})
    assert [p.check for p in observation_problems(twin, wrong_unit, _contract(_alarm()))] == ["observation_units"]
    no_slot = twin.model_copy(update={"observations": (obs.model_copy(update={"params": {
        k: v for k, v in obs.params.items() if k != "noise_sd"}}),)})
    assert [p.check for p in observation_problems(no_slot, ps, _contract(_alarm()))] == ["params_exist"]
    second = obs.model_copy(update={"id": "bearings_on_leak_flow", "binding": "bearings", "params": {
        **obs.params, "y_healthy": "shaft_seal.leak_flow_y_limit"}})
    shared = twin.model_copy(update={"observations": (second, obs)})
    assert "observation_shared" in [p.check for p in observation_problems(shared, ps, _contract(_alarm()))]


def test_quarantine_drops_the_observations_of_a_quarantined_component() -> None:
    twin, _, _, _ = _run(_machine(**_leak()))
    reduced = reduce(twin, (Quarantine(component="shaft_seal", checks=("x",), reason="r"),), {**BASE})
    assert reduced.observations == () and "shaft_seal" not in reduced.components


@pytest.mark.parametrize(("build", "message"), [
    (lambda: DeclaredObservation(component="c", channel="x", form="linear", reason="r"), "y_limit is required"),
    (lambda: DeclaredObservation(component="c", channel="x", form="threshold_event", reason="r",
                                 y_limit=_value(value=1.0)), "y_limit is required"),
    (lambda: _value(value=1.0, source="manual"), "needs its quote"),
    (lambda: _value(value=1.0, doc_id="manual"), "doc_id, page and quote"),
    (lambda: _value(low=2.0, high=1.0), "low must be below high"),
    (lambda: _value(value=1.0, low=0.5, high=2.0), "give `value`"),
    (lambda: _machine(observation=_leak()["observation"]), "must cover exactly"),
    (lambda: _machine(observation_channel=_leak()["observation_channel"]), "must cover exactly"),
])
def test_declarations_are_validated(build: Any, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        build()
