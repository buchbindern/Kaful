"""Artifact models: round-trip, strictness, ordering and reference checks."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from kaful.artifacts import ARTIFACT_MODELS, RunManifest
from kaful.artifacts.base import ArtifactBase, Evidence
from kaful.artifacts.machine_model import Component
from kaful.artifacts.sensor_contract import SensorChannel
from kaful.phases import PHASE_ORDER
from kaful.runs import dump_json
from tests.harness import kaful_factories as f

ALL = sorted([(kind, "rich", fn) for kind, fn in f.RICH.items()]
             + [(kind, "minimal", fn) for kind, fn in f.MINIMAL.items()])


def _dump(model: ArtifactBase | RunManifest) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(dump_json(model))
    return data


def test_every_artifact_kind_has_factories() -> None:
    assert set(f.RICH) == set(ARTIFACT_MODELS) == set(f.MINIMAL)


@pytest.mark.parametrize(("kind", "variant", "factory"), ALL, ids=[f"{k}-{v}" for k, v, _ in ALL])
def test_artifact_round_trips_through_json(kind: str, variant: str, factory: Any) -> None:
    artifact = factory()
    text = dump_json(artifact)
    loaded = ARTIFACT_MODELS[kind].model_validate_json(text)
    assert loaded == artifact
    assert dump_json(loaded) == text
    assert loaded.schema_version == 1 and getattr(loaded, "kind") == kind


def test_manifest_round_trips() -> None:
    manifest = f.run_manifest()
    assert RunManifest.model_validate_json(dump_json(manifest)) == manifest


def test_manifest_requires_every_phase_in_order() -> None:
    data = _dump(f.run_manifest())
    data["phases"] = list(reversed(data["phases"]))
    with pytest.raises(ValidationError, match="pipeline order"):
        RunManifest.model_validate(data)
    assert tuple(r.phase for r in f.run_manifest().phases) == PHASE_ORDER


@pytest.mark.parametrize("kind", sorted(f.RICH))
def test_unknown_fields_are_rejected(kind: str) -> None:
    data = _dump(f.RICH[kind]())
    data["surprise"] = 1
    with pytest.raises(ValidationError, match="surprise"):
        ARTIFACT_MODELS[kind].model_validate(data)


@pytest.mark.parametrize("kind", sorted(f.RICH))
def test_wrong_schema_version_or_kind_is_rejected(kind: str) -> None:
    model = ARTIFACT_MODELS[kind]
    data = _dump(f.RICH[kind]())
    with pytest.raises(ValidationError):
        model.model_validate({**data, "schema_version": 2})
    with pytest.raises(ValidationError):
        model.model_validate({**data, "kind": "something_else"})


def test_artifacts_are_frozen() -> None:
    artifact = f.scope()
    with pytest.raises(ValidationError):
        artifact.run_id = "other"  # type: ignore[misc]


def test_non_finite_floats_are_rejected() -> None:
    data = _dump(f.sensor_contract())
    data["channels"][0]["range"] = [0.0, float("inf")]
    with pytest.raises(ValidationError):
        ARTIFACT_MODELS["sensor_contract"].model_validate(data)


def test_unsorted_collections_are_rejected() -> None:
    data = _dump(f.machine_model())
    data["components"] = list(reversed(data["components"]))
    with pytest.raises(ValidationError, match="must be sorted"):
        ARTIFACT_MODELS["machine_model"].model_validate(data)


def test_duplicate_ids_are_rejected() -> None:
    data = _dump(f.scope())
    data["decisions"] = [data["decisions"][0]] * 2
    with pytest.raises(ValidationError, match="duplicate"):
        ARTIFACT_MODELS["scope"].model_validate(data)


def test_manual_source_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence"):
        Component(id="seal", name="Seal", description="", category="seal",
                  source="manual", ref="manual p.3", confidence=0.5)


def test_downgrade_must_change_the_source() -> None:
    with pytest.raises(ValidationError, match="downgraded_from"):
        Component(id="seal", name="Seal", description="", category="seal", source="llm_judgement",
                  ref="x", confidence=0.5, downgraded_from="llm_judgement")
    downgraded = Component(id="seal", name="Seal", description="", category="seal",
                           source="llm_judgement", ref="x", confidence=0.2, downgraded_from="manual")
    assert downgraded.downgraded_from == "manual"


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_confidence_is_a_fraction(confidence: float) -> None:
    with pytest.raises(ValidationError):
        Component(id="seal", name="Seal", description="", category="seal",
                  source="default", ref="x", confidence=confidence)


@pytest.mark.parametrize("path", ["/abs/manual.pdf", "../other/manual.pdf", "a\\b.pdf", "C:/m.pdf", ""])
def test_absolute_or_escaping_paths_are_rejected(path: str) -> None:
    data = _dump(f.document_index())
    data["documents"][0]["path"] = path
    with pytest.raises(ValidationError):
        ARTIFACT_MODELS["document_index"].model_validate(data)


def test_document_index_hash_must_match_chunks() -> None:
    data = _dump(f.document_index())
    data["index_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="index_hash"):
        ARTIFACT_MODELS["document_index"].model_validate(data)


def test_machine_model_references_must_resolve() -> None:
    data = _dump(f.machine_model())
    data["fault_modes"][0]["component"] = "ghost"
    with pytest.raises(ValidationError, match="ghost"):
        ARTIFACT_MODELS["machine_model"].model_validate(data)


def test_twin_definition_events_must_name_a_binding() -> None:
    data = _dump(f.twin_definition())
    data["events"][0]["binding"] = "ghost"
    with pytest.raises(ValidationError, match="ghost"):
        ARTIFACT_MODELS["twin_definition"].model_validate(data)


def test_template_binding_records_damage_state_and_threshold() -> None:
    binding = f.twin_definition().bindings[0]
    assert (binding.damage_state, binding.damage_unit, binding.threshold_param) == (
        "seal_wear_mm", "mm", "seal.wear_limit")


@pytest.mark.parametrize(("changes", "message"), [
    ({"dtype": "float", "unit": None}, "need a unit"),
    ({"dtype": "category", "categories": None}, "categories"),
    ({"range": (5.0, 1.0)}, "low must be <= high"),
    ({"role": "event"}, "event channels"),
])
def test_sensor_channel_rules(changes: dict[str, Any], message: str) -> None:
    base: dict[str, Any] = {"name": "flow", "description": "d", "role": "observation", "dtype": "float",
                            "unit": "L/min", "source": "default", "ref": "x", "confidence": 1.0}
    with pytest.raises(ValidationError, match=message):
        SensorChannel(**{**base, **changes})


def test_sensor_contract_usage_clock_must_match_its_channel() -> None:
    data = _dump(f.sensor_contract())
    data["usage_clock"] = "flow_rate"
    with pytest.raises(ValidationError, match="usage_clock"):
        ARTIFACT_MODELS["sensor_contract"].model_validate(data)


def test_maintenance_events_are_category_event_channels() -> None:
    channel = f.sensor_contract().channel("maintenance")
    assert channel.role == "event" and channel.categories == ("replace_seal", "service")
    assert f.operating_profile().events[0].channel == "maintenance"


def test_operating_profile_segments_must_be_contiguous() -> None:
    data = _dump(f.forecast())
    data["profile"]["segments"][1]["start"] = 150.0
    with pytest.raises(ValidationError, match="contiguous"):
        ARTIFACT_MODELS["forecast"].model_validate(data)


def test_event_forecast_counts_must_add_up() -> None:
    data = _dump(f.forecast())
    data["events"][0]["n_censored"] = 0
    with pytest.raises(ValidationError, match="n_censored"):
        ARTIFACT_MODELS["forecast"].model_validate(data)


def test_verification_passes_only_when_failures_are_quarantined() -> None:
    report = f.verification_report()
    assert report.passed
    assert not report.model_copy(update={"quarantined": ()}).passed
    assert report.model_copy(update={"quarantined": (), "allowed_failures": True}).passed


def test_calibration_target_is_85_to_95_percent() -> None:
    calibration = f.evaluation_report().calibration
    assert calibration is not None and calibration.coverage_target == (0.85, 0.95)
    assert calibration.meets_target
    assert not calibration.model_copy(update={"coverage_90": 0.8}).meets_target


def test_synthetic_dataset_carries_the_sanity_check_disclaimer() -> None:
    assert f.synthetic_dataset().disclaimer.startswith("Pipeline sanity check, not validation")


def test_evidence_pages_are_one_based() -> None:
    with pytest.raises(ValidationError):
        Evidence(doc_id="manual", page=0, quote="q")
