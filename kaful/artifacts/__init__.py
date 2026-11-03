"""Typed artifacts: one Pydantic model per phase output, plus the run manifest.

`ARTIFACT_MODELS` maps each `kind` to its model, in pipeline order.
"""

from kaful.artifacts.base import ArtifactBase, ArtifactRef, Evidence, Provenanced
from kaful.artifacts.document_index import DocumentIndex
from kaful.artifacts.evaluation import EvaluationReport
from kaful.artifacts.forecast import Forecast, OperatingProfile
from kaful.artifacts.machine_model import MachineModel
from kaful.artifacts.manifest import RunManifest
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.scope import Scope
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.synthetic import SyntheticDataset
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.artifacts.verification import VerificationReport

ARTIFACT_MODELS: dict[str, type[ArtifactBase]] = {
    "document_index": DocumentIndex,
    "machine_model": MachineModel,
    "sensor_contract": SensorContract,
    "scope": Scope,
    "param_set": ParamSet,
    "twin_definition": TwinDefinition,
    "verification_report": VerificationReport,
    "synthetic_dataset": SyntheticDataset,
    "forecast": Forecast,
    "evaluation_report": EvaluationReport,
}

__all__ = [
    "ARTIFACT_MODELS",
    "ArtifactBase",
    "ArtifactRef",
    "DocumentIndex",
    "EvaluationReport",
    "Evidence",
    "Forecast",
    "MachineModel",
    "OperatingProfile",
    "ParamSet",
    "Provenanced",
    "RunManifest",
    "Scope",
    "SensorContract",
    "SyntheticDataset",
    "TwinDefinition",
    "VerificationReport",
]
