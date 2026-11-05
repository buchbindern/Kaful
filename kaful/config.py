"""Settings (kaful.toml + KAFUL_* env vars) and per-machine config (machine.toml)."""

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from kaful.artifacts.base import EvidenceClass, Identifier, NonEmptyStr, StrictModel, by_attr
from kaful.artifacts.document_index import RuledTablesRoute
from kaful.artifacts.sensor_contract import ChannelDtype, ChannelRole, Sampling
from kaful.paths import RelPath

REPO_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = REPO_ROOT / "kaful.toml"
MACHINE_CONFIG_FILE = "machine.toml"


class SandboxSettings(BaseModel):
    """Limits for running generated code (see kaful.sandbox)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timeout_s: float = Field(default=60.0, gt=0.0)
    cpu_limit_s: int = Field(default=120, ge=1)
    max_output_bytes: int = Field(default=65536, ge=1024)
    memory_limit_mb: int | None = Field(default=None, ge=64)
    hash_seed: int = Field(default=0, ge=0)


class ClassifierSettings(BaseModel):
    """Page classifier thresholds (see kaful.phases.p1_ingest.c_classify_pages). Rules apply in order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    blank_max_chars: int = Field(default=50, ge=0)
    blank_max_image_frac: float = Field(default=0.10, ge=0.0, le=1.0)
    blank_max_drawings: int = Field(default=20, ge=0)
    scanned_max_chars: int = Field(default=200, ge=0)
    scanned_min_image_frac: float = Field(default=0.5, ge=0.0, le=1.0)
    garbled_min_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    figure_min_image_frac: float = Field(default=0.05, ge=0.0, le=1.0, description="Screenshots are ~10% of a page.")
    figure_min_curves: int = Field(default=150, ge=1, description="Bezier or diagonal path items; icons give < 150.")
    table_min_rows: int = Field(default=2, ge=1)
    table_min_cols: int = Field(default=2, ge=1)
    table_max_empty_frac: float = Field(default=0.4, ge=0.0, le=1.0)
    unruled_min_lines: int = Field(default=15, ge=1)
    unruled_min_short_frac: float = Field(default=0.6, ge=0.0, le=1.0)
    unruled_min_numeric_frac: float = Field(default=0.15, ge=0.0, le=1.0)


class IngestSettings(BaseModel):
    """Ingest phase settings: classifier, page labels, rendering, chunking, embeddings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    classifier: ClassifierSettings = ClassifierSettings()
    classifier_min_recall: float = Field(default=0.90, ge=0.0, le=1.0, description="M6 gate: needs-LLM recall.")
    review_per_class: int = Field(default=6, ge=1, description="Pages sampled per predicted class per machine.")
    grounding_min_tokens: int = Field(default=3, ge=1, description="Shorter legacy quotes are left out.")
    grounding_near_ratio: float = Field(default=0.85, gt=0.0, le=1.0, description="SequenceMatcher ratio for 'near'.")
    grounding_candidate_overlap: float = Field(default=0.6, gt=0.0, le=1.0,
                                               description="Share of quote tokens a passage needs to be compared.")
    label_band_frac: float = Field(default=0.08, gt=0.0, lt=0.5, description="Header/footer band height.")
    label_min_pages: int = Field(default=5, ge=1)
    label_min_share: float = Field(default=0.5, gt=0.0, le=1.0)
    render_max_dpi: float = Field(default=150.0, gt=0.0)
    render_long_edge_px: int = Field(default=1568, ge=200)
    chunk_max_tokens: int = Field(default=1200, ge=100)
    chunk_chars_per_token: float = Field(default=4.0, gt=0.0, description="Chunk sizing only, not budgeting.")
    toc_max_level: int = Field(default=2, ge=1)
    embedding_model: str = Field(default="text-embedding-3-large", min_length=1)
    embedding_dimensions: int = Field(default=3072, ge=1)
    embedding_batch_size: int = Field(default=64, ge=1)
    embedding_encodings: dict[str, str] = Field(
        default={"text-embedding-3-large": "cl100k_base", "text-embedding-3-small": "cl100k_base"},
        description="Tokenizer per embedding model, for exact token counts (kaful.llm.tokens).")
    embedding_max_tokens: int = Field(default=8191, ge=1, description="The embedding model's input limit.")
    embedding_window_tokens: int = Field(default=8000, ge=1, description="A longer chunk is embedded in windows.")
    embedding_window_overlap: int = Field(default=200, ge=0, description="Tokens each window repeats from the last.")

    @model_validator(mode="after")
    def _check_windows(self) -> "IngestSettings":
        if not self.embedding_window_overlap < self.embedding_window_tokens <= self.embedding_max_tokens:
            raise ValueError("need embedding_window_overlap < embedding_window_tokens <= embedding_max_tokens")
        return self

    def embedding_encoding(self) -> str:
        encoding = self.embedding_encodings.get(self.embedding_model)
        if encoding is None:
            raise ValueError(f"no tokenizer configured for embedding model {self.embedding_model!r}")
        return encoding


RetrievalTopic = Literal["components", "fault_modes", "sensor_channels"]


class UnderstandSettings(BaseModel):
    """Understand phase settings: context selection, self-consistency, evidence, confidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_samples: int = Field(default=3, ge=1, description="Self-consistency N per prompt; tuned against evals.")
    context_max_tokens: int = Field(default=40000, ge=1000, description="Whole index if it fits, else retrieval.")
    retrieval_top_k: int = Field(default=6, ge=1, description="Chunks per query in retrieval mode.")
    queries: dict[RetrievalTopic, tuple[NonEmptyStr, ...]] = {
        "components": (
            "main components and assemblies of the machine and what they are part of",
            "how the fluid, power or material flows through the machine",
            "parts list, spare parts and wear parts",
        ),
        "fault_modes": (
            "alarms, warnings and error codes and what causes them",
            "troubleshooting: faults, symptoms and causes",
            "wear, service intervals, inspection and replacement of parts",
        ),
        "sensor_channels": (
            "signals, sensors and measured values the controller reports or displays",
            "diagnostic data, data output and readable parameters",
            "sampling interval, logging rate and update period of measured data",
        ),
    }
    evidence_near_ratio: float = Field(default=0.85, gt=0.0, le=1.0)
    evidence_candidate_overlap: float = Field(default=0.6, gt=0.0, le=1.0)
    evidence_weights: dict[EvidenceClass, float] = Field(
        default={"verbatim": 1.0, "near": 0.8, "in_layer": 0.5, "unsupported": 0.2, "unverified": 0.0},
        description="Confidence factor for an item's best evidence; confidence = agreement x weight (rule 15).",
    )


Effort = Literal["low", "medium", "high", "xhigh", "max"]


# first_pass_failed: the first model's reply stayed invalid (e.g. it ran out of max_tokens);
# output_has_tables: its valid output has a table.
EscalationRule = Literal["first_pass_failed", "output_has_tables"]


class Escalation(BaseModel):
    """Re-run a call on a stronger model when the first pass meets any rule in `when`, and use
    that output instead. The phase that calls the prompt implements the rules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    effort: Effort | None = None
    when: tuple[EscalationRule, ...] = Field(min_length=1, description="Sorted, no repeats.")

    @model_validator(mode="after")
    def _sorted(self) -> "Escalation":
        if list(self.when) != sorted(set(self.when)):
            raise ValueError("escalate.when must be sorted, without repeats")
        return self

    def rule(self) -> str:
        return " or ".join(self.when)


class ModelChoice(BaseModel):
    """The model (and optional effort level, request timeout and escalation) one prompt runs on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    effort: Effort | None = None
    request_timeout_s: float | None = Field(
        default=None, gt=0.0, description="Overrides LLMSettings.request_timeout_s for this prompt."
    )
    escalate: Escalation | None = None

    def escalated(self) -> "ModelChoice":
        """The choice an escalated call runs on (same timeout, no further escalation)."""
        if self.escalate is None:
            raise ValueError(f"{self.model} has no escalation")
        return ModelChoice(model=self.escalate.model, effort=self.escalate.effort,
                           request_timeout_s=self.request_timeout_s)


class ReuseRule(BaseModel):
    """Reuse an older version's cached response for a request that differs from it only in
    `max_tokens` (the older, lower limit), when that response ended normally: it never
    reached the old limit, so the higher one could not have changed it. The reused page
    records the older request (see kaful.llm.gateway)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(ge=1, description="The older prompt version whose responses may be reused.")
    max_tokens: int = Field(gt=0, description="That version's max_tokens (the only difference allowed).")


class LLMSettings(BaseModel):
    """LLM gateway settings (see kaful.llm). Model choice is per prompt, never in code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_model: str = Field(default="claude-sonnet-5-5", min_length=1)
    default_effort: Effort | None = None
    prompts: dict[str, ModelChoice] = Field(
        default_factory=dict, description="Per-prompt model choice, keyed by prompt id."
    )
    token_estimate: Literal["count_tokens", "chars"] = Field(
        default="count_tokens",
        description="Input-token estimate for the pre-call budget check. count_tokens is exact and "
        "free but needs the network; chars is used whenever the transport cannot count.",
    )
    chars_per_token_worst: float = Field(default=2.5, gt=0.0)
    pdf_tokens_per_page_worst: int = Field(default=3000, gt=0)
    image_pixels_per_token: float = Field(default=750.0, gt=0.0, description="Anthropic: tokens ~ w*h/750.")
    cache_reads: bool = Field(default=True, description="False: ignore cached responses (still write).")
    reuse: dict[str, ReuseRule] = Field(default_factory=dict, description="Per prompt id: older responses to reuse.")
    batch_poll_s: float = Field(default=60.0, gt=0.0)
    batch_max_wait_s: float = Field(default=86400.0, gt=0.0)
    request_timeout_s: float = Field(default=120.0, gt=0.0, description="Per request; prompts may override.")
    sdk_max_retries: int = Field(default=2, ge=0)

    def choice(self, prompt_id: str) -> ModelChoice:
        return self.prompts.get(prompt_id) or ModelChoice(model=self.default_model, effort=self.default_effort)

    def timeout_for(self, prompt_id: str) -> float:
        """Request timeout in seconds: the prompt's own, else the global one."""
        override = self.choice(prompt_id).request_timeout_s
        return self.request_timeout_s if override is None else override


class KafulSettings(BaseSettings):
    """Global settings. Precedence: constructor args > KAFUL_* env > kaful.toml > defaults."""

    model_config = SettingsConfigDict(
        env_prefix="KAFUL_",
        env_nested_delimiter="__",
        toml_file=SETTINGS_FILE,
        extra="forbid",
        frozen=True,
    )

    machines_dir: Path = Field(default=Path("machines"), description="Relative to the repo root.")
    default_budget_usd: float = Field(default=10.0, gt=0.0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    sandbox: SandboxSettings = SandboxSettings()
    llm: LLMSettings = LLMSettings()
    ingest: IngestSettings = IngestSettings()
    understand: UnderstandSettings = UnderstandSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, TomlConfigSettingsSource(settings_cls)

    def machine_dir(self, machine_id: str) -> Path:
        base = self.machines_dir if self.machines_dir.is_absolute() else REPO_ROOT / self.machines_dir
        return base / machine_id

    def snapshot(self) -> dict[str, object]:
        """Settings recorded in the run manifest (no paths, no secrets)."""
        return self.model_dump(mode="json", exclude={"machines_dir"})


class DocumentEntry(StrictModel):
    """A document the pipeline reads, relative to the machine directory."""

    path: RelPath
    kind: Literal["manual", "standard", "datasheet", "catalogue", "other"]
    ruled_tables: RuledTablesRoute = Field(
        default="pymupdf", description="'llm': also send this document's table_ruled pages to the LLM layer "
        "(PyMuPDF loses the rows a merged cell spans).")


class DeclaredChannel(StrictModel):
    """A channel the user's installation adds (e.g. a retrofitted accelerometer).

    It enters the SensorContract with source `deployment`, citing machine.toml; the
    SensorChannel rules (units, sampling) are checked when the contract is built.
    """

    name: Identifier
    description: NonEmptyStr
    role: ChannelRole
    dtype: ChannelDtype
    unit: str | None = None
    range: tuple[float, float] | None = None
    categories: tuple[NonEmptyStr, ...] | None = None
    sampling: Sampling
    sample_period_s: float | None = Field(default=None, gt=0.0)


class MachineConfig(StrictModel):
    """Contents of machines/<id>/machine.toml."""

    machine_id: Identifier
    display_name: NonEmptyStr
    time_unit: NonEmptyStr = Field(description="The one time unit every rate is expressed in.")
    time_unit_provisional: bool = Field(
        default=False, description="True until M9 confirms the unit; see `confirmed_time_unit`."
    )
    documents: Annotated[tuple[DocumentEntry, ...], by_attr("path")] = ()
    instrumentation: Annotated[tuple[DeclaredChannel, ...], by_attr("name")] = Field(
        default=(), description="[[instrumentation]]: channels added by the deployment, not in any document."
    )

    def confirmed_time_unit(self) -> str:
        """The time unit, for code that converts or expresses rates. Refuses a provisional one."""
        if self.time_unit_provisional:
            raise MachineConfigError(f"{self.machine_id}: time_unit {self.time_unit!r} is provisional (M9 decides)")
        return self.time_unit


class MachineConfigError(ValueError):
    """machine.toml is missing or invalid."""


def load_machine_config(machine_dir: Path) -> MachineConfig:
    """Load and validate machine.toml; its machine_id must match the directory name."""
    path = machine_dir / MACHINE_CONFIG_FILE
    if not path.is_file():
        raise MachineConfigError(f"{machine_dir.name}: no {MACHINE_CONFIG_FILE} found")
    with path.open("rb") as handle:
        config = MachineConfig.model_validate(tomllib.load(handle))
    if config.machine_id != machine_dir.name:
        raise MachineConfigError(
            f"{MACHINE_CONFIG_FILE} machine_id {config.machine_id!r} != directory {machine_dir.name!r}"
        )
    return config
