"""Settings (kaful.toml + KAFUL_* env vars) and per-machine config (machine.toml)."""

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from kaful.artifacts.base import EvidenceClass, Identifier, NonEmptyStr, StrictModel, by_attr, sorted_unique
from kaful.artifacts.document_index import LEGACY_KINDS, DocumentKind, RuledTablesRoute
from kaful.artifacts.param_set import LifeKind
from kaful.artifacts.scope import Decision
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
    part_max_tokens: int = Field(default=80000, ge=1000, description=(
        "Machine documents larger than whole_max_tokens are read in parts of at most this many estimated tokens "
        "(M7c, approved 2026-11-09): about eversys's whole context, and below Haiku's 100k-token price step."))
    batch_parts: bool = Field(default=True, description=(
        "Sample the prompts of a context with several parts through the Batch API (half price); one part stays "
        "sync, so a whole-context run sends the same requests as before M7c."))
    whole_max_tokens: int | None = Field(default=None, ge=1000, description=(
        "The whole index goes in when its token count (count_tokens when available, else the estimate) is at most "
        "this; None means context_max_tokens. Else retrieval, limited by context_max_tokens."))
    context_chars_per_token: float = Field(default=2.1, gt=0.0, description=(
        "Context size estimates. Calibrated 2026-11-07 with count_tokens: eversys 2.37 (retrieved) and 2.46 "
        "(whole), CNC 2.10; the lowest, so an estimate is never below the real count."))
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
    example_markers: tuple[NonEmptyStr, ...] = Field(
        default=("example", "e.g.", "for example", "for instance"),
        description="A value quoted shortly after one of these is from a worked example (kaful.value_roles).")
    example_window_tokens: int = Field(default=40, ge=1, description="How far before a quote a marker counts.")
    example_figure_kinds: tuple[NonEmptyStr, ...] = Field(
        default=("screenshot", "photo"), description="Figure kinds whose values are examples (one display state).")
    evidence_near_ratio: float = Field(default=0.85, gt=0.0, le=1.0)
    machine_kinds: tuple[DocumentKind, ...] = Field(default=("manual",), description=(
        "Document kinds that define the machine: components and channels come only from these (plus machine.toml "
        "[[instrumentation]]); catalogues, standards and literature add fault-mode evidence (2026-11-08)."))
    merge_generic_heads: tuple[NonEmptyStr, ...] = Field(
        default=("module", "unit", "system", "assembly"),
        description="Trailing words that do not tell two component names apart (grinder / grinder module).")
    merge_variant_words: tuple[NonEmptyStr, ...] = Field(
        default=("series", "type"),
        description=("Words that make a component a variant of another (spindle motor αi series / spindle "
                     "motor); a variant folds into its base (M7c, approved 2026-11-09)."))
    golden_text_overlap: float = Field(default=0.5, gt=0.0, le=1.0, description=(
        "Share of a golden fault mode's what_degrades words a run's mode must name to match it (golden_score)."))
    golden_min_word_chars: int = Field(default=3, ge=1, description="Shorter words are ignored when matching.")
    evidence_candidate_overlap: float = Field(default=0.6, gt=0.0, le=1.0)
    evidence_weights: dict[EvidenceClass, float] = Field(
        default={"verbatim": 1.0, "near": 0.8, "in_layer": 0.5, "unsupported": 0.2, "unverified": 0.0},
        description="Confidence factor for an item's best evidence; confidence = agreement x weight (rule 15).",
    )


class ScopeSettings(BaseModel):
    """Scope (M8, approved 2026-11-09): which components get full models, by rules over what understand produced.

    The life signal (L) is found by code: a sentence of a machine document that names a component, states a
    number with a time or usage unit and has a life or interval word; or a reference-document page (a kind not
    in understand.machine_kinds) that the component's fault-mode evidence cites, prints every word of the
    component's name and has a life-rating or life-model phrase."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    life_words: tuple[NonEmptyStr, ...] = Field(default=(
        "every", "replace", "change", "interval", "life", "service", "maintenance", "inspect", "overhaul"),
        description="A machine-document sentence or table row states a life or interval only with a word that starts "
        "with one of these (replaced, replacement, lifetime); 'after' is left out: eversys warranty and stop messages.")
    life_periods: tuple[NonEmptyStr, ...] = Field(default=(
        "daily", "weekly", "monthly", "quarterly", "yearly", "annually"), description=(
        "Words that state a period without a number (a schedule's Period column: 3-monthly, Yearly)."))
    life_units: tuple[NonEmptyStr, ...] = Field(default=(
        "h", "hr", "hrs", "hour", "hours", "day", "days", "week", "weeks", "month", "months", "year", "years",
        "cycle", "cycles", "product", "products", "operation", "operations", "stroke", "strokes", "km",
        "revolution", "revolutions", "shot", "shots", "brew", "brews"), description=(
        "Time or usage units a stated life or interval is counted in, written after its number."))
    life_model_phrases: tuple[NonEmptyStr, ...] = Field(default=(
        "L10", "rating life", "rated life", "basic dynamic load rating", "dynamic load rating", "MTBF", "B10 life",
        "fatigue life", "service life", "life expectancy"), description=(
        "Phrases that mark a life rating or life model on a reference-document page, strongest first: a page's "
        "quote is its line with the earliest phrase."))
    machine_words: tuple[NonEmptyStr, ...] = Field(default=("machine",), description=(
        "A statement with one of these words is about the whole machine (eversys p84 'The service of Eversys coffee "
        "machines ... 12 months or every 50'000 products'): machine-wide, even when it also names a component."))
    name_word_min_chars: int = Field(default=3, ge=1, description=(
        "A machine-wide statement that prints a word of a component's name this long (generic heads aside) is "
        "about that part, not the machine: air_compressor p29 'Have all flexibles ... inspected'."))
    statement_max_words: int = Field(default=60, ge=1, description=(
        "A longer 'sentence' is a table or list read without full stops (CNC p672 parameter table), not a statement."))
    observation_limit_roles: tuple[NonEmptyStr, ...] = Field(default=("spec", "setting_range"), description=(
        "Value roles of a channel range that count as a documented limit (an example never does)."))
    channel_quantity_words: tuple[NonEmptyStr, ...] = Field(default=(
        "temperature", "pressure", "speed", "flow", "rate", "level", "current", "voltage", "torque", "load",
        "vibration", "position", "error", "dewpoint", "humidity", "count", "counter", "time", "power", "frequency",
        "data", "information", "reading", "value", "signal", "state", "status", "feedback", "difference",
        "deviation", "noise", "resistance", "consumption"), description=(
        "A channel's name or description names a component when the name is followed by one of these words, by "
        "punctuation or by nothing: 'Coffee boiler temperature', 'dewpoint temperature of the air dryer, ...'. "
        "Followed by any other word, the name only qualifies another part: 'the coffee boiler flowmeter' (M8, "
        "Natalie 2026-11-09)."))
    named_channel_factor: float = Field(default=0.5, gt=0, lt=1, description=(
        "A channel that only names the component observes it with this fraction of the channel's confidence; a "
        "fault-mode link keeps the lower of the channel's and the fault mode's confidence."))


class FamilyWords(StrictModel):
    """A template family and the mechanism word prefixes that select it."""

    family: Literal["accumulation", "arrhenius", "bearing_fatigue"]
    words: tuple[NonEmptyStr, ...]


class IntervalPrior(StrictModel):
    """A lognormal prior on life T from a lower-bound interval I: P(T < I) = q, log-sd sigma.

    The median is I * exp(-z_q * sigma), z_q the standard normal q-quantile."""

    q: float = Field(gt=0.0, lt=0.5)
    sigma: float = Field(gt=0.0)
    why: NonEmptyStr


class DutyDefault(StrictModel):
    """A wide default duty cycle (time units per calendar year) for a calendar-only interval (flagged)."""

    per_year_median: float = Field(gt=0.0)
    sigma: float = Field(gt=0.0)
    why: NonEmptyStr


class ThroughputDefault(StrictModel):
    """A wide default for an amount processed per time unit (e.g. water per product), when machine.toml [[spec]]
    declares none and no document value was extracted; flagged."""

    median: float = Field(gt=0.0)
    unit: NonEmptyStr = Field(description="The amount's unit (per one time unit), e.g. 'L'.")
    sigma: float = Field(gt=0.0)
    why: NonEmptyStr


class FPrior(StrictModel):
    """A literature or standard prior on a template f parameter."""

    median: float
    sigma: float = Field(gt=0.0, description="log-sd for a lognormal, sd for a truncated normal.")
    unit: NonEmptyStr
    source: Literal["standard", "handbook", "literature", "default"]
    ref: NonEmptyStr


class ParameterizeSettings(BaseModel):
    """Parameterize (M9, approved 2026-11-09): one template family per full-model component, then priors."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    families: tuple[FamilyWords, ...] = Field(default=(
        FamilyWords(family="bearing_fatigue", words=("flaking", "spalling", "fatigue", "pitting")),
        FamilyWords(family="arrhenius", words=("overheat", "ageing", "aging", "thermal", "oxidat", "insulation")),
        FamilyWords(family="accumulation", words=("clog", "scal", "residue", "deposit", "foul", "exhaust", "wear",
                                                  "block", "contamin", "buildup")),
    ), description=(
        "Strongest first. A component gets the first family any of its counted fault modes' mechanism or name "
        "words start with: a bearing with flaking among wear and contamination is bearing_fatigue. Other modes "
        "are listed as not separately modelled (one template per component, approved 2026-11-09)."))
    fallback_family: Literal["accumulation", "arrhenius", "bearing_fatigue"] = Field(default="accumulation", description=(
        "The family of a component no word matches; flagged for the review."))
    kind_words: dict[str, tuple[NonEmptyStr, ...]] = Field(default={
        "rated_life": ("l10", "rated", "rating"), "mtbf": ("mtbf",), "replacement_interval": ("replace", "change", "renew"),
        "inspection_interval": ("inspect", "check", "clean"), "service_interval": ("service", "maintenance", "overhaul")},
        description=("A life statement's kind: the first kind (in this order) any of its word prefixes appears in. The "
                     "action words come before 'service': 'Check air filter service indicator' is an inspection."))
    consumable_words: tuple[NonEmptyStr, ...] = Field(default=(
        "filter", "element", "oil", "fluid", "cartridge", "separator", "grease", "lubricant"), description=(
        "A replacement interval is a consumable's when a word of the component's name ends with one of these "
        "('Roto-injectfluid' is a fluid; 'boiler' does not end with 'oil')."))
    interval_priors: dict[str, IntervalPrior] = Field(default={
        "consumable_replacement": IntervalPrior(q=0.20, sigma=0.35, why=(
            "consumables are replaced at about their useful life with a small margin; some wear out early")),
        "replacement": IntervalPrior(q=0.05, sigma=0.6, why="preventive replacement leaves more margin"),
        "service": IntervalPrior(q=0.05, sigma=0.8, why="a conservative lower bound; says little about life"),
        "machine_wide": IntervalPrior(q=0.05, sigma=1.0, why="a machine-wide interval, not even about the part"),
    }, description="Approved 2026-11-09 (Natalie, M9 checkpoint 1); shown as assumptions in the sensitivity report.")
    rated_life_weibull_beta: float = Field(default=1.5, gt=0.0, description=(
        "ISO 281 life scatter: L10 is the 10% point of a Weibull with this shape; the prior is the lognormal with "
        "the same median and 10% point."))
    transcription_extra_sigma: float = Field(default=0.5, gt=0.0, description=(
        "Added (in quadrature) to the log-sd of a life prior whose statement comes from the transcription of a "
        "scanned page (Natalie 2026-11-09): the number may be misread."))
    max_probability_below_interval: float = Field(default=0.25, gt=0.0, lt=1.0, description=(
        "Gate: under a component's life prior, P(T < interval) may not exceed this for any of its lower bounds."))
    calendar_days: dict[str, float] = Field(default={
        "daily": 1.0, "weekly": 7.0, "monthly": 30.4375, "quarterly": 91.3125, "yearly": 365.25, "annually": 365.25,
        "day": 1.0, "week": 7.0, "month": 30.4375, "year": 365.25}, description=(
        "Calendar periods and units, in days ('2-yearly' doubles). Converted through a duty cycle, never by pint: "
        "a running hour is not a calendar hour."))
    judgement_samples: int = Field(default=3, ge=1, description="life_judgement samples per component.")
    judgement_sigma_floor: float = Field(default=1.2, gt=0.0, description=(
        "Minimum log-sd of a life_judgement prior (approved plan 2026-11-09: wide, a last resort)."))
    judgement_confidence_base: float = Field(default=0.3, gt=0.0, le=1.0, description=(
        "Confidence of a life_judgement prior when its samples agree exactly; it falls as exp(-sd of the samples' "
        "log medians), so disagreement lowers it (computed by code, rule 15)."))
    quantity_units: dict[str, NonEmptyStr] = Field(default={"volume": "L", "mass": "kg"}, description=(
        "An amount-processed life (a filter's capacity) is pooled in this unit of its quantity."))
    throughput_defaults: dict[str, ThroughputDefault] = Field(default={
        "volume_per_product": ThroughputDefault(median=0.12, unit="L", sigma=0.7, why=(
            "filtered water per product: espressos 25-40 ml (eversys p78, label 78: 25 ml each), coffees and hot "
            "water 120-250 ml, plus rinses and steam-boiler refill; no water-per-product setting was extracted")),
    }, description="Keyed '<quantity>_per_<time unit>'; machine.toml [[spec]] (component 'machine') replaces one.")
    declared_spec_sigma: float = Field(default=0.1, gt=0.0, description=(
        "Log-sd of a value declared in machine.toml [[spec]]: a stated value, nearly exact."))
    activation_energy: FPrior = Field(default=FPrior(
        median=80e3, sigma=0.35, unit="J/mol", source="literature", ref=(
            "thermal ageing of oils, greases and insulation: the '10 K halves life' rule near 100 °C, Ea about "
            "80 kJ/mol (IEC 60216 thermal endurance); wide")), description="arrhenius.activation_energy prior.")
    bearing_life_exponents: dict[str, float] = Field(default={"ball": 3.0, "roller": 10 / 3}, description=(
        "ISO 281:2007 basic rating life exponent p; 'roller' when the component name says roller, else ball."))
    bearing_exponent_sd: float = Field(default=0.05, gt=0.0, description="sd of the truncated-normal prior on p.")
    duty_defaults: dict[str, DutyDefault] = Field(default={
        "hour": DutyDefault(per_year_median=2000.0, sigma=0.7, why="about one shift a day; no duty cycle declared"),
        "product": DutyDefault(per_year_median=60000.0, sigma=0.8, why=(
            "about 165 products a day for a commercial machine; no duty cycle declared")),
    }, description="Per time unit, when machine.toml declares no duty cycle (Natalie 2026-11-09: wide, flagged).")


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
    batch_poll_max_errors: int = Field(default=10, ge=0, description="Transient poll errors in a row before a "
                                       "batch run stops cleanly (rerun to collect).")
    batch_max_wait_s: float = Field(default=86400.0, gt=0.0)
    request_timeout_s: float = Field(default=120.0, gt=0.0, description="Per request; prompts may override.")
    sdk_max_retries: int = Field(default=2, ge=0)
    stream_min_max_tokens: int = Field(default=8000, ge=1, description=(
        "Calls whose max_tokens is at least this are streamed, so a long generation keeps the connection busy."))
    call_deadline_s: float = Field(default=600.0, gt=0.0, description=(
        "Wall-clock limit per call, retries included; the call is abandoned (TransportError) after it."))

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
    scope: ScopeSettings = ScopeSettings()
    parameterize: ParameterizeSettings = ParameterizeSettings()

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
    kind: DocumentKind
    ruled_tables: RuledTablesRoute = Field(
        default="pymupdf", description="'llm': also send this document's table_ruled pages to the LLM layer "
        "(PyMuPDF loses the rows a merged cell spans).")

    @field_validator("kind", mode="before")
    @classmethod
    def _renamed_kind(cls, kind: object) -> object:
        """machine.toml files written before M7 renamed kind 'other' to 'literature' are read with the new name."""
        return LEGACY_KINDS.get(kind, kind) if isinstance(kind, str) else kind


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


class DeclaredComponent(StrictModel):
    """A component the user knows the machine has but its own documents don't name (e.g. the spindle bearings).

    It enters the MachineModel with source `deployment`, citing machine.toml, and is given to
    `understand.fault_modes` like an extracted one, so reference documents can attach mechanisms to it.
    """

    id: Identifier
    name: NonEmptyStr
    description: str
    category: NonEmptyStr
    parent: Identifier | None = None
    note: str | None = Field(default=None, description="Where the declaration comes from; noted in the report.")


class DeclaredScope(StrictModel):
    """[[scope]]: a scope decision the engineer sets for a component (an onboarding-review override, M8)."""

    component: Identifier
    decision: Decision
    reason: NonEmptyStr
    note: str | None = None


class DeclaredSpec(StrictModel):
    """[[spec]]: a value the engineer states (M9): a template input such as a bearing's load rating or nominal
    speed (component id), or a machine-level conversion such as volume_per_product (component 'machine')."""

    component: Identifier
    name: Identifier
    value: float
    unit: NonEmptyStr
    note: str | None = None


class DeclaredLife(StrictModel):
    """[[life]]: a life or interval the engineer states for a component (M8 counts it for L; M9 calibrates on it,
    with service and replacement intervals as lower bounds)."""

    component: Identifier
    kind: LifeKind
    value: float = Field(gt=0.0)
    unit: NonEmptyStr
    note: str | None = None


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
    instances: dict[Identifier, Annotated[int, Field(ge=1)]] = Field(
        default_factory=dict, description="[instances]: how many of a part the documentation lists per instance "
        "the machine has (e.g. spindle = 1); any part not stated has 1.")
    components: Annotated[tuple[DeclaredComponent, ...], by_attr("id")] = Field(
        default=(), description="[[components]]: parts the machine has that its own documents don't name."
    )
    scope: Annotated[tuple[DeclaredScope, ...], by_attr("component")] = Field(
        default=(), description="[[scope]]: scope decisions set by the engineer (decided_by 'override').")
    life: Annotated[tuple[DeclaredLife, ...], sorted_unique(lambda d: (d.component, d.kind), "life")] = Field(
        default=(), description="[[life]]: lives and intervals stated by the engineer, per component and kind.")
    spec: Annotated[tuple[DeclaredSpec, ...], sorted_unique(lambda d: (d.component, d.name), "spec")] = Field(
        default=(), description="[[spec]]: template inputs and conversions stated by the engineer (M9).")

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
