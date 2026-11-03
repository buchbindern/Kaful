"""Settings (kaful.toml + KAFUL_* env vars) and per-machine config (machine.toml)."""

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from kaful.artifacts.base import Identifier, NonEmptyStr, StrictModel, by_attr
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


class MachineConfig(StrictModel):
    """Contents of machines/<id>/machine.toml."""

    machine_id: Identifier
    display_name: NonEmptyStr
    time_unit: NonEmptyStr = Field(description="The one time unit every rate is expressed in.")
    documents: Annotated[tuple[DocumentEntry, ...], by_attr("path")] = ()


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
