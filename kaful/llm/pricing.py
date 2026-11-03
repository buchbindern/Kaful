"""Prices from the repo-root `pricing.toml` (shared with the legacy ledger) and call cost.

The file is the single price table; see its header for sources. Costs are USD.
"""

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field

from kaful.artifacts.base import NonEmptyStr, StrictModel
from kaful.config import REPO_ROOT

PRICING_FILE = REPO_ROOT / "pricing.toml"
PER_TOKENS = 1_000_000

Price = float  # USD per million tokens
CacheTtl = Literal["5m", "1h"]


class PricingError(LookupError):
    """A model has no price, so the gateway refuses to call it."""


class Multipliers(StrictModel):
    cache_write_5m: float = Field(gt=0.0)
    cache_write_1h: float = Field(gt=0.0)
    cache_read: float = Field(gt=0.0)
    batch: float = Field(gt=0.0, le=1.0)


class LongContextPrice(StrictModel):
    """Prices for a request whose prompt is over `threshold_tokens` (all input tokens)."""

    threshold_tokens: int = Field(gt=0)
    input: Price = Field(ge=0.0)
    output: Price = Field(ge=0.0)


class ModelPrice(StrictModel):
    provider: Literal["anthropic", "openai"]
    input: Price = Field(ge=0.0)
    output: Price = Field(ge=0.0)
    cache_read_multiplier: float | None = Field(default=None, gt=0.0)
    long_context: LongContextPrice | None = None


class TokenUsage(StrictModel):
    """Billed token counts of one call. `input_tokens` excludes cache reads and writes."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_write_5m_tokens: int = Field(default=0, ge=0)
    cache_write_1h_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)

    @property
    def prompt_tokens(self) -> int:
        """Every input token, cached or not (what the long-context threshold counts)."""
        return (self.input_tokens + self.cache_write_5m_tokens + self.cache_write_1h_tokens
                + self.cache_read_tokens)


class PricingTable(StrictModel):
    as_of: NonEmptyStr
    multipliers: Multipliers
    models: dict[str, ModelPrice]

    def model(self, model: str) -> ModelPrice:
        price = self.models.get(model)
        if price is None:
            raise PricingError(f"no price for model {model!r} in pricing.toml; refusing to call it")
        return price

    def rates(self, model: str, prompt_tokens: int) -> tuple[Price, Price]:
        """(input, output) price per million tokens for a prompt of this size."""
        price = self.model(model)
        tier = price.long_context
        if tier is not None and prompt_tokens > tier.threshold_tokens:
            return tier.input, tier.output
        return price.input, price.output

    def cost(self, model: str, usage: TokenUsage, batch: bool = False) -> float:
        """USD cost of one call."""
        in_rate, out_rate = self.rates(model, usage.prompt_tokens)
        m = self.multipliers
        read = self.model(model).cache_read_multiplier or m.cache_read
        total = (
            usage.input_tokens * in_rate
            + usage.cache_write_5m_tokens * in_rate * m.cache_write_5m
            + usage.cache_write_1h_tokens * in_rate * m.cache_write_1h
            + usage.cache_read_tokens * in_rate * read
            + usage.output_tokens * out_rate
        )
        return total * (m.batch if batch else 1.0) / PER_TOKENS

    def worst_case_cost(
        self, model: str, input_tokens: int, max_tokens: int, cache_write: CacheTtl | None, batch: bool
    ) -> float:
        """Upper bound on one call: every input token written to cache (if caching) and
        the full `max_tokens` generated."""
        if cache_write == "5m":
            usage = TokenUsage(cache_write_5m_tokens=input_tokens, output_tokens=max_tokens)
        elif cache_write == "1h":
            usage = TokenUsage(cache_write_1h_tokens=input_tokens, output_tokens=max_tokens)
        else:
            usage = TokenUsage(input_tokens=input_tokens, output_tokens=max_tokens)
        return self.cost(model, usage, batch)


@lru_cache(maxsize=None)
def load_pricing(path: Path = PRICING_FILE) -> PricingTable:
    with path.open("rb") as handle:
        return PricingTable.model_validate(tomllib.load(handle))
