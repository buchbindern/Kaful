"""
utils/pricing.py
----------------
Load pricing.toml and turn token counts into USD.

Usage:
    from utils.pricing import cost_usd
    cost = cost_usd("claude-sonnet-4-6", input_tokens=1000, output_tokens=200)
"""

import logging
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

PRICING_PATH = Path(__file__).resolve().parent.parent / "pricing.toml"
PER_TOKENS = 1_000_000

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def load_pricing(path: Path = PRICING_PATH) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def cost_usd(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_5m_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
    cache_read_tokens: int = 0,
    pricing: dict[str, Any] | None = None,
) -> float | None:
    """Cost of one call in USD, or None if the model is not in the table."""
    pricing = pricing or load_pricing()
    rates = pricing["models"].get(model)
    if rates is None:
        logger.warning("no price for model %r; cost recorded as null", model)
        return None
    m = pricing["multipliers"]
    in_rate = rates["input"]
    total = (
        input_tokens * in_rate
        + cache_write_5m_tokens * in_rate * m["cache_write_5m"]
        + cache_write_1h_tokens * in_rate * m["cache_write_1h"]
        + cache_read_tokens * in_rate * m["cache_read"]
        + output_tokens * rates["output"]
    )
    return total / PER_TOKENS


def pricing_as_of() -> str:
    return str(load_pricing()["as_of"])
