"""kaful.llm.hashing matches M1/M2's prompt_hash; kaful.llm.pricing matches pricing.toml."""

import json
from pathlib import Path

import pytest

from kaful.llm.hashing import prompt_hash
from kaful.llm.pricing import PricingError, TokenUsage, load_pricing
from utils.prompt_hash import prompt_hash as legacy_prompt_hash

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "tiny_pump"
REQUEST = {
    "model": "claude-sonnet-4-6", "max_tokens": 100, "temperature": 0.2, "system": "Be brief.",
    "messages": [{"role": "user", "content": "List the channels."}],
}


def _text_only_fixtures() -> list[dict[str, object]]:
    records = [json.loads(p.read_text()) for p in sorted(FIXTURES.glob("*.json"))]
    # Phase-1 requests carry PDFs whose base64 is redacted in the fixture, so they can't be re-hashed.
    return [r for r in records if "data_sha256" not in json.dumps(r["request"])]


def test_hash_matches_every_recorded_text_fixture() -> None:
    records = _text_only_fixtures()
    assert len(records) >= 15
    for record in records:
        assert prompt_hash(record["request"]) == record["prompt_sha256"]  # type: ignore[arg-type]


def test_hash_matches_legacy_when_no_new_fields_are_present() -> None:
    assert prompt_hash(REQUEST) == legacy_prompt_hash(REQUEST)
    assert prompt_hash({**REQUEST, "stream": True, "timeout": 5}) == legacy_prompt_hash(REQUEST)


def test_new_output_fields_change_the_hash_but_cache_control_does_not() -> None:
    with_format = {**REQUEST, "output_config": {"format": {"type": "json_schema", "schema": {}}}}
    assert prompt_hash(with_format) != prompt_hash(REQUEST)
    cached = {**REQUEST, "system": [{"type": "text", "text": "Be brief.",
                                     "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}
    uncached = {**REQUEST, "system": [{"type": "text", "text": "Be brief."}]}
    assert prompt_hash(cached) == prompt_hash(uncached)
    assert prompt_hash({**cached, "cache_control": {"type": "ephemeral"}}) == prompt_hash(uncached)


def test_cost_uses_model_rates_and_multipliers() -> None:
    pricing = load_pricing()
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert pricing.cost("claude-sonnet-5-5", usage) == pytest.approx(2.0 + 10.0)
    assert pricing.cost("claude-sonnet-5-5", usage, batch=True) == pytest.approx(6.0)
    reads = TokenUsage(cache_read_tokens=1_000_000)
    assert pricing.cost("claude-sonnet-5-5", reads) == pytest.approx(0.10)  # 0.05x override
    assert pricing.cost("claude-sonnet-4-6", reads) == pytest.approx(0.30)  # standard 0.1x
    writes = TokenUsage(cache_write_5m_tokens=1_000_000, cache_write_1h_tokens=1_000_000)
    assert pricing.cost("claude-opus-5-5", writes) == pytest.approx(5.0 + 8.0)


def test_haiku_5_5_long_context_tier_counts_cached_tokens() -> None:
    pricing = load_pricing()
    short = TokenUsage(input_tokens=100_000, output_tokens=1_000_000)
    assert pricing.cost("claude-haiku-5-5", short) == pytest.approx(0.01 + 0.50)
    # 60k fresh + 50k cache reads = 110k prompt tokens: the whole request pays the higher tier.
    long = TokenUsage(input_tokens=60_000, cache_read_tokens=50_000, output_tokens=1_000_000)
    assert pricing.cost("claude-haiku-5-5", long) == pytest.approx(0.03 + 0.0025 + 2.50)


def test_worst_case_prices_input_as_cache_write_and_full_max_tokens() -> None:
    pricing = load_pricing()
    worst = pricing.worst_case_cost("claude-sonnet-5-5", 1_000_000, 100_000, cache_write="1h", batch=False)
    assert worst == pytest.approx(4.0 + 1.0)
    assert pricing.worst_case_cost("claude-sonnet-5-5", 1_000_000, 0, None, batch=True) == pytest.approx(1.0)


def test_unknown_model_is_refused() -> None:
    with pytest.raises(PricingError, match="no price"):
        load_pricing().cost("claude-imaginary-9", TokenUsage())
