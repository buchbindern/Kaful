"""M1 baseline report aggregation (pure, offline)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import m1_baseline_report as rep  # noqa: E402
from utils.pricing import cost_usd  # noqa: E402

SONNET = "claude-sonnet-4-6"


def _call(phase: str, step: str, inp: int, out: int, component: str | None = None,
          model: str = SONNET) -> dict:
    return {"phase": phase, "step": step, "component": component, "model": model,
            "input_tokens": inp, "output_tokens": out, "output_basis": "exact", "input_basis": "count_tokens"}


def test_step_rows_sum_llm_and_embedding_costs() -> None:
    calls = [_call("phase2", "step_c_extract", 1000, 100), _call("phase2", "step_c_extract", 1000, 100)]
    emb = [{"phase": "phase2", "step": "step_c_extract", "model": "text-embedding-3-large",
            "n_texts": 2, "chars": 4000}]
    [row] = rep.step_rows(calls, emb)
    llm = 2 * cost_usd(SONNET, input_tokens=1000, output_tokens=100)
    assert row["calls"] == 2 and row["embed_requests"] == 2
    assert row["llm_cost"] == pytest.approx(llm)
    assert row["cost"] == pytest.approx(llm + cost_usd("text-embedding-3-large", input_tokens=1000))


def test_scenarios_exclude_phase1_from_rerun() -> None:
    rows = rep.step_rows([_call("phase1", "ingest", 10_000, 1000), _call("phase3", "step_b_plan", 5000, 500)], [])
    s = rep.scenario_totals(rows)
    assert s["cold"] == pytest.approx(s["from_phase2"] + cost_usd(SONNET, input_tokens=10_000, output_tokens=1000))


def test_missing_phase1_outputs_use_ratio_and_cap() -> None:
    recs = [{"input_tokens": 1000, "output_tokens": None}, {"input_tokens": 100_000, "output_tokens": None}]
    rep.fill_missing_outputs(recs, (0.1, 0.2, 0.3))
    assert recs[0]["output_tokens_range"] == [100, 200, 300]
    assert recs[1]["output_tokens"] == rep.PDF_MAX_TOKENS
    assert "estimated" in recs[0]["output_basis"]


def test_output_ratio_quartiles_ignore_unmeasured_chunks() -> None:
    recs = [{"input_tokens": 100, "output_tokens": o} for o in (10, 20, 30, 40, 50)]
    recs.append({"input_tokens": 100, "output_tokens": None})
    lo, mid, hi = rep.output_ratio_quartiles(recs)
    assert lo < mid == 0.3 < hi


def test_component_rows_and_retry_extra() -> None:
    calls = [_call("phase6", "step_b_generate", 2000, 800, "pump"),
             _call("phase6", "step_b_generate", 2000, 800, "boiler"),
             _call("phase6", "step_a_spec", 1000, 500, "pump"),
             _call("phase5", "step_b_triage", 1000, 500)]
    rows = rep.component_rows(calls)
    assert {(r["step"], r["component"]) for r in rows} == {
        ("step_b_generate", "pump"), ("step_b_generate", "boiler"), ("step_a_spec", "pump")}
    one = cost_usd(SONNET, input_tokens=2000, output_tokens=800)
    assert rep.codegen_retry_extra(calls) == pytest.approx(2 * one * (rep.CODEGEN_MAX_ATTEMPTS - 1))


def test_stand_in_low_bound_shrinks_retrieved_share_only() -> None:
    call = {**_call("phase4", "step_b_physics", 10_000, 500, "pump"),
            "input_basis": "count_tokens (BM25 stand-in retrieval)", "request_chars": 40_000}
    other = {**_call("phase5", "step_b_triage", 10_000, 500), "request_chars": 40_000}
    retrievals = [{"phase": "phase4", "step": "step_b_physics", "component": "pump",
                   "hits": [["a", 15_000], ["b", 15_000]], "corpus_mean_chars": 5_000}]
    rep.stand_in_low_bound([call, other], retrievals)
    # 30k of 40k chars retrieved; typical size would give 10k, so the retrieved part shrinks by 2/3.
    assert call["input_tokens_low"] == round(10_000 * (1 - 0.75 * (2 / 3)))
    assert "input_tokens_low" not in other


def test_plain_rerun_phase1_counts_only_unindexed_chunks() -> None:
    indexed = {**_call("phase1", "ingest", 1000, 200), "output_basis": "count_tokens(stored extraction)"}
    missing = {**_call("phase1", "ingest", 1000, 200), "output_basis": "estimated (ratio)"}
    assert rep.unindexed_phase1_cost([indexed, missing]) == pytest.approx(rep.price(missing))
