"""
scripts/m1_baseline_report.py
-----------------------------
Build docs/m1_cost_baseline.md from the per-call records written by
scripts/m1_estimate_baseline.py. Offline; prices come from pricing.toml.

Usage:
    python -B scripts/m1_baseline_report.py eversys_coffee_machine cnc_spindle
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.ledger import EMBEDDING_CHARS_PER_TOKEN  # noqa: E402
from utils.pricing import cost_usd, pricing_as_of  # noqa: E402

DATA_DIR = ROOT / "docs" / "m1_baseline"
REPORT = ROOT / "docs" / "m1_cost_baseline.md"
PDF_MAX_TOKENS = 8192            # utils/rag.py extraction max_tokens
CODEGEN_MAX_ATTEMPTS = 3         # phase6 step_b: 1 try + 2 syntax retries
PER_COMPONENT_STEPS = ("step_b_physics", "step_a_spec", "step_b_generate")
STEP_LABELS = {
    ("phase1", "ingest"): "1 ingest (PDF extraction)",
    ("phase2", "step_a_retrieve"): "2a retrieve", ("phase2", "step_b_context"): "2b context",
    ("phase2", "step_c_extract"): "2c extract (5 runs)",
    ("phase3", "step_a_retrieve"): "3a retrieve", ("phase3", "step_b_plan"): "3b plan",
    ("phase3", "step_c_codegen"): "3c simulator codegen",
    ("phase4", "step_a_comprehend"): "4a comprehend", ("phase4", "step_b_physics"): "4b physics",
    ("phase4", "step_c_resolve"): "4c unit resolution",
    ("phase5", "step_a_assign"): "5a assign", ("phase5", "step_b_triage"): "5b triage",
    ("phase6", "step_a_spec"): "6a spec", ("phase6", "step_b_generate"): "6b component code",
    ("phase6", "step_d_composite"): "6d composite",
}


# ── Loading and pricing ───────────────────────────────────────────────────────

def load(machine: str) -> dict[str, Any]:
    return json.loads((DATA_DIR / f"{machine}_calls.json").read_text())


def output_ratio_quartiles(records: list[dict[str, Any]]) -> tuple[float, float, float]:
    """Output/input token ratio of phase-1 chunks whose extraction is stored."""
    ratios = sorted(r["output_tokens"] / r["input_tokens"] for r in records
                    if r.get("output_tokens") is not None and r["input_tokens"])
    q = statistics.quantiles(ratios, n=4)
    return q[0], statistics.median(ratios), q[2]


def fill_missing_outputs(records: list[dict[str, Any]], ratios: tuple[float, float, float]) -> None:
    """Estimate output for chunks with no stored extraction (low/central/high)."""
    for r in records:
        if r.get("output_tokens") is None:
            est = [min(round(r["input_tokens"] * k), PDF_MAX_TOKENS) for k in ratios]
            r["output_tokens_range"] = est
            r["output_tokens"] = est[1]
            r["output_basis"] = "estimated (output/input ratio of stored chunks)"


def price(r: dict[str, Any], low: bool = False) -> float:
    inp = r.get("input_tokens_low", r["input_tokens"]) if low else r["input_tokens"]
    return cost_usd(r["model"], input_tokens=inp, output_tokens=r["output_tokens"]) or 0.0


def stand_in_low_bound(calls: list[dict[str, Any]], retrievals: list[dict[str, Any]]) -> None:
    """Set input_tokens_low on calls whose context came from the BM25 stand-in.

    BM25 favours long chunks. The low bound assumes the real retrieval returned the
    same number of distinct chunks at the store's mean chunk size.
    """
    for c in calls:
        if "stand-in" not in c.get("input_basis", ""):
            continue
        recs = [e for e in retrievals if (e["phase"], e["step"]) == (c["phase"], c["step"]) and "hits" in e]
        recs = [e for e in recs if e["component"] == c["component"]] or \
               [e for e in recs if e["component"] is None]
        hits = {cid: n for e in recs for cid, n in e["hits"]}
        if not hits:
            continue
        retrieved = sum(hits.values())
        corpus_mean = recs[0]["corpus_mean_chars"]
        shrink = max(0.0, 1 - corpus_mean * len(hits) / retrieved)
        share = min(retrieved / c["request_chars"], 1.0)
        c["input_tokens_low"] = round(c["input_tokens"] * (1 - share * shrink))


def embedding_cost(e: dict[str, Any]) -> float:
    return cost_usd(e["model"], input_tokens=round(e["chars"] / EMBEDDING_CHARS_PER_TOKEN)) or 0.0


def index_embeddings(phase1: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Phase 1 embeds each stored extraction with text-embedding-3-large."""
    return [{"phase": "phase1", "step": "ingest", "component": None, "model": "text-embedding-3-large",
             "n_texts": 1, "chars": r["output_chars"] or round(r["output_tokens"] * EMBEDDING_CHARS_PER_TOKEN)}
            for r in phase1]


# ── Aggregation ───────────────────────────────────────────────────────────────

def step_rows(calls: list[dict[str, Any]], embeddings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (phase, step): calls, tokens, LLM cost, embedding cost."""
    rows: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {
        "calls": 0, "input_tokens": 0, "output_tokens": 0, "llm_cost": 0.0, "llm_cost_low": 0.0,
        "embed_requests": 0, "embed_cost": 0.0, "models": set(), "bases": set()})
    for r in calls:
        row = rows[(r["phase"], r["step"])]
        row["calls"] += 1
        row["input_tokens"] += r["input_tokens"]
        row["output_tokens"] += r["output_tokens"]
        row["llm_cost"] += price(r)
        row["llm_cost_low"] += price(r, low=True)
        row["models"].add(r["model"])
        row["bases"].add(r["output_basis"])
        if "stand-in" in r.get("input_basis", ""):
            row["bases"].add("input: BM25 stand-in retrieval")
    for e in embeddings:
        row = rows[(e["phase"], e["step"])]
        row["embed_requests"] += e["n_texts"]
        row["embed_cost"] += embedding_cost(e)
    out = [{"phase": k[0], "step": k[1], **v, "cost": v["llm_cost"] + v["embed_cost"],
            "cost_low": v["llm_cost_low"] + v["embed_cost"]}
           for k, v in rows.items()]
    return sorted(out, key=lambda r: (r["phase"], r["step"]))


def component_rows(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0,
                                                                       "cost": 0.0})
    for r in calls:
        if r["step"] in PER_COMPONENT_STEPS:
            row = rows[(r["step"], r["component"])]
            row["input_tokens"] += r["input_tokens"]
            row["output_tokens"] += r["output_tokens"]
            row["cost"] += price(r)
    return [{"step": k[0], "component": k[1], **v} for k, v in sorted(rows.items())]


def scenario_totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = sum(r["cost"] for r in rows)
    low = sum(r.get("cost_low", r["cost"]) for r in rows)
    phase1 = sum(r["cost"] for r in rows if r["phase"] == "phase1")
    return {"cold": total, "from_phase2": total - phase1, "cold_low": low, "from_phase2_low": low - phase1}


def phase1_range(phase1: list[dict[str, Any]]) -> tuple[float, float]:
    """Phase-1 LLM cost with estimated outputs at their low and high ratios."""
    lo = hi = 0.0
    for r in phase1:
        o_lo, _, o_hi = r.get("output_tokens_range", [r["output_tokens"]] * 3)
        lo += cost_usd(r["model"], input_tokens=r["input_tokens"], output_tokens=o_lo) or 0.0
        hi += cost_usd(r["model"], input_tokens=r["input_tokens"], output_tokens=o_hi) or 0.0
    return lo, hi


def unindexed_phase1_cost(phase1: list[dict[str, Any]]) -> float:
    """Phase 1 without --from extracts every chunk whose ID is not indexed yet.

    Chunk IDs are a running counter, so this equals the chunks with no stored
    extraction (checked for cnc_spindle: indexed IDs 000-047 are the first 48 chunks).
    """
    return sum(price(r) for r in phase1 if "estimated" in r["output_basis"])


def codegen_retry_extra(calls: list[dict[str, Any]]) -> float:
    """Extra cost if every 6b component needed all attempts (failed attempts are not saved)."""
    return sum(price(r) for r in calls if r["step"] == "step_b_generate") * (CODEGEN_MAX_ATTEMPTS - 1)


def analyse(machine: str, ratios: tuple[float, float, float]) -> dict[str, Any]:
    data = load(machine)
    fill_missing_outputs(data["phase1"], ratios)
    stand_in_low_bound(data["calls"], data["embeddings"])
    calls = data["phase1"] + data["calls"]
    embeddings = index_embeddings(data["phase1"]) + data["embeddings"]
    rows = step_rows(calls, embeddings)
    return {
        "machine": machine, "rows": rows, "components": component_rows(data["calls"]),
        "scenarios": scenario_totals(rows), "phase1_range": phase1_range(data["phase1"]),
        "retry_extra": codegen_retry_extra(data["calls"]), "phase1": data["phase1"],
        "plain_rerun_phase1": unindexed_phase1_cost(data["phase1"]),
        "calls": data["calls"],
    }


def top_drivers(rows: list[dict[str, Any]], n: int = 3) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: r["cost"], reverse=True)[:n]


# ── Markdown ──────────────────────────────────────────────────────────────────

def usd(x: float) -> str:
    return f"${x:,.2f}" if x >= 0.01 else f"${x:,.4f}"


def ktok(n: int) -> str:
    return f"{n / 1000:,.1f}k"


def label(r: dict[str, Any]) -> str:
    return STEP_LABELS.get((r["phase"], r["step"]), f"{r['phase']} {r['step']}")


def summary_table(analyses: list[dict[str, Any]]) -> list[str]:
    lines = ["| Machine | Cold run (incl. phase 1) | Rerun `--from phase2` | Phase 1 alone | "
             "Phase-1 range* | Plain rerun (no `--from`): phase 1 | +6b if all retries** |",
             "|---|---|---|---|---|---|---|"]
    for a in analyses:
        s, (lo, hi) = a["scenarios"], a["phase1_range"]
        p1 = s["cold"] - s["from_phase2"]
        lines.append(f"| {a['machine']} | **{usd(s['cold'])}** (low {usd(s['cold_low'])}) | "
                     f"**{usd(s['from_phase2'])}** (low {usd(s['from_phase2_low'])}) | {usd(p1)} | "
                     f"{usd(lo)} – {usd(hi)} | {usd(a['plain_rerun_phase1'])} | +{usd(a['retry_extra'])} |")
    return lines


def step_table(a: dict[str, Any]) -> list[str]:
    cold, rerun = a["scenarios"]["cold"], a["scenarios"]["from_phase2"]
    lines = ["| Step | Calls | Input tok | Output tok | Model | Cost | Low*** | % of cold | % of rerun | Basis |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in a["rows"]:
        share_rerun = "–" if r["phase"] == "phase1" else f"{100 * r['cost'] / rerun:.1f}%"
        models = ", ".join(sorted(m.replace("claude-", "") for m in r["models"])) or "embeddings only"
        calls = f"{r['calls']}" + (f" (+{r['embed_requests']} emb)" if r["embed_requests"] else "")
        lines.append(f"| {label(r)} | {calls} | {ktok(r['input_tokens'])} | {ktok(r['output_tokens'])} | "
                     f"{models} | {usd(r['cost'])} | {usd(r['cost_low']) if r['cost_low'] < r['cost'] - 1e-9 else ''} | "
                     f"{100 * r['cost'] / cold:.1f}% | {share_rerun} | "
                     f"{'; '.join(sorted(r['bases'])) or 'embedding (chars/4)'} |")
    lines.append(f"| **Total** | | | | | **{usd(cold)}** | {usd(a['scenarios']['cold_low'])} | 100% | | |")
    return lines


def component_table(a: dict[str, Any]) -> list[str]:
    steps = [s for s in PER_COMPONENT_STEPS if any(c["step"] == s for c in a["components"])]
    comps = sorted({c["component"] for c in a["components"]})
    cost = {(c["step"], c["component"]): c["cost"] for c in a["components"]}
    head = " | ".join(STEP_LABELS[(("phase4" if s == "step_b_physics" else "phase6"), s)] for s in steps)
    lines = [f"| Component | {head} | Total |", "|---|" + "---|" * (len(steps) + 1)]
    for comp in comps:
        vals = [cost.get((s, comp)) for s in steps]
        cells = " | ".join(usd(v) if v is not None else "–" for v in vals)
        lines.append(f"| {comp} | {cells} | {usd(sum(v for v in vals if v))} |")
    per = [sum(cost.get((s, c), 0) for s in steps) for c in comps]
    lines.append(f"| **Mean per component** | " + " | ".join(
        usd(statistics.mean(cost[(s, c)] for c in comps if (s, c) in cost)) for s in steps)
        + f" | {usd(statistics.mean(per))} |")
    return lines


def drivers(a: dict[str, Any]) -> list[str]:
    cold, rerun = a["scenarios"]["cold"], a["scenarios"]["from_phase2"]
    out = [f"Cold run: " + "; ".join(f"{i}. {label(r)} {usd(r['cost'])} ({100 * r['cost'] / cold:.0f}%)"
                                     for i, r in enumerate(top_drivers(a["rows"]), 1))]
    rows2 = [r for r in a["rows"] if r["phase"] != "phase1"]
    out.append("Rerun `--from phase2`: " + "; ".join(
        f"{i}. {label(r)} {usd(r['cost'])} ({100 * r['cost'] / rerun:.0f}%)"
        for i, r in enumerate(top_drivers(rows2), 1)))
    return out


def phase1_facts(a: dict[str, Any]) -> str:
    p1 = a["phase1"]
    pages = sum(r["pages"][1] - r["pages"][0] + 1 for r in p1)
    tin = sum(r["input_tokens"] for r in p1)
    known = [r for r in p1 if "estimated" not in r["output_basis"]]
    src = p1[0].get("chunk_source", "rechunk")
    return (f"{len(p1)} chunks, {pages} pages ({src} chunks), {tin / pages:,.0f} input tokens/page "
            f"measured; {len(known)}/{len(p1)} chunk outputs measured from stored extractions.")


def render(analyses: list[dict[str, Any]], ratios: tuple[float, float, float], notes: str) -> str:
    out = ["# M1 cost baseline (legacy pipeline)", "",
           f"Prices: `pricing.toml` (as of {pricing_as_of()}). Token counts: Anthropic `count_tokens` "
           "on the reconstructed requests (see Method). Generated by `scripts/m1_baseline_report.py`.", "",
           "## Summary (USD per run)", "", *summary_table(analyses), "",
           f"\\* Phase-1 cost with estimated chunk outputs at the 25th/75th percentile output/input ratio "
           f"({ratios[0]:.2f} / {ratios[2]:.2f}; central {ratios[1]:.2f}). Equal to the point value when "
           "every chunk output is measured.  ",
           "\\*\\* Failed 6b attempts are not saved; this is the extra cost if every component used all "
           f"{CODEGEN_MAX_ATTEMPTS} attempts.  ",
           "\\*\\*\\* Steps whose retrieved context was not saved (4a, 4b, 4c, and the ProgPy context in "
           "6b/6d) are re-rendered with a local BM25 stand-in, which favours long chunks. The bold figure "
           "uses the stand-in's chunks; \"low\" assumes the same number of chunks at the store's mean size.", ""]
    for a in analyses:
        out += [f"## {a['machine']}", "", f"Phase 1: {phase1_facts(a)}", "", "### Per step", "",
                *step_table(a), "", "### Per component (4b, 6a, 6b)", "", *component_table(a), "",
                "### Top cost drivers", "", *[f"- {d}" for d in drivers(a)], ""]
    return "\n".join(out + [notes])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("machines", nargs="+")
    args = parser.parse_args()
    phase1 = [r for m in args.machines for r in load(m)["phase1"]]
    ratios = output_ratio_quartiles(phase1)
    analyses = [analyse(m, ratios) for m in args.machines]
    notes = (DATA_DIR / "notes.md").read_text() if (DATA_DIR / "notes.md").exists() else ""
    REPORT.write_text(render(analyses, ratios, notes))
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
