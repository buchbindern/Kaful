"""
utils/audit_chunks.py
---------------------
Audit indexed manual chunks for extractor-refusal / meta-commentary pollution.

Two sources, both scanned by default:
  1. The live Chroma collections at machines/<id>/rag/vector_db
  2. The committed pipeline outputs (outputs/schema/step_a_chunks.json,
     outputs/simulate/step_a_context.json)

Reports, per machine and per source PDF:
  - total chunks
  - POISONED  : document opens with extractor meta-commentary / refusal
  - SHORT     : suspiciously short document (likely truncated or a refusal)
  - EMPTY     : no document text at all
  - the inferred printed-vs-physical page offset, where the extractor
    stated it explicitly

Writes a JSON report with the full poisoned chunk-id list so you can see the
blast radius before deciding whether to re-ingest.

Usage:
    python utils/audit_chunks.py                    # all machines
    python utils/audit_chunks.py cnc_spindle cdv    # specific machines
    python utils/audit_chunks.py --json report.json
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MACHINES_DIR = ROOT / "machines"

# Phrases the extractor uses when it argues with the prompt instead of
# extracting. Matched against the HEAD of the document only — these words can
# legitimately appear deep inside a long technical extraction.
REFUSAL_MARKERS = (
    "i notice there",
    "i notice that",
    "i need to note",
    "i should note",
    "discrepancy in your request",
    "discrepancy between",
    "the document provided only contains",
    "the document provided contains",
    "the pages provided",
    "but the document",
    "i don't see pages",
    "i do not see pages",
    "i cannot extract",
    "i can't extract",
    "i'm unable to",
    "i am unable to",
    "let me extract all the",
    "i'll extract all the",
    "there appears to be a mismatch",
)

HEAD_CHARS = 400
MIN_DOC_CHARS = 500

# "...only contains pages 518-525", "...actually provided (568-575)"
OFFSET_RE = re.compile(
    r"(?:only contains|actually (?:provided|contains?|contained)|provided (?:are|is)|contains)"
    r"[^\n\d]{0,20}pages?\s*\(?\s*(\d{1,4})\s*[-–—]\s*(\d{1,4})",
    re.IGNORECASE,
)


# ── Classification ────────────────────────────────────────────────────────────

def classify(text: str) -> str:
    """Return one of: 'ok', 'poisoned', 'short', 'empty'."""
    if not text or not text.strip():
        return "empty"
    head = text[:HEAD_CHARS].lower()
    if any(marker in head for marker in REFUSAL_MARKERS):
        return "poisoned"
    if len(text.strip()) < MIN_DOC_CHARS:
        return "short"
    return "ok"


def stated_pages(text: str):
    """Pages the extractor claims it actually saw, if it said so. Else None."""
    m = OFFSET_RE.search(text[:1500])
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def page_offset(requested_start, text: str):
    """physical_index - printed_number, if both are known."""
    actual = stated_pages(text)
    if actual is None or requested_start is None:
        return None
    try:
        return int(requested_start) - actual[0]
    except (TypeError, ValueError):
        return None


# ── Sources ───────────────────────────────────────────────────────────────────

class _NoEmbed:
    """Stub EF so get_collection() never downloads a default embedding model.

    We only call collection.get(), which does not embed anything.
    """

    def __call__(self, input):  # noqa: A002 - chroma requires this signature
        raise RuntimeError("audit_chunks does not embed; query() is not supported here")

    def name(self):
        return "audit-noop"


def scan_chroma(machine_id: str) -> list[dict]:
    """Read every document + metadata out of the machine's Chroma collection."""
    rag_dir = MACHINES_DIR / machine_id / "rag" / "vector_db"
    if not rag_dir.exists():
        return []

    try:
        import chromadb
    except ImportError:
        print("  ! chromadb not installed — skipping live collection", file=sys.stderr)
        return []

    collection_name = f"{machine_id}_manual"
    try:
        client = chromadb.PersistentClient(path=str(rag_dir))
        collection = client.get_collection(name=collection_name)   # reuse persisted EF
        payload = collection.get(include=["documents", "metadatas"])  # no embedding call
    except Exception as exc:
        print(f"  ! could not open collection '{collection_name}': {exc}", file=sys.stderr)
        return []

    records = []
    ids = payload.get("ids") or []
    docs = payload.get("documents") or []
    metas = payload.get("metadatas") or []

    for i, chunk_id in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        meta = meta or {}
        records.append({
            "chunk_id": chunk_id,
            "text": docs[i] if i < len(docs) else "",
            "start_page": meta.get("start_page"),
            "end_page": meta.get("end_page"),
            "heading": meta.get("heading", ""),
            "pdf_file": meta.get("pdf_file", "?"),
            "source_type": meta.get("source_type", "?"),
            "source": "chroma",
        })
    return records


def _walk_chunks(obj):
    """Yield every dict with a 'text' key, at any depth."""
    if isinstance(obj, dict):
        if isinstance(obj.get("text"), str):
            yield obj
        for value in obj.values():
            yield from _walk_chunks(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_chunks(value)


def scan_outputs(machine_id: str) -> list[dict]:
    """Read chunks out of the committed pipeline output JSON."""
    machine_dir = MACHINES_DIR / machine_id / "outputs"
    targets = [
        machine_dir / "schema" / "step_a_chunks.json",
        machine_dir / "simulate" / "step_a_context.json",
    ]

    records = []
    for path in targets:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"  ! {path.name} is not valid JSON: {exc}", file=sys.stderr)
            continue

        for chunk in _walk_chunks(data):
            start = chunk.get("start_page")
            end = chunk.get("end_page")
            pages = chunk.get("pages")
            if start is None and isinstance(pages, str) and "-" in pages:
                head, _, tail = pages.partition("-")
                start = head.strip() if head.strip().isdigit() else None
                end = tail.strip() if tail.strip().isdigit() else None

            records.append({
                "chunk_id": chunk.get("chunk_id", "?"),
                "text": chunk.get("text", ""),
                "start_page": start,
                "end_page": end,
                "heading": chunk.get("heading", ""),
                "pdf_file": chunk.get("pdf_file", "?"),
                "source_type": chunk.get("source_type", "?"),
                "source": path.name,
            })
    return records


# ── Reporting ─────────────────────────────────────────────────────────────────

def audit_machine(machine_id: str) -> dict:
    records = scan_chroma(machine_id) + scan_outputs(machine_id)

    # A chunk can appear in both Chroma and the outputs; count it once,
    # preferring the live collection.
    deduped = {}
    for record in records:
        key = (record["chunk_id"], record["source"] != "chroma")
        deduped.setdefault(key, record)
    seen_ids = set()
    unique = []
    for (chunk_id, _), record in sorted(deduped.items(), key=lambda kv: kv[0][1]):
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        unique.append(record)

    verdicts = Counter()
    by_pdf = defaultdict(Counter)
    offsets = defaultdict(Counter)
    poisoned = []

    for record in unique:
        verdict = classify(record["text"])
        verdicts[verdict] += 1
        by_pdf[record["pdf_file"]][verdict] += 1

        if verdict in ("poisoned", "short", "empty"):
            offset = page_offset(record["start_page"], record["text"])
            if offset is not None:
                offsets[record["pdf_file"]][offset] += 1
            poisoned.append({
                "chunk_id": record["chunk_id"],
                "verdict": verdict,
                "pdf_file": record["pdf_file"],
                "pages": f"{record['start_page']}-{record['end_page']}",
                "stated_pages": stated_pages(record["text"]),
                "page_offset": offset,
                "heading": (record["heading"] or "")[:80],
                "source": record["source"],
                "head": record["text"][:160].replace("\n", " "),
            })

    return {
        "machine_id": machine_id,
        "total": len(unique),
        "verdicts": dict(verdicts),
        "by_pdf": {k: dict(v) for k, v in by_pdf.items()},
        "page_offsets": {k: dict(v) for k, v in offsets.items()},
        "bad_chunks": poisoned,
    }


def print_report(report: dict) -> None:
    total = report["total"]
    verdicts = report["verdicts"]
    bad = total - verdicts.get("ok", 0)
    rate = (bad / total * 100) if total else 0.0

    print(f"\n{'=' * 60}")
    print(f"{report['machine_id']}  —  {total} chunks, {bad} bad ({rate:.1f}%)")
    print(f"{'=' * 60}")

    if not total:
        print("  (nothing indexed and no committed outputs found)")
        return

    for verdict in ("ok", "poisoned", "short", "empty"):
        if verdicts.get(verdict):
            print(f"  {verdict:<9} {verdicts[verdict]:>4}")

    print("\n  per source PDF:")
    for pdf, counts in sorted(report["by_pdf"].items()):
        pdf_total = sum(counts.values())
        pdf_bad = pdf_total - counts.get("ok", 0)
        flag = "  <-- re-ingest" if pdf_bad else ""
        print(f"    {pdf:<44} {pdf_bad:>3}/{pdf_total:<4} bad{flag}")

    if report["page_offsets"]:
        print("\n  inferred physical-minus-printed page offset:")
        for pdf, counts in sorted(report["page_offsets"].items()):
            summary = ", ".join(
                f"{offset:+d} (x{n})" for offset, n in sorted(counts.items())
            )
            verdict = "CONSTANT" if len(counts) == 1 else "INCONSISTENT"
            print(f"    {pdf:<44} {summary}   [{verdict}]")

    if report["bad_chunks"]:
        print("\n  worst offenders:")
        for chunk in report["bad_chunks"][:8]:
            print(f"    {chunk['chunk_id']:<34} {chunk['verdict']:<9} p{chunk['pages']}")
            print(f"      {chunk['head'][:100]}...")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("machines", nargs="*", help="machine ids (default: all)")
    parser.add_argument("--json", metavar="PATH", help="write full JSON report here")
    args = parser.parse_args()

    if args.machines:
        machine_ids = args.machines
    else:
        machine_ids = sorted(
            p.name for p in MACHINES_DIR.iterdir()
            if p.is_dir() and not p.name.startswith((".", "_"))
        )

    reports = []
    for machine_id in machine_ids:
        report = audit_machine(machine_id)
        reports.append(report)
        print_report(report)

    grand_total = sum(r["total"] for r in reports)
    grand_bad = sum(r["total"] - r["verdicts"].get("ok", 0) for r in reports)
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {grand_bad}/{grand_total} chunks bad "
          f"({(grand_bad / grand_total * 100) if grand_total else 0:.1f}%)")
    print(f"{'=' * 60}\n")

    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=2))
        print(f"Full report → {args.json}")

    return 1 if grand_bad else 0


if __name__ == "__main__":
    sys.exit(main())