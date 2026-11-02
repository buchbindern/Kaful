"""
scripts/m1_estimate_baseline.py
-------------------------------
M1 cost baseline: reconstruct every API call a legacy run makes, from saved
checkpoints, and measure its tokens with Anthropic's free count_tokens endpoint.

How it works:
- Phase 1: rebuild each chunk request exactly as utils/rag.py sends it (the PDF
  slice plus the extraction prompt) and count it. The output is the extraction
  text stored in Chroma, where the current index has it.
- Phases 2-6: copy the machine's outputs to a scratch dir, delete the LLM-step
  checkpoints there, and run the real step code with call_claude replaced by a
  stub. The stub captures the exact request and returns the saved output, so
  prompts are rendered by the pipeline's own code.
- Retrieval that was never saved (4a, 4b, 4c, the ProgPy context) is replaced
  by a local BM25 ranking over the same stores. Those prompts are marked
  estimated.

No Messages calls and no OpenAI calls are made: OPENAI_API_KEY is removed from
the environment during the replay, and utils.llm.call_claude raises if reached.
Token counts are cached in docs/m1_baseline/token_counts.json, so with --offline
the script reruns without network.

Usage:
    python -B scripts/m1_estimate_baseline.py eversys_coffee_machine cnc_spindle
    python -B scripts/m1_estimate_baseline.py eversys_coffee_machine --offline
"""

import argparse
import base64
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import MACHINES_DIR, PROGPY_CHROMA_DIR, get_machine_config  # noqa: E402
from utils import ledger  # noqa: E402
from utils.prompt_hash import prompt_hash  # noqa: E402

OUT_DIR = ROOT / "docs" / "m1_baseline"
COUNTS_CACHE = OUT_DIR / "token_counts.json"
DEFAULT_MODEL = "claude-sonnet-4-6"  # utils/llm.py default; also the PDF extraction model
TEXT_PROBE = "x"                     # baseline message for counting bare text
# Phase-1 chunk source per machine. "indexed": the chunk PDFs behind the live index
# (eversys: the 75 chunks actually extracted; today's chunker would cut 70). "rechunk":
# what a cold run would extract today (CNC: the index covers only 48 of 229 chunks, and
# rag/chunks/ mixes files from two indexing runs).
PHASE1_SOURCE = {"eversys_coffee_machine": "indexed"}
# 4c: the unit-resolution response's `reasoning` strings are not saved. Assume this
# many characters per signal (the step prints the first 60 of each).
UNIT_REASONING_CHARS = 120

log = logging.getLogger("m1_baseline")

STEP_MODULES = [
    "phases.phase2_schema.step_b_context", "phases.phase2_schema.step_c_extract",
    "phases.phase3_simulate.step_b_plan", "phases.phase3_simulate.step_c_codegen",
    "phases.phase4_twin.step_a_comprehend", "phases.phase4_twin.step_b_physics",
    "phases.phase4_twin.step_c_resolve", "phases.phase5_triage.step_a_assign",
    "phases.phase5_triage.step_b_triage", "phases.phase6_codegen.step_a_spec",
    "phases.phase6_codegen.step_b_generate", "phases.phase6_codegen.step_d_composite",
]


# ── Token counting (free endpoint, cached) ────────────────────────────────────

class TokenCounter:
    """count_tokens with a JSON cache keyed by prompt hash / text hash."""

    def __init__(self, offline: bool) -> None:
        self.offline = offline
        self.cache: dict[str, int] = json.loads(COUNTS_CACHE.read_text()) if COUNTS_CACHE.exists() else {}
        self.calls = 0
        self._client: Any = None

    def _api(self) -> Any:
        if self.offline:
            raise RuntimeError("token count missing from cache and --offline was given")
        if self._client is None:
            import anthropic
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return self._client

    def request(self, request: dict[str, Any]) -> int:
        key = "req:" + prompt_hash(request)
        if key not in self.cache:
            kwargs = {k: request[k] for k in ("model", "messages") }
            if request.get("system"):
                kwargs["system"] = request["system"]
            self.cache[key] = self._api().messages.count_tokens(**kwargs).input_tokens
            self.calls += 1
        return self.cache[key]

    def text(self, text: str, model: str) -> int:
        """Tokens in a bare text: count it as a message, minus a one-char probe."""
        key = f"txt:{model}:" + hashlib.sha256(text.encode()).hexdigest()
        if key not in self.cache:
            probe = self._message_tokens(TEXT_PROBE, model)
            self.cache[key] = max(self._message_tokens(text, model) - probe + 1, 0)
        return self.cache[key]

    def _message_tokens(self, text: str, model: str) -> int:
        return self.request({"model": model, "messages": [{"role": "user", "content": text}]})

    def save(self) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        COUNTS_CACHE.write_text(json.dumps(self.cache, indent=1, sort_keys=True))


# ── Local stand-in for vector retrieval (BM25 over the stored chunk texts) ────

class Bm25:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs
        self.tfs = [Counter(_tokens(d["text"])) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def top(self, query: str, k: int, k1: float = 1.5, b: float = 0.75) -> list[dict[str, Any]]:
        q = _tokens(query)
        scores = []
        for i, tf in enumerate(self.tfs):
            s = sum(self.idf.get(t, 0) * tf[t] * (k1 + 1)
                    / (tf[t] + k1 * (1 - b + b * self.lens[i] / self.avg)) for t in q if t in tf)
            scores.append((s, i))
        return [self.docs[i] for _, i in sorted(scores, reverse=True)[:k]]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def chroma_docs(db: Path, collection: str) -> list[dict[str, Any]]:
    """Documents and metadata of one Chroma collection, read with sqlite (no embeddings)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        """select e.embedding_id, m.key, m.string_value, m.int_value
           from embeddings e join segments s on e.segment_id = s.id
           join collections c on s.collection = c.id
           join embedding_metadata m on m.id = e.id where c.name = ?""", (collection,)).fetchall()
    con.close()
    docs: dict[str, dict[str, Any]] = {}
    for emb_id, key, sval, ival in rows:
        d = docs.setdefault(emb_id, {"chunk_id": emb_id})
        d["text" if key == "chroma:document" else key] = sval if sval is not None else ival
    return list(docs.values())


class FakeRag:
    """Stands in for ManualRAG.retrieve_chunks; records the query embeddings it replaces."""

    def __init__(self, docs: list[dict[str, Any]], embeddings: list[dict[str, Any]]) -> None:
        self.index = Bm25(docs)
        self.embeddings = embeddings
        self.corpus_mean = sum(len(d["text"]) for d in docs) / max(len(docs), 1)

    def retrieve_chunks(self, queries: list[str], n_results_per_query: int = 5) -> list[dict]:
        found: dict[str, dict] = {}
        for q in queries:
            for d in self.index.top(q, n_results_per_query):
                c = found.setdefault(d["chunk_id"], {
                    "chunk_id": d["chunk_id"], "text": d["text"],
                    "start_page": d.get("start_page"), "end_page": d.get("end_page"),
                    "heading": d.get("heading", ""), "source_type": d.get("source_type", ""),
                    "pdf_file": d.get("pdf_file", ""), "matched_queries": []})
                c["matched_queries"].append(q)
        self.embeddings.append({**ledger.caller_scope(), "component": ledger.current_component(),
                                "model": "text-embedding-3-large", "n_texts": len(queries),
                                "chars": sum(len(q) for q in queries),
                                "hits": [[c["chunk_id"], len(c["text"])] for c in found.values()],
                                "corpus_mean_chars": self.corpus_mean})
        return list(found.values())


@contextmanager
def lexical_progpy_store(embeddings: list[dict[str, Any]]) -> Iterator[None]:
    """Make utils.progpy_rag.get_framework_context retrieve with BM25 instead of OpenAI."""
    sys.path.insert(0, str(ROOT / "rag_frameworks" / "progpy" / "progpy_rag"))
    from vector_store import ChromaDBStore  # type: ignore[import-not-found]
    docs = chroma_docs(PROGPY_CHROMA_DIR / "chroma.sqlite3", "progpy")
    index = Bm25(docs)
    corpus_mean = sum(len(d["text"]) for d in docs) / len(docs)
    original = ChromaDBStore.query

    def query(self: Any, text: str, filter: Any = None, top_k: int = 5) -> list[Any]:
        hits = index.top(text, top_k)
        embeddings.append({**ledger.caller_scope(), "component": None,
                           "model": "text-embedding-3-small", "n_texts": 1, "chars": len(text),
                           "hits": [[d["chunk_id"], len(d["text"])] for d in hits],
                           "corpus_mean_chars": corpus_mean})
        meta = lambda d: {"type": d.get("type", ""), "name": d.get("name", "")}  # noqa: E731
        return [type("Hit", (), {"chunk_id": d["chunk_id"], "text": d["text"], "metadata": meta(d)})
                for d in hits]

    def init(self: Any, *_a: Any, **_k: Any) -> None:
        """No client: the persisted collection is bound to OpenAI embeddings."""

    original_init = ChromaDBStore.__init__
    ChromaDBStore.query, ChromaDBStore.__init__ = query, init
    try:
        yield
    finally:
        ChromaDBStore.query, ChromaDBStore.__init__ = original, original_init


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def extraction_prompt() -> str:
    """The first-attempt prompt from ManualRAG._extract_rich_content, read from source."""
    src = (ROOT / "utils" / "rag.py").read_text()
    return re.search(r'base_prompt = """(.*?)"""', src, re.S).group(1)


def phase1_chunks(machine_id: str) -> list[dict[str, Any]]:
    """The chunks a cold phase-1 run would extract, built by the real chunker."""
    from utils.rag import ManualRAG
    cfg = get_machine_config(machine_id)
    rag = ManualRAG.__new__(ManualRAG)  # chunking needs no clients
    rag.cfg = cfg
    chunks = []
    for manual in rag._load_manuals_list():
        path = str(manual["path"])
        texts = {p["page_num"]: p["text"] for p in rag._extract_page_texts(path)}
        for ch in rag._build_section_chunks(pdf_path=path, max_section_pages=8, fallback_chunk_size=5):
            chunks.append({
                "pdf_file": manual["path"].name, "start": ch["start"], "end": ch["end"],
                "pdf_b64": rag._extract_pdf_chunk(path, ch["start"], ch["end"]),
                "page_text_chars": sum(len(texts.get(p, "")) for p in range(ch["start"], ch["end"] + 1)),
            })
    return chunks


def indexed_chunks(machine_id: str) -> list[dict[str, Any]]:
    """The chunk PDFs behind the live index, with their stored extraction text."""
    import fitz
    cfg = get_machine_config(machine_id)
    docs = chroma_docs(cfg["rag_dir"] / "chroma.sqlite3", cfg["collection_name"])
    manual = _manuals(machine_id)[0]["path"]
    pages = [p.get_text() or "" for p in fitz.open(str(manual))]
    out = []
    for d in sorted(docs, key=lambda d: d["chunk_id"]):
        chunk_file = d.get("chunk_file") or d["pdf_file"]  # older index: pdf_file is the chunk file
        out.append({
            "pdf_file": manual.name, "start": d["start_page"], "end": d["end_page"], "text": d["text"],
            "pdf_b64": base64.b64encode((cfg["chunks_dir"] / chunk_file).read_bytes()).decode(),
            "page_text_chars": sum(len(pages[i - 1]) for i in range(d["start_page"], d["end_page"] + 1)),
        })
    return out


def stored_extractions(machine_id: str) -> dict[tuple[str, int, int], str]:
    """Extraction text in the live Chroma index, keyed by (pdf_file, start, end)."""
    cfg = get_machine_config(machine_id)
    docs = chroma_docs(cfg["rag_dir"] / "chroma.sqlite3", cfg["collection_name"])
    pdf_files = {m["path"].name for m in _manuals(machine_id)}
    out = {}
    for d in docs:
        # The eversys index predates the pdf_file metadata; it has one manual.
        pdf = d.get("pdf_file") or next(iter(pdf_files))
        out[(pdf, d["start_page"], d["end_page"])] = d["text"]
    return out


def _manuals(machine_id: str) -> list[dict[str, Any]]:
    from utils.rag import ManualRAG
    rag = ManualRAG.__new__(ManualRAG)
    rag.cfg = get_machine_config(machine_id)
    return rag._load_manuals_list()


def measure_phase1(machine_id: str, counter: TokenCounter) -> list[dict[str, Any]]:
    prompt = extraction_prompt()
    if PHASE1_SOURCE.get(machine_id) == "indexed":
        chunks = indexed_chunks(machine_id)
        stored = {(c["pdf_file"], c["start"], c["end"]): c["text"] for c in chunks}
    else:
        chunks, stored = phase1_chunks(machine_id), stored_extractions(machine_id)
    records = []
    for i, ch in enumerate(chunks):
        request = {"model": DEFAULT_MODEL, "max_tokens": 8192, "messages": [{"role": "user", "content": [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                            "data": ch["pdf_b64"]}},
            {"type": "text", "text": prompt}]}]}
        text = stored.get((ch["pdf_file"], ch["start"], ch["end"]))
        records.append({
            "machine": machine_id, "phase": "phase1", "step": "ingest", "component": None,
            "event": "pdf_extract", "model": DEFAULT_MODEL, "chunk": i, "pdf_file": ch["pdf_file"],
            "pages": [ch["start"], ch["end"]], "page_text_chars": ch["page_text_chars"],
            "prompt_sha256": prompt_hash(request), "input_tokens": counter.request(request),
            "input_basis": "count_tokens", "chunk_source": PHASE1_SOURCE.get(machine_id, "rechunk"),
            "output_tokens": counter.text(text, DEFAULT_MODEL) if text else None,
            "output_chars": len(text) if text else None,
            "output_basis": "count_tokens(stored extraction)" if text else "missing",
        })
        log.info("phase1 %s chunk %d pages %d-%d: %d in", ch["pdf_file"], i, ch["start"], ch["end"],
                 records[-1]["input_tokens"])
    return records


# ── Phases 2-6: replay the step code with a capturing call_claude ─────────────

LLM_CHECKPOINTS = [
    "schema/step_b_context.json", "schema/step_c_runs.json",
    "simulate/step_b_plan.json", "simulate/simulator.py",
    "twin/step_a_comprehension.json", "twin/components", "twin/step_c_machine_model.json",
    "triage/step_a_field_assignments.json", "triage/step_b_triage.json",
    "codegen/specs", "codegen/components", "codegen/step_d_external_inputs.json",
    "codegen/composite_model.py",
]


def scratch_cfg(machine_id: str, scratch: Path) -> dict[str, Any]:
    """Machine cfg with every outputs path redirected into a scratch copy."""
    cfg = get_machine_config(machine_id)
    real_out = cfg["machine_dir"] / "outputs"
    shutil.copytree(real_out, scratch / "outputs",
                    ignore=shutil.ignore_patterns("__pycache__", "*SNAPSHOT*", "estimate", "*.csv"))
    for rel in LLM_CHECKPOINTS:
        p = scratch / "outputs" / rel
        shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
    for key, val in list(cfg.items()):
        if isinstance(val, Path) and real_out in (val, *val.parents):
            cfg[key] = scratch / "outputs" / val.relative_to(real_out)
    return cfg


class Responder:
    """Returns the saved model output for each captured call, and says how exact it is."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.counts: Counter[str] = Counter()

    def _json(self, rel: str) -> Any:
        return json.loads((self.out / rel).read_text())

    def __call__(self, step: str, component: str | None, request: dict) -> tuple[str, str]:
        n = self.counts[step]
        self.counts[step] += 1
        handler: Callable[[], tuple[str, str]] = {
            "step_b_context": lambda: self._context(n),
            "step_c_extract": lambda: (self._json("schema/step_c_runs.json")[n], "exact"),
            "step_b_plan": lambda: (json.dumps(self._json("simulate/step_b_plan.json"), indent=2), "reformatted"),
            "step_c_codegen": lambda: ((self.out / "simulate/simulator.py").read_text(), "saved code"),
            "step_a_comprehend": lambda: (json.dumps(self._json("twin/step_a_comprehension.json"), indent=2), "reformatted"),
            "step_b_physics": lambda: (json.dumps(self._json(f"twin/components/{component}.json"), indent=2), "reformatted"),
            "step_c_resolve": self._units,
            "step_a_assign": lambda: (json.dumps(self._json("triage/step_a_field_assignments.json"), indent=2), "reformatted"),
            "step_b_triage": lambda: (json.dumps(self._json("triage/step_b_triage.json"), indent=2), "reformatted"),
            "step_a_spec": lambda: (json.dumps(self._json(f"codegen/specs/{component}.json"), indent=2), "reformatted"),
            "step_b_generate": lambda: ((self.out / f"codegen/components/{component}.py").read_text(), "saved code"),
            "step_d_composite": lambda: self._composite(request),
        }[step]
        return handler()

    def _context(self, n: int) -> tuple[str, str]:
        saved = self._json("schema/step_b_context.json")
        event_keys = ["machine_type", "machine_name", "event_type", "event_description"]
        part = {k: v for k, v in saved.items() if (k in event_keys) == (n == 0)}
        return json.dumps(part, indent=2), "reformatted (split of merged file)"

    def _units(self) -> tuple[str, str]:
        model = self._json("twin/step_c_machine_model.json")
        sigs = [r for c in model.get("components", []) for r in c.get("operating_ranges", [])]
        sigs += [s for fp in model.get("flow_paths", []) for s in fp.get("boundary_signals", [])]
        resolved = {s["signal_name"]: {"unit": s["unit"], "confidence": s["unit_confidence"],
                                       "reasoning": "x" * UNIT_REASONING_CHARS}
                    for s in sigs if "unit_confidence" in s}
        return json.dumps(resolved, indent=2), "estimated (reasoning text not saved)"

    def _composite(self, request: dict) -> tuple[str, str]:
        if request.get("model") == "claude-opus-4-6":
            return (self.out / "codegen/composite_model.py").read_text(), "saved code"
        saved = self._json("codegen/step_d_external_inputs.json")
        prompt = request["messages"][0]["content"]
        return json.dumps({k: v for k, v in saved.items() if k in prompt}, indent=2), "reformatted"


@contextmanager
def capture_calls(responder: Responder, captured: list[dict[str, Any]]) -> Iterator[None]:
    """Replace call_claude in every LLM step module with a capturing stub."""
    import importlib
    import utils.llm

    def fake_call_claude(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 4000,
                         system: str | None = None, temperature: float = 1.0, max_retries: int = 5) -> str:
        request = {"model": model, "max_tokens": max_tokens,
                   "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
        if system:
            request["system"] = system
        scope, component = ledger.caller_scope(), ledger.current_component()
        text, basis = responder(scope["step"], component, request)
        captured.append({**scope, "component": component, "request": request,
                         "response": text, "output_basis": basis})
        return text

    def forbidden(*_a: Any, **_k: Any) -> str:
        raise RuntimeError("real call_claude reached during replay")

    modules = [importlib.import_module(m) for m in STEP_MODULES]
    saved = [(m, m.call_claude) for m in modules]
    original_llm = utils.llm.call_claude
    utils.llm.call_claude = forbidden
    for m in modules:
        m.call_claude = fake_call_claude
    try:
        yield
    finally:
        utils.llm.call_claude = original_llm
        for m, fn in saved:
            m.call_claude = fn


@contextmanager
def no_openai_key() -> Iterator[None]:
    key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        yield
    finally:
        if key is not None:
            os.environ["OPENAI_API_KEY"] = key


def replay(machine_id: str) -> tuple[list[dict], list[dict]]:
    """Run phases 2-6 against a scratch copy; return captured calls and embedding queries."""
    from phases.phase2_schema import run as phase2
    from phases.phase3_simulate import run as phase3
    from phases.phase4_twin import run as phase4
    from phases.phase5_triage import run as phase5
    from phases.phase6_codegen import step_a_spec, step_b_generate
    from phases.phase6_codegen import step_d_composite as sd

    real = get_machine_config(machine_id)
    docs = chroma_docs(real["rag_dir"] / "chroma.sqlite3", real["collection_name"])
    captured: list[dict] = []
    embeddings: list[dict] = []
    # Fixed scratch path: 6d's composite prompt embeds the component directory, and a
    # random temp path would change the prompt hash (and miss the token-count cache).
    tmp = Path(tempfile.gettempdir()) / "kaful_m1_replay" / machine_id
    shutil.rmtree(tmp, ignore_errors=True)
    with no_openai_key(), lexical_progpy_store(embeddings), \
            capture_calls(Responder(real["machine_dir"] / "outputs"), captured):
        cfg = scratch_cfg(machine_id, tmp)
        ledger.start_run({"machine_dir": tmp, "machine_id": machine_id})
        rag = FakeRag(docs, embeddings)
        schema = phase2(cfg, rag)
        phase3(cfg, rag, schema)
        model, _, _ = phase4(cfg, rag)
        full = phase5(cfg, schema, model)
        specs = step_a_spec.run(cfg, full)
        step_b_generate.run(cfg, specs)
        interfaces = sd._build_interfaces(full, specs, cfg)
        connections = sd._propose_connections(cfg, model, interfaces)
        external = sd._resolve_external_inputs(cfg, connections, interfaces, full, model)
        sd._generate_composite(cfg, interfaces, connections, external)
        ledger.end_run()
    shutil.rmtree(tmp, ignore_errors=True)
    return captured, embeddings


ESTIMATED_RETRIEVAL = {"step_a_comprehend", "step_b_physics", "step_c_resolve"}
PROGPY_CONTEXT = {"step_b_generate", "step_d_composite"}


def measure_steps(machine_id: str, counter: TokenCounter) -> tuple[list[dict], list[dict]]:
    captured, embeddings = replay(machine_id)
    records = []
    for c in captured:
        req = c["request"]
        uses_progpy = c["step"] in PROGPY_CONTEXT and (req.get("model") == "claude-opus-4-6"
                                                       or c["step"] == "step_b_generate")
        records.append({
            "machine": machine_id, "phase": c["phase"], "step": c["step"], "component": c["component"],
            "call_site": c["call_site"], "event": "llm_call", "model": req["model"],
            "prompt_sha256": prompt_hash(req), "request_chars": len(req.get("system") or "")
            + len(req["messages"][0]["content"]),
            "input_tokens": counter.request(req),
            "input_basis": "count_tokens (BM25 stand-in retrieval)" if c["step"] in ESTIMATED_RETRIEVAL
            else "count_tokens (BM25 stand-in ProgPy context)" if uses_progpy else "count_tokens",
            "output_tokens": counter.text(c["response"], req["model"]),
            "output_chars": len(c["response"]), "output_basis": c["output_basis"],
        })
    return records, embeddings


# ── Retrieval queries for steps whose chunks are checkpointed (2a, 3a) ────────

def checkpointed_query_embeddings(machine_id: str) -> list[dict[str, Any]]:
    """2a and 3a load from checkpoints in the replay; their query embeddings still cost."""
    import importlib.util
    cfg = get_machine_config(machine_id)
    spec = importlib.util.spec_from_file_location("queries", cfg["machine_dir"] / "queries.py")
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)  # type: ignore[union-attr]
    rows = []
    for phase, step, names in [("phase2", "step_a_retrieve", ["schema_queries", "process_queries"]),
                               ("phase3", "step_a_retrieve", ["simulation_queries"])]:
        texts = [t for n in names for t in _flatten(getattr(q, n, []))]
        rows.append({"phase": phase, "step": step, "component": None, "model": "text-embedding-3-large",
                     "n_texts": len(texts), "chars": sum(len(t) for t in texts)})
    return rows


def _flatten(items: Any) -> list[str]:
    """simulation_queries may be grouped by topic (dict of lists)."""
    if isinstance(items, dict):
        return [t for v in items.values() for t in _flatten(v)]
    if isinstance(items, (list, tuple)):
        return [t for v in items for t in _flatten(v)]
    return [items] if isinstance(items, str) else []


# ── Main ──────────────────────────────────────────────────────────────────────

def run_machine(machine_id: str, counter: TokenCounter) -> None:
    p1 = measure_phase1(machine_id, counter)
    counter.save()
    steps, embeddings = measure_steps(machine_id, counter)
    counter.save()
    embeddings += checkpointed_query_embeddings(machine_id)
    for e in embeddings:
        e.pop("call_site", None)
    out = {"machine": machine_id, "phase1": p1, "calls": steps, "embeddings": embeddings}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{machine_id}_calls.json").write_text(json.dumps(out, indent=1))
    log.info("%s: %d phase-1 chunks, %d step calls, %d embedding requests",
             machine_id, len(p1), len(steps), len(embeddings))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("machines", nargs="+")
    parser.add_argument("--offline", action="store_true", help="use cached token counts only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    counter = TokenCounter(offline=args.offline)
    try:
        for machine_id in args.machines:
            assert (MACHINES_DIR / machine_id).is_dir(), machine_id
            run_machine(machine_id, counter)
    finally:
        counter.save()
        log.info("count_tokens API calls this run: %d", counter.calls)


if __name__ == "__main__":
    main()
