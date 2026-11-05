"""The 3a hash-order case (PLAN M7): prompts and artifacts must not depend on PYTHONHASHSEED.

Legacy 3a (`_organize_by_topic`) put each retrieved chunk under its topics by iterating
a set, so a chunk matched by several topics' queries landed in the prompt in hash
order and the prompt changed between runs. Here every chunk is matched by all three
topic queries (tiny_pump's legacy `simulation_queries`) with tied scores, and the whole
understand phase runs on tiny_pump; under three hash seeds the retrieved context, every
request's prompt hash (the response-cache key) and every artifact must be identical.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from kaful.config import REPO_ROOT

SCRIPT = """
import hashlib, json, sys
from pathlib import Path
import numpy as np
from kaful import pipeline
from kaful.artifacts.document_index import Chunk, DocumentIndex, DocumentRecord, PageText, chunk_id_for, index_hash_for, text_sha256
from kaful.config import IngestSettings, LLMSettings, UnderstandSettings
from kaful.llm.budget import RunBudget
from kaful.llm.embeddings import Embedder, VectorStore
from kaful.llm.ledger import Ledger, read_ledger
from kaful.phases.ingest import phase as ingest_phase
from kaful.phases.understand import phase as understand_phase
from kaful.phases.understand.context import select_context
from tests.harness.kaful_factories import BASE, FINGERPRINT, SHA, SIGNALS
from tests.harness.kaful_ingest import FakeEmbeddings
from tests.harness.kaful_llm import ScriptedTransport
from tests.harness.kaful_understand import MACHINE_ID, ScriptedUnderstanding, locked_prompts, make_tiny_pump, understand_settings

tmp = Path(sys.argv[1])
# 1. Retrieval with ties: 12 one-page chunks, equal vectors, three topic queries each matching every chunk.
chunks = tuple(sorted((Chunk(chunk_id=chunk_id_for(SHA, p, p, "v1"), doc_id="manual", page_start=p, page_end=p,
                             text=t, text_sha256=text_sha256(t), chunker_version="v1")
                       for p in range(1, 13) for t in [f"[manual]\\n--- page {p} ---\\n" + "wear clogging " * 100]),
                      key=lambda c: c.chunk_id))
index = DocumentIndex(**BASE, index_hash=index_hash_for(chunks, FINGERPRINT), fingerprint=FINGERPRINT, chunks=chunks,
                      documents=(DocumentRecord(doc_id="manual", path="manual/pump.pdf", kind="manual", sha256=SHA, n_pages=12),),
                      pages=tuple(PageText(doc_id="manual", page=p, text="x", page_class="text", signals=SIGNALS) for p in range(1, 13)))
settings = UnderstandSettings(context_max_tokens=2000, queries={
    "components": ("duty operating hours per day rated flow water temperature",),
    "fault_modes": ("wear clogging service interval replace inspect life alarm warning",),
    "sensor_channels": ("permitted range limits discharge pressure inlet pressure motor current vibration",)})
embedder = Embedder(settings=IngestSettings(embedding_dimensions=8), llm=LLMSettings(), ledger=Ledger(tmp / "l.jsonl"),
                    budget=RunBudget(1.0), transport=FakeEmbeddings(), store=VectorStore(tmp / "vectors"))
context = select_context(index, np.ones((12, 8), dtype=np.float32), embedder, settings, IngestSettings())
# 2. The whole phase on tiny_pump.
make_tiny_pump(tmp)
ingest_phase.read_lock = locked_prompts
understand_phase.read_lock = locked_prompts
request = pipeline.RunRequest(machine_id=MACHINE_ID, argv=("run",), only=("ingest", "understand"), budget_usd=1.0)
ctx, base = pipeline.start_run(understand_settings(tmp), request, transport=ScriptedTransport(ScriptedUnderstanding()),
                               embedding_transport=FakeEmbeddings())
pipeline.execute(ctx, ("ingest", "understand"), base)
rows = read_ledger(ctx.run_dir / "ledger.jsonl")
def normalized(path):
    # The run id differs per run (not per hash seed), and so does the sha256 of the index
    # file each artifact cites, because that file records its run id.
    data = json.loads(path.read_text().replace(ctx.manifest.run_id, "RUN"))
    data["inputs"] = [{**ref, "sha256": "-"} for ref in data["inputs"]]
    return json.dumps(data, sort_keys=True)
artifacts = {r.kind: normalized(ctx.machine_dir / r.path) for r in ctx.manifest.phase("understand").artifacts}
sys.stdout.write(json.dumps({
    "retrieved": list(context.selection.chunk_ids), "context_sha256": context.selection.context_sha256,
    "prompt_hashes": [(r.prompt_id, r.sample, r.prompt_sha256) for r in rows if r.event == "llm_call"],
    "artifacts": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sorted(artifacts.items())},
}, sort_keys=True))
"""


def test_prompts_and_artifacts_are_identical_across_hash_seeds(tmp_path: Path) -> None:
    outputs = {}
    for seed in ("0", "1", "4242"):
        env = {k: v for k, v in os.environ.items() if k not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}}
        env["PYTHONHASHSEED"] = seed
        done = subprocess.run([sys.executable, "-B", "-c", SCRIPT, str(tmp_path / seed)], cwd=REPO_ROOT, env=env,
                              capture_output=True, text=True, check=True)
        outputs[seed] = done.stdout.splitlines()[-1]  # PyMuPDF prints a notice first
    assert len(set(outputs.values())) == 1, outputs
    result = json.loads(outputs["0"])
    assert len(result["retrieved"]) == 5  # 12 tied chunks; the budget fits five, picked by chunk id
    assert len(result["prompt_hashes"]) == 9 and set(result["artifacts"]) == {
        "machine_model", "sensor_contract", "understand_report"}
