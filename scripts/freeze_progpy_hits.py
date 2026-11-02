"""
scripts/freeze_progpy_hits.py
-----------------------------
Freeze the ProgPy framework-context retrieval into a test fixture (M2).

The ProgPy Chroma store is gitignored and bound to OpenAI embeddings, so tests
cannot query it. This ranks the stored chunks for each FRAMEWORK_QUERIES entry
with M1's local BM25 stand-in (no API calls) and writes the top hits to
tests/fixtures/progpy/framework_hits.json.

Re-running it changes every 6b/6d prompt hash, so recorded fixtures must be
re-recorded afterwards (an approved step).

Usage:
    python -B scripts/freeze_progpy_hits.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config import PROGPY_CHROMA_DIR  # noqa: E402
from m1_estimate_baseline import Bm25, chroma_docs  # noqa: E402
from tests.harness.fake_embedder import PROGPY_HITS_PATH  # noqa: E402
from utils.progpy_rag import FRAMEWORK_QUERIES  # noqa: E402

TOP_K = 4  # matches get_framework_context's store.query(top_k=4)

# NASA Open Source Agreement 1.3, §3B (notice) and §3C (modification record);
# see tests/fixtures/progpy/NOTICE.md.
NOSA_COPYRIGHT = ("Copyright © 2021 United States Government as represented by the Administrator "
                  "of the National Aeronautics and Space Administration. No copyright is claimed "
                  "in the United States under Title 17, U.S. Code. All Other Rights Reserved.")
NOSA_MODIFICATION = ("Modification of ProgPy (Python Prognostics Packages, NASA Ames Research Center), "
                     "derived from Original Software provided by Government Agency: excerpts of "
                     "source docstrings, examples and documentation selected by BM25 ranking. "
                     "Distributed under the NASA Open Source Agreement 1.3 (see licenses/).")


def main() -> None:
    db = PROGPY_CHROMA_DIR / "chroma.sqlite3"
    index = Bm25(chroma_docs(db, "progpy"))
    keep = ("chunk_id", "text", "type", "name")
    hits = {q: [{k: d.get(k, "") for k in keep} for d in index.top(q, TOP_K)]
            for q in FRAMEWORK_QUERIES}
    PROGPY_HITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGPY_HITS_PATH.write_text(json.dumps({
        "about": "BM25 stand-in hits over the ProgPy store; see scripts/freeze_progpy_hits.py",
        "copyright": NOSA_COPYRIGHT,
        "license": "NASA Open Source Agreement 1.3 (tests/fixtures/progpy/licenses/)",
        "modification": NOSA_MODIFICATION,
        "top_k": TOP_K,
        "hits": hits,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n = sum(len(v) for v in hits.values())
    print(f"wrote {n} hits for {len(hits)} queries → {PROGPY_HITS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
