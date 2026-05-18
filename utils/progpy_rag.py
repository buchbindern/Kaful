"""
utils/progpy_rag.py
--------------------
Wrapper around the ProgPy framework RAG.

Retrieves relevant ProgPy documentation chunks for use in component codegen.
This is a shared utility — not per-machine, used by all machines.

Usage:
    from utils.progpy_rag import get_framework_context
    framework_context = get_framework_context()
"""

import os
import sys

from config import PROGPY_CHROMA_DIR, PROGPY_RAG_DIR


FRAMEWORK_QUERIES = [
    "PrognosticsModel base class required methods dx next_state output",
    "PrognosticsModel inputs outputs states events default_parameters class attributes",
    "initialize method signature input container output container None handling",
    "next_state method discrete model signature dt",
    "dx method continuous model state derivative",
    "output method OutputContainer signature",
    "event_state method return dict float 0 1",
    "threshold_met method return dict bool",
    "default_parameters tunable constants",
    "degradation state wear fouling prognostics model example",
]


def get_framework_context(twin_type: str = "physics") -> str:
    """
    Retrieve ProgPy framework context for component codegen.

    Args:
        twin_type: "physics" or "data_driven" — selects which manual chunks to inject

    Returns:
        formatted context string ready to inject into COMPONENT_CODEGEN_PROMPT
    """
    # Add progpy_rag to path so its internal imports work
    rag_path = str(PROGPY_RAG_DIR)
    if rag_path not in sys.path:
        sys.path.insert(0, rag_path)

    from vector_store import ChromaDBStore
    from ingestion.manual_chunks import MANUAL_CHUNKS_PHYSICS, MANUAL_CHUNKS_DATA_DRIVEN
    from ingestion.chunker import process as chunk_process

    store = ChromaDBStore(
        collection_name="progpy",
        persist_dir=str(PROGPY_CHROMA_DIR),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

    seen_ids = set()
    sections = []

    # Query the vector store
    for query_text in FRAMEWORK_QUERIES:
        results = store.query(query_text, top_k=4)
        for chunk in results:
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            label = f"[{chunk.metadata['type'].upper()}] {chunk.metadata['name']}"
            sections.append(f"### {label}\n{chunk.text}")

    # Inject manual chunks
    manual_chunks = MANUAL_CHUNKS_PHYSICS if twin_type == "physics" else MANUAL_CHUNKS_DATA_DRIVEN
    try:
        manual_final = chunk_process(manual_chunks)
        for chunk in manual_final:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                sections.append(f"### [CRITICAL API RULES] {chunk.metadata['name']}\n{chunk.text}")
        print(f"    Injected {len(manual_final)} manual ProgPy chunks")
    except Exception as e:
        print(f"    Warning: Could not inject manual chunks: {e}")

    print(f"    ProgPy context: {len(sections)} chunks")

    return "\n\n---\n\n".join(sections)