"""
phases/phase2_schema/step_a_retrieve.py
----------------------------------------
Step A: Retrieve relevant chunks from the RAG using machine-specific queries.

Loads queries from the machine's queries.py, retrieves chunks from ChromaDB,
dedupes them, and saves the result to disk.

Output saved to: outputs/schema/step_a_chunks.json
"""

import json
import importlib.util
import sys
from pathlib import Path

from utils.helpers import dedupe_chunks, chunks_to_context


def run(cfg: dict, rag) -> tuple[list[dict], str]:
    """
    Retrieve and deduplicate chunks using machine-specific queries.

    Args:
        cfg: result of get_machine_config()
        rag: ManualRAG instance (already indexed)

    Returns:
        tuple of (chunks, context_string)
    """
    output_path = cfg["step_a_chunks"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_a already done — loading from disk")
        with open(output_path) as f:
            chunks = json.load(f)
        return chunks, chunks_to_context(chunks)

    print("  Running step_a — retrieving chunks...")

    # Load machine-specific queries
    schema_queries, process_queries = _load_queries(cfg)

    # Retrieve chunks
    chunks_1 = rag.retrieve_chunks(schema_queries, n_results_per_query=4)
    chunks_2 = rag.retrieve_chunks(process_queries, n_results_per_query=3)
    chunks   = dedupe_chunks(chunks_1 + chunks_2)

    # test (delete this later)
    for c in chunks:
        print("\n--- CHUNK ---")
        print("id:", c.get("chunk_id"))
        print("pages:", c.get("start_page"), "-", c.get("end_page"))
        print("heading:", c.get("heading"))
        print("matched_queries:", c.get("matched_queries"))
        print(c.get("text", "")[:500])

    print(f"    Retrieved {len(chunks_1) + len(chunks_2)} chunks → {len(chunks)} after dedup")

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"    Saved → {output_path.name}")

    return chunks, chunks_to_context(chunks)


def _load_queries(cfg: dict) -> tuple[list[str], list[str]]:
    """
    Dynamically load queries from the machine's queries.py.

    Args:
        cfg: result of get_machine_config()

    Returns:
        tuple of (schema_queries, process_queries)
    """
    queries_path = cfg["machine_dir"] / "queries.py"

    if not queries_path.exists():
        raise FileNotFoundError(
            f"No queries.py found at {queries_path}.\n"
            f"Create it with schema_queries and process_queries lists."
        )

    # Dynamically import the machine's queries.py
    spec   = importlib.util.spec_from_file_location("queries", queries_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "schema_queries") or not hasattr(module, "process_queries"):
        raise AttributeError(
            f"{queries_path} must define both 'schema_queries' and 'process_queries' lists."
        )

    return module.schema_queries, module.process_queries