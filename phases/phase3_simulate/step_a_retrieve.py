"""
phases/phase3_simulate/step_a_retrieve.py
------------------------------------------
Step A: Retrieve and organize manual context for simulation.

Uses topic-tagged simulation queries to retrieve chunks from the RAG,
then organizes them by topic for structured injection into the simulator prompts.

Output saved to: outputs/simulate/step_a_context.json
"""

import json
import importlib.util

from utils.helpers import dedupe_chunks


def run(cfg: dict, rag) -> tuple[dict, str]:
    """
    Retrieve and organize manual context for simulation.

    Args:
        cfg: result of get_machine_config()
        rag: ManualRAG instance (already indexed)

    Returns:
        tuple of (grouped_context dict, formatted context string)
    """
    output_path = cfg["sim_step_a_context"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_a already done — loading from disk")
        with open(output_path) as f:
            grouped_context = json.load(f)
        return grouped_context, _format_manual_context(grouped_context)

    print("  Running step_a — retrieving simulation context...")

    # Load simulation queries from machine's queries.py
    simulation_queries = _load_simulation_queries(cfg)

    # Retrieve chunks using query strings only
    query_strings = [q for _, q in simulation_queries]
    chunks        = rag.retrieve_chunks(query_strings, n_results_per_query=5)
    chunks        = dedupe_chunks(chunks)

    print(f"    Retrieved {len(chunks)} chunks")

    # Organize by topic
    grouped_context = _organize_by_topic(chunks, simulation_queries)

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(grouped_context, f, indent=2)

    print(f"    Topics: {list(grouped_context.keys())}")
    print(f"    Saved → {output_path.name}")

    return grouped_context, _format_manual_context(grouped_context)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_simulation_queries(cfg: dict) -> list[tuple[str, str]]:
    """Load simulation_queries from the machine's queries.py."""
    queries_path = cfg["machine_dir"] / "queries.py"

    if not queries_path.exists():
        raise FileNotFoundError(
            f"No queries.py found at {queries_path}."
        )

    spec   = importlib.util.spec_from_file_location("queries", queries_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "simulation_queries"):
        raise AttributeError(
            f"{queries_path} must define a 'simulation_queries' list of (topic, query) tuples."
        )

    return module.simulation_queries


def _organize_by_topic(chunks: list[dict], simulation_queries: list[tuple]) -> dict:
    """Organize chunks by topic based on which query retrieved them."""
    query_to_topic = {query: topic for topic, query in simulation_queries}
    grouped        = {}

    for chunk in chunks:
        grouped_topics = set()
        for matched_query in chunk.get("matched_queries", []):
            topic = query_to_topic.get(matched_query, "general")
            grouped_topics.add(topic)

        for topic in grouped_topics:
            grouped.setdefault(topic, []).append({
                "chunk_id": chunk["chunk_id"],
                "pages":    f"{chunk['start_page']}-{chunk['end_page']}",
                "text":     chunk["text"],
            })

    return grouped


def _format_manual_context(grouped_context: dict) -> str:
    """Format grouped context into a string for prompt injection."""
    parts = []

    for topic, topic_chunks in grouped_context.items():
        parts.append(f"## {topic}")
        for chunk in topic_chunks:
            parts.append(f"[Pages {chunk['pages']}]\n{chunk['text']}")

    return "\n\n".join(parts)