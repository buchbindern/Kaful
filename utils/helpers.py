"""
utils/helpers.py
----------------
General purpose helpers used across pipeline phases.

Usage:
    from utils.helpers import dedupe_chunks, chunks_to_context
"""


def dedupe_chunks(chunks: list[dict]) -> list[dict]:
    """
    Remove duplicate chunks based on normalized text.
    Duplicates happen when multiple queries retrieve the same chunk.

    Args:
        chunks: list of chunk dicts from ManualRAG.retrieve_chunks()

    Returns:
        list of unique chunk dicts with all original fields preserved
    """
    seen         = set()
    unique_chunks = []

    for chunk in chunks:
        text = chunk.get("text", "").strip() if isinstance(chunk, dict) else str(chunk).strip()

        if not text:
            continue

        norm = " ".join(text.lower().split())

        if norm in seen:
            continue

        seen.add(norm)
        unique_chunks.append(chunk)

    return unique_chunks


def chunks_to_context(chunks: list[dict]) -> str:
    """
    Convert deduplicated chunks into a formatted context string for prompting.

    Args:
        chunks: list of chunk dicts from dedupe_chunks()

    Returns:
        formatted string ready to inject into a prompt
    """
    parts = []

    for i, chunk in enumerate(chunks, start=1):
        text = chunk["text"]
        page = chunk.get("start_page", chunk.get("page", "?"))
        parts.append(f"--- Chunk {i} | page={page} ---\n{text}")

    return "\n\n".join(parts)

############
def load_module_from_path(path):
    module_name = os.path.splitext(os.path.basename(path))[0]
    
    # Remove cached version if it exists
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def container_to_dict(obj):
    """Convert any ProgPy container to a plain Python dict."""
    try:
        return dict(obj)
    except Exception:
        return {k: obj[k] for k in obj.keys()}

def find_component_class(module):
    """
    Find the ProgPy PrognosticsModel class in a loaded module.
    Accepts both continuous (dx) and discrete (next_state) models.
    Returns (class_name, class) or raises ValueError.
    """
    required_attrs   = ["inputs", "states", "outputs", "events"]
    required_methods = ["initialize", "output", "event_state", "threshold_met"]
    candidates = [
        (name, obj)
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__
        and all(hasattr(obj, a) for a in required_attrs)
        and all(hasattr(obj, m) for m in required_methods)
        and (hasattr(obj, "next_state") or hasattr(obj, "dx"))
    ]
    if not candidates:
        raise ValueError("No valid ProgPy component class found")
    return candidates[0]