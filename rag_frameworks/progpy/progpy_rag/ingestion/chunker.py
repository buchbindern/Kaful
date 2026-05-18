"""
ingestion/chunker.py

Converts RawChunks → final Chunks ready for embedding.

Responsibilities:
  - Token-length enforcement (split oversized chunks, skip tiny ones)
  - Stable chunk ID generation (hash of source + name so re-indexing is idempotent)
  - Final metadata assembly with all required fields
"""

import hashlib

from .github_crawler import RawChunk
from vector_store.base import Chunk


# Embedding model token limits
# text-embedding-3-small: 8191 tokens max
# We use a lower ceiling to leave room for metadata context
MAX_TOKENS = 2500
MIN_TOKENS = 30

FRAMEWORK = "progpy"


def count_tokens(text: str) -> int:
    """
    Approximate token count without tiktoken (no network required).
    Rule of thumb: 1 token ≈ 4 characters for English/code.
    Accurate enough for chunking purposes.
    """
    return len(text) // 4


def split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """
    Split text into token-safe chunks.
    Tries to split on double newlines (paragraph boundaries) first.
    Falls back to hard token splits if needed.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    # Try paragraph splits first
    paragraphs = text.split("\n\n")
    parts = []
    current = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current_tokens + para_tokens > max_tokens and current:
            parts.append("\n\n".join(current))
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        parts.append("\n\n".join(current))

    # Hard fallback: slice by char count for any part still over limit
    result = []
    char_limit = max_tokens * 4
    for part in parts:
        if count_tokens(part) > max_tokens:
            for i in range(0, len(part), char_limit):
                result.append(part[i : i + char_limit])
        else:
            result.append(part)

    return result


def make_chunk_id(source: str, name: str, index: int, text: str = "") -> str:
    raw = f"{source}::{name}::{index}::{text}"
    return hashlib.md5(raw.encode()).hexdigest()


def raw_to_chunks(raw: RawChunk) -> list[Chunk]:
    """Convert one RawChunk into one or more final Chunks."""
    parts = split_by_tokens(raw.text, MAX_TOKENS)
    chunks = []

    for i, part in enumerate(parts):
        if count_tokens(part) < MIN_TOKENS:
            continue  # skip near-empty fragments

        chunk_id = make_chunk_id(raw.source, raw.name, i, part)

        metadata = {
            "framework": FRAMEWORK,
            "domain":    raw.domain,
            "type":      raw.chunk_type,
            "pattern":   raw.pattern,
            "source":    raw.source,
            "name":      raw.name,
            "part":      i,
        }

        chunks.append(Chunk(
            text=part,
            metadata=metadata,
            chunk_id=chunk_id,
        ))

    return chunks


def process(raw_chunks: list[RawChunk]) -> list[Chunk]:
    """
    Convert all RawChunks to final Chunks.
    Validates every chunk before returning.
    """
    final: list[Chunk] = []

    for raw in raw_chunks:
        converted = raw_to_chunks(raw)
        for chunk in converted:
            chunk.validate()
        final.extend(converted)

    # Stats
    type_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for c in final:
        type_counts[c.metadata["type"]] = type_counts.get(c.metadata["type"], 0) + 1
        domain_counts[c.metadata["domain"]] = domain_counts.get(c.metadata["domain"], 0) + 1

    print(f"\n  Chunking complete: {len(final)} chunks total")
    print(f"  By type:   {type_counts}")
    print(f"  By domain: {domain_counts}")

    return final