"""
ingest.py — Main ingestion pipeline runner

Usage:
    python ingest.py                    # full ingestion (GitHub + docs)
    python ingest.py --source github    # GitHub only
    python ingest.py --source docs      # docs site only
    python ingest.py --wipe             # wipe collection before re-indexing

Environment variables:
    OPENAI_API_KEY   — if set, uses text-embedding-3-small (recommended)
                       if not set, falls back to local sentence-transformers (dev mode)
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from ingestion import crawl_github, crawl_docs, chunk
from ingestion.manual_chunks import MANUAL_CHUNKS
from vector_store import ChromaDBStore


def parse_args():
    parser = argparse.ArgumentParser(description="ProgPy RAG ingestion pipeline")
    parser.add_argument(
        "--source",
        choices=["github", "docs", "all"],
        default="all",
        help="Which source to ingest (default: all)",
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Wipe the vector store collection before ingesting",
    )
    parser.add_argument(
        "--collection",
        default="progpy",
        help="ChromaDB collection name (default: progpy)",
    )
    parser.add_argument(
        "--persist-dir",
        default="./data/chroma",
        help="ChromaDB persistence directory (default: ./data/chroma)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Upsert batch size (default: 50)",
    )
    return parser.parse_args()


def run():
    args = parse_args()

    print("=" * 60)
    print("ProgPy RAG Ingestion Pipeline")
    print("=" * 60)

    # --- Init vector store ---
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("\n⚠️  No OPENAI_API_KEY found — using local embeddings (dev mode)")
        print("   Set OPENAI_API_KEY for production quality embeddings\n")

    store = ChromaDBStore(
        collection_name=args.collection,
        persist_dir=args.persist_dir,
        openai_api_key=openai_key,
    )

    if args.wipe:
        print("Wiping existing collection...")
        store.delete_collection()
        # Re-create after wipe
        store = ChromaDBStore(
            collection_name=args.collection,
            persist_dir=args.persist_dir,
            openai_api_key=openai_key,
        )

    print(f"Vector store ready. Current chunk count: {store.count()}")

    # --- Crawl sources ---
    raw_chunks = []

    if args.source in ("github", "all"):
        print("\n[1/3] Crawling GitHub repo...")
        raw_chunks.extend(crawl_github())

    if args.source in ("docs", "all"):
        print("\n[2/3] Crawling docs site...")
        raw_chunks.extend(crawl_docs())

    if not raw_chunks:
        print("No chunks extracted. Exiting.")
        return

    # Add manually curated chunks
    print(f"\n[+] Adding {len(MANUAL_CHUNKS)} manual curated chunks...")
    raw_chunks.extend(MANUAL_CHUNKS)

    # --- Chunk + validate ---
    print("\n[3/3] Chunking and validating...")
    final_chunks = chunk(raw_chunks)

    # --- Upsert in batches ---
    print(f"\nUpserting {len(final_chunks)} chunks in batches of {args.batch_size}...")
    batches = [
        final_chunks[i : i + args.batch_size]
        for i in range(0, len(final_chunks), args.batch_size)
    ]

    for i, batch in enumerate(batches):
        store.upsert(batch)
        if (i + 1) % 10 == 0 or i == len(batches) - 1:
            print(f"  {(i+1)*args.batch_size}/{len(final_chunks)} chunks upserted...")

    # --- Final stats ---
    print(f"\n✅ Ingestion complete!")
    print(f"   Total chunks in store: {store.count()}")
    print(f"\nTo query the store:")
    print(f"   from vector_store import ChromaDBStore")
    print(f"   store = ChromaDBStore(persist_dir='{args.persist_dir}')")
    print(f"   results = store.query('thermal degradation model', filter={{'framework': 'progpy'}})")


if __name__ == "__main__":
    run()