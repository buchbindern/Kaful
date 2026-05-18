import sys, os
sys.path.insert(0, "twin_framework_rags/ProgPy/progpy_rag")

from dotenv import load_dotenv
load_dotenv(".env")

from ingestion.manual_chunks import MANUAL_CHUNKS
from ingestion.chunker import process
from vector_store import ChromaDBStore

store = ChromaDBStore(
    persist_dir="twin_framework_rags/ProgPy/progpy_rag/data/chroma",
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

print(f"Current chunks: {store.count()}")
chunks = process(MANUAL_CHUNKS)
print(f"Upserting {len(chunks)} manual chunks...")
store.upsert(chunks)
print(f"Done. Total: {store.count()}")