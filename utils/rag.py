"""
utils/rag.py
------------
Manual ingestion and retrieval for the digital twin pipeline.

Uses:
- Claude (claude-sonnet) for rich text extraction from PDF chunks
- OpenAI (text-embedding-3-large) for vector embeddings
- ChromaDB for persistent vector storage
- fitz (PyMuPDF) for fast page text extraction

Usage:
    from utils.rag import ManualRAG
    from config import get_machine_config

    cfg = get_machine_config("eversys_coffee_machine")
    rag = ManualRAG(cfg)

    # Index once — skips automatically if already indexed
    rag.index_manual()

    # Re-index from scratch
    rag.index_manual(force=True)
"""

import base64
import io
import os
import re
import time

import anthropic
import chromadb
import fitz  # PyMuPDF — faster text extraction than pypdf
from anthropic import RateLimitError
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from pypdf import PdfReader, PdfWriter


class ManualRAG:
    def __init__(self, cfg: dict, model: str = "claude-sonnet-4-20250514"):
        """
        Args:
            cfg:   result of get_machine_config()
            model: Claude model to use for rich text extraction
        """
        self.cfg             = cfg
        self.model           = model
        self.collection_name = cfg["collection_name"]

        # Claude client — for rich text extraction
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # OpenAI embedding function — for vector search
        self.embedding_fn = OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-large",
        )

        # ChromaDB — persistent vector store
        self.chroma_client = chromadb.PersistentClient(path=str(cfg["rag_dir"]))

        os.makedirs(cfg["chunks_dir"], exist_ok=True)

    # ── Public interface ──────────────────────────────────────────────────────

    def collection_exists(self) -> bool:
        """Check if this machine's collection is already indexed."""
        try:
            self.chroma_client.get_collection(name=self.collection_name)
            return True
        except Exception:
            return False

    def index_manual(self, force: bool = False, max_section_pages: int = 8,
                     fallback_chunk_size: int = 5, delay_between_chunks: int = 3):
        """
        Index the machine manual into ChromaDB.

        Skips automatically if the collection already exists.
        Pass force=True to re-index from scratch.

        Args:
            force:                delete and re-index even if collection exists
            max_section_pages:    max pages per section chunk before splitting
            fallback_chunk_size:  chunk size if heading detection fails
            delay_between_chunks: seconds to wait between Claude API calls
        """
        if self.collection_exists() and not force:
            print(f"✓ '{self.collection_name}' already indexed — skipping.")
            print(f"  Pass force=True to re-index.")
            return self._get_collection()

        # Find the PDF
        pdf_path = self._find_manual_pdf()
        if not pdf_path:
            raise FileNotFoundError(
                f"No PDF found in {self.cfg['manual_dir']}. "
                f"Place the manual PDF there and try again."
            )

        print(f"\nIndexing '{self.collection_name}' from {pdf_path.name}...")

        # Delete existing collection if force
        if force and self.collection_exists():
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"  Deleted existing collection.")

        collection = self.chroma_client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": f"Manual: {self.collection_name}"}
        )

        chunks = self._build_section_chunks(
            pdf_path=str(pdf_path),
            max_section_pages=max_section_pages,
            fallback_chunk_size=fallback_chunk_size,
        )

        estimated_mins = len(chunks) * delay_between_chunks / 60
        print(f"  {len(chunks)} chunks to index (~{estimated_mins:.1f} minutes)...")

        for i, chunk in enumerate(chunks):
            print(
                f"  [{i+1}/{len(chunks)}] pages {chunk['start']}-{chunk['end']} "
                f"| {chunk['heading']}... ",
                end="", flush=True
            )

            pdf_chunk_b64 = self._extract_pdf_chunk(str(pdf_path), chunk["start"], chunk["end"])

            # Save raw chunk to disk
            chunk_filename = f"{self.collection_name}_chunk_{chunk['id']:03d}.pdf"
            chunk_path     = self.cfg["chunks_dir"] / chunk_filename
            with open(chunk_path, "wb") as f:
                f.write(base64.b64decode(pdf_chunk_b64))

            # Extract rich text via Claude
            rich_text = self._extract_rich_content(
                pdf_chunk_b64,
                pages=f"{chunk['start']}-{chunk['end']}"
            )

            collection.add(
                documents=[rich_text],
                metadatas=[{
                    "chunk_id":   chunk["id"],
                    "start_page": chunk["start"],
                    "end_page":   chunk["end"],
                    "heading":    chunk["heading"],
                    "manual_id":  self.collection_name,
                    "pdf_file":   chunk_filename,
                    "page_count": chunk["end"] - chunk["start"] + 1,
                }],
                ids=[f"{self.collection_name}_chunk_{chunk['id']:03d}"]
            )

            print("✓")

            if i < len(chunks) - 1:
                time.sleep(delay_between_chunks)

        print(f"\n✅ Indexed {len(chunks)} chunks successfully.")
        return collection

    def retrieve_chunks(self, queries: list[str], n_results_per_query: int = 5) -> list[dict]:
        """
        Query the collection and return deduplicated chunks.
        Will be upgraded with reranking and BM25 in a later phase.

        Args:
            queries:             list of query strings
            n_results_per_query: how many chunks to retrieve per query

        Returns:
            list of chunk dicts with text, metadata, and matched_queries
        """
        collection = self._get_collection()
        all_chunks = {}

        for query in queries:
            results = collection.query(query_texts=[query], n_results=n_results_per_query)

            for i, metadata in enumerate(results["metadatas"][0]):
                chunk_id = results["ids"][0][i]
                doc      = results["documents"][0][i]

                if chunk_id not in all_chunks:
                    all_chunks[chunk_id] = {
                        "chunk_id":        chunk_id,
                        "text":            doc,
                        "start_page":      metadata["start_page"],
                        "end_page":        metadata["end_page"],
                        "heading":         metadata.get("heading", ""),
                        "matched_queries": [query],
                    }
                else:
                    all_chunks[chunk_id]["matched_queries"].append(query)

        return list(all_chunks.values())

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_collection(self):
        """Get the existing collection with the embedding function attached."""
        return self.chroma_client.get_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
        )

    def _find_manual_pdf(self):
        """Find the first PDF in the machine's manual directory."""
        pdfs = list(self.cfg["manual_dir"].glob("*.pdf"))
        return pdfs[0] if pdfs else None

    def _extract_pdf_chunk(self, pdf_path: str, start_page: int, end_page: int) -> str:
        """Extract a page range from the PDF and return as base64."""
        reader = PdfReader(pdf_path)
        writer = PdfWriter()

        for page_num in range(start_page - 1, end_page):
            if page_num < len(reader.pages):
                writer.add_page(reader.pages[page_num])

        output = io.BytesIO()
        writer.write(output)
        return base64.b64encode(output.getvalue()).decode()

    def _extract_rich_content(self, pdf_chunk_b64: str, pages: str, max_retries: int = 3) -> str:
        """Extract rich text from a PDF chunk using Claude."""
        prompt = f"""Extract ALL information from pages {pages} of this technical manual.

CRITICAL REQUIREMENTS:
1. Extract COMPLETE technical specifications with exact values, ranges, and units
2. Include ALL model numbers, part numbers, error codes, and reference numbers
3. Preserve ALL tables, lists, and structured data exactly as shown
4. Describe ALL diagrams, images, and visual content in detail
5. Keep technical terminology EXACTLY as written
6. Do NOT summarize - extract verbatim content with full detail
7. Organize by sections/topics but include ALL information

Focus on:
- Specifications (temperatures, pressures, speeds, capacities, dimensions)
- Component details (motors, sensors, valves, pumps)
- Procedures (step-by-step instructions)
- Error codes and troubleshooting
- Part identification and diagrams
- Settings and parameters

Extract everything - this will be used for technical search and retrieval."""

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_chunk_b64,
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }]
                )
                return response.content[0].text

            except RateLimitError:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 60
                    print(f"\n    Rate limit — waiting {wait_time}s (retry {attempt+2}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise

    def _extract_page_texts(self, pdf_path: str) -> list[dict]:
        """
        Extract raw text page-by-page using fitz (PyMuPDF).
        Used only for heading detection — faster than pypdf for this purpose.
        """
        doc        = fitz.open(pdf_path)
        page_texts = []

        for i, page in enumerate(doc, start=1):
            try:
                text = page.get_text() or ""
            except Exception:
                text = ""
            page_texts.append({"page_num": i, "text": text})

        doc.close()
        return page_texts

    def _find_section_heading(self, text: str):
        """Detect a section heading in the first 15 lines of a page."""
        if not text:
            return None

        lines    = [line.strip() for line in text.splitlines() if line.strip()]
        patterns = [
            re.compile(r'^\d+\.\d+\.?\s+.+$'),  # e.g. "6.4 Brew Group"
        ]

        for line in lines[:15]:
            for pattern in patterns:
                if pattern.match(line):
                    return line

        return None

    def _build_section_chunks(self, pdf_path: str, max_section_pages: int = 8,
                               fallback_chunk_size: int = 5) -> list[dict]:
        """
        Build chunks based on detected section headings.
        Falls back to fixed-size chunks if heading detection is weak.
        """
        page_texts  = self._extract_page_texts(pdf_path)
        total_pages = len(page_texts)

        # Detect section starts
        section_starts = []
        for page in page_texts:
            heading = self._find_section_heading(page["text"])
            if heading:
                section_starts.append({"page_num": page["page_num"], "heading": heading})

        # Deduplicate consecutive duplicates
        deduped = []
        seen    = set()
        for sec in section_starts:
            key = (sec["page_num"], sec["heading"])
            if key not in seen:
                deduped.append(sec)
                seen.add(key)
        section_starts = deduped

        # Fallback if heading detection is weak
        if len(section_starts) < 3:
            print("  Section detection weak — falling back to fixed-size chunks.")
            chunks   = []
            current  = 1
            chunk_id = 0
            while current <= total_pages:
                end = min(current + fallback_chunk_size - 1, total_pages)
                chunks.append({
                    "id":      chunk_id,
                    "start":   current,
                    "end":     end,
                    "heading": f"pages_{current}_{end}",
                })
                chunk_id += 1
                current  += fallback_chunk_size
            return chunks

        # Build section ranges
        section_chunks = []
        for i, sec in enumerate(section_starts):
            start = sec["page_num"]
            end   = section_starts[i+1]["page_num"] - 1 if i < len(section_starts) - 1 else total_pages

            if end < start:
                continue

            section_chunks.append({
                "id":      i,
                "start":   start,
                "end":     end,
                "heading": sec["heading"],
            })

        # Split oversized sections
        final_chunks = []
        chunk_id     = 0

        for chunk in section_chunks:
            length = chunk["end"] - chunk["start"] + 1

            if length <= max_section_pages:
                final_chunks.append({
                    "id":      chunk_id,
                    "start":   chunk["start"],
                    "end":     chunk["end"],
                    "heading": chunk["heading"],
                })
                chunk_id += 1
            else:
                current = chunk["start"]
                sub_idx = 0
                while current <= chunk["end"]:
                    sub_end = min(current + max_section_pages - 1, chunk["end"])
                    final_chunks.append({
                        "id":      chunk_id,
                        "start":   current,
                        "end":     sub_end,
                        "heading": f"{chunk['heading']} [part {sub_idx+1}]",
                    })
                    chunk_id += 1
                    sub_idx  += 1
                    current   = sub_end + 1

        return final_chunks